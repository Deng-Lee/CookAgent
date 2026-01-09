"""
Lightweight parent-level retriever with aggregation.

Flow:
- Query Chroma collection using enriched chunks embeddings.
- Aggregate hits by parent_id (RRF sum + max score + coverage).
- Optionally load parent markdown for downstream generation.

Usage:
  python src/retrieval/parent_retriever.py \
    --query "白灼虾怎么做" \
    --db-path data/chroma \
    --collection cook_chunks_v1 \
    --top-k 25 \
    --top-parents 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from evidence_utils import (
    build_evidence_set,
    build_evidence_set_for_parent,
    default_clarify_question,
    evidence_sufficient,
    format_auto_answer,
)
from generation_utils import (
    classify_query,
    route_blocks,
    build_layer_evidence,
    generate_answer,
    parse_steps,
    parse_ingredients,
    normalize_block_type,
)
from logging_utils import (
    log_evidence_built,
    log_evidence_routing,
    log_evidence_insufficient,
    log_parent_lock,
    log_retrieval,
)
from model_utils import resolve_local_model_path
from retrieval_types import (
    ChunkHit,
    ParentHit,
    ParentLock,
    RetrievalState,
    DEFAULT_EVIDENCE_LOG,
    DEFAULT_LOCK_LOG,
)
from session_utils import load_session, parse_option_id, purge_expired_pending, save_session




def _distance_to_score(distance: float) -> float:
    """Convert Chroma distance (cosine) to a similarity score."""
    return 1.0 / (1.0 + distance)


def aggregate_hits(
    res: Dict,
    *,
    k: int = 60,
) -> List[ParentHit]:
    """
    Aggregate chunk-level hits into parent-level scores using RRF sum + max score.
    """
    parents: Dict[str, ParentHit] = {}
    ids = res["ids"][0]
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    seen_chunk_ids = set()

    for idx, (hit_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        if hit_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(hit_id)
        parent_id = meta.get("parent_id")
        if parent_id is None:
            continue
        score = _distance_to_score(dist)
        if parent_id not in parents:
            parents[parent_id] = ParentHit(parent_id=parent_id)
        ph = parents[parent_id]
        ph.total_chunks = ph.total_chunks or int(meta.get("total_chunks") or 0)
        ph.rrf_sum += 1.0 / (k + idx + 1)
        ph.max_chunk_score = max(ph.max_chunk_score, score)
        ph.coverage += 1
        ph.hits.append(
            ChunkHit(
                score=score,
                rank=idx + 1,
                text=doc,
                metadata={**meta, "chunk_id": hit_id},
            )
        )

    # Sort hits inside each parent by score desc.
    for ph in parents.values():
        ph.hits.sort(key=lambda h: h.score, reverse=True)

    # Compute coverage ratio.
    for ph in parents.values():
        if ph.total_chunks > 0:
            ph.coverage_ratio = ph.coverage / ph.total_chunks
        else:
            ph.coverage_ratio = 0.0

    # Normalize rrf_sum and max_chunk_score for parents that pass coverage threshold.
    coverage_threshold = 0.5
    eps = 1e-8
    eligible = [ph for ph in parents.values() if ph.coverage_ratio >= coverage_threshold]
    if eligible:
        min_rrf = min(p.rrf_sum for p in eligible)
        max_rrf = max(p.rrf_sum for p in eligible)
        min_max = min(p.max_chunk_score for p in eligible)
        max_max = max(p.max_chunk_score for p in eligible)
        w1 = 0.7
        w2 = 0.3
        for ph in eligible:
            ph.rrf_sum_norm = (ph.rrf_sum - min_rrf) / (max_rrf - min_rrf + eps)
            ph.max_chunk_norm = (ph.max_chunk_score - min_max) / (max_max - min_max + eps)
            ph.overall_score = w1 * ph.rrf_sum_norm + w2 * ph.max_chunk_norm

        # Sort only eligible parents by overall_score desc.
        return sorted(
            eligible,
            key=lambda p: p.overall_score,
            reverse=True,
        )

    return list(parents.values())


def load_parent_doc(parent_id: str) -> Optional[str]:
    path = Path(parent_id)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _retrieve_state(
    query: str,
    db_path: Path,
    collection_name: str,
    *,
    top_k: int = 25,
    top_parents: int = 5,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
) -> RetrievalState:
    model_path = resolve_local_model_path()
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=model_path,
        model_kwargs={"local_files_only": True},
    )
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    res = collection.query(query_texts=[query], n_results=top_k)
    parents = aggregate_hits(res)
    coverage_threshold = 0.5
    any_eligible = any(ph.coverage_ratio >= coverage_threshold for ph in parents)
    t_min = 0.40
    auto_recommend = True
    parent_lock = ParentLock(
        status="none",
        parent_id=None,
        lock_reason=None,
        lock_score=None,
        locked_at_turn=turn,
        pending_reason=None,
    )
    if not any_eligible:
        for ph in parents:
            ph.low_evidence = True
        auto_recommend = False
    if parents and parents[0].overall_score < t_min:
        for ph in parents:
            ph.low_evidence = True
        auto_recommend = False
    if parents and parents[0].overall_score > t_min and len(parents) > 1:
        score1 = parents[0].overall_score
        score2 = parents[1].overall_score
        if score1 > 0 and (score2 / score1) > 0.92:
            for ph in parents:
                ph.good_but_ambiguous = True
            auto_recommend = False
    if auto_recommend:
        for ph in parents:
            ph.auto_recommend = True
        if parents:
            parent_lock = ParentLock(
                status="locked",
                parent_id=parents[0].parent_id,
                lock_reason="auto",
                lock_score=parents[0].overall_score,
                locked_at_turn=turn,
                pending_reason=None,
            )
    elif parents and parents[0].good_but_ambiguous:
        parent_lock = ParentLock(
            status="pending",
            parent_id=None,
            lock_reason=None,
            lock_score=None,
            locked_at_turn=turn,
            pending_reason="ambiguous_top1_top2",
        )
    for ph in parents[:top_parents]:
        ph.parent_doc = load_parent_doc(ph.parent_id)

    if log_path:
        log_retrieval(query, res, parents[:top_parents])
    if lock_log_path:
        log_parent_lock(query, parent_lock, parents[:top_parents], log_path=lock_log_path)
    evidence_set = None
    if parent_lock.status == "locked" and parents:
        evidence_set = build_evidence_set(parents[0])
        if evidence_log_path:
            log_evidence_built(query, evidence_set, log_path=evidence_log_path)
    if evidence_insufficient and parent_lock.status == "locked" and evidence_set:
        log_evidence_insufficient(
            query,
            turn,
            parent_lock,
            evidence_set,
            evidence_insufficient,
            log_path=evidence_log_path or DEFAULT_EVIDENCE_LOG,
        )
    return RetrievalState(
        parents=parents[:top_parents],
        parent_lock=parent_lock,
        evidence_set=evidence_set,
    )


def retrieve(
    query: str,
    db_path: Path,
    collection_name: str,
    *,
    top_k: int = 25,
    top_parents: int = 5,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
) -> List[ParentHit]:
    state = _retrieve_state(
        query=query,
        db_path=db_path,
        collection_name=collection_name,
        top_k=top_k,
        top_parents=top_parents,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=turn,
    )
    return state.parents


def _build_output_from_state(
    query: str,
    trace_id: str,
    state: RetrievalState,
    *,
    include_candidates: bool = False,
) -> Dict:
    parents = state.parents
    parent_lock = state.parent_lock
    evidence_set = state.evidence_set

    if parent_lock.status == "locked" and parents:
        sufficient, missing = evidence_sufficient(evidence_set)
        if sufficient:
            answer = format_auto_answer(evidence_set)
            return {
                "trace_id": trace_id,
                "state": "AUTO_RECOMMEND",
                "parent_id": parent_lock.parent_id,
                "lock_status": "locked",
                "answer": answer,
            }
        return {
            "trace_id": trace_id,
            "state": "LOW_EVIDENCE",
            "lock_status": "none",
            "message": "evidence_insufficient",
            "missing_block_types": missing,
            "clarify_question": default_clarify_question(query),
        }

    if parent_lock.status == "pending":
        candidates = []
        for idx, ph in enumerate(parents, 1):
            item = {
                "dish_name": ph.hits[0].metadata.get("dish_name") if ph.hits else None,
                "overall_score": ph.overall_score,
                "top_evidence_snippets": [h.text.replace("\n", " ")[:80] for h in ph.hits[:3]],
            }
            if include_candidates:
                item["option_id"] = str(idx)
            candidates.append(item)
        return {
            "trace_id": trace_id,
            "state": "GOOD_BUT_AMBIGUOUS",
            "lock_status": "pending",
            "candidates": candidates,
        }

    return {
        "trace_id": trace_id,
        "state": "LOW_EVIDENCE",
        "lock_status": "none",
        "message": "evidence_too_low_or_no_candidates",
        "clarify_question": default_clarify_question(query),
    }


def run_once(
    query: str,
    trace_id: str,
    *,
    db_path: Path,
    collection_name: str,
    top_k: int = 25,
    top_parents: int = 5,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
) -> Dict:
    state = _retrieve_state(
        query=query,
        db_path=db_path,
        collection_name=collection_name,
        top_k=top_k,
        top_parents=top_parents,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=turn,
    )
    return _build_output_from_state(query, trace_id, state, include_candidates=False)


def run_session_once(
    query: str,
    trace_id: str,
    session_id: str,
    *,
    db_path: Path,
    collection_name: str,
    top_k: int = 25,
    top_parents: int = 5,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
) -> Dict:
    session = load_session(session_id)
    now = time.time()
    last_turn = session.get("turn") or 0
    current_turn = turn if turn is not None else last_turn + 1
    session["turn"] = current_turn

    purge_expired_pending(session)

    option_id = parse_option_id(query, session.get("pending_candidates", []))
    if option_id:
        candidate = next(
            (c for c in session.get("pending_candidates", []) if c.get("option_id") == option_id), None
        )
        if candidate:
            parent_id = candidate.get("parent_id")
            lock_score = candidate.get("overall_score")
            parent_lock = ParentLock(
                status="locked",
                parent_id=parent_id,
                lock_reason="user_select",
                lock_score=lock_score,
                locked_at_turn=current_turn,
                pending_reason=None,
            )
            evidence_set = build_evidence_set_for_parent(db_path, collection_name, parent_id)
            if evidence_log_path:
                log_evidence_built(query, evidence_set, log_path=evidence_log_path)
            sufficient, missing = evidence_sufficient(evidence_set)
            if lock_log_path:
                log_parent_lock(query, parent_lock, [], log_path=lock_log_path)
            op_texts = [
                c.get("text", "")
                for c in evidence_set.get("chunks", [])
                if normalize_block_type(c.get("block_type")) == "operation"
            ]
            ing_texts = [
                c.get("text", "")
                for c in evidence_set.get("chunks", [])
                if normalize_block_type(c.get("block_type")) == "ingredients"
            ]
            session["parent_cache"] = {
                "parent_id": parent_id,
                "parsed_steps": parse_steps(op_texts),
                "parsed_ingredients": parse_ingredients(ing_texts),
            }
            session["parent_lock"] = {
                "status": "locked",
                "parent_id": parent_id,
                "pending_reason": None,
                "lock_score": lock_score,
            }
            session["pending_candidates"] = []
            session["last_trace_id"] = trace_id
            session["updated_at"] = now
            save_session(session)
            if sufficient:
                return {
                    "trace_id": trace_id,
                    "state": "AUTO_RECOMMEND",
                    "parent_id": parent_id,
                    "lock_status": "locked",
                    "answer": format_auto_answer(evidence_set),
                }
            return {
                "trace_id": trace_id,
                "state": "LOW_EVIDENCE",
                "lock_status": "none",
                "message": "evidence_insufficient",
                "missing_block_types": missing,
                "clarify_question": default_clarify_question(query),
            }

    if session.get("parent_lock", {}).get("status") == "locked":
        parent_id = session.get("parent_lock", {}).get("parent_id")
        if parent_id:
            evidence_set = build_evidence_set_for_parent(db_path, collection_name, parent_id)
            if evidence_log_path:
                log_evidence_built(query, evidence_set, log_path=evidence_log_path)

            routing = classify_query(query)
            intent = routing["intent"]
            confidence = routing["confidence"]
            slots = routing["slots"]
            intent_min = 0.5
            layer1_blocks = route_blocks(intent) if confidence >= intent_min else []
            selected_blocks = layer1_blocks
            evidence_layer1 = build_layer_evidence(evidence_set, layer1_blocks) if layer1_blocks else None
            parent_cache = session.get("parent_cache") or {}
            cached_steps = None
            cached_ingredients = None
            if parent_cache.get("parent_id") == parent_id:
                cached_steps = parent_cache.get("parsed_steps")
                cached_ingredients = parent_cache.get("parsed_ingredients")
            if cached_steps is None or cached_ingredients is None:
                op_texts = [
                    c.get("text", "")
                    for c in evidence_set.get("chunks", [])
                    if normalize_block_type(c.get("block_type")) == "operation"
                ]
                ing_texts = [
                    c.get("text", "")
                    for c in evidence_set.get("chunks", [])
                    if normalize_block_type(c.get("block_type")) == "ingredients"
                ]
                cached_steps = cached_steps if cached_steps is not None else parse_steps(op_texts)
                cached_ingredients = (
                    cached_ingredients if cached_ingredients is not None else parse_ingredients(ing_texts)
                )
                session["parent_cache"] = {
                    "parent_id": parent_id,
                    "parsed_steps": cached_steps,
                    "parsed_ingredients": cached_ingredients,
                }

            routing_log = {
                "intent": intent,
                "confidence": confidence,
                "selected_blocks_layer1": selected_blocks,
                "upgraded_to_layer2": False,
                "upgrade_reason": None,
                "final_evidence_chunk_ids": [],
            }

            answer = None
            missing_slots: List[str] = []
            if confidence < intent_min:
                routing_log["upgraded_to_layer2"] = True
                routing_log["upgrade_reason"] = "low_confidence"
            elif not evidence_layer1 or not evidence_layer1.get("chunks"):
                routing_log["upgraded_to_layer2"] = True
                routing_log["upgrade_reason"] = "missing_block"
            else:
                answer, missing_slots = generate_answer(
                    intent,
                    slots,
                    evidence_layer1,
                    parsed_steps=cached_steps,
                    parsed_ingredients=cached_ingredients,
                )
                routing_log["final_evidence_chunk_ids"] = [
                    c.get("chunk_id") for c in evidence_layer1.get("chunks", [])
                ]
                if (not answer) or missing_slots:
                    routing_log["upgraded_to_layer2"] = True
                    routing_log["upgrade_reason"] = "no_hit_sentence"

            if routing_log["upgraded_to_layer2"]:
                answer, missing_slots = generate_answer(
                    intent,
                    slots,
                    evidence_set,
                    parsed_steps=cached_steps,
                    parsed_ingredients=cached_ingredients,
                )
                routing_log["final_evidence_chunk_ids"] = [
                    c.get("chunk_id") for c in evidence_set.get("chunks", [])
                ]

            if evidence_log_path:
                log_evidence_routing(
                    trace_id,
                    current_turn,
                    query,
                    routing_log,
                    log_path=evidence_log_path,
                )

            sufficient, missing = evidence_sufficient(evidence_set)
            if answer and sufficient:
                session["last_trace_id"] = trace_id
                session["updated_at"] = now
                save_session(session)
                return {
                    "trace_id": trace_id,
                    "state": "AUTO_RECOMMEND",
                    "parent_id": parent_id,
                    "lock_status": "locked",
                    "answer": answer,
                }
            session["last_trace_id"] = trace_id
            session["updated_at"] = now
            save_session(session)
            if evidence_log_path:
                log_evidence_insufficient(
                    query,
                    current_turn,
                    ParentLock(
                        status="locked",
                        parent_id=parent_id,
                        lock_reason="auto",
                        lock_score=session.get("parent_lock", {}).get("lock_score"),
                        locked_at_turn=current_turn,
                        pending_reason=None,
                    ),
                    evidence_set,
                    {
                        "output_intent": intent,
                        "missing_block_types": [],
                        "missing_slots": missing_slots or missing,
                        "decision": {"action": "clarify", "reason": "missing_slots"},
                    },
                    log_path=evidence_log_path,
                )
            return {
                "trace_id": trace_id,
                "state": "LOW_EVIDENCE",
                "lock_status": "locked",
                "message": "evidence_insufficient",
                "missing_block_types": missing,
                "clarify_question": "该菜谱未提及这一点，是否需要切换做法？",
            }

    state = _retrieve_state(
        query=query,
        db_path=db_path,
        collection_name=collection_name,
        top_k=top_k,
        top_parents=top_parents,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=current_turn,
    )
    output = _build_output_from_state(query, trace_id, state, include_candidates=True)
    if output.get("state") == "GOOD_BUT_AMBIGUOUS":
        pending_candidates = []
        for idx, ph in enumerate(state.parents, 1):
            pending_candidates.append(
                {
                    "option_id": str(idx),
                    "parent_id": ph.parent_id,
                    "dish_name": ph.hits[0].metadata.get("dish_name") if ph.hits else None,
                    "overall_score": ph.overall_score,
                }
            )
        top1_score = state.parents[0].overall_score if state.parents else None
        session["parent_lock"] = {
            "status": "pending",
            "parent_id": None,
            "pending_reason": "ambiguous_top1_top2",
            "lock_score": top1_score,
        }
        session["pending_candidates"] = pending_candidates
        session["parent_cache"] = None
    else:
        session["parent_lock"] = {
            "status": state.parent_lock.status,
            "parent_id": state.parent_lock.parent_id,
            "pending_reason": state.parent_lock.pending_reason,
            "lock_score": state.parent_lock.lock_score,
        }
        session["pending_candidates"] = []
        if state.parent_lock.status != "locked":
            session["parent_cache"] = None
    session["last_trace_id"] = trace_id
    session["updated_at"] = now
    save_session(session)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parent-level retrieval with aggregation.")
    parser.add_argument("--query", help="User query text. If omitted, will prompt interactively.")
    parser.add_argument("--db-path", type=Path, default=Path("data/chroma"), help="Chroma persistence path.")
    parser.add_argument("--collection", type=str, default="cook_chunks_v1", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=25, help="Chunk-level n_results.")
    parser.add_argument("--top-parents", type=int, default=5, help="How many parent docs to return.")
    parser.add_argument("--log-path", type=Path, default=Path("logs/retriever.log"), help="Path to append JSON logs.")
    parser.add_argument(
        "--lock-log-path",
        type=Path,
        default=DEFAULT_LOCK_LOG,
        help="Path to append parent lock logs.",
    )
    parser.add_argument(
        "--evidence-log-path",
        type=Path,
        default=DEFAULT_EVIDENCE_LOG,
        help="Path to append evidence-driven logs.",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=None,
        help="Conversation turn index for parent locking logs.",
    )
    parser.add_argument(
        "--trace-id",
        type=str,
        default="cli",
        help="Trace id for run_once output.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="cli_default",
        help="Session id for minimal persistence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query:
        result = run_session_once(
            query=args.query.strip(),
            trace_id=args.trace_id,
            session_id=args.session_id,
            db_path=args.db_path,
            collection_name=args.collection,
            top_k=args.top_k,
            top_parents=args.top_parents,
            log_path=args.log_path,
            lock_log_path=args.lock_log_path,
            evidence_log_path=args.evidence_log_path,
            turn=args.turn,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("进入交互模式，输入 exit 退出。")
    while True:
        query = input("请输入问题: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        result = run_session_once(
            query=query,
            trace_id=args.trace_id,
            session_id=args.session_id,
            db_path=args.db_path,
            collection_name=args.collection,
            top_k=args.top_k,
            top_parents=args.top_parents,
            log_path=args.log_path,
            lock_log_path=args.lock_log_path,
            evidence_log_path=args.evidence_log_path,
            turn=args.turn,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
