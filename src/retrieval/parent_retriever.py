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
import os
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
    output_intent_for,
    build_generation_mapping,
)
from llm_utils import (
    LLMCall,
    build_extraction_payload,
    build_polish_payload,
    call_llm_extract_with_debug,
    call_llm_polish_with_debug,
    validate_extraction,
    validate_polish,
)
from llm_client import llm_extract, llm_polish
from logging_utils import (
    log_evidence_built,
    log_evidence_routing,
    log_evidence_insufficient,
    log_parent_lock,
    log_retrieval,
    log_generation_started,
    log_generation_completed,
    log_generation_mapping,
    log_llm_call,
)
from model_utils import resolve_local_model_path
from retrieval_types import (
    ChunkHit,
    ParentHit,
    ParentLock,
    RetrievalState,
    DEFAULT_EVIDENCE_LOG,
    DEFAULT_LOCK_LOG,
    DEFAULT_GENERATION_LOG,
)
from session_utils import load_session, parse_option_id, purge_expired_pending, save_session


DEFAULT_EXCLUDED_CATEGORIES = {"示例菜", "调料", "半成品"}

CATEGORY_SYNONYMS = {
    "素菜": ["素", "素菜", "素食", "无肉"],
    "荤菜": ["荤", "荤菜", "肉", "肉菜", "有肉"],
    "水产": ["水产", "海鲜", "鱼", "虾", "蟹"],
    "汤": ["汤", "羹", "煲汤"],
    "甜点": ["甜点", "甜品", "点心"],
    "饮料": ["饮料", "饮品", "奶茶", "果汁"],
    "早餐": ["早餐", "早饭", "早点"],
    "主食": ["主食", "饭", "面", "粥", "饼"],
}


def detect_categories(query: str) -> List[str]:
    hits = []
    text = query or ""
    for category, words in CATEGORY_SYNONYMS.items():
        if any(word in text for word in words):
            hits.append(category)
    return hits


def resolve_category_choice(query: str, pending: List[Dict]) -> Optional[str]:
    if not query or not pending:
        return None
    option_id = parse_option_id(query, pending)
    if option_id:
        hit = next((c for c in pending if c.get("option_id") == option_id), None)
        return hit.get("category") if hit else None
    for item in pending:
        category = item.get("category")
        if category and category in query:
            return category
    return None


def build_category_where_filter(dir_category: Optional[str]) -> Optional[Dict]:
    filters = []
    if dir_category:
        filters.append({"dir_category": dir_category})
    filters.append({"is_excluded_default": False})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


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
    dir_category: Optional[str] = None,
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

    where_filter = build_category_where_filter(dir_category)
    res = collection.query(query_texts=[query], n_results=top_k, where=where_filter)
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
        detected_categories = detect_categories(query)
        log_retrieval(
            query,
            res,
            parents[:top_parents],
            detected_categories=detected_categories,
            category_filter_applied=bool(dir_category),
            category_conflict=len(detected_categories) > 1,
            category_filtered_count=len(parents[:top_parents]),
        )
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
    dir_category: Optional[str] = None,
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
        dir_category=dir_category,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=turn,
    )
    return state.parents


def _collect_field_texts(fields: Dict, names: List[str]) -> List[str]:
    texts: List[str] = []
    for name in names:
        items = fields.get(name)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return texts


def _build_recipe_from_extraction(extraction: Dict) -> Optional[str]:
    fields = extraction.get("fields", {}) if isinstance(extraction, dict) else {}
    if not isinstance(fields, dict):
        return None
    ingredients = _collect_field_texts(fields, ["ingredients"])
    steps = _collect_field_texts(fields, ["operation", "steps"])
    tips = _collect_field_texts(fields, ["tips"])
    if not ingredients and not steps and not tips:
        return None
    sections = []
    if ingredients:
        sections.append("## 原料")
        sections.append("\n".join([f"- {item}" for item in ingredients]))
    if steps:
        sections.append("## 步骤")
        sections.append("\n".join([f"{idx}. {item}" for idx, item in enumerate(steps, 1)]))
    if tips:
        sections.append("## 注意事项")
        sections.append("\n".join([f"- {item}" for item in tips]))
    return "\n\n".join(sections).strip()


def _build_recipe_from_parsed(ingredients: List[str], steps: List[str]) -> Optional[str]:
    if not ingredients or not steps:
        return None
    sections = [
        "## 原料",
        "\n".join([f"- {item}" for item in ingredients]),
        "## 步骤",
        "\n".join([f"{idx}. {item}" for idx, item in enumerate(steps, 1)]),
    ]
    return "\n\n".join(sections).strip()


def _build_answer_from_extraction(extraction: Dict) -> Optional[str]:
    fields = extraction.get("fields", {}) if isinstance(extraction, dict) else {}
    if not isinstance(fields, dict):
        return None
    texts: List[str] = []
    for field_name in fields:
        items = fields.get(field_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n".join(texts).strip() if texts else None


def _build_output_from_state(
    query: str,
    trace_id: str,
    state: RetrievalState,
    *,
    include_candidates: bool = False,
    session_id: Optional[str] = None,
    turn: Optional[int] = None,
    generation_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    llm_call_extractor: Optional[LLMCall] = None,
    llm_call_polish: Optional[LLMCall] = None,
) -> Dict:
    parents = state.parents
    parent_lock = state.parent_lock
    evidence_set = state.evidence_set

    if parent_lock.status == "locked" and parents:
        sufficient, missing = evidence_sufficient(evidence_set)
        if sufficient:
            output_intent = "full_recipe"
            evidence_chunks = evidence_set.get("chunks", []) if evidence_set else []
            block_types = sorted(
                {normalize_block_type(c.get("block_type")) for c in evidence_chunks if c.get("block_type")}
            )
            answer = None
            answer_source = None
            if llm_call_extractor:
                payload = build_extraction_payload(evidence_set, "FULL_RECIPE", evidence_scope="full")
                extraction, llm_error = call_llm_extract_with_debug(llm_call_extractor, payload)
                ok, errors = validate_extraction(extraction, evidence_set, expected_intent="FULL_RECIPE")
                if ok:
                    answer = _build_recipe_from_extraction(extraction or {})
                    if not answer:
                        ok = False
                        errors = ["empty_fields"]
                    else:
                        answer_source = "llm_extract"
                log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                if log_path:
                    fallback_reason = None
                    if not ok:
                        if "citation_chunk_id_invalid" in errors or "citation_quote_not_found" in errors:
                            fallback_reason = "bad_citation"
                        elif "number_out_of_bounds" in errors:
                            fallback_reason = "out_of_bounds"
                        elif "empty_fields" in errors:
                            fallback_reason = "empty_fields"
                        elif "intent_mismatch" in errors or "fields_missing_or_invalid" in errors:
                            fallback_reason = "invalid_schema"
                        else:
                            fallback_reason = "validation_failed"
                    log_llm_call(
                        {
                            "span_name": "llm_call",
                            "trace_id": trace_id,
                            "stage": "extract",
                            "intent": "FULL_RECIPE",
                            "evidence_scope": "full",
                            "llm_called": True,
                            "llm_success": ok,
                            "fallback_used": not ok,
                            "fallback_reason": fallback_reason,
                            "fallback_target": "rule_extract" if not ok else None,
                            "llm_error": llm_error,
                        },
                        log_path=log_path,
                    )
                if not ok:
                    answer = None

            if not answer:
                op_texts = [
                    c.get("text", "")
                    for c in evidence_chunks
                    if normalize_block_type(c.get("block_type")) == "operation"
                ]
                ing_texts = [
                    c.get("text", "")
                    for c in evidence_chunks
                    if normalize_block_type(c.get("block_type")) == "ingredients"
                ]
                parsed_steps = parse_steps(op_texts)
                parsed_ingredients = parse_ingredients(ing_texts)
                answer = _build_recipe_from_parsed(parsed_ingredients, parsed_steps)
                if answer:
                    answer_source = "rule_extract"
            if not answer:
                answer = format_auto_answer(evidence_set)
                answer_source = "format_auto_answer"

            polish_output_ts = None
            if llm_call_polish and answer:
                polish_output_ts = time.time()
                polish_payload = build_polish_payload(answer, intent="FULL_RECIPE")
                polished, llm_error = call_llm_polish_with_debug(llm_call_polish, polish_payload)
                ok, errors = validate_polish(answer, polished)
                log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                fallback_reason = None
                if not ok:
                    if "number_out_of_bounds" in errors:
                        fallback_reason = "out_of_bounds"
                    elif "polished_empty" in errors:
                        fallback_reason = "empty_output"
                    else:
                        fallback_reason = "validation_failed"
                log_llm_call(
                    {
                        "span_name": "llm_call",
                        "trace_id": trace_id,
                        "stage": "polish",
                        "intent": "FULL_RECIPE",
                        "evidence_scope": "full",
                        "llm_called": True,
                        "llm_success": ok,
                        "fallback_used": not ok,
                        "fallback_reason": fallback_reason,
                        "fallback_target": "use_draft" if not ok else None,
                        "llm_error": llm_error,
                    },
                    log_path=log_path,
                )
                if ok and polished:
                    answer = polished

            start_ts = time.time()
            if generation_log_path:
                log_generation_started(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": turn,
                        "mode": "single_turn",
                        "query": query,
                        "output_intent": output_intent,
                        "decision": {
                            "state": "AUTO_RECOMMEND",
                            "layer_used": 2,
                            "intent": "FULL_RECIPE",
                            "intent_conf": None,
                            "upgraded_to_layer2": True,
                            "upgrade_reason": None,
                        },
                        "lock": {
                            "status": "locked",
                            "parent_id": parent_lock.parent_id,
                            "lock_reason": parent_lock.lock_reason,
                            "lock_score": parent_lock.lock_score,
                            "locked_at_turn": parent_lock.locked_at_turn,
                        },
                        "evidence": {
                            "parent_id": parent_lock.parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                            "block_types": block_types,
                            "size": len(evidence_chunks),
                        },
                        "scoring": {
                            "top1_overall_score": parents[0].overall_score if parents else None,
                            "top2_overall_score": parents[1].overall_score if len(parents) > 1 else None,
                            "ratio12": (parents[1].overall_score / parents[0].overall_score)
                            if len(parents) > 1 and parents[0].overall_score > 0
                            else None,
                        },
                    },
                    log_path=generation_log_path,
                )
                mapping = build_generation_mapping(output_intent, evidence_set)
                log_generation_mapping(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": turn,
                        "mapping_strategy": "by_block_type_v1",
                        "sections": mapping,
                    },
                    log_path=generation_log_path,
                )
                log_generation_completed(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": turn,
                        "status": "ok",
                        "finish_reason": "ok",
                        "latency_ms": int((time.time() - start_ts) * 1000),
                        "output": {
                            "format": "markdown",
                            "sections": [m["section"] for m in mapping],
                            "char_count": len(answer),
                            "preview": answer[:200],
                        },
                        "answer_source": answer_source,
                        "evidence": {
                            "parent_id": parent_lock.parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                        },
                        "error": {"type": None, "message": None},
                    },
                    log_path=generation_log_path,
                )
                if polish_output_ts:
                    print(f"润色后到输出耗时: {time.time() - polish_output_ts:.2f}s")
            return {
                "trace_id": trace_id,
                "state": "AUTO_RECOMMEND",
                "parent_id": parent_lock.parent_id,
                "lock_status": "locked",
                "answer": answer,
            }
        if generation_log_path:
            evidence_chunks = evidence_set.get("chunks", []) if evidence_set else []
            log_generation_completed(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "turn": turn,
                    "status": "refused",
                    "finish_reason": "evidence_insufficient",
                    "latency_ms": 0,
                    "output": {
                        "format": "markdown",
                        "sections": [],
                        "char_count": 0,
                        "preview": "",
                    },
                    "evidence": {
                        "parent_id": parent_lock.parent_id,
                        "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                    },
                    "error": {"type": None, "message": None},
                },
                log_path=generation_log_path,
            )
        state = (
            "EVIDENCE_INSUFFICIENT"
            if any(m in {"Ingredients", "Operation"} for m in missing)
            else "LOW_EVIDENCE"
        )
        return {
            "trace_id": trace_id,
            "state": state,
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
    dir_category: Optional[str] = None,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
    llm_call_extractor: Optional[LLMCall] = None,
    llm_call_polish: Optional[LLMCall] = None,
) -> Dict:
    detected_categories = detect_categories(query)
    if len(detected_categories) > 1:
        pending = [
            {"option_id": str(idx), "category": cat} for idx, cat in enumerate(sorted(detected_categories), 1)
        ]
        return {
            "trace_id": trace_id,
            "state": "GOOD_BUT_AMBIGUOUS",
            "lock_status": "pending",
            "pending_categories": pending,
            "clarify_question": f"你是想看【{' / '.join(sorted(detected_categories))}】哪一类？",
        }
    dir_category = detected_categories[0] if detected_categories else None
    state = _retrieve_state(
        query=query,
        db_path=db_path,
        collection_name=collection_name,
        top_k=top_k,
        top_parents=top_parents,
        dir_category=dir_category,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=turn,
    )
    return _build_output_from_state(
        query,
        trace_id,
        state,
        include_candidates=False,
        session_id=None,
        turn=turn,
        generation_log_path=DEFAULT_GENERATION_LOG,
        evidence_log_path=evidence_log_path,
        llm_call_extractor=llm_call_extractor,
        llm_call_polish=llm_call_polish,
    )


def run_session_once(
    query: str,
    trace_id: str,
    session_id: str,
    *,
    db_path: Path,
    collection_name: str,
    top_k: int = 25,
    top_parents: int = 5,
    dir_category: Optional[str] = None,
    log_path: Optional[Path] = None,
    lock_log_path: Optional[Path] = None,
    evidence_log_path: Optional[Path] = None,
    evidence_insufficient: Optional[Dict] = None,
    turn: Optional[int] = None,
    llm_call_extractor: Optional[LLMCall] = None,
    llm_call_polish: Optional[LLMCall] = None,
) -> Dict:
    print("正在对相关文档进行排序...")
    phase_ts = time.time()
    session = load_session(session_id)
    now = time.time()
    last_turn = session.get("turn") or 0
    current_turn = turn if turn is not None else last_turn + 1
    session["turn"] = current_turn

    purge_expired_pending(session)

    pending_categories = session.get("pending_categories") or []
    selected_category: Optional[str] = None
    if pending_categories:
        selected_category = resolve_category_choice(query, pending_categories)
        if selected_category:
            session["pending_categories"] = []
            session["updated_at"] = now
            save_session(session)
        else:
            return {
                "trace_id": trace_id,
                "state": "GOOD_BUT_AMBIGUOUS",
                "lock_status": "pending",
                "pending_categories": pending_categories,
                "clarify_question": f"你是想看【{' / '.join([c['category'] for c in pending_categories])}】哪一类？",
            }

    option_id = parse_option_id(query, session.get("pending_candidates", []))
    if option_id:
        candidate = next(
            (c for c in session.get("pending_candidates", []) if c.get("option_id") == option_id), None
        )
        if candidate:
            print("已对文档进行锁定。")
            print(f"锁定耗时: {time.time() - phase_ts:.2f}s")
            phase_ts = time.time()
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
            print(f"证据加载耗时: {time.time() - phase_ts:.2f}s")
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
                "lock_reason": "user_select",
            }
            session["pending_candidates"] = []
            session["pending_categories"] = []
            session["last_trace_id"] = trace_id
            session["updated_at"] = now
            save_session(session)
            if sufficient:
                print("正在抽取证据链...")
                if llm_call_extractor:
                    print("LLM 正在抽取信息...")
                phase_ts = time.time()
                start_ts = time.time()
                evidence_chunks = evidence_set.get("chunks", []) if evidence_set else []
                block_types = sorted(
                    {normalize_block_type(c.get("block_type")) for c in evidence_chunks if c.get("block_type")}
                )
                output_intent = "full_recipe"
                log_generation_started(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "mode": "session_followup",
                        "query": query,
                        "output_intent": output_intent,
                        "decision": {
                            "state": "AUTO_RECOMMEND",
                            "layer_used": 2,
                            "intent": "FULL_RECIPE",
                            "intent_conf": None,
                            "upgraded_to_layer2": False,
                            "upgrade_reason": None,
                        },
                        "lock": {
                            "status": "locked",
                            "parent_id": parent_id,
                            "lock_reason": "user_select",
                            "lock_score": lock_score,
                            "locked_at_turn": current_turn,
                        },
                        "evidence": {
                            "parent_id": parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                            "block_types": block_types,
                            "size": len(evidence_chunks),
                        },
                        "scoring": {
                            "top1_overall_score": lock_score,
                            "top2_overall_score": None,
                            "ratio12": None,
                        },
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                mapping = build_generation_mapping(output_intent, evidence_set)
                log_generation_mapping(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "mapping_strategy": "by_block_type_v1",
                        "sections": mapping,
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                answer = format_auto_answer(evidence_set)
                print(f"抽取/生成耗时: {time.time() - phase_ts:.2f}s")
                polish_output_ts = None
                if llm_call_polish:
                    print("LLM 正在润色答案...")
                    phase_ts = time.time()
                    polish_output_ts = phase_ts
                    polish_payload = build_polish_payload(answer, intent="FULL_RECIPE")
                    polished, llm_error = call_llm_polish_with_debug(llm_call_polish, polish_payload)
                    ok, errors = validate_polish(answer, polished)
                    log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                    fallback_reason = None
                    if not ok:
                        if "number_out_of_bounds" in errors:
                            fallback_reason = "out_of_bounds"
                        elif "polished_empty" in errors:
                            fallback_reason = "empty_output"
                        else:
                            fallback_reason = "validation_failed"
                    log_llm_call(
                        {
                            "span_name": "llm_call",
                            "trace_id": trace_id,
                            "stage": "polish",
                            "intent": "FULL_RECIPE",
                            "evidence_scope": "full",
                            "llm_called": True,
                            "llm_success": ok,
                            "fallback_used": not ok,
                            "fallback_reason": fallback_reason,
                            "fallback_target": "use_draft" if not ok else None,
                        },
                        log_path=log_path,
                    )
                    if ok and polished:
                        answer = polished
                log_generation_completed(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "status": "ok",
                        "finish_reason": "ok",
                        "latency_ms": int((time.time() - start_ts) * 1000),
                        "output": {
                            "format": "markdown",
                            "sections": [m["section"] for m in mapping],
                            "char_count": len(answer),
                            "preview": answer[:200],
                        },
                        "answer_source": "format_auto_answer",
                        "evidence": {
                            "parent_id": parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                        },
                        "error": {"type": None, "message": None},
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                if llm_call_polish:
                    print(f"润色耗时: {time.time() - phase_ts:.2f}s")
                if polish_output_ts:
                    print(f"润色后到输出耗时: {time.time() - polish_output_ts:.2f}s")
                return {
                    "trace_id": trace_id,
                    "state": "AUTO_RECOMMEND",
                    "parent_id": parent_id,
                    "lock_status": "locked",
                    "answer": answer,
                }
            log_generation_completed(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "turn": current_turn,
                    "status": "refused",
                    "finish_reason": "evidence_insufficient",
                    "latency_ms": 0,
                    "output": {
                        "format": "markdown",
                        "sections": [],
                        "char_count": 0,
                        "preview": "",
                    },
                    "evidence": {
                        "parent_id": parent_id,
                        "chunk_ids": [
                            c.get("chunk_id") for c in evidence_set.get("chunks", []) if c.get("chunk_id")
                        ],
                    },
                    "error": {"type": None, "message": None},
                },
                log_path=DEFAULT_GENERATION_LOG,
            )
            state = (
                "EVIDENCE_INSUFFICIENT"
                if any(m in {"Ingredients", "Operation"} for m in missing)
                else "LOW_EVIDENCE"
            )
            return {
                "trace_id": trace_id,
                "state": state,
                "lock_status": "locked",
                "message": "evidence_insufficient",
                "missing_block_types": missing,
                "clarify_question": default_clarify_question(query),
            }

    if session.get("parent_lock", {}).get("status") == "locked":
        parent_id = session.get("parent_lock", {}).get("parent_id")
        if parent_id:
            print("已对文档进行锁定。")
            print(f"锁定耗时: {time.time() - phase_ts:.2f}s")
            phase_ts = time.time()
            print("正在加载锁定文档的证据...")
            evidence_set = build_evidence_set_for_parent(db_path, collection_name, parent_id)
            if evidence_log_path:
                log_evidence_built(query, evidence_set, log_path=evidence_log_path)
            print(f"证据加载耗时: {time.time() - phase_ts:.2f}s")

            routing = classify_query(query)
            intent = routing["intent"]
            confidence = routing["confidence"]
            slots = routing["slots"]
            print(f"意图识别完成：{intent}（置信度 {confidence:.2f}）。")
            print(f"意图识别详情: intent={intent}, confidence={confidence:.2f}, slots={slots}")
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
            answer_source = None
            missing_slots: List[str] = []
            if confidence < intent_min:
                routing_log["upgraded_to_layer2"] = True
                routing_log["upgrade_reason"] = "low_confidence"
                print("意图置信度较低，已升级到全量证据。")
            elif not evidence_layer1 or not evidence_layer1.get("chunks"):
                routing_log["upgraded_to_layer2"] = True
                routing_log["upgrade_reason"] = "missing_block"
                print("意图相关块不足，已升级到全量证据。")
            else:
                print("正在抽取证据链...")
                phase_ts = time.time()
                if llm_call_extractor:
                    print("LLM 正在抽取信息...")
                    payload = build_extraction_payload(evidence_layer1, intent, evidence_scope="layer1")
                    extraction, llm_error = call_llm_extract_with_debug(llm_call_extractor, payload)
                    ok, errors = validate_extraction(extraction, evidence_layer1, expected_intent=intent)
                    if ok:
                        answer = _build_answer_from_extraction(extraction or {})
                        if not answer:
                            ok = False
                            errors = ["empty_fields"]
                        else:
                            answer_source = "llm_extract_layer1"
                    log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                    fallback_reason = None
                    if not ok:
                        if "citation_chunk_id_invalid" in errors or "citation_quote_not_found" in errors:
                            fallback_reason = "bad_citation"
                        elif "number_out_of_bounds" in errors:
                            fallback_reason = "out_of_bounds"
                        elif "empty_fields" in errors:
                            fallback_reason = "empty_fields"
                        elif "intent_mismatch" in errors or "fields_missing_or_invalid" in errors:
                            fallback_reason = "invalid_schema"
                        else:
                            fallback_reason = "validation_failed"
                    log_llm_call(
                        {
                            "span_name": "llm_call",
                            "trace_id": trace_id,
                            "stage": "extract",
                            "intent": intent,
                            "evidence_scope": "layer1",
                            "llm_called": True,
                            "llm_success": ok,
                            "fallback_used": not ok,
                            "fallback_reason": fallback_reason,
                            "fallback_target": "rule_generate" if not ok else None,
                            "llm_error": llm_error,
                        },
                        log_path=log_path,
                    )
                    if not ok:
                        answer = None

                if not answer:
                    print("正在生成回答...")
                    answer, missing_slots = generate_answer(
                        intent,
                        slots,
                        evidence_layer1,
                        parsed_steps=cached_steps,
                        parsed_ingredients=cached_ingredients,
                    )
                    if answer:
                        answer_source = "rule_generate_layer1"
                print(f"抽取/生成耗时: {time.time() - phase_ts:.2f}s")
                routing_log["final_evidence_chunk_ids"] = [
                    c.get("chunk_id") for c in evidence_layer1.get("chunks", [])
                ]
                if (not answer) or missing_slots:
                    routing_log["upgraded_to_layer2"] = True
                    routing_log["upgrade_reason"] = "no_hit_sentence"
                    print("局部证据未命中，升级到全量证据。")

            if routing_log["upgraded_to_layer2"]:
                print("正在抽取证据链（扩展范围）...")
                phase_ts = time.time()
                if llm_call_extractor:
                    print("LLM 正在抽取信息...")
                    payload = build_extraction_payload(evidence_set, intent, evidence_scope="layer2")
                    extraction, llm_error = call_llm_extract_with_debug(llm_call_extractor, payload)
                    ok, errors = validate_extraction(extraction, evidence_set, expected_intent=intent)
                    if ok:
                        answer = _build_answer_from_extraction(extraction or {})
                        if not answer:
                            ok = False
                            errors = ["empty_fields"]
                        else:
                            answer_source = "llm_extract_layer2"
                    log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                    fallback_reason = None
                    if not ok:
                        if "citation_chunk_id_invalid" in errors or "citation_quote_not_found" in errors:
                            fallback_reason = "bad_citation"
                        elif "number_out_of_bounds" in errors:
                            fallback_reason = "out_of_bounds"
                        elif "empty_fields" in errors:
                            fallback_reason = "empty_fields"
                        elif "intent_mismatch" in errors or "fields_missing_or_invalid" in errors:
                            fallback_reason = "invalid_schema"
                        else:
                            fallback_reason = "validation_failed"
                    log_llm_call(
                        {
                            "span_name": "llm_call",
                            "trace_id": trace_id,
                            "stage": "extract",
                            "intent": intent,
                            "evidence_scope": "layer2",
                            "llm_called": True,
                            "llm_success": ok,
                            "fallback_used": not ok,
                            "fallback_reason": fallback_reason,
                            "fallback_target": "rule_generate" if not ok else None,
                            "llm_error": llm_error,
                        },
                        log_path=log_path,
                    )
                    if not ok:
                        answer = None

                if not answer:
                    print("正在生成回答...")
                    answer, missing_slots = generate_answer(
                        intent,
                        slots,
                        evidence_set,
                        parsed_steps=cached_steps,
                        parsed_ingredients=cached_ingredients,
                    )
                    if answer:
                        answer_source = "rule_generate_layer2"
                print(f"抽取/生成耗时: {time.time() - phase_ts:.2f}s")
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
                polish_output_ts = None
                if llm_call_polish:
                    print("LLM 正在润色答案...")
                    polish_output_ts = time.time()
                    polish_payload = build_polish_payload(answer, intent=intent)
                    polished, llm_error = call_llm_polish_with_debug(llm_call_polish, polish_payload)
                    ok, errors = validate_polish(answer, polished)
                    log_path = evidence_log_path or DEFAULT_EVIDENCE_LOG
                    fallback_reason = None
                    if not ok:
                        if "number_out_of_bounds" in errors:
                            fallback_reason = "out_of_bounds"
                        elif "polished_empty" in errors:
                            fallback_reason = "empty_output"
                        else:
                            fallback_reason = "validation_failed"
                    log_llm_call(
                        {
                            "span_name": "llm_call",
                            "trace_id": trace_id,
                            "stage": "polish",
                            "intent": intent,
                            "evidence_scope": "layer2" if routing_log["upgraded_to_layer2"] else "layer1",
                            "llm_called": True,
                            "llm_success": ok,
                            "fallback_used": not ok,
                            "fallback_reason": fallback_reason,
                            "fallback_target": "use_draft" if not ok else None,
                            "llm_error": llm_error,
                        },
                        log_path=log_path,
                    )
                    if ok and polished:
                        answer = polished

                start_ts = time.time()
                output_intent = output_intent_for(intent, slots)
                final_evidence = evidence_set if routing_log["upgraded_to_layer2"] else evidence_layer1
                evidence_chunks = final_evidence.get("chunks", []) if final_evidence else []
                block_types = sorted(
                    {normalize_block_type(c.get("block_type")) for c in evidence_chunks if c.get("block_type")}
                )
                log_generation_started(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "mode": "session_followup",
                        "query": query,
                        "output_intent": output_intent,
                        "decision": {
                            "state": "AUTO_RECOMMEND",
                            "layer_used": 2 if routing_log["upgraded_to_layer2"] else 1,
                            "intent": intent,
                            "intent_conf": confidence,
                            "upgraded_to_layer2": routing_log["upgraded_to_layer2"],
                            "upgrade_reason": routing_log["upgrade_reason"],
                        },
                        "lock": {
                            "status": "locked",
                            "parent_id": parent_id,
                            "lock_reason": session.get("parent_lock", {}).get("lock_reason"),
                            "lock_score": session.get("parent_lock", {}).get("lock_score"),
                            "locked_at_turn": current_turn,
                        },
                        "evidence": {
                            "parent_id": parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                            "block_types": block_types,
                            "size": len(evidence_chunks),
                        },
                        "scoring": {
                            "top1_overall_score": session.get("parent_lock", {}).get("lock_score"),
                            "top2_overall_score": None,
                            "ratio12": None,
                        },
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                mapping = build_generation_mapping(output_intent, final_evidence)
                log_generation_mapping(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "mapping_strategy": "by_block_type_v1",
                        "sections": mapping,
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                log_generation_completed(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "turn": current_turn,
                        "status": "ok",
                        "finish_reason": "ok",
                        "latency_ms": int((time.time() - start_ts) * 1000),
                        "output": {
                            "format": "markdown",
                            "sections": [m["section"] for m in mapping],
                            "char_count": len(answer),
                            "preview": answer[:200],
                        },
                        "answer_source": answer_source,
                        "evidence": {
                            "parent_id": parent_id,
                            "chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
                        },
                        "error": {"type": None, "message": None},
                    },
                    log_path=DEFAULT_GENERATION_LOG,
                )
                if polish_output_ts:
                    print(f"润色后到输出耗时: {time.time() - polish_output_ts:.2f}s")
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
            log_generation_completed(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "turn": current_turn,
                    "status": "refused",
                    "finish_reason": "evidence_insufficient",
                    "latency_ms": 0,
                    "output": {
                        "format": "markdown",
                        "sections": [],
                        "char_count": 0,
                        "preview": "",
                    },
                    "evidence": {
                        "parent_id": parent_id,
                        "chunk_ids": [
                            c.get("chunk_id") for c in evidence_set.get("chunks", []) if c.get("chunk_id")
                        ],
                    },
                    "error": {"type": None, "message": None},
                },
                log_path=DEFAULT_GENERATION_LOG,
            )
            state = (
                "EVIDENCE_INSUFFICIENT"
                if any(m in {"Ingredients", "Operation"} for m in missing)
                else "LOW_EVIDENCE"
            )
            return {
                "trace_id": trace_id,
                "state": state,
                "lock_status": "locked",
                "message": "evidence_insufficient",
                "missing_block_types": missing,
                "clarify_question": "该菜谱未提及这一点，是否需要切换做法？",
            }

    detected_categories = detect_categories(query)
    if len(detected_categories) > 1:
        pending = [
            {"option_id": str(idx), "category": cat} for idx, cat in enumerate(sorted(detected_categories), 1)
        ]
        session["parent_lock"] = {
            "status": "pending",
            "parent_id": None,
            "pending_reason": "category_conflict",
            "lock_score": None,
            "lock_reason": None,
        }
        session["pending_categories"] = pending
        session["pending_candidates"] = []
        session["parent_cache"] = None
        session["last_trace_id"] = trace_id
        session["updated_at"] = now
        save_session(session)
        return {
            "trace_id": trace_id,
            "state": "GOOD_BUT_AMBIGUOUS",
            "lock_status": "pending",
            "pending_categories": pending,
            "clarify_question": f"你是想看【{' / '.join(sorted(detected_categories))}】哪一类？",
        }
    if detected_categories:
        dir_category = detected_categories[0]
    else:
        dir_category = selected_category

    state = _retrieve_state(
        query=query,
        db_path=db_path,
        collection_name=collection_name,
        top_k=top_k,
        top_parents=top_parents,
        dir_category=dir_category,
        log_path=log_path,
        lock_log_path=lock_log_path,
        evidence_log_path=evidence_log_path,
        evidence_insufficient=evidence_insufficient,
        turn=current_turn,
    )
    if state.parent_lock.status == "locked":
        print("已对文档进行锁定。")
        print(f"锁定耗时: {time.time() - phase_ts:.2f}s")
    output = _build_output_from_state(
        query,
        trace_id,
        state,
        include_candidates=True,
        evidence_log_path=evidence_log_path,
        llm_call_extractor=llm_call_extractor,
        llm_call_polish=llm_call_polish,
    )
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
            "lock_reason": None,
        }
        session["pending_candidates"] = pending_candidates
        session["parent_cache"] = None
        session["pending_categories"] = []
    else:
        session["parent_lock"] = {
            "status": state.parent_lock.status,
            "parent_id": state.parent_lock.parent_id,
            "pending_reason": state.parent_lock.pending_reason,
            "lock_score": state.parent_lock.lock_score,
            "lock_reason": state.parent_lock.lock_reason,
        }
        session["pending_candidates"] = []
        session["pending_categories"] = []
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
    parser.add_argument("--collection", type=str, default="cook_chunks", help="Collection name.")
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
    parser.add_argument("--llm", action="store_true", help="Enable LLM extraction and polish.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_call_extractor = None
    llm_call_polish = None
    if args.llm:
        env_path = Path(".env")
        env_vars: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
        model = env_vars.get("LLM_MODEL") or os.getenv("LLM_MODEL")
        api_key = env_vars.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
        if not model or not api_key:
            print("LLM is enabled but LLM_MODEL or LLM_API_KEY is missing; falling back to rules.")
        else:
            llm_call_extractor = lambda payload: llm_extract(payload, model=model, api_key=api_key)
            llm_call_polish = lambda payload: llm_polish(payload, model=model, api_key=api_key)
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
            llm_call_extractor=llm_call_extractor,
            llm_call_polish=llm_call_polish,
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
            llm_call_extractor=llm_call_extractor,
            llm_call_polish=llm_call_polish,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
