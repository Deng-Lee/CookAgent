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
import os
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

LOCAL_MODEL_ENV = "LOCAL_BGE_M3_PATH"
DEFAULT_LOCK_LOG = Path("logs/parent_locking.log")
DEFAULT_EVIDENCE_LOG = Path("logs/evidence_driven.log")


def resolve_local_model_path() -> str:
    """
    Resolve local path to BAAI/bge-m3 snapshot to avoid network calls.
    Priority:
    1) Env LOCAL_BGE_M3_PATH if set and exists.
    2) Latest snapshot under ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<hash> that contains config.json
    3) Fallback to ~/.cache/huggingface/hub/models--BAAI--bge-m3 if it contains config.json
    """
    env_path = os.environ.get(LOCAL_MODEL_ENV)
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return str(p)
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
    snapshots_root = root / "snapshots"
    if snapshots_root.exists():
        snapshots = sorted(
            [p for p in snapshots_root.glob("*") if (p / "config.json").exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    if root.exists() and (root / "config.json").exists():
        return str(root)
    raise FileNotFoundError(
        f"Local BGE model not found. Set {LOCAL_MODEL_ENV} to the model directory or ensure HF cache exists."
    )


def log_retrieval(query: str, res: Dict, parents: List[ParentHit], log_path: Path = Path("logs/retriever.log")) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ids = res["ids"][0]
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    vector_topk = []
    seen_chunk_ids = set()
    for idx, (hit_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        if hit_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(hit_id)
        vector_topk.append(
            {
                "chunk_id": hit_id,
                "parent_id": meta.get("parent_id"),
                "block_type": meta.get("category"),
                "rank": idx,
                "score": _distance_to_score(dist),
                "snippet_80": doc.replace("\n", " ")[:80],
            }
        )
    parent_candidates = []
    for ph in parents:
        flags = []
        if ph.low_evidence:
            flags.append("LOW_EVIDENCE")
        if ph.good_but_ambiguous:
            flags.append("GOOD_BUT_AMBIGUOUS")
        if ph.auto_recommend:
            flags.append("AUTO_RECOMMEND")
        parent_candidates.append(
            {
                "parent_id": ph.parent_id,
                "dish_name": ph.hits[0].metadata.get("dish_name") if ph.hits else None,
                "rrf_sum_score": ph.rrf_sum,
                "coverage_score": ph.coverage_ratio,
                "coverage_raw": ph.coverage,
                "total_chunks": ph.total_chunks,
                "max_chunk_score": ph.max_chunk_score,
                "rrf_sum_norm": ph.rrf_sum_norm,
                "max_chunk_norm": ph.max_chunk_norm,
                "overall_score": ph.overall_score,
                "flags": flags,
                "hit_block_types": sorted({h.metadata.get("category") for h in ph.hits if h.metadata.get("category")}),
                "top_evidence_snippets": [h.text.replace("\n", " ")[:80] for h in ph.hits[:3]],
            }
        )
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "span_name": "retriever_vector",
        "query": query,
        "topk": len(vector_topk),
        "vector_topk": vector_topk,
        "parent_candidates": parent_candidates,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_parent_lock(
    query: str,
    parent_lock: ParentLock,
    parents: List[ParentHit],
    log_path: Path = DEFAULT_LOCK_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if parent_lock.status == "pending":
        candidates = [
            {
                "parent_id": ph.parent_id,
                "dish_name": ph.hits[0].metadata.get("dish_name") if ph.hits else None,
                "overall_score": ph.overall_score,
                "top_evidence_snippets": [h.text.replace("\n", " ")[:80] for h in ph.hits[:3]],
            }
            for ph in parents
        ]
    else:
        candidates = []
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "span_name": "parent_lock",
        "query": query,
        "event": "lock_created"
        if parent_lock.status == "locked"
        else "lock_pending"
        if parent_lock.status == "pending"
        else "lock_none",
        "parent_lock": {
            "status": parent_lock.status,
            "parent_id": parent_lock.parent_id,
            "lock_reason": parent_lock.lock_reason,
            "lock_score": parent_lock.lock_score,
            "locked_at_turn": parent_lock.locked_at_turn,
            "pending_reason": parent_lock.pending_reason,
        },
        "candidates": candidates,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_evidence_set(parent: ParentHit) -> Dict:
    seen_chunk_ids = set()
    chunks = []
    for hit in parent.hits:
        chunk_id = hit.metadata.get("chunk_id")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "block_type": hit.metadata.get("category"),
                "text": hit.text,
                "score": hit.score,
            }
        )
    return {
        "parent_id": parent.parent_id,
        "chunks": chunks,
    }


def log_evidence_event(
    query: str,
    evidence_set: Dict,
    log_path: Path = DEFAULT_EVIDENCE_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_ids = [c.get("chunk_id") for c in evidence_set.get("chunks", [])]
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "span_name": "evidence_built",
        "query": query,
        "evidence_set": {
            "parent_id": evidence_set.get("parent_id"),
            "chunk_ids": chunk_ids,
        },
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_evidence_insufficient(
    query: str,
    turn: Optional[int],
    parent_lock: ParentLock,
    evidence_set: Dict,
    payload: Dict,
    log_path: Path = DEFAULT_EVIDENCE_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_chunks = evidence_set.get("chunks", [])
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "evidence_insufficient",
        "turn": turn,
        "query": query,
        "locked_parent_id": parent_lock.parent_id,
        "output_intent": payload.get("output_intent"),
        "evidence_chunk_ids": [c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")],
        "evidence_block_types": [
            c.get("block_type") for c in evidence_chunks if c.get("block_type") is not None
        ],
        "missing_block_types": payload.get("missing_block_types", []),
        "missing_slots": payload.get("missing_slots", []),
        "decision": payload.get("decision"),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _distance_to_score(distance: float) -> float:
    """Convert Chroma distance (cosine) to a similarity score."""
    return 1.0 / (1.0 + distance)


@dataclass
class ChunkHit:
    score: float
    rank: int
    text: str
    metadata: Dict


@dataclass
class ParentHit:
    parent_id: str
    rrf_sum: float = 0.0
    max_chunk_score: float = 0.0
    coverage: int = 0  # hit count
    total_chunks: int = 0
    coverage_ratio: float = 0.0
    rrf_sum_norm: float = 0.0
    max_chunk_norm: float = 0.0
    overall_score: float = 0.0
    low_evidence: bool = False
    good_but_ambiguous: bool = False
    auto_recommend: bool = False
    hits: List[ChunkHit] = field(default_factory=list)
    parent_doc: Optional[str] = None


@dataclass
class ParentLock:
    status: str  # locked | pending | none
    parent_id: Optional[str]
    lock_reason: Optional[str]
    lock_score: Optional[float]
    locked_at_turn: Optional[int]
    pending_reason: Optional[str]


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
            pending_reason="GOOD_BUT_AMBIGUOUS",
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
            log_evidence_event(query, evidence_set, log_path=evidence_log_path)
    if evidence_insufficient and parent_lock.status == "locked" and evidence_set:
        log_evidence_insufficient(
            query,
            turn,
            parent_lock,
            evidence_set,
            evidence_insufficient,
            log_path=evidence_log_path or DEFAULT_EVIDENCE_LOG,
        )
    return parents[:top_parents]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query or input("请输入问题: ").strip()
    if not query:
        print("未输入问题，退出。")
        return
    parents = retrieve(
        query=query,
        db_path=args.db_path,
        collection_name=args.collection,
        top_k=args.top_k,
        top_parents=args.top_parents,
        log_path=args.log_path,
        lock_log_path=args.lock_log_path,
        evidence_log_path=args.evidence_log_path,
        turn=args.turn,
    )
    for i, ph in enumerate(parents, 1):
        print(
            f"[parent {i}] id={ph.parent_id} "
            f"rrf_sum={ph.rrf_sum:.4f} max_score={ph.max_chunk_score:.4f} "
            f"coverage={ph.coverage} total_chunks={ph.total_chunks} coverage_ratio={ph.coverage_ratio:.4f}"
        )
        print(" top chunks:")
        for hit in ph.hits[:3]:
            cat = hit.metadata.get("category")
            text_preview = hit.text.replace("\n", " ")[:80]
            print(f"   - score={hit.score:.4f} rank={hit.rank} category={cat} text={text_preview}...")
        if ph.parent_doc:
            print(f" parent_doc_loaded: {len(ph.parent_doc)} chars")
        else:
            print(" parent_doc_loaded: None")
        print()


if __name__ == "__main__":
    main()
