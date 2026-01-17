from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from retrieval_types import (
    ParentHit,
    ParentLock,
    DEFAULT_EVIDENCE_LOG,
    DEFAULT_LOCK_LOG,
    DEFAULT_GENERATION_LOG,
)


def log_retrieval(
    query: str,
    res: Dict,
    parents: List[ParentHit],
    *,
    detected_categories: Optional[List[str]] = None,
    category_filter_applied: bool = False,
    category_conflict: bool = False,
    category_filtered_count: Optional[int] = None,
    log_path: Path = Path("logs/retriever.log"),
) -> None:
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
                "score": 1.0 / (1.0 + dist),
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
        "detected_categories": detected_categories or [],
        "category_filter_applied": category_filter_applied,
        "category_conflict": category_conflict,
        "category_filtered_count": category_filtered_count,
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


def log_evidence_built(
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


def log_evidence_routing(
    trace_id: str,
    turn: Optional[int],
    query: str,
    routing: Dict,
    *,
    log_path: Path = DEFAULT_EVIDENCE_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "span_name": "evidence_routing",
        "trace_id": trace_id,
        "turn": turn,
        "query": query,
        "routing": routing,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_generation_started(
    record: Dict,
    log_path: Path = DEFAULT_GENERATION_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["event"] = "generation_started"
    record["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_generation_completed(
    record: Dict,
    log_path: Path = DEFAULT_GENERATION_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["event"] = "generation_completed"
    record["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_generation_mapping(
    record: Dict,
    log_path: Path = DEFAULT_GENERATION_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["event"] = "generation_mapping"
    record["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_llm_call(
    record: Dict,
    log_path: Path = DEFAULT_EVIDENCE_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.datetime.utcnow().isoformat() + "Z", **record}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
