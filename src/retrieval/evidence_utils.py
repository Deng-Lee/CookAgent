from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from retrieval_types import ParentHit
from model_utils import resolve_local_model_path
from generation_utils import normalize_block_type, BLOCK_CANONICAL_TO_ORIG


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


def build_evidence_set_for_parent(
    db_path: Path,
    collection_name: str,
    parent_id: str,
) -> Dict:
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
    res = collection.get(where={"parent_id": parent_id})
    chunks = []
    for hit_id, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
        chunks.append(
            {
                "chunk_id": hit_id,
                "block_type": meta.get("category") if meta else None,
                "text": doc,
                "score": None,
            }
        )
    return {"parent_id": parent_id, "chunks": chunks}


def evidence_sufficient(evidence_set: Optional[Dict]) -> Tuple[bool, List[str]]:
    if not evidence_set:
        return False, ["Ingredients", "Operation"]
    block_types = {
        normalize_block_type(c.get("block_type")) for c in evidence_set.get("chunks", []) if c.get("block_type")
    }
    required = {"ingredients", "operation"}
    missing = sorted(required - block_types)
    missing_orig = [BLOCK_CANONICAL_TO_ORIG[m] for m in missing]
    return not missing_orig, missing_orig


def format_auto_answer(evidence_set: Optional[Dict]) -> str:
    if not evidence_set:
        return "未找到可用证据。"
    grouped: Dict[str, List[str]] = {}
    for chunk in evidence_set.get("chunks", []):
        canonical = normalize_block_type(chunk.get("block_type")) or "tips"
        grouped.setdefault(canonical, []).append(chunk.get("text", ""))
    sections = []
    section_order = [
        ("ingredients", "原料"),
        ("operation", "步骤"),
        ("tips", "注意事项"),
    ]
    for block_type, label in section_order:
        items = grouped.get(block_type)
        if not items:
            continue
        sections.append(f"## {label}")
        sections.append("\n\n".join([it for it in items if it]))
    return "\n\n".join(sections).strip()


def extract_first_step(evidence_set: Optional[Dict]) -> Optional[str]:
    if not evidence_set:
        return None
    for chunk in evidence_set.get("chunks", []):
        if chunk.get("block_type") != "Operation":
            continue
        text = chunk.get("text", "")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            return stripped.lstrip("-*0123456789.、) ").strip() or stripped
    return None


def default_clarify_question(query: str) -> str:
    if query.strip():
        return "你想要哪一种做法或口味？"
    return "你想查询哪道菜的做法？"
