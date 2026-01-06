"""
Batch splitter + embedding builder for all recipe markdown files.

Workflow:
- Traverse recipe markdowns under a root folder.
-,Split each file into structured chunks using splitter.split_file.
-,Embed the enriched text (带菜名/类别前缀) and store into a Chroma collection.

Dependencies:
  pip install chromadb sentence-transformers
  (model: BAAI/bge-m3, aligns with readme.md)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from splitter import Chunk, split_file

LOCAL_MODEL_ENV = "LOCAL_BGE_M3_PATH"


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


def iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield all markdown recipe files under root."""
    yield from root.rglob("*.md")


def chunk_records(path: Path, *, strict: bool) -> List[dict]:
    """Split one markdown file and build records for Chroma upsert."""
    chunks: List[Chunk] = split_file(path, strict=strict)
    records: List[dict] = []
    total_chunks = len(chunks)
    # Inject a title-only chunk to boost dish name recall.
    if chunks:
        dish_name = chunks[0].meta.get("dish_name", path.stem)
        title_text = f"[菜名: {dish_name}] [类别: Title] {dish_name}"
        total_chunks_with_title = total_chunks + 1
        records.append(
            {
                "id": f"{path.stem}-title",
                "text": title_text,
                "metadata": {
                    "dish_name": dish_name,
                    "category": "Title",
                    "parent_id": str(path),
                    "chunk_index": -1,
                    "total_chunks": total_chunks_with_title,
                },
            }
        )
    total_chunks_with_title = total_chunks + (1 if chunks else 0)
    for idx, chunk in enumerate(chunks):
        doc_id = f"{path.stem}-{idx}-{uuid4().hex[:8]}"
        records.append(
            {
                "id": doc_id,
                "text": chunk.enriched,  # enriched text for embedding to reduce semantic drift.
                "metadata": {
                    **chunk.meta,
                    "parent_id": chunk.parent_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks_with_title,
                },
            }
        )
    return records


def build_collection(
    root: Path,
    db_path: Path,
    collection_name: str,
    *,
    strict: bool = False,
) -> None:
    """Traverse all recipes, split, and upsert embeddings into Chroma."""
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

    total_chunks = 0
    for md_file in iter_markdown_files(root):
        records = chunk_records(md_file, strict=strict)
        if not records:
            continue
        collection.upsert(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )
        total_chunks += len(records)
        print(f"[ok] {md_file} -> {len(records)} chunks")
    print(f"[done] total chunks: {total_chunks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch split + embed recipes into Chroma.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/cook/dishes"),
        help="Root folder containing recipe markdown files.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/chroma"),
        help="Chroma persistence path.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="cook_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown section headings to enforce markdown schema.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_collection(args.root, args.db_path, args.collection, strict=args.strict)


if __name__ == "__main__":
    main()
