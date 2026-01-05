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
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from splitter import Chunk, split_file


def iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield all markdown recipe files under root."""
    yield from root.rglob("*.md")


def chunk_records(path: Path, *, strict: bool) -> List[dict]:
    """Split one markdown file and build records for Chroma upsert."""
    chunks: List[Chunk] = split_file(path, strict=strict)
    records: List[dict] = []
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
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
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
