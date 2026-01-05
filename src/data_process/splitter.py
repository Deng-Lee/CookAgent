"""
Markdown splitter for GitCook-Agent.

Implements the 5-class structural splitting described in readme.md:
- Intro
- Ingredients
- Operation
- Calculation
- Additional

Each chunk is enriched with metadata header:
`[菜名: <name>] [类别: <category>] <content>`
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class ChunkCategory(str, Enum):
    INTRO = "Intro"
    INGREDIENTS = "Ingredients"
    OPERATION = "Operation"
    CALCULATION = "Calculation"
    ADDITIONAL = "Additional"


# Heuristic heading patterns mapped to structural categories.
CATEGORY_PATTERNS: Dict[ChunkCategory, Iterable[str]] = {
    ChunkCategory.INTRO: ("简介", "介绍", "概述"),
    ChunkCategory.INGREDIENTS: (
        "必备原料",
        "原料",
        "食材",
        "用料",
        "材料",
        "工具",
    ),
    ChunkCategory.OPERATION: ("操作", "步骤", "做法"),
    ChunkCategory.CALCULATION: ("计算", "用量", "时间"),
    ChunkCategory.ADDITIONAL: ("附加内容", "附加", "备注", "提示"),
}


@dataclass
class Chunk:
    parent_id: str
    category: ChunkCategory
    content: str
    enriched: str
    meta: Dict[str, str]


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _match_category(heading: str) -> Optional[ChunkCategory]:
    norm = _normalize_heading(heading)
    for category, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if pat in norm:
                return category
    return None


def _parse_title(lines: List[str]) -> Tuple[str, int]:
    """
    Returns (title, next_index_after_title_heading).
    Falls back to first non-empty line if no H1 heading exists.
    """
    for idx, line in enumerate(lines):
        m = re.match(r"^#\s+(.*)", line.strip())
        if m:
            return m.group(1).strip(), idx + 1
    for idx, line in enumerate(lines):
        if line.strip():
            return line.strip(), idx + 1
    return "未知菜品", 0


def _split_sections(
    lines: List[str],
    start_idx: int,
    strict: bool = False,
    source_path: Optional[str] = None,
) -> List[Tuple[Optional[ChunkCategory], List[str]]]:
    """
    Split markdown lines into sections keyed by optional category.
    Intro is captured as content before the first recognized category heading.

    If strict=True, encountering an H2 heading that does not map to a known category
    will raise, so that bad/misspelled headings can be fixed upstream.
    """
    sections: List[Tuple[Optional[ChunkCategory], List[str]]] = []
    current_category: Optional[ChunkCategory] = None
    current_lines: List[str] = []

    def push():
        if current_lines:
            sections.append((current_category, current_lines.copy()))

    for line in lines[start_idx:]:
        heading_match = re.match(r"^##\s+(.*)", line.strip())
        if heading_match:
            heading_text = heading_match.group(1).strip()
            cat = _match_category(heading_text)
            if cat is not None:
                push()
                current_category = cat
                current_lines = []
                continue
            if strict:
                raise ValueError(f"Unrecognized section heading: {heading_text}")
            else:
                print(
                    f"[warn] Unrecognized section heading '{heading_text}' in "
                    f"{source_path or 'unknown source'}, fallback to Additional."
                )
        current_lines.append(line.rstrip())

    push()
    return sections


def _build_chunk(
    dish_name: str,
    parent_id: str,
    category: ChunkCategory,
    lines: List[str],
    source_path: Optional[str],
) -> Chunk:
    content = "\n".join([ln for ln in lines if ln.strip()]).strip()
    enriched = f"[菜名: {dish_name}] [类别: {category.value}] {content}"
    meta = {
        "dish_name": dish_name,
        "category": category.value,
    }
    if source_path:
        meta["source_path"] = source_path
    return Chunk(
        parent_id=parent_id,
        category=category,
        content=content,
        enriched=enriched,
        meta=meta,
    )


def split_markdown(
    md_text: str,
    source_path: Optional[str] = None,
    parent_id: Optional[str] = None,
    *,
    strict: bool = False,
) -> List[Chunk]:
    """
    Split a markdown recipe into 5 structural chunks, enriching each with metadata.
    Missing sections are skipped; intro is built from text before the first recognized section heading.
    If strict=True, any H2 heading that does not map to a known category will raise.
    """
    lines = md_text.splitlines()
    dish_name, start_idx = _parse_title(lines)
    sections = _split_sections(lines, start_idx, strict=strict, source_path=source_path)

    chunks: List[Chunk] = []
    pid = parent_id or (source_path or dish_name)

    # Intro from content without explicit category before first heading.
    if sections and sections[0][0] is None:
        intro_lines = sections[0][1]
        if any(ln.strip() for ln in intro_lines):
            chunks.append(_build_chunk(dish_name, pid, ChunkCategory.INTRO, intro_lines, source_path))
        sections = sections[1:]

    for category, body in sections:
        cat = category or ChunkCategory.ADDITIONAL  # Preserve content even if heading not mapped.
        if not body or not any(ln.strip() for ln in body):
            continue
        chunks.append(_build_chunk(dish_name, pid, cat, body, source_path))

    return chunks


def split_file(path: str | Path, *, strict: bool = False) -> List[Chunk]:
    """Convenience helper to split a file path."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return split_markdown(text, source_path=str(p), parent_id=str(p), strict=strict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a recipe markdown into structured chunks.")
    parser.add_argument("path", help="Path to markdown file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if any H2 heading is not mapped to a known category.",
    )
    args = parser.parse_args()

    chunks = split_file(args.path, strict=args.strict)
    print(json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
