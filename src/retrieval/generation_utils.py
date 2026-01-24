from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from config import (
    ASK_STEP_N_SCORE,
    ASK_STEPS_SCORE,
    ASK_INGREDIENTS_SCORE,
    ASK_INGREDIENTS_QUANTITY_SCORE,
    ASK_TIME_SCORE,
    ASK_HEAT_SCORE,
    ASK_SUBSTITUTION_SCORE,
    ASK_TIPS_SCORE,
)

_BLOCK_ALIASES = {
    "Intro": "intro",
    "Ingredients": "ingredients",
    "Operation": "operation",
    "Calculation": "calculation",
    "Additional": "tips",
    "Title": "title",
}

BLOCK_CANONICAL_TO_ORIG = {
    "ingredients": "Ingredients",
    "operation": "Operation",
    "tips": "Additional",
    "intro": "Intro",
    "calculation": "Calculation",
    "title": "Title",
}


def _parse_cn_number(text: str) -> Optional[int]:
    if not text:
        return None
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if text == "十":
        return 10
    if "十" in text:
        parts = text.split("十", 1)
        tens = mapping.get(parts[0], 1) if parts[0] else 1
        ones = mapping.get(parts[1], 0) if parts[1] else 0
        return tens * 10 + ones
    total = 0
    for ch in text:
        total += mapping.get(ch, 0)
    return total or None


def classify_query(query: str) -> Dict:
    text = query.strip()
    scores = {}
    slots: Dict[str, int] = {}

    if re.search(r"第(\d+)步", text):
        m = re.search(r"第(\d+)步", text)
        if m:
            slots["step_n"] = int(m.group(1))
        scores["ASK_STEP_N"] = ASK_STEP_N_SCORE
    if re.search(r"第\s*([一二三四五六七八九十两]+)\s*步", text):
        m = re.search(r"第\s*([一二三四五六七八九十两]+)\s*步", text)
        if m:
            step_n = _parse_cn_number(m.group(1))
            if step_n:
                slots["step_n"] = step_n
        scores["ASK_STEP_N"] = max(scores.get("ASK_STEP_N", 0.0), ASK_STEP_N_SCORE)
    if re.search(r"第一步|第1步", text):
        slots["step_n"] = 1
        scores["ASK_STEP_N"] = max(scores.get("ASK_STEP_N", 0.0), ASK_STEP_N_SCORE)
    if re.search(r"怎么做|步骤|流程|做法", text):
        scores["ASK_STEPS"] = max(scores.get("ASK_STEPS", 0.0), ASK_STEPS_SCORE)
    if re.search(r"原料|材料|食材|需要什么|用什么|用料", text):
        scores["ASK_INGREDIENTS"] = max(scores.get("ASK_INGREDIENTS", 0.0), ASK_INGREDIENTS_SCORE)
    if re.search(r"多少|几克|几勺|用量", text):
        scores["ASK_INGREDIENTS"] = max(
            scores.get("ASK_INGREDIENTS", 0.0),
            ASK_INGREDIENTS_QUANTITY_SCORE,
        )
    if re.search(r"多久|几分钟|多长时间|炖多久|煮多久", text):
        scores["ASK_TIME"] = max(scores.get("ASK_TIME", 0.0), ASK_TIME_SCORE)
    if re.search(r"大火|小火|中火|火候|火力", text):
        scores["ASK_HEAT"] = max(scores.get("ASK_HEAT", 0.0), ASK_HEAT_SCORE)
    if re.search(r"可以不放|能换|替代|没有.+怎么办", text):
        scores["ASK_SUBSTITUTION"] = max(scores.get("ASK_SUBSTITUTION", 0.0), ASK_SUBSTITUTION_SCORE)
    if re.search(r"注意什么|技巧|为什么|更好吃|避免", text):
        scores["ASK_TIPS"] = max(scores.get("ASK_TIPS", 0.0), ASK_TIPS_SCORE)

    if not scores:
        return {"intent": "UNKNOWN", "confidence": 0.0, "slots": {}}
    intent, confidence = max(scores.items(), key=lambda x: x[1])
    return {"intent": intent, "confidence": confidence, "slots": slots}


def route_blocks(intent: str) -> List[str]:
    mapping = {
        "ASK_INGREDIENTS": ["ingredients"],
        "ASK_STEPS": ["operation"],
        "ASK_STEP_N": ["operation"],
        "ASK_TIME": ["operation", "tips"],
        "ASK_HEAT": ["operation", "tips"],
        "ASK_SUBSTITUTION": ["ingredients", "tips"],
        "ASK_TIPS": ["tips", "operation"],
    }
    return mapping.get(intent, [])


def normalize_block_type(block_type: Optional[str]) -> Optional[str]:
    if not block_type:
        return None
    if block_type in _BLOCK_ALIASES:
        return _BLOCK_ALIASES[block_type]
    return block_type.strip().lower()


def build_layer_evidence(evidence_set: Dict, block_types: List[str]) -> Dict:
    if not evidence_set:
        return {"parent_id": None, "chunks": []}
    selected = []
    for chunk in evidence_set.get("chunks", []):
        canonical = normalize_block_type(chunk.get("block_type"))
        if canonical in block_types:
            selected.append(chunk)
    return {"parent_id": evidence_set.get("parent_id"), "chunks": selected}


def _normalize_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_steps(texts: List[str]) -> List[str]:
    steps: List[str] = []
    for text in texts:
        for line in _normalize_lines(text):
            if line.startswith("#"):
                continue
            if re.match(r"^\d+[\.\、\)]", line) or line.startswith(("-", "*")):
                cleaned = line.lstrip("-*0123456789.、) ").strip()
                if cleaned:
                    steps.append(cleaned)
            else:
                steps.append(line)
    if not steps:
        merged = " ".join(texts)
        steps = [s.strip() for s in re.split(r"[。；;]\s*", merged) if s.strip()]
    return steps


def parse_ingredients(texts: List[str]) -> List[str]:
    items: List[str] = []
    for text in texts:
        for line in _normalize_lines(text):
            if line.startswith("#"):
                continue
            if line.startswith(("-", "*")):
                cleaned = line.lstrip("-* ").strip()
                if cleaned:
                    items.append(cleaned)
            else:
                items.append(line)
    return items


def extract_sentences(texts: List[str], keywords: List[str]) -> List[str]:
    hits: List[str] = []
    for text in texts:
        for line in _normalize_lines(text):
            if any(k in line for k in keywords):
                hits.append(line)
    return hits


def generate_answer(
    intent: str,
    slots: Dict[str, int],
    evidence_set: Dict,
    *,
    parsed_steps: Optional[List[str]] = None,
    parsed_ingredients: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    chunks = evidence_set.get("chunks", [])
    op_texts = [c.get("text", "") for c in chunks if normalize_block_type(c.get("block_type")) == "operation"]
    ing_texts = [c.get("text", "") for c in chunks if normalize_block_type(c.get("block_type")) == "ingredients"]
    add_texts = [c.get("text", "") for c in chunks if normalize_block_type(c.get("block_type")) == "tips"]

    if intent == "ASK_STEP_N":
        steps = parsed_steps if parsed_steps is not None else parse_steps(op_texts)
        step_n = slots.get("step_n")
        if step_n and 1 <= step_n <= len(steps):
            return f"第{step_n}步：{steps[step_n - 1]}", []
        return None, ["step_n"]
    if intent == "ASK_STEPS":
        steps = parsed_steps if parsed_steps is not None else parse_steps(op_texts)
        if steps:
            return "完整步骤：\n" + "\n".join([f"- {s}" for s in steps]), []
        return None, ["steps"]
    if intent == "ASK_INGREDIENTS":
        items = parsed_ingredients if parsed_ingredients is not None else parse_ingredients(ing_texts)
        if items:
            return "所需原料：\n" + "\n".join([f"- {i}" for i in items]), []
        return None, ["ingredients"]
    if intent == "ASK_TIME":
        hits = extract_sentences(op_texts + add_texts, ["分钟", "小时", "时间", "多久", "炖", "煮", "焯"])
        if hits:
            return "与时间相关的信息：\n" + "\n".join([f"- {h}" for h in hits[:3]]), []
        return None, ["time_info"]
    if intent == "ASK_HEAT":
        hits = extract_sentences(op_texts + add_texts, ["大火", "小火", "中火", "火候", "火力"])
        if hits:
            return "与火候相关的信息：\n" + "\n".join([f"- {h}" for h in hits[:3]]), []
        return None, ["heat_info"]
    if intent == "ASK_SUBSTITUTION":
        hits = extract_sentences(ing_texts + add_texts, ["可选", "可以不放", "替代", "没有", "可用"])
        if hits:
            return "替代/可选说明：\n" + "\n".join([f"- {h}" for h in hits[:3]]), []
        return None, ["substitution"]
    if intent == "ASK_TIPS":
        tips = parse_ingredients(add_texts)
        if tips:
            return "注意事项：\n" + "\n".join([f"- {t}" for t in tips[:5]]), []
        return None, ["tips"]

    return None, ["unknown_intent"]


def output_intent_for(intent: str, slots: Dict[str, int]) -> str:
    if intent == "ASK_STEP_N":
        return "step_n"
    if intent == "ASK_STEPS":
        return "steps_overview"
    if intent == "ASK_INGREDIENTS":
        return "ingredients_only"
    if intent == "ASK_TIPS":
        return "tips_only"
    if intent in {"ASK_TIME", "ASK_HEAT", "ASK_SUBSTITUTION"}:
        return "qa"
    return "full_recipe"


def build_generation_mapping(output_intent: str, evidence_set: Dict) -> List[Dict]:
    chunks = evidence_set.get("chunks", []) if evidence_set else []
    by_block: Dict[str, List[str]] = {}
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        canonical = normalize_block_type(chunk.get("block_type")) or "tips"
        by_block.setdefault(canonical, []).append(cid)
    sections = []
    if output_intent in {"ingredients_only", "full_recipe"}:
        sections.append({"section": "ingredients", "used_chunk_ids": by_block.get("ingredients", [])})
    if output_intent in {"steps_overview", "step_n", "full_recipe", "qa"}:
        sections.append({"section": "steps", "used_chunk_ids": by_block.get("operation", [])})
    if output_intent in {"tips_only", "full_recipe"}:
        sections.append({"section": "tips", "used_chunk_ids": by_block.get("tips", [])})
    return [s for s in sections if s["used_chunk_ids"]]
