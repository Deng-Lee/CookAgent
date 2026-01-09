from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


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

def classify_query(query: str) -> Dict:
    text = query.strip()
    scores = {}
    slots: Dict[str, int] = {}

    if re.search(r"第(\d+)步", text):
        m = re.search(r"第(\d+)步", text)
        if m:
            slots["step_n"] = int(m.group(1))
        scores["ASK_STEP_N"] = 0.95
    if re.search(r"第一步|第1步", text):
        slots["step_n"] = 1
        scores["ASK_STEP_N"] = max(scores.get("ASK_STEP_N", 0.0), 0.95)
    if re.search(r"怎么做|步骤|流程|做法", text):
        scores["ASK_STEPS"] = max(scores.get("ASK_STEPS", 0.0), 0.7)
    if re.search(r"原料|材料|食材|需要什么|用什么|用料", text):
        scores["ASK_INGREDIENTS"] = max(scores.get("ASK_INGREDIENTS", 0.0), 0.8)
    if re.search(r"多少|几克|几勺|用量", text):
        scores["ASK_INGREDIENTS"] = max(scores.get("ASK_INGREDIENTS", 0.0), 0.6)
    if re.search(r"多久|几分钟|多长时间|炖多久|煮多久", text):
        scores["ASK_TIME"] = max(scores.get("ASK_TIME", 0.0), 0.8)
    if re.search(r"大火|小火|中火|火候|火力", text):
        scores["ASK_HEAT"] = max(scores.get("ASK_HEAT", 0.0), 0.8)
    if re.search(r"可以不放|能换|替代|没有.+怎么办", text):
        scores["ASK_SUBSTITUTION"] = max(scores.get("ASK_SUBSTITUTION", 0.0), 0.7)
    if re.search(r"注意什么|技巧|为什么|更好吃|避免", text):
        scores["ASK_TIPS"] = max(scores.get("ASK_TIPS", 0.0), 0.7)

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
            preview = steps[:3]
            return "步骤要点：\n" + "\n".join([f"- {s}" for s in preview]), []
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
