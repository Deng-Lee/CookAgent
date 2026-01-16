from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


LLMCall = Callable[[Dict[str, Any]], Any]


def build_extraction_payload(
    evidence_set: Dict,
    intent: str,
    *,
    evidence_scope: Optional[str] = None,
) -> Dict:
    return {
        "intent": intent,
        "evidence_scope": evidence_scope,
        "evidence_set": evidence_set,
    }


def build_polish_payload(
    answer_draft: str,
    *,
    intent: Optional[str] = None,
) -> Dict:
    return {
        "intent": intent,
        "answer_draft": answer_draft,
    }


def call_llm_extract(llm_call: Optional[LLMCall], payload: Dict) -> Optional[Dict]:
    if not llm_call:
        return None
    response = llm_call(payload)
    if response is None:
        return None
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    return None


def call_llm_polish(llm_call: Optional[LLMCall], payload: Dict) -> Optional[str]:
    if not llm_call:
        return None
    response = llm_call(payload)
    if response is None:
        return None
    if isinstance(response, str):
        return response.strip() or None
    if isinstance(response, dict):
        text = response.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None
    return None


def _collect_evidence_chunks(evidence_set: Dict) -> Dict[str, str]:
    chunks = {}
    for chunk in evidence_set.get("chunks", []):
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            chunks[chunk_id] = chunk.get("text", "") or ""
    return chunks


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _validate_schema(extraction: Dict, *, expected_intent: Optional[str]) -> List[str]:
    errors = []
    if not isinstance(extraction, dict):
        return ["extraction_not_dict"]
    intent = extraction.get("intent")
    fields = extraction.get("fields")
    missing = extraction.get("missing")
    if not isinstance(intent, str) or not intent:
        errors.append("intent_missing_or_invalid")
    if expected_intent and intent != expected_intent:
        errors.append("intent_mismatch")
    if not isinstance(fields, dict):
        errors.append("fields_missing_or_invalid")
    if not isinstance(missing, list):
        errors.append("missing_missing_or_invalid")
    return errors


def _iter_field_items(fields: Dict) -> Iterable[Tuple[str, Dict]]:
    for field_name, items in fields.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield field_name, item


def validate_extraction(
    extraction: Optional[Dict],
    evidence_set: Dict,
    *,
    expected_intent: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    if not extraction:
        return False, ["empty_extraction"]
    errors = _validate_schema(extraction, expected_intent=expected_intent)
    if errors:
        return False, errors

    chunks = _collect_evidence_chunks(evidence_set)
    fields = extraction.get("fields") or {}

    for _, item in _iter_field_items(fields):
        text = item.get("text")
        citations = item.get("citations")
        if not isinstance(text, str) or not text.strip():
            errors.append("item_text_missing")
            continue
        if not isinstance(citations, list) or not citations:
            errors.append("citations_missing")
            continue
        quotes_for_item: List[str] = []
        for citation in citations:
            if not isinstance(citation, dict):
                errors.append("citation_not_dict")
                continue
            chunk_id = citation.get("chunk_id")
            quote = citation.get("quote")
            if not chunk_id or chunk_id not in chunks:
                errors.append("citation_chunk_id_invalid")
                continue
            if not isinstance(quote, str) or not quote:
                errors.append("citation_quote_missing")
                continue
            if quote not in chunks[chunk_id]:
                errors.append("citation_quote_not_found")
                continue
            quotes_for_item.append(quote)

        if quotes_for_item:
            quoted_text = " ".join(quotes_for_item)
            for number in _extract_numbers(text):
                if number not in quoted_text:
                    errors.append("number_out_of_bounds")
                    break

    return not errors, errors


def validate_polish(original: str, polished: Optional[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(original, str) or not original.strip():
        return False, ["original_empty"]
    if not isinstance(polished, str) or not polished.strip():
        return False, ["polished_empty"]
    original_numbers = set(_extract_numbers(original))
    polished_numbers = set(_extract_numbers(polished))
    if not polished_numbers.issubset(original_numbers):
        errors.append("number_out_of_bounds")
    return not errors, errors
