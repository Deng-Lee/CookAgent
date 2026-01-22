import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "retrieval"))

from retrieval.parent_retriever import run_session_once


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(json.loads(stripped))
    return items


def _normalize_state(state: str) -> str:
    return "LOW_EVIDENCE" if state == "EVIDENCE_INSUFFICIENT" else state


def _state_matches(expected: Any, actual: str) -> bool:
    actual_norm = _normalize_state(actual)
    if isinstance(expected, list):
        return actual_norm in {_normalize_state(s) for s in expected}
    return actual_norm == _normalize_state(expected)


def _category_matches(expected: str, result: Dict[str, Any]) -> bool:
    if not expected:
        return True
    if result.get("parent_id") and f"/{expected}/" in result["parent_id"]:
        return True
    pending = result.get("pending_categories") or []
    if any(c.get("category") == expected for c in pending):
        return True
    return False


def _parent_matches(expected_ids: List[str], result: Dict[str, Any]) -> bool:
    if not expected_ids:
        return True
    parent_id = result.get("parent_id")
    if parent_id and parent_id in expected_ids:
        return True
    candidates = result.get("candidates") or []
    for cand in candidates:
        if cand.get("parent_id") in expected_ids:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch eval runner for eval_set_v1.jsonl")
    parser.add_argument("--eval-path", type=Path, default=Path("src/test/eval_set_v1.jsonl"))
    parser.add_argument("--db-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", type=str, default="cook_chunks")
    parser.add_argument("--output-path", type=Path, default=Path("logs/eval_report.jsonl"))
    parser.add_argument("--fail-path", type=Path, default=Path("logs/eval_failures.jsonl"))
    args = parser.parse_args()

    items = _load_jsonl(args.eval_path)
    passed = 0
    failed = 0
    output_lines: List[str] = []
    failure_lines: List[str] = []
    for item in items:
        eval_id = item["id"]
        query = item["query"]
        expected = item.get("expected", {})
        result = run_session_once(
            query=query,
            trace_id=eval_id,
            session_id=f"eval_{eval_id}",
            db_path=args.db_path,
            collection_name=args.collection,
        )

        ok_state = _state_matches(expected.get("state"), result.get("state"))
        ok_category = _category_matches(expected.get("category"), result)
        ok_parent = _parent_matches(expected.get("parent_ids") or [], result)
        ok = ok_state and ok_category and ok_parent

        def _strip_answer(result_obj: Dict[str, Any]) -> Dict[str, Any]:
            if "answer" in result_obj:
                result_obj = {**result_obj, "answer": None}
            return result_obj

        output_lines.append(
            json.dumps(
                {
                    "id": eval_id,
                    "query": query,
                    "expected": expected,
                    "result": _strip_answer(result),
                    "pass": ok,
                },
                ensure_ascii=False,
            )
        )

        if ok:
            passed += 1
        else:
            failed += 1
            failure_lines.append(
                json.dumps(
                    {
                        "id": eval_id,
                        "query": query,
                        "expected": expected,
                        "result": _strip_answer(result),
                    },
                    ensure_ascii=False,
                )
            )

    total = passed + failed
    summary = {"total": total, "passed": passed, "failed": failed}
    output_lines.append(json.dumps({"summary": summary}, ensure_ascii=False))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    args.fail_path.write_text("\n".join(failure_lines) + ("\n" if failure_lines else ""), encoding="utf-8")
    print(f"[done] eval report -> {args.output_path}")
    print(f"[done] eval failures -> {args.fail_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
