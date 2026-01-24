import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "retrieval"))

from retrieval.parent_retriever import run_session_once, retrieve, detect_categories


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


def _score_from_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    scores = [
        c.get("overall_score")
        for c in candidates
        if isinstance(c, dict) and isinstance(c.get("overall_score"), (int, float))
    ]
    if not scores:
        return {"top1_overall_score": None, "score2_over_score1": None}
    top1 = scores[0]
    ratio = (scores[1] / scores[0]) if len(scores) > 1 and scores[0] else None
    return {"top1_overall_score": top1, "score2_over_score1": ratio}


def _score_from_retrieval(query: str, db_path: Path, collection: str) -> Dict[str, Optional[float]]:
    categories = detect_categories(query)
    dir_category = categories[0] if len(categories) == 1 else None
    parents = retrieve(
        query=query,
        db_path=db_path,
        collection_name=collection,
        dir_category=dir_category,
    )
    if not parents:
        return {"top1_overall_score": None, "score2_over_score1": None}
    parents_sorted = sorted(parents, key=lambda p: p.overall_score, reverse=True)
    top1 = parents_sorted[0].overall_score
    ratio = (
        (parents_sorted[1].overall_score / parents_sorted[0].overall_score)
        if len(parents_sorted) > 1 and parents_sorted[0].overall_score
        else None
    )
    return {"top1_overall_score": top1, "score2_over_score1": ratio}


def _next_index(output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{prefix}_*.jsonl"))
    return len(existing) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch eval runner for eval_set_v1.jsonl")
    parser.add_argument("--eval-path", type=Path, default=Path("src/test/eval_set_v1.jsonl"))
    parser.add_argument("--db-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", type=str, default="cook_chunks")
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    args = parser.parse_args()

    eval_log_path = project_root / "logs" / "retrieval_eval_runner.log"
    eval_lock_log_path = project_root / "logs" / "parent_locking_eval_runner.log"
    eval_evidence_log_path = project_root / "logs" / "evidence_driven_eval_runner.log"
    eval_generation_log_path = project_root / "logs" / "generation_eval_runner.log"
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path.write_text("", encoding="utf-8")
    eval_lock_log_path.write_text("", encoding="utf-8")
    eval_evidence_log_path.write_text("", encoding="utf-8")
    eval_generation_log_path.write_text("", encoding="utf-8")

    eval_path = args.eval_path
    if not eval_path.exists():
        fallback_path = Path("result/eval_set_v1.jsonl")
        if fallback_path.exists():
            eval_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Eval set not found at {args.eval_path} or {fallback_path}. "
                "Provide --eval-path or create the file locally."
            )
    items = _load_jsonl(eval_path)
    passed = 0
    failed = 0
    output_lines: List[str] = []
    failure_lines: List[str] = []
    run_id = uuid.uuid4().hex[:8]
    for item in items:
        eval_id = item["id"]
        query = item["query"]
        expected = item.get("expected", {})
        result = run_session_once(
            query=query,
            trace_id=eval_id,
            session_id=f"eval_{run_id}_{eval_id}",
            db_path=args.db_path,
            collection_name=args.collection,
            log_path=eval_log_path,
            lock_log_path=eval_lock_log_path,
            evidence_log_path=eval_evidence_log_path,
            generation_log_path=eval_generation_log_path,
        )

        ok_state = _state_matches(expected.get("state"), result.get("state"))
        ok_category = _category_matches(expected.get("category"), result)
        ok_parent = _parent_matches(expected.get("parent_ids") or [], result)
        ok = ok_state and ok_category and ok_parent

        def _strip_result_fields(result_obj: Dict[str, Any]) -> Dict[str, Any]:
            if "answer" in result_obj:
                result_obj = {**result_obj, "answer": None}
            result_obj.pop("message", None)
            result_obj.pop("clarify_question", None)
            return result_obj

        scoring = {}
        state = result.get("state")
        if state not in {"LOW_EVIDENCE", "EVIDENCE_INSUFFICIENT"}:
            candidates = result.get("candidates") or []
            if candidates:
                scoring = _score_from_candidates(candidates)
            else:
                scoring = _score_from_retrieval(query, args.db_path, args.collection)

        output_lines.append(
            json.dumps(
                {
                    "id": eval_id,
                    "query": query,
                    "expected": expected,
                    "result": {**_strip_result_fields(result), **scoring},
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
                        "result": {**_strip_result_fields(result), **scoring},
                    },
                    ensure_ascii=False,
                )
            )

    total = passed + failed
    summary = {"total": total, "passed": passed, "failed": failed}
    output_lines.append(json.dumps({"summary": summary}, ensure_ascii=False))
    report_index = _next_index(args.output_dir, "eval_report")
    failure_index = _next_index(args.output_dir, "eval_failures")
    report_path = args.output_dir / f"eval_report_{report_index}.jsonl"
    failure_path = args.output_dir / f"eval_failures_{failure_index}.jsonl"
    report_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    failure_path.write_text("\n".join(failure_lines) + ("\n" if failure_lines else ""), encoding="utf-8")
    print(f"[done] eval report -> {report_path}")
    print(f"[done] eval failures -> {failure_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
