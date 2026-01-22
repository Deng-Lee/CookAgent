import argparse
import sys
import uuid
from pathlib import Path

from retrieval.parent_retriever import run_session_once
from retrieval.generation_utils import classify_query
from retrieval.parent_retriever import detect_categories


def _print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    if detail:
        print(f"[{status}] {name}: {detail}")
    else:
        print(f"[{status}] {name}")


def test_classify_step_cn() -> bool:
    result = classify_query("第三步是什么")
    ok = result["intent"] == "ASK_STEP_N" and result["slots"].get("step_n") == 3
    _print_result("classify_step_cn", ok, str(result))
    return ok


def test_detect_category_conflict() -> bool:
    hits = detect_categories("素菜汤推荐")
    ok = "素菜" in hits and "汤" in hits and len(hits) >= 2
    _print_result("detect_category_conflict", ok, str(hits))
    return ok


def test_detect_single_category() -> bool:
    hits = detect_categories("推荐几个素菜")
    ok = hits == ["素菜"]
    _print_result("detect_single_category", ok, str(hits))
    return ok


def test_category_pending(db_path: Path, collection: str) -> bool:
    session_id = f"regress_{uuid.uuid4().hex[:6]}"
    result = run_session_once(
        query="素菜汤推荐",
        trace_id="regress",
        session_id=session_id,
        db_path=db_path,
        collection_name=collection,
    )
    ok = result.get("state") == "GOOD_BUT_AMBIGUOUS" and result.get("pending_categories")
    _print_result("category_pending", ok, str(result))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal regression runner.")
    parser.add_argument("--db-path", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", type=str, default="cook_chunks")
    args = parser.parse_args()

    results = [
        test_classify_step_cn(),
        test_detect_category_conflict(),
        test_detect_single_category(),
        test_category_pending(args.db_path, args.collection),
    ]
    failed = [r for r in results if not r]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
