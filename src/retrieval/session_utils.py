from __future__ import annotations

import json
import re
import time
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional

from retrieval_types import DEFAULT_SESSION_DIR


def _safe_session_name(session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", session_id.strip() or "cli_default")


def session_path(session_id: str) -> Path:
    safe = _safe_session_name(session_id)
    digest = sha1(session_id.encode("utf-8")).hexdigest()[:8]
    return DEFAULT_SESSION_DIR / f"{safe}-{digest}.json"


def legacy_session_path(session_id: str) -> Path:
    safe = _safe_session_name(session_id)
    return DEFAULT_SESSION_DIR / f"{safe}.json"


def load_session(session_id: str) -> Dict:
    path = session_path(session_id)
    legacy_path = legacy_session_path(session_id)
    if not path.exists():
        if legacy_path.exists():
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        return {
            "session_id": session_id,
            "turn": 0,
            "last_trace_id": None,
            "parent_lock": {
                "status": "none",
                "parent_id": None,
                "pending_reason": None,
                "lock_score": None,
                "lock_reason": None,
            },
            "pending_candidates": [],
            "parent_cache": None,
            "ttl_seconds": 900,
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(session: Dict) -> None:
    path = session_path(session["session_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def purge_expired_pending(session: Dict) -> None:
    ttl_seconds = session.get("ttl_seconds", 900)
    updated_at = session.get("updated_at")
    if not updated_at:
        return
    if session.get("parent_lock", {}).get("status") != "pending":
        return
    if (time.time() - updated_at) > ttl_seconds:
        session["parent_lock"] = {
            "status": "none",
            "parent_id": None,
            "pending_reason": None,
            "lock_score": None,
            "lock_reason": None,
        }
        session["pending_candidates"] = []


def parse_option_id(text: str, candidates: List[Dict]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    option = match.group(1)
    valid = {c.get("option_id") for c in candidates}
    return option if option in valid else None
