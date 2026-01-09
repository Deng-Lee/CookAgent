from __future__ import annotations

import os
from pathlib import Path

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
