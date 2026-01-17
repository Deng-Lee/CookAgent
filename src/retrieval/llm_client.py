from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _deepseek_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.2,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            last_error = None
            break
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1)
                continue
            raise
    if last_error is not None:
        raise last_error
    content = payload["choices"][0]["message"]["content"]
    return content.strip()


def llm_extract(payload: Dict[str, Any], *, model: str, api_key: str) -> Dict[str, Any]:
    system = (
        "You extract structured data from evidence_set. "
        "Return JSON only, following the required schema. "
        "Citations must include chunk_id and exact quote from chunk.text. "
        "Do not add facts beyond evidence_set."
    )
    user = json.dumps(payload, ensure_ascii=False)
    content = _deepseek_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        api_key=api_key,
    )
    return json.loads(content)


def llm_polish(payload: Dict[str, Any], *, model: str, api_key: str) -> str:
    system = (
        "You polish the given answer_draft for fluency. "
        "Do not add facts, numbers, times, or quantities. "
        "Keep meaning and order unchanged. "
        "Return the polished text only."
    )
    user = json.dumps(payload, ensure_ascii=False)
    return _deepseek_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
