import json
import os
import sys
import urllib.request
from pathlib import Path


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = load_env(Path(".env"))
    base_url = env.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
    model = env.get("LLM_MODEL") or os.getenv("LLM_MODEL")
    api_key = env.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not model or not api_key:
        print("Missing LLM_MODEL or LLM_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "今天上海天气如何？"},
        ],
        "temperature": 0.0,
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
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    content = payload["choices"][0]["message"]["content"]
    print(content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
