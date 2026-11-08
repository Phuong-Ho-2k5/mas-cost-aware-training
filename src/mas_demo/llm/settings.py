"""Provider configuration; credentials never become part of a TaskSpec snapshot."""
import math
import os
from urllib.parse import urlsplit

from mas_demo.llm.gateway import OpenAICompatibleGateway


def create_gateway(spec):
    if spec.analysis_mode == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter backend")
        return OpenAICompatibleGateway(key, spec.model_id)
    url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must be an HTTP(S) URL without credentials, query or fragment")
    try:
        timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
        deadline = float(os.environ.get("LLM_DEADLINE_SECONDS", "130"))
        max_tokens = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "1024"))
        if not all(math.isfinite(x) and x > 0 for x in (timeout, deadline)) or not 1 <= max_tokens <= 2048:
            raise ValueError()
    except ValueError:
        raise ValueError("LLM timeout/deadline must be positive finite seconds; output tokens must be 1..2048") from None
    return OpenAICompatibleGateway(os.environ.get("LLM_API_KEY", "").strip(), spec.model_id,
        base_url=url, timeout=timeout, deadline=deadline, mode="local", attempts=1, max_tokens=max_tokens)
