"""Exact local token-budget preflight; never shorten observed metric evidence."""

from copy import deepcopy
import json
import httpx
from urllib.parse import urlsplit, urlunsplit


async def count_local_prompt(client, base_url, body, headers, backend="vllm"):
    """Use mandatory vLLM tokenization, including the response schema."""
    from mas_demo.llm.gateway import GatewayError

    if backend != "vllm":
        raise GatewayError("Only vLLM token preflight is supported")
    url = urlsplit(base_url.rstrip("/"))
    if url.path != "/v1" or url.query or url.fragment:
        raise GatewayError("vLLM token preflight requires a root /v1 base URL")
    endpoint = urlunsplit((url.scheme, url.netloc, "/tokenize", "", ""))
    request = {name: body[name] for name in
               ("model", "messages", "response_format", "chat_template_kwargs")}
    request.update(add_generation_prompt=True, add_special_tokens=False)
    try:
        response = await client.post(endpoint, json=request, headers=headers, timeout=5)
    except httpx.TransportError:
        raise GatewayError("vLLM token-budget preflight transport failure") from None
    if response.status_code != 200:
        raise GatewayError("Local token-budget preflight failed")
    try:
        counts = response.json()
        if isinstance(counts, dict):
            counts = {"prompt_tokens": counts.get("count"),
                      "context_tokens": counts.get("max_model_len")}
        if (not isinstance(counts, dict)
            or type(counts.get("prompt_tokens")) is not int
            or type(counts.get("context_tokens")) is not int
            or counts["prompt_tokens"] < 0 or counts["context_tokens"] <= 0):
            raise ValueError("Invalid token counts")
        return counts
    except ValueError:
        raise GatewayError("Invalid local token-budget preflight response") from None


async def fit_local_context(client, base_url, body, metadata, headers, backend="vllm"):
    payload = json.loads(body["messages"][1]["content"])
    original = deepcopy(payload)
    compact = {"history_removed": 0, "previous_response_referenced": False}
    while True:
        counts = await count_local_prompt(client, base_url, body, headers, backend)
        from mas_demo.llm.gateway import GatewayError

        expected_context = metadata.get("server", {}).get("context_tokens")
        if expected_context is not None and counts["context_tokens"] != expected_context:
            raise GatewayError("vLLM preflight context differs from verified runtime")
        compact.setdefault("initial_prompt_tokens", counts["prompt_tokens"])
        compact.update(
            prompt_tokens=counts["prompt_tokens"],
            context_tokens=counts["context_tokens"],
        )
        metadata["context_compaction"] = compact
        if counts["prompt_tokens"] + body["max_tokens"] <= counts["context_tokens"]:
            return
        if payload.get("history"):
            payload["history"] = payload["history"][1:]
            compact["history_removed"] += 1
        elif (
            payload.get("previous_response")
            and not compact["previous_response_referenced"]
        ):
            previous = payload["previous_response"]
            schema_name = body["response_format"]["json_schema"]["name"]
            analysis = schema_name in ("training_diagnosis", "training_detection", "training_hypotheses")
            payload["previous_response"] = {
                "stored_in": (
                    "previous analysis call in metadata.calls"
                    if analysis
                    else "previous planning event"
                ),
                "proposal_ids": (
                    [p.get("proposal_id") for p in previous.get("proposals", [])]
                    if isinstance(previous, dict)
                    else []
                ),
                "instruction": (
                    "Trả đầy đủ ba hypothesis đã sửa, dùng detection đã khóa, evidence và validation_errors. Không đổi status của detection."
                    if schema_name == "training_hypotheses" else
                    "Trả kết quả detection đã sửa đầy đủ, dựa trên evidence và validation_errors; tự chọn status phù hợp logs."
                    if schema_name == "training_detection" else
                    "Return a complete corrected diagnosis using the evidence and validation errors; choose status yourself."
                    if analysis
                    else "Generate a corrected complete plan using the validation errors."
                ),
            }
            compact["previous_response_referenced"] = True
        else:
            raise GatewayError(
                "Local context exceeds token budget; evidence was kept intact"
            )
        assert payload.get("evidence") == original.get("evidence")
        body["messages"][1]["content"] = json.dumps(payload, allow_nan=False, separators=(",", ":"))
