"""Exact local token-budget preflight; never shorten observed metric evidence."""

from copy import deepcopy
import json
import httpx


async def fit_local_context(client, base_url, body, metadata, headers):
    payload = json.loads(body["messages"][1]["content"])
    original = deepcopy(payload)
    compact = {"history_removed": 0, "previous_response_referenced": False}
    while True:
        try:
            response = await client.post(
                base_url + "/token-count", json=body, headers=headers, timeout=5
            )
        except httpx.TransportError:
            # External OpenAI-compatible servers may not provide this extension.
            return
        if response.status_code in (404, 405, 501):
            return
        from mas_demo.llm.gateway import GatewayError

        if response.status_code != 200:
            raise GatewayError("Local token-budget preflight failed")
        try:
            counts = response.json()
            if (
                not isinstance(counts, dict)
                or type(counts.get("prompt_tokens")) is not int
                or type(counts.get("context_tokens")) is not int
                or counts["prompt_tokens"] < 0
                or counts["context_tokens"] <= 0
            ):
                raise ValueError("Invalid token counts")
        except ValueError as error:
            raise GatewayError(
                "Invalid local token-budget preflight response"
            ) from error
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
            analysis = (
                body["response_format"]["json_schema"]["name"] == "training_diagnosis"
            )
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
        body["messages"][1]["content"] = json.dumps(payload, allow_nan=False)
