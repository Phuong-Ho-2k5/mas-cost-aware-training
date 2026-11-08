import asyncio
import json
import time
import httpx
from pydantic import ValidationError
from typing import Literal
from mas_demo.contracts import DiagnosisReport, Hypothesis, Record


class GatewayError(RuntimeError):
    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}


class ResponsePayload(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    hypotheses: list[Hypothesis]
    reason: str


class OpenRouterGateway:
    def __init__(self, key, model="xingchen-agi/xing4.0-29b-a4b:free",
                 base_url="https://openrouter.ai/api/v1", timeout=30, retry_delay=1, deadline=65):
        self.key, self.model = key, model
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retry_delay, self.deadline = timeout, retry_delay, deadline

    def complete(self, evidence):
        payload, metadata = self.structured(
            "analysis", evidence, ResponsePayload,
            "Diagnose ML training from provided evidence only. Return hypotheses with "
            "evidence_epochs and interventions from allowed_interventions. Each intervention has "
            "parameter, operation and value. Learning rate uses multiply; momentum, weight_decay "
            "and grad_clip_norm use set. Use observed settings and metrics only, and consider "
            "uncertainty: a proposed cause must be verified against control. No commands. "
            "ABSTAIN if evidence insufficient. Confidence is an estimate, not calibrated.")
        return DiagnosisReport(mode="openrouter", **payload.model_dump()), metadata

    def structured(self, role, payload, response_type, instruction):
        started = time.monotonic()
        metadata = {"agent": role, "mode": "openrouter", "model": self.model,
                    "attempts": 0, "usage": {}, "latency_seconds": 0}
        async def bounded():
            try:
                return await asyncio.wait_for(
                    self._complete(role, payload, response_type, instruction, metadata), timeout=self.deadline)
            except asyncio.TimeoutError:
                raise GatewayError("Gateway absolute deadline exceeded") from None
        try:
            if not self.key:
                raise GatewayError("OPENROUTER_API_KEY is not set")
            result = asyncio.run(bounded())
        except GatewayError as error:
            metadata["latency_seconds"] = time.monotonic() - started
            error.metadata = dict(metadata)
            raise
        metadata["latency_seconds"] = time.monotonic() - started
        return result, metadata

    async def _complete(self, role, payload, response_type, instruction, metadata):
        schema = response_type.model_json_schema()
        schema_name = {"analysis": "training_diagnosis", "planner": "training_plan",
                       "evaluation": "training_evaluation"}[role]
        body = {"model": self.model, "temperature": 0, "max_tokens": 2048,
                "messages": [
                    {"role": "system", "content": instruction + " Return only JSON conforming to this schema, "
                     "without markdown or extra fields: " + json.dumps(schema)},
                    {"role": "user", "content": json.dumps(payload, allow_nan=False)}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema_name, "strict": False, "schema": schema}}}
        started = time.monotonic()
        async with httpx.AsyncClient(trust_env=False) as client:
            for attempt in range(2):
                metadata["attempts"] = attempt + 1
                remaining = self.deadline - (time.monotonic() - started)
                if remaining <= 0:
                    raise GatewayError("Gateway deadline exceeded")
                try:
                    response = await client.post(self.base_url + "/chat/completions", json=body,
                                           headers={"Authorization": "Bearer " + self.key},
                                           timeout=min(self.timeout, remaining))
                except httpx.TransportError:
                    if attempt == 1:
                        raise GatewayError("Upstream transport failure") from None
                    delay = self.retry_delay
                else:
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if not isinstance(data, dict):
                                raise ValueError("Response must be an object")
                            metadata.update(model=data.get("model", self.model), request_id=data.get("id"),
                                            usage=data.get("usage", {}))
                            result = response_type.model_validate_json(data["choices"][0]["message"]["content"])
                        except (ValueError, KeyError, IndexError, TypeError, ValidationError):
                            raise GatewayError("Invalid upstream response schema") from None
                        return result
                    # Keep the provider's explanation for future quota/rate-limit diagnosis.
                    # Never persist a credential even if an upstream error echoes it.
                    detail = {}
                    try:
                        envelope = response.json()
                        if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
                            detail = envelope["error"]
                    except ValueError:
                        pass
                    def sanitized(value):
                        return str(value).replace(self.key, "[REDACTED]")[:500]
                    metadata.setdefault("upstream_errors", []).append({
                        "attempt": attempt + 1, "status": response.status_code,
                        "code": sanitized(detail.get("code", "")),
                        "message": sanitized(detail.get("message", "")),
                        "retry_after": sanitized(response.headers.get("Retry-After", ""))})
                    if response.status_code != 429 and response.status_code < 500:
                        raise GatewayError(f"Upstream HTTP {response.status_code}")
                    if attempt == 1:
                        raise GatewayError(f"Upstream HTTP {response.status_code}; retry limit reached")
                    try:
                        delay = max(self.retry_delay, float(response.headers.get("Retry-After", 0)))
                    except ValueError:
                        delay = self.retry_delay
                if delay >= self.deadline - (time.monotonic() - started):
                    raise GatewayError("Retry exceeds gateway deadline")
                await asyncio.sleep(delay)
        raise GatewayError("No usable response")


def failure_metadata(error, agent, spec, metadata=None):
    return {"agent": agent, "mode": "openrouter", "model": spec.model_id,
            **(metadata or {}), **error.metadata, "error": str(error)}
