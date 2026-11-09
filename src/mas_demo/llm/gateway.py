import asyncio
import json
import time
import httpx
from pydantic import ValidationError
from jsonschema import (
    Draft202012Validator,
    ValidationError as JSONSchemaValidationError,
)
from typing import Literal
from mas_demo.contracts import DiagnosisReport, Hypothesis, Record
from mas_demo.prompts import load_prompt


class GatewayError(RuntimeError):
    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}


class ResponsePayload(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    hypotheses: list[Hypothesis]
    reason: str


def response_schema(role, response_type, mode="openrouter", max_trials=None):
    """The exact schema sent to a provider, also used by benchmark scoring."""
    schema = response_type.model_json_schema()
    if mode == "local":
        schema["properties"]["reason"]["maxLength"] = 160
        if role == "analysis":
            schema["properties"]["hypotheses"]["maxItems"] = 1
            hypothesis = schema["$defs"]["AnalysisHypothesis"]["properties"]
            hypothesis["cause"]["maxLength"] = 160
            hypothesis["hypothesis_id"]["maxLength"] = 48
            hypothesis["evidence_epochs"]["maxItems"] = 8

    if role == "planner":
        schema["properties"]["proposals"]["maxItems"] = (
            3 if max_trials is None else max_trials
        )

    def compact(node):
        if isinstance(node, dict):
            return {
                k: compact(v) for k, v in node.items() if k not in ("title", "default")
            }
        if isinstance(node, list):
            return [compact(item) for item in node]
        return node

    return compact(schema)


class OpenAICompatibleGateway:
    def __init__(
        self,
        key,
        model="xingchen-agi/xing4.0-29b-a4b:free",
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
        retry_delay=1,
        deadline=65,
        mode="openrouter",
        attempts=2,
        max_tokens=2048,
    ):
        self.key, self.model = key, model
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retry_delay, self.deadline = timeout, retry_delay, deadline
        self.mode, self.attempts, self.max_tokens = mode, attempts, max_tokens

    def complete(self, evidence):
        payload, metadata = self.structured(
            "analysis", evidence, ResponsePayload, load_prompt("analysis")
        )
        return DiagnosisReport(mode=self.mode, **payload.model_dump()), metadata

    def structured(self, role, payload, response_type, instruction):
        started = time.monotonic()
        metadata = {
            "agent": role,
            "mode": self.mode,
            "model": self.model,
            "attempts": 0,
            "usage": {},
            "latency_seconds": 0,
        }

        async def bounded():
            try:
                return await asyncio.wait_for(
                    self._complete(role, payload, response_type, instruction, metadata),
                    timeout=self.deadline,
                )
            except asyncio.TimeoutError:
                raise GatewayError("Gateway absolute deadline exceeded") from None

        try:
            if not self.key and self.mode == "openrouter":
                raise GatewayError("OPENROUTER_API_KEY is not set")
            result = asyncio.run(bounded())
        except GatewayError as error:
            metadata["latency_seconds"] = time.monotonic() - started
            error.metadata = dict(metadata)
            raise
        metadata["latency_seconds"] = time.monotonic() - started
        return result, metadata

    async def _complete(self, role, payload, response_type, instruction, metadata):
        schema = response_schema(
            role, response_type, self.mode, payload.get("max_trials")
        )
        schema_name = {
            "analysis": "training_diagnosis",
            "planner": "training_plan",
            "evaluation": "training_evaluation",
        }[role]
        if self.mode == "local":
            instruction += load_prompt("local")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": instruction
                    + load_prompt("response_format")
                    + json.dumps(schema),
                },
                {"role": "user", "content": json.dumps(payload, allow_nan=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": False, "schema": schema},
            },
        }
        started = time.monotonic()
        async with httpx.AsyncClient(trust_env=False) as client:
            if self.mode == "local":
                from mas_demo.llm.context import fit_local_context

                await fit_local_context(
                    client,
                    self.base_url,
                    body,
                    metadata,
                    {"Authorization": "Bearer " + self.key} if self.key else {},
                )
            for attempt in range(self.attempts):
                metadata["attempts"] = attempt + 1
                remaining = self.deadline - (time.monotonic() - started)
                if remaining <= 0:
                    raise GatewayError("Gateway deadline exceeded")
                try:
                    response = await client.post(
                        self.base_url + "/chat/completions",
                        json=body,
                        headers=(
                            {"Authorization": "Bearer " + self.key} if self.key else {}
                        ),
                        timeout=min(self.timeout, remaining),
                    )
                except httpx.TransportError:
                    if attempt + 1 == self.attempts:
                        raise GatewayError("Upstream transport failure") from None
                    delay = self.retry_delay
                else:
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if not isinstance(data, dict):
                                raise ValueError("Response must be an object")
                            metadata.update(
                                model=data.get("model", self.model),
                                request_id=data.get("id"),
                                usage=data.get("usage", {}),
                            )
                            if isinstance(data.get("local_metadata"), dict):
                                metadata["server"] = {
                                    name: data["local_metadata"][name]
                                    for name in (
                                        "revision",
                                        "dtype",
                                        "device",
                                        "engine",
                                        "context_tokens",
                                        "load_seconds",
                                        "torch_version",
                                        "generation_seconds",
                                        "rss_mb",
                                        "peak_vram_mb",
                                    )
                                    if name in data["local_metadata"]
                                }
                            choice = data["choices"][0]
                            content = choice["message"]["content"]
                            metadata["raw_output"] = (
                                content.replace(self.key, "[REDACTED]")
                                if self.key
                                else content
                            )
                            metadata["finish_reason"] = choice.get("finish_reason")
                            if choice.get("finish_reason") not in (None, "stop"):
                                raise ValueError("Incomplete or refused model output")
                            Draft202012Validator(schema).validate(json.loads(content))
                            result = response_type.model_validate_json(content)
                        except (
                            ValueError,
                            KeyError,
                            IndexError,
                            TypeError,
                            ValidationError,
                            JSONSchemaValidationError,
                        ):
                            raise GatewayError(
                                "Invalid upstream response schema"
                            ) from None
                        return result
                    # Keep the provider's explanation for future quota/rate-limit diagnosis.
                    # Never persist a credential even if an upstream error echoes it.
                    detail = {}
                    try:
                        envelope = response.json()
                        if isinstance(envelope, dict) and isinstance(
                            envelope.get("error"), dict
                        ):
                            detail = envelope["error"]
                    except ValueError:
                        pass

                    def sanitized(value):
                        value = str(value)
                        return (
                            value.replace(self.key, "[REDACTED]") if self.key else value
                        )[:500]

                    if isinstance(detail.get("raw_output"), str):
                        raw = detail["raw_output"]
                        metadata["raw_output"] = (
                            raw.replace(self.key, "[REDACTED]") if self.key else raw
                        )[:20000]
                        metadata["finish_reason"] = detail.get("finish_reason")
                    if isinstance(detail.get("usage"), dict):
                        metadata["usage"] = detail["usage"]
                    metadata.setdefault("upstream_errors", []).append(
                        {
                            "attempt": attempt + 1,
                            "status": response.status_code,
                            "code": sanitized(detail.get("code", "")),
                            "message": sanitized(detail.get("message", "")),
                            "retry_after": sanitized(
                                response.headers.get("Retry-After", "")
                            ),
                        }
                    )
                    if detail.get("code") == "invalid_model_output" or (
                        response.status_code != 429 and response.status_code < 500
                    ):
                        raise GatewayError(f"Upstream HTTP {response.status_code}")
                    if attempt + 1 == self.attempts:
                        raise GatewayError(
                            f"Upstream HTTP {response.status_code}; retry limit reached"
                        )
                    try:
                        delay = max(
                            self.retry_delay,
                            float(response.headers.get("Retry-After", 0)),
                        )
                    except ValueError:
                        delay = self.retry_delay
                if delay >= self.deadline - (time.monotonic() - started):
                    raise GatewayError("Retry exceeds gateway deadline")
                await asyncio.sleep(delay)
        raise GatewayError("No usable response")


def failure_metadata(error, agent, spec, metadata=None):
    return {
        "agent": agent,
        "mode": spec.analysis_mode,
        "model": spec.model_id,
        **(metadata or {}),
        **error.metadata,
        "error": str(error),
    }


# Retain the public constructor used by existing scripts and saved tests.
OpenRouterGateway = OpenAICompatibleGateway
