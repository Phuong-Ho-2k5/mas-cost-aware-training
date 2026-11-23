import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


def response_schema(role, response_type, mode="vllm", max_trials=None, payload=None):
    """The exact schema sent to a provider, also used by benchmark scoring."""
    schema = response_type.model_json_schema()
    if mode != "vllm":
        raise ValueError("Only vLLM response schemas are supported")
    schema["properties"]["reason"]["maxLength"] = (
        512 if role == "analysis" else 240 if role == "analysis_detection" else 160
    )
    if role == "analysis":
        schema["properties"]["hypotheses"]["maxItems"] = 3
        hypothesis = schema["$defs"]["AnalysisHypothesis"]["properties"]
        hypothesis["cause"]["maxLength"] = 160
        hypothesis["hypothesis_id"]["maxLength"] = 48
        hypothesis["evidence_epochs"]["maxItems"] = 8

    if role == "planner":
        schema["properties"]["reason"]["minLength"] = 1
        schema["properties"]["proposals"]["maxItems"] = 3

    def compact(node):
        if isinstance(node, dict):
            return {
                k: compact(v) for k, v in node.items() if k not in ("title", "default")
            }
        if isinstance(node, list):
            return [compact(item) for item in node]
        return node

    schema = compact(schema)
    if payload is not None:
        from mas_demo.llm.output_schema import grounded_schema
        schema = grounded_schema(role, schema, payload)
    return schema


def _with_thought_field(schema, max_length=360):
    """Prepend a bounded `thought` field to the wire schema for structured CoT."""
    from copy import deepcopy
    wired = deepcopy(schema)
    thought_prop = {"type": "string", "minLength": 1, "maxLength": max_length}

    def inject(node):
        if isinstance(node, dict) and node.get("type") == "object" and "properties" in node:
            node["properties"] = {"thought": thought_prop, **node["properties"]}
            req = list(node.get("required", []))
            if "thought" not in req:
                node["required"] = ["thought", *req]

    if "anyOf" in wired:
        for branch in wired["anyOf"]:
            inject(branch)
    else:
        inject(wired)
    return wired


class VLLMGateway:
    def __init__(
        self,
        key,
        model="Qwen/Qwen3.5-0.8B",
        base_url="http://127.0.0.1:8000/v1",
        timeout=120,
        retry_delay=1,
        deadline=130,
        mode="vllm",
        attempts=1,
        max_tokens=1024,
        local_backend="vllm",
        enable_thinking=False,
        runtime_manifest=None,
    ):
        if mode != "vllm" or local_backend != "vllm":
            raise ValueError("Only the vLLM backend is supported")
        self.key, self.model = key, model
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retry_delay, self.deadline = timeout, retry_delay, deadline
        self.mode, self.attempts, self.max_tokens = mode, attempts, max_tokens
        self.local_backend = "vllm"
        self.enable_thinking = bool(enable_thinking)
        self.runtime_manifest = runtime_manifest or str(
            Path(__file__).resolve().parents[3] / "tmp/vllm/runtime-manifest.json"
        )

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
            result = asyncio.run(bounded())
        except GatewayError as error:
            metadata["latency_seconds"] = time.monotonic() - started
            error.metadata = dict(metadata)
            raise
        metadata["latency_seconds"] = time.monotonic() - started
        return result, metadata

    async def _complete(self, role, payload, response_type, instruction, metadata):
        from mas_demo.llm.health import verify_vllm_runtime

        # A separate executor keeps read-only WSL inspection within the deadline.
        inspector = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-identity")
        try:
            metadata["server"] = await asyncio.get_running_loop().run_in_executor(
                inspector, verify_vllm_runtime, self
            )
            if metadata["server"].get("structured_outputs_config") != {"backend": "xgrammar", "disable_any_whitespace": True}:
                raise RuntimeError("Start vLLM with the updated launcher for compact structured JSON")
        except RuntimeError as error:
            raise GatewayError("vLLM runtime verification failed: " + str(error)) from None
        finally:
            inspector.shutdown(wait=False, cancel_futures=True)
        schema = response_schema(
            role, response_type, self.mode, payload.get("max_trials"), payload=payload
        )
        schema_name = {
            "analysis": "training_diagnosis",
            "analysis_detection": "training_detection",
            "analysis_hypotheses": "training_hypotheses",
            "planner": "training_plan",
            "evaluation": "training_evaluation",
        }[role]
        if role in ("analysis_detection", "analysis_hypotheses"):
            instruction = instruction.rstrip() + "\n\n" + load_prompt("local_vllm_analysis").strip()
        elif role == "planner":
            instruction = instruction.rstrip() + "\n\n" + load_prompt("local_vllm_planner").strip()
        else:
            instruction = (
                instruction.rstrip()
                + "\n\n"
                + load_prompt("local").strip()
                + " "
                + load_prompt("local_vllm").strip()
            )
        system = instruction
        metadata["response_schema"] = schema
        thinking = self.enable_thinking and bool(metadata["server"].get("reasoning_parser"))
        structured_thought = (
            not thinking
            and role in ("analysis_detection", "analysis_hypotheses", "planner", "evaluation")
        )
        wire_schema = _with_thought_field(schema) if structured_thought else schema
        metadata["response_schema"] = wire_schema
        body = {
            "model": self.model,
            "temperature": 0.6 if thinking else 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {"role": "user", "content": json.dumps(payload, allow_nan=False, separators=(",", ":"))},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": False, "schema": wire_schema},
            },
        }
        if thinking:
            body["top_p"] = 0.95
            body["presence_penalty"] = 0.2
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
        started = time.monotonic()
        async with httpx.AsyncClient(trust_env=False) as client:
            from mas_demo.llm.context import fit_local_context

            await fit_local_context(
                client, self.base_url, body, metadata,
                {"Authorization": "Bearer " + self.key} if self.key else {},
            )
            metadata["input_payload"] = json.loads(body["messages"][1]["content"])
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
                            choice = data["choices"][0]
                            message = choice["message"]
                            content = message["content"]
                            reasoning = message.get("reasoning") or message.get("reasoning_content")
                            if reasoning:
                                metadata["reasoning"] = (
                                    reasoning.replace(self.key, "[REDACTED]")
                                    if self.key
                                    else reasoning
                                )
                            metadata["raw_output"] = (
                                content.replace(self.key, "[REDACTED]")
                                if self.key and content
                                else content
                            )
                            metadata["finish_reason"] = choice.get("finish_reason")
                            if data.get("model") != self.model:
                                raise GatewayError("Upstream model identity mismatch")
                            if choice.get("finish_reason") != "stop":
                                raise ValueError("Incomplete or refused model output")
                            parsed = json.loads(content)
                            if structured_thought and isinstance(parsed, dict) and "thought" in parsed:
                                thought = parsed.pop("thought")
                                metadata["thought"] = (
                                    thought.replace(self.key, "[REDACTED]")
                                    if self.key and isinstance(thought, str)
                                    else thought
                                )
                            Draft202012Validator(schema).validate(parsed)
                            result = response_type.model_validate(parsed)
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
