"""OpenAI-compatible JSON-schema subset for local MAS inference.

Run through scripts/start_local_llm.ps1; one Uvicorn worker only.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import BaseModel, ConfigDict, Field

from services.local_llm.generation import TransformersEngine


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=100_000)


class SchemaSpec(StrictModel):
    name: str = Field(max_length=100)
    strict: bool = False
    schema_: dict = Field(alias="schema")


class ResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: SchemaSpec


class ChatRequest(StrictModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    temperature: Literal[0] = 0
    max_tokens: int = Field(default=1024, ge=1, le=2048, strict=True)
    response_format: ResponseFormat
    stream: Literal[False] = False


def validate_schema(schema):
    """Accept MAS's structural subset; refuse external references and other dialects."""
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "$defs",
        "$ref",
        "title",
        "description",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
    }
    if len(json.dumps(schema)) > 65536:
        raise ValueError("Schema too large")
    Draft202012Validator.check_schema(schema)

    def visit(node, depth=0, references=frozenset()):
        if not isinstance(node, dict) or depth > 32:
            raise ValueError("Unsupported schema node/depth")
        unknown = set(node) - supported
        if unknown:
            raise ValueError(
                "Unsupported schema keywords: " + ", ".join(sorted(unknown))
            )
        if "$ref" in node and not node["$ref"].startswith("#/$defs/"):
            raise ValueError("Only internal $defs references are supported")
        if "$ref" in node:
            reference = node["$ref"]
            name = reference[len("#/$defs/") :].replace("~1", "/").replace("~0", "~")
            target = schema.get("$defs", {}).get(name)
            if target is None or reference in references:
                raise ValueError("Unresolved or recursive schema reference")
            visit(target, depth + 1, references | {reference})
        for key in ("properties", "$defs"):
            for child in node.get(key, {}).values():
                visit(child, depth + 1, references)
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                visit(node[key], depth + 1, references)
        for child in node.get("anyOf", []):
            visit(child, depth + 1, references)

    visit(schema)


def create_app(engine_factory=TransformersEngine.from_environment):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await asyncio.to_thread(engine_factory)
        app.state.worker = asyncio.Semaphore(1)
        app.state.pending = 0
        app.state.jobs = set()
        app.state.cancellations = set()
        yield
        for cancel in app.state.cancellations:
            cancel.set()
        if app.state.jobs:
            await asyncio.gather(*app.state.jobs, return_exceptions=True)

    app = FastAPI(title="MAS Local Qwen", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        detail = (
            error.detail
            if isinstance(error.detail, dict)
            else {"message": str(error.detail), "code": "invalid_request"}
        )
        return JSONResponse(status_code=error.status_code, content={"error": detail})

    @app.exception_handler(RequestValidationError)
    async def request_error(request, error):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "Request does not match the supported chat/schema contract",
                }
            },
        )

    @app.get("/health")
    async def health():
        return {
            "ready": True,
            "pending": app.state.pending,
            "pid": os.getpid(),
            "service_root": str(Path(__file__).resolve().parents[2]),
            **app.state.engine.info,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.engine.model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/token-count")
    async def token_count(body: ChatRequest):
        if body.model != app.state.engine.model_id:
            raise HTTPException(404, "Model not served")
        if not hasattr(app.state.engine, "count_tokens"):
            raise HTTPException(501, "Token counting unavailable")
        count = await asyncio.to_thread(app.state.engine.count_tokens, body)
        return {
            "prompt_tokens": count,
            "context_tokens": app.state.engine.info.get("context_tokens", 4096),
        }

    @app.post("/v1/chat/completions")
    async def complete(body: ChatRequest, request: Request):
        if body.model != app.state.engine.model_id:
            raise HTTPException(
                404,
                {"code": "model_not_found", "message": "Requested model is not loaded"},
            )
        try:
            validate_schema(body.response_format.json_schema.schema_)
        except (ValueError, SchemaError) as error:
            raise HTTPException(
                422, {"code": "unsupported_schema", "message": str(error)[:300]}
            ) from None
        if app.state.pending >= 2:
            raise HTTPException(
                429,
                {
                    "code": "server_busy",
                    "message": "One active and one waiting request already admitted",
                },
            )
        cancel = threading.Event()
        app.state.pending += 1
        app.state.cancellations.add(cancel)

        async def work():
            try:
                async with app.state.worker:
                    return await asyncio.to_thread(
                        app.state.engine.generate, body, cancel
                    )
            finally:
                app.state.pending -= 1
                app.state.cancellations.discard(cancel)

        job = asyncio.create_task(work())
        app.state.jobs.add(job)

        def finished(task):
            app.state.jobs.discard(task)
            # Retrieve exceptions even when a disconnected client stopped awaiting.
            if not task.cancelled():
                task.exception()

        job.add_done_callback(finished)
        try:
            while not job.done():
                await asyncio.wait({job}, timeout=0.1)
                if await request.is_disconnected():
                    cancel.set()
                    raise HTTPException(
                        499,
                        {
                            "code": "client_disconnected",
                            "message": "Generation cancellation requested",
                        },
                    )
            result = await asyncio.shield(job)
        except asyncio.CancelledError:
            cancel.set()
            raise
        except TimeoutError:
            raise HTTPException(
                504,
                {
                    "code": "generation_timeout",
                    "message": "Generation deadline exceeded",
                },
            ) from None
        except ValueError as error:
            raise HTTPException(
                422, {"code": "invalid_context_or_schema", "message": str(error)[:300]}
            ) from None
        except RuntimeError:
            raise HTTPException(
                503,
                {
                    "code": "generation_failed",
                    "message": "Model resource or generation failure",
                },
            ) from None
        try:
            if result["finish_reason"] != "stop":
                raise ValueError("Incomplete output")
            value = json.loads(
                result["content"],
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("Non-finite output")
                ),
            )
            Draft202012Validator(body.response_format.json_schema.schema_).validate(
                value
            )
        except (ValueError, ValidationError):
            raise HTTPException(
                502,
                {
                    "code": "invalid_model_output",
                    "message": "Model output was incomplete or failed full JSON Schema validation",
                    "raw_output": result["content"],
                    "finish_reason": result["finish_reason"],
                    "usage": result["usage"],
                    "local_metadata": {
                        **app.state.engine.info,
                        **result.get("resources", {}),
                    },
                },
            ) from None
        return {
            "id": "local-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": app.state.engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": result["content"]},
                }
            ],
            "usage": result["usage"],
            "local_metadata": {**app.state.engine.info, **result.get("resources", {})},
        }

    return app


app = create_app()
