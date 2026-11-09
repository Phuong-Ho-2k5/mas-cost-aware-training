"""Measure cold load and warm structured inference without starting the API."""
import argparse
import json
from pathlib import Path
import threading
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.local_llm.generation import TransformersEngine, MODEL_ID, MODEL_REVISION
from services.local_llm.server import ChatRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", default="artifacts/local-qwen-profile.json")
    args = parser.parse_args()
    engine = TransformersEngine(args.model, args.revision, args.device)
    body = ChatRequest.model_validate({"model": args.model, "max_tokens": 128,
        "messages": [{"role": "system", "content": "Return only a short JSON object with answer: a greeting of at most five words."},
                     {"role": "user", "content": "Say hello."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "greeting", "schema": {
            "type": "object", "properties": {"answer": {"type": "string", "maxLength": 40}},
            "required": ["answer"], "additionalProperties": False}}}})
    responses = [engine.generate(body, threading.Event()) for _ in range(2)]
    report = {**engine.info, "measurements": responses}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(r["finish_reason"] != "stop" for r in responses):
        raise SystemExit("Incomplete structured generation")


if __name__ == "__main__":
    main()
