"""One loaded Transformers model with schema-constrained greedy generation."""

import os
import time

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def validate_token_budget(prompt_tokens, output_tokens, context_tokens):
    if prompt_tokens + output_tokens > context_tokens:
        raise ValueError(
            f"Requested context {prompt_tokens}+{output_tokens} exceeds {context_tokens}; evidence was not truncated"
        )


class TransformersEngine:
    def __init__(
        self,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        context_tokens=4096,
        generation_seconds=90,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; select cpu explicitly")
        started = time.monotonic()
        torch.set_num_threads(4)
        self.model_id = model_id
        self.context_tokens, self.generation_seconds = (
            context_tokens,
            generation_seconds,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        # Building the 152k-token vocabulary trie per request costs seconds.
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )

        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        from copy import deepcopy

        self.generation_config = deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.info = {
            "model": model_id,
            "revision": self.model.config._commit_hash or revision,
            "dtype": str(dtype).removeprefix("torch."),
            "device": device,
            "engine": "transformers",
            "load_seconds": time.monotonic() - started,
            "context_tokens": context_tokens,
            "generation_seconds": generation_seconds,
            "torch_version": torch.__version__,
        }

    @classmethod
    def from_environment(cls):
        return cls(
            model_id=os.environ.get("LOCAL_LLM_MODEL", MODEL_ID),
            revision=os.environ.get("LOCAL_LLM_REVISION", MODEL_REVISION),
            device=os.environ.get("LOCAL_LLM_DEVICE", "cpu"),
        )

    def count_tokens(self, body):
        messages = [item.model_dump() for item in body.messages]
        tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return len(tokens)

    def generate(self, body, cancel):
        from transformers import StoppingCriteria, StoppingCriteriaList
        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        torch = self.torch
        messages = [item.model_dump() for item in body.messages]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        validate_token_budget(prompt_tokens, body.max_tokens, self.context_tokens)
        parser = JsonSchemaParser(
            body.response_format.json_schema.schema_,
            config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=2, force_json_field_order=True
            ),
        )
        prefix = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, parser
        )
        started = time.monotonic()

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set() or time.monotonic() - started >= self_limit

        self_limit = self.generation_seconds
        if cancel.is_set():
            raise TimeoutError("Request cancelled before generation")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=body.max_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        new_tokens = output[0, prompt_tokens:]
        elapsed = time.monotonic() - started
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        finish = (
            "stop" if len(new_tokens) and int(new_tokens[-1]) in eos_ids else "length"
        )
        if cancel.is_set() or elapsed >= self.generation_seconds:
            raise TimeoutError("Generation deadline or client cancellation")
        import psutil

        resources = {
            "rss_mb": psutil.Process().memory_info().rss / 1024**2,
            "generation_seconds": elapsed,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if self.model.device.type == "cuda"
                else None
            ),
        }
        return {
            "content": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(new_tokens),
                "total_tokens": prompt_tokens + len(new_tokens),
            },
            "resources": resources,
        }
