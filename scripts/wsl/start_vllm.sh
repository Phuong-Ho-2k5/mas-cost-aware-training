#!/usr/bin/env bash
set -euo pipefail
source /opt/mas-vllm/environment.sh
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export CUDA_CACHE_PATH=/opt/mas-vllm/cache/cuda
export TORCH_EXTENSIONS_DIR=/opt/mas-vllm/cache/torch_extensions
mkdir -p "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
command -v cc >/dev/null || { echo 'Triton JIT needs build-essential and python3.12-dev in the dedicated distro.' >&2; exit 2; }
port=${1:-8001}
batch_tokens=${2:-1024}
memory_fraction=${3:-0.65}
compact_json=${4:-false}
context_tokens=${5:-4096}
reasoning_parser=${6:-none}
[[ "$context_tokens" == 4096 || "$context_tokens" == 8192 ]] || exit 2
[[ "$compact_json" == true || "$compact_json" == false ]] || exit 2
[[ "$reasoning_parser" == none || "$reasoning_parser" == qwen3 ]] || exit 2
decoder_args=()
if [[ "$compact_json" == true ]]; then
    decoder_args=(--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
fi
if [[ "$reasoning_parser" != none ]]; then
    decoder_args+=(--reasoning-parser "$reasoning_parser")
fi
[[ "$port" == 8001 || "$port" == 8000 ]] || { echo 'Use candidate8001 or final8000.' >&2; exit 2; }
[[ "$batch_tokens" == 1024 || "$batch_tokens" == 512 ]] || exit 2
[[ "$memory_fraction" == 0.65 || "$memory_fraction" == 0.80 || "$memory_fraction" == 0.85 ]] || exit 2
exec /opt/mas-vllm/venv/bin/vllm serve Qwen/Qwen3.5-0.8B \
    --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --tokenizer-revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --served-model-name Qwen/Qwen3.5-0.8B \
    --host 127.0.0.1 --port "$port" \
    --dtype float16 --language-model-only \
    --max-model-len "$context_tokens" --max-num-seqs 1 \
    --max-num-batched-tokens "$batch_tokens" \
    --tensor-parallel-size 1 --enforce-eager \
    --gpu-memory-utilization "$memory_fraction" \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking":false}' "${decoder_args[@]}"
