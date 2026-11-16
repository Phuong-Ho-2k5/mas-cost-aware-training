#!/usr/bin/env bash
set -euo pipefail

# This script runs only in the dedicated distro whose VHDX is on D:.
requirements_input=${1:?Pass the requirements-vllm.in path}
runtime_root=/opt/mas-vllm
mkdir -p "$runtime_root"/{cache,tmp,reports}
export TMPDIR="$runtime_root/tmp"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export UV_CACHE_DIR="$runtime_root/cache/uv"
export HF_HOME="$runtime_root/cache/huggingface"
export XDG_CACHE_HOME="$runtime_root/cache"
export VLLM_CACHE_ROOT="$runtime_root/cache/vllm"
export TRITON_CACHE_DIR="$runtime_root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/cache/torchinductor"
export DEBIAN_FRONTEND=noninteractive

# Only a completed environment with unchanged input can skip a fresh install.
cached_ready=$(python3 - "$runtime_root" "$requirements_input" <<'PY'
import json
from pathlib import Path
import sys
root, requested = map(Path, sys.argv[1:])
try:
    ready = json.loads((root / 'environment.json').read_text()).get('environment_ready') is True
    same_input = (root / 'requirements.requested.in').read_bytes() == requested.read_bytes()
    reusable = ready and same_input and (root / 'environment.lock').is_file() and (root / 'bootstrap/bin/uv').is_file()
except (OSError, ValueError):
    reusable = False
print(int(reusable))
PY
)
if [[ "$cached_ready" == 1 ]]; then
    cp "$runtime_root/environment.json" "$runtime_root/environment.previous.json"
fi
python3 - "$runtime_root/environment.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'task': 1, 'state': 'running', 'environment_ready': False, 'model_hardware_gate': 'not_run'}))
PY
record_failure() {
    local exit_code=$?
    python3 - "$runtime_root/environment.json" "$exit_code" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'task': 1, 'state': 'failed', 'environment_ready': False, 'exit_code': int(sys.argv[2]), 'model_hardware_gate': 'not_run'}))
PY
    exit "$exit_code"
}
trap record_failure ERR

cat > "$runtime_root/environment.sh" <<'ENV'
export TMPDIR=/opt/mas-vllm/tmp
export PIP_CACHE_DIR=/opt/mas-vllm/cache/pip
export UV_CACHE_DIR=/opt/mas-vllm/cache/uv
export HF_HOME=/opt/mas-vllm/cache/huggingface
export XDG_CACHE_HOME=/opt/mas-vllm/cache
export VLLM_CACHE_ROOT=/opt/mas-vllm/cache/vllm
export TRITON_CACHE_DIR=/opt/mas-vllm/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/opt/mas-vllm/cache/torchinductor
ENV

uv_bin="$runtime_root/bootstrap/bin/uv"
python_bin="$runtime_root/venv/bin/python"
if [[ "$cached_ready" == 1 ]]; then
    echo 'Reusing the completed environment; no dependency downloads or synchronization.'
else
    apt-get update
    apt-get install -y --no-install-recommends python3.12-venv ca-certificates
    if [[ ! -x "$runtime_root/bootstrap/bin/python" ]]; then
        python3.12 -m venv "$runtime_root/bootstrap"
    fi
    "$runtime_root/bootstrap/bin/python" -m pip install 'uv==0.13.8'
    if [[ ! -x "$python_bin" ]]; then
        "$uv_bin" venv --python python3.12 "$runtime_root/venv"
    fi
    cp "$requirements_input" "$runtime_root/requirements.requested.in"
    if ! "$uv_bin" pip compile --python "$python_bin" "$runtime_root/requirements.requested.in" --output-file "$runtime_root/environment.lock" 2> "$runtime_root/reports/initial-resolve.log"; then
        echo 'Initial candidate did not resolve; retaining resolver evidence and trying the coherent official v0.20.0 family.'
        printf 'vllm==0.20.0\ntransformers==5.12.0\n' > "$runtime_root/requirements.fallback.in"
        "$uv_bin" pip compile --python "$python_bin" "$runtime_root/requirements.fallback.in" --output-file "$runtime_root/environment.lock" 2> "$runtime_root/reports/fallback-resolve.log"
    fi
    "$uv_bin" pip sync --python "$python_bin" "$runtime_root/environment.lock"
fi
"$uv_bin" pip check --python "$python_bin"
"$uv_bin" pip freeze --python "$python_bin" > "$runtime_root/environment.freeze.txt"

"$python_bin" - <<'PY'
import importlib.metadata as metadata
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import torch
import vllm

extension = 'vllm._C_stable_libtorch' if importlib.util.find_spec('vllm._C_stable_libtorch') else 'vllm._C'
importlib.import_module(extension)

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available inside the dedicated distro')
capability = torch.cuda.get_device_capability(0)
if capability < (7, 5):
    raise RuntimeError(f'Unsupported GPU capability: {capability}')
x = torch.arange(16, device='cuda', dtype=torch.float16)
assert (x + 1).sum().item() == 136
torch.cuda.synchronize()
report = {
    'schema_version': 1,
    'task': 1,
    'environment_ready': True,
    'state': 'ready',
    'model_downloaded': False,
    'model_hardware_gate': 'not_run',
    'python': sys.version,
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'versions': {name: metadata.version(name) for name in ('vllm', 'torch', 'transformers', 'triton')},
    'torch_cuda': torch.version.cuda,
    'torch_arch_list': torch.cuda.get_arch_list(),
    'gpu': torch.cuda.get_device_name(0),
    'compute_capability': list(capability),
    'cuda_allocation_test': 'passed',
    'vllm_extension': extension,
    'environment_lock': '/opt/mas-vllm/environment.lock',
    'cache_root': '/opt/mas-vllm/cache',
    'package_index': 'https://pypi.org/simple',
    'artifact_hashes_recorded': False,
    'provenance_note': 'Resolved package versions and actual CUDA runtime recorded; exact installed wheel hashes are not captured by this environment-only check.',
}
Path('/opt/mas-vllm/environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
