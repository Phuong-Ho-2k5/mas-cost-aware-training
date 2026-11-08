param(
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [string]$ModelCache = 'D:\HK261\model-cache',
    [ValidateRange(1024, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$serverPython = Join-Path $repoRoot 'tmp\llm-server-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $serverPython)) { throw 'Run scripts/setup_local_llm.ps1 first.' }
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot 'tmp'), $ModelCache | Out-Null
$env:TEMP = Join-Path $repoRoot 'tmp'
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $ModelCache 'huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:LOCAL_LLM_DEVICE = $Device
$env:TOKENIZERS_PARALLELISM = 'false'
Write-Host "Qwen on $Device; loopback API at http://127.0.0.1:$Port/v1. Ctrl+C stops the server."
& $serverPython -m uvicorn services.local_llm.server:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
