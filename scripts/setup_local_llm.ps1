param([string]$ModelCache)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$basePython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $basePython)) { throw 'Run scripts/setup.ps1 first to install the MAS environment and CUDA PyTorch.' }
. (Join-Path $PSScriptRoot 'load_environment.ps1')
Import-ProjectEnvironment -Python $basePython -EnvFile (Join-Path $repoRoot '.env')
if (-not $ModelCache) { $ModelCache = $env:LOCAL_LLM_MODEL_CACHE }
if (-not $ModelCache) { $ModelCache = Join-Path $repoRoot 'tmp\model-cache' }
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot 'tmp'), $ModelCache | Out-Null
$env:TEMP = Join-Path $repoRoot 'tmp'
$env:TMP = $env:TEMP
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $ModelCache 'huggingface' }
$env:HF_HUB_DISABLE_XET = '1'
$serverEnv = Join-Path $repoRoot 'tmp\llm-server-venv'
if (-not (Test-Path -LiteralPath (Join-Path $serverEnv 'Scripts\python.exe'))) {
    & $basePython -m venv --system-site-packages $serverEnv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create serving environment' }
}
$basePackages = Join-Path $repoRoot '.venv\Lib\site-packages'
# Add the existing CUDA torch install without changing its package versions.
Set-Content -Encoding ascii -LiteralPath (Join-Path $serverEnv 'Lib\site-packages\mas-base.pth') -Value @($basePackages, (Join-Path $repoRoot 'src'))
$serverPython = Join-Path $serverEnv 'Scripts\python.exe'
& $serverPython -m pip install --cache-dir (Join-Path $ModelCache 'pip') -r requirements-llm-server.txt
if ($LASTEXITCODE -ne 0) { throw 'Serving dependency installation failed' }
Write-Host 'Ready. Start: .\scripts\start_local_llm.ps1 -Device cuda'
