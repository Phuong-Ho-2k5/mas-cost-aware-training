$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv }
New-Item -ItemType Directory -Force -Path 'tmp/pip' | Out-Null
$env:TEMP = Join-Path (Get-Location) 'tmp/pip'
$env:TMP = $env:TEMP
& .venv/Scripts/python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv/Scripts/python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv/Scripts/python -m pip install -e '.[dev]'
exit $LASTEXITCODE
