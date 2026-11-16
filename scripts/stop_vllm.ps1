param(
    [ValidateSet(8000,8001)][int]$Port = 8000,
    [ValidateSet('MAS-vLLM')][string]$Distro = 'MAS-vLLM'
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/vllm_common.ps1"
$root = Assert-VllmDataPath -Path (Split-Path -Parent $PSScriptRoot)
Assert-VllmDistroTarget -Distro $Distro -InstallRoot 'D:/HK261/wsl/mas-vllm' -Records @(Get-VllmDistroRecords)
& "$root/.venv/Scripts/python.exe" "$root/scripts/vllm_service.py" stop --port $Port
if ($LASTEXITCODE -ne 0) { throw 'vLLM stop refused or failed; no unrelated process will be stopped.' }
