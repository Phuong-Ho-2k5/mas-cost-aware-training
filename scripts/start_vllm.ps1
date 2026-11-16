param(
    [ValidateSet(8000,8001)][int]$Port = 8000,
    [ValidateSet('MAS-vLLM')][string]$Distro = 'MAS-vLLM',
    [ValidateRange(1,600)][int]$TimeoutSeconds = 600
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/vllm_common.ps1"
$root = Assert-VllmDataPath -Path (Split-Path -Parent $PSScriptRoot)
Assert-VllmDistroTarget -Distro $Distro -InstallRoot 'D:/HK261/wsl/mas-vllm' -Records @(Get-VllmDistroRecords)
if (-not (@(Get-VllmDistroRecords | Where-Object DistributionName -eq $Distro).Count)) { throw 'Dedicated distro is not registered.' }
& "$root/.venv/Scripts/python.exe" "$root/scripts/vllm_service.py" start --port $Port --timeout $TimeoutSeconds
if ($LASTEXITCODE -ne 0) { throw 'vLLM start failed; inspect tmp/vllm/runtime-manifest.json and its owned run log.' }
