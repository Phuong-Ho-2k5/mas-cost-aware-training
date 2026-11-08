param([ValidateRange(1024, 65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -eq 0) { Write-Host "No server listening on port $Port."; exit 0 }
$identity = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
if (-not [string]::Equals($identity.service_root, $repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The listening service is not the local Qwen service from this repository.'
}
foreach ($taskProcessId in ($listeners.OwningProcess | Select-Object -Unique)) {
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $taskProcessId"
    if ($taskProcess.CommandLine -notlike '*uvicorn services.local_llm.server:app*' -or $identity.pid -ne $taskProcessId) {
        throw "Port $Port belongs to another process; refusing to stop PID $taskProcessId."
    }
    Stop-Process -Id $taskProcessId
    Write-Host "Stopped local Qwen server PID $taskProcessId."
}
