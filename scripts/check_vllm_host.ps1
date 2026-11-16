param(
    [string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp\vllm\preflight.json'),
    [string]$Distro = 'Ubuntu',
    [double]$MinimumCFreeGiB = 4,
    [double]$MinimumDFreeGiB = 35
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vllm_common.ps1')
$Output = Assert-VllmDataPath -Path $Output
$cFree = (Get-Volume -DriveLetter C).SizeRemaining / 1GB
$dFree = (Get-Volume -DriveLetter D).SizeRemaining / 1GB
$checks = [ordered]@{}
$records = @(Get-VllmDistroRecords)
try {
    Assert-VllmHostSpace -CFreeGiB $cFree -DFreeGiB $dFree -MinimumCFreeGiB $MinimumCFreeGiB -MinimumDFreeGiB $MinimumDFreeGiB
    $checks.storage = @{passed=$true}
} catch { $checks.storage = @{passed=$false; error=$_.Exception.Message} }
try {
    $wslVersion = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('--version')
    $record = @($records | Where-Object DistributionName -eq $Distro)
    if ($record.Count -ne 1 -or $record[0].Version -ne 2) { throw "WSL2 distro $Distro not found." }
    $os = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/bin/cat', '/etc/os-release')
    $kernel = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/bin/uname', '-r')
    $gpu = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader')
    $capability = [double]::Parse(($gpu.Split("`n")[0].Split(',')[2].Trim()), [Globalization.CultureInfo]::InvariantCulture)
    if ($capability -lt 7.5) { throw 'GPU compute capability is below 7.5.' }
    $checks.wsl_gpu = @{passed=$true; distro=$Distro; os_release=$os; kernel=$kernel; gpu=$gpu}
} catch { $checks.wsl_gpu = @{passed=$false; error=$_.Exception.Message} }
$warnings = @()
if ($cFree -lt 5) { $warnings += 'C is below the original 5 GiB planning reserve. All task payloads must stay on D; this run uses a 4 GiB host reserve.' }
$report = [ordered]@{
    schema_version=1; task=1; timestamp_utc=[DateTime]::UtcNow.ToString('o')
    ready=(@($checks.Values | Where-Object { -not $_.passed }).Count -eq 0)
    host=@{c_free_gib=[math]::Round($cFree,3); d_free_gib=[math]::Round($dFree,3); minimum_c_gib=$MinimumCFreeGiB; minimum_d_gib=$MinimumDFreeGiB}
    wsl_version=$wslVersion; registered_distros=$records; checks=$checks; warnings=$warnings
    model_downloaded=$false; model_hardware_gate='not_run'
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath $Output
$report | ConvertTo-Json -Depth 8
if (-not $report.ready) { exit 1 }
