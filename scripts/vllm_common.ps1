# Shared placement/ownership guards. Dot-sourcing this file has no side effects.
function Assert-VllmDataPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -notmatch '^[dD]:[\\/]') { throw 'vLLM task data must be on drive D using an absolute path.' }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($full -eq 'D:') { throw 'Use a dedicated directory, not the root of drive D.' }
    $ancestor = $full
    while ($ancestor -and $ancestor -ne 'D:') {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -Force -LiteralPath $ancestor
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not allowed for vLLM data: $ancestor"
            }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $full
}

function Assert-VllmHostSpace {
    param([double]$CFreeGiB, [double]$DFreeGiB,
          [double]$MinimumCFreeGiB = 4, [double]$MinimumDFreeGiB = 35)
    if ($CFreeGiB -lt $MinimumCFreeGiB -or $DFreeGiB -lt $MinimumDFreeGiB) {
        throw "Insufficient physical host storage: C=$CFreeGiB GiB (need $MinimumCFreeGiB), D=$DFreeGiB GiB (need $MinimumDFreeGiB). Virtual Linux df capacity is not host free space."
    }
}

function Get-VllmOutputPath {
    param([string]$Root, [string]$Child)
    return Assert-VllmDataPath -Path (Join-Path $Root $Child)
}

function Get-VllmMinimumD {
    param([bool]$EnvironmentReady)
    if ($EnvironmentReady) { return 10 }
    return 35
}

function Write-VllmSetupStatus {
    param([string]$Output, [ValidateSet('running','ready','failed')][string]$State,
          [string]$ErrorMessage, $Payload, [string]$PreviousOutput)
    $Output = Assert-VllmDataPath -Path $Output
    if ($PreviousOutput) {
        $PreviousOutput = Assert-VllmDataPath -Path $PreviousOutput
        if (Test-Path -LiteralPath $Output) {
            $prior = Get-Content -Raw -LiteralPath $Output | ConvertFrom-Json
            if ($prior.environment_ready -eq $true) {
                Copy-Item -LiteralPath $Output -Destination $PreviousOutput -Force
            }
        }
    }
    $status = [ordered]@{}
    if ($Payload) {
        foreach ($entry in $Payload.PSObject.Properties) { $status[$entry.Name] = $entry.Value }
    }
    $status.schema_version = 1
    $status.task = 1
    $status.state = $State
    $status.timestamp_utc = [DateTime]::UtcNow.ToString('o')
    $status.environment_ready = ($State -eq 'ready')
    $status.model_downloaded = $false
    $status.model_hardware_gate = 'not_run'
    if ($ErrorMessage) { $status.error = $ErrorMessage }
    $status | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $Output
}

function Get-VllmDistroRecords {
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path -LiteralPath $key) {
        Get-ChildItem -LiteralPath $key | ForEach-Object {
            $record = Get-ItemProperty -LiteralPath $_.PSPath
            [pscustomobject]@{DistributionName=$record.DistributionName; BasePath=$record.BasePath; Version=$record.Version}
        }
    }
}

function Assert-VllmDistroTarget {
    param([string]$Distro, [string]$InstallRoot, [array]$Records)
    if ($Distro -notmatch '^MAS-vLLM(?:-[A-Za-z0-9_-]+)?$') { throw 'Use a dedicated MAS-vLLM distro name.' }
    $full = Assert-VllmDataPath -Path $InstallRoot
    $match = @($Records | Where-Object { $_.DistributionName -eq $Distro })
    $marker = Get-VllmOutputPath -Root $full -Child 'mas-vllm-owner.json'
    $null = Get-VllmOutputPath -Root $full -Child 'ext4.vhdx'
    if ($match.Count -gt 0) {
        $actual = [IO.Path]::GetFullPath(($match[0].BasePath -replace '^\\\\\?\\', '')).TrimEnd('\')
        if ($actual -ine $full -or $match[0].Version -ne 2) { throw 'Existing distro does not match the dedicated D: WSL2 target.' }
        if (-not (Test-Path -LiteralPath $marker)) { throw 'Existing distro has no task ownership marker.' }
    }
    if (Test-Path -LiteralPath $full) {
        if (-not (Test-Path -LiteralPath $marker)) { throw 'Refusing an existing unowned install directory.' }
        $owner = Get-Content -Raw -LiteralPath $marker | ConvertFrom-Json
        if ($owner.distro -ne $Distro -or $owner.install_root -ine $full) { throw 'Ownership marker does not match the requested target.' }
        if ($match.Count -eq 0 -and @(Get-ChildItem -Force -LiteralPath $full | Where-Object Name -ne 'mas-vllm-owner.json').Count -gt 0) {
            throw 'Unregistered target contains data; inspect it before retrying import.'
        }
    }
}

function Invoke-VllmSetupProcess {
    param([string]$Command, [string[]]$Arguments, [string]$Output, [string]$ErrorOutput, [scriptblock]$Monitor)
    $Output = Assert-VllmDataPath -Path $Output
    $ErrorOutput = Assert-VllmDataPath -Path $ErrorOutput
    $quoted = @($Arguments | ForEach-Object {
        if ($_ -match '["\r\n]') { throw 'Invalid native argument.' }
        # WSL parses option names from the raw command line: leave flags unquoted.
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    })
    $process = Start-Process -FilePath $Command -ArgumentList ($quoted -join ' ') -PassThru -WindowStyle Hidden -RedirectStandardOutput $Output -RedirectStandardError $ErrorOutput
    # Cache the process handle before exit; PS 5.1 can otherwise lose ExitCode.
    $null = $process.Handle
    while (-not $process.WaitForExit(3000)) { & $Monitor }
    $process.WaitForExit()
    if ($null -eq $process.ExitCode) { throw 'Native process exited without an observable exit code.' }
    return $process.ExitCode
}

function Invoke-VllmNative {
    param([string]$Command, [string[]]$Arguments)
    $result = & $Command @Arguments 2>&1
    $nativeExit = $LASTEXITCODE
    $output = (($result | ForEach-Object { "$_" }) -join "`n") -replace "`0", ''
    if ($nativeExit -ne 0) { throw "$Command failed ($nativeExit): $output" }
    return $output
}
