param(
    [string]$Distro = 'MAS-vLLM',
    [string]$InstallRoot = 'D:\HK261\wsl\mas-vllm',
    [string]$DataRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp\vllm'),
    [double]$MinimumCFreeGiB = 4
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot 'vllm_common.ps1')
$InstallRoot = Assert-VllmDataPath -Path $InstallRoot
$DataRoot = Assert-VllmDataPath -Path $DataRoot
$paths = @{}
foreach ($child in @('downloads','temp','environment.json','environment.previous.json','environment.freeze.txt','wsl-distribution-catalog.json','setup-transcript.log','linux-setup.log','linux-setup.err')) {
    $paths[$child] = Get-VllmOutputPath -Root $DataRoot -Child $child
}
$records = @(Get-VllmDistroRecords)
Assert-VllmDistroTarget -Distro $Distro -InstallRoot $InstallRoot -Records $records
$registered = @($records | Where-Object DistributionName -eq $Distro).Count -gt 0
$requirementsPath = Assert-VllmDataPath -Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'requirements-vllm.in')
$scriptPath = Assert-VllmDataPath -Path (Join-Path $PSScriptRoot 'wsl\setup_vllm.sh')
$reusable = $false
if ($registered) {
    try {
        $previous = (Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/bin/cat', '/opt/mas-vllm/environment.json')) | ConvertFrom-Json
        $requestedHash = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/usr/bin/sha256sum', '/opt/mas-vllm/requirements.requested.in')
        $reusable = ($previous.environment_ready -eq $true -and $requestedHash.Split(' ')[0] -ieq (Get-FileHash -Algorithm SHA256 -LiteralPath $requirementsPath).Hash)
    } catch { $reusable = $false }
}
New-Item -ItemType Directory -Force -Path $DataRoot, $paths.downloads, $paths.temp | Out-Null
$oldTemp, $oldTmp = $env:TEMP, $env:TMP
$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $paths['setup-transcript.log'] -Append | Out-Null
    $transcriptStarted = $true
    Write-VllmSetupStatus -Output $paths['environment.json'] -State running -PreviousOutput $paths['environment.previous.json']
    Assert-VllmHostSpace -CFreeGiB ((Get-Volume -DriveLetter C).SizeRemaining/1GB) -DFreeGiB ((Get-Volume -DriveLetter D).SizeRemaining/1GB) -MinimumCFreeGiB $MinimumCFreeGiB -MinimumDFreeGiB (Get-VllmMinimumD -EnvironmentReady $reusable)
    $env:TEMP = $paths.temp
    $env:TMP = $env:TEMP
    if (-not $registered) {
        $catalogUrl = 'https://raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json'
        $catalog = Invoke-RestMethod -Uri $catalogUrl
        $image = @($catalog.ModernDistributions.Ubuntu | Where-Object Name -eq 'Ubuntu-24.04')[0].Amd64Url
        if ($image.Url -notmatch '^https://releases\.ubuntu\.com/' -or $image.Sha256 -notmatch '^[a-fA-F0-9]{64}$') { throw 'Unexpected official Ubuntu rootfs metadata.' }
        $archive = Get-VllmOutputPath -Root $paths.downloads -Child ([IO.Path]::GetFileName($image.Url))
        $catalog | ConvertTo-Json -Depth 15 | Set-Content -Encoding utf8 -LiteralPath $paths['wsl-distribution-catalog.json']
        if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -ine $image.Sha256) {
            Write-Host "Downloading Ubuntu rootfs to $archive"
            & curl.exe --fail --location --retry 3 --output $archive $image.Url
            if ($LASTEXITCODE -ne 0) { throw 'Ubuntu rootfs download failed.' }
        }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -ine $image.Sha256) { throw 'Ubuntu rootfs checksum mismatch.' }
        New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
        $ownerPath = Get-VllmOutputPath -Root $InstallRoot -Child 'mas-vllm-owner.json'
        @{distro=$Distro; install_root=$InstallRoot; created_utc=[DateTime]::UtcNow.ToString('o'); rootfs_url=$image.Url; rootfs_sha256=$image.Sha256} | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $ownerPath
        Write-Host "Importing only $Distro into $InstallRoot"
        & wsl.exe --import $Distro $InstallRoot $archive --version 2
        if ($LASTEXITCODE -ne 0) { throw 'WSL import failed. Target retained for inspection; no existing distro was changed.' }
    }
    Assert-VllmDistroTarget -Distro $Distro -InstallRoot $InstallRoot -Records @(Get-VllmDistroRecords)
    $linuxScript = '/mnt/d/' + ($scriptPath.Substring(3) -replace '\\', '/')
    $inputFile = '/mnt/d/' + ($requirementsPath.Substring(3) -replace '\\', '/')
    Write-Host 'Preparing /opt/mas-vllm in the D-backed distro; no model weights will be downloaded.'
    $nativeArgs = @('-d', $Distro, '-u', 'root', '--exec', '/bin/bash', $linuxScript, $inputFile)
    try {
        $setupExit = Invoke-VllmSetupProcess -Command (Get-Command wsl.exe).Source -Arguments $nativeArgs -Output $paths['linux-setup.log'] -ErrorOutput $paths['linux-setup.err'] -Monitor {
            Assert-VllmHostSpace -CFreeGiB ((Get-Volume -DriveLetter C).SizeRemaining/1GB) -DFreeGiB ((Get-Volume -DriveLetter D).SizeRemaining/1GB) -MinimumCFreeGiB $MinimumCFreeGiB -MinimumDFreeGiB 10
        }
    } catch {
        # Only the dedicated, ownership-checked distro belongs to this setup.
        & wsl.exe --terminate $Distro | Out-Null
        throw
    }
    if ($setupExit -ne 0) { throw "Linux setup failed ($setupExit); inspect $($paths['linux-setup.err'])." }
    $linuxReport = (Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/bin/cat', '/opt/mas-vllm/environment.json')) | ConvertFrom-Json
    if (-not $linuxReport.environment_ready) { throw 'Linux did not report a ready environment.' }
    $freeze = Invoke-VllmNative -Command 'wsl.exe' -Arguments @('-d', $Distro, '-u', 'root', '--exec', '/bin/cat', '/opt/mas-vllm/environment.freeze.txt')
    $freeze | Set-Content -Encoding utf8 -LiteralPath $paths['environment.freeze.txt']
    Assert-VllmHostSpace -CFreeGiB ((Get-Volume -DriveLetter C).SizeRemaining/1GB) -DFreeGiB ((Get-Volume -DriveLetter D).SizeRemaining/1GB) -MinimumCFreeGiB $MinimumCFreeGiB -MinimumDFreeGiB 10
    Write-VllmSetupStatus -Output $paths['environment.json'] -State ready -Payload $linuxReport
    Write-Host "Task 1 environment ready in $Distro; report: $($paths['environment.json']). Model feasibility is Task 2."
} catch {
    Write-VllmSetupStatus -Output $paths['environment.json'] -State failed -ErrorMessage $_.Exception.Message
    throw
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
