# Parse dotenv with python-dotenv; never execute its contents as PowerShell code.
function Import-ProjectEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [string]$EnvFile = (Join-Path (Split-Path -Parent $PSScriptRoot) '.env')
    )
    if (-not (Test-Path -LiteralPath $EnvFile)) { return }
    $dotenvJson = & $Python -c 'import json, sys; from dotenv import dotenv_values; print(json.dumps({k:v for k,v in dotenv_values(sys.argv[1]).items() if v is not None}))' $EnvFile
    if ($LASTEXITCODE -ne 0) { throw 'Cannot load project .env; check python-dotenv installation.' }
    $dotenvValues = $dotenvJson | ConvertFrom-Json
    foreach ($entry in $dotenvValues.PSObject.Properties) {
        if (-not (Test-Path -LiteralPath "Env:$($entry.Name)")) {
            [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, 'Process')
        }
    }
}
