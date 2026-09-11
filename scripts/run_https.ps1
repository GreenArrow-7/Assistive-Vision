param(
    [Parameter(Mandatory=$true)][string]$Certificate,
    [Parameter(Mandatory=$true)][string]$Key,
    [int]$Port = 8443
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (!(Test-Path -LiteralPath $Certificate) -or !(Test-Path -LiteralPath $Key)) {
    throw 'Certificate or key missing. See README HTTPS Smartphone Testing.'
}
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$serverArgs = @('-m','uvicorn','server.main:app','--host','0.0.0.0','--port',"$Port",
    '--ssl-certfile',$Certificate,'--ssl-keyfile',$Key)
if (Test-Path -LiteralPath '.env') { $serverArgs += @('--env-file','.env') }
& $pythonPath @serverArgs
