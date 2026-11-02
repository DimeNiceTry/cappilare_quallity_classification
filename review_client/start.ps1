$ErrorActionPreference = 'Stop'
$reviewUrl = 'http://127.0.0.1:8766'
$reviewHealthy = $false
try {
    $reply = Invoke-RestMethod -Uri ($reviewUrl + '/api/health') -TimeoutSec 2
    $reviewHealthy = $reply.app -eq 'capillary-cross-review'
} catch {}
if (-not $reviewHealthy) {
    $reviewPython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $reviewPython)) {
        $reviewPython = (Get-Command python -ErrorAction Stop).Source
    }
    $reviewServer = Join-Path $PSScriptRoot 'server.py'
    Start-Process -FilePath $reviewPython -ArgumentList @('"' + $reviewServer + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server-error.log')
    for ($reviewAttempt = 0; $reviewAttempt -lt 30; $reviewAttempt++) {
        Start-Sleep -Milliseconds 200
        try {
            $reply = Invoke-RestMethod -Uri ($reviewUrl + '/api/health') -TimeoutSec 1
            if ($reply.app -eq 'capillary-cross-review') { $reviewHealthy = $true; break }
        } catch {}
    }
}
if (-not $reviewHealthy) { throw 'Cannot start annotation client. See review_client/server-error.log.' }
Start-Process $reviewUrl
