param([switch]$Upload)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
$url = 'http://127.0.0.1:8767'
$running = $false
try {
    $state = Invoke-RestMethod -Uri ($url + '/api/health') -TimeoutSec 2
    if ($state.app -ne 'capillary-quality-comparison') { throw 'Port 8767 belongs to another application.' }
    $running = $true
} catch {}
if (-not $running) {
    Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'focus_classifier.serve') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server-error.log')
    for ($attempt=0; $attempt -lt 80; $attempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $state = Invoke-RestMethod -Uri ($url + '/api/health') -TimeoutSec 1
            if ($state.app -eq 'capillary-quality-comparison') { $running = $true; break }
        } catch {}
    }
}
if (-not $running) { throw 'Comparison server did not start. See focus_classifier/server-error.log.' }
if ($Upload) { Start-Process ($url + '/upload') } else { Start-Process $url }
