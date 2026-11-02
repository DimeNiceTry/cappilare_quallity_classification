param([switch]$Research)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$environmentPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    $pythonCommand = $null
    $pythonPrefix = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.11','-3.10')) {
            & py $version -c 'import sys; assert sys.version_info[:2] in [(3,10),(3,11)]' 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonCommand = 'py'; $pythonPrefix = @($version); break }
        }
    }
    if (-not $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c 'import sys; assert sys.version_info[:2] in [(3,10),(3,11)]' 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonCommand = 'python' }
    }
    if (-not $pythonCommand) { throw 'Install Python 3.11 or 3.10, then run setup.cmd again: https://www.python.org/downloads/' }
    & $pythonCommand @pythonPrefix -m venv (Join-Path $taskRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv.' }
}
& $environmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to update pip.' }
$requirements = if ($Research) { 'requirements-research.txt' } else { 'requirements.txt' }
& $environmentPython -m pip install -r (Join-Path $taskRoot $requirements)
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and Python version.' }
& $environmentPython (Join-Path $taskRoot 'scripts\check_install.py')
if ($LASTEXITCODE -ne 0) { throw 'Installation check failed.' }
Write-Output 'Setup complete. Run start_photo_evaluation.cmd.'
