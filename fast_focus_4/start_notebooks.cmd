@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m notebook .
if errorlevel 1 pause
