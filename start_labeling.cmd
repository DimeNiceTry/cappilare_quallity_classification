@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0review_client\start.ps1"
if errorlevel 1 pause
