@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0focus_classifier\start.ps1" -Upload
if errorlevel 1 pause
