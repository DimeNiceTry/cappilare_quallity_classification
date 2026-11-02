@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0focus_classifier\start.ps1"
if errorlevel 1 pause
