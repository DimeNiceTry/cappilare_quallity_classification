@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" goto install
py -3.10 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" goto install
python -m venv .venv
if errorlevel 1 goto fail
:install
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] in [(3,10),(3,11)], 'Install Python 3.10 or 3.11'"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ipykernel install --prefix .venv --name capillary-focus4 --display-name "Capillary: 4 features"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" predict.py --check
if errorlevel 1 goto fail
echo Ready. Run start_notebooks.cmd or use predict.py.
exit /b 0
:fail
echo Setup failed. See the error above.
pause
exit /b 1
