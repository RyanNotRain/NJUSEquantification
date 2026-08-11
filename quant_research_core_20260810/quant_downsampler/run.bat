@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 scripts\%*
    exit /b %errorlevel%
)

where python >nul 2>nul
if not errorlevel 1 (
    python scripts\%*
    exit /b %errorlevel%
)

echo Python not found. Install Python 3.10+ and run: pip install -r requirements.txt
exit /b 1
