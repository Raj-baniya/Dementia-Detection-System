@echo off
REM NeuroScan AI Launcher for Windows CMD
REM Run this batch file to start the Streamlit app

echo.
echo 🧠  Launching NeuroScan AI...
echo.

set VENV_PYTHON=%~dp0venv\Scripts\python.exe
set HARDCODED_PYTHON=f:\Mini-Project-ML\venv\Scripts\python.exe
set APP_PATH=%~dp0files\app.py

if exist "%VENV_PYTHON%" (
    echo ✓ Using workspace venv: %VENV_PYTHON%
    goto :RUN
)

if exist "%HARDCODED_PYTHON%" (
    set VENV_PYTHON=%HARDCODED_PYTHON%
    echo ✓ Using configured venv: %VENV_PYTHON%
    goto :RUN
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set VENV_PYTHON=python
    echo ✓ Using system Python
    goto :RUN
)

echo ❌ Python environment not found.
echo Create it with: python -m venv venv
pause
exit /b 1

:RUN
echo ✓ Launching app at http://localhost:8501
echo.

%VENV_PYTHON% -m streamlit run "%APP_PATH%"
pause
