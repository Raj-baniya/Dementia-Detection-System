@echo off
REM NeuroScan AI Launcher for Windows CMD
REM Run this batch file to start the Streamlit app

echo.
echo 🧠  Launching NeuroScan AI...
echo.

set VENV_PYTHON=f:\Mini-Project-ML\venv\Scripts\python.exe
set APP_PATH=f:\Mini-Project-ML\files\app.py

if not exist %VENV_PYTHON% (
    echo ❌ Virtual environment not found at %VENV_PYTHON%
    echo Create it with: python -m venv f:\Mini-Project-ML\venv
    pause
    exit /b 1
)

echo ✓ Using Python from venv
echo ✓ Launching app at http://localhost:8501
echo.

%VENV_PYTHON% -m streamlit run %APP_PATH%
pause
