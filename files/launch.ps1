# NeuroScan AI Launch Script
# This script activates the venv and runs the Streamlit app

$VenvPython = "$(Split-Path -Parent $PSScriptRoot)\venv\Scripts\python.exe"

Write-Host "🧠 NeuroScan AI — Launching..." -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "❌ Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please ensure venv is installed at f:\Mini-Project-ML\venv" -ForegroundColor Yellow
    exit 1
}

# Launch Streamlit
Write-Host "📍 Using Python: $VenvPython" -ForegroundColor Gray
Write-Host "🌐 Opening http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

& $VenvPython -m streamlit run app.py
