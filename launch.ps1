# NeuroScan AI Launcher
# Run this script to start the Streamlit app with the correct Python environment

Write-Host "🧠 Launching NeuroScan AI..." -ForegroundColor Cyan

$venvPython = "f:\Mini-Project-ML\venv\Scripts\python.exe"
$appPath = "f:\Mini-Project-ML\files\app.py"

if (-not (Test-Path $venvPython)) {
    Write-Host "❌ Virtual environment not found at $venvPython" -ForegroundColor Red
    Write-Host "Create it with: python -m venv f:\Mini-Project-ML\venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Using Python from venv" -ForegroundColor Green
Write-Host "✓ Launching app at http://localhost:8501" -ForegroundColor Green
Write-Host ""

& $venvPython -m streamlit run $appPath
