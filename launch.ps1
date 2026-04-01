# NeuroScan AI Launcher
# Run this script to start the Streamlit app with the correct Python environment
# Usage: powershell -ExecutionPolicy Bypass -File .\launch.ps1

Write-Host "🧠 Launching NeuroScan AI..." -ForegroundColor Cyan

$localVenv = "$PSScriptRoot\venv\Scripts\python.exe"
$hardcodedVenv = "f:\Mini-Project-ML\venv\Scripts\python.exe"
$appPath = "$PSScriptRoot\files\app.py"

if (Test-Path $localVenv) {
    $venvPython = $localVenv
    Write-Host "✓ Using workspace venv Python: $venvPython" -ForegroundColor Green
} elseif (Test-Path $hardcodedVenv) {
    $venvPython = $hardcodedVenv
    Write-Host "✓ Using configured venv Python: $venvPython" -ForegroundColor Green
} else {
    $sysPython = (Get-Command python.exe -ErrorAction SilentlyContinue)
    if ($sysPython) {
        $venvPython = $sysPython.Source
        Write-Host "✓ Using system Python: $venvPython" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Python environment not found." -ForegroundColor Red
        Write-Host "Create a venv with: python -m venv venv" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $appPath)) {
    $appPath = "files/app.py"
}

Write-Host "✓ Launching app at http://localhost:8501" -ForegroundColor Green
Write-Host ""

& $venvPython -m streamlit run $appPath
