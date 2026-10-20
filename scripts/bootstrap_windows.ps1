
$ErrorActionPreference = "Stop"

Write-Host "=== Adaptive Latent MAS - Windows bootstrap ===" -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11+ first."
}

$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run UI with: streamlit run ui/app.py"
