$ErrorActionPreference = "Stop"

Write-Host "=== Adaptive Latent MAS - Windows bootstrap ===" -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11+ first."
}

$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run UI with: .\scripts\run_ui.ps1 (from the project folder)"
