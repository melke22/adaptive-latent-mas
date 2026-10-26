$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
        throw "Virtual environment not found. Run scripts\bootstrap_windows.ps1 first."
    }
    & .\.venv\Scripts\python.exe -m streamlit run ui/app.py
    if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
