$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run scripts\setup_local.ps1 first."
}

& ".\.venv\Scripts\python.exe" -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
