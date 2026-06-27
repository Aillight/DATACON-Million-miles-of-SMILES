$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Test-PythonModule {
    param([string]$ModuleName)

    & python -m $ModuleName --version *> $null
    return $LASTEXITCODE -eq 0
}

$UvExecutable = (Get-Command uv -ErrorAction SilentlyContinue).Source
$UsePythonModuleUv = $false

if (-not $UvExecutable) {
    $UsePythonModuleUv = Test-PythonModule -ModuleName "uv"
}

if ($UvExecutable -or $UsePythonModuleUv) {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        if ($UvExecutable) {
            & $UvExecutable venv --python 3.12 .venv
        } else {
            & python -m uv venv --python 3.12 .venv
        }
    }

    if ($UvExecutable) {
        & $UvExecutable pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    } else {
        & python -m uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    }
} else {
    Write-Warning "uv is not installed; falling back to python -m venv + pip. RDKit may not install on Python 3.14."
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        python -m venv .venv
    }

    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
}

Write-Output "Local environment is ready. Activate it with: .venv\Scripts\Activate.ps1"
