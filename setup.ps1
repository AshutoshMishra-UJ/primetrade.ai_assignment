param(
    [switch]$ForceRecreateVenv
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"

if ($ForceRecreateVenv -and (Test-Path $venvPath)) {
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    py -m venv .venv
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

Write-Host "Setup completed."
Write-Host "Run core analysis: $pythonExe src/analyze_sentiment_trader.py"
Write-Host "Run bonus model: $pythonExe src/bonus/bonus_modeling.py"
Write-Host "Run dashboard: $pythonExe -m streamlit run src/bonus/dashboard.py"
