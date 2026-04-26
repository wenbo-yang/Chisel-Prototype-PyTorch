# scripts/setup.ps1 — Windows environment setup
# Usage: .\scripts\setup.ps1
#
# Requirements: Python 3.11 must already be installed.
# Download from https://www.python.org/downloads/

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Python check ----------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.11 from https://www.python.org/downloads/ and re-run."
    exit 1
}

$pyVersion = python --version
Write-Host "Using $pyVersion"

# --- Python dependencies ---------------------------------------------
Write-Host "`nInstalling dependencies..."
python -m pip install --upgrade pip

python -m pip install pytest
python -m pip install torch torchvision torchaudio
python -m pip install fastapi "uvicorn[standard]" cryptography
python -m pip install opencv-python
python -m pip install numpy
python -m pip install matplotlib
python -m pip install ultralytics

# --- Download models -------------------------------------------------
Write-Host "`nDownloading models..."
$modelsDir = Join-Path $PSScriptRoot "..\models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

python - @"
from ultralytics import YOLO
import shutil, pathlib, os

dest = pathlib.Path(r'$($modelsDir.Replace("\","\\"))\yolov8n-seg.pt')
if dest.exists():
    print(f'Model already exists: {dest}')
else:
    model = YOLO('yolov8n-seg.pt')   # downloads to cwd
    src = pathlib.Path('yolov8n-seg.pt')
    if src.exists():
        shutil.move(str(src), str(dest))
    print(f'Model saved to {dest}')
"@

Write-Host "`nSetup complete. Run tests with: pytest test/ -v"
