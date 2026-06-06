# scripts/setup.ps1 — Windows environment setup
# Usage: .\scripts\setup.ps1
#
# Installs Python 3.11 (via winget if not present), dependencies, and downloads models.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REQUIRED_PYTHON_VERSION = "3.11"

# --- Python check & install ------------------------------------------
function Get-PythonVersion {
    try {
        $v = python --version 2>&1
        if ($v -match "Python (\d+\.\d+)") { return $Matches[1] }
    } catch {}
    return $null
}

$installedVersion = Get-PythonVersion

if (-not $installedVersion) {
    Write-Host "Python not found. Installing Python $REQUIRED_PYTHON_VERSION via winget..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error "winget not available. Install Python $REQUIRED_PYTHON_VERSION manually from https://www.python.org/downloads/ and re-run."
        exit 1
    }
    winget install --id Python.Python.3.11 --source winget --accept-source-agreements --accept-package-agreements
    # Refresh PATH so python is available in this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    $installedVersion = Get-PythonVersion
    if (-not $installedVersion) {
        Write-Error "Python installation succeeded but 'python' is still not on PATH. Open a new terminal and re-run."
        exit 1
    }
}

$major, $minor = $installedVersion.Split(".")
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
    Write-Warning "Python $installedVersion found but $REQUIRED_PYTHON_VERSION+ is recommended."
    Write-Warning "Install Python $REQUIRED_PYTHON_VERSION from https://www.python.org/downloads/ for best compatibility."
}

Write-Host "Using Python $installedVersion"

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

# Download YOLOv8 Nano segmentation model
dest_nano = pathlib.Path(r'$($modelsDir.Replace("\","\\"))\yolov8n-seg.pt')
if dest_nano.exists():
    print(f'✓ Nano model already exists: {dest_nano}')
else:
    print('Downloading YOLOv8 Nano segmentation model...')
    model = YOLO('yolov8n-seg.pt')   # downloads to cwd
    src = pathlib.Path('yolov8n-seg.pt')
    if src.exists():
        shutil.move(str(src), str(dest_nano))
    print(f'✓ Nano model saved to {dest_nano}')

# Download YOLOv8 Extra-Large segmentation model
dest_large = pathlib.Path(r'$($modelsDir.Replace("\","\\"))\yolov8x-seg.pt')
if dest_large.exists():
    print(f'✓ Extra-Large model already exists: {dest_large}')
else:
    print('Downloading YOLOv8 Extra-Large segmentation model (137 MB)...')
    model = YOLO('yolov8x-seg.pt')   # downloads to cwd
    src = pathlib.Path('yolov8x-seg.pt')
    if src.exists():
        shutil.move(str(src), str(dest_large))
    print(f'✓ Extra-Large model saved to {dest_large}')
"@

Write-Host "`nSetup complete!`n"
Write-Host "Models available:"
Write-Host "  - YOLOv8 Nano (7 MB):        models/yolov8n-seg.pt"
Write-Host "  - YOLOv8 Extra-Large (137 MB): models/yolov8x-seg.pt"
Write-Host "`nRun tests with: pytest test/ -v"
