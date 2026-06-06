#!/usr/bin/env bash
# scripts/setup.sh — macOS/Linux environment setup
# Usage: bash scripts/setup.sh

set -e

# --- pyenv + Python --------------------------------------------------
brew install pyenv
pyenv install 3.11
pyenv global 3.11

# Add pyenv to shell (append to .zshrc and .bashrc if not already present)
for RC in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if ! grep -q 'pyenv init' "$RC" 2>/dev/null; then
    echo '' >> "$RC"
    echo '# pyenv' >> "$RC"
    echo 'if command -v pyenv 1>/dev/null 2>&1; then' >> "$RC"
    echo '  eval "$(pyenv init -)"' >> "$RC"
    echo 'fi' >> "$RC"
  fi
done

eval "$(pyenv init -)"

# --- Python dependencies ---------------------------------------------
pip install --upgrade pip

pip install pytest
pip install torch torchvision torchaudio
pip install fastapi uvicorn[standard] cryptography
pip install opencv-python
pip install numpy
pip install matplotlib
pip install ultralytics

# --- Download models -------------------------------------------------
MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

python - <<'EOF'
from ultralytics import YOLO
import shutil, pathlib

# Download YOLOv8 Nano segmentation model
dest_nano = pathlib.Path('models/yolov8n-seg.pt')
if dest_nano.exists():
    print(f'✓ Nano model already exists: {dest_nano}')
else:
    print('Downloading YOLOv8 Nano segmentation model...')
    model = YOLO('yolov8n-seg.pt')   # downloads to cwd if not present
    if pathlib.Path('yolov8n-seg.pt').exists():
        shutil.move('yolov8n-seg.pt', dest_nano)
    print(f'✓ Nano model saved to {dest_nano}')

# Download YOLOv8 Extra-Large segmentation model
dest_large = pathlib.Path('models/yolov8x-seg.pt')
if dest_large.exists():
    print(f'✓ Extra-Large model already exists: {dest_large}')
else:
    print('Downloading YOLOv8 Extra-Large segmentation model (137 MB)...')
    model = YOLO('yolov8x-seg.pt')   # downloads to cwd if not present
    if pathlib.Path('yolov8x-seg.pt').exists():
        shutil.move('yolov8x-seg.pt', dest_large)
    print(f'✓ Extra-Large model saved to {dest_large}')
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Models available:"
echo "  - YOLOv8 Nano (7 MB):          models/yolov8n-seg.pt"
echo "  - YOLOv8 Extra-Large (137 MB):  models/yolov8x-seg.pt"
echo ""
echo "Run tests with: pytest test/ -v"
