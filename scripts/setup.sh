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
model = YOLO('yolov8n-seg.pt')   # downloads to cwd if not present
dest = pathlib.Path('models/yolov8n-seg.pt')
if not dest.exists():
    shutil.move('yolov8n-seg.pt', dest)
print(f'Model saved to {dest}')
EOF

echo ""
echo "Setup complete. Run tests with: pytest test/ -v"
