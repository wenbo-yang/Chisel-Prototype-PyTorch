brew install pyenv
pyenv install 3.9
pyenv global 3.9
pip install pytest
pip install torch torchvision torchaudio
pip install fastapi
pip install fastapi uvicorn[standard] cryptography
pip install opencv

# # add this to .bashrc .bash_profile .zshrc
# if command -v pyenv 1>/dev/null 2>&1; then
#   eval "$(pyenv init -)"
# fi
