#!/bin/bash
set -e
echo "=== Fish Speech S2 Pro Install (RTX 3090) ==="

# --- System deps ---
apt-get update -qq
apt-get install -y -qq git ffmpeg wget curl screen python3.11-dev python3.11-venv build-essential libsndfile1

# --- Fish Speech repo ---
cd /root
if [ ! -d fish-speech ]; then
    git clone https://github.com/fishaudio/fish-speech.git
fi
cd fish-speech

# --- Python venv ---
python3.11 -m venv /root/fish-env
source /root/fish-env/bin/activate

pip install --upgrade pip wheel

# PyTorch cu121 (CUDA 12.x — Ampere/3090 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Fish Speech deps (--no-deps to preserve our cu121 torch)
pip install -e ".[stable]" --no-deps
# All project deps (from pyproject.toml) excluding torch/torchaudio already installed
pip install \
    faster-whisper soundfile requests loguru pyrootutils huggingface_hub \
    natsort gradio transformers datasets lightning hydra-core einops librosa \
    rich grpcio uvicorn loralib resampy "einx[torch]==0.2.2" zstandard pydub \
    modelscope opencc-python-reimplemented silero-vad ormsgpack tiktoken \
    "pydantic==2.9.2" cachetools safetensors "kui>=1.6.0" \
    descript-audio-codec descript-audiotools

# --- Model checkpoints (fishaudio/s2-pro) ---
mkdir -p /root/fish-speech/checkpoints
cd /root/fish-speech/checkpoints

if [ ! -f s2-pro/codec.pth ]; then
    echo "Downloading S2 Pro checkpoints (~11GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='fishaudio/s2-pro', local_dir='s2-pro')
"
fi

# --- Voices dir ---
mkdir -p /root/fish-speech/voices

# --- ElevenLabs voice metadata (10,790 voices for gallery) ---
cd /root/fish-speech
if [ ! -f el_voices.json ]; then
    echo "Fetching ElevenLabs voice metadata..."
    /root/fish-env/bin/python3 fetch_el_metadata.py
fi

# --- cloudflared ---
if ! command -v cloudflared &>/dev/null; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# --- llama.cpp (CPU inference for LLM annotation/enhance) ---
# See fish_install_llamacpp.sh to install after this script completes

# --- Pull our modified webui files from GitHub ---
cd /root/fish-speech
REPO_RAW="https://raw.githubusercontent.com/Dankular/Voice.Clone/main"
wget -q "$REPO_RAW/tools/webui/__init__.py" -O tools/webui/__init__.py
wget -q "$REPO_RAW/tools/run_webui.py" -O tools/run_webui.py
wget -q "$REPO_RAW/fetch_el_metadata.py" -O fetch_el_metadata.py

echo "=== Install complete ==="
echo "Run: /root/fish_start_3090.sh"
