#!/bin/bash
set -e
# Install llama.cpp with CUDA support + download a GGUF model
# Run AFTER fish_install_3090.sh

echo "=== llama.cpp install ==="

apt-get install -y -qq cmake

cd /root
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Build with CUDA (for partial GPU offload) — set -DGGML_CUDA=OFF for CPU-only
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j$(nproc) --target llama-server

mkdir -p /root/models

echo ""
echo "=== llama.cpp built ==="
echo "Download a model, e.g.:"
echo "  wget -P /root/models/ https://huggingface.co/<user>/<repo>/resolve/main/<model>.gguf"
echo ""
echo "Then run: /root/fish_start_3090.sh"
echo ""
echo "The server listens on port 11434 (OpenAI-compatible at /v1)"
echo "CPU-only: set --n-gpu-layers 0 in fish_start_3090.sh"
echo "GPU offload: set --n-gpu-layers N (limited by remaining VRAM after Fish Speech)"
