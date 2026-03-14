#!/bin/bash
set -e
# Install llama.cpp with CUDA support + download a GGUF model
# Run AFTER fish_install_3090.sh

echo "=== llama.cpp install ==="

apt-get install -y -qq cmake

# --- Install CUDA toolkit (nvcc) if not present ---
if ! command -v nvcc &>/dev/null; then
    echo "nvcc not found — installing CUDA toolkit..."
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.replace('.', '-'))" 2>/dev/null || echo "12-1")
    # Add NVIDIA apt repo if needed
    if ! dpkg -l cuda-keyring &>/dev/null 2>&1; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq
    fi
    apt-get install -y -qq cuda-nvcc-${CUDA_VER} libcublas-dev-${CUDA_VER}
fi

CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
echo "CUDA toolkit: $CUDA_ROOT"

cd /root
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Build with CUDA (sm86 = RTX 3090 Ampere)
rm -rf build
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCUDAToolkit_ROOT="$CUDA_ROOT"
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
