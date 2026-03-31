#!/bin/bash
# Fish Speech S2 Pro — init script for RTX 3090 (Vast.ai / CUDA 12.x)
#
# Usage:
#   ./init.sh                — auto: install if fresh, pull latest files, start
#   ./init.sh install        — full first-time install (no start)
#   ./init.sh llamacpp       — build llama.cpp (run after install)
#   ./init.sh start          — (re)start all services
#   ./init.sh pull           — pull latest custom files from GitHub then restart

set -e

CMD="${1:-auto}"
REPO_RAW="https://raw.githubusercontent.com/Dankular/Voice.Clone/main"

# ─── INSTALL ──────────────────────────────────────────────────────────────────

install_base() {
    echo "=== Fish Speech S2 Pro Install (RTX 3090) ==="

    apt-get update -qq
    apt-get install -y -qq \
        git ffmpeg wget curl screen \
        python3.11-dev python3.11-venv \
        build-essential libsndfile1

    cd /root
    if [ ! -d fish-speech ]; then
        git clone https://github.com/fishaudio/fish-speech.git
    fi
    cd fish-speech

    python3.11 -m venv /root/fish-env
    source /root/fish-env/bin/activate

    pip install --upgrade pip wheel

    # PyTorch cu121 — Ampere/RTX 3090 compatible
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

    # Fish Speech deps (--no-deps preserves our cu121 torch)
    pip install -e ".[stable]" --no-deps

    # Remaining deps
    pip install \
        faster-whisper soundfile requests loguru pyrootutils huggingface_hub \
        natsort gradio transformers datasets lightning hydra-core einops librosa \
        rich grpcio uvicorn loralib resampy "einx[torch]==0.2.2" zstandard pydub \
        modelscope opencc-python-reimplemented silero-vad ormsgpack tiktoken \
        "pydantic==2.9.2" cachetools safetensors "kui>=1.6.0" \
        descript-audio-codec descript-audiotools fastapi \
        sentence-transformers

    # S2 Pro checkpoints (~11 GB)
    mkdir -p /root/fish-speech/checkpoints
    if [ ! -f /root/fish-speech/checkpoints/s2-pro/codec.pth ]; then
        echo "Downloading S2 Pro checkpoints (~11GB)..."
        /root/fish-env/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='fishaudio/s2-pro', local_dir='/root/fish-speech/checkpoints/s2-pro')
"
    fi

    mkdir -p /root/fish-speech/voices

    # cloudflared
    if ! command -v cloudflared &>/dev/null; then
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
            -O /usr/local/bin/cloudflared
        chmod +x /usr/local/bin/cloudflared
    fi

    # libcuda.so symlink (required for torch.compile / inductor)
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
        /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

    pull_files

    # ElevenLabs voice metadata (requires EL_API_KEY env var)
    if [ ! -f /root/fish-speech/el_voices.json ]; then
        if [ -n "$EL_API_KEY" ]; then
            echo "Fetching ElevenLabs voice metadata..."
            cd /root/fish-speech && /root/fish-env/bin/python3 fetch_el_metadata.py
        else
            echo "NOTE: EL_API_KEY not set — skipping voice metadata fetch."
            echo "      Set EL_API_KEY and run: cd /root/fish-speech && /root/fish-env/bin/python3 fetch_el_metadata.py"
        fi
    fi

    echo ""
    echo "=== Install complete ==="
    echo "Tag classifier uses sentence-transformers (auto-downloaded on first run)."
    echo "Optional: ./init.sh llamacpp   (builds llama.cpp for other LLM uses)"
    echo "Next: ./init.sh start"
}

install_llamacpp() {
    echo "=== llama.cpp install (CUDA sm86 / RTX 3090) ==="

    apt-get install -y -qq cmake

    if ! command -v nvcc &>/dev/null; then
        CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.replace('.', '-'))" 2>/dev/null || echo "12-1")
        if ! dpkg -l cuda-keyring &>/dev/null 2>&1; then
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
                -O /tmp/cuda-keyring.deb
            dpkg -i /tmp/cuda-keyring.deb
            apt-get update -qq
        fi
        apt-get install -y -qq cuda-nvcc-${CUDA_VER} libcublas-dev-${CUDA_VER}
    fi

    NVCC=$(which nvcc 2>/dev/null || find /usr/local/cuda*/bin -name nvcc 2>/dev/null | head -1)
    CUDA_ROOT=$(dirname "$(dirname "$NVCC")")
    echo "Using CUDA: $CUDA_ROOT"

    cd /root
    if [ ! -d llama.cpp ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    cd llama.cpp
    rm -rf build
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCUDAToolkit_ROOT="$CUDA_ROOT" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
    cmake --build build --config Release -j"$(nproc)" --target llama-server

    mkdir -p /root/models

    echo ""
    echo "=== llama.cpp built ==="
    echo "Download a model to /root/models/*.gguf, then: ./init.sh start"
    echo "Suggested: Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf"
}

# ─── PULL CUSTOM FILES ────────────────────────────────────────────────────────

pull_files() {
    echo "Pulling custom files from GitHub..."
    cd /root/fish-speech
    wget -q "$REPO_RAW/tools/webui/__init__.py"  -O tools/webui/__init__.py
    wget -q "$REPO_RAW/tools/webui/inference.py" -O tools/webui/inference.py
    wget -q "$REPO_RAW/tools/run_webui.py"        -O tools/run_webui.py
    wget -q "$REPO_RAW/tools/fish_api.py"         -O tools/fish_api.py
    wget -q "$REPO_RAW/tools/tag_classifier.py"   -O tools/tag_classifier.py
    wget -q "$REPO_RAW/fetch_el_metadata.py"      -O fetch_el_metadata.py
    echo "Done."
}

# ─── AMPERE PATCHES (applied on every start) ──────────────────────────────────

apply_patches() {
    echo "Applying Ampere patches..."

    INFERENCE=/root/fish-speech/fish_speech/models/text2semantic/inference.py
    LLAMA=/root/fish-speech/fish_speech/models/text2semantic/llama.py

    # 1. FlashAttention: replace SDPBackend.MATH with Flash+Efficient
    python3 -c "
path = '$INFERENCE'
with open(path) as f:
    src = f.read()
old = 'with sdpa_kernel(SDPBackend.MATH):'
new = 'with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):'
if old not in src:
    print('FlashAttention: already patched or not found, skipping')
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new))
    print('Patched: MATH -> FLASH_ATTENTION + EFFICIENT_ATTENTION')
"

    # 2. Disable coordinate_descent_tuning (causes VRAM OOM on 24GB)
    python3 -c "
path = '$INFERENCE'
with open(path) as f:
    src = f.read()
old = 'torch._inductor.config.coordinate_descent_tuning = True'
new = 'torch._inductor.config.coordinate_descent_tuning = False'
if old not in src:
    print('coordinate_descent_tuning: already patched or not found, skipping')
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new))
    print('Patched: coordinate_descent_tuning True -> False')
"

    # 3. Add dynamic=True to torch.compile (prevents per-shape recompilation)
    python3 -c "
path = '$INFERENCE'
with open(path) as f:
    src = f.read()
old = '''        decode_one_token = torch.compile(
            decode_one_token,
            backend=\"inductor\" if torch.cuda.is_available() else \"aot_eager\",
            mode=\"default\" if torch.cuda.is_available() else None,
            fullgraph=True,
        )'''
new = '''        decode_one_token = torch.compile(
            decode_one_token,
            backend=\"inductor\" if torch.cuda.is_available() else \"aot_eager\",
            mode=\"default\" if torch.cuda.is_available() else None,
            fullgraph=True,
            dynamic=True,
        )'''
if old not in src:
    print('dynamic=True: already patched or not found, skipping')
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new, 1))
    print('Patched: torch.compile added dynamic=True')
"

    # 4. Replace in-place masked ops in llama.py (fixes inductor kernel aliasing crash)
    python3 -c "
path = '$LLAMA'
with open(path) as f:
    src = f.read()
changes = 0
replacements = [
    ('vq_embeds_sum[~is_semantic] = 0',   'vq_embeds_sum = vq_embeds_sum * is_semantic.unsqueeze(-1)'),
    ('vq_embeds_sum[~vq_masks] = 0',      'vq_embeds_sum = vq_embeds_sum * vq_masks.unsqueeze(-1)'),
    ('x[audio_masks] = audio_embeds / math.sqrt(2)', 'x = torch.where(audio_masks.unsqueeze(-1), audio_embeds / math.sqrt(2), x)'),
    ('x[audio_masks] = audio_embeds',     'x = torch.where(audio_masks.unsqueeze(-1), audio_embeds, x)'),
]
for old, new in replacements:
    if old in src:
        src = src.replace(old, new)
        changes += 1
with open(path, 'w') as f:
    f.write(src)
print(f'llama.py: {changes}/4 in-place masked ops replaced with functional equivalents')
"
}

# ─── START SERVICES ───────────────────────────────────────────────────────────

start_services() {
    echo "=== Starting Fish Speech services ==="

    pkill -9 -f run_webui.py  2>/dev/null || true
    pkill -9 -f llama-server  2>/dev/null || true
    pkill -f   cloudflared    2>/dev/null || true
    sleep 1

    # llama.cpp (optional — no longer needed for tagging, kept for other uses)
    LLAMA_MODEL=$(ls /root/models/*.gguf 2>/dev/null | head -1)
    if [ -n "$LLAMA_MODEL" ] && [ -x /root/llama.cpp/build/bin/llama-server ]; then
        screen -dmS llamacpp bash -c "
            /root/llama.cpp/build/bin/llama-server \
                --model \"$LLAMA_MODEL\" \
                --host 0.0.0.0 --port 11434 \
                --ctx-size 2048 \
                --n-gpu-layers 0 \
                --threads 8 \
                >> /root/llamacpp.log 2>&1
        "
        echo "llama.cpp started: $(basename "$LLAMA_MODEL")"
    else
        echo "NOTE: llama.cpp not found or no GGUF model — tagging uses embedding classifier"
    fi

    apply_patches

    # Fish Speech WebUI
    truncate -s 0 /root/webui.log
    screen -dmS fishwebui bash -c "
        source /root/fish-env/bin/activate
        cd /root/fish-speech
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export LD_LIBRARY_PATH=/root/fish-env/lib/python3.11/site-packages/nvidia/cusparselt/lib:\$LD_LIBRARY_PATH
        exec python tools/run_webui.py \
            --llama-checkpoint-path checkpoints/s2-pro \
            --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
            --compile \
            >> /root/webui.log 2>&1
    "
    echo "fishwebui started"

    # cloudflared tunnel
    sleep 5
    truncate -s 0 /root/cloudflared.log
    setsid cloudflared tunnel --url http://localhost:7860 >> /root/cloudflared.log 2>&1 &

    echo ""
    echo "Services started."
    echo "Logs: tail -f /root/webui.log"
    echo "URL:  grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' /root/cloudflared.log | head -1"
}

# ─── DISPATCH ─────────────────────────────────────────────────────────────────

case "$CMD" in
    install)  install_base ;;
    llamacpp) install_llamacpp ;;
    start)    start_services ;;
    pull)     pull_files && start_services ;;
    auto)
        if [ ! -d /root/fish-speech ]; then
            install_base
        else
            pull_files
        fi
        start_services
        ;;
    *)
        echo "Usage: $0 [install|llamacpp|start|pull|auto]"
        echo ""
        echo "  install   full first-time install"
        echo "  llamacpp  build llama.cpp (run after install)"
        echo "  start     (re)start all services"
        echo "  pull      pull latest files from GitHub then restart"
        echo "  auto      install if fresh, pull latest, start (default)"
        exit 1
        ;;
esac
