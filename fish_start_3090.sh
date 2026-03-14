#!/bin/bash
# Start Fish Speech WebUI + llama.cpp (CPU) + cloudflared

# Kill any existing sessions cleanly
pkill -9 -f run_webui.py 2>/dev/null || true
pkill -9 -f llama-server 2>/dev/null || true
sleep 1

# --- llama.cpp server on CPU (keeps GPU free for Fish Speech + Whisper) ---
# Requires a GGUF model in /root/models/ — see fish_install_llamacpp.sh
LLAMA_MODEL=$(ls /root/models/*.gguf 2>/dev/null | head -1)
if [ -n "$LLAMA_MODEL" ]; then
    screen -dmS llamacpp bash -c "
        /root/llama.cpp/build/bin/llama-server \
            --model \"$LLAMA_MODEL\" \
            --host 0.0.0.0 --port 11434 \
            --ctx-size 2048 \
            --n-gpu-layers 99 \
            --threads $(nproc) \
            >> /root/llamacpp.log 2>&1
    "
    echo "llama.cpp started with: $LLAMA_MODEL"
else
    echo "WARNING: No GGUF model found in /root/models/ — enhance feature will fail"
fi

# --- Fish Speech WebUI ---
truncate -s 0 /root/webui.log
export TORCHINDUCTOR_CACHE_DIR=/root/.inductor_cache
# LIBRARY_PATH needed so inductor/triton gcc can link libcuda.so
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
screen -dmS fishwebui bash -c '
    source /root/fish-env/bin/activate
    cd /root/fish-speech
    export TORCHINDUCTOR_CACHE_DIR=/root/.inductor_cache
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
    python tools/run_webui.py \
        --llama-checkpoint-path checkpoints/s2-pro \
        --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
        >> /root/webui.log 2>&1
'

# --- cloudflared tunnel ---
sleep 5
truncate -s 0 /root/cloudflared.log
screen -dmS cloudflared bash -c 'cloudflared tunnel --url http://localhost:7860 >> /root/cloudflared.log 2>&1'

echo "Started: fishwebui + llamacpp (CPU) + cloudflared"
echo "Logs:    tail -f /root/webui.log"
echo "URL:     grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /root/cloudflared.log | head -1"
