#!/bin/bash
# Start Fish Speech WebUI + llama.cpp (CPU) + cloudflared

# Kill any existing sessions cleanly
pkill -9 -f run_webui.py 2>/dev/null || true
pkill -9 -f llama-server 2>/dev/null || true
sleep 1

# --- llama.cpp server on CPU (keep all VRAM for Fish Speech allocator headroom) ---
# Requires a GGUF model in /root/models/ — see fish_install_llamacpp.sh
LLAMA_MODEL=$(ls /root/models/*.gguf 2>/dev/null | head -1)
if [ -n "$LLAMA_MODEL" ]; then
    screen -dmS llamacpp bash -c "
        /root/llama.cpp/build/bin/llama-server \
            --model \"$LLAMA_MODEL\" \
            --host 0.0.0.0 --port 11434 \
            --ctx-size 2048 \
            --n-gpu-layers 0 \
            --threads 8 \
            >> /root/llamacpp.log 2>&1
    "
    echo "llama.cpp started with: $LLAMA_MODEL"
else
    echo "WARNING: No GGUF model found in /root/models/ — enhance feature will fail"
fi

# --- Apply Ampere-specific patches to fish-speech ---
bash /root/fish_patch_ampere.sh

# --- Fish Speech WebUI ---
truncate -s 0 /root/webui.log
# Strip Windows CRLF line endings that break bash scripts uploaded from Windows
sed -i 's/\r//' /root/fishwebui_launch.sh
screen -dmS fishwebui /root/fishwebui_launch.sh

# --- cloudflared tunnel ---
sleep 5
pkill -f cloudflared 2>/dev/null || true
sleep 1
truncate -s 0 /root/cloudflared.log
setsid cloudflared tunnel --url http://localhost:7860 >> /root/cloudflared.log 2>&1 &

echo "Started: fishwebui + llamacpp (GPU) + cloudflared"
echo "Logs:    tail -f /root/webui.log"
echo "URL:     grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /root/cloudflared.log | head -1"
