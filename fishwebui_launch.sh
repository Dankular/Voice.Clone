#!/bin/bash
source /root/fish-env/bin/activate
cd /root/fish-speech
export TORCHINDUCTOR_CACHE_DIR=/root/.inductor_cache
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force inductor to compile in-process (single thread) instead of subprocess pool
# Fixes BrokenProcessPool crash with torch 2.5.x + triton 3.x
export TORCHINDUCTOR_COMPILE_THREADS=1
exec python tools/run_webui.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --half \
    --compile \
    >> /root/webui.log 2>&1
