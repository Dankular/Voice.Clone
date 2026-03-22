#!/bin/bash
source /root/fish-env/bin/activate
cd /root/fish-speech
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/fish-env/lib/python3.11/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
exec python tools/run_webui.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --compile \
    >> /root/webui.log 2>&1
