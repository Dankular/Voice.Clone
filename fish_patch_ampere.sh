#!/bin/bash
# Patch Fish Speech for Ampere GPUs (RTX 3090, A100, etc.)
# Must run after fish-speech is cloned to /root/fish-speech

INFERENCE=/root/fish-speech/fish_speech/models/text2semantic/inference.py

# Replace SDPBackend.MATH with Flash+Efficient in AR decode loop
python3 -c "
path = '$INFERENCE'
with open(path) as f:
    src = f.read()

old = 'with sdpa_kernel(SDPBackend.MATH):'
new = 'with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):'

if old not in src:
    print('Already patched or not found, skipping')
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new))
    print('Patched: MATH -> FLASH_ATTENTION + EFFICIENT_ATTENTION fallback')
"
