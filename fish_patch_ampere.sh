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

# Replace torch.compile inductor (crashes on triton 3.1) with cudagraphs backend
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
            backend=\"cudagraphs\",
        )'''

if old not in src:
    print('compile backend already patched or not found, skipping')
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new, 1))
    print('Patched: inductor/fullgraph -> cudagraphs (no triton)')
"
