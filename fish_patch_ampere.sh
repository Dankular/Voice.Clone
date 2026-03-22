#!/bin/bash
# Patch Fish Speech for Ampere GPUs (RTX 3090, A100, etc.)
# Must run after fish-speech is cloned to /root/fish-speech
# Applies on every restart via fish_start_3090.sh

INFERENCE=/root/fish-speech/fish_speech/models/text2semantic/inference.py
LLAMA=/root/fish-speech/fish_speech/models/text2semantic/llama.py

# 1. FlashAttention: replace SDPBackend.MATH with Flash+Efficient in AR decode loop
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

# 2. Disable coordinate_descent_tuning (causes VRAM OOM during triton autotuning on 24GB)
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

# 4. Replace in-place masked assignments with functional equivalents (causes inductor kernel aliasing crash)
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
