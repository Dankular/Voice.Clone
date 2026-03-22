# Voice Clone

A production-ready voice cloning API and web UI built on top of [Fish Speech S2 Pro](https://huggingface.co/fishaudio/s2-pro), with a 10,000+ voice gallery sourced from ElevenLabs shared voices.

---

## Features

- **10,790+ voice gallery** — all ElevenLabs public shared voices, searchable and filterable by language, gender, age, accent
- **Zero storage** — voice metadata only (~2MB JSON), previews fetched on-demand
- **Auto-transcription** — Whisper (CPU) automatically transcribes reference audio, no manual input needed
- **REST API** — `/gallery` and `/generate` endpoints for programmatic access
- **Gradio WebUI** — browser-based interface for testing
- **Voice library** — save and manage custom voices
- **LRU cache** — repeated calls to the same voice skip download + transcription
- **Optimised inference** — `torch.compile`, TF32 tensor cores, cuDNN benchmark
- **LLM prosody tagger** — llama.cpp server (CPU) with any GGUF model annotates text with `[pause]`, `[emphasis]`, etc. before synthesis
- **Auto max tokens** — output token budget estimated from input character count automatically

---

## Setup

### Requirements

- Python 3.11
- CUDA GPU with 24GB VRAM (tested on RTX 3090 — Ampere sm86, CUDA 12.1)
- 32GB+ system RAM
- ~80GB disk:
  - Fish Speech S2 Pro: ~11GB (safetensors + codec)
  - GGUF LLM (`Qwen2.5-1.5B-Instruct-abliterated Q4_K_M`): ~940MB, ~1GB disk
  - llama.cpp build + Python env: ~10GB
  - Voices/cache: varies
- CUDA driver 535+ (CUDA 12.2 runtime, toolkit 12.1)

### Install

Everything is handled by `init.sh`:

```bash
# 1. Full install (deps, Fish Speech, model download, cloudflared, custom files)
bash init.sh install

# 2. Build llama.cpp with CUDA (for the prosody tagger LLM)
bash init.sh llamacpp

# 3. Download a GGUF model
mkdir -p /root/models
wget -P /root/models/ https://huggingface.co/mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf

# 4. Launch
bash init.sh start
```

**Subsequent restarts** — pull latest files from GitHub and restart everything:

```bash
bash init.sh pull
# or just: bash init.sh   (auto mode — pulls if already installed, then starts)
```

#### `init.sh` subcommands

| Command | Description |
|---------|-------------|
| `install` | Full first-time install (deps, env, checkpoints, cloudflared, custom files) |
| `llamacpp` | Build llama.cpp with CUDA support |
| `start` | Kill and restart all services (applies Ampere patches, starts webui + llama + cloudflared) |
| `pull` | Pull latest custom files from GitHub then restart |
| `auto` *(default)* | Install if not installed, pull latest files, then start |

**Key Python deps** (installed by `init.sh install`):
- `torch==2.5.1+cu121` / `torchaudio` — pinned for CUDA 12.1 / RTX 3090
- `faster-whisper` — CPU transcription (preserves all VRAM for Fish Speech)
- `descript-audio-codec`, `descript-audiotools` — DAC codec
- `gradio>5.0`, `uvicorn`, `fastapi` — WebUI + REST API
- `huggingface_hub` — model download

### Fetch ElevenLabs voice metadata

Required once per instance. Set your ElevenLabs API key and run:

```bash
export EL_API_KEY="your_key_here"
cd /root/fish-speech
python fetch_el_metadata.py
# Saves el_voices.json (~2MB, 10,790 voices)
```

`init.sh install` will do this automatically if `EL_API_KEY` is set in the environment.

### Launch

`init.sh start` handles this, but the underlying command is:

```bash
cd /root/fish-speech
source /root/fish-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tools/run_webui.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --compile
```

Gradio UI: `http://localhost:7860`
API docs: `http://localhost:7860/docs`

---

## API

### `GET /gallery`

Search and filter voices from the 10,790-voice library.

**Query params:**

| Param | Description | Example |
|-------|-------------|---------|
| `search` | Name or description keyword | `husky` |
| `language` | Language code | `en` |
| `gender` | `male` / `female` | `male` |
| `age` | `young` / `middle_aged` / `old` | `middle_aged` |
| `accent` | Accent filter | `british` |
| `source` | `all` / `elevenlabs` / `saved` | `elevenlabs` |
| `limit` | Results per page (max 500) | `20` |
| `offset` | Pagination offset | `0` |

**Example:**
```bash
curl "http://localhost:7860/gallery?search=james&language=en&gender=male&limit=5"
```

**Response:**
```json
{
  "total": 82,
  "offset": 0,
  "limit": 5,
  "voices": [
    {
      "voice_id": "rm143ZlE6RfHtN634wZ8",
      "name": "Jay - British, Well-spoken & Husky",
      "description": "A British, calm and well spoken London accent...",
      "gender": "male",
      "age": "middle_aged",
      "accent": "british",
      "language": "en",
      "source": "elevenlabs",
      "preview_url": "https://..."
    }
  ]
}
```

---

### `POST /generate`

Generate audio from a voice ID and text prompt. Reference audio and transcription are handled automatically.

**Request body:**

```json
{
  "voice_id": "rm143ZlE6RfHtN634wZ8",
  "text": "Your text to synthesise goes here.",
  "temperature": 0.8,
  "top_p": 0.8,
  "repetition_penalty": 1.1,
  "chunk_length": 300,
  "max_new_tokens": 0,
  "seed": 0
}
```

**Response:** `audio/wav` binary stream with header `Content-Disposition: attachment; filename="output.wav"`

**Example:**
```bash
# Step 1 — find a voice
curl "http://localhost:7860/gallery?search=husky&language=en&gender=male&limit=3"

# Step 2 — generate with that voice_id
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"voice_id": "rm143ZlE6RfHtN634wZ8", "text": "Hello, this is a test."}' \
  --output output.wav
```

---

## Changes from upstream fish-speech

### `tools/run_webui.py`
- Replaced `app.launch()` with FastAPI + `gr.mount_gradio_app` so REST API and Gradio share one port
- Added `--compile` flag; VQ-GAN decoder is also compiled when `--compile` is set
- Enabled `torch.backends.cudnn.benchmark` and TF32 tensor cores
- Sets `TORCHINDUCTOR_CACHE_DIR` to persist compiled kernels across restarts
- Calls `init_api(inference_engine)` to share the loaded model with the API

### `tools/fish_api.py` *(new file)*
- `GET /gallery` — searches `el_voices.json` + saved voices with filtering and pagination
- `POST /generate` — resolves voice_id → fetches preview on-demand → Whisper transcription → prosody tagging → TTS inference → returns WAV
- LRU cache (500 voices) for audio bytes + transcription to avoid redundant downloads
- Prosody tagger calls local llama.cpp server to annotate text before synthesis
- Whisper runs on CPU (`int8`) to keep all VRAM available for Fish Speech

### `tools/webui/__init__.py`
- Added **Voice Gallery** tab: searchable/filterable dropdown over 10,790 ElevenLabs voices, loads preview on-demand
- Added **My Saved Voices** tab: save/load/delete custom voices with name, description, publish flag
- Auto-transcription: `reference_audio.change` → Whisper → populates reference text field automatically
- `app.load()` refresh on page load so saved voices always reflect current state

### `tools/webui/variables.py`
- Updated header to say **Fish Speech S2 Pro** with correct HuggingFace link

### `fetch_el_metadata.py` *(new file)*
- Paginates ElevenLabs `/v1/shared-voices` API and saves slim metadata JSON
- Reads API key from `EL_API_KEY` environment variable
- No audio downloaded — purely metadata (name, description, gender, age, accent, language, preview_url)

### `init.sh` *(replaces individual .sh scripts)*
- Consolidates `fish_install_3090.sh`, `fish_install_llamacpp.sh`, `fish_patch_ampere.sh`, `fish_start_3090.sh`, `fishwebui_launch.sh`, `start_cloudflared.sh`
- Ampere GPU patches applied on every start (FlashAttention, `dynamic=True` compile, in-place op fixes)
- `pull` subcommand fetches latest custom files from GitHub before restarting

---

## Voice Library

Users can save any voice (from gallery or uploaded audio) to a local library:

- Saved to `voices/{uuid}/` with `audio.{ext}` + `meta.json`
- Accessible via the **My Saved Voices** tab in the WebUI or `source=saved` in `/gallery`
- `published` flag controls visibility

---

## ElevenLabs Integration

Voice metadata is fetched once via `fetch_el_metadata.py` and stored as `el_voices.json`. No ElevenLabs API key is needed at inference time — preview audio is fetched directly from public URLs embedded in the metadata.

The API key is only needed for the one-time metadata fetch and is read from the `EL_API_KEY` environment variable (never hardcoded).

The LRU cache ensures each unique voice is only downloaded and transcribed once per server session.
