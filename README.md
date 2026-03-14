# Voice Clone

A production-ready voice cloning API and web UI built on top of [Fish Speech S2 Pro](https://huggingface.co/fishaudio/s2-pro), with a 10,000+ voice gallery sourced from ElevenLabs shared voices.

---

## Features

- **10,790+ voice gallery** — all ElevenLabs public shared voices, searchable and filterable by language, gender, age, accent
- **Zero storage** — voice metadata only (~2MB JSON), previews fetched on-demand
- **Auto-transcription** — Whisper (GPU) automatically transcribes reference audio, no manual input needed
- **REST API** — `/gallery` and `/generate` endpoints for programmatic access
- **Gradio WebUI** — browser-based interface for testing
- **Voice library** — save and manage custom voices
- **LRU cache** — repeated calls to the same voice skip download + transcription
- **Optimised inference** — `torch.compile`, TF32 tensor cores, cuDNN benchmark
- **LLM enhance** — llama.cpp server (CPU) with any GGUF model for prosody annotation
- **Auto max tokens** — output token budget estimated from input character count automatically

---

## Setup

### Requirements

- Python 3.11
- CUDA GPU with 24GB VRAM (tested on RTX 3090 — Ampere sm86, CUDA 12.1)
- 32GB+ system RAM
- ~80GB disk:
  - Fish Speech S2 Pro: ~11GB (safetensors + codec)
  - GGUF LLM (`Qwen2.5-1.5B-Instruct-abliterated Q4_K_M`): ~940MB VRAM, ~1GB disk
  - llama.cpp build + Python env: ~10GB
  - Voices/cache: varies
- CUDA driver 535+ (CUDA 12.2 runtime, toolkit 12.1)

### Install

Use the provided install scripts — they handle everything automatically:

```bash
# 1. Full Fish Speech install (deps, model download, EL voices, cloudflared)
bash fish_install_3090.sh

# 2. Build llama.cpp with CUDA (for enhance input LLM)
bash fish_install_llamacpp.sh

# 3. Download a GGUF model
mkdir -p /root/models
wget -P /root/models/ https://huggingface.co/mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf

# 4. Launch
bash fish_start_3090.sh
```

**Key Python deps** (installed by `fish_install_3090.sh`):
- `torch==2.5.1+cu121` / `torchaudio` — pinned for CUDA 12.1 / RTX 3090
- `faster-whisper` — GPU transcription
- `descript-audio-codec`, `descript-audiotools` — DAC codec
- `gradio>5.0`, `uvicorn`, `fastapi` — WebUI + REST API
- `huggingface_hub` — model download

### Copy modified files

Copy the files from this repo into your `fish-speech` directory (handled automatically by `fish_install_3090.sh`):

```
tools/run_webui.py          → fish-speech/tools/run_webui.py
tools/fish_api.py           → fish-speech/tools/fish_api.py
tools/webui/__init__.py     → fish-speech/tools/webui/__init__.py
tools/webui/variables.py    → fish-speech/tools/webui/variables.py
fetch_el_metadata.py        → fish-speech/fetch_el_metadata.py
```

### Fetch ElevenLabs voice metadata

```bash
python fetch_el_metadata.py
# Saves el_voices.json (~2MB, 10,790 voices)
```

### Launch

```bash
cd fish-speech
python tools/run_webui.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq \
  --half \
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
- Added `--half` and `--compile` to launch flags
- Enabled `torch.backends.cudnn.benchmark` and TF32
- Calls `init_api(inference_engine)` to share the loaded model with the API

### `tools/fish_api.py` *(new file)*
- `GET /gallery` — searches `el_voices.json` + saved voices with filtering and pagination
- `POST /generate` — resolves voice_id → fetches preview on-demand → Whisper transcription → TTS inference → returns WAV
- LRU cache (500 voices) for audio bytes + transcription to avoid redundant downloads

### `tools/webui/__init__.py`
- Added **Voice Gallery** tab: searchable/filterable dropdown over 10,790 ElevenLabs voices, loads preview on-demand
- Added **My Saved Voices** tab: save/load/delete custom voices with name, description, publish flag
- Auto-transcription: `reference_audio.change` → Whisper → populates reference text field automatically
- `app.load()` refresh on page load so saved voices always reflect current state

### `tools/webui/variables.py`
- Updated header to say **Fish Speech S2 Pro** with correct HuggingFace link

### `fetch_el_metadata.py` *(new file)*
- Paginates ElevenLabs `/v1/shared-voices` API and saves slim metadata JSON
- No audio downloaded — purely metadata (name, description, gender, age, accent, language, preview_url)

---

## Voice Library

Users can save any voice (from gallery or uploaded audio) to a local library:

- Saved to `voices/{uuid}/` with `audio.{ext}` + `meta.json`
- Accessible via the **My Saved Voices** tab in the WebUI or `source=saved` in `/gallery`
- `published` flag controls visibility

---

## ElevenLabs Integration

Voice metadata is fetched once via `fetch_el_metadata.py` and stored as `el_voices.json`. No ElevenLabs API key is needed at inference time — preview audio is fetched directly from public Google Cloud Storage URLs embedded in the metadata.

The LRU cache ensures each unique voice is only downloaded and transcribed once per server session.
