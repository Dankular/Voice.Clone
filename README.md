# Voice Clone

A production-ready voice cloning API and web UI powered by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — lightweight 600+ language zero-shot voice cloning, with a 10,000+ voice gallery sourced from ElevenLabs shared voices.

---

## Features

- **10,790+ voice gallery** — all ElevenLabs public shared voices, searchable and filterable by language, gender, age, accent
- **Zero storage** — voice metadata only (~2MB JSON), previews fetched on-demand
- **Auto-transcription** — Whisper (CPU) automatically transcribes reference audio, no manual input needed
- **REST API** — `/gallery` and `/generate` endpoints for programmatic access
- **Gradio WebUI** — browser-based interface for testing
- **Voice library** — save and manage custom voices
- **LRU cache** — repeated calls to the same voice skip download + transcription
- **600+ languages** — OmniVoice supports zero-shot cloning across 600+ languages
- **Lightweight** — model weights auto-download on first use, no large checkpoint required
- **Prosody tagger** — embedding-based classifier annotates text with `[pause]`, `[emphasis]`, etc. before synthesis

---

## Setup

### Requirements

- Python 3.11
- CUDA GPU (tested on RTX 3090 — Ampere sm86, CUDA 12.1)
- 32GB+ system RAM
- CUDA driver 535+ (CUDA 12.2 runtime, toolkit 12.1)

### Install

Everything is handled by `init.sh`:

```bash
# 1. Full install (deps, OmniVoice, cloudflared, custom files)
bash init.sh install

# 2. (Optional) Build llama.cpp with CUDA
bash init.sh llamacpp

# 3. Launch
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
| `install` | Full first-time install (deps, env, cloudflared, custom files) |
| `llamacpp` | Build llama.cpp with CUDA support |
| `start` | Kill and restart all services (starts webui + cloudflared) |
| `pull` | Pull latest custom files from GitHub then restart |
| `auto` *(default)* | Install if not installed, pull latest files, then start |

**Key Python deps** (installed by `init.sh install`):
- `torch==2.5.1+cu121` / `torchaudio` — pinned for CUDA 12.1 / RTX 3090
- `omnivoice` — zero-shot voice cloning engine
- `faster-whisper` — CPU transcription
- `gradio>5.0`, `uvicorn`, `fastapi` — WebUI + REST API

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
python tools/run_webui.py
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
  "text": "Your text to synthesise goes here."
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
