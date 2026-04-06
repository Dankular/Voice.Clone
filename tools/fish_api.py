"""
Fish Speech API routes — mounted alongside the Gradio UI.

GET  /gallery          — list all voices (ElevenLabs + saved)
POST /generate         — generate audio from voice_id + text
"""
import io
import json
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from tag_classifier import classify_and_tag

EL_VOICES_FILE = Path("/root/fish-speech/el_voices.json")
VOICES_DIR = Path("/root/fish-speech/voices")

router = APIRouter()

# ── shared state (set by run_webui.py after model load) ──────────────────────
_inference_engine = None
_whisper_model = None

# ── LRU cache: voice_id → (audio_bytes, transcription) ───────────────────────
_VOICE_CACHE_MAX = 500
_voice_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()


def _cache_get(voice_id: str):
    if voice_id in _voice_cache:
        _voice_cache.move_to_end(voice_id)
        return _voice_cache[voice_id]
    return None


def _cache_set(voice_id: str, audio_bytes: bytes, transcription: str):
    _voice_cache[voice_id] = (audio_bytes, transcription)
    _voice_cache.move_to_end(voice_id)
    if len(_voice_cache) > _VOICE_CACHE_MAX:
        _voice_cache.popitem(last=False)
    logger.info(f"Cached voice {voice_id[:12]} ({len(_voice_cache)}/{_VOICE_CACHE_MAX})")


def init_api(inference_engine):
    global _inference_engine
    _inference_engine = inference_engine


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_el_voices() -> list[dict]:
    if EL_VOICES_FILE.exists():
        with open(EL_VOICES_FILE) as f:
            return json.load(f)
    return []


def _load_saved_voices() -> list[dict]:
    voices = []
    for p in sorted(VOICES_DIR.glob("*/meta.json")):
        try:
            with open(p) as f:
                voices.append(json.load(f))
        except Exception:
            pass
    return voices


def _transcribe(audio_path: str) -> str:
    segments, _ = _get_whisper().transcribe(audio_path, beam_size=1)
    return " ".join(s.text.strip() for s in segments)


def _fetch_and_transcribe_el(voice_id: str):
    """Return (tmp_audio_path, transcription, is_tmp) for an ElevenLabs voice."""
    # Check cache first
    cached = _cache_get(voice_id)
    if cached:
        audio_bytes, transcription = cached
        logger.info(f"Cache hit for voice {voice_id[:12]}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            return tmp.name, transcription

    voices = _load_el_voices()
    voice = next((v for v in voices if v["id"] == voice_id), None)
    if not voice or not voice.get("preview_url"):
        raise HTTPException(404, f"ElevenLabs voice '{voice_id}' not found")

    resp = requests.get(voice["preview_url"], timeout=20)
    resp.raise_for_status()
    audio_bytes = resp.content

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    transcription = _transcribe(tmp_path)
    _cache_set(voice_id, audio_bytes, transcription)
    return tmp_path, transcription


def _resolve_saved_voice(voice_id: str):
    """Return (audio_path, transcription) for a saved voice."""
    cached = _cache_get(voice_id)
    if cached:
        audio_bytes, transcription = cached
        logger.info(f"Cache hit for saved voice {voice_id[:12]}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            return tmp.name, transcription

    voices = _load_saved_voices()
    voice = next((v for v in voices if v["id"] == voice_id or v["id"].startswith(voice_id)), None)
    if not voice:
        raise HTTPException(404, f"Saved voice '{voice_id}' not found")

    audio_path = voice["audio_file"]
    transcription = voice.get("transcription", "")

    with open(audio_path, "rb") as f:
        _cache_set(voice_id, f.read(), transcription)

    return audio_path, transcription


def _tag_text(text: str, voice_meta: dict) -> str:
    """Classify sentences and insert prosody tags. Returns original text on failure."""
    return classify_and_tag(text, voice_meta)


def _estimate_max_tokens(text: str) -> int:
    """~3 semantic tokens per char at normal speech rate, 3x safety buffer, capped at 4096."""
    return max(512, min(4096, len(text) * 10))


def _run_omnivoice_inference(text: str, audio_path: str, transcription: str) -> tuple[int, np.ndarray]:
    """Run inference via OmniVoice (lazy-loaded)."""
    from tools.webui.inference import _get_omnivoice

    model = _get_omnivoice()

    kwargs = {"text": text}
    if audio_path:
        kwargs["ref_audio"] = audio_path
        if transcription:
            kwargs["ref_text"] = transcription

    audio_tensors = model.generate(**kwargs)
    wav = audio_tensors[0].cpu().numpy().squeeze()
    return 24000, wav.astype(np.float32)


def _run_inference(text: str, audio_path: str, transcription: str, **kwargs) -> tuple[int, np.ndarray]:
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    max_new_tokens = kwargs.get("max_new_tokens", 0) or _estimate_max_tokens(text)

    req = ServeTTSRequest(
        text=text,
        references=[ServeReferenceAudio(audio=audio_bytes, text=transcription)],
        reference_id=None,
        max_new_tokens=max_new_tokens,
        chunk_length=kwargs.get("chunk_length", 300),
        top_p=kwargs.get("top_p", 0.8),
        repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        temperature=kwargs.get("temperature", 0.8),
        seed=kwargs.get("seed") or None,
        format="wav",
    )

    for result in _inference_engine.inference(req):
        if result.code == "final":
            return result.audio  # (sample_rate, np.ndarray)
        if result.code == "error":
            raise HTTPException(500, str(result.error))

    raise HTTPException(500, "No audio generated")


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/gallery")
def gallery(
    search: str = Query("", description="Filter by name or description"),
    language: str = Query("", description="Filter by language code e.g. en"),
    gender: str = Query("", description="male / female"),
    age: str = Query("", description="young / middle-aged / old"),
    accent: str = Query("", description="Filter by accent"),
    source: str = Query("all", description="all | elevenlabs | saved"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    results = []

    if source in ("all", "elevenlabs"):
        for v in _load_el_voices():
            if search and search.lower() not in v["name"].lower() and search.lower() not in v.get("description", "").lower():
                continue
            if language and v.get("language", "") != language:
                continue
            if gender and v.get("gender", "") != gender:
                continue
            if age and v.get("age", "") != age:
                continue
            if accent and v.get("accent", "") != accent:
                continue
            results.append({
                "voice_id":    v["id"],
                "name":        v["name"],
                "description": v.get("description", ""),
                "gender":      v.get("gender", ""),
                "age":         v.get("age", ""),
                "accent":      v.get("accent", ""),
                "language":    v.get("language", ""),
                "category":    v.get("category", ""),
                "source":      "elevenlabs",
                "preview_url": v.get("preview_url", ""),
            })

    if source in ("all", "saved"):
        for v in _load_saved_voices():
            if search and search.lower() not in v["name"].lower():
                continue
            results.append({
                "voice_id":    v["id"],
                "name":        v["name"],
                "description": v.get("description", ""),
                "gender":      v.get("gender", ""),
                "language":    v.get("language", ""),
                "source":      "saved",
                "published":   v.get("published", False),
            })

    total = len(results)
    page = results[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "voices": page}


class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    model: str = "fish-speech"  # "fish-speech" or "omnivoice"
    max_new_tokens: int = 0
    chunk_length: int = 300
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    temperature: float = 0.8
    seed: int = 0
    format: str = "wav"


@router.post("/generate")
def generate(req: GenerateRequest):
    logger.info(f"API /generate: model={req.model}, voice_id={req.voice_id}, text={req.text[:60]}")

    # Resolve voice metadata for tagger
    el_voices = _load_el_voices()
    voice_meta = next((v for v in el_voices if v["id"] == req.voice_id), {})

    # Tag text with prosody annotations before inference
    tagged_text = _tag_text(req.text, voice_meta)

    # Resolve voice → audio path + transcription
    saved_ids = {v["id"] for v in _load_saved_voices()}
    if req.voice_id in saved_ids or any(v["id"].startswith(req.voice_id) for v in _load_saved_voices()):
        audio_path, transcription = _resolve_saved_voice(req.voice_id)
        cleanup = False
    else:
        audio_path, transcription = _fetch_and_transcribe_el(req.voice_id)
        cleanup = True

    try:
        if req.model == "omnivoice":
            sample_rate, audio_data = _run_omnivoice_inference(
                text=tagged_text,
                audio_path=audio_path,
                transcription=transcription,
            )
        else:
            if _inference_engine is None:
                raise HTTPException(503, "Fish Speech inference engine not ready")
            sample_rate, audio_data = _run_inference(
                text=tagged_text,
                audio_path=audio_path,
                transcription=transcription,
                max_new_tokens=req.max_new_tokens,
                chunk_length=req.chunk_length,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                temperature=req.temperature,
                seed=req.seed,
            )
    finally:
        if cleanup:
            Path(audio_path).unlink(missing_ok=True)

    # Encode to WAV in memory
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )
