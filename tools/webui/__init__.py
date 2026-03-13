import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import gradio as gr
import requests
from loguru import logger

from fish_speech.i18n import i18n
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER

VOICES_DIR = Path("/root/fish-speech/voices")
VOICES_DIR.mkdir(exist_ok=True)
EL_VOICES_FILE = Path("/root/fish-speech/el_voices.json")

_whisper_model = None
_el_voices: list[dict] = []


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper model (CUDA)...")
        _whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
        logger.info("Whisper ready.")
    return _whisper_model


def _load_el_voices() -> list[dict]:
    global _el_voices
    if not _el_voices and EL_VOICES_FILE.exists():
        with open(EL_VOICES_FILE) as f:
            _el_voices = json.load(f)
        logger.info(f"Loaded {len(_el_voices)} ElevenLabs voices from metadata.")
    return _el_voices


def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""
    try:
        segments, info = _get_whisper().transcribe(audio_path, beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        logger.info(f"Transcribed [{info.language}]: {text[:80]}")
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


# ── ElevenLabs gallery ────────────────────────────────────────────────────────

def filter_voices(search: str, lang: str, gender: str, age: str, accent: str) -> gr.update:
    voices = _load_el_voices()
    results = []
    search_l = search.lower().strip()
    for v in voices:
        if search_l and search_l not in v["name"].lower() and search_l not in v.get("description", "").lower():
            continue
        if lang and lang != "Any" and v.get("language", "") != lang:
            continue
        if gender and gender != "Any" and v.get("gender", "") != gender:
            continue
        if age and age != "Any" and v.get("age", "") != age:
            continue
        if accent and accent != "Any" and v.get("accent", "") != accent:
            continue
        results.append(v)
        if len(results) >= 100:
            break

    choices = [f"{v['name']} | {v.get('gender','')} {v.get('age','')} {v.get('accent','')} [{v.get('language','')}]  ({v['id'][:8]})"
               for v in results]
    return gr.update(choices=choices, value=None), f"{len(results)} voices found"


def load_el_voice(label: str):
    """Download preview on-demand, transcribe, return as reference audio + text."""
    if not label:
        return None, ""
    short_id = label.split("(")[-1].rstrip(")")
    voice = next((v for v in _load_el_voices() if v["id"].startswith(short_id)), None)
    if not voice or not voice.get("preview_url"):
        return None, ""

    logger.info(f"Fetching preview for: {voice['name']}")
    try:
        resp = requests.get(voice["preview_url"], timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Preview download failed: {e}")
        return None, ""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    transcription = transcribe_audio(tmp_path)
    return tmp_path, transcription


def _unique_filter_values(key: str) -> list[str]:
    vals = sorted(set(v.get(key, "") for v in _load_el_voices() if v.get(key)))
    return ["Any"] + vals


# ── Custom voice library (user-saved) ─────────────────────────────────────────

def _load_saved_voices() -> list[dict]:
    voices = []
    for p in sorted(VOICES_DIR.glob("*/meta.json")):
        try:
            with open(p) as f:
                voices.append(json.load(f))
        except Exception:
            pass
    return voices


def _saved_choices() -> list[str]:
    return [f"{v['name']}  ({v['id'][:8]})" for v in _load_saved_voices()]


def _find_saved(label: str) -> dict | None:
    sid = label.split("(")[-1].rstrip(")").strip() if "(" in label else ""
    return next((v for v in _load_saved_voices() if v["id"].startswith(sid)), None)


def save_voice(audio_path, transcription, name, description, published):
    if not audio_path:
        return "⚠️ No audio to save.", gr.update()
    if not name.strip():
        return "⚠️ Enter a voice name.", gr.update()
    vid = str(uuid.uuid4())
    vdir = VOICES_DIR / vid
    vdir.mkdir()
    ext = Path(audio_path).suffix or ".wav"
    dest = vdir / f"audio{ext}"
    shutil.copy2(audio_path, dest)
    meta = {
        "id": vid, "name": name.strip(), "description": description.strip(),
        "transcription": transcription, "published": published,
        "audio_file": str(dest), "created_at": datetime.utcnow().isoformat(),
    }
    with open(vdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return f"✅ Saved **{name}**", gr.update(choices=_saved_choices())


def load_saved_voice(label: str):
    v = _find_saved(label)
    if not v:
        return None, ""
    return v["audio_file"], v["transcription"]


def delete_saved_voice(label: str):
    v = _find_saved(label)
    if not v:
        return "⚠️ Not found.", gr.update(choices=_saved_choices())
    shutil.rmtree(VOICES_DIR / v["id"], ignore_errors=True)
    return f"🗑️ Deleted **{v['name']}**", gr.update(choices=_saved_choices())


# ── App ────────────────────────────────────────────────────────────────────────

def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    _load_el_voices()  # pre-load at startup

    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)
        app.load(
            None, None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}" % theme,
        )

        with gr.Row():
            # ── Left ──────────────────────────────────────────────────────────
            with gr.Column(scale=3):
                text = gr.Textbox(label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10)

                with gr.Row():
                    with gr.Column():

                        with gr.Tab(label="🎙️ Voice"):
                            reference_id = gr.Textbox(label="Reference ID", placeholder="Leave empty to use gallery/uploaded audio", visible=False)

                            with gr.Tabs():

                                with gr.Tab("🌐 Voice Gallery"):
                                    gr.Markdown(f"**{len(_el_voices):,} voices** — search and filter, preview fetched on demand.")
                                    with gr.Row():
                                        search_box = gr.Textbox(label="Search by name / description", scale=3)
                                        gallery_status = gr.Textbox(label="", value="", interactive=False, scale=1)
                                    with gr.Row():
                                        lang_filter   = gr.Dropdown(label="Language", choices=_unique_filter_values("language"), value="Any", scale=1)
                                        gender_filter = gr.Dropdown(label="Gender",   choices=_unique_filter_values("gender"),   value="Any", scale=1)
                                        age_filter    = gr.Dropdown(label="Age",      choices=_unique_filter_values("age"),      value="Any", scale=1)
                                        accent_filter = gr.Dropdown(label="Accent",   choices=_unique_filter_values("accent"),   value="Any", scale=1)
                                    gallery_dropdown = gr.Dropdown(
                                        label="Select voice (fetches preview on load)",
                                        choices=[], value=None, interactive=True,
                                    )
                                    load_gallery_btn = gr.Button("⬇️ Load Voice", variant="primary")

                                with gr.Tab("⭐ My Saved Voices"):
                                    saved_dropdown = gr.Dropdown(
                                        label="Saved voice", choices=_saved_choices(), value=None, interactive=True,
                                    )
                                    with gr.Row():
                                        load_saved_btn  = gr.Button("Load", variant="primary", scale=2)
                                        delete_saved_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                                    saved_status = gr.Markdown("")

                                with gr.Tab("📤 Upload"):
                                    gr.Markdown("Upload your own reference audio (5–10 sec).")

                            gr.Markdown("---")
                            reference_audio = gr.Audio(label="Reference Audio", type="filepath")
                            reference_text  = gr.Textbox(label="Reference Text (auto-transcribed)", lines=2,
                                                         placeholder="Auto-filled on upload or voice load.")

                            gr.Markdown("---")
                            gr.Markdown("**Save current audio as a voice**")
                            with gr.Row():
                                save_name  = gr.Textbox(label="Name", scale=3)
                                save_pub   = gr.Checkbox(label="Public", value=True, scale=1)
                            save_desc      = gr.Textbox(label="Description", lines=1)
                            save_btn       = gr.Button("💾 Save Voice")
                            save_status    = gr.Markdown("")

                        with gr.Tab(label="Advanced Config"):
                            with gr.Row():
                                chunk_length = gr.Slider(label="Iterative Prompt Length (0=off)", minimum=100, maximum=400, value=300, step=8)
                                max_new_tokens = gr.Slider(label="Max tokens per batch (0=unlimited)", minimum=0, maximum=2048, value=0, step=8)
                            with gr.Row():
                                top_p = gr.Slider(label="Top-P", minimum=0.7, maximum=0.95, value=0.8, step=0.01)
                                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1, maximum=1.2, value=1.1, step=0.01)
                            with gr.Row():
                                temperature = gr.Slider(label="Temperature", minimum=0.7, maximum=1.0, value=0.8, step=0.01)
                                seed = gr.Number(label="Seed (0=random)", value=0)

            # ── Right ─────────────────────────────────────────────────────────
            with gr.Column(scale=3):
                error = gr.HTML(label="Error", visible=True)
                audio = gr.Audio(label="Generated Audio", type="numpy", interactive=False)
                generate = gr.Button("🎧 Generate", variant="primary")

        # ── Events ────────────────────────────────────────────────────────────

        # Auto-transcribe on upload
        reference_audio.change(transcribe_audio, [reference_audio], [reference_text])

        # Gallery search + filter
        for component in [search_box, lang_filter, gender_filter, age_filter, accent_filter]:
            component.change(
                filter_voices,
                [search_box, lang_filter, gender_filter, age_filter, accent_filter],
                [gallery_dropdown, gallery_status],
            )

        # Load from gallery (on-demand fetch + transcribe)
        load_gallery_btn.click(
            load_el_voice,
            [gallery_dropdown],
            [reference_audio, reference_text],
        )

        # Load / delete saved voice
        load_saved_btn.click(load_saved_voice, [saved_dropdown], [reference_audio, reference_text])
        delete_saved_btn.click(delete_saved_voice, [saved_dropdown], [saved_status, saved_dropdown])

        # Save voice
        save_btn.click(
            save_voice,
            [reference_audio, reference_text, save_name, save_desc, save_pub],
            [save_status, saved_dropdown],
        )

        use_memory_cache = gr.State("on")

        # Generate
        generate.click(
            inference_fct,
            [text, reference_id, reference_audio, reference_text,
             max_new_tokens, chunk_length, top_p, repetition_penalty,
             temperature, seed, use_memory_cache],
            [audio, error],
            concurrency_limit=1,
        )

    return app
