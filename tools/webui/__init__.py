import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests
from loguru import logger

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
        segments, info = _get_whisper().transcribe(audio_path, beam_size=1)
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
    """Download preview on-demand, transcribe, return as reference audio + text + voice meta."""
    if not label:
        return None, "", {}
    short_id = label.split("(")[-1].rstrip(")")
    voice = next((v for v in _load_el_voices() if v["id"].startswith(short_id)), None)
    if not voice or not voice.get("preview_url"):
        return None, "", {}

    logger.info(f"Fetching preview for: {voice['name']}")
    try:
        resp = requests.get(voice["preview_url"], timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Preview download failed: {e}")
        return None, "", {}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    transcription = transcribe_audio(tmp_path)
    return tmp_path, transcription, voice


def _unique_filter_values(key: str) -> list[str]:
    vals = sorted(set(v.get(key, "") for v in _load_el_voices() if v.get(key)))
    return ["Any"] + vals


# ── Custom voice library (user-saved) ─────────────────────────────────────────

def _find_saved_by_id(voice_id: str) -> dict | None:
    voice_id = voice_id.strip()
    if not voice_id:
        return None
    # Exact match first
    p = VOICES_DIR / voice_id / "meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # Prefix match (short IDs)
    for meta_path in VOICES_DIR.glob("*/meta.json"):
        if meta_path.parent.name.startswith(voice_id):
            with open(meta_path) as f:
                return json.load(f)
    return None


def save_voice(audio_path, transcription, name, description, published):
    if not audio_path:
        return "⚠️ No audio to save.", ""
    if not name.strip():
        return "⚠️ Enter a voice name.", ""
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
    return f"✅ Saved **{name.strip()}**", vid


def load_saved_voice_by_id(voice_id: str):
    v = _find_saved_by_id(voice_id)
    if not v:
        return None, "", "<span style='color:red'>⚠️ Voice ID not found.</span>", {}
    # Saved voices have name/description but typically no gender/age/accent
    meta = {"name": v.get("name", ""), "description": v.get("description", "")}
    return v["audio_file"], v["transcription"], f"<span style='color:green'>✓ Loaded **{v['name']}**</span>", meta


# ── App ────────────────────────────────────────────────────────────────────────

def build_app(inference_fct, theme: str = "light") -> gr.Blocks:
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
                text = gr.Textbox(label="Input Text", placeholder=TEXTBOX_PLACEHOLDER, lines=10, elem_id="text-input")

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

                                with gr.Tab("📤 Upload"):
                                    gr.Markdown("Upload your own reference audio (5–10 sec).")

                            gr.Markdown("---")
                            reference_audio = gr.Audio(label="Reference Audio", type="filepath")
                            reference_text  = gr.Textbox(label="Reference Text (auto-transcribed)", lines=2,
                                                         placeholder="Auto-filled on upload or voice load.")

                            gr.Markdown("---")
                            gr.Markdown("**Load Custom Voice**")
                            with gr.Row():
                                load_custom_id  = gr.Textbox(label="Voice ID", placeholder="Paste your voice ID here…", scale=3)
                                load_custom_btn = gr.Button("⬇️ Load", variant="primary", scale=1)
                            load_custom_status = gr.HTML("")

                            gr.Markdown("---")
                            gr.Markdown("**Save current audio as a voice**")
                            with gr.Row():
                                save_name  = gr.Textbox(label="Name", scale=3)
                                save_pub   = gr.Checkbox(label="Public", value=True, scale=1)
                            save_desc      = gr.Textbox(label="Description", lines=1)
                            save_btn       = gr.Button("💾 Save Voice")
                            save_status    = gr.Markdown("")
                            saved_id_box   = gr.Textbox(label="Voice ID (copy to reuse across sessions)", interactive=False, visible=False)

            # ── Right ─────────────────────────────────────────────────────────
            with gr.Column(scale=3):
                error = gr.HTML(label="Error", visible=True)
                audio = gr.Audio(label="Generated Audio", type="numpy", interactive=False)
                generate = gr.Button("🎧 Generate", variant="primary")

        # ── Events ────────────────────────────────────────────────────────────

        current_voice_meta = gr.State({})

        # Auto-transcribe on upload
        reference_audio.change(transcribe_audio, [reference_audio], [reference_text])

        # Gallery search + filter
        for component in [search_box, lang_filter, gender_filter, age_filter, accent_filter]:
            component.change(
                filter_voices,
                [search_box, lang_filter, gender_filter, age_filter, accent_filter],
                [gallery_dropdown, gallery_status],
            )

        # Load from gallery (on-demand fetch + transcribe + store voice meta)
        load_gallery_btn.click(
            load_el_voice,
            [gallery_dropdown],
            [reference_audio, reference_text, current_voice_meta],
        )

        # Load custom voice by ID (stores minimal meta for enhance)
        load_custom_btn.click(
            load_saved_voice_by_id,
            [load_custom_id],
            [reference_audio, reference_text, load_custom_status, current_voice_meta],
        )

        # Save voice — display generated ID in copyable box
        def _save_and_show(*args):
            status, vid = save_voice(*args)
            visible = bool(vid)
            return status, gr.update(value=vid, visible=visible)

        save_btn.click(
            _save_and_show,
            [reference_audio, reference_text, save_name, save_desc, save_pub],
            [save_status, saved_id_box],
        )

        # Generate
        generate.click(
            inference_fct,
            [text, reference_audio, reference_text],
            [audio, error],
            concurrency_limit=1,
        )

    return app
