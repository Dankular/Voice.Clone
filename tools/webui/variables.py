from fish_speech.i18n import i18n

HEADER_MD = """# Fish Speech S2 Pro

Choose from **10,790+ voices** in the gallery or upload your own reference audio.
Use `GET /gallery` to search voices by name, language, gender, age, and accent.
Use `POST /generate` with a `voice_id` and `text` to synthesise audio — reference audio and transcription are handled automatically.

**Fine-Grained Inline Control** — embed `[tag]` syntax anywhere in your text to control tone, emotion, and delivery at word level. S2 Pro accepts free-form descriptions: `[whisper]`, `[excited]`, `[pause]`, `[laughing]`, and 15,000+ more. Click any tag below to insert it at your cursor.
"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
