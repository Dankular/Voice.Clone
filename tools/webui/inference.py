import html
import tempfile
from functools import partial
from typing import Any, Callable

import numpy as np
from loguru import logger

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

# ── OmniVoice lazy singleton ────────────────────────────────────────────────
_omnivoice_model = None


def _get_omnivoice():
    global _omnivoice_model
    if _omnivoice_model is None:
        import torch
        from omnivoice import OmniVoice

        logger.info("Loading OmniVoice model (first use)...")
        _omnivoice_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
        )
        logger.info("OmniVoice ready.")
    return _omnivoice_model


def _estimate_max_tokens(text: str) -> int:
    """~3 semantic tokens per char at normal speech rate, 3x safety buffer, capped at 4096."""
    return max(512, min(4096, len(text) * 10))


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    model_choice,
    engine,
):
    """
    Wrapper for the inference function.
    Routes to Fish Speech or OmniVoice based on model_choice.
    """

    if model_choice == "OmniVoice":
        return _omnivoice_inference(text, reference_audio, reference_text)

    # ── Fish Speech S2 Pro (default) ─────────────────────────────────────────
    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    req = ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=int(max_new_tokens) or _estimate_max_tokens(text),
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio, None
            case "error":
                return None, build_html_error_message(i18n(result.error))
            case _:
                pass

    return None, i18n("No audio generated")


def _omnivoice_inference(text: str, reference_audio: str | None, reference_text: str):
    """Run inference via the OmniVoice model."""
    try:
        import torchaudio

        model = _get_omnivoice()

        kwargs: dict[str, Any] = {"text": text}
        if reference_audio:
            kwargs["ref_audio"] = reference_audio
            if reference_text:
                kwargs["ref_text"] = reference_text

        audio_tensors = model.generate(**kwargs)
        # audio_tensors is a list of tensors, each (1, T) at 24 kHz
        wav = audio_tensors[0].cpu().numpy().squeeze()
        return (24000, wav.astype(np.float32)), None
    except ImportError:
        return None, build_html_error_message(
            Exception("OmniVoice is not installed. Run: pip install omnivoice")
        )
    except Exception as e:
        logger.error(f"OmniVoice inference failed: {e}")
        return None, build_html_error_message(e)


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red;
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
