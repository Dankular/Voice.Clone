import html
from typing import Any

import numpy as np
from loguru import logger

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


def inference_wrapper(
    text,
    reference_audio,
    reference_text,
):
    """
    Run inference via OmniVoice.
    """
    try:
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


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red;
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """
