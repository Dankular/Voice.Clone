"""
Sentence-level tag classifier using sentence-transformers embeddings.

Replaces the Qwen2.5 LLM tagger with a lightweight embedding model
(all-MiniLM-L6-v2, ~22 MB) for faster, purpose-built tag prediction.
"""
import re
from typing import Optional

import numpy as np
from loguru import logger

# ── Tag definitions: tag → natural-language description for embedding ────────
# Richer descriptions yield better cosine-similarity matches.

TAG_DESCRIPTIONS: dict[str, str] = {
    # Paralinguistic
    "[laughing]": "something very funny, burst of laughter, hilarious",
    "[chuckling]": "mildly amusing, light chuckle, quiet amusement",
    "[chuckle]": "brief quiet laugh, slightly amused",
    "[giggling]": "playful giggle, teasing, childlike amusement",
    "[sighing]": "deep sigh, weariness, resignation, tired exhale",
    "[sigh]": "brief sigh, mild frustration or acceptance",
    "[exhale]": "breathing out, relief, letting go of tension",
    "[inhale]": "breathing in, preparing to speak, bracing",
    "[tsk]": "disapproval, mild scolding, tutting sound",
    "[gasp]": "sudden shock, surprise, alarming revelation",
    "[crying]": "deep sorrow, tears, emotional breakdown, weeping",
    "[sobbing]": "uncontrollable crying, grief, devastation",
    "[panting]": "out of breath, exhausted, physical exertion",
    "[clearing throat]": "preparing to speak formally, getting attention",
    "[moaning]": "pain, discomfort, suffering, groaning",
    # Emotion
    "[excited]": "thrilling news, enthusiasm, eagerness, can't wait",
    "[angry]": "fury, rage, outrage, furious confrontation",
    "[sad]": "melancholy, grief, loss, heartbreak, sorrow",
    "[nervous]": "anxiety, worry, uncertainty, fear of outcome",
    "[confident]": "assertive, self-assured, commanding, certain",
    "[surprised]": "unexpected event, astonishment, didn't see that coming",
    "[disappointed]": "let down, unmet expectations, deflated hopes",
    "[disgusted]": "revulsion, repulsed, nauseated, moral outrage",
    "[scared]": "fear, terror, dread, frightened, danger",
    "[happy]": "joyful, cheerful, pleased, good news, content",
    "[upset]": "distressed, bothered, emotionally hurt, troubled",
    "[confused]": "bewildered, puzzled, not understanding, perplexed",
    "[delight]": "pure joy, wonderful surprise, enchanted",
    "[shocked]": "stunned disbelief, jaw-dropping, can't believe it",
    # Delivery
    "[whisper]": "secret, intimate, hushed voice, quiet confession",
    "[low voice]": "subdued tone, speaking quietly, gravely serious",
    "[shouting]": "yelling loudly, calling out, raising voice in anger",
    "[screaming]": "extreme fear or rage, shrieking, terrified scream",
    "[loud]": "speaking loudly, projecting voice, being heard",
    "[singing]": "musical, melodic speech, humming a tune",
    "[interrupting]": "cutting someone off, urgent interjection",
    "[with strong accent]": "heavily accented speech, regional dialect",
    # Prosody
    "[pause]": "dramatic silence, moment of reflection, beat",
    "[short pause]": "brief hesitation, thinking, slight beat",
    "[long pause]": "extended silence, heavy moment, processing",
    "[emphasis]": "stressing an important word, underlining significance",
    "[slow]": "deliberate pacing, careful articulation, gravity",
    "[fast]": "rapid speech, urgency, excitement, rushing",
    # Style
    "[warm tone]": "friendly, caring, compassionate, gentle warmth",
    "[cold tone]": "detached, dismissive, emotionally distant, icy",
    "[sarcastic]": "irony, mocking, saying opposite of meaning, dry wit",
    "[dramatic]": "theatrical, heightened emotion, intense moment",
}

# Voice profile keywords → tags to suppress
_PROFILE_SUPPRESS: dict[str, set[str]] = {
    "calm": {"[screaming]", "[shouting]", "[giggling]", "[excited]", "[loud]"},
    "measured": {"[screaming]", "[shouting]", "[giggling]", "[excited]", "[fast]"},
    "professional": {"[laughing]", "[giggling]", "[crying]", "[sobbing]", "[moaning]",
                     "[screaming]", "[singing]", "[panting]"},
    "broadcaster": {"[laughing]", "[giggling]", "[crying]", "[sobbing]", "[moaning]",
                    "[screaming]", "[singing]", "[panting]"},
    "stoic": {"[giggling]", "[laughing]", "[excited]", "[delight]", "[happy]", "[singing]"},
    "narrator": {"[screaming]", "[shouting]", "[giggling]", "[panting]", "[moaning]"},
    "serious": {"[giggling]", "[laughing]", "[singing]", "[delight]"},
}

# ── Singleton model holder ───────────────────────────────────────────────────

_model = None
_tag_embeddings: Optional[np.ndarray] = None
_tag_names: list[str] = []

# Density: ~1 tag per 15-25 words as in original prompt
_SIMILARITY_THRESHOLD = 0.35
_MAX_TAGS_PER_SENTENCE = 2
_MIN_SENTENCE_WORDS = 3


def _load_model():
    """Lazy-load sentence-transformers model and pre-compute tag embeddings."""
    global _model, _tag_embeddings, _tag_names
    if _model is not None:
        return

    from sentence_transformers import SentenceTransformer

    logger.info("Loading tag classifier model (all-MiniLM-L6-v2)...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    _tag_names = list(TAG_DESCRIPTIONS.keys())
    descriptions = list(TAG_DESCRIPTIONS.values())
    _tag_embeddings = _model.encode(descriptions, normalize_embeddings=True)
    logger.info(f"Tag classifier ready — {len(_tag_names)} tags indexed")


def _get_suppressed_tags(voice_meta: Optional[dict]) -> set[str]:
    """Determine which tags to suppress based on voice profile description."""
    if not voice_meta:
        return set()
    desc = (voice_meta.get("description", "") + " " + voice_meta.get("name", "")).lower()
    suppressed: set[str] = set()
    for keyword, tags in _PROFILE_SUPPRESS.items():
        if keyword in desc:
            suppressed |= tags
    return suppressed


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving whitespace/punctuation."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?…])\s+', text)
    # Also split on em-dash or semicolon boundaries for long clauses
    result = []
    for part in parts:
        if len(part.split()) > 30:
            # Further split long segments on — or ;
            sub = re.split(r'(?<=—)\s*|(?<=;)\s+', part)
            result.extend(sub)
        else:
            result.append(part)
    return [s for s in result if s.strip()]


def classify_and_tag(text: str, voice_meta: Optional[dict] = None) -> str:
    """
    Classify each sentence and insert the best-matching tag(s) inline.

    Returns the annotated text, or the original text on failure.
    """
    if not text.strip():
        return text

    try:
        _load_model()

        sentences = _split_sentences(text)
        if not sentences:
            return text

        suppressed = _get_suppressed_tags(voice_meta)

        # Embed all sentences at once
        sentence_embeddings = _model.encode(sentences, normalize_embeddings=True)

        # Cosine similarity (already normalized → dot product)
        similarities = sentence_embeddings @ _tag_embeddings.T  # (n_sentences, n_tags)

        tagged_parts = []
        tags_used = 0

        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())

            # Skip very short fragments
            if word_count < _MIN_SENTENCE_WORDS:
                tagged_parts.append(sentence)
                continue

            # Get top tag candidates
            scores = similarities[i]
            top_indices = np.argsort(scores)[::-1]

            inserted_tags = []
            for idx in top_indices:
                if len(inserted_tags) >= _MAX_TAGS_PER_SENTENCE:
                    break
                tag = _tag_names[idx]
                score = scores[idx]

                if score < _SIMILARITY_THRESHOLD:
                    break
                if tag in suppressed:
                    continue

                inserted_tags.append(tag)

            if inserted_tags:
                # Density control: aim for ~1 tag per 15-25 words overall
                tagged_parts.append(" ".join(inserted_tags) + " " + sentence)
                tags_used += len(inserted_tags)
            else:
                tagged_parts.append(sentence)

        result = " ".join(tagged_parts)
        logger.info(f"TagClassifier: '{text[:40]}' → '{result[:60]}' ({tags_used} tags)")
        return result

    except Exception as e:
        logger.warning(f"TagClassifier failed (using raw text): {e}")
        return text
