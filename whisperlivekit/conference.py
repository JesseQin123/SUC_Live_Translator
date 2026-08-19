"""Local three-track conference translation.

This module deliberately sits beside the streaming ASR pipeline.  WhisperLiveKit
continues to own microphone ingestion and transcript stability; conference mode
only translates committed text into the two missing languages.
"""

import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CONFERENCE_LANGUAGES = ("zh", "en", "ja")
_NLLB_CODES = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
}
_LANGUAGE_ALIASES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "cmn": "zh",
    "cmn-hans": "zh",
    "chinese": "zh",
    "中文": "zh",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "ja-jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "日本語": "ja",
}
_JAPANESE_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HAN_SCRIPT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_SCRIPT = re.compile(r"[A-Za-z]")


def normalize_conference_language(value: Optional[str]) -> Optional[str]:
    """Normalize common ASR language identifiers to zh, en, or ja."""
    if not value:
        return None
    key = str(value).strip().replace("_", "-").lower()
    return _LANGUAGE_ALIASES.get(key)


def detect_conference_language(text: str) -> str:
    """Small fallback for backends that do not emit language metadata.

    Kana unambiguously identifies Japanese.  Han-only text is treated as
    Chinese, while Latin text is treated as English.  An ASR-provided language
    is always preferred by :class:`ConferenceTranslator`.
    """
    if _JAPANESE_SCRIPT.search(text or ""):
        return "ja"
    if _HAN_SCRIPT.search(text or ""):
        return "zh"
    if _LATIN_SCRIPT.search(text or ""):
        return "en"
    return "en"


@dataclass(frozen=True)
class ConferenceTranslation:
    source_language: str
    subtitles: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "source_language": self.source_language,
            "subtitles": self.subtitles,
        }


class TranslationSuperseded(RuntimeError):
    """Raised when a newer live-caption request replaces queued work."""


class ConferenceTranslator:
    """Route a committed phrase to all three conference subtitle tracks."""

    def __init__(self, backend):
        self.backend = backend

    async def translate(
        self,
        text: str,
        source_language: Optional[str] = None,
    ) -> ConferenceTranslation:
        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Conference translation text cannot be empty.")

        source = normalize_conference_language(source_language)
        if source is None:
            source = detect_conference_language(normalized_text)
        targets = [language for language in CONFERENCE_LANGUAGES if language != source]
        translated = await self.backend.translate(normalized_text, source, targets)

        missing = [language for language in targets if not translated.get(language)]
        if missing:
            raise RuntimeError("Translation backend omitted track(s): " + ", ".join(missing))

        subtitles = {source: normalized_text, **translated}
        return ConferenceTranslation(
            source_language=source,
            subtitles={language: subtitles[language] for language in CONFERENCE_LANGUAGES},
        )


class NllbConferenceBackend:
    """Generate two NLLB targets while sharing one locally loaded model."""

    def __init__(self, translation_model):
        self.model = translation_model
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._latest_request_id = 0

    async def translate(self, text, source_language, target_languages):
        with self._request_lock:
            self._latest_request_id += 1
            request_id = self._latest_request_id
        return await asyncio.to_thread(
            self._translate_sync,
            text,
            source_language,
            target_languages,
            request_id,
        )

    def _ensure_latest(self, request_id):
        with self._request_lock:
            if request_id != self._latest_request_id:
                raise TranslationSuperseded("A newer conference subtitle replaced this request.")

    def _translate_sync(self, text, source_language, target_languages, request_id):
        source_code = _NLLB_CODES[source_language]
        target_codes = [_NLLB_CODES[language] for language in target_languages]
        tokenizer = self.model.get_tokenizer(source_code)
        with self._lock:
            # Coalesce browser updates while an older inference is running. At
            # most one stale translation can consume model time; queued stale
            # requests exit here instead of building an uncancellable backlog.
            self._ensure_latest(request_id)
            if self.model.backend_type == "ctranslate2":
                outputs = self._translate_ctranslate2(text, tokenizer, target_codes, request_id)
            else:
                outputs = self._translate_transformers(text, tokenizer, target_codes, request_id)
        return dict(zip(target_languages, outputs))

    def _translate_transformers(self, text, tokenizer, target_codes, request_id):
        import torch

        model = self.model.translator
        device = next(model.parameters()).device
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        outputs = []
        with torch.inference_mode():
            for target_code in target_codes:
                self._ensure_latest(request_id)
                target_id = tokenizer.convert_tokens_to_ids(target_code)
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    max_new_tokens=256,
                    num_beams=1,
                )
                outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
        return outputs

    def _translate_ctranslate2(self, text, tokenizer, target_codes, request_id):
        self._ensure_latest(request_id)
        encoded = tokenizer(text, truncation=True, max_length=512)
        source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        results = self.model.translator.translate_batch(
            [source_tokens for _ in target_codes],
            target_prefix=[[target_code] for target_code in target_codes],
            beam_size=1,
            max_batch_size=len(target_codes),
        )
        outputs = []
        for result in results:
            hypothesis = result.hypotheses[0]
            if hypothesis and hypothesis[0] in target_codes:
                hypothesis = hypothesis[1:]
            token_ids = tokenizer.convert_tokens_to_ids(hypothesis)
            outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
        return outputs


def build_conference_translator(config) -> ConferenceTranslator:
    """Load the local NLLB model needed by conference mode."""
    try:
        from nllw import load_model
    except ImportError as exc:
        raise ImportError("Conference mode requires NLLW. Install `whisperlivekit[translation]`.") from exc

    backend_name = getattr(config, "nllb_backend", "transformers")
    model = load_model(
        [_NLLB_CODES[language] for language in CONFERENCE_LANGUAGES],
        nllb_backend=backend_name,
        nllb_size=getattr(config, "nllb_size", "600M"),
    )

    if backend_name == "transformers":
        try:
            import torch

            if torch.backends.mps.is_available():
                model.translator = model.translator.to("mps")
                model.device = "mps"
                logger.info("Conference NLLB translation is using Apple MPS.")
        except (AttributeError, RuntimeError):
            logger.warning(
                "NLLB could not move to Apple MPS; continuing on CPU.",
                exc_info=True,
            )

    return ConferenceTranslator(NllbConferenceBackend(model))
