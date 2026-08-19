from types import SimpleNamespace

import pytest

from whisperlivekit.conference import (
    ConferenceTranslator,
    detect_conference_language,
    normalize_conference_language,
)
from whisperlivekit.timed_objects import ASRToken, State
from whisperlivekit.tokens_alignment import TokensAlignment
from whisperlivekit.web.web_interface import get_conference_ui_html


class FakeTranslationBackend:
    def __init__(self):
        self.calls = []

    async def translate(self, text, source_language, target_languages):
        self.calls.append((text, source_language, target_languages))
        return {language: f"{language}:{text}" for language in target_languages}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("zh-CN", "zh"),
        ("cmn", "zh"),
        ("Chinese", "zh"),
        ("en-US", "en"),
        ("English", "en"),
        ("ja-JP", "ja"),
        ("Japanese", "ja"),
        ("fr", None),
        (None, None),
    ],
)
def test_conference_language_aliases_are_normalized(value, expected):
    assert normalize_conference_language(value) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I think agents will change how companies work.", "en"),
        ("人工知能は企業の働き方を変えます。", "ja"),
        ("我认为人工智能会改变企业的工作方式。", "zh"),
    ],
)
def test_conference_language_fallback_handles_english_japanese_and_chinese(text, expected):
    assert detect_conference_language(text) == expected


@pytest.mark.asyncio
async def test_conference_translation_preserves_source_and_requests_only_other_tracks():
    backend = FakeTranslationBackend()
    translator = ConferenceTranslator(backend)

    result = await translator.translate("大家好", source_language="zh-CN")

    assert result.source_language == "zh"
    assert result.subtitles == {
        "zh": "大家好",
        "en": "en:大家好",
        "ja": "ja:大家好",
    }
    assert backend.calls == [("大家好", "zh", ["en", "ja"])]


@pytest.mark.asyncio
async def test_conference_translation_uses_text_fallback_when_asr_has_no_language():
    backend = FakeTranslationBackend()
    translator = ConferenceTranslator(backend)

    result = await translator.translate("今日はよろしくお願いします。")

    assert result.source_language == "ja"
    assert result.subtitles["ja"] == "今日はよろしくお願いします。"
    assert backend.calls[0][1:] == ("ja", ["zh", "en"])


def test_conference_alignment_starts_a_new_line_when_detected_language_changes():
    state = State(
        new_tokens=[
            ASRToken(0.0, 0.5, "Hello ", detected_language="en"),
            ASRToken(0.5, 1.0, "world.", detected_language="en"),
            ASRToken(1.1, 1.5, "大家", detected_language="zh"),
            ASRToken(1.5, 2.0, "好。", detected_language="zh"),
        ]
    )
    args = SimpleNamespace(diarization=False, conference_mode=True)
    alignment = TokensAlignment(state, args, sep="")

    alignment.update()
    lines, _, _ = alignment.get_lines(audio_time=2.0)

    assert [(line.text, line.detected_language) for line in lines] == [
        ("Hello world.", "en"),
        ("大家好。", "zh"),
    ]


def test_conference_ui_contains_three_tracks_and_meeting_controls():
    html = get_conference_ui_html()

    for element_id in (
        'id="track-zh"',
        'id="track-en"',
        'id="track-ja"',
        'id="microphone-select"',
        'id="record-button"',
        'id="fullscreen-button"',
        'id="font-size"',
    ):
        assert element_id in html
    assert "/conference/translate" in html
