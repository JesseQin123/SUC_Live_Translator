import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN_INTEGRATION = os.environ.get("WLK_RUN_CONFERENCE_INTEGRATION") == "1"
MODEL_DIR = os.environ.get("WLK_FUNASR_MODEL_DIR")

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION or platform.system() != "Darwin" or not MODEL_DIR,
    reason=(
        "set WLK_RUN_CONFERENCE_INTEGRATION=1 and WLK_FUNASR_MODEL_DIR "
        "on macOS to run the real three-language conference test"
    ),
)


def _synthesize_turns(directory: Path):
    turns = [
        ("zh", "Tingting", "大家好，我们今天讨论人工智能的发展计划。"),
        (
            "en",
            "Samantha",
            "Hello everyone, today we will discuss the artificial intelligence project.",
        ),
        ("ja", "Kyoko", "皆さんこんにちは。今日は人工知能の計画について話します。"),
    ]
    paths = []
    for language, voice, text in turns:
        path = directory / f"turn-{language}.aiff"
        subprocess.run(
            ["say", "-v", voice, "-o", str(path), text],
            check=True,
            capture_output=True,
        )
        paths.append((language, path))
    return paths


@pytest.mark.asyncio
async def test_real_local_conference_switches_languages_and_translates_all_tracks(tmp_path):
    pytest.importorskip("funasr")
    pytest.importorskip("nllw")
    from whisperlivekit.conference import build_conference_translator
    from whisperlivekit.test_harness import TestHarness

    turns = _synthesize_turns(tmp_path)
    async with TestHarness(
        backend="funasr",
        backend_policy="localagreement",
        model_dir=MODEL_DIR,
        lan="auto",
        conference_mode=True,
        pause_segmentation_seconds=0.8,
        warmup_file=None,
    ) as harness:
        for _, audio_path in turns:
            await harness.feed(str(audio_path), speed=0, chunk_duration=0.5)
            await harness.pause(6.0, speed=0)
        result = await harness.finish(timeout=90)

    lines = [line for line in result.speech_lines if line.get("detected_language")]
    language_order = []
    for line in lines:
        if not language_order or language_order[-1] != line["detected_language"]:
            language_order.append(line["detected_language"])
    assert language_order == ["zh", "en", "ja"]

    translator = build_conference_translator(SimpleNamespace(nllb_backend="transformers", nllb_size="600M"))
    for language in language_order:
        line = next(item for item in lines if item["detected_language"] == language)
        translated = await translator.translate(line["text"], language)
        assert translated.source_language == language
        assert translated.subtitles[language] == line["text"].strip()
        assert all(translated.subtitles[target] for target in ("zh", "en", "ja"))
