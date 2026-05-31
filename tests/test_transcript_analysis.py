import json
import tempfile
import unittest
from pathlib import Path

from src.utils.transcript_analysis import (
    _engine_name,
    _parakeet_lines_to_source_frames,
    frame_to_srt_timestamp,
    read_transcript_sidecar,
    transcript_keywords,
    transcript_sidecar_path,
    transcript_srt_path,
    transcript_to_srt,
)


class FakeToken:
    def __init__(self, text, start, end, confidence=0.9):
        self.text = text
        self.start = start
        self.end = end
        self.duration = end - start
        self.confidence = confidence


class FakeSentence:
    def __init__(self, text, tokens):
        self.text = text
        self.tokens = tokens
        self.start = tokens[0].start
        self.end = tokens[-1].end
        self.confidence = 0.8


class TranscriptAnalysisTests(unittest.TestCase):
    def test_transcript_paths_are_project_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = transcript_sidecar_path("BTM/Edit:Day", "abc/123", root=tmp)
            srt = transcript_srt_path("BTM/Edit:Day", "abc/123", "Clip A.mov", root=tmp)
            self.assertEqual(sidecar, Path(tmp) / "BTM-Edit-Day" / "abc_123_transcript.json")
            self.assertEqual(srt, Path(tmp) / "BTM-Edit-Day" / "Subtitles" / "Clip A_abc_123.srt")

    def test_parakeet_seconds_convert_to_source_frames(self):
        sentence = FakeSentence(
            "hello world",
            [FakeToken("hello", 0.5, 0.75), FakeToken("world", 0.75, 1.0)],
        )
        lines = _parakeet_lines_to_source_frames([sentence], fps=24, duration_frames=100)
        self.assertEqual(lines[0]["start"], 12)
        self.assertEqual(lines[0]["end"], 24)
        self.assertEqual(lines[0]["words"][0]["w"], "hello")
        self.assertEqual(lines[0]["words"][1]["start"], 18)

    def test_srt_export_uses_clip_relative_time(self):
        payload = {
            "fps": 24,
            "lines": [{"text": "hello world", "start": 12, "end": 36}],
        }
        srt = transcript_to_srt(payload)
        self.assertIn("00:00:00,500 --> 00:00:01,500", srt)
        self.assertIn("hello world", srt)
        self.assertEqual(frame_to_srt_timestamp(24, 24), "00:00:01,000")

    def test_read_transcript_sidecar_can_strip_words(self):
        payload = {
            "schema_version": 1,
            "media_id": "m1",
            "clip_name": "clip.mov",
            "file_path": "/tmp/clip.mov",
            "fps": 24,
            "duration_frames": 48,
            "engine": "parakeet",
            "model": "model",
            "language": "en",
            "analyzed_at": "2026-05-31T00:00:00Z",
            "text": "hello world",
            "lines": [{"text": "hello world", "start": 0, "end": 24, "words": [{"w": "hello"}]}],
            "metadata_rollup": {"word_count": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            compact = read_transcript_sidecar(path, include_words=False)
            self.assertNotIn("words", compact["lines"][0])
            full = read_transcript_sidecar(path, include_words=True)
            self.assertEqual(full["lines"][0]["words"][0]["w"], "hello")

    def test_keywords_and_engine_name(self):
        keywords = transcript_keywords("Faith faith sermon video editing because the faith matters")
        self.assertEqual(keywords[0], "faith")
        self.assertEqual(_engine_name("mlx-community/parakeet-tdt-0.6b-v3"), "parakeet-tdt-0.6b-v3")


if __name__ == "__main__":
    unittest.main()
