import json
import tempfile
import unittest
from pathlib import Path

from src.utils.visual_analysis import (
    DEFAULT_OLLAMA_VLM_MODEL,
    _boolish,
    _compress_camera_timeline,
    _image_to_base64_jpeg,
    _is_ollama_vlm_model,
    _metadata_rollup,
    _ollama_model_name,
    _parse_vlm_json,
    _select_vlm_keyframes,
    _shot_size_from_box,
    read_visual_sidecar,
    visual_sidecar_path,
)


class VisualAnalysisTests(unittest.TestCase):
    def test_visual_sidecar_path_uses_visual_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = visual_sidecar_path("BTM/Edit:Day", "abc/123", root=tmp)
            self.assertEqual(path, Path(tmp) / "BTM-Edit-Day" / "abc_123_visual.json")

    def test_read_visual_sidecar_strips_frames_by_default(self):
        payload = {
            "schema_version": 1,
            "media_id": "m1",
            "clip_name": "clip.mov",
            "file_path": "/tmp/clip.mov",
            "fps": 24,
            "duration_frames": 120,
            "sampled_every_n_frames": 5,
            "proxy_width": 640,
            "analyzed_at": "2026-05-31T00:00:00Z",
            "tier": "fast",
            "models": {},
            "frames": [{"f": 0}],
            "events": [{"type": "high_motion", "start": 0, "peak": 5, "end": 10}],
            "expression": {"frames": [{"f": 0, "label": "smiling"}], "events": [], "dominant": "smiling"},
            "objects": {"frames": [{"f": 0, "classes": ["book"]}], "per_clip": ["book"]},
            "shot_size": {"frames": [{"f": 0, "size": "MS"}], "dominant": "MS"},
            "camera": {"timeline": [{"start": 0, "end": 5, "move": "static"}]},
            "vlm": {"status": "skipped"},
            "metadata_rollup": {"shot_size": "MS", "keywords": ["book"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visual.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            compact = read_visual_sidecar(path)
            self.assertNotIn("frames", compact)
            self.assertNotIn("frames", compact["objects"])
            full = read_visual_sidecar(path, include_frames=True)
            self.assertEqual(full["frames"], [{"f": 0}])
            self.assertEqual(full["objects"]["frames"][0]["classes"], ["book"])

    def test_shot_size_and_rollup(self):
        self.assertEqual(_shot_size_from_box([0.2, 0.1, 0.8, 0.9]), "CU")
        self.assertEqual(_shot_size_from_box([0.2, 0.2, 0.8, 0.55]), "MS")
        rollup = _metadata_rollup(
            pose_events=[{"type": "point"}],
            expression_events=[{"expression": "animated"}],
            objects={"per_clip": ["book", "cup"]},
            shot_size={"dominant": "MS"},
            camera={"timeline": [{"move": "push_in"}, {"move": "push_in"}, {"move": "static"}]},
        )
        self.assertEqual(rollup["shot_size"], "MS")
        self.assertEqual(rollup["camera"], "push_in")
        self.assertIn("animated", rollup["keywords"])
        self.assertIn("book", rollup["keywords"])

    def test_camera_timeline_compresses_adjacent_moves(self):
        compressed = _compress_camera_timeline([
            {"start": 0, "end": 5, "move": "static", "magnitude": 0.1},
            {"start": 5, "end": 10, "move": "static", "magnitude": 0.2},
            {"start": 10, "end": 15, "move": "pan_left", "magnitude": 0.8},
        ])
        self.assertEqual(len(compressed), 2)
        self.assertEqual(compressed[0]["end"], 10)
        self.assertEqual(compressed[0]["magnitude"], 0.2)

    def test_deep_keyframe_selection_prioritizes_events(self):
        frames = _select_vlm_keyframes(
            duration_frames=300,
            sample_every_n=5,
            pose_events=[{"peak": 100}, {"start": 150}],
            expression_events=[{"peak": 175}],
            camera={"timeline": [{"start": 0, "end": 60}, {"start": 60, "end": 120}]},
            max_keyframes=4,
        )
        self.assertIn(100, frames)
        self.assertIn(150, frames)
        self.assertLessEqual(len(frames), 4)

    def test_parse_vlm_json_extracts_embedded_json(self):
        parsed = _parse_vlm_json('Here: {"shot_type":"static","story_beat":true,"keywords":["podium"]}')
        self.assertEqual(parsed["shot_type"], "static")
        self.assertTrue(parsed["story_beat"])
        self.assertTrue(_boolish(parsed["story_beat"]))
        self.assertFalse(_boolish("This is a prose explanation, not a boolean."))

    def test_ollama_model_prefixes(self):
        self.assertEqual(DEFAULT_OLLAMA_VLM_MODEL, "ollama:qwen3-vl:8b")
        self.assertTrue(_is_ollama_vlm_model("ollama:qwen3-vl:8b"))
        self.assertEqual(_ollama_model_name("ollama:qwen3-vl:8b"), "qwen3-vl:8b")
        self.assertEqual(_ollama_model_name("ollama/qwen3-vl:8b"), "qwen3-vl:8b")
        self.assertEqual(_ollama_model_name("ollama://qwen3-vl:8b"), "qwen3-vl:8b")

    def test_image_to_base64_jpeg_returns_ascii(self):
        from PIL import Image

        encoded = _image_to_base64_jpeg(Image.new("RGB", (8, 8), "white"))
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 20)
        encoded.encode("ascii")


if __name__ == "__main__":
    unittest.main()
