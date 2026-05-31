import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils.visual_analysis import (
    DEFAULT_OLLAMA_VLM_MODEL,
    OLLAMA_VLM_JSON_SCHEMA,
    _boolish,
    _compress_camera_timeline,
    _image_to_base64_jpeg,
    _is_ollama_vlm_model,
    _merge_vlm_metadata_rollup,
    _metadata_rollup,
    _next_sparse_sample_frame,
    _ollama_model_name,
    _ollama_num_ctx,
    _parse_vlm_json,
    _select_vlm_keyframes,
    _shot_size_from_box,
    _vlm_camera_context,
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
            "object_sampled_every_n_frames": 120,
            "object_sampled_every_seconds": 5.0,
            "batch_size": 1,
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
            self.assertEqual(full["object_sampled_every_n_frames"], 120)
            self.assertEqual(full["object_sampled_every_seconds"], 5.0)
            self.assertEqual(full["batch_size"], 1)

    def test_sparse_object_cadence_uses_source_frames(self):
        self.assertEqual(_next_sparse_sample_frame(0, 24), 24)
        self.assertEqual(_next_sparse_sample_frame(32, 24), 56)

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

    def test_ollama_num_ctx_defaults_small_for_keyframes(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_ollama_num_ctx(), 4096)
        with mock.patch.dict("os.environ", {"RESOLVE_MCP_OLLAMA_NUM_CTX": "2048"}):
            self.assertEqual(_ollama_num_ctx(), 2048)
        with mock.patch.dict("os.environ", {"RESOLVE_MCP_OLLAMA_NUM_CTX": "32"}):
            self.assertEqual(_ollama_num_ctx(), 512)

    def test_ollama_vlm_schema_rejects_extra_keys(self):
        self.assertIn("shot_type", OLLAMA_VLM_JSON_SCHEMA["required"])
        self.assertFalse(OLLAMA_VLM_JSON_SCHEMA["additionalProperties"])

    def test_vlm_camera_context_is_compact(self):
        context = _vlm_camera_context(
            {
                "timeline": [
                    {"start": 0, "end": 50, "move": "static"},
                    {"start": 50, "end": 100, "move": "static"},
                    {"start": 100, "end": 200, "move": "push_in"},
                    {"start": 500, "end": 600, "move": "pan_left"},
                ]
            },
            120,
        )
        self.assertEqual(context["dominant_moves"][0], "static")
        self.assertLessEqual(len(context["nearby_motion"]), 5)
        self.assertNotIn("timeline", context)

    def test_vlm_metadata_rollup_enriches_search_projection(self):
        merged = _merge_vlm_metadata_rollup(
            {"description": "CU; gesturing", "keywords": ["cu"], "tone": "unknown"},
            {
                "status": "analyzed",
                "shot_type": "talking-head",
                "description": "Speaker gesturing in a modern office",
                "keyframes": [
                    {"keywords": ["speaker", "office"], "tone": "professional"},
                    {"keywords": ["gesture"], "tone": "professional"},
                ],
            },
        )
        self.assertEqual(merged["description"], "Speaker gesturing in a modern office")
        self.assertEqual(merged["tone"], "professional")
        self.assertIn("talking-head", merged["keywords"])
        self.assertIn("office", merged["keywords"])

    def test_image_to_base64_jpeg_returns_ascii(self):
        from PIL import Image

        encoded = _image_to_base64_jpeg(Image.new("RGB", (8, 8), "white"))
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 20)
        encoded.encode("ascii")


if __name__ == "__main__":
    unittest.main()
