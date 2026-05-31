import tempfile
import unittest
from pathlib import Path

from src.utils.motion_analysis import (
    add_motion_energy,
    build_pose_pointer,
    extract_motion_events,
    motion_sidecar_path,
    read_motion_sidecar,
    summarize_motion_events,
    write_json_atomic,
)


def _frame(f, *, wrist_y=0.70, nose_x=0.50, energy=None):
    frame = {
        "f": f,
        "wrist_l": [0.42, wrist_y],
        "wrist_r": [0.58, wrist_y],
        "elbow_l": [0.40, 0.55],
        "elbow_r": [0.60, 0.55],
        "shoulder_l": [0.43, 0.40],
        "shoulder_r": [0.57, 0.40],
        "nose": [nose_x, 0.30],
        "head_yaw": round(((nose_x - 0.50) / 0.14) * 45.0, 2),
        "conf": 0.95,
    }
    if energy is not None:
        frame["motion_energy"] = energy
    return frame


class MotionAnalysisTests(unittest.TestCase):
    def test_sidecar_path_and_pointer_use_project_and_media_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = motion_sidecar_path("BTM/Edit:Day", "abc/123", root=tmp)
            self.assertEqual(path, Path(tmp) / "BTM-Edit-Day" / "abc_123_pose.json")
            pointer = build_pose_pointer(
                "BTM/Edit:Day",
                "abc/123",
                event_count=14,
                model="yolo11n-pose",
                analyzed_at="2026-05-30T18:00:00Z",
            )
            self.assertEqual(pointer, "2026-05-30|events:14|yolo11n-pose|sidecar:BTM-Edit-Day/abc_123_pose.json")

    def test_read_motion_sidecar_keeps_frames_optional(self):
        payload = {
            "schema_version": 1,
            "media_id": "m1",
            "clip_name": "clip.mov",
            "file_path": "/tmp/clip.mov",
            "fps": 23.976,
            "duration_frames": 100,
            "sampled_every_n_frames": 3,
            "proxy_width": 640,
            "analyzed_at": "2026-05-30T18:00:00Z",
            "model": "yolo11n-pose",
            "person_count_mode": "single",
            "frames": [_frame(0)],
            "events": [{"type": "still", "start": 0, "peak": None, "end": 9, "energy": 0.02}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.json"
            write_json_atomic(path, payload)
            compact = read_motion_sidecar(path)
            self.assertNotIn("frames", compact)
            full = read_motion_sidecar(path, include_frames=True)
            self.assertEqual(len(full["frames"]), 1)

    def test_extract_motion_events_from_synthetic_pose_track(self):
        frames = [
            _frame(0),
            _frame(3),
            _frame(6),
            _frame(9, wrist_y=0.56),
            _frame(12, wrist_y=0.20),
            _frame(15, wrist_y=0.10),
            _frame(18, wrist_y=0.28),
            _frame(21),
            _frame(24, nose_x=0.58),
            _frame(27, nose_x=0.59),
            _frame(30, nose_x=0.60),
            _frame(33),
            _frame(36),
            _frame(39),
        ]
        add_motion_energy(frames)
        events = extract_motion_events(frames, fps=24.0, sample_every_n=3)
        summary = summarize_motion_events(events)
        self.assertGreaterEqual(summary.get("hand_raise", 0), 1)
        self.assertGreaterEqual(summary.get("head_turn_right", 0), 1)
        self.assertGreaterEqual(summary.get("high_motion", 0), 1)
        self.assertGreaterEqual(summary.get("still", 0), 1)
        hand_raise = next(event for event in events if event["type"] == "hand_raise")
        self.assertEqual(hand_raise["which"], "both")
        self.assertLessEqual(hand_raise["start"], hand_raise["peak"])
        self.assertLessEqual(hand_raise["peak"], hand_raise["end"])


if __name__ == "__main__":
    unittest.main()
