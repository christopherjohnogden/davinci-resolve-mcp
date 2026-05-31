"""YOLO pose motion-analysis helpers for Resolve media pool clips.

This module is intentionally source-safe: it reads source media through ffmpeg
and writes only project-level sidecar JSON files under ``~/Resolve_Analysis``.
No source media is modified or transcoded.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


POSE_SCHEMA_VERSION = 1
DEFAULT_MOTION_ROOT_NAME = "Resolve_Analysis"
POSE_METADATA_KEY = "pose_analysis"
POSE_KEYPOINTS = (
    "wrist_l",
    "wrist_r",
    "elbow_l",
    "elbow_r",
    "shoulder_l",
    "shoulder_r",
    "nose",
)
_COCO_TO_POSE_KEY = {
    0: "nose",
    5: "shoulder_l",
    6: "shoulder_r",
    7: "elbow_l",
    8: "elbow_r",
    9: "wrist_l",
    10: "wrist_r",
}


def resolve_pose_model_path(model: str) -> str:
    """Prefer local model weights so analysis does not depend on GitHub at run time."""
    requested = str(model or "yolo11n-pose").strip() or "yolo11n-pose"
    expanded = Path(requested).expanduser()
    if expanded.exists():
        return str(expanded)

    candidates = []
    if not expanded.suffix and os.sep not in requested and (not os.altsep or os.altsep not in requested):
        candidates.append(f"{requested}.pt")
    candidates.append(requested)

    repo_root = Path(__file__).resolve().parents[2]
    search_roots = [Path.cwd(), repo_root, Path.home()]
    seen = set()
    for root in search_roots:
        for candidate in candidates:
            path = (root / candidate).expanduser()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return str(path)
    return requested


def _safe_segment(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace(os.sep, "-")
    if os.altsep:
        text = text.replace(os.altsep, "-")
    text = re.sub(r"[\x00-\x1f:]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or fallback


def motion_root(root: Optional[str] = None) -> Path:
    return Path(root).expanduser() if root else Path.home() / DEFAULT_MOTION_ROOT_NAME


def motion_project_name(project_name: Any) -> str:
    return _safe_segment(project_name, "Project")


def motion_media_id(media_id: Any) -> str:
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(media_id or "").strip()).strip("._")
        or "unknown"
    )


def motion_project_dir(project_name: Any, root: Optional[str] = None) -> Path:
    return motion_root(root) / motion_project_name(project_name)


def motion_sidecar_path(project_name: Any, media_id: Any, root: Optional[str] = None) -> Path:
    return motion_project_dir(project_name, root) / f"{motion_media_id(media_id)}_pose.json"


def motion_sidecar_rel(project_name: Any, media_id: Any) -> str:
    return f"{motion_project_name(project_name)}/{motion_media_id(media_id)}_pose.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_pose_pointer(
    project_name: Any,
    media_id: Any,
    event_count: int,
    model: str,
    analyzed_at: str,
) -> str:
    date = str(analyzed_at or utc_now_iso()).split("T", 1)[0]
    return f"{date}|events:{int(event_count)}|{model}|sidecar:{motion_sidecar_rel(project_name, media_id)}"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_motion_sidecar(path: Path, include_frames: bool = False) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    out = {
        "analyzed": True,
        "schema_version": data.get("schema_version"),
        "media_id": data.get("media_id"),
        "clip_name": data.get("clip_name"),
        "file_path": data.get("file_path"),
        "fps": data.get("fps"),
        "duration_frames": data.get("duration_frames"),
        "sampled_every_n_frames": data.get("sampled_every_n_frames"),
        "proxy_width": data.get("proxy_width"),
        "analyzed_at": data.get("analyzed_at"),
        "model": data.get("model"),
        "person_count_mode": data.get("person_count_mode"),
        "events": data.get("events") or [],
    }
    if include_frames:
        out["frames"] = data.get("frames") or []
    return out


def summarize_motion_events(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for event in events or []:
        typ = str(event.get("type") or "unknown")
        summary[typ] = summary.get(typ, 0) + 1
    return dict(sorted(summary.items()))


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        text = str(value).strip()
        if not text:
            return default
        # Resolve values are often "23.976 fps" or "24".
        return float(text.split()[0])
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(round(float(str(value).strip().split()[0])))
    except (TypeError, ValueError):
        return default


def _run_json(cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "command failed").strip())
    return json.loads(result.stdout or "{}")


def ffprobe_video_info(file_path: str) -> Dict[str, Any]:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed or is not on PATH")
    data = _run_json([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        file_path,
    ])
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe could not find a video stream")
    stream = streams[0]
    fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")) or 0.0
    duration_seconds = parse_float(stream.get("duration"), 0.0)
    nb_frames = parse_int(stream.get("nb_frames"), 0)
    if not nb_frames and duration_seconds and fps:
        nb_frames = int(round(duration_seconds * fps))
    return {
        "width": parse_int(stream.get("width")),
        "height": parse_int(stream.get("height")),
        "fps": fps,
        "duration_frames": nb_frames,
        "duration_seconds": duration_seconds,
    }


def _parse_rate(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text == "0/0":
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def proxy_dimensions(source_width: int, source_height: int, proxy_width: int) -> Tuple[int, int]:
    source_width = max(1, int(source_width or proxy_width or 640))
    source_height = max(1, int(source_height or proxy_width or 640))
    width = max(64, int(proxy_width or 640))
    height = max(2, int(round(source_height * (width / source_width))))
    if height % 2:
        height += 1
    return width, height


def _frame_chunks(proc: subprocess.Popen, frame_bytes: int):
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(frame_bytes)
        if not chunk:
            break
        if len(chunk) != frame_bytes:
            break
        yield chunk


def run_yolo_pose_analysis(
    *,
    file_path: str,
    media_id: str,
    clip_name: str,
    fps: float,
    duration_frames: int,
    sample_every_n: int = 3,
    proxy_width: int = 640,
    model: str = "yolo11n-pose",
) -> Dict[str, Any]:
    """Run ffmpeg sampling and YOLO pose inference, returning the sidecar payload."""
    if not file_path or not os.path.exists(file_path):
        raise RuntimeError(f"Media file is offline or missing: {file_path}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or is not on PATH")
    sample_every_n = max(1, int(sample_every_n or 3))
    proxy_width = max(64, int(proxy_width or 640))

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("numpy is required for YOLO pose analysis") from exc
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO pose analysis. Install it in the MCP Python environment."
        ) from exc

    probe = ffprobe_video_info(file_path)
    source_width = probe.get("width") or proxy_width
    source_height = probe.get("height") or proxy_width
    if not fps:
        fps = float(probe.get("fps") or 0.0)
    if not duration_frames:
        duration_frames = int(probe.get("duration_frames") or 0)
    width, height = proxy_dimensions(source_width, source_height, proxy_width)
    frame_bytes = width * height * 3

    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-vf",
        f"select='not(mod(n,{sample_every_n}))',scale={width}:{height}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    resolved_model = resolve_pose_model_path(model)
    yolo = YOLO(resolved_model)
    frames: List[Dict[str, Any]] = []
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for sample_index, chunk in enumerate(_frame_chunks(proc, frame_bytes)):
            image = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            frame = _pose_frame_from_yolo(
                yolo=yolo,
                image=image,
                true_frame=sample_index * sample_every_n,
                width=width,
                height=height,
            )
            frames.append(frame)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg exited with status {returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()

    add_motion_energy(frames)
    events = extract_motion_events(frames, fps=fps, sample_every_n=sample_every_n)
    analyzed_at = utc_now_iso()
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "media_id": media_id,
        "clip_name": clip_name,
        "file_path": file_path,
        "fps": fps,
        "duration_frames": duration_frames,
        "sampled_every_n_frames": sample_every_n,
        "proxy_width": width,
        "analyzed_at": analyzed_at,
        "model": model,
        "model_path": resolved_model if os.path.exists(resolved_model) else None,
        "person_count_mode": "single",
        "frames": frames,
        "events": events,
    }


def _pose_frame_from_yolo(
    *,
    yolo: Any,
    image: Any,
    true_frame: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    result_list = yolo(image, verbose=False)
    result = result_list[0] if result_list else None
    frame: Dict[str, Any] = {"f": int(true_frame), "conf": 0.0}
    if result is None or getattr(result, "keypoints", None) is None:
        frame["low_conf"] = True
        return frame

    keypoints = result.keypoints
    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)
    boxes = getattr(result, "boxes", None)
    if xy is None:
        frame["low_conf"] = True
        return frame

    try:
        xy_arr = xy.cpu().numpy()
        conf_arr = conf.cpu().numpy() if conf is not None else None
    except Exception:
        xy_arr = xy.numpy()
        conf_arr = conf.numpy() if conf is not None else None
    if len(xy_arr) == 0:
        frame["low_conf"] = True
        return frame

    person_index = 0
    box_conf = None
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        try:
            box_conf_arr = boxes.conf.cpu().numpy()
            box_xyxy_arr = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else None
        except Exception:
            box_conf_arr = boxes.conf.numpy()
            box_xyxy_arr = boxes.xyxy.numpy() if getattr(boxes, "xyxy", None) is not None else None
        if len(box_conf_arr):
            person_index = int(
                max(range(len(box_conf_arr)), key=lambda idx: float(box_conf_arr[idx]))
            )
            box_conf = float(box_conf_arr[person_index])
            if box_xyxy_arr is not None and len(box_xyxy_arr) > person_index:
                box = box_xyxy_arr[person_index]
                frame["person_box"] = [
                    round(float(box[0]) / width, 6),
                    round(float(box[1]) / height, 6),
                    round(float(box[2]) / width, 6),
                    round(float(box[3]) / height, 6),
                ]

    kp_xy = xy_arr[person_index]
    kp_conf = conf_arr[person_index] if conf_arr is not None and len(conf_arr) > person_index else None
    valid_conf_values: List[float] = []
    for coco_idx, key in _COCO_TO_POSE_KEY.items():
        if coco_idx >= len(kp_xy):
            continue
        point_conf = float(kp_conf[coco_idx]) if kp_conf is not None and coco_idx < len(kp_conf) else 1.0
        x, y = kp_xy[coco_idx]
        if point_conf <= 0 or float(x) <= 0 or float(y) <= 0:
            continue
        frame[key] = [round(float(x) / width, 6), round(float(y) / height, 6)]
        valid_conf_values.append(point_conf)

    if box_conf is not None:
        frame["conf"] = round(max(0.0, min(1.0, box_conf)), 4)
    elif valid_conf_values:
        frame["conf"] = round(sum(valid_conf_values) / len(valid_conf_values), 4)
    else:
        frame["conf"] = 0.0
    if frame["conf"] < 0.4:
        frame["low_conf"] = True
    frame["head_yaw"] = estimate_head_yaw(frame)
    return frame


def estimate_head_yaw(frame: Dict[str, Any]) -> Optional[float]:
    nose = frame.get("nose")
    shoulder_l = frame.get("shoulder_l")
    shoulder_r = frame.get("shoulder_r")
    if not (
        isinstance(nose, list)
        and isinstance(shoulder_l, list)
        and isinstance(shoulder_r, list)
    ):
        return None
    mid_x = (float(shoulder_l[0]) + float(shoulder_r[0])) / 2.0
    shoulder_width = max(0.05, abs(float(shoulder_r[0]) - float(shoulder_l[0])))
    yaw = ((float(nose[0]) - mid_x) / shoulder_width) * 45.0
    return round(max(-45.0, min(45.0, yaw)), 2)


def _valid_point(frame: Dict[str, Any], key: str) -> Optional[Tuple[float, float]]:
    if frame.get("conf", 0.0) < 0.4:
        return None
    value = frame.get(key)
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def add_motion_energy(frames: List[Dict[str, Any]], scale: float = 14.0) -> None:
    previous: Optional[Dict[str, Any]] = None
    raw: List[float] = []
    for frame in frames:
        if previous is None:
            raw.append(0.0)
            previous = frame if frame.get("conf", 0.0) >= 0.4 else previous
            continue
        distances = []
        for key in POSE_KEYPOINTS:
            a = _valid_point(previous, key)
            b = _valid_point(frame, key)
            if a is None or b is None:
                continue
            distances.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        if distances:
            energy = min(1.0, (sum(distances) / len(distances)) * scale)
            raw.append(energy)
            if frame.get("conf", 0.0) >= 0.4:
                previous = frame
        else:
            raw.append(0.0)
    smoothed = _smooth(raw)
    for frame, energy in zip(frames, smoothed):
        frame["motion_energy"] = round(float(energy), 4)


def _smooth(values: List[float]) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for index in range(len(values)):
        lo = max(0, index - 1)
        hi = min(len(values), index + 2)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def extract_motion_events(
    frames: List[Dict[str, Any]],
    *,
    fps: float,
    sample_every_n: int,
    high_motion_threshold: float = 0.6,
    still_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    min_hold_samples = max(3, int(round((fps or 24.0) / max(1, sample_every_n) * 0.35)))
    events.extend(_energy_events(frames, "high_motion", high_motion_threshold, above=True, min_len=2))
    events.extend(_energy_events(frames, "still", still_threshold, above=False, min_len=min_hold_samples))
    events.extend(_hand_raise_events(frames, min_len=2))
    events.extend(_point_events(frames, min_len=2))
    events.extend(_head_turn_events(frames, min_len=2))
    events.sort(key=lambda event: (event.get("start", 0), event.get("type", "")))
    return events


def _event_from_group(
    group: List[Dict[str, Any]],
    typ: str,
    energy: bool = False,
) -> Dict[str, Any]:
    if energy:
        peak_frame = max(group, key=lambda item: float(item.get("motion_energy") or 0.0))
        out = {
            "type": typ,
            "start": group[0]["f"],
            "peak": peak_frame["f"] if typ != "still" else None,
            "end": group[-1]["f"],
            "energy": round(float(peak_frame.get("motion_energy") or 0.0), 4),
        }
    else:
        out = {
            "type": typ,
            "start": group[0]["f"],
            "peak": group[len(group) // 2]["f"],
            "end": group[-1]["f"],
        }
    return out


def _energy_events(
    frames: List[Dict[str, Any]],
    typ: str,
    threshold: float,
    *,
    above: bool,
    min_len: int,
) -> List[Dict[str, Any]]:
    groups = _groups_where(
        frames,
        lambda frame: (
            float(frame.get("motion_energy") or 0.0) >= threshold
            if above
            else frame.get("conf", 0.0) >= 0.4 and float(frame.get("motion_energy") or 0.0) <= threshold
        ),
        min_len=min_len,
    )
    return [_event_from_group(group, typ, energy=True) for group in groups]


def _groups_where(
    frames: List[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool],
    min_len: int,
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for frame in frames:
        if predicate(frame):
            current.append(frame)
        else:
            if len(current) >= min_len:
                groups.append(current)
            current = []
    if len(current) >= min_len:
        groups.append(current)
    return groups


def _wrist_baseline(frames: List[Dict[str, Any]], wrist_key: str) -> Optional[float]:
    still_values = [
        float(frame[wrist_key][1])
        for frame in frames
        if frame.get("conf", 0.0) >= 0.4
        and isinstance(frame.get(wrist_key), list)
        and float(frame.get("motion_energy") or 0.0) <= 0.2
    ]
    values = still_values or [
        float(frame[wrist_key][1])
        for frame in frames
        if frame.get("conf", 0.0) >= 0.4 and isinstance(frame.get(wrist_key), list)
    ]
    return median(values) if values else None


def _hand_raise_events(frames: List[Dict[str, Any]], min_len: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    baselines = {
        "left": _wrist_baseline(frames, "wrist_l"),
        "right": _wrist_baseline(frames, "wrist_r"),
    }
    side_key = {"left": "wrist_l", "right": "wrist_r"}
    side_groups: Dict[str, List[List[Dict[str, Any]]]] = {}
    for side, baseline in baselines.items():
        if baseline is None:
            side_groups[side] = []
            continue
        key = side_key[side]
        side_groups[side] = _groups_where(
            frames,
            lambda frame, wrist_key=key, base=baseline: (
                frame.get("conf", 0.0) >= 0.4
                and isinstance(frame.get(wrist_key), list)
                and float(frame[wrist_key][1]) < float(base) - 0.12
            ),
            min_len=min_len,
        )

    consumed: Set[Tuple[str, int]] = set()
    for left_group in side_groups.get("left", []):
        matched_index = None
        for index, right_group in enumerate(side_groups.get("right", [])):
            overlaps = (
                int(left_group[0]["f"]) <= int(right_group[-1]["f"])
                and int(right_group[0]["f"]) <= int(left_group[-1]["f"])
            )
            if overlaps:
                matched_index = index
                merged = sorted(left_group + right_group, key=lambda frame: frame["f"])
                event = _hand_raise_group_event(merged, which="both")
                out.append(event)
                consumed.add(("right", index))
                break
        if matched_index is None:
            out.append(_hand_raise_group_event(left_group, which="left"))
    for index, right_group in enumerate(side_groups.get("right", [])):
        if ("right", index) not in consumed:
            out.append(_hand_raise_group_event(right_group, which="right"))
    return out


def _hand_raise_group_event(group: List[Dict[str, Any]], which: str) -> Dict[str, Any]:
    keys = (
        ["wrist_l", "wrist_r"]
        if which == "both"
        else ["wrist_l" if which == "left" else "wrist_r"]
    )

    def score(frame: Dict[str, Any]) -> float:
        ys = [float(frame[key][1]) for key in keys if isinstance(frame.get(key), list)]
        return min(ys) if ys else 1.0

    peak = min(group, key=score)
    confidence = max(float(frame.get("conf") or 0.0) for frame in group)
    return {
        "type": "hand_raise",
        "start": group[0]["f"],
        "peak": peak["f"],
        "end": group[-1]["f"],
        "confidence": round(confidence, 4),
        "which": which,
    }


def _point_events(frames: List[Dict[str, Any]], min_len: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for side, wrist_key, shoulder_key, direction in (
        ("left", "wrist_l", "shoulder_l", -1),
        ("right", "wrist_r", "shoulder_r", 1),
    ):
        groups = _groups_where(
            frames,
            lambda frame, wk=wrist_key, sk=shoulder_key, sign=direction: _is_point_frame(
                frame,
                wk,
                sk,
                sign,
            ),
            min_len=min_len,
        )
        for group in groups:
            peak = max(
                group,
                key=lambda frame, wk=wrist_key, sk=shoulder_key: abs(
                    float(frame[wk][0]) - float(frame[sk][0])
                ),
            )
            out.append(
                {
                    "type": "point",
                    "start": group[0]["f"],
                    "peak": peak["f"],
                    "end": group[-1]["f"],
                    "confidence": round(
                        max(float(frame.get("conf") or 0.0) for frame in group),
                        4,
                    ),
                    "which": side,
                }
            )
    return out


def _is_point_frame(frame: Dict[str, Any], wrist_key: str, shoulder_key: str, direction: int) -> bool:
    wrist = _valid_point(frame, wrist_key)
    shoulder = _valid_point(frame, shoulder_key)
    if wrist is None or shoulder is None:
        return False
    horizontal = (wrist[0] - shoulder[0]) * direction
    near_shoulder_height = abs(wrist[1] - shoulder[1]) <= 0.18
    return horizontal >= 0.18 and near_shoulder_height


def _head_turn_events(frames: List[Dict[str, Any]], min_len: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for typ, predicate, peak_fn in (
        ("head_turn_right", lambda frame: (frame.get("head_yaw") or 0.0) >= 12.0, max),
        ("head_turn_left", lambda frame: (frame.get("head_yaw") or 0.0) <= -12.0, min),
    ):
        groups = _groups_where(
            frames,
            lambda frame, pred=predicate: (
                frame.get("conf", 0.0) >= 0.4
                and frame.get("head_yaw") is not None
                and pred(frame)
            ),
            min_len=min_len,
        )
        for group in groups:
            peak_yaw = peak_fn(float(frame.get("head_yaw") or 0.0) for frame in group)
            peak = min(
                group,
                key=lambda frame, target=peak_yaw: abs(
                    float(frame.get("head_yaw") or 0.0) - target
                ),
            )
            out.append(
                {
                    "type": typ,
                    "start": group[0]["f"],
                    "peak": peak["f"],
                    "end": group[-1]["f"],
                    "confidence": round(
                        max(float(frame.get("conf") or 0.0) for frame in group),
                        4,
                    ),
                }
            )
    return out
