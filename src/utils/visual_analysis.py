"""Source-safe visual analysis sidecars for Resolve media pool clips.

The fast tier shares one ffmpeg decode loop across dense CV passes:
YOLO pose, YOLO objects, shot-size box math, camera optical-flow, and optional
HSEmotion expression scoring. Heavy sampled-frame data stays in a sidecar under
``~/Resolve_Analysis``; Resolve metadata receives only a compact search rollup.
"""

from __future__ import annotations

import base64
import io
import os
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from src.utils.motion_analysis import (
    _frame_chunks,
    _pose_frame_from_result,
    add_motion_energy,
    extract_motion_events,
    ffprobe_video_info,
    motion_media_id,
    motion_project_dir,
    motion_project_name,
    proxy_dimensions,
    read_motion_sidecar,
    resolve_pose_model_path,
    summarize_motion_events,
    utc_now_iso,
)


VISUAL_SCHEMA_VERSION = 1
VISUAL_METADATA_FIELDS = ("Description", "Comments", "Keywords", "Shot")
DEFAULT_OLLAMA_VLM_MODEL = "ollama:qwen3-vl:8b"
OLLAMA_VLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "shot_type": {"type": "string"},
        "description": {"type": "string"},
        "story_beat": {"type": "boolean"},
        "story_note": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
    },
    "required": ["shot_type", "description", "story_beat", "story_note", "keywords", "tone"],
    "additionalProperties": False,
}
_OBJECT_IGNORE = {"person"}
_VLM_CACHE: Dict[str, Any] = {}


def visual_sidecar_path(project_name: Any, media_id: Any, root: Optional[str] = None) -> Path:
    return motion_project_dir(project_name, root) / f"{motion_media_id(media_id)}_visual.json"


def visual_sidecar_rel(project_name: Any, media_id: Any) -> str:
    return f"{motion_project_name(project_name)}/{motion_media_id(media_id)}_visual.json"


def read_visual_sidecar(path: Path, include_frames: bool = False) -> Optional[Dict[str, Any]]:
    data = read_motion_sidecar(path, include_frames=True)
    if data is None:
        return None
    # read_motion_sidecar intentionally returns pose-oriented keys. Reload the
    # full payload here so the visual sections are preserved.
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {
        "analyzed": True,
        "schema_version": payload.get("schema_version"),
        "media_id": payload.get("media_id"),
        "clip_name": payload.get("clip_name"),
        "file_path": payload.get("file_path"),
        "fps": payload.get("fps"),
        "duration_frames": payload.get("duration_frames"),
        "sampled_every_n_frames": payload.get("sampled_every_n_frames"),
        "object_sampled_every_n_frames": payload.get("object_sampled_every_n_frames"),
        "object_sampled_every_seconds": payload.get("object_sampled_every_seconds"),
        "batch_size": payload.get("batch_size"),
        "proxy_width": payload.get("proxy_width"),
        "analyzed_at": payload.get("analyzed_at"),
        "tier": payload.get("tier"),
        "models": payload.get("models") or {},
        "events": payload.get("events") or [],
        "expression": _strip_frames(payload.get("expression") or {}, include_frames),
        "objects": _strip_frames(payload.get("objects") or {}, include_frames),
        "shot_size": _strip_frames(payload.get("shot_size") or {}, include_frames),
        "camera": payload.get("camera") or {},
        "vlm": payload.get("vlm") or {},
        "metadata_rollup": payload.get("metadata_rollup") or {},
    }
    if include_frames:
        out["frames"] = payload.get("frames") or []
    return out


def _strip_frames(section: Dict[str, Any], include_frames: bool) -> Dict[str, Any]:
    out = dict(section or {})
    if not include_frames:
        out.pop("frames", None)
    return out


def run_clip_visual_analysis(
    *,
    file_path: str,
    media_id: str,
    clip_name: str,
    fps: float,
    duration_frames: int,
    tier: str = "fast",
    sample_every_n: int = 10,
    object_every_n: Optional[int] = None,
    object_every_seconds: float = 5.0,
    batch_size: int = 1,
    proxy_width: int = 640,
    pose_model: str = "yolo11n-pose",
    object_model: str = "yolo11n",
    expression: bool = True,
    expression_every_n: int = 2,
    vlm_model: Optional[str] = None,
    vlm_max_keyframes: int = 12,
) -> Dict[str, Any]:
    if not file_path or not os.path.exists(file_path):
        raise RuntimeError(f"Media file is offline or missing: {file_path}")
    tier = (tier or "fast").strip().lower()
    if tier not in {"fast", "deep"}:
        raise RuntimeError("tier must be 'fast' or 'deep'")
    if tier == "deep" and not vlm_model:
        vlm_model = DEFAULT_OLLAMA_VLM_MODEL
    sample_every_n = max(1, int(sample_every_n or 10))
    object_every_seconds = max(0.1, float(object_every_seconds or 5.0))
    batch_size = max(1, int(batch_size or 1))
    expression_every_n = max(1, int(expression_every_n or 2))

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "visual analysis requires numpy, opencv-python, and ultralytics in the MCP Python environment"
        ) from exc

    probe = ffprobe_video_info(file_path)
    source_width = probe.get("width") or proxy_width
    source_height = probe.get("height") or proxy_width
    if not fps:
        fps = float(probe.get("fps") or 0.0)
    if not duration_frames:
        duration_frames = int(probe.get("duration_frames") or 0)
    if object_every_n is None or int(object_every_n or 0) <= 0:
        object_every_n = max(sample_every_n, int(round(float(fps or 24.0) * object_every_seconds)))
    else:
        object_every_n = max(1, int(object_every_n))
    width, height = proxy_dimensions(source_width, source_height, proxy_width)
    frame_bytes = width * height * 3

    pose_path = resolve_pose_model_path(pose_model)
    object_path = resolve_pose_model_path(object_model)
    pose_yolo = YOLO(pose_path)
    object_yolo = YOLO(object_path)
    expression_model, expression_error = _load_expression_model() if expression else (None, None)

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

    pose_frames: List[Dict[str, Any]] = []
    object_frames: List[Dict[str, Any]] = []
    shot_frames: List[Dict[str, Any]] = []
    expression_frames: List[Dict[str, Any]] = []
    camera_intervals: List[Dict[str, Any]] = []
    prev_gray = None
    prev_f: Optional[int] = None

    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    next_object_f = 0

    def process_batch(batch: List[Tuple[int, Any]]) -> None:
        nonlocal next_object_f, prev_gray, prev_f
        if not batch:
            return
        images = [image for _, image in batch]
        pose_results = pose_yolo(images, verbose=False)
        object_indices: List[int] = []
        for index, (f, _) in enumerate(batch):
            if f >= next_object_f:
                object_indices.append(index)
                next_object_f = _next_sparse_sample_frame(f, object_every_n)
        object_results = object_yolo([batch[index][1] for index in object_indices], verbose=False) if object_indices else []
        object_result_by_index = {
            index: object_results[offset] if offset < len(object_results) else None
            for offset, index in enumerate(object_indices)
        }

        for index, (f, image) in enumerate(batch):
            pose_result = pose_results[index] if index < len(pose_results) else None
            pose_frame = _pose_frame_from_result(
                pose_result,
                true_frame=f,
                width=width,
                height=height,
            )
            pose_frames.append(pose_frame)

            detection = {"objects": [], "person_boxes": []}
            if index in object_result_by_index:
                detection = _detect_objects_from_result(object_result_by_index.get(index), width, height)
                object_frames.append({"f": f, "objects": detection["objects"]})
            person_box = _best_person_box(detection["person_boxes"], pose_frame.get("person_box"))
            shot_frames.append({"f": f, "size": _shot_size_from_box(person_box), "person_box": person_box})

            sample_index = f // sample_every_n
            if expression_model is not None and sample_index % expression_every_n == 0:
                expression_frames.append(_expression_frame(expression_model, image, f, person_box))

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None and prev_f is not None:
                camera_intervals.append(_camera_interval(cv2, prev_gray, gray, prev_f, f))
            prev_gray = gray
            prev_f = f

    try:
        batch: List[Tuple[int, Any]] = []
        for sample_index, chunk in enumerate(_frame_chunks(proc, frame_bytes)):
            f = sample_index * sample_every_n
            image = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            batch.append((f, image))
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []
        process_batch(batch)

        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg exited with status {returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()

    add_motion_energy(pose_frames)
    pose_events = extract_motion_events(pose_frames, fps=fps, sample_every_n=sample_every_n)
    expression_events = _expression_events(expression_frames)
    objects = _object_rollup(object_frames)
    shot_size = _shot_size_rollup(shot_frames)
    camera = {"timeline": _compress_camera_timeline(camera_intervals)}
    metadata_rollup = _metadata_rollup(
        pose_events=pose_events,
        expression_events=expression_events,
        objects=objects,
        shot_size=shot_size,
        camera=camera,
    )
    vlm = _run_deep_vlm(
        tier=tier,
        vlm_model=vlm_model,
        file_path=file_path,
        fps=fps,
        duration_frames=duration_frames,
        width=width,
        height=height,
        sample_every_n=sample_every_n,
        pose_events=pose_events,
        expression_events=expression_events,
        camera=camera,
        rollup=metadata_rollup,
        max_keyframes=vlm_max_keyframes,
    )
    metadata_rollup = _merge_vlm_metadata_rollup(metadata_rollup, vlm)
    analyzed_at = utc_now_iso()

    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "media_id": media_id,
        "clip_name": clip_name,
        "file_path": file_path,
        "fps": fps,
        "duration_frames": duration_frames,
        "sampled_every_n_frames": sample_every_n,
        "object_sampled_every_n_frames": object_every_n,
        "object_sampled_every_seconds": object_every_seconds,
        "batch_size": batch_size,
        "proxy_width": width,
        "analyzed_at": analyzed_at,
        "tier": tier,
        "models": {
            "pose": pose_model,
            "pose_path": pose_path if os.path.exists(pose_path) else None,
            "objects": object_model,
            "objects_path": object_path if os.path.exists(object_path) else None,
            "expression": "hsemotion/enet_b0_8_best_vgaf" if expression_model is not None else None,
            "expression_error": expression_error,
            "vlm": vlm_model,
        },
        "person_count_mode": "single",
        "frames": pose_frames,
        "events": pose_events,
        "events_summary": summarize_motion_events(pose_events),
        "expression": {
            "frames": expression_frames,
            "events": expression_events,
            "dominant": _dominant_expression(expression_frames),
        },
        "objects": objects,
        "shot_size": shot_size,
        "camera": camera,
        "vlm": vlm,
        "metadata_rollup": metadata_rollup,
    }


def _next_sparse_sample_frame(current_frame: int, every_n: int) -> int:
    return int(current_frame) + max(1, int(every_n or 1))


def _detect_objects(yolo: Any, image: Any, f: int, width: int, height: int) -> Dict[str, Any]:
    result_list = yolo(image, verbose=False)
    result = result_list[0] if result_list else None
    return _detect_objects_from_result(result, width, height)


def _detect_objects_from_result(result: Any, width: int, height: int) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    person_boxes: List[List[float]] = []
    if result is None or getattr(result, "boxes", None) is None:
        return {"objects": objects, "person_boxes": person_boxes}
    boxes = result.boxes
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
    except Exception:
        xyxy = boxes.xyxy.numpy()
        conf = boxes.conf.numpy()
        cls = boxes.cls.numpy()
    names = getattr(result, "names", {}) or {}
    for index, box in enumerate(xyxy):
        klass = names.get(int(cls[index]), str(int(cls[index])))
        norm_box = [
            round(float(box[0]) / width, 6),
            round(float(box[1]) / height, 6),
            round(float(box[2]) / width, 6),
            round(float(box[3]) / height, 6),
        ]
        item = {
            "class": str(klass),
            "confidence": round(float(conf[index]), 4),
            "box": norm_box,
        }
        objects.append(item)
        if str(klass).lower() == "person":
            person_boxes.append(norm_box)
    return {"objects": objects, "person_boxes": person_boxes}


def _best_person_box(detected_boxes: List[List[float]], pose_box: Any) -> Optional[List[float]]:
    if detected_boxes:
        return max(detected_boxes, key=lambda box: max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    if isinstance(pose_box, list) and len(pose_box) == 4:
        return [float(v) for v in pose_box]
    return None


def _shot_size_from_box(box: Optional[List[float]]) -> str:
    if not box:
        return "unknown"
    h = max(0.0, float(box[3]) - float(box[1]))
    if h >= 0.85:
        return "ECU"
    if h >= 0.68:
        return "CU"
    if h >= 0.50:
        return "MCU"
    if h >= 0.34:
        return "MS"
    if h >= 0.22:
        return "MWS"
    if h >= 0.12:
        return "WS"
    return "EWS"


def _camera_interval(cv2: Any, prev_gray: Any, gray: Any, start_f: int, end_f: int) -> Dict[str, Any]:
    import numpy as np  # type: ignore

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    dx = float(np.mean(flow[:, :, 0]))
    dy = float(np.mean(flow[:, :, 1]))
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    mean_mag = float(np.mean(mag))
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx = (xx - (w / 2.0)) / max(1.0, w / 2.0)
    ry = (yy - (h / 2.0)) / max(1.0, h / 2.0)
    radial = float(np.mean(flow[:, :, 0] * rx + flow[:, :, 1] * ry))
    move = "static"
    if mean_mag >= 0.25:
        if abs(radial) > max(abs(dx), abs(dy), 0.1) * 0.9:
            move = "push_in" if radial > 0 else "pull_out"
        elif abs(dx) >= abs(dy):
            move = "pan_right" if dx < 0 else "pan_left"
        else:
            move = "tilt_up" if dy > 0 else "tilt_down"
        if mean_mag >= 2.0 and float(np.std(mag)) / max(mean_mag, 0.01) > 1.6:
            move = "handheld"
    return {
        "start": int(start_f),
        "end": int(end_f),
        "move": move,
        "magnitude": round(mean_mag, 4),
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "radial": round(radial, 4),
    }


def _compress_camera_timeline(intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compressed: List[Dict[str, Any]] = []
    for interval in intervals:
        if compressed and compressed[-1]["move"] == interval["move"]:
            previous = compressed[-1]
            previous["end"] = interval["end"]
            previous["magnitude"] = round(max(previous.get("magnitude", 0.0), interval.get("magnitude", 0.0)), 4)
        else:
            compressed.append(dict(interval))
    return compressed


def _load_expression_model():
    try:
        import torch  # type: ignore
        from hsemotion.facial_emotions import HSEmotionRecognizer  # type: ignore
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf", device="cpu"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        torch.load = original_load


def _expression_frame(model: Any, image: Any, f: int, person_box: Optional[List[float]]) -> Dict[str, Any]:
    crop = _faceish_crop(image, person_box)
    if crop is None:
        return {"f": f, "label": "unknown", "confidence": 0.0}
    try:
        label, scores = model.predict_emotions(crop, logits=False)
        labels = [model.idx_to_class[idx] for idx in sorted(model.idx_to_class)]
        score_map = {labels[i]: float(scores[i]) for i in range(min(len(labels), len(scores)))}
        normalized = _normalize_expression(str(label), score_map)
        return {
            "f": f,
            "label": normalized,
            "raw_label": str(label),
            "confidence": round(max(score_map.values()) if score_map else 0.0, 4),
        }
    except Exception as exc:
        return {"f": f, "label": "unknown", "confidence": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def _faceish_crop(image: Any, person_box: Optional[List[float]]):
    h, w = image.shape[:2]
    if not person_box:
        return None
    x1 = max(0, int(float(person_box[0]) * w))
    y1 = max(0, int(float(person_box[1]) * h))
    x2 = min(w, int(float(person_box[2]) * w))
    y2 = min(h, int((float(person_box[1]) + (float(person_box[3]) - float(person_box[1])) * 0.45) * h))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return image[y1:y2, x1:x2]


def _normalize_expression(label: str, scores: Dict[str, float]) -> str:
    lower = label.lower()
    if "happiness" in lower:
        conf = max(scores.get("Happiness", 0.0), scores.get("happiness", 0.0))
        return "animated" if conf >= 0.70 else "smiling"
    if "surprise" in lower:
        return "surprised"
    if any(part in lower for part in ("anger", "fear", "contempt", "disgust")):
        return "intense"
    if "sad" in lower:
        return "sad-ish"
    if "neutral" in lower:
        return "neutral"
    return lower or "unknown"


def _expression_events(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for frame in frames:
        active = frame.get("label") not in {"neutral", "unknown", None} and float(frame.get("confidence") or 0.0) >= 0.35
        if active:
            current.append(frame)
        else:
            if current:
                events.append(_expression_event(current))
            current = []
    if current:
        events.append(_expression_event(current))
    return events


def _expression_event(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    peak = max(group, key=lambda frame: float(frame.get("confidence") or 0.0))
    label = Counter(str(frame.get("label") or "unknown") for frame in group).most_common(1)[0][0]
    return {
        "type": "expression",
        "expression": label,
        "start": group[0]["f"],
        "peak": peak["f"],
        "end": group[-1]["f"],
        "confidence": round(float(peak.get("confidence") or 0.0), 4),
    }


def _dominant_expression(frames: List[Dict[str, Any]]) -> str:
    labels = [str(frame.get("label")) for frame in frames if frame.get("label") not in {None, "unknown"}]
    return Counter(labels).most_common(1)[0][0] if labels else "unknown"


def _object_rollup(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter = Counter()
    compact_frames = []
    for frame in frames:
        classes = []
        for obj in frame.get("objects") or []:
            klass = str(obj.get("class") or "").lower()
            if not klass:
                continue
            counts[klass] += 1
            if klass not in classes:
                classes.append(klass)
        compact_frames.append({"f": frame.get("f"), "classes": classes})
    per_clip = [klass for klass, _ in counts.most_common(12) if klass not in _OBJECT_IGNORE]
    return {"per_clip": per_clip, "frames": compact_frames}


def _shot_size_rollup(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [str(frame.get("size")) for frame in frames if frame.get("size") and frame.get("size") != "unknown"]
    dominant = Counter(sizes).most_common(1)[0][0] if sizes else "unknown"
    return {"frames": frames, "dominant": dominant}


def _metadata_rollup(
    *,
    pose_events: List[Dict[str, Any]],
    expression_events: List[Dict[str, Any]],
    objects: Dict[str, Any],
    shot_size: Dict[str, Any],
    camera: Dict[str, Any],
) -> Dict[str, Any]:
    shot = shot_size.get("dominant") or "unknown"
    camera_moves = [entry.get("move") for entry in camera.get("timeline") or [] if entry.get("move")]
    camera_move = Counter(camera_moves).most_common(1)[0][0] if camera_moves else "static"
    tone = Counter(event.get("expression") for event in expression_events if event.get("expression")).most_common(1)
    tone_value = tone[0][0] if tone else "unknown"
    object_list = objects.get("per_clip") or []
    gesture = "gesturing" if any(event.get("type") in {"hand_raise", "point", "high_motion"} for event in pose_events) else "steady"
    description_parts = [part for part in [shot, camera_move, gesture, tone_value if tone_value != "unknown" else None] if part]
    if object_list:
        description_parts.append("objects: " + ", ".join(object_list[:5]))
    description = "; ".join(description_parts)
    keywords = [str(camera_move), gesture]
    if shot != "unknown":
        keywords.extend(["talking-head", str(shot).lower()])
    else:
        keywords.append("visual")
    if tone_value != "unknown":
        keywords.append(str(tone_value))
    keywords.extend(str(obj) for obj in object_list[:8])
    return {
        "shot_size": shot,
        "camera": camera_move,
        "description": description,
        "keywords": sorted({kw for kw in keywords if kw and kw != "unknown"}),
        "tone": tone_value,
    }


def _merge_vlm_metadata_rollup(rollup: Dict[str, Any], vlm: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(vlm, dict) or vlm.get("status") != "analyzed":
        return rollup
    merged = dict(rollup)
    description = str(vlm.get("description") or "").strip()
    if description:
        merged["description"] = description
    shot_type = str(vlm.get("shot_type") or "").strip()
    if shot_type:
        merged["shot_type"] = shot_type
    keywords = list(merged.get("keywords") or [])
    if shot_type:
        keywords.append(shot_type)
    tones = []
    for keyframe in vlm.get("keyframes") or []:
        keywords.extend(str(keyword) for keyword in (keyframe.get("keywords") or []) if keyword)
        tone = str(keyframe.get("tone") or "").strip()
        if tone and tone != "unknown":
            tones.append(tone)
    if tones:
        merged["tone"] = Counter(tones).most_common(1)[0][0]
    merged["keywords"] = sorted({keyword for keyword in keywords if keyword and keyword != "unknown"})
    return merged


def _run_deep_vlm(
    *,
    tier: str,
    vlm_model: Optional[str],
    file_path: str,
    fps: float,
    duration_frames: int,
    width: int,
    height: int,
    sample_every_n: int,
    pose_events: List[Dict[str, Any]],
    expression_events: List[Dict[str, Any]],
    camera: Dict[str, Any],
    rollup: Dict[str, Any],
    max_keyframes: int = 12,
) -> Dict[str, Any]:
    if tier != "deep":
        return {"status": "skipped", "reason": "tier=fast"}
    if not vlm_model:
        return {
            "status": "not_configured",
            "reason": f"Pass a local VLM model path via vlm_model, or use {DEFAULT_OLLAMA_VLM_MODEL}.",
            "shot_type": rollup.get("camera"),
            "description": rollup.get("description"),
            "keyframes": [],
            "story_beats": [],
        }
    keyframes = _select_vlm_keyframes(
        duration_frames=duration_frames,
        sample_every_n=sample_every_n,
        pose_events=pose_events,
        expression_events=expression_events,
        camera=camera,
        max_keyframes=max_keyframes,
    )
    try:
        images = _extract_keyframe_images(file_path, keyframes, width, height)
        if not images:
            raise RuntimeError("No keyframes could be decoded for deep visual analysis")
        analyzed = []
        for frame, image in images:
            analyzed.append(_analyze_vlm_keyframe(vlm_model, image, frame, fps, rollup, camera))
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "model": vlm_model,
            "shot_type": rollup.get("camera"),
            "description": rollup.get("description"),
            "keyframes": [{"frame": frame} for frame in keyframes],
            "story_beats": [],
        }
    shot_types = [item.get("shot_type") for item in analyzed if item.get("shot_type")]
    descriptions = [item.get("description") for item in analyzed if item.get("description")]
    story_beats = [
        {"frame": item.get("frame"), "note": item.get("story_note") or item.get("description")}
        for item in analyzed
        if item.get("story_beat")
    ]
    return {
        "status": "analyzed",
        "model": vlm_model,
        "shot_type": Counter(shot_types).most_common(1)[0][0] if shot_types else rollup.get("camera"),
        "description": " / ".join(descriptions[:3]) if descriptions else rollup.get("description"),
        "keyframes": analyzed,
        "story_beats": story_beats,
    }


def _select_vlm_keyframes(
    *,
    duration_frames: int,
    sample_every_n: int,
    pose_events: List[Dict[str, Any]],
    expression_events: List[Dict[str, Any]],
    camera: Dict[str, Any],
    max_keyframes: int = 12,
) -> List[int]:
    max_keyframes = max(1, int(max_keyframes or 12))
    frame_limit = max(0, int(duration_frames or 0) - 1)
    if frame_limit <= 0:
        return [0]
    bucket_count = min(max_keyframes, frame_limit + 1)
    candidates: List[Tuple[int, int]] = []

    def add(priority: int, value: Any) -> None:
        try:
            frame = int(value)
        except Exception:
            return
        if frame < 0:
            return
        if frame_limit:
            frame = min(frame, frame_limit)
        candidates.append((priority, frame))

    for event in pose_events:
        event_type = str(event.get("type") or "")
        priority = 3 if event_type == "still" else 0
        add(priority, event.get("peak") if event.get("peak") is not None else event.get("start"))
    for event in expression_events:
        add(1, event.get("peak") if event.get("peak") is not None else event.get("start"))
    for segment in camera.get("timeline") or []:
        add(2, segment.get("start"))
        if segment.get("start") is not None and segment.get("end") is not None:
            add(3, (int(segment["start"]) + int(segment["end"])) // 2)

    selected: List[int] = []
    used = set()
    bucket_ranges: List[Tuple[int, int, int]] = []
    for index in range(bucket_count):
        start = int(round(index * (frame_limit + 1) / bucket_count))
        end = int(round((index + 1) * (frame_limit + 1) / bucket_count)) - 1
        end = min(frame_limit, max(start, end))
        center = min(frame_limit, max(0, (start + end) // 2))
        bucket_ranges.append((start, end, center))

    for start, end, center in bucket_ranges:
        bucket_candidates = [
            (priority, abs(frame - center), frame)
            for priority, frame in candidates
            if start <= frame <= end and frame not in used
        ]
        if bucket_candidates:
            _, _, frame = sorted(bucket_candidates)[0]
        else:
            frame = center
        selected.append(frame)
        used.add(frame)

    if len(selected) < bucket_count:
        for _, _, center in bucket_ranges:
            if center in used:
                continue
            selected.append(center)
            used.add(center)
            if len(selected) >= bucket_count:
                break

    return sorted(selected[:bucket_count] or [0])
    selected: List[int] = []
    for _, frame in sorted(candidates, key=lambda item: (item[0], item[1])):
        if any(abs(frame - existing) < min_gap for existing in selected):
            continue
        selected.append(frame)
        if len(selected) >= max_keyframes:
            break
    return sorted(selected or [0])


def _extract_keyframe_images(file_path: str, frames: List[int], width: int, height: int) -> List[Tuple[int, Any]]:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    unique_frames = sorted({max(0, int(frame)) for frame in frames})
    if not unique_frames:
        return []
    frame_bytes = width * height * 3
    selector = "+".join(f"eq(n\\,{frame})" for frame in unique_frames)
    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-vf",
        f"select='{selector}',scale={width}:{height}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    images: List[Tuple[int, Any]] = []
    try:
        for index, chunk in enumerate(_frame_chunks(proc, frame_bytes)):
            if index >= len(unique_frames):
                break
            array = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            images.append((unique_frames[index], Image.fromarray(array.copy(), "RGB")))
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg exited with status {returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()
    return images


def _analyze_vlm_keyframe(
    model_name: str,
    image: Any,
    frame: int,
    fps: float,
    rollup: Dict[str, Any],
    camera: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = _vlm_prompt(frame, fps, rollup, camera)
    if _is_ollama_vlm_model(model_name):
        raw = _generate_ollama_response(model_name, image, prompt)
    else:
        model, processor = _load_qwen_vlm(model_name)
        raw = _generate_qwen_response(model, processor, image, prompt)
    parsed = _parse_vlm_json(raw)
    return {
        "frame": frame,
        "time_seconds": round(frame / fps, 3) if fps else None,
        "shot_type": parsed.get("shot_type") or rollup.get("camera"),
        "description": parsed.get("description") or raw.strip(),
        "story_beat": _boolish(parsed.get("story_beat")),
        "story_note": parsed.get("story_note") or parsed.get("note") or "",
        "keywords": parsed.get("keywords") or [],
        "tone": parsed.get("tone") or rollup.get("tone"),
        "raw_response": raw,
    }


def _load_qwen_vlm(model_name: str):
    key = str(model_name)
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
    except Exception as exc:
        raise RuntimeError("deep visual analysis requires transformers, torch, and qwen-vl-utils") from exc

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    _VLM_CACHE[key] = (model, processor)
    return _VLM_CACHE[key]


def _generate_qwen_response(model: Any, processor: Any, image: Any, prompt: str) -> str:
    import torch  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = _model_device(model)
    try:
        inputs = inputs.to(device)
    except Exception:
        pass
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=220)
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cpu")


def _is_ollama_vlm_model(model_name: str) -> bool:
    value = str(model_name or "").strip().lower()
    return value.startswith("ollama:") or value.startswith("ollama/") or value.startswith("ollama://")


def _ollama_model_name(model_name: str) -> str:
    value = str(model_name or "").strip()
    lowered = value.lower()
    if lowered.startswith("ollama://"):
        return value[len("ollama://"):].strip()
    if lowered.startswith("ollama:"):
        return value[len("ollama:"):].strip()
    if lowered.startswith("ollama/"):
        return value[len("ollama/"):].strip()
    return value


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def _ollama_num_ctx() -> int:
    try:
        return max(512, int(os.environ.get("RESOLVE_MCP_OLLAMA_NUM_CTX", "4096") or 4096))
    except Exception:
        return 4096


def _image_to_base64_jpeg(image: Any) -> str:
    buffer = io.BytesIO()
    if getattr(image, "mode", "RGB") != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _generate_ollama_response(model_name: str, image: Any, prompt: str) -> str:
    model = _ollama_model_name(model_name)
    if not model:
        raise RuntimeError(f"Ollama VLM model must be provided as ollama:<model>, for example {DEFAULT_OLLAMA_VLM_MODEL}")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_image_to_base64_jpeg(image)],
        "stream": False,
        "format": OLLAMA_VLM_JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": _ollama_num_ctx(),
            "num_predict": 220,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    endpoint = f"{_ollama_host()}/api/generate"
    request = urlrequest.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    timeout = float(os.environ.get("RESOLVE_MCP_OLLAMA_TIMEOUT", "180") or 180)
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Ollama is not reachable at {_ollama_host()}: {exc.reason}") from exc
    message = result.get("message") if isinstance(result.get("message"), dict) else {}
    return str(result.get("response") or message.get("content") or result.get("thinking") or "")


def _vlm_prompt(frame: int, fps: float, rollup: Dict[str, Any], camera: Dict[str, Any]) -> str:
    seconds = round(frame / fps, 2) if fps else None
    camera_context = _vlm_camera_context(camera, frame)
    return (
        "You are providing visual judgment for a film editor. "
        "Analyze this single keyframe in the context of the dense CV signals. "
        "Return exactly one JSON object with only these keys: shot_type, description, "
        "story_beat, story_note, keywords, tone. Do not return start, end, frame, or time keys. "
        "story_beat must be a JSON boolean, not prose. keywords must be short searchable tags. "
        "Use concise editorial language. story_beat must be true only if the frame looks like a strong "
        "delivery, reaction, gesture peak, or useful cutaway. Do not copy keys from the dense_rollup or "
        "camera_context; use those only as context for judging the image.\n"
        f"Frame: {frame}; seconds: {seconds}; dense_rollup: {json.dumps(rollup, sort_keys=True)}; "
        f"camera_context: {json.dumps(camera_context, sort_keys=True)}"
    )


def _vlm_camera_context(camera: Dict[str, Any], frame: int) -> Dict[str, Any]:
    timeline = camera.get("timeline") or []
    moves = [str(segment.get("move") or "") for segment in timeline if segment.get("move")]
    nearby: List[Dict[str, Any]] = []
    try:
        target = int(frame)
    except Exception:
        target = 0
    for segment in timeline:
        try:
            start = int(segment.get("start") or 0)
            end = int(segment.get("end") if segment.get("end") is not None else start)
        except Exception:
            continue
        if start <= target <= end or min(abs(target - start), abs(target - end)) <= 120:
            nearby.append({
                "move": segment.get("move"),
                "from": start,
                "to": end,
            })
        if len(nearby) >= 5:
            break
    return {
        "dominant_moves": [move for move, _ in Counter(moves).most_common(3)],
        "nearby_motion": nearby,
    }


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def _parse_vlm_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
