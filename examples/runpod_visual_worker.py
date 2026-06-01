#!/usr/bin/env python3
"""RunPod Serverless worker for Resolve MCP visual analysis.

Deploy this in a container that includes the Resolve MCP repo plus the visual
runtime dependencies: ffmpeg, opencv-python, ultralytics, numpy, and any optional
expression/VLM packages. The worker reads a source-safe media URL or a mounted
RunPod network-volume path, writes only to temporary worker storage, and returns
the visual sidecar JSON to the local MCP.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib import parse as urlparse
from urllib import request as urlrequest

from src.utils.env_loader import load_local_env
from src.utils.runpod_analysis import runpod_volume_path
from src.utils.visual_analysis import (
    _analyze_vlm_keyframe,
    _merge_vlm_metadata_rollup,
    _summarize_vlm_keyframes,
    run_clip_visual_analysis,
)

load_local_env()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("input", event)
    if not isinstance(payload, dict):
        return {"error": "RunPod input must be a JSON object."}
    media = payload.get("media") or {}
    analysis = payload.get("analysis") or {}
    if payload.get("job_type") == "resolve_vlm_keyframes":
        return _handle_vlm_keyframes(media, analysis, payload)
    file_url = media.get("file_url")
    volume_path = media.get("volume_path")
    if not file_url and not volume_path:
        return {"error": "media.file_url or media.volume_path is required."}

    with tempfile.TemporaryDirectory(prefix="resolve-mcp-runpod-") as tmp:
        try:
            local_path = _resolve_media_path(media, Path(tmp))
        except Exception as exc:
            return {"error": str(exc)}
        visual = run_clip_visual_analysis(
            file_path=str(local_path),
            media_id=str(media.get("media_id") or ""),
            clip_name=str(media.get("clip_name") or local_path.name),
            fps=float(media.get("fps") or 0.0),
            duration_frames=int(media.get("duration_frames") or 0),
            tier=str(analysis.get("tier") or "fast"),
            sample_every_n=int(analysis.get("sample_every_n") or 10),
            object_every_n=_optional_int(analysis.get("object_every_n")),
            object_every_seconds=float(analysis.get("object_every_seconds") or 5.0),
            batch_size=int(analysis.get("batch_size") or 1),
            proxy_width=int(analysis.get("proxy_width") or 640),
            pose_model=str(analysis.get("pose_model") or "yolo11n-pose"),
            object_model=str(analysis.get("object_model") or "yolo11n"),
            expression=bool(analysis.get("expression", True)),
            expression_every_n=int(analysis.get("expression_every_n") or 2),
            vlm_model=analysis.get("vlm_model"),
            vlm_max_keyframes=int(analysis.get("vlm_max_keyframes") or 0),
        )
        if media.get("source_file_path"):
            visual["file_path"] = str(media["source_file_path"])
        visual["analysis_backend"] = {"name": "runpod_worker"}
        return {"visual_sidecar": visual}


def _handle_vlm_keyframes(media: Dict[str, Any], analysis: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        return {"error": f"Pillow is required for VLM keyframe jobs: {exc}"}
    vlm_model = analysis.get("vlm_model")
    if not vlm_model:
        return {"error": "analysis.vlm_model is required."}
    fps = float(media.get("fps") or 0.0)
    rollup = payload.get("metadata_rollup") or {}
    camera = payload.get("camera") or {}
    analyzed = []
    for item in media.get("keyframes") or []:
        try:
            frame = int(item.get("frame"))
            path = Path(str(item.get("volume_path") or runpod_volume_path(str(item.get("object_key") or ""))))
            if not path.exists():
                return {"error": f"Keyframe image does not exist: {path}"}
            with Image.open(path) as image:
                analyzed.append(_analyze_vlm_keyframe(str(vlm_model), image.convert("RGB"), frame, fps, rollup, camera))
        except Exception as exc:
            return {"error": f"VLM keyframe analysis failed: {type(exc).__name__}: {exc}"}
    descriptions = [str(item.get("description") or "").strip() for item in analyzed if item.get("description")]
    shot_types = [str(item.get("shot_type") or "").strip() for item in analyzed if item.get("shot_type")]
    clip_summary = _summarize_vlm_keyframes(
        model_name=str(vlm_model),
        clip_name=str(media.get("clip_name") or ""),
        fps=fps,
        duration_frames=int(media.get("duration_frames") or 0),
        keyframes=analyzed,
        rollup=rollup,
        camera=camera,
    )
    vlm = {
        "status": "analyzed",
        "model": vlm_model,
        "description": str(clip_summary.get("clip_summary") or "").strip() or " / ".join(descriptions[:3]),
        "shot_type": shot_types[0] if shot_types else None,
        "keyframes": analyzed,
        "clip_summary": clip_summary,
    }
    return {
        "media_id": media.get("media_id"),
        "clip_name": media.get("clip_name"),
        "vlm": vlm,
        "metadata_rollup": _merge_vlm_metadata_rollup(dict(rollup), vlm),
        "analysis_backend": {"name": "runpod_worker", "mode": "vlm_keyframes"},
    }


def _resolve_media_path(media: Dict[str, Any], directory: Path) -> Path:
    volume_path = media.get("volume_path")
    if volume_path:
        path = Path(str(volume_path))
        if not path.exists():
            raise FileNotFoundError(f"RunPod volume media path does not exist: {path}")
        return path

    file_url = str(media.get("file_url") or "")
    if file_url.startswith("runpod-volume://"):
        object_key = file_url[len("runpod-volume://"):]
        path = Path(runpod_volume_path(object_key))
        if not path.exists():
            raise FileNotFoundError(f"RunPod volume media path does not exist: {path}")
        return path

    if not file_url:
        raise ValueError("media.file_url is required when media.volume_path is not provided.")
    return _download_media(file_url, directory, str(media.get("clip_name") or "clip"))


def _download_media(file_url: str, directory: Path, fallback_name: str) -> Path:
    parsed = urlparse.urlparse(file_url)
    name = Path(urlparse.unquote(parsed.path)).name or fallback_name
    if "." not in name and "." in fallback_name:
        name = fallback_name
    target = directory / _safe_name(name)
    request = urlrequest.Request(file_url, headers={"User-Agent": "resolve-mcp-runpod-worker"})
    with urlrequest.urlopen(request, timeout=int(os.environ.get("RUNPOD_MEDIA_DOWNLOAD_TIMEOUT", "1800"))) as response:
        with target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return target


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in name)
    return safe or "clip"


def _optional_int(value: Any) -> Any:
    if value in {None, ""}:
        return None
    return int(value)


if __name__ == "__main__":
    try:
        import runpod  # type: ignore
    except Exception as exc:  # pragma: no cover - only used in deployed worker
        raise SystemExit(f"runpod package is required in the worker image: {exc}") from exc
    runpod.serverless.start({"handler": handler})
