"""Source-safe local transcript sidecars for Resolve media pool clips.

Parakeet/MLX writes source-frame transcript timings to
``~/Resolve_Analysis/<project>/<media_id>_transcript.json``. The sidecar is the
source of truth for edit decisions; SRT and Resolve metadata are projections.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.motion_analysis import (
    motion_media_id,
    motion_project_dir,
    motion_project_name,
    utc_now_iso,
)


TRANSCRIPT_SCHEMA_VERSION = 1
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_TRANSCRIPT_ENGINE = "parakeet-tdt-0.6b-v3"
TRANSCRIPT_METADATA_FIELDS = ("Description", "Comments", "Keywords")
_PARAKEET_CACHE: Dict[Tuple[str, str, Optional[str], bool, int], Any] = {}
_KEYWORD_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "being",
    "could",
    "every",
    "from",
    "have",
    "here",
    "just",
    "like",
    "more",
    "some",
    "that",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def transcript_sidecar_path(project_name: Any, media_id: Any, root: Optional[str] = None) -> Path:
    return motion_project_dir(project_name, root) / f"{motion_media_id(media_id)}_transcript.json"


def transcript_sidecar_rel(project_name: Any, media_id: Any) -> str:
    return f"{motion_project_name(project_name)}/{motion_media_id(media_id)}_transcript.json"


def transcript_srt_path(project_name: Any, media_id: Any, clip_name: Any = "", root: Optional[str] = None) -> Path:
    stem = Path(str(clip_name or "")).stem.strip() or motion_media_id(media_id)
    safe_stem = re.sub(r"[^A-Za-z0-9_. -]+", "_", stem).strip(" ._") or motion_media_id(media_id)
    return motion_project_dir(project_name, root) / "Subtitles" / f"{safe_stem}_{motion_media_id(media_id)}.srt"


def read_transcript_sidecar(path: Path, include_words: bool = True) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    lines = payload.get("lines") or []
    if not include_words:
        lines = [{key: value for key, value in line.items() if key != "words"} for line in lines]
    return {
        "analyzed": True,
        "schema_version": payload.get("schema_version"),
        "media_id": payload.get("media_id"),
        "clip_name": payload.get("clip_name"),
        "file_path": payload.get("file_path"),
        "fps": payload.get("fps"),
        "duration_frames": payload.get("duration_frames"),
        "engine": payload.get("engine"),
        "model": payload.get("model"),
        "language": payload.get("language"),
        "analyzed_at": payload.get("analyzed_at"),
        "text": payload.get("text") or "",
        "lines": lines,
        "metadata_rollup": payload.get("metadata_rollup") or {},
        "srt_path": payload.get("srt_path"),
    }


def run_parakeet_transcription(
    *,
    file_path: str,
    media_id: str,
    clip_name: str,
    fps: float,
    duration_frames: int,
    model: str = DEFAULT_PARAKEET_MODEL,
    cache_dir: Optional[str] = None,
    fp32: bool = False,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
    max_words: Optional[int] = 16,
    silence_gap: Optional[float] = 0.7,
    max_duration: Optional[float] = 6.0,
    local_attention: bool = False,
    local_attention_context_size: int = 256,
) -> Dict[str, Any]:
    if not file_path or not os.path.exists(file_path):
        raise RuntimeError(f"Media file is offline or missing: {file_path}")
    if not fps:
        fps = 24.0
    loaded_model, dtype = _load_parakeet_model(
        model=model,
        cache_dir=cache_dir,
        fp32=fp32,
        local_attention=local_attention,
        local_attention_context_size=local_attention_context_size,
    )
    try:
        from parakeet_mlx.alignment import SentenceConfig  # type: ignore
        from parakeet_mlx.parakeet import DecodingConfig, Greedy  # type: ignore
    except Exception as exc:
        raise RuntimeError("parakeet-mlx is not installed in the MCP Python environment") from exc

    decoding_config = DecodingConfig(
        decoding=Greedy(),
        sentence=SentenceConfig(max_words=max_words, silence_gap=silence_gap, max_duration=max_duration),
    )
    result = loaded_model.transcribe(
        file_path,
        dtype=dtype,
        chunk_duration=chunk_duration if float(chunk_duration or 0) > 0 else None,
        overlap_duration=overlap_duration,
        decoding_config=decoding_config,
    )
    text = str(getattr(result, "text", "") or "").strip()
    lines = _parakeet_lines_to_source_frames(getattr(result, "sentences", []) or [], fps, duration_frames)
    analyzed_at = utc_now_iso()
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "media_id": media_id,
        "clip_name": clip_name,
        "file_path": file_path,
        "fps": fps,
        "duration_frames": duration_frames,
        "engine": _engine_name(model),
        "model": model,
        "language": "en",
        "analyzed_at": analyzed_at,
        "text": text,
        "lines": lines,
        "metadata_rollup": transcript_metadata_rollup(text, lines),
    }


def _load_parakeet_model(
    *,
    model: str,
    cache_dir: Optional[str],
    fp32: bool,
    local_attention: bool,
    local_attention_context_size: int,
):
    try:
        import mlx.core as mx  # type: ignore
        from parakeet_mlx import from_pretrained  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "local transcription requires parakeet-mlx in the MCP environment: "
            "venv/bin/python -m pip install parakeet-mlx"
        ) from exc
    dtype = mx.float32 if fp32 else mx.bfloat16
    key = (str(model), "fp32" if fp32 else "bf16", str(cache_dir) if cache_dir else None, bool(local_attention), int(local_attention_context_size))
    if key not in _PARAKEET_CACHE:
        loaded = from_pretrained(str(model), dtype=dtype, cache_dir=cache_dir)
        if local_attention:
            loaded.encoder.set_attention_model(
                "rel_pos_local_attn",
                (int(local_attention_context_size), int(local_attention_context_size)),
            )
        _PARAKEET_CACHE[key] = loaded
    return _PARAKEET_CACHE[key], dtype


def _engine_name(model: str) -> str:
    name = str(model or DEFAULT_PARAKEET_MODEL).rstrip("/").split("/")[-1]
    return name or DEFAULT_TRANSCRIPT_ENGINE


def _seconds_to_frame(seconds: Any, fps: float, duration_frames: int = 0) -> int:
    try:
        frame = int(round(float(seconds) * float(fps or 24.0)))
    except Exception:
        frame = 0
    if duration_frames > 0:
        frame = min(max(frame, 0), max(duration_frames - 1, 0))
    return max(frame, 0)


def _parakeet_lines_to_source_frames(sentences: Iterable[Any], fps: float, duration_frames: int) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for sentence in sentences:
        tokens = list(getattr(sentence, "tokens", []) or [])
        words = [
            {
                "w": str(getattr(token, "text", "") or "").strip(),
                "start": _seconds_to_frame(getattr(token, "start", 0.0), fps, duration_frames),
                "end": _seconds_to_frame(getattr(token, "end", getattr(token, "start", 0.0)), fps, duration_frames),
                "confidence": round(float(getattr(token, "confidence", 1.0) or 0.0), 4),
            }
            for token in tokens
            if str(getattr(token, "text", "") or "").strip()
        ]
        text = str(getattr(sentence, "text", "") or "").strip()
        if not text and words:
            text = " ".join(word["w"] for word in words)
        if not text:
            continue
        start = _seconds_to_frame(getattr(sentence, "start", 0.0), fps, duration_frames)
        end = _seconds_to_frame(getattr(sentence, "end", getattr(sentence, "start", 0.0)), fps, duration_frames)
        if words:
            start = min(start, words[0]["start"])
            end = max(end, words[-1]["end"])
        lines.append(
            {
                "text": text,
                "start": start,
                "end": max(end, start),
                "confidence": round(float(getattr(sentence, "confidence", 1.0) or 0.0), 4),
                "words": words,
            }
        )
    return lines


def transcript_metadata_rollup(text: str, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = " ".join(str(text or "").split())
    keywords = transcript_keywords(clean)
    return {
        "description": f"Transcript: {_truncate(clean, 700)}" if clean else "",
        "comments": f"Transcript:\n{_truncate(clean, 8000)}" if clean else "",
        "keywords": ["transcript", "speech", *keywords],
        "line_count": len(lines or []),
        "word_count": sum(len(line.get("words") or []) for line in lines or []),
    }


def transcript_keywords(text: str, limit: int = 24) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']{2,}", str(text or "").lower())
    counts = Counter(word.strip("'") for word in words if word not in _KEYWORD_STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def frame_to_srt_timestamp(frame: int, fps: float) -> str:
    seconds = max(0.0, float(frame) / float(fps or 24.0))
    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def transcript_to_srt(payload: Dict[str, Any]) -> str:
    fps = float(payload.get("fps") or 24.0)
    entries: List[str] = []
    index = 1
    for line in payload.get("lines") or []:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        start = int(line.get("start") or 0)
        end = int(line.get("end") if line.get("end") is not None else start)
        if end <= start:
            end = start + max(1, int(math.ceil(fps * 0.5)))
        entries.extend(
            [
                str(index),
                f"{frame_to_srt_timestamp(start, fps)} --> {frame_to_srt_timestamp(end, fps)}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(entries)


def write_transcript_srt(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript_to_srt(payload), encoding="utf-8")
