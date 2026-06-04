"""Editorial intelligence helpers for transcript + visual sidecars.

These functions are intentionally deterministic and dependency-light. They turn
cached analysis sidecars into reusable edit signals: transcript meaning vectors,
long-video visual section maps, and scored cut/punch-in candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.motion_analysis import (
    motion_media_id,
    motion_project_dir,
    utc_now_iso,
)


EMBEDDING_SCHEMA_VERSION = 1
CUT_PLAN_SCHEMA_VERSION = 1
DEFAULT_EMBEDDING_DIMENSIONS = 384

_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "could",
    "every", "from", "have", "here", "just", "like", "more", "some", "that",
    "their", "them", "then", "there", "they", "this", "through", "what", "when",
    "where", "which", "with", "would", "your", "into", "onto", "over", "under",
    "really", "right", "going", "gonna", "okay", "yeah", "things", "thing",
}

_SEMANTIC_EXPANSIONS = {
    "faith": ["belief", "trust", "god", "lord", "spiritual", "church"],
    "pressure": ["stress", "trial", "hardship", "struggle", "challenge", "burden"],
    "fear": ["afraid", "anxiety", "worry", "scared"],
    "hope": ["confidence", "future", "promise", "encourage"],
    "family": ["home", "parents", "kids", "children", "marriage"],
    "money": ["financial", "budget", "cost", "pay", "price"],
    "work": ["job", "career", "business", "labor"],
    "love": ["care", "compassion", "relationship"],
    "pain": ["hurt", "suffering", "grief", "loss"],
}

_FILLER_WORDS = {
    "um", "uh", "erm", "ah", "like", "you know", "i mean", "sort of", "kind of",
}

_STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean_talking_head": {
        "label": "Clean talking head",
        "selection_bias": "clarity and natural pacing",
        "ideal_line_words": (8, 36),
        "prefer_short_lines": False,
        "prefer_motion": True,
        "needs_punch_ins": True,
        "needs_cutaways": False,
        "notes": [
            "Preserve clarity and natural breath.",
            "Remove obvious resets/dead air without over-compressing.",
            "Use punch-ins as polish, not as the edit's foundation.",
        ],
    },
    "punchy_social": {
        "label": "Punchy social",
        "selection_bias": "hook, directness, and pace",
        "ideal_line_words": (5, 22),
        "prefer_short_lines": True,
        "prefer_motion": True,
        "needs_punch_ins": True,
        "needs_cutaways": False,
        "notes": [
            "Front-load the strongest hook.",
            "Prefer short, direct lines over caveats and setup.",
            "Use visual energy and punch-ins to keep momentum.",
        ],
    },
    "interview_doc": {
        "label": "Interview/doc",
        "selection_bias": "story arc, specificity, and emotional delivery",
        "ideal_line_words": (10, 55),
        "prefer_short_lines": False,
        "prefer_motion": False,
        "needs_punch_ins": False,
        "needs_cutaways": True,
        "notes": [
            "Build an emotional/logical arc, not just clean answers.",
            "Preserve meaningful pauses and expression changes.",
            "Use VLM/expression/story beats when available.",
        ],
    },
    "promo": {
        "label": "Promo",
        "selection_bias": "offer, benefit, proof, and CTA",
        "ideal_line_words": (5, 24),
        "prefer_short_lines": True,
        "prefer_motion": True,
        "needs_punch_ins": True,
        "needs_cutaways": True,
        "notes": [
            "Prioritize the offer/benefit/proof/CTA chain.",
            "Remove rambling setup unless it builds trust.",
            "End with a button or call to action.",
        ],
    },
    "fast_hype": {
        "label": "Fast hype",
        "selection_bias": "energy, action, rhythm, escalation",
        "ideal_line_words": (2, 14),
        "prefer_short_lines": True,
        "prefer_motion": True,
        "needs_punch_ins": True,
        "needs_cutaways": True,
        "notes": [
            "Favor short phrases, motion, and visual escalation.",
            "Coverage and rhythm can matter more than full sentence continuity.",
            "Use action/motion peaks aggressively, then let the skill decide final density.",
        ],
    },
    "sermon_event": {
        "label": "Sermon/event",
        "selection_bias": "chronology, clarity, and restraint",
        "ideal_line_words": (8, 70),
        "prefer_short_lines": False,
        "prefer_motion": False,
        "needs_punch_ins": False,
        "needs_cutaways": False,
        "notes": [
            "Preserve chronology unless the user explicitly asks for highlights.",
            "Tighten dead air carefully.",
            "Use VLM mostly for chaptering/context, not aggressive reconstruction.",
        ],
    },
}

_STYLE_ALIASES = {
    "clean": "clean_talking_head",
    "talking_head": "clean_talking_head",
    "clean punchy talking-head": "punchy_social",
    "social": "punchy_social",
    "short social cut": "punchy_social",
    "punchy": "punchy_social",
    "interview": "interview_doc",
    "doc": "interview_doc",
    "documentary": "interview_doc",
    "hype": "fast_hype",
    "event": "sermon_event",
    "sermon": "sermon_event",
}

_PROMO_TERMS = {"join", "come", "register", "sign", "today", "now", "free", "offer", "get", "try", "call", "visit"}
_HYPE_TERMS = {"go", "now", "win", "build", "move", "start", "fast", "big", "new", "best", "never"}
_INTERVIEW_TERMS = {"felt", "realized", "learned", "remember", "because", "changed", "hard", "honest", "moment"}

_CONTENT_TYPE_ALIASES = {
    "talking_head": "talking_head",
    "talking-head": "talking_head",
    "interview": "interview",
    "doc": "interview",
    "documentary": "interview",
    "promo": "promo",
    "ad": "promo",
    "commercial": "promo",
    "hype": "fast_hype",
    "fast_hype": "fast_hype",
    "sermon": "event_sermon",
    "event": "event_sermon",
    "event_sermon": "event_sermon",
    "podcast": "podcast",
    "tutorial": "tutorial",
    "product_demo": "product_demo",
    "demo": "product_demo",
    "vlog": "vlog",
}

_CUT_INTENT_ALIASES = {
    "narrative": "narrative",
    "story": "narrative",
    "peak": "peak_highlight",
    "highlight": "peak_highlight",
    "peak_highlight": "peak_highlight",
    "multi_clip": "multi_clip",
    "multiclip": "multi_clip",
    "assembled_short": "assembled_short",
    "short": "assembled_short",
    "surgical": "surgical_tighten",
    "tighten": "surgical_tighten",
    "surgical_tighten": "surgical_tighten",
}

_TIMELINE_MODE_ALIASES = {
    "raw": "raw_dump",
    "raw_dump": "raw_dump",
    "source": "raw_dump",
    "assembled": "assembled",
    "stringout": "assembled",
    "string_out": "assembled",
    "multicam": "multicam",
    "assembly": "assembly",
}


def transcript_embeddings_path(project_name: Any, media_id: Any, root: Optional[str] = None) -> Path:
    return motion_project_dir(project_name, root) / f"{motion_media_id(media_id)}_transcript_embeddings.json"


def ai_cut_plan_path(project_name: Any, plan_id: str, root: Optional[str] = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(plan_id or "ai_cut_plan")).strip("._") or "ai_cut_plan"
    return motion_project_dir(project_name, root) / "edit_plans" / f"{safe}.json"


def build_transcript_embeddings(
    transcript: Dict[str, Any],
    *,
    media_id: str,
    clip_id: str = "",
    clip_name: str = "",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    window_lines: int = 1,
    engine: str = "hashing-v1",
) -> Dict[str, Any]:
    """Build a lightweight semantic index from transcript lines.

    `hashing-v1` is not a neural embedding model, but it persists dense vectors
    and supports meaning-ish query expansion without downloads. The schema is
    designed so a sentence-transformer backend can replace the vector generator
    later without changing cut/search consumers.
    """

    dims = max(64, int(dimensions or DEFAULT_EMBEDDING_DIMENSIONS))
    lines = [line for line in transcript.get("lines") or [] if str(line.get("text") or "").strip()]
    entries: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        start_i = max(0, index - max(0, int(window_lines or 1) // 2))
        end_i = min(len(lines), index + max(1, int(window_lines or 1)))
        context_lines = lines[start_i:end_i]
        text = " ".join(str(item.get("text") or "").strip() for item in context_lines if str(item.get("text") or "").strip())
        vector = _hash_embedding(text, dims)
        entries.append(
            {
                "index": index,
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name or transcript.get("clip_name"),
                "text": str(line.get("text") or "").strip(),
                "context_text": text,
                "start": int(_num(line.get("start"), 0)),
                "end": int(_num(line.get("end"), _num(line.get("start"), 0))),
                "keywords": _keywords(text, limit=12),
                "vector": vector,
            }
        )
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "media_id": media_id,
        "clip_id": clip_id,
        "clip_name": clip_name or transcript.get("clip_name"),
        "fps": transcript.get("fps"),
        "duration_frames": transcript.get("duration_frames"),
        "engine": engine,
        "dimensions": dims,
        "analyzed_at": utc_now_iso(),
        "entry_count": len(entries),
        "entries": entries,
    }


def read_transcript_embeddings(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def query_transcript_embeddings(indexes: Sequence[Dict[str, Any]], query: str, limit: int = 10) -> List[Dict[str, Any]]:
    dims = DEFAULT_EMBEDDING_DIMENSIONS
    for payload in indexes:
        if payload.get("dimensions"):
            dims = int(payload.get("dimensions") or dims)
            break
    query_vector = _hash_embedding(query, dims)
    query_terms = set(_tokens(_expand_query(query)))
    results: List[Dict[str, Any]] = []
    for payload in indexes:
        for entry in payload.get("entries") or []:
            vector = entry.get("vector") or []
            score = _cosine(query_vector, vector)
            text_terms = set(_tokens(entry.get("context_text") or entry.get("text") or ""))
            overlap = len(query_terms & text_terms)
            if overlap:
                score += min(0.25, overlap * 0.04)
            results.append(
                {
                    "score": round(float(score), 4),
                    "clip_id": entry.get("clip_id") or payload.get("clip_id"),
                    "media_id": entry.get("media_id") or payload.get("media_id"),
                    "clip_name": entry.get("clip_name") or payload.get("clip_name"),
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "text": entry.get("text"),
                    "context_text": entry.get("context_text"),
                    "keywords": entry.get("keywords") or [],
                }
            )
    results.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    return results[: max(1, int(limit or 10))]


def resolve_edit_axes(
    *,
    content_type: str = "auto",
    cut_intent: str = "auto",
    target_duration: str = "",
    timeline_mode: str = "raw_dump",
    num_clips: int = 1,
    takes_already_scrubbed: bool = False,
    reorder_allowed: bool = True,
    style_key: str = "clean_talking_head",
) -> Dict[str, Any]:
    """Resolve CutMaster-style edit axes into deterministic planning rules.

    This keeps taste out of the MCP: the engine exposes axis-derived pacing,
    reorder, and selection strategy; the skill still applies thresholds.
    """

    content = _canonical_content_type(content_type, style_key)
    mode = _canonical_timeline_mode(timeline_mode)
    intent = _canonical_cut_intent(cut_intent)
    if intent == "auto":
        intent = _infer_cut_intent(
            content_type=content,
            target_duration=target_duration,
            timeline_mode=mode,
            num_clips=num_clips,
            takes_already_scrubbed=takes_already_scrubbed,
            style_key=style_key,
        )

    strategy = {
        "narrative": "narrative_arc",
        "peak_highlight": "peak_hunt",
        "multi_clip": "top_n_moments",
        "assembled_short": "montage_short",
        "surgical_tighten": "preserve_structure",
    }.get(intent, "narrative_arc")

    if not reorder_allowed or content in {"event_sermon", "tutorial"} or intent == "surgical_tighten":
        reorder_mode = "locked"
    elif intent in {"peak_highlight", "assembled_short"} and content in {"promo", "fast_hype", "vlog"}:
        reorder_mode = "free"
    elif intent == "multi_clip":
        reorder_mode = "per_clip_chronological"
    else:
        reorder_mode = "story_safe"

    pacing = _segment_pacing_for_axes(content, intent, style_key)
    warnings: List[str] = []
    if intent == "surgical_tighten" and mode not in {"assembled", "assembly", "multicam"}:
        warnings.append("surgical_tighten is safest on an already assembled timeline; raw source needs stronger transcript boundaries.")
    if content == "event_sermon" and reorder_mode != "locked":
        warnings.append("event/sermon content should normally preserve chronology.")
    if intent in {"peak_highlight", "assembled_short"} and not str(target_duration or "").strip():
        warnings.append("highlight/short intent works best with a target_duration.")

    return {
        "content_type": content,
        "cut_intent": intent,
        "timeline_mode": mode,
        "num_clips": int(num_clips or 0),
        "takes_already_scrubbed": bool(takes_already_scrubbed),
        "reorder_allowed": bool(reorder_allowed),
        "reorder_mode": reorder_mode,
        "selection_strategy": strategy,
        "segment_pacing": pacing,
        "warnings": warnings,
    }


def probe_source_preflight(file_path: Any, expected_fps: Optional[float] = None) -> Dict[str, Any]:
    """Small ffprobe preflight for frame-math risk such as VFR footage."""

    path = Path(str(file_path or "")).expanduser()
    result: Dict[str, Any] = {
        "file_path": str(path) if str(file_path or "") else "",
        "exists": path.exists() if str(file_path or "") else False,
        "ffprobe_available": bool(shutil.which("ffprobe")),
        "video": None,
        "vfr": {"is_vfr": None, "severity": "unknown"},
        "warnings": [],
    }
    if not str(file_path or ""):
        result["warnings"].append("No source file path available.")
        return result
    if not path.exists():
        result["warnings"].append("Source file is offline or not reachable from this machine.")
        return result
    if not result["ffprobe_available"]:
        result["warnings"].append("ffprobe not found; cannot verify constant frame rate.")
        return result
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            result["warnings"].append((proc.stderr or "ffprobe failed").strip()[:300])
            return result
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
        r_rate = _rate_to_float(stream.get("r_frame_rate"))
        avg_rate = _rate_to_float(stream.get("avg_frame_rate"))
        expected = float(expected_fps or 0.0) or None
        base = expected or avg_rate or r_rate or 0.0
        delta = abs((r_rate or 0.0) - (avg_rate or 0.0))
        ratio = delta / base if base else 0.0
        is_vfr = bool(r_rate and avg_rate and ratio > 0.001)
        severity = "high" if ratio > 0.01 else ("warn" if is_vfr else "pass")
        if is_vfr:
            result["warnings"].append(
                f"Potential VFR source: r_frame_rate={stream.get('r_frame_rate')} avg_frame_rate={stream.get('avg_frame_rate')}."
            )
        result.update(
            {
                "video": {
                    "codec": stream.get("codec_name"),
                    "width": _safe_int(stream.get("width")),
                    "height": _safe_int(stream.get("height")),
                    "duration_seconds": _safe_float(stream.get("duration")),
                    "nb_frames": _safe_int(stream.get("nb_frames")),
                    "r_frame_rate": stream.get("r_frame_rate"),
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                    "r_frame_rate_float": r_rate,
                    "avg_frame_rate_float": avg_rate,
                    "expected_fps": expected,
                },
                "vfr": {
                    "is_vfr": is_vfr,
                    "severity": severity,
                    "delta_fps": round(delta, 6),
                    "delta_ratio": round(ratio, 6),
                },
            }
        )
    except Exception as exc:
        result["warnings"].append(f"ffprobe preflight failed: {exc}")
    return result


def plan_ai_cut_from_sidecars(
    clips: Sequence[Dict[str, Any]],
    *,
    goal: str = "",
    style: str = "clean punchy talking-head",
    target_duration: str = "",
    undercut_bias: bool = True,
    content_type: str = "auto",
    cut_intent: str = "auto",
    timeline_mode: str = "raw_dump",
    takes_already_scrubbed: bool = False,
    reorder_allowed: bool = True,
    max_ranges: int = 12,
    max_punch_ins: int = 16,
    max_cutaways: int = 12,
) -> Dict[str, Any]:
    style_key = _canonical_style(style)
    style_profile = _style_profile(style_key)
    edit_axes = resolve_edit_axes(
        content_type=content_type,
        cut_intent=cut_intent,
        target_duration=target_duration,
        timeline_mode=timeline_mode,
        num_clips=len(clips),
        takes_already_scrubbed=takes_already_scrubbed,
        reorder_allowed=reorder_allowed,
        style_key=style_key,
    )
    selected_ranges: List[Dict[str, Any]] = []
    punch_ins: List[Dict[str, Any]] = []
    cutaways: List[Dict[str, Any]] = []
    clip_roles: List[Dict[str, Any]] = []
    inputs_used: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        media_id = str(clip.get("media_id") or "")
        clip_name = str(clip.get("clip_name") or "")
        transcript = clip.get("transcript") or {}
        visual = clip.get("visual") or {}
        if not transcript:
            warnings.append(f"{clip_name or clip_id}: missing transcript sidecar")
        if not visual:
            warnings.append(f"{clip_name or clip_id}: missing visual sidecar")
        inputs_used.append(_clip_inputs_used(clip, transcript, visual))
        role = classify_clip_role(visual, transcript)
        clip_roles.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "role": role,
                "reason": _clip_role_reason(visual, transcript),
            }
        )
        selected_ranges.extend(
            _selected_ranges_for_clip(
                clip_id,
                media_id,
                clip_name,
                transcript,
                visual,
                goal,
                role,
                style_key,
            )
        )
        punch_ins.extend(_punch_ins_for_clip(clip_id, media_id, clip_name, visual, transcript))
        cutaways.extend(_cutaways_for_clip(clip_id, media_id, clip_name, visual, role))

    redundancy_groups = _annotate_redundancy_groups(selected_ranges)
    selected_ranges.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    punch_ins.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    cutaways.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    all_candidate_counts = {
        "selected_ranges": len(selected_ranges),
        "punch_ins": len(punch_ins),
        "cutaways": len(cutaways),
    }
    return {
        "schema_version": CUT_PLAN_SCHEMA_VERSION,
        "plan_id": _plan_id(
            "|".join([goal, edit_axes.get("content_type", ""), edit_axes.get("cut_intent", ""), target_duration]),
            style,
            [clip.get("media_id") for clip in clips],
        ),
        "created_at": utc_now_iso(),
        "goal": goal,
        "style": style,
        "style_key": style_key,
        "target_duration": target_duration,
        "edit_axes": edit_axes,
        "edit_intent": {
            "style_key": style_key,
            "style_label": style_profile.get("label"),
            "content_type": edit_axes.get("content_type"),
            "cut_intent": edit_axes.get("cut_intent"),
            "timeline_mode": edit_axes.get("timeline_mode"),
            "selection_strategy": edit_axes.get("selection_strategy"),
            "reorder_mode": edit_axes.get("reorder_mode"),
            "selection_bias": style_profile.get("selection_bias"),
            "target_duration": target_duration or None,
            "undercut_bias": bool(undercut_bias),
            "style_notes": style_profile.get("notes") or [],
        },
        "status": "planned",
        "plan_semantics": {
            "type": "ranked_candidates",
            "thresholds_applied": False,
            "decision_owner": "assistant-editor skill",
            "note": "MCP scores candidates deterministically; the skill applies taste, under-cut bias, and approval thresholds.",
            "candidate_contract": "Candidates may include low-confidence or redundant options. Do not treat returned candidates as final keep/drop commands.",
        },
        "analysis_depth": _overall_analysis_depth(inputs_used),
        "inputs_used": inputs_used,
        "clip_roles": clip_roles,
        "redundancy_groups": redundancy_groups,
        "selected_ranges": selected_ranges[: max(1, int(max_ranges or 12))],
        "punch_ins": punch_ins[: max(1, int(max_punch_ins or 16))],
        "cutaways": cutaways[: max(1, int(max_cutaways or 12))],
        "candidate_counts": {
            **all_candidate_counts,
            "redundancy_groups": len(redundancy_groups),
            "returned_selected_ranges": min(all_candidate_counts["selected_ranges"], max(1, int(max_ranges or 12))),
            "returned_punch_ins": min(all_candidate_counts["punch_ins"], max(1, int(max_punch_ins or 16))),
            "returned_cutaways": min(all_candidate_counts["cutaways"], max(1, int(max_cutaways or 12))),
        },
        "quality_gates": _quality_gates_for_plan(
            style_key=style_key,
            target_duration=target_duration,
            analysis_depth=_overall_analysis_depth(inputs_used),
            candidate_counts=all_candidate_counts,
            selected_ranges=selected_ranges,
            punch_ins=punch_ins,
            cutaways=cutaways,
            redundancy_groups=redundancy_groups,
            clip_roles=clip_roles,
            inputs_used=inputs_used,
            edit_axes=edit_axes,
        ),
        "warnings": warnings,
        "apply_status": {
            "implemented": False,
            "reason": "This first pass persists a deterministic edit plan. Timeline assembly/application can consume this plan in a follow-up action.",
        },
    }


def _clip_inputs_used(clip: Dict[str, Any], transcript: Dict[str, Any], visual: Dict[str, Any]) -> Dict[str, Any]:
    vlm = visual.get("vlm") or {}
    summary = vlm.get("clip_summary") if isinstance(vlm.get("clip_summary"), dict) else {}
    return {
        "clip_id": clip.get("clip_id"),
        "media_id": clip.get("media_id"),
        "clip_name": clip.get("clip_name"),
        "transcript": {
            "available": bool(transcript),
            "line_count": len(transcript.get("lines") or []),
            "engine": transcript.get("engine"),
        },
        "visual": {
            "available": bool(visual),
            "tier": visual.get("tier"),
            "event_count": len(visual.get("events") or []),
            "has_expression": bool((visual.get("expression") or {}).get("events") or (visual.get("expression") or {}).get("dominant")),
            "has_objects": bool((visual.get("objects") or {}).get("per_clip")),
            "has_camera_motion": bool((visual.get("camera") or {}).get("timeline")),
            "has_vlm": bool(vlm.get("keyframes") or summary),
            "vlm_summary_mode": summary.get("summary_mode") or ("single" if summary else None),
            "vlm_status": summary.get("status"),
        },
        "embeddings": {
            "available": bool(clip.get("transcript_embeddings")),
            "engine": (clip.get("transcript_embeddings") or {}).get("engine") if isinstance(clip.get("transcript_embeddings"), dict) else None,
        },
        "source_preflight": clip.get("source_preflight") or {},
    }


def _overall_analysis_depth(inputs: Sequence[Dict[str, Any]]) -> str:
    if not inputs:
        return "none"
    has_transcript = any(((item.get("transcript") or {}).get("available")) for item in inputs)
    has_visual = any(((item.get("visual") or {}).get("available")) for item in inputs)
    has_vlm = any(((item.get("visual") or {}).get("has_vlm")) for item in inputs)
    has_embeddings = any(((item.get("embeddings") or {}).get("available")) for item in inputs)
    if has_transcript and has_visual and has_vlm and has_embeddings:
        return "transcript+embeddings+deep_visual"
    if has_transcript and has_visual and has_vlm:
        return "transcript+deep_visual"
    if has_transcript and has_visual:
        return "transcript+fast_visual"
    if has_visual and has_vlm:
        return "deep_visual_only"
    if has_visual:
        return "fast_visual_only"
    if has_transcript and has_embeddings:
        return "transcript+embeddings"
    if has_transcript:
        return "transcript_only"
    return "insufficient"


def _canonical_style(style: Any) -> str:
    raw = str(style or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in _STYLE_PROFILES:
        return raw
    loose = str(style or "").strip().lower()
    if loose in _STYLE_ALIASES:
        return _STYLE_ALIASES[loose]
    for key in _STYLE_PROFILES:
        if key in raw:
            return key
    if "promo" in raw:
        return "promo"
    if "hype" in raw or "fast" in raw:
        return "fast_hype"
    if "interview" in raw or "doc" in raw:
        return "interview_doc"
    if "sermon" in raw or "event" in raw:
        return "sermon_event"
    if "social" in raw or "punch" in raw:
        return "punchy_social"
    return "clean_talking_head"


def _canonical_content_type(content_type: Any, style_key: str) -> str:
    raw = str(content_type or "").strip().lower().replace(" ", "_")
    if not raw or raw == "auto":
        if style_key == "interview_doc":
            return "interview"
        if style_key == "promo":
            return "promo"
        if style_key == "fast_hype":
            return "fast_hype"
        if style_key == "sermon_event":
            return "event_sermon"
        return "talking_head"
    return _CONTENT_TYPE_ALIASES.get(raw, raw if raw in set(_CONTENT_TYPE_ALIASES.values()) else "unknown")


def _canonical_cut_intent(cut_intent: Any) -> str:
    raw = str(cut_intent or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw or raw == "auto":
        return "auto"
    return _CUT_INTENT_ALIASES.get(raw, "narrative")


def _canonical_timeline_mode(timeline_mode: Any) -> str:
    raw = str(timeline_mode or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return "raw_dump"
    return _TIMELINE_MODE_ALIASES.get(raw, raw if raw in set(_TIMELINE_MODE_ALIASES.values()) else "raw_dump")


def _infer_cut_intent(
    *,
    content_type: str,
    target_duration: str,
    timeline_mode: str,
    num_clips: int,
    takes_already_scrubbed: bool,
    style_key: str,
) -> str:
    if takes_already_scrubbed or timeline_mode == "assembled":
        return "surgical_tighten"
    if style_key in {"promo", "fast_hype", "punchy_social"} and _target_duration_seconds(target_duration, 0) <= 120:
        return "assembled_short"
    if int(num_clips or 0) > 1 and content_type in {"promo", "fast_hype", "vlog"}:
        return "multi_clip"
    if str(target_duration or "").strip() and _target_duration_seconds(target_duration, 99999) <= 90:
        return "peak_highlight"
    return "narrative"


def _segment_pacing_for_axes(content_type: str, cut_intent: str, style_key: str) -> Dict[str, Any]:
    if cut_intent == "surgical_tighten":
        return {"mode": "preserve_existing", "min_seconds": 2, "target_seconds": None, "max_seconds": None}
    if cut_intent == "peak_highlight":
        return {"mode": "highlight", "min_seconds": 3, "target_seconds": 8, "max_seconds": 18}
    if cut_intent == "assembled_short":
        return {"mode": "short_form", "min_seconds": 2, "target_seconds": 6, "max_seconds": 14}
    if cut_intent == "multi_clip":
        return {"mode": "multi_clip", "min_seconds": 4, "target_seconds": 12, "max_seconds": 24}
    if content_type in {"interview", "podcast", "event_sermon"}:
        return {"mode": "long_form_story", "min_seconds": 8, "target_seconds": 24, "max_seconds": 55}
    if style_key in {"promo", "fast_hype"}:
        return {"mode": "energetic", "min_seconds": 2, "target_seconds": 7, "max_seconds": 18}
    return {"mode": "clean_story", "min_seconds": 5, "target_seconds": 16, "max_seconds": 36}


def _target_duration_seconds(value: Any, default: int) -> int:
    text = str(value or "").strip().lower()
    if not text:
        return default
    nums = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return default
    seconds = max(nums)
    if "min" in text or (seconds <= 20 and "s" not in text and "sec" not in text):
        seconds *= 60
    return int(round(seconds))


def _style_profile(style_key: str) -> Dict[str, Any]:
    return dict(_STYLE_PROFILES.get(style_key) or _STYLE_PROFILES["clean_talking_head"])


def classify_clip_role(visual: Dict[str, Any], transcript: Dict[str, Any]) -> str:
    rollup = visual.get("metadata_rollup") or {}
    vlm = visual.get("vlm") or {}
    summary = " ".join(
        str(value or "")
        for value in [
            (vlm.get("clip_summary") or {}).get("clip_summary") if isinstance(vlm.get("clip_summary"), dict) else "",
            vlm.get("description"),
            rollup.get("description"),
            " ".join(rollup.get("keywords") or []),
        ]
    ).lower()
    line_text = " ".join(str(line.get("text") or "").strip() for line in transcript.get("lines") or [])
    text = " ".join(part for part in [str(transcript.get("text") or "").strip(), line_text] if part)
    word_count = len(_tokens(text))
    if any(term in summary for term in ["audience", "clapping", "applause", "reaction"]):
        return "reaction"
    if any(term in summary for term in ["wide", "establishing", "atrium", "building", "exterior", "staircase", "room"]):
        if word_count < 40:
            return "establishing"
        return "main_take_with_context"
    if word_count >= 8:
        return "main_take"
    if any(term in summary for term in ["podium", "microphone", "book", "stage", "screen"]):
        return "cutaway"
    return "supporting"


def _selected_ranges_for_clip(
    clip_id: str,
    media_id: str,
    clip_name: str,
    transcript: Dict[str, Any],
    visual: Dict[str, Any],
    goal: str,
    role: str,
    style_key: str,
) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    profile = _style_profile(style_key)
    goal_terms = set(_tokens(_expand_query(goal)))
    visual_events = visual.get("events") or []
    stills = [event for event in visual_events if event.get("type") == "still"]
    action_events = [event for event in visual_events if event.get("type") in {"hand_raise", "point", "head_turn_left", "head_turn_right", "high_motion"}]
    for index, line in enumerate(transcript.get("lines") or []):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        start = int(_num(line.get("start"), 0))
        end = int(_num(line.get("end"), start))
        words = _tokens(text)
        if len(words) < 4:
            continue
        overlap = len(set(words) & goal_terms) if goal_terms else 0
        filler_penalty = _filler_penalty(text)
        action = _nearest_event(action_events, (start + end) // 2, max_distance=180)
        still_conflict = any(_ranges_overlap(start, end, int(_num(event.get("start"), 0)), int(_num(event.get("end"), 0))) for event in stills)
        style_fit = _style_fit_for_line(text, len(words), bool(action), style_key, profile)
        base_score = 0.35 + min(0.25, len(words) / 80.0) + min(0.3, overlap * 0.08)
        score = base_score - filler_penalty + float(style_fit.get("score_adjustment") or 0.0)
        if action:
            score += 0.15
        if still_conflict:
            score -= 0.08
        if role.startswith("main_take"):
            score += 0.08
        score_breakdown = {
            "base": round(base_score, 4),
            "goal_overlap": overlap,
            "filler_penalty": round(filler_penalty, 4),
            "visual_action_bonus": 0.15 if action else 0.0,
            "stillness_penalty": 0.08 if still_conflict else 0.0,
            "main_take_bonus": 0.08 if role.startswith("main_take") else 0.0,
            "style_adjustment": round(float(style_fit.get("score_adjustment") or 0.0), 4),
        }
        ranges.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_in": start,
                "source_out": max(end, start + 1),
                "score": round(max(0.0, min(1.0, score)), 4),
                "score_breakdown": score_breakdown,
                "text": text,
                "reason": _range_reason(text, action, still_conflict, overlap),
                "style_fit": {key: value for key, value in style_fit.items() if key != "score_adjustment"},
                "visual_anchor": _event_anchor(action) if action else None,
                "line_index": index,
            }
        )
    return ranges


def _punch_ins_for_clip(clip_id: str, media_id: str, clip_name: str, visual: Dict[str, Any], transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    events = visual.get("events") or []
    shot_size = ((visual.get("shot_size") or {}).get("dominant") or (visual.get("metadata_rollup") or {}).get("shot_size") or "")
    for event in events:
        kind = str(event.get("type") or "")
        if kind not in {"hand_raise", "point", "head_turn_left", "head_turn_right", "high_motion"}:
            continue
        start = int(_num(event.get("start"), 0))
        peak = int(_num(event.get("peak"), start))
        end = int(_num(event.get("end"), peak))
        if peak <= start:
            punch_frame = peak or start
        else:
            punch_frame = int(round(start + 0.55 * (peak - start)))
        nearby_text = _nearest_transcript_text(transcript, punch_frame)
        base = {
            "hand_raise": 0.82,
            "point": 0.86,
            "high_motion": 0.72,
            "head_turn_left": 0.68,
            "head_turn_right": 0.68,
        }.get(kind, 0.6)
        out.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_frame": punch_frame,
                "event_start": start,
                "event_peak": peak,
                "event_end": end,
                "scale": _punch_scale_for_shot_size(str(shot_size)),
                "duration_frames": max(48, min(144, max(1, end - start))),
                "score": round(base, 4),
                "event_type": kind,
                "nearby_text": nearby_text,
                "reason": f"Punch in during {kind.replace('_', ' ')}; land around the gesture build/peak.",
            }
        )
    expr = (visual.get("expression") or {}).get("events") or []
    for event in expr:
        label = str(event.get("label") or event.get("type") or "").lower()
        if not any(term in label for term in ["smile", "happy", "laugh", "intense", "surprise"]):
            continue
        peak = int(_num(event.get("peak"), _num(event.get("start"), 0)))
        out.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_frame": peak,
                "event_start": int(_num(event.get("start"), peak)),
                "event_peak": peak,
                "event_end": int(_num(event.get("end"), peak + 72)),
                "scale": _punch_scale_for_shot_size(str(shot_size)),
                "duration_frames": 96,
                "score": 0.78,
                "event_type": label or "expression",
                "nearby_text": _nearest_transcript_text(transcript, peak),
                "reason": "Punch or hold on expression change; expression spikes often carry the beat better than words alone.",
            }
        )
    return out


def _cutaways_for_clip(clip_id: str, media_id: str, clip_name: str, visual: Dict[str, Any], role: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fps = float(_num(visual.get("fps"), 24.0)) or 24.0
    vlm = visual.get("vlm") or {}
    summary = vlm.get("clip_summary") if isinstance(vlm.get("clip_summary"), dict) else {}
    timeline = summary.get("visual_timeline") or []
    for segment in timeline:
        start_sec = _num(segment.get("start_seconds"), 0.0)
        end_sec = _num(segment.get("end_seconds"), start_sec + 3.0)
        text = str(segment.get("summary") or "").strip()
        if not text:
            continue
        score = 0.55
        lower = text.lower()
        if role in {"establishing", "reaction", "cutaway"}:
            score += 0.2
        if any(term in lower for term in ["wide", "staircase", "atrium", "audience", "clapping", "podium", "microphone"]):
            score += 0.15
        out.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_in": int(round(start_sec * fps)),
                "source_out": int(round(max(end_sec, start_sec + 2.0) * fps)),
                "score": round(min(1.0, score), 4),
                "role": role,
                "summary": text,
                "reason": str(segment.get("editorial_value") or "Useful visual context or transition coverage.").strip(),
            }
        )
    if not out and role in {"establishing", "reaction", "cutaway", "main_take_with_context"}:
        duration = int(_num(visual.get("duration_frames"), 0))
        out.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_in": 0,
                "source_out": min(duration or int(fps * 5), int(fps * 5)),
                "score": 0.55,
                "role": role,
                "summary": str((visual.get("metadata_rollup") or {}).get("description") or "").strip(),
                "reason": "Fallback cutaway candidate from clip role and visual rollup.",
            }
        )
    return out


def validate_ai_cut_plan_preview(plan: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if not isinstance(plan, dict) or not plan:
        errors.append({"id": "missing_plan", "message": "No plan payload supplied."})
        plan = {}
    if plan.get("schema_version") != CUT_PLAN_SCHEMA_VERSION:
        warnings.append(
            {
                "id": "schema_version",
                "message": f"Expected schema_version {CUT_PLAN_SCHEMA_VERSION}; got {plan.get('schema_version')}.",
            }
        )
    semantics = plan.get("plan_semantics") or {}
    if semantics.get("type") != "ranked_candidates":
        errors.append({"id": "plan_semantics", "message": "Plan must be a ranked_candidates contract, not a final command list."})
    if semantics.get("thresholds_applied") is not False:
        warnings.append({"id": "thresholds", "message": "Plan does not clearly declare thresholds_applied=false."})

    ranges = plan.get("selected_ranges") or []
    if not ranges:
        errors.append({"id": "no_selected_ranges", "message": "Plan has no selected range candidates."})
    for index, item in enumerate(ranges):
        source_in = int(_num(item.get("source_in"), -1))
        source_out = int(_num(item.get("source_out"), -1))
        if source_in < 0 or source_out <= source_in:
            errors.append({"id": "invalid_range", "index": index, "message": "selected_ranges source_out must be greater than source_in."})
        score = _num(item.get("score"), -1)
        if score < 0 or score > 1:
            warnings.append({"id": "score_range", "index": index, "message": "Candidate score should be normalized 0-1."})

    inputs = plan.get("inputs_used") or []
    has_transcript = any(((item.get("transcript") or {}).get("available")) for item in inputs)
    if not has_transcript:
        warnings.append({"id": "missing_transcript", "message": "No transcript sidecars were used; content selection is weak."})
    source_warnings = []
    for item in inputs:
        preflight = item.get("source_preflight") or {}
        vfr = preflight.get("vfr") or {}
        if vfr.get("is_vfr"):
            source_warnings.append(
                {
                    "clip_id": item.get("clip_id"),
                    "clip_name": item.get("clip_name"),
                    "severity": vfr.get("severity"),
                    "message": "Potential VFR source; source-frame math may drift unless conformed/proxied consistently.",
                }
            )
    warnings.extend({"id": "source_preflight", **warning} for warning in source_warnings)

    axes = plan.get("edit_axes") or {}
    for message in axes.get("warnings") or []:
        warnings.append({"id": "edit_axes", "message": str(message)})
    gates = plan.get("quality_gates") or []
    gate_warnings = [gate for gate in gates if gate.get("status") == "warn"]
    warnings.extend({"id": f"quality_gate:{gate.get('id')}", "message": gate.get("message")} for gate in gate_warnings)

    valid = not errors
    return {
        "success": valid,
        "valid": valid,
        "plan_id": plan.get("plan_id"),
        "schema_version": plan.get("schema_version"),
        "analysis_depth": plan.get("analysis_depth"),
        "edit_axes": axes,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "candidate_counts": plan.get("candidate_counts") or {},
        "recommendation": (
            "Plan contract is structurally usable; the skill should apply thresholds and under-cut bias."
            if valid
            else "Do not apply this plan until validation errors are fixed."
        ),
    }


def apply_ai_cut_plan_preview(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return an application preview.

    The actual timeline mutation should stay separate from planning. This
    placeholder gives agents a stable second action now without pretending that
    a full assembly edit was made.
    """

    validation = validate_ai_cut_plan_preview(plan)
    return {
        "success": False,
        "implemented": False,
        "plan_id": plan.get("plan_id"),
        "analysis_depth": plan.get("analysis_depth"),
        "validation": validation,
        "selected_range_count": len(plan.get("selected_ranges") or []),
        "punch_in_count": len(plan.get("punch_ins") or []),
        "cutaway_count": len(plan.get("cutaways") or []),
        "verification": verify_ai_cut_plan_preview(plan),
        "reason": "Timeline mutation is intentionally not enabled in this first pass. Use the plan as structured edit intelligence or wire it into the existing assembly tools next.",
    }


def verify_ai_cut_plan_preview(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": False,
        "implemented": False,
        "plan_id": plan.get("plan_id"),
        "validation": validate_ai_cut_plan_preview(plan),
        "expected": {
            "selected_ranges": len(plan.get("selected_ranges") or []),
            "punch_ins": len(plan.get("punch_ins") or []),
            "cutaways": len(plan.get("cutaways") or []),
        },
        "actual": None,
        "diff": None,
        "reason": "No timeline mutation has been implemented yet, so there is no applied timeline to diff. This is the stable verification contract for the future apply step.",
    }


def _hash_embedding(text: str, dimensions: int) -> List[float]:
    vector = [0.0] * max(1, int(dimensions))
    tokens = _tokens(_expand_query(text))
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % len(vector)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _expand_query(text: str) -> str:
    terms = _tokens(text)
    expanded = list(terms)
    for term in terms:
        expanded.extend(_SEMANTIC_EXPANSIONS.get(term, []))
    return " ".join(expanded)


def _tokens(text: Any) -> List[str]:
    raw = str(text or "").lower()
    tokens = [token.strip("'") for token in re.findall(r"[a-z][a-z']{1,}", raw)]
    return [token for token in tokens if token and token not in _STOPWORDS]


def _keywords(text: str, limit: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for token in _tokens(text):
        counts[token] = counts.get(token, 0) + 1
    return [token for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    na = math.sqrt(sum(float(value) * float(value) for value in a[:size])) or 1.0
    nb = math.sqrt(sum(float(value) * float(value) for value in b[:size])) or 1.0
    return dot / (na * nb)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "N/A":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "N/A":
            return None
        return int(float(value))
    except Exception:
        return None


def _rate_to_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text == "0/0":
        return None
    if "/" in text:
        left, right = text.split("/", 1)
        den = _safe_float(right)
        if not den:
            return None
        num = _safe_float(left)
        if num is None:
            return None
        return num / den
    return _safe_float(text)


def _filler_penalty(text: str) -> float:
    lower = str(text or "").lower()
    penalty = 0.0
    for filler in _FILLER_WORDS:
        if filler in lower:
            penalty += 0.025
    return min(0.15, penalty)


def _style_fit_for_line(text: str, word_count: int, has_action: bool, style_key: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    tokens = set(_tokens(text))
    low, high = profile.get("ideal_line_words") or (6, 40)
    adjustment = 0.0
    reasons: List[str] = []
    if low <= word_count <= high:
        adjustment += 0.04
        reasons.append("line length fits style")
    elif word_count > high:
        if profile.get("prefer_short_lines"):
            adjustment -= min(0.12, (word_count - high) * 0.006)
            reasons.append("longer than this style prefers")
        else:
            adjustment -= min(0.05, (word_count - high) * 0.002)
            reasons.append("long line; review pacing")
    elif word_count < low:
        adjustment -= 0.03
        reasons.append("short fragment; may need context")
    if profile.get("prefer_motion") and has_action:
        adjustment += 0.04
        reasons.append("visual motion supports energetic cut")
    if style_key == "promo" and tokens & _PROMO_TERMS:
        adjustment += 0.08
        reasons.append("promo/CTA language")
    if style_key == "fast_hype" and tokens & _HYPE_TERMS:
        adjustment += 0.08
        reasons.append("hype/action language")
    if style_key == "interview_doc" and tokens & _INTERVIEW_TERMS:
        adjustment += 0.06
        reasons.append("story/emotion language")
    if style_key == "sermon_event":
        adjustment += 0.02
        reasons.append("chronological/event candidate; skill should preserve order")
    return {
        "style_key": style_key,
        "word_count": int(word_count),
        "ideal_word_range": [int(low), int(high)],
        "reasons": reasons,
        "score_adjustment": round(adjustment, 4),
    }


def _annotate_redundancy_groups(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    assigned: set[int] = set()
    token_sets = [set(_tokens(item.get("text") or "")) for item in candidates]
    for index, tokens in enumerate(token_sets):
        if index in assigned or len(tokens) < 3:
            continue
        members = [index]
        for other_index in range(index + 1, len(candidates)):
            if other_index in assigned:
                continue
            other = token_sets[other_index]
            if len(other) < 3:
                continue
            union = tokens | other
            if not union:
                continue
            similarity = len(tokens & other) / len(union)
            containment = len(tokens & other) / max(1, min(len(tokens), len(other)))
            if similarity >= 0.48 or containment >= 0.72:
                members.append(other_index)
        if len(members) < 2:
            continue
        group_id = f"redundant_{len(groups) + 1:03d}"
        ranked = sorted(members, key=lambda idx: candidates[idx].get("score") or 0.0, reverse=True)
        for rank, member_index in enumerate(ranked, start=1):
            candidates[member_index]["redundancy_group_id"] = group_id
            candidates[member_index]["redundancy_rank"] = rank
        assigned.update(members)
        best = candidates[ranked[0]]
        groups.append(
            {
                "group_id": group_id,
                "candidate_count": len(members),
                "recommended_keep_count": 1,
                "decision_owner": "assistant-editor skill",
                "top_candidate": {
                    "clip_id": best.get("clip_id"),
                    "source_in": best.get("source_in"),
                    "source_out": best.get("source_out"),
                    "score": best.get("score"),
                    "text": best.get("text"),
                },
                "candidates": [
                    {
                        "clip_id": candidates[idx].get("clip_id"),
                        "source_in": candidates[idx].get("source_in"),
                        "source_out": candidates[idx].get("source_out"),
                        "score": candidates[idx].get("score"),
                        "text": candidates[idx].get("text"),
                    }
                    for idx in ranked
                ],
                "note": "Likely repeated idea. The MCP ranks options; the skill decides whether repetition is intentional.",
            }
        )
    return groups


def _quality_gates_for_plan(
    *,
    style_key: str,
    target_duration: str,
    analysis_depth: str,
    candidate_counts: Dict[str, int],
    selected_ranges: Sequence[Dict[str, Any]],
    punch_ins: Sequence[Dict[str, Any]],
    cutaways: Sequence[Dict[str, Any]],
    redundancy_groups: Sequence[Dict[str, Any]],
    clip_roles: Sequence[Dict[str, Any]],
    inputs_used: Sequence[Dict[str, Any]],
    edit_axes: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    profile = _style_profile(style_key)
    gates: List[Dict[str, Any]] = []
    gates.append(
        {
            "id": "style_target",
            "status": "pass",
            "message": f"Plan scored for {style_key}.",
            "detail": profile.get("selection_bias"),
        }
    )
    axes = edit_axes or {}
    gates.append(
        {
            "id": "edit_axes",
            "status": "warn" if axes.get("warnings") else "pass",
            "message": (
                f"{axes.get('content_type', 'unknown')} / {axes.get('cut_intent', 'unknown')} "
                f"using {axes.get('selection_strategy', 'unknown')} strategy."
            ),
            "detail": axes.get("warnings") or [],
        }
    )
    has_transcript = any(((item.get("transcript") or {}).get("available")) for item in inputs_used)
    gates.append(
        {
            "id": "transcript_basis",
            "status": "pass" if has_transcript else "warn",
            "message": "Transcript sidecars available for content selection." if has_transcript else "No transcript sidecars; content selection is weak.",
        }
    )
    has_deep = "deep_visual" in str(analysis_depth)
    if style_key in {"interview_doc", "promo"}:
        gates.append(
            {
                "id": "deep_visual_context",
                "status": "pass" if has_deep else "warn",
                "message": "Deep VLM context is available." if has_deep else "Deep VLM context missing; story/context judgment should be conservative.",
            }
        )
    if profile.get("needs_punch_ins"):
        gates.append(
            {
                "id": "punch_in_coverage",
                "status": "pass" if len(punch_ins) >= 3 else "warn",
                "message": f"{len(punch_ins)} punch-in candidates found.",
                "minimum_expected": 3,
            }
        )
    if profile.get("needs_cutaways"):
        gates.append(
            {
                "id": "coverage_cutaways",
                "status": "pass" if len(cutaways) > 0 else "warn",
                "message": f"{len(cutaways)} cutaway/context candidates found.",
            }
        )
    gates.append(
        {
            "id": "redundancy_review",
            "status": "warn" if redundancy_groups else "pass",
            "message": f"{len(redundancy_groups)} repeated-idea group(s) need skill thresholding." if redundancy_groups else "No obvious repeated-idea groups detected.",
        }
    )
    gates.append(
        {
            "id": "candidate_depth",
            "status": "pass" if candidate_counts.get("selected_ranges", 0) >= 3 else "warn",
            "message": f"{candidate_counts.get('selected_ranges', 0)} selected-range candidates found.",
        }
    )
    if target_duration:
        gates.append(
            {
                "id": "target_duration_declared",
                "status": "pass",
                "message": f"Target duration declared: {target_duration}. Skill should apply final density/runtime thresholds.",
            }
        )
    roles = {str(item.get("role") or "") for item in clip_roles}
    if style_key in {"promo", "fast_hype"} and roles <= {"main_take", "main_take_with_context"}:
        gates.append(
            {
                "id": "visual_variety",
                "status": "warn",
                "message": "Only main-take roles detected; promo/hype edits may need more coverage or stronger punch-ins.",
            }
        )
    return gates


def _nearest_event(events: Sequence[Dict[str, Any]], frame: int, max_distance: int) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_distance = max_distance + 1
    for event in events:
        peak = int(_num(event.get("peak"), _num(event.get("start"), 0)))
        distance = abs(int(frame) - peak)
        if distance < best_distance:
            best = event
            best_distance = distance
    return best if best_distance <= max_distance else None


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _event_anchor(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not event:
        return None
    return {
        "type": event.get("type"),
        "start": event.get("start"),
        "peak": event.get("peak"),
        "end": event.get("end"),
    }


def _range_reason(text: str, action: Optional[Dict[str, Any]], still_conflict: bool, overlap: int) -> str:
    parts = []
    if overlap:
        parts.append(f"matches the edit goal on {overlap} transcript term(s)")
    parts.append("complete transcript line with usable source-frame timing")
    if action:
        parts.append(f"near {str(action.get('type') or '').replace('_', ' ')}")
    if still_conflict:
        parts.append("has stillness overlap; use with care or hold deliberately")
    return "; ".join(parts)


def _nearest_transcript_text(transcript: Dict[str, Any], frame: int) -> str:
    best = ""
    best_distance = 10**9
    for line in transcript.get("lines") or []:
        start = int(_num(line.get("start"), 0))
        end = int(_num(line.get("end"), start))
        center = (start + end) // 2
        distance = 0 if start <= frame <= end else abs(center - frame)
        if distance < best_distance:
            best = str(line.get("text") or "").strip()
            best_distance = distance
    return best[:300]


def _punch_scale_for_shot_size(shot_size: str) -> float:
    value = str(shot_size or "").upper()
    if value in {"WS", "MWS"}:
        return 1.22
    if value in {"MS", "MCU"}:
        return 1.16
    if value in {"CU", "ECU"}:
        return 1.08
    return 1.18


def _clip_role_reason(visual: Dict[str, Any], transcript: Dict[str, Any]) -> str:
    role = classify_clip_role(visual, transcript)
    if role == "main_take":
        return "Transcript has enough spoken content to drive the edit."
    if role == "main_take_with_context":
        return "Spoken content plus wider/environmental visual context."
    if role == "establishing":
        return "Visual summary suggests location/context coverage with limited transcript content."
    if role == "reaction":
        return "Visual summary suggests audience/reaction/applause coverage."
    if role == "cutaway":
        return "Objects or context make this useful as coverage."
    return "Supporting visual evidence without a stronger role signal."


def _plan_id(goal: str, style: str, media_ids: Iterable[Any]) -> str:
    raw = "|".join([str(goal or ""), str(style or ""), *[str(item or "") for item in media_ids]])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"ai_cut_{digest}"
