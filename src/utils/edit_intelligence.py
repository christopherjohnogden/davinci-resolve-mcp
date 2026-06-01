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


def plan_ai_cut_from_sidecars(
    clips: Sequence[Dict[str, Any]],
    *,
    goal: str = "",
    style: str = "clean punchy talking-head",
    max_ranges: int = 12,
    max_punch_ins: int = 16,
    max_cutaways: int = 12,
) -> Dict[str, Any]:
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
        selected_ranges.extend(_selected_ranges_for_clip(clip_id, media_id, clip_name, transcript, visual, goal, role))
        punch_ins.extend(_punch_ins_for_clip(clip_id, media_id, clip_name, visual, transcript))
        cutaways.extend(_cutaways_for_clip(clip_id, media_id, clip_name, visual, role))

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
        "plan_id": _plan_id(goal, style, [clip.get("media_id") for clip in clips]),
        "created_at": utc_now_iso(),
        "goal": goal,
        "style": style,
        "status": "planned",
        "plan_semantics": {
            "type": "ranked_candidates",
            "thresholds_applied": False,
            "decision_owner": "assistant-editor skill",
            "note": "MCP scores candidates deterministically; the skill applies taste, under-cut bias, and approval thresholds.",
        },
        "analysis_depth": _overall_analysis_depth(inputs_used),
        "inputs_used": inputs_used,
        "clip_roles": clip_roles,
        "selected_ranges": selected_ranges[: max(1, int(max_ranges or 12))],
        "punch_ins": punch_ins[: max(1, int(max_punch_ins or 16))],
        "cutaways": cutaways[: max(1, int(max_cutaways or 12))],
        "candidate_counts": {
            **all_candidate_counts,
            "returned_selected_ranges": min(all_candidate_counts["selected_ranges"], max(1, int(max_ranges or 12))),
            "returned_punch_ins": min(all_candidate_counts["punch_ins"], max(1, int(max_punch_ins or 16))),
            "returned_cutaways": min(all_candidate_counts["cutaways"], max(1, int(max_cutaways or 12))),
        },
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
) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
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
        score = 0.35 + min(0.25, len(words) / 80.0) + min(0.3, overlap * 0.08) - filler_penalty
        if action:
            score += 0.15
        if still_conflict:
            score -= 0.08
        if role.startswith("main_take"):
            score += 0.08
        ranges.append(
            {
                "clip_id": clip_id,
                "media_id": media_id,
                "clip_name": clip_name,
                "source_in": start,
                "source_out": max(end, start + 1),
                "score": round(max(0.0, min(1.0, score)), 4),
                "text": text,
                "reason": _range_reason(text, action, still_conflict, overlap),
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


def apply_ai_cut_plan_preview(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return an application preview.

    The actual timeline mutation should stay separate from planning. This
    placeholder gives agents a stable second action now without pretending that
    a full assembly edit was made.
    """

    return {
        "success": False,
        "implemented": False,
        "plan_id": plan.get("plan_id"),
        "analysis_depth": plan.get("analysis_depth"),
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


def _filler_penalty(text: str) -> float:
    lower = str(text or "").lower()
    penalty = 0.0
    for filler in _FILLER_WORDS:
        if filler in lower:
            penalty += 0.025
    return min(0.15, penalty)


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
