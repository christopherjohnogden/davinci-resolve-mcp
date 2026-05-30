# Fix `clip_transcript` — Register It and Return Full (Untruncated) Text

**Date:** 2026-05-30
**Status:** Design approved, pending spec review
**Branch:** `feat/clip-transcript-tool`

## Summary

The `clip_transcript` tool reads DaVinci Resolve audio transcriptions. Two bugs,
both verified live against Resolve Studio, prevent it from working as intended:

1. **Not registered.** `clip_transcript(action, params)` is fully implemented in
   `src/granular/media_pool_item.py`, but it is never registered as a tool in
   the compound `src/server.py`. MCP clients cannot call it, so an agent asking
   for a clip's transcript falls back to `media_pool_item` / `media_analysis`,
   which return Resolve's truncated metadata.
2. **Truncated text.** `get` / `get_all` read transcript text from
   `GetClipProperty("Transcription")`, which Resolve truncates to ~699
   characters with a trailing `…`. The complete transcript (e.g. 1370 chars for
   clip `20260521_Cam A0367.MP4`) only exists in the subtitle track produced by
   `CreateSubtitlesFromAudio`.

This fix registers the tool and makes `get`/`get_all` return the full text by
falling back to the subtitle track when the property is truncated.

## Evidence (verified 2026-05-30)

For clip `20260521_Cam A0367.MP4`:
- `GetClipProperty("Transcription")` → 699 chars, ends with `…` (truncated).
- `GetClipProperty("Transcription Status")` → `"Transcribed"`.
- Building a timeline + `CreateSubtitlesFromAudio()` → 52 subtitle items,
  1370 chars when joined (the full transcript).
- No other clip property (`Transcript`, `Subtitle`, `Caption`, `Captions`)
  holds the text.

## Goals

- `clip_transcript` is callable as a standalone MCP tool in the compound server.
- `get` and `get_all` return the complete transcript, not Resolve's ~699-char
  preview.
- Fast path preserved: when the property text is already complete, no subtitle
  generation is triggered.
- Callers can tell where the text came from and whether truncation was detected.

## Non-Goals

- No change to how transcription is *started* (`transcribe`, `transcribe_audio`).
- No change to the `with_timecodes` subtitle path beyond reuse.
- No new transcription backend; this only reads what Resolve already has.
- Not folding transcript actions into `media_pool_item` (kept standalone per
  decision below).

## Key Decisions

| Decision | Choice |
|---|---|
| Full-text source | **Property first, subtitle fallback on truncation** — `get` returns the fast property text, but detects truncation and falls back to the subtitle track for the complete text. |
| Registration | **Standalone `clip_transcript` tool** — its own `@mcp.tool` in `server.py` dispatching to the existing `media_pool_item.clip_transcript`. |

## Architecture

Three focused changes.

### 1. Register the tool (`src/server.py`)

Add, following the exact pattern of the neighboring `media_pool_item` tool
(`@mcp.tool()` decorator; function name becomes the tool name; `action` + `params`
signature; docstring enumerates actions):

```python
@mcp.tool()
def clip_transcript(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read and manage DaVinci Resolve audio transcriptions for media pool clips.

    Actions:
      list() -> {clips}                      # all transcribed clips
      get(clip_id, with_timecodes=False, wait_seconds=30) -> {name, status, text, source, truncated, [lines]}
      get_all(with_timecodes=False, wait_seconds=30) -> {clips}
      transcribe(clip_ids | scope, skip_existing=True) -> {started, skipped, failed}
      status(clip_ids | scope) -> {clips}

    get/get_all return the FULL transcript: text comes from the clip property,
    but if Resolve truncated it (~699 chars, trailing ellipsis) the full text is
    read from the subtitle track. `source` is "property" or "subtitles";
    `truncated` flags when the property was incomplete.
    """
    return _media_pool_item_module.clip_transcript(action, params)
```

Annotation: inherits the default `WRITE_TOOL` hint (the `transcribe` action
mutates Resolve), consistent with sibling tools — no entry in the
`external_tools`/`destructive_tools` lists is required.

The dispatch target is the already-imported granular module (the same module
object `server.py` already uses for `media_pool_item`). Confirm the import
alias during implementation; reuse it rather than adding a new import.

### 2. Untruncate `get`/`get_all` (`src/granular/media_pool_item.py`)

In `_build_transcript_payload`, after reading the property text, detect
truncation and fall back to subtitle-derived text:

```python
def _build_transcript_payload(project, clip, *, with_timecodes, wait_seconds):
    status = _transcription_status(clip)
    name = clip.GetName()
    if status != "Transcribed":
        return {"name": name, "status": status, "text": None,
                "note": "Clip is not transcribed. Run transcribe first."}

    text = _transcription_text(clip)          # fast path: clip property
    source = "property"
    truncated = _is_truncated(text)
    lines = None

    if truncated or with_timecodes:
        tc = _transcript_timecode_lines(project, clip, wait_seconds=wait_seconds)
        full = _join_subtitle_text(tc)        # complete text from subtitle items
        if full:
            if truncated:                     # only replace when property was short
                text, source = full, "subtitles"
            lines = tc.get("lines")

    payload = {"name": name, "status": status, "text": text,
               "source": source, "truncated": truncated}
    if with_timecodes and lines is not None:
        payload["lines"] = lines
    return payload
```

Notes:
- If subtitle generation fails or yields nothing, keep the property text with
  `source:"property"`, `truncated:true` — never error; partial beats none.
- `_join_subtitle_text(tc)` derives plain text from the same structure
  `_transcript_timecode_lines` already returns (the helper we used live).
  Implementation reads its `lines` (each carrying the segment text) and joins
  them with newlines; returns `""` if absent.

### 3. Testable truncation helper (`src/granular/media_pool_item.py`)

```python
# Resolve's "Transcription" clip property is a preview that is cut off (trailing
# ellipsis) for long transcripts; the full text lives in the subtitle track.
def _is_truncated(text: str) -> bool:
    if not text:
        return False
    return text.rstrip().endswith("…") or text.rstrip().endswith("...")
```

Ellipsis detection (both the `…` glyph and a literal `...`) is the reliable
signal — Resolve appends it. We deliberately do NOT hard-code the ~699 byte cap
(it is undocumented and may vary by version); the ellipsis is the contract.

## Data Flow (`get`)

```
get(clip_id)
  → status != "Transcribed"  → {text:null, note:"not transcribed"}
  → text = property text
  → _is_truncated(text)?
       no  → {text, source:"property", truncated:false}
       yes → subtitle track → join segments → full
              full?  yes → {text:full, source:"subtitles", truncated:true}
                     no  → {text, source:"property", truncated:true, note:...}
```

## Error Handling

- Clip not transcribed → existing `not transcribed` note, `text:null`.
- Subtitle generation fails / empty → fall back to property text, keep
  `truncated:true`, add a `note`. No exception escapes the action.
- Unknown `clip_id` / scope → existing `_resolve_target_clips` error behavior,
  unchanged.

## Testing

- **Unit (CI-safe):** `_is_truncated` — trailing `…` → true; trailing `...` →
  true; normal text → false; empty/None → false. `_join_subtitle_text` — joins
  segment text; returns `""` for missing/empty input.
- **Live verification (manual, against running Resolve Studio):** call the
  registered `clip_transcript(action="get", params={clip_id:<A0367>})` and
  confirm it returns the full ~1370-char transcript with
  `source:"subtitles", truncated:true`; call it on a short clip and confirm
  `source:"property", truncated:false`; confirm `list` enumerates transcribed
  clips and the tool is discoverable by name in the MCP tool list.

## Out of Scope / Follow-ups

- Caching subtitle-derived text to avoid regenerating on repeat `get`s (the
  subtitle build is the slow part). Note it; do not build it now (YAGNI) unless
  repeated calls prove painful in use.
