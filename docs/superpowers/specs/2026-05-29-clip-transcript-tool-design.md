# Design: `clip_transcript` MCP tool

**Date:** 2026-05-29
**Status:** Approved for implementation
**Author:** Claude (with christopherjohnogden)

## Summary

Add a new MCP tool, `clip_transcript`, that lets the server **read** audio
transcriptions from DaVinci Resolve clips. Today the MCP can only *trigger*
transcription (`transcribe_clip_audio` → `TranscribeAudio()` returns a bool) and
clear it; it cannot retrieve the resulting text. This tool closes that gap.

By default it returns transcript **text**. On request (`with_timecodes=true`) it
also returns frame-accurate **timecodes** at caption-chunk granularity.

## Motivation

The user wants an LLM to read interview/sermon transcripts and reason about them
(find soundbites, plan cuts, locate where something is said). That requires the
MCP to surface the transcript text — and, when planning cuts, the timecodes for
each span. The cut-*applying* step and per-word precision are explicitly out of
scope for this tool (see Non-Goals).

## Background: what the Resolve API exposes (verified live)

Established by live probing against the user's open project, not docs:

- `MediaPoolItem.GetClipProperty("Transcription")` returns the **full transcript
  text** (newline-separated, sentence-ish segments). Verified: returned 699+
  chars for a transcribed clip.
- `MediaPoolItem.GetClipProperty("Transcription Status")` returns `'Transcribed'`
  when a transcription exists (empty otherwise).
- The clip-property transcript has **no timecodes and no speaker labels**.
- Frame-accurate timecodes are available only via the **subtitle path**:
  `Timeline.CreateSubtitlesFromAudio({})` writes captions onto a subtitle track,
  and each subtitle item exposes `GetName()` (text), `GetStart()`, `GetEnd()`
  (timeline frames). Granularity is caption chunks (~6–8 words), not per-word.
  This call is **asynchronous** (items may not appear immediately) and **mutates
  the timeline** (adds a subtitle track).

Note: richer data (per-word timecodes + speaker labels) exists only inside the
project's `Project.db` as a zstd-compressed protobuf BLOB. Reading it is
unsupported, version-fragile, and risky while Resolve is running. It is **not**
used by this tool; it is recorded as a possible future enhancement.

## Scope

### In scope
- A single `clip_transcript(action, params)` tool with three actions: `list`,
  `get`, `get_all`.
- Text retrieval via the supported `GetClipProperty` path.
- Optional, opt-in timecode retrieval via the subtitle path.

### Non-goals (explicitly out)
- **Per-word timecodes / speaker labels** (the `Project.db` BLOB crack). Recorded
  as a future `deep_read` option; not built here.
- **Applying cuts** / building a new timeline from chosen in/out points. Separate
  future tool.
- **Auto-transcribing** clips that aren't transcribed yet — the existing
  `transcribe_clip_audio` tool already does that.

## Tool design

Location: `src/granular/media_pool_item.py`, alongside `transcribe_clip_audio`
and `clear_clip_transcription`. Follows existing conventions:
- `_get_mp()` → `(_, mp, err)` for media-pool access.
- `_find_clip_by_id(mp.GetRootFolder(), clip_id)` to resolve a clip.
- Returns plain dicts; errors as `{"error": ...}` consistent with neighboring
  tools in this file.

Signature:

```python
@mcp.tool()
def clip_transcript(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read audio transcriptions from clips.

    Actions:
      list() -> {clips: [{clip_id, name, status, char_count}], count}
      get(clip_id, with_timecodes=False)
          -> {name, status, text, lines?}
      get_all(with_timecodes=False)
          -> {transcripts: [{clip_id, name, status, text, lines?}], count}
    """
```

### Action: `list`
Walks the media pool (recursively, like other tools here), and for every clip
whose `GetClipProperty("Transcription Status") == "Transcribed"` returns
`{clip_id, name, status, char_count}`. Lets the caller discover what's available
before pulling full text. Returns `{clips: [...], count: N}`.

### Action: `get`
Params: `clip_id` (required), `with_timecodes` (optional, default `False`).

1. Resolve the clip; if not found → `{"error": "Clip <id> not found"}`.
2. Read `status = GetClipProperty("Transcription Status")`.
   - If status is not `"Transcribed"`: return
     `{name, status, text: None, note: "Clip is not transcribed. Run transcribe_clip_audio first."}`.
3. Read `text = GetClipProperty("Transcription")`.
4. If `with_timecodes` is `False`: return `{name, status, text}`.
5. If `with_timecodes` is `True`: additionally produce `lines` via the subtitle
   path (see "Timecode assembly"). Always include `text` so the caller still
   gets the transcript even if timecode generation is pending/unavailable.

### Action: `get_all`
Params: `with_timecodes` (optional, default `False`).
Runs `get`'s logic for every transcribed clip and returns
`{transcripts: [...], count: N}`. Intended for batch interview workflows where
the LLM reads everything at once. Non-transcribed clips are skipped (not errors).

`with_timecodes` is honored but **discouraged** for `get_all`: timecodes come
from the timeline's subtitle track, which reflects the *current timeline*, not an
arbitrary clip. So timecode assembly is only meaningful for clips actually on the
open timeline. For `get_all` with `with_timecodes=true`, the tool returns text for
all transcribed clips but only populates `lines` for clips present on the current
timeline; others get a `note` explaining timecodes need that clip on a timeline.
This avoids generating/cleaning many subtitle tracks in one call.

## Timecode assembly (`with_timecodes=true`)

This is the only path that mutates state, so it is opt-in and guarded.

1. Get the current timeline (`project.GetCurrentTimeline()`). If none →
   return text with `note: "No current timeline; timecodes require an open timeline."`
2. If a subtitle track with items already exists, read it (no mutation).
3. Otherwise call `timeline.CreateSubtitlesFromAudio({})`, then **poll**
   `GetItemListInTrack("subtitle", idx)` until items appear or a timeout
   (default ~30s, capped). `CreateSubtitlesFromAudio` is async; first call may
   return before items populate.
4. For each subtitle item, emit
   `{text: GetName(), start_tc, end_tc, start_frame}` where `*_tc` are formatted
   from `GetStart()/GetEnd()` using the timeline frame rate.
5. If items never populate within the timeout: return text plus
   `note: "Timecode generation pending; try again."` (do **not** fail the whole
   call — text is still useful).

Constraints surfaced honestly in the response:
- `granularity: "caption-chunk"` so the caller knows lines are ~6–8 words, not
  per-word.
- A flag indicating whether a subtitle track was created by this call.

Source timecode and speaker labels are NOT included (out of scope; documented).

## Error handling

- No media pool / no project: return the `_get_mp()` error envelope.
- Clip not found: `{"error": "Clip <id> not found"}` (matches neighbors).
- Not transcribed: informative `note`, not an error, pointing at
  `transcribe_clip_audio`.
- Timecode generation pending/unavailable: text returned with a `note`; never a
  hard failure.

## Testing

Unit tests (fakes, no live Resolve), in `tests/`:
- `list` returns only transcribed clips with correct fields.
- `get` returns text for a transcribed clip.
- `get` on a non-transcribed clip returns the informative note, not text.
- `get` on a missing clip returns the not-found error.
- `get_all` aggregates transcribed clips and skips non-transcribed ones.
- Timecode assembly: given fake subtitle items, lines carry text + formatted TC;
  empty/never-populating track yields the "pending" note with text still present.

Live verification (manual, against the user's transcribed `A0367` clip):
- `list` shows the clip as transcribed.
- `get` returns the expected sermon text.
- `get` with `with_timecodes=true` returns caption lines with frame-accurate TC,
  and cleans up appropriately per the chosen behavior.

## Future enhancements (not in this work)

- `deep_read=true`: per-word timecodes + speaker labels by parsing the
  `Project.db` `BtLockableBlob.FieldsBlob` (zstd + protobuf). Guarded, reads a
  copy, fails loudly on format-version mismatch.
- A companion tool to **apply** a cut list (in/out points → new timeline via
  `AppendToTimeline`).
