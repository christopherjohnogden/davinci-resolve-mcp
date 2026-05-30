# Fix `clip_transcript` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the `clip_transcript` tool in the compound MCP server and make `get`/`get_all` return the full transcript instead of Resolve's ~699-char truncated preview.

**Architecture:** Add a truncation-detection helper and a subtitle-text joiner in `src/granular/media_pool_item.py`, wire them into the existing `_build_transcript_payload` so truncated property text falls back to the subtitle track, and register a standalone `clip_transcript` `@mcp.tool` in `src/server.py` that dispatches to the granular module.

**Tech Stack:** Python, FastMCP (`@mcp.tool`), `unittest` with fakes (no live Resolve in CI), DaVinci Resolve scripting API.

**Source spec:** `docs/superpowers/specs/2026-05-30-clip-transcript-fix-design.md`

**Branch:** `feat/clip-transcript-tool`

---

## File Structure

- `src/granular/media_pool_item.py` — add `_is_truncated`, `_join_subtitle_text`; modify `_build_transcript_payload` to fall back to subtitles on truncation. (existing file)
- `src/server.py` — import the granular module under an alias; register `@mcp.tool() clip_transcript` dispatching to it. (existing file)
- `tests/test_clip_transcript.py` — add unit tests for `_is_truncated`, `_join_subtitle_text`, and the truncation-fallback path in `_build_transcript_payload`. (existing file; follows its `unittest` + fakes pattern)

Run tests with: `venv/bin/python -m pytest tests/test_clip_transcript.py -v` (or `venv/bin/python -m unittest tests.test_clip_transcript -v`).

---

### Task 1: `_is_truncated` helper

**Files:**
- Modify: `src/granular/media_pool_item.py` (add helper near `_transcription_text`, ~line 845)
- Test: `tests/test_clip_transcript.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_clip_transcript.py` (before the `if __name__` block):

```python
class IsTruncatedTests(unittest.TestCase):
    def test_trailing_ellipsis_glyph_is_truncated(self):
        self.assertTrue(mpi._is_truncated("a long transcript…"))

    def test_trailing_three_dots_is_truncated(self):
        self.assertTrue(mpi._is_truncated("a long transcript..."))

    def test_trailing_whitespace_after_ellipsis_still_truncated(self):
        self.assertTrue(mpi._is_truncated("text…  \n"))

    def test_normal_text_not_truncated(self):
        self.assertFalse(mpi._is_truncated("a complete sentence."))

    def test_empty_or_none_not_truncated(self):
        self.assertFalse(mpi._is_truncated(""))
        self.assertFalse(mpi._is_truncated(None))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py::IsTruncatedTests -v`
Expected: FAIL — `module 'src.granular.media_pool_item' has no attribute '_is_truncated'`.

- [ ] **Step 3: Implement the helper**

Add to `src/granular/media_pool_item.py` immediately after the `_transcription_text` function (after ~line 851):

```python
# Resolve's "Transcription" clip property is a preview that is cut off with a
# trailing ellipsis for long transcripts; the full text lives in the subtitle
# track. The ellipsis is the reliable truncation signal (the ~699-char cap is
# undocumented and may vary by version).
def _is_truncated(text) -> bool:
    if not text:
        return False
    stripped = text.rstrip()
    return stripped.endswith("…") or stripped.endswith("...")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py::IsTruncatedTests -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/granular/media_pool_item.py tests/test_clip_transcript.py
git commit -m "feat(transcript): add _is_truncated helper for Resolve preview detection"
```

---

### Task 2: `_join_subtitle_text` helper

**Files:**
- Modify: `src/granular/media_pool_item.py` (add helper after `_transcript_timecode_lines`, before `_build_transcript_payload`, ~line 944)
- Test: `tests/test_clip_transcript.py`

`_transcript_timecode_lines` returns either `{"lines": [{"text", "start_tc", "end_tc", "start_frame"}, ...], ...}` or `{"note": "..."}`. This helper joins the `text` of each line into the full transcript, or returns `""` when no lines exist.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_clip_transcript.py`:

```python
class JoinSubtitleTextTests(unittest.TestCase):
    def test_joins_line_texts_with_newlines(self):
        tc = {"lines": [{"text": "First line."}, {"text": "Second line."}]}
        self.assertEqual(mpi._join_subtitle_text(tc), "First line.\nSecond line.")

    def test_skips_blank_segments(self):
        tc = {"lines": [{"text": "A"}, {"text": ""}, {"text": "B"}]}
        self.assertEqual(mpi._join_subtitle_text(tc), "A\nB")

    def test_note_only_returns_empty(self):
        self.assertEqual(mpi._join_subtitle_text({"note": "unavailable"}), "")

    def test_missing_lines_returns_empty(self):
        self.assertEqual(mpi._join_subtitle_text({}), "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py::JoinSubtitleTextTests -v`
Expected: FAIL — no attribute `_join_subtitle_text`.

- [ ] **Step 3: Implement the helper**

Add to `src/granular/media_pool_item.py` immediately before `_build_transcript_payload`:

```python
def _join_subtitle_text(tc: Dict[str, Any]) -> str:
    """Join the text of subtitle caption lines (from _transcript_timecode_lines)
    into the full transcript. Returns '' when no lines are present."""
    lines = (tc or {}).get("lines") or []
    parts = [str(ln.get("text", "")).strip() for ln in lines]
    return "\n".join(p for p in parts if p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py::JoinSubtitleTextTests -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/granular/media_pool_item.py tests/test_clip_transcript.py
git commit -m "feat(transcript): add _join_subtitle_text helper"
```

---

### Task 3: Untruncate `_build_transcript_payload`

**Files:**
- Modify: `src/granular/media_pool_item.py` (`_build_transcript_payload`, ~line 945)
- Test: `tests/test_clip_transcript.py`

Current body:

```python
def _build_transcript_payload(project, clip, *, with_timecodes: bool, wait_seconds: int) -> Dict[str, Any]:
    status = _transcription_status(clip)
    name = clip.GetName()
    if status != "Transcribed":
        return {
            "name": name,
            "status": status,
            "text": None,
            "note": "Clip is not transcribed. Run transcribe_clip_audio first.",
        }
    payload = {"name": name, "status": status, "text": _transcription_text(clip)}
    if with_timecodes:
        tc = _transcript_timecode_lines(project, clip, wait_seconds=wait_seconds)
        payload.update(tc)
    return payload
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_clip_transcript.py`. These reuse the existing `FakeClip`, `FakeTimeline`, `FakeProject`, `FakeSubtitleItem`/subtitle wiring, and `_Patch` defined earlier in the file (study `test_get_with_timecodes_generates_when_absent` for the subtitle-fake setup).

```python
class TranscriptTruncationFallbackTests(unittest.TestCase):
    def test_truncated_property_falls_back_to_subtitles(self):
        # Property text ends with the ellipsis Resolve appends → truncated.
        clip = FakeClip("c1", "Clip1", status="Transcribed", text="Hello world…")
        tl = FakeTimeline(subtitle_items=[
            FakeSubtitleItem("Hello world, this is the full thing.", 0, 24),
        ], fps="24")
        res = mpi._build_transcript_payload(
            FakeProject(tl), clip, with_timecodes=False, wait_seconds=1)
        self.assertEqual(res["source"], "subtitles")
        self.assertTrue(res["truncated"])
        self.assertEqual(res["text"], "Hello world, this is the full thing.")

    def test_complete_property_used_directly_no_subtitle_build(self):
        clip = FakeClip("c1", "Clip1", status="Transcribed", text="Short and complete.")
        # No timeline → if it tried subtitles it would note unavailability; it must not.
        res = mpi._build_transcript_payload(
            FakeProject(None), clip, with_timecodes=False, wait_seconds=1)
        self.assertEqual(res["source"], "property")
        self.assertFalse(res["truncated"])
        self.assertEqual(res["text"], "Short and complete.")

    def test_truncated_but_subtitles_unavailable_keeps_property(self):
        clip = FakeClip("c1", "Clip1", status="Transcribed", text="Partial…")
        res = mpi._build_transcript_payload(
            FakeProject(None), clip, with_timecodes=False, wait_seconds=1)  # no timeline
        self.assertEqual(res["source"], "property")
        self.assertTrue(res["truncated"])
        self.assertEqual(res["text"], "Partial…")
        self.assertIn("note", res)
```

If `FakeSubtitleItem` does not already exist in the file, add this minimal fake near the other fakes (it mirrors what `_transcript_timecode_lines` reads — `GetName`, `GetStart`, `GetEnd`):

```python
class FakeSubtitleItem:
    def __init__(self, text, start, end):
        self._text, self._start, self._end = text, start, end
    def GetName(self): return self._text
    def GetStart(self): return self._start
    def GetEnd(self): return self._end
```

And ensure `FakeTimeline` supports `subtitle_items` returning these via
`GetItemListInTrack("subtitle", idx)` and `GetTrackCount("subtitle")` — the
existing `test_get_with_timecodes_*` tests already rely on this wiring, so reuse
it as-is; only add `FakeSubtitleItem` if it is missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py::TranscriptTruncationFallbackTests -v`
Expected: FAIL — payload has no `source`/`truncated` keys; truncated text not replaced.

- [ ] **Step 3: Implement the change**

Replace the body of `_build_transcript_payload` in `src/granular/media_pool_item.py` with:

```python
def _build_transcript_payload(project, clip, *, with_timecodes: bool, wait_seconds: int) -> Dict[str, Any]:
    """Shared get/get_all body for one clip. Always returns text; replaces a
    truncated property preview with the full subtitle-track text when possible.
    Adds caption lines only when with_timecodes and they are available."""
    status = _transcription_status(clip)
    name = clip.GetName()
    if status != "Transcribed":
        return {
            "name": name,
            "status": status,
            "text": None,
            "note": "Clip is not transcribed. Run transcribe_clip_audio first.",
        }

    text = _transcription_text(clip)        # fast path: clip property
    source = "property"
    truncated = _is_truncated(text)
    lines = None
    note = None

    if truncated or with_timecodes:
        tc = _transcript_timecode_lines(project, clip, wait_seconds=wait_seconds)
        lines = tc.get("lines")
        full = _join_subtitle_text(tc)
        if truncated:
            if full:
                text, source = full, "subtitles"
            else:
                note = tc.get("note", "Full transcript unavailable; showing truncated preview.")

    payload = {"name": name, "status": status, "text": text,
               "source": source, "truncated": truncated}
    if note:
        payload["note"] = note
    if with_timecodes and lines is not None:
        payload["lines"] = lines
    return payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py -v`
Expected: the new `TranscriptTruncationFallbackTests` pass AND all pre-existing
`ClipTranscriptTests` still pass. (The existing `test_get_returns_text` asserts
`text` on a non-truncated clip; the new `source`/`truncated` keys are additive
and must not break it. If that test asserted an exact dict equality, update it to
also expect `source:"property", truncated:False` — check and adjust.)

- [ ] **Step 5: Commit**

```bash
git add src/granular/media_pool_item.py tests/test_clip_transcript.py
git commit -m "fix(transcript): return full text via subtitle fallback when property is truncated"
```

---

### Task 4: Register `clip_transcript` in the compound server

**Files:**
- Modify: `src/server.py` (add import alias near other granular usage; add `@mcp.tool` near the `media_pool_item` tool, ~line 13035)

No unit test for the FastMCP registration itself (consistent with how the other
compound tools in this file are untested at the registration layer); correctness
of dispatch is covered by Task 3's tests on the underlying function plus the
Task 5 live check. This task must not break server import.

- [ ] **Step 1: Add the granular module import alias**

In `src/server.py`, find where other granular helpers are imported at module
scope. Add (top-level, alongside other `from src...` imports, e.g. near line 105):

```python
from src.granular import media_pool_item as _granular_media_pool_item
```

(If an alias for this module already exists, reuse it instead of adding a new one — grep first: `grep -n "granular import media_pool_item\|granular.media_pool_item" src/server.py`.)

- [ ] **Step 2: Register the tool**

In `src/server.py`, immediately after the existing `media_pool_item` tool
function ends (it ends with its `return _unknown(...)` around line 13262, before
the next `@mcp.tool()` at ~13269), add:

```python
@mcp.tool()
def clip_transcript(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read and manage DaVinci Resolve audio transcriptions for media pool clips.

    Actions:
      list() -> {clips}
        All transcribed clips in the media pool.
      get(clip_id, with_timecodes=False, wait_seconds=30)
          -> {name, status, text, source, truncated, [lines]}
        Full transcript for one clip. `text` is sourced from the clip property,
        but if Resolve truncated it (long transcripts end with an ellipsis) the
        full text is read from the subtitle track. `source` is "property" or
        "subtitles"; `truncated` flags an incomplete property preview.
        with_timecodes=True also returns caption-chunk `lines` with frame-accurate
        timecodes (timeline-scoped; may generate a subtitle track on the current timeline).
      get_all(with_timecodes=False, wait_seconds=30) -> {clips}
        Same as get for every transcribed clip.
      transcribe(clip_ids | scope=mediapool|timeline|folder, skip_existing=True)
          -> {started, skipped, failed}
        Starts transcription (asynchronous). Poll with status.
      status(clip_ids | scope) -> {clips}
        Per-clip transcription status, for polling completion.

    Read-first: prefer get/list over media_pool_item metadata, which Resolve truncates.
    """
    return _granular_media_pool_item.clip_transcript(action, params)
```

- [ ] **Step 3: Verify the server still imports and the tool is registered**

Run:
```bash
venv/bin/python -c "import src.server as s; names=[t.name for t in __import__('asyncio').get_event_loop().run_until_complete(s.mcp.list_tools())]; print('clip_transcript' in names, len(names))"
```
Expected: prints `True <count>`.

If that introspection API differs in this FastMCP version, fall back to a syntax
+ attribute check:
```bash
venv/bin/python -c "import src.server as s; print(hasattr(s, 'clip_transcript'))"
```
Expected: `True`. Either way, the import must succeed with no exception.

- [ ] **Step 4: Run the full transcript test suite (no regressions)**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/server.py
git commit -m "feat(transcript): register clip_transcript as a compound MCP tool"
```

---

### Task 5: Live verification against Resolve (manual)

**Files:** none (manual; record results in the PR description).

Requires DaVinci Resolve Studio running with the project containing the
transcribed clip `20260521_Cam A0367.MP4`. Cannot run in CI.

- [ ] **Step 1: Full unit suite green**

Run: `venv/bin/python -m pytest tests/test_clip_transcript.py -v`
Expected: all pass.

- [ ] **Step 2: Confirm the tool is discoverable**

Start the compound server and confirm `clip_transcript` appears in the tool list
(via the MCP client / the Resolve chat panel, or `npx davinci-resolve-mcp` doctor
path you normally use).

- [ ] **Step 3: get on the known-truncated clip returns full text**

Call `clip_transcript(action="get", params={"clip_id": "<A0367 unique id>"})`.
Expected: `status:"Transcribed"`, `truncated:true`, `source:"subtitles"`,
`text` length ~1370 chars (the full transcript ending "...We'll see you in
November. (indistinct)"), NOT the 699-char `…` preview.

- [ ] **Step 4: get on a short clip uses the property**

Call `get` on a clip whose transcript is short/complete. Expected:
`truncated:false`, `source:"property"`.

- [ ] **Step 5: list enumerates transcribed clips**

Call `clip_transcript(action="list")`. Expected: includes the transcribed clips.

- [ ] **Step 6: Record results** (which clips, lengths, source values) in the PR.

---

## Notes for the implementer

- The ellipsis (`…` glyph or literal `...`) is the truncation contract — do not
  hard-code the ~699 byte cap.
- `_transcript_timecode_lines` is async (it polls after `CreateSubtitlesFromAudio`);
  the `wait_seconds` param flows through unchanged. Tests use small `wait_seconds`
  with fakes that return subtitles immediately.
- Subtitle generation mutates the current timeline (adds a caption track). This is
  pre-existing behavior of the `with_timecodes` path; the truncation fallback reuses
  it. Caching to avoid repeat generation is an explicit out-of-scope follow-up.
- Keep `source`/`truncated` additive to the payload so `get_all` consumers and the
  existing tests are not broken.
