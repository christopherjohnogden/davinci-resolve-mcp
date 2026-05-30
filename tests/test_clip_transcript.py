"""Tests for the clip_transcript MCP tool (read transcriptions from clips).

Covers the supported read paths (text via GetClipProperty) and the opt-in
timecode assembly (caption-chunk lines from a subtitle track). No live Resolve;
fakes model the relevant Resolve objects.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.granular import media_pool_item as mpi


class FakeClip:
    def __init__(self, uid, name, status="", text="", file_path="/m/x.mp4",
                 transcribe_returns=True):
        self._uid = uid
        self._name = name
        self._props = {
            "Transcription Status": status,
            "Transcription": text,
            "File Path": file_path,
        }
        self._transcribe_returns = transcribe_returns
        self.transcribe_calls = 0

    def GetUniqueId(self):
        return self._uid

    def GetName(self):
        return self._name

    def GetClipProperty(self, key=None):
        return self._props.get(key, "")

    def TranscribeAudio(self, use_speaker_detection=None):
        self.transcribe_calls += 1
        # Simulate async start: status flips to a pending state, not Transcribed.
        if self._transcribe_returns:
            self._props["Transcription Status"] = "Transcribing"
        return self._transcribe_returns


class FakeTimelineItem:
    def __init__(self, media_pool_item):
        self._mpi = media_pool_item

    def GetMediaPoolItem(self):
        return self._mpi


class FakeSubtitleItem:
    def __init__(self, text, start, end):
        self._text, self._start, self._end = text, start, end

    def GetName(self):
        return self._text

    def GetStart(self):
        return self._start

    def GetEnd(self):
        return self._end


class FakeTimeline:
    def __init__(self, subtitle_items=None, fps="24", video_items=None):
        self._items = subtitle_items
        self._fps = fps
        self._video_items = video_items or []  # list of FakeTimelineItem
        self.create_called = 0

    def GetSetting(self, key):
        return self._fps if key == "timelineFrameRate" else None

    def GetTrackCount(self, ttype):
        if ttype == "subtitle":
            return 1 if self._items else 0
        if ttype == "video":
            return 1 if self._video_items else 0
        return 0

    def GetItemListInTrack(self, ttype, idx):
        if ttype == "subtitle" and self._items and idx == 1:
            return list(self._items)
        if ttype == "video" and self._video_items and idx == 1:
            return list(self._video_items)
        return []

    def CreateSubtitlesFromAudio(self, settings):
        self.create_called += 1
        # Simulate async generation populating the track.
        self._items = [
            FakeSubtitleItem("What's up guys?", 0, 24),
            FakeSubtitleItem("Glad you're here", 24, 60),
        ]
        return True


class FakeProject:
    def __init__(self, timeline=None):
        self._timeline = timeline

    def GetCurrentTimeline(self):
        return self._timeline


class _Patch:
    """Patch the module's _get_mp / get_all_media_pool_clips / _find_clip_by_id."""

    def __init__(self, clips, project=None):
        self._clips = clips
        self._project = project or FakeProject()

    def __enter__(self):
        self._orig = (mpi._get_mp, mpi.get_all_media_pool_clips, mpi._find_clip_by_id)

        class FakeMediaPool:
            def GetRootFolder(self):
                return None

        fake_mp = FakeMediaPool()
        mpi._get_mp = lambda: (self._project, fake_mp, None)
        mpi.get_all_media_pool_clips = lambda mp: list(self._clips)
        mpi._find_clip_by_id = lambda root, cid: next(
            (c for c in self._clips if c.GetUniqueId() == cid), None
        )
        return self

    def __exit__(self, *a):
        mpi._get_mp, mpi.get_all_media_pool_clips, mpi._find_clip_by_id = self._orig


class ClipTranscriptTests(unittest.TestCase):
    def test_list_returns_only_transcribed(self):
        clips = [
            FakeClip("a", "ClipA", status="Transcribed", text="hello world"),
            FakeClip("b", "ClipB", status="", text=""),
        ]
        with _Patch(clips):
            res = mpi.clip_transcript("list")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["clips"][0]["clip_id"], "a")
        self.assertEqual(res["clips"][0]["char_count"], len("hello world"))

    def test_get_returns_text(self):
        clips = [FakeClip("a", "ClipA", status="Transcribed", text="the transcript")]
        with _Patch(clips):
            res = mpi.clip_transcript("get", {"clip_id": "a"})
        self.assertEqual(res["status"], "Transcribed")
        self.assertEqual(res["text"], "the transcript")
        self.assertNotIn("lines", res)  # no timecodes unless requested

    def test_get_not_transcribed_returns_note(self):
        clips = [FakeClip("a", "ClipA", status="", text="")]
        with _Patch(clips):
            res = mpi.clip_transcript("get", {"clip_id": "a"})
        self.assertIsNone(res["text"])
        self.assertIn("transcribe_clip_audio", res["note"])

    def test_get_missing_clip_errors(self):
        with _Patch([]):
            res = mpi.clip_transcript("get", {"clip_id": "nope"})
        self.assertIn("not found", res["error"])

    def test_get_requires_clip_id(self):
        with _Patch([]):
            res = mpi.clip_transcript("get", {})
        self.assertIn("clip_id is required", res["error"])

    def test_get_all_aggregates_and_skips_untranscribed(self):
        clips = [
            FakeClip("a", "A", status="Transcribed", text="aaa"),
            FakeClip("b", "B", status="", text=""),
            FakeClip("c", "C", status="Transcribed", text="ccc"),
        ]
        with _Patch(clips):
            res = mpi.clip_transcript("get_all")
        self.assertEqual(res["count"], 2)
        ids = {t["clip_id"] for t in res["transcripts"]}
        self.assertEqual(ids, {"a", "c"})

    def test_unknown_action(self):
        with _Patch([]):
            res = mpi.clip_transcript("frobnicate")
        self.assertIn("Unknown action", res["error"])

    # --- timecode assembly ---

    def test_get_with_timecodes_reads_existing_subtitles(self):
        clips = [FakeClip("a", "A", status="Transcribed", text="t")]
        tl = FakeTimeline(subtitle_items=[
            FakeSubtitleItem("hello there", 0, 48),
        ], fps="24")
        with _Patch(clips, project=FakeProject(tl)):
            res = mpi.clip_transcript("get", {"clip_id": "a", "with_timecodes": True})
        self.assertEqual(tl.create_called, 0)  # existing track -> no mutation
        self.assertEqual(res["granularity"], "caption-chunk")
        self.assertFalse(res["subtitle_track_created"])
        self.assertEqual(res["lines"][0]["text"], "hello there")
        self.assertEqual(res["lines"][0]["start_tc"], "00:00:00:00")
        self.assertEqual(res["lines"][0]["end_tc"], "00:00:02:00")  # 48f @ 24fps

    def test_get_with_timecodes_generates_when_absent(self):
        clips = [FakeClip("a", "A", status="Transcribed", text="t")]
        tl = FakeTimeline(subtitle_items=None, fps="24")
        with _Patch(clips, project=FakeProject(tl)):
            res = mpi.clip_transcript("get", {"clip_id": "a", "with_timecodes": True})
        self.assertEqual(tl.create_called, 1)
        self.assertTrue(res["subtitle_track_created"])
        self.assertEqual(len(res["lines"]), 2)
        self.assertEqual(res["text"], "t")  # text still present alongside lines

    def test_get_with_timecodes_no_timeline(self):
        clips = [FakeClip("a", "A", status="Transcribed", text="t")]
        with _Patch(clips, project=FakeProject(None)):
            res = mpi.clip_transcript("get", {"clip_id": "a", "with_timecodes": True})
        self.assertIn("timeline", res["note"].lower())
        self.assertEqual(res["text"], "t")  # text always returned

    # --- transcribe ---

    def test_transcribe_explicit_clip_ids_starts_and_skips(self):
        a = FakeClip("a", "A", status="")               # needs transcription
        b = FakeClip("b", "B", status="Transcribed")    # already done -> skipped
        with _Patch([a, b]):
            res = mpi.clip_transcript("transcribe", {"clip_ids": ["a", "b"]})
        self.assertEqual(res["count_started"], 1)
        self.assertEqual(res["started"][0]["clip_id"], "a")
        self.assertEqual(a.transcribe_calls, 1)
        self.assertEqual(b.transcribe_calls, 0)         # skipped, not called
        self.assertEqual(res["skipped"][0]["clip_id"], "b")

    def test_transcribe_scope_mediapool(self):
        clips = [FakeClip("a", "A", status=""), FakeClip("b", "B", status="")]
        with _Patch(clips):
            res = mpi.clip_transcript("transcribe", {"scope": "mediapool"})
        self.assertEqual(res["count_started"], 2)

    def test_transcribe_scope_timeline_maps_to_media_pool_items(self):
        a = FakeClip("a", "A", status="")
        b = FakeClip("b", "B", status="")
        tl = FakeTimeline(video_items=[FakeTimelineItem(a), FakeTimelineItem(b),
                                       FakeTimelineItem(a)])  # dup a
        with _Patch([a, b], project=FakeProject(tl)):
            res = mpi.clip_transcript("transcribe", {"scope": "timeline"})
        # 'a' appears twice on the timeline but should be deduped.
        self.assertEqual(res["count_started"], 2)
        self.assertEqual(a.transcribe_calls, 1)

    def test_transcribe_failure_reported(self):
        a = FakeClip("a", "A", status="", transcribe_returns=False)
        with _Patch([a]):
            res = mpi.clip_transcript("transcribe", {"clip_ids": ["a"]})
        self.assertEqual(res["count_started"], 0)
        self.assertEqual(len(res["failed"]), 1)

    def test_transcribe_no_skip_runs_all(self):
        a = FakeClip("a", "A", status="Transcribed")
        with _Patch([a]):
            res = mpi.clip_transcript("transcribe", {"clip_ids": ["a"], "skip_existing": False})
        self.assertEqual(res["count_started"], 1)
        self.assertEqual(a.transcribe_calls, 1)

    # --- status ---

    def test_status_reports_per_clip(self):
        clips = [FakeClip("a", "A", status="Transcribed"),
                 FakeClip("b", "B", status="")]
        with _Patch(clips):
            res = mpi.clip_transcript("status", {"clip_ids": ["a", "b"]})
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["transcribed"], 1)
        self.assertFalse(res["all_done"])

    def test_status_all_done(self):
        clips = [FakeClip("a", "A", status="Transcribed")]
        with _Patch(clips):
            res = mpi.clip_transcript("status", {"clip_ids": ["a"]})
        self.assertTrue(res["all_done"])

    def test_unknown_scope_errors(self):
        with _Patch([]):
            res = mpi.clip_transcript("transcribe", {"scope": "bogus"})
        self.assertIn("Unknown scope", res["error"])


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


if __name__ == "__main__":
    unittest.main()
