import unittest

from src.server import (
    _audio_capabilities,
    _audio_mapping_report,
    _normalize_auto_sync_settings,
    _audio_track_probe,
    _probe_audio_item,
    _safe_auto_sync_audio,
    _safe_set_audio_properties,
    _subtitle_generation_probe,
    _transcription_capabilities,
    _voice_isolation_capabilities,
)


class MediaPoolItemStub:
    def __init__(self, name="audio.mov", item_id="mpi-1", clip_type="Video + Audio", synced_audio=""):
        self.name = name
        self.item_id = item_id
        self.properties = {
            "File Path": f"/tmp/{self.name}",
            "Type": clip_type,
            "Duration": "00:00:02:00",
            "Synced Audio": synced_audio,
            "Sound Roll #": synced_audio,
            "Audio Offset": "0",
            "Audio Ch": "1",
        }

    def GetName(self):
        return self.name

    def GetUniqueId(self):
        return self.item_id

    def GetClipProperty(self, key=""):
        if key:
            return self.properties.get(key, "")
        return dict(self.properties)

    def GetMediaId(self):
        return "media-1"

    def GetAudioMapping(self):
        return {"tracks": 1, "channels": 2}

    def TranscribeAudio(self):
        return True

    def ClearTranscription(self):
        return True


class TimelineItemStub:
    def __init__(self, media_pool_item=None):
        self.media_pool_item = media_pool_item or MediaPoolItemStub()
        self.props = {
            "Volume": 0,
            "Pan": 0,
            "AudioSyncOffsetIsManual": False,
            "AudioSyncOffset": 0,
        }

    def GetName(self):
        return "Audio Item"

    def GetUniqueId(self):
        return "item-1"

    def GetStart(self):
        return 0

    def GetEnd(self):
        return 48

    def GetDuration(self):
        return 48

    def GetLeftOffset(self):
        return 0

    def GetTrackTypeAndIndex(self):
        return ["audio", 1]

    def GetMediaPoolItem(self):
        return self.media_pool_item

    def GetProperty(self, key=""):
        if not key:
            return dict(self.props)
        return self.props.get(key)

    def SetProperty(self, key, value):
        self.props[key] = value
        return True

    def GetSourceAudioChannelMapping(self):
        return {"source": "stereo"}

    def GetVoiceIsolationState(self):
        return {"isEnabled": False, "amount": 0}

    def SetVoiceIsolationState(self, state):
        return True


class TimelineStub:
    def __init__(self):
        self.item = TimelineItemStub()
        self.subtitle_calls = []

    def GetTrackCount(self, track_type):
        return 1 if track_type in {"audio", "video"} else 0

    def GetItemListInTrack(self, track_type, track_index):
        return [self.item] if track_type in {"audio", "video"} and track_index == 1 else []

    def GetTrackSubType(self, track_type, track_index):
        return "stereo"

    def GetTrackName(self, track_type, track_index):
        return "A1"

    def GetIsTrackEnabled(self, track_type, track_index):
        return True

    def GetIsTrackLocked(self, track_type, track_index):
        return False

    def GetVoiceIsolationState(self, track_index):
        return {"isEnabled": False, "amount": 0}

    def SetVoiceIsolationState(self, track_index, state):
        return True

    def CreateSubtitlesFromAudio(self, settings):
        self.subtitle_calls.append(settings)
        return True


class FolderStub:
    def GetName(self):
        return "Master"

    def TranscribeAudio(self):
        return True

    def ClearTranscription(self):
        return True


class MediaPoolStub:
    def __init__(self, clip=None, clips=None):
        self.clip = clip or MediaPoolItemStub()
        self.clips = clips or [self.clip]
        self.auto_sync_calls = []

    def GetRootFolder(self):
        return self

    def GetClipList(self):
        return self.clips

    def GetSubFolderList(self):
        return []

    def GetCurrentFolder(self):
        return FolderStub()

    def GetSelectedClips(self):
        return self.clips

    def AutoSyncAudio(self, clips, settings=None):
        self.auto_sync_calls.append([clip.GetName() for clip in clips])
        for clip in clips:
            if "video" in str(clip.GetClipProperty("Type")).lower():
                clip.properties["Synced Audio"] = "lav.wav"
                clip.properties["Sound Roll #"] = "lav.wav"
                clip.properties["Audio Offset"] = "12.34"
        return True


class ResolveAudioSyncStub:
    AUDIO_SYNC_MODE = "key:mode"
    AUDIO_SYNC_CHANNEL_NUMBER = "key:channel"
    AUDIO_SYNC_RETAIN_EMBEDDED_AUDIO = "key:retain_audio"
    AUDIO_SYNC_RETAIN_VIDEO_METADATA = "key:retain_meta"

    AUDIO_SYNC_WAVEFORM = "mode:waveform"
    AUDIO_SYNC_CHANNEL_AUTOMATIC = -1


class AudioFairlightProbeTest(unittest.TestCase):
    def test_capabilities_include_voice_and_transcription(self):
        caps = _audio_capabilities()

        self.assertIn("track_state", caps["supported"])
        self.assertIn("transcription_subtitles", caps["partially_supported"])

    def test_audio_track_and_item_probe(self):
        timeline = TimelineStub()

        self.assertTrue(_audio_track_probe(timeline, {"track_index": 1})["available"])
        self.assertEqual(_probe_audio_item(timeline, {})["audio_properties"]["Volume"], 0)

    def test_safe_set_audio_properties_restores(self):
        timeline = TimelineStub()
        result = _safe_set_audio_properties(timeline, {"properties": {"Volume": -3}, "restore": True})

        self.assertTrue(result["success"])
        self.assertEqual(timeline.item.props["Volume"], 0)

    def test_voice_isolation_capabilities(self):
        result = _voice_isolation_capabilities(TimelineStub(), {})

        self.assertTrue(result["timeline_track"]["get_available"])
        self.assertTrue(result["item"]["get_available"])

    def test_audio_mapping_report(self):
        timeline = TimelineStub()
        pool = MediaPoolStub(timeline.item.media_pool_item)
        result = _audio_mapping_report(pool, timeline, {})

        self.assertEqual(len(result["timeline_items"]), 2)
        self.assertEqual(len(result["media_pool_items"]), 1)

    def test_auto_sync_dry_run_and_transcription_caps(self):
        pool = MediaPoolStub()
        sync = _safe_auto_sync_audio(pool, {"selected": True})
        transcription = _transcription_capabilities(pool, {"selected": True})

        self.assertTrue(sync["success"])
        self.assertTrue(sync["would_auto_sync"])
        self.assertTrue(transcription["clip_methods"][0]["transcribe_audio"])

    def test_auto_sync_append_alias_maps_to_retain_embedded_audio(self):
        settings = _normalize_auto_sync_settings(
            {"syncBy": "waveform", "channel": "auto", "append": True},
            ResolveAudioSyncStub(),
        )

        self.assertEqual(settings["key:mode"], "mode:waveform")
        self.assertEqual(settings["key:channel"], -1)
        self.assertTrue(settings["key:retain_audio"])

    def test_auto_sync_batches_multiple_cameras_against_same_lavs(self):
        cam_a = MediaPoolItemStub("cam-a.mov", "cam-a", "Video + Audio")
        cam_b = MediaPoolItemStub("cam-b.mov", "cam-b", "Video + Audio")
        lav_a = MediaPoolItemStub("lav-a.wav", "lav-a", "Audio")
        lav_b = MediaPoolItemStub("lav-b.wav", "lav-b", "Audio")
        pool = MediaPoolStub(clips=[cam_a, cam_b, lav_a, lav_b])

        dry = _safe_auto_sync_audio(pool, {"clip_ids": ["cam-a", "cam-b", "lav-a", "lav-b"], "dry_run": True})
        result = _safe_auto_sync_audio(pool, {"clip_ids": ["cam-a", "cam-b", "lav-a", "lav-b"], "dry_run": False})

        self.assertTrue(dry["batch_by_video"])
        self.assertEqual(len(dry["execution_plan"]), 2)
        self.assertTrue(result["batch_by_video"])
        self.assertEqual(pool.auto_sync_calls, [
            ["cam-a.mov", "lav-a.wav", "lav-b.wav"],
            ["cam-b.mov", "lav-a.wav", "lav-b.wav"],
        ])
        self.assertEqual(result["linked_count"], 2)
        self.assertEqual(result["linked"][0]["sound_roll"], "lav.wav")
        self.assertEqual(result["linked"][0]["audio_offset"], "12.34")

    def test_subtitle_generation_probe_is_dry_run_by_default(self):
        timeline = TimelineStub()
        result = _subtitle_generation_probe(timeline, {"settings": {"language": "en"}})

        self.assertTrue(result["success"])
        self.assertFalse(timeline.subtitle_calls)


if __name__ == "__main__":
    unittest.main()
