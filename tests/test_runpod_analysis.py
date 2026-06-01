import unittest
from unittest import mock

from src.utils.runpod_analysis import (
    RunPodConfigError,
    build_runpod_endpoint_base,
    build_runpod_file_url,
    build_runpod_visual_input,
    extract_visual_payload_from_runpod,
    resolve_runpod_network_volume_config,
    runpod_s3_endpoint_for_datacenter,
    runpod_staging_object_key,
    runpod_staging_retention_days,
    runpod_volume_path,
    stage_file_to_runpod_network_volume,
)


class RunPodAnalysisTests(unittest.TestCase):
    def test_endpoint_base_uses_explicit_endpoint_id(self):
        self.assertEqual(
            build_runpod_endpoint_base(endpoint_id="abc123"),
            "https://api.runpod.ai/v2/abc123",
        )

    def test_endpoint_base_accepts_full_run_url(self):
        self.assertEqual(
            build_runpod_endpoint_base(endpoint_url="https://api.runpod.ai/v2/abc123/run"),
            "https://api.runpod.ai/v2/abc123",
        )

    def test_endpoint_base_uses_env_endpoint_id(self):
        with mock.patch.dict("os.environ", {"RUNPOD_ANALYSIS_ENDPOINT_ID": "env-endpoint"}, clear=True):
            self.assertEqual(
                build_runpod_endpoint_base(),
                "https://api.runpod.ai/v2/env-endpoint",
            )

    def test_file_url_explicit_wins(self):
        self.assertEqual(
            build_runpod_file_url("/local/clip.mov", file_url="https://example.test/clip.mov"),
            "https://example.test/clip.mov",
        )

    def test_file_url_prefix_mapping_encodes_relative_path(self):
        url = build_runpod_file_url(
            "/Volumes/Media/Project A/Cam 01.mov",
            local_prefix="/Volumes/Media",
            url_prefix="https://cdn.example.test/media",
        )
        self.assertEqual(url, "https://cdn.example.test/media/Project%20A/Cam%2001.mov")

    def test_file_url_requires_mapping_for_local_paths(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RunPodConfigError):
                build_runpod_file_url("/Volumes/Media/clip.mov")

    def test_visual_input_preserves_local_sampling_settings(self):
        payload = build_runpod_visual_input(
            file_url="https://cdn.example.test/clip.mov",
            file_path="/Volumes/Media/clip.mov",
            media_id="m1",
            clip_name="clip.mov",
            fps=23.976,
            duration_frames=5400,
            tier="deep",
            sample_every_n=10,
            object_every_n=None,
            object_every_seconds=5.0,
            batch_size=1,
            proxy_width=640,
            pose_model="yolo11n-pose",
            object_model="yolo11n",
            expression=True,
            expression_every_n=2,
            vlm_model="qwen3-vl:8b",
            vlm_max_keyframes=12,
        )
        self.assertEqual(payload["job_type"], "resolve_clip_visual_analysis")
        self.assertEqual(payload["media"]["media_id"], "m1")
        self.assertIsNone(payload["media"]["volume_path"])
        self.assertEqual(payload["analysis"]["sample_every_n"], 10)
        self.assertEqual(payload["analysis"]["object_every_seconds"], 5.0)
        self.assertTrue(payload["output"]["allow_visual_sidecar_url"])
        self.assertEqual(payload["staging"]["retention_days"], 14)
        self.assertFalse(payload["staging"]["delete_after_analysis"])

    def test_visual_input_can_point_to_network_volume_path(self):
        payload = build_runpod_visual_input(
            file_url="runpod-volume://resolve/project/m1/clip.mov",
            file_path="/Volumes/Media/clip.mov",
            media_id="m1",
            clip_name="clip.mov",
            fps=23.976,
            duration_frames=5400,
            tier="fast",
            sample_every_n=10,
            object_every_n=None,
            object_every_seconds=5.0,
            batch_size=1,
            proxy_width=640,
            pose_model="yolo11n-pose",
            object_model="yolo11n",
            expression=True,
            expression_every_n=2,
            vlm_model=None,
            vlm_max_keyframes=12,
            staged_volume_path="/runpod-volume/resolve/project/m1/clip.mov",
            network_volume_id="vol123",
            staging_object_key="resolve/project/m1/clip.mov",
            analysis_proxy={"proxy_path": "/tmp/proxy.mp4", "frame_mapping": "one_proxy_frame_per_source_frame"},
        )
        self.assertEqual(payload["media"]["volume_path"], "/runpod-volume/resolve/project/m1/clip.mov")
        self.assertEqual(payload["media"]["network_volume_id"], "vol123")
        self.assertEqual(payload["media"]["staging_object_key"], "resolve/project/m1/clip.mov")
        self.assertEqual(payload["media"]["analysis_proxy"]["proxy_path"], "/tmp/proxy.mp4")

    def test_network_volume_endpoint_helpers(self):
        self.assertEqual(
            runpod_s3_endpoint_for_datacenter("US-NC-1"),
            "https://s3api-us-nc-1.runpod.io/",
        )
        self.assertEqual(
            runpod_volume_path("resolve/project/clip.mov"),
            "/runpod-volume/resolve/project/clip.mov",
        )

    def test_network_volume_config_uses_env_without_metadata_fetch(self):
        env = {
            "RUNPOD_NETWORK_VOLUME_ID": "vol123",
            "RUNPOD_NETWORK_VOLUME_ENDPOINT_URL": "https://s3api-us-nc-1.runpod.io",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            config = resolve_runpod_network_volume_config()
        self.assertEqual(config["network_volume_id"], "vol123")
        self.assertEqual(config["data_center_id"], "US-NC-1")
        self.assertEqual(config["s3_endpoint_url"], "https://s3api-us-nc-1.runpod.io/")

    def test_network_volume_stage_dry_run_builds_stable_paths(self):
        env = {
            "RUNPOD_NETWORK_VOLUME_ID": "vol123",
            "RUNPOD_NETWORK_VOLUME_DATACENTER_ID": "US-NC-1",
            "RUNPOD_NETWORK_VOLUME_ENDPOINT_URL": "https://s3api-us-nc-1.runpod.io",
            "RUNPOD_VOLUME_REMOTE_PREFIX": "resolve-mcp-staging",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            result = stage_file_to_runpod_network_volume(
                file_path="/Volumes/Media/Project A/Cam 01.mov",
                project_name="Project A",
                media_id="media:01",
                dry_run=True,
            )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["network_volume_id"], "vol123")
        self.assertEqual(result["s3_uri"], "s3://vol123/resolve-mcp-staging/Project-A/media-01/Cam 01.mov")
        self.assertEqual(result["volume_path"], "/runpod-volume/resolve-mcp-staging/Project-A/media-01/Cam 01.mov")
        self.assertEqual(result["file_url"], "runpod-volume://resolve-mcp-staging/Project-A/media-01/Cam 01.mov")

    def test_staging_object_key_sanitizes_project_and_media_id(self):
        self.assertEqual(
            runpod_staging_object_key(
                file_path="/Volumes/Media/Project A/Cam 01.mov",
                project_name="Project A",
                media_id="media:01",
                remote_prefix="prefix",
            ),
            "prefix/Project-A/media-01/Cam 01.mov",
        )

    def test_staging_retention_days_uses_env_and_defaults(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runpod_staging_retention_days(), 14)
        with mock.patch.dict("os.environ", {"RUNPOD_STAGING_RETENTION_DAYS": "21"}, clear=True):
            self.assertEqual(runpod_staging_retention_days(), 21)
        with mock.patch.dict("os.environ", {"RUNPOD_STAGING_RETENTION_DAYS": "bad"}, clear=True):
            self.assertEqual(runpod_staging_retention_days(), 14)

    def test_extract_visual_payload_accepts_nested_sidecar(self):
        payload = extract_visual_payload_from_runpod(
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "visual_sidecar": {
                        "schema_version": 1,
                        "media_id": "m1",
                        "events": [],
                        "metadata_rollup": {"keywords": ["speaker"]},
                    }
                },
            },
            expected_media_id="m1",
        )
        self.assertEqual(payload["analysis_backend"]["name"], "runpod")
        self.assertEqual(payload["analysis_backend"]["job_id"], "job-1")

    def test_extract_visual_payload_rejects_media_id_mismatch(self):
        with self.assertRaisesRegex(Exception, "media_id mismatch"):
            extract_visual_payload_from_runpod(
                {
                    "status": "COMPLETED",
                    "output": {
                        "visual_sidecar": {
                            "media_id": "other",
                            "events": [],
                            "metadata_rollup": {},
                        }
                    },
                },
                expected_media_id="m1",
            )


if __name__ == "__main__":
    unittest.main()
