from src.utils.edit_intelligence import (
    build_transcript_embeddings,
    plan_ai_cut_from_sidecars,
    query_transcript_embeddings,
)


def test_transcript_embeddings_support_meaning_query():
    transcript = {
        "lines": [
            {"text": "We keep trusting God when life feels heavy.", "start": 10, "end": 40},
            {"text": "The camera battery is on the table.", "start": 50, "end": 80},
        ]
    }
    index = build_transcript_embeddings(transcript, media_id="m1", clip_id="c1", clip_name="Clip")
    results = query_transcript_embeddings([index], "faith under pressure", limit=1)
    assert results[0]["text"] == "We keep trusting God when life feels heavy."
    assert results[0]["score"] > 0


def test_plan_ai_cut_combines_transcript_and_visual_events():
    clip = {
        "clip_id": "c1",
        "media_id": "m1",
        "clip_name": "Talk",
        "transcript": {
            "text": "This is the strong point.",
            "lines": [
                {"text": "This is the strong point about faith under pressure.", "start": 100, "end": 170},
            ],
        },
        "visual": {
            "fps": 24,
            "duration_frames": 500,
            "events": [
                {"type": "hand_raise", "start": 110, "peak": 140, "end": 180},
            ],
            "shot_size": {"dominant": "MS"},
            "vlm": {
                "clip_summary": {
                    "clip_summary": "A man speaks directly to camera and gestures.",
                    "visual_timeline": [
                        {
                            "start_seconds": 0,
                            "end_seconds": 5,
                            "summary": "A man speaks in a medium shot.",
                            "editorial_value": "Primary talking-head content.",
                        }
                    ],
                }
            },
            "metadata_rollup": {"keywords": ["talking-head"]},
        },
    }
    plan = plan_ai_cut_from_sidecars([clip], goal="faith under pressure")
    assert plan["plan_semantics"]["thresholds_applied"] is False
    assert plan["analysis_depth"] == "transcript+deep_visual"
    assert plan["inputs_used"][0]["transcript"]["available"] is True
    assert plan["inputs_used"][0]["visual"]["has_vlm"] is True
    assert plan["candidate_counts"]["punch_ins"] == 1
    assert plan["selected_ranges"][0]["clip_id"] == "c1"
    assert plan["punch_ins"][0]["source_frame"] == 126
    assert plan["clip_roles"][0]["role"] == "main_take"
