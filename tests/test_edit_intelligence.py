from src.utils.edit_intelligence import (
    build_transcript_embeddings,
    plan_ai_cut_from_sidecars,
    query_transcript_embeddings,
    resolve_edit_axes,
    validate_ai_cut_plan_preview,
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


def test_plan_ai_cut_exposes_style_intent_quality_gates_and_redundancy():
    clip = {
        "clip_id": "c1",
        "media_id": "m1",
        "clip_name": "Promo",
        "transcript": {
            "lines": [
                {"text": "Register today for the free event.", "start": 10, "end": 40},
                {"text": "Sign up today for the free event.", "start": 50, "end": 80},
                {"text": "This is a longer setup line about why the event matters to families.", "start": 90, "end": 150},
            ],
        },
        "visual": {
            "fps": 24,
            "duration_frames": 240,
            "events": [{"type": "high_motion", "start": 14, "peak": 24, "end": 45}],
            "metadata_rollup": {"keywords": ["talking-head"]},
        },
    }
    plan = plan_ai_cut_from_sidecars(
        [clip],
        goal="event registration",
        style="promo",
        target_duration="60-90s",
        undercut_bias=True,
    )
    assert plan["style_key"] == "promo"
    assert plan["edit_intent"]["undercut_bias"] is True
    assert plan["target_duration"] == "60-90s"
    assert plan["edit_axes"]["content_type"] == "promo"
    assert plan["edit_axes"]["cut_intent"] == "assembled_short"
    assert plan["edit_axes"]["selection_strategy"] == "montage_short"
    assert plan["selected_ranges"][0]["style_fit"]["style_key"] == "promo"
    assert plan["selected_ranges"][0]["score_breakdown"]["style_adjustment"] > 0
    assert plan["candidate_counts"]["redundancy_groups"] == 1
    assert plan["redundancy_groups"][0]["recommended_keep_count"] == 1
    gate_ids = {gate["id"] for gate in plan["quality_gates"]}
    assert {"style_target", "edit_axes", "target_duration_declared", "redundancy_review"} <= gate_ids


def test_edit_axes_can_represent_surgical_tighten_without_taste_thresholds():
    axes = resolve_edit_axes(
        content_type="interview",
        cut_intent="surgical_tighten",
        timeline_mode="assembled",
        target_duration="",
        num_clips=1,
        takes_already_scrubbed=True,
        style_key="interview_doc",
    )
    assert axes["cut_intent"] == "surgical_tighten"
    assert axes["selection_strategy"] == "preserve_structure"
    assert axes["reorder_mode"] == "locked"
    assert axes["segment_pacing"]["mode"] == "preserve_existing"


def test_validate_ai_cut_plan_flags_contract_and_vfr_warnings():
    clip = {
        "clip_id": "c1",
        "media_id": "m1",
        "clip_name": "Talk",
        "source_preflight": {
            "vfr": {"is_vfr": True, "severity": "warn"},
        },
        "transcript": {
            "lines": [
                {"text": "This is a useful line for the cut.", "start": 12, "end": 48},
            ],
        },
        "visual": {"events": []},
    }
    plan = plan_ai_cut_from_sidecars([clip], content_type="talking_head", cut_intent="narrative")
    validation = validate_ai_cut_plan_preview(plan)
    assert validation["valid"] is True
    assert validation["warning_count"] >= 1
    assert any(item["id"] == "source_preflight" for item in validation["warnings"])
