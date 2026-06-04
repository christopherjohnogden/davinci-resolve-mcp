"""Focused tests for the native After Effects decoder helpers.

These tests use synthetic semantic layer/property dictionaries instead of real
.aep files. They lock down the decoder contract while the binary chunk mapping
is still being validated with real AE fixture projects.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AE_NATIVE_EXTRACT_PATH = PROJECT_ROOT / "scripts" / "ae_native_extract.py"

spec = importlib.util.spec_from_file_location("ae_native_extract", AE_NATIVE_EXTRACT_PATH)
assert spec and spec.loader
ae_native_extract = importlib.util.module_from_spec(spec)
sys.modules["ae_native_extract"] = ae_native_extract
spec.loader.exec_module(ae_native_extract)


def prop(match_name: str, value: Any, name: str | None = None) -> dict[str, Any]:
    return {
        "type": "property",
        "name": name,
        "matchName": match_name,
        "value": {"value": value},
    }


def group(match_name: str, children: list[dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    return {
        "type": "group",
        "name": name,
        "matchName": match_name,
        "children": children,
    }


class AENativeDecoderTest(unittest.TestCase):
    def test_text_style_override_from_sidecar_strings_decodes_all_caps(self):
        override = ae_native_extract.text_style_override_from_sidecar_strings(
            [
                "Core - Lower Third",
                "en_US",
                "Core - Lower Third",
                "Notes",
                "en_US",
                "Notes",
                "4e28bf87-2f79-484f-be62-fa97fad42aec",
                "The consequence of sin is always later and greater.",
                "The consequence of sin is always later and greater.",
                (
                    '{"capPropFontEdit":false,"fontEditValue":"BrandonGrotesque-Bold",'
                    '"fontFSAllCapsValue":true,"fontFSSmallCapsValue":false,'
                    '"fontSizeEditValue":106}'
                ),
                (
                    '{"capPropFontEdit":false,"fontEditValue":"BebasNeueBold",'
                    '"fontFSAllCapsValue":false,"fontFSSmallCapsValue":false,'
                    '"fontSizeEditValue":104}'
                ),
            ]
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(override["compName"], "Core - Lower Third")
        self.assertEqual(override["layerName"], "Notes")
        self.assertTrue(override["style"]["fontFSAllCapsValue"])
        self.assertEqual(override["style"]["fontSizeEditValue"], 106)

    def test_layer_switch_summary_exposes_raw_flags_and_sdk_candidates(self):
        raw_flags = (1 << 3) | (1 << 6) | (1 << 9) | (1 << 18)

        summary = ae_native_extract.layer_switch_summary(
            {"rawLayerFlags": raw_flags},
            "Layr",
            {"kind": "text"},
            [{"kind": "fill"}],
            {"id": 12, "name": "Source"},
            {"postLayerStreams": {"streams": [{"index": 1, "streamId": 42, "typeName": "AE Layer", "flags": {"fiop": 1}}]}},
        )

        self.assertEqual(summary["rawLayerFlagsHex"], "0x00040248")
        self.assertEqual(summary["layerKind"], "renderLayer")
        self.assertEqual(summary["visualKind"], "text")
        self.assertEqual(summary["effectKinds"], ["fill"])
        self.assertIn(3, summary["setBitPositions"])
        self.assertTrue(summary["friendlyLayerSwitchCandidates"]["motionBlur"])
        self.assertTrue(summary["friendlyLayerSwitchCandidates"]["shy"])
        self.assertTrue(summary["friendlyLayerSwitchCandidates"]["adjustmentLayer"])
        self.assertTrue(summary["friendlyLayerSwitchCandidates"]["guideLayer"])
        self.assertFalse(summary["friendlyLayerSwitchCandidates"]["solo"])
        self.assertIn("motionBlur", summary["enabledFriendlyLayerSwitchCandidates"])
        self.assertIn("guideLayer", summary["enabledFriendlyLayerSwitchCandidates"])
        candidate_names = {item["name"] for item in summary["aegpLayerFlagNameCandidates"]}
        self.assertIn("AEGP_LayerFlag_MOTION_BLUR", candidate_names)
        self.assertIn("AEGP_LayerFlag_SHY", candidate_names)
        self.assertIn("AEGP_LayerFlag_ADJUSTMENT_LAYER", candidate_names)
        self.assertIn("AEGP_LayerFlag_GUIDE_LAYER", candidate_names)
        self.assertTrue(summary["postLayerStreams"])
        self.assertEqual(summary["postLayerStreamHints"][0]["nonZeroFlags"], {"fiop": 1})
        self.assertEqual(summary["postLayerStreamHintSummary"]["nonZeroFlagKinds"], ["fiop"])

    def test_track_matte_candidate_is_attached_to_adjacent_layer(self):
        layers = [
            {
                "name": "Background MATTE",
                "ordinal": 7,
                "visual": {"kind": "shape"},
                "timing": {"rawLayerFlags": 0},
                "compositingHints": {"postLayerStreams": []},
            },
            {
                "name": "Background",
                "ordinal": 8,
                "visual": {"kind": "shape"},
                "timing": {"rawLayerFlags": 1},
                "compositingHints": {"postLayerStreams": []},
            },
        ]

        ae_native_extract.apply_layer_compositing_relationships(layers)

        self.assertEqual(layers[0]["transfer"]["usedAsTrackMatteCandidateFor"]["layerName"], "Background")
        self.assertEqual(layers[1]["transfer"]["trackMatteCandidate"]["matteLayerName"], "Background MATTE")
        self.assertEqual(layers[1]["transfer"]["trackMatte"], "candidate")

    def test_parent_layer_summary_decodes_parent_index(self):
        properties = group(
            "ADBE Root",
            [prop("ADBE Parent", [1], "Parent")],
        )

        summary = ae_native_extract.layer_parent_summary(properties)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["parentIndex"], 1)
        self.assertEqual(summary["parentIndexBase"], "ae-layer-index-candidate")
        self.assertEqual(summary["raw"]["value"], [1])

    def test_parent_layer_relationship_resolves_parent_name_and_graph_edge(self):
        layers = [
            {
                "name": "Parent Null",
                "ordinal": 1,
                "parentComp": {"itemId": 10, "name": "Main"},
                "visual": {"kind": "layer"},
                "timing": {},
            },
            {
                "name": "Child Text",
                "ordinal": 2,
                "parentComp": {"itemId": 10, "name": "Main"},
                "visual": {"kind": "text"},
                "timing": {},
                "parentIndex": 1,
                "parentLayer": {
                    "decodeStatus": "best-effort-parent-layer",
                    "parentIndex": 1,
                    "parentIndexBase": "ae-layer-index-candidate",
                },
            },
        ]

        ae_native_extract.apply_layer_parent_relationships(layers)
        graph = ae_native_extract.project_graph([], layers)
        summary = ae_native_extract.layer_conversion_summary(layers[1])

        self.assertEqual(layers[1]["parentLayer"]["parentName"], "Parent Null")
        self.assertEqual(layers[1]["parentLayer"]["parentOrdinal"], 1)
        self.assertEqual(layers[0]["childLayers"][0]["name"], "Child Text")
        self.assertEqual(graph["parentLayerEdges"][0]["fromLayerName"], "Child Text")
        self.assertEqual(graph["parentLayerEdges"][0]["toLayerName"], "Parent Null")
        self.assertEqual(summary["parentLayer"]["parentName"], "Parent Null")

    def test_transfer_mode_summary_exposes_stream_hints(self):
        layer = {
            "name": "Fill Layer",
            "compositingHints": {
                "postLayerStreams": [
                    {
                        "index": 1,
                        "streamId": 101,
                        "typeName": "Track Matte Mode",
                        "flags": {"fmat": 2, "fifl": 1},
                    },
                    {
                        "index": 2,
                        "streamId": 102,
                        "typeName": "Blend Opacity",
                        "flags": {"fiop": 0},
                    },
                ]
            },
        }

        summary = ae_native_extract.transfer_mode_summary(layer)

        self.assertEqual(summary["blendMode"], "normal")
        self.assertEqual(summary["trackMatte"], "alphaInverted")
        self.assertEqual(summary["decodedTrackMatteCandidates"][0]["name"], "alphaInverted")
        self.assertTrue(summary["trackMatteStreamCandidate"])
        self.assertTrue(summary["blendModeStreamCandidate"])
        self.assertTrue(summary["frameBlendingStreamFlagCandidate"])
        self.assertEqual(
            summary["transferStreamHints"][0]["decodedTransferCandidates"][0]["kind"],
            "trackMatte",
        )
        self.assertEqual(summary["transferStreamHints"][0]["semanticNameSignals"], ["matte", "track", "mode"])
        self.assertEqual(summary["transferStreamHintSummary"]["nonZeroFlagKinds"], ["fifl", "fmat"])
        self.assertEqual(
            summary["transferStreamHintSummary"]["decodedTransferCandidateCounts"],
            {"trackMatte": 1},
        )
        self.assertEqual(
            summary["transferStreamHintSummary"]["semanticNameSignals"],
            ["blend", "matte", "mode", "opacity", "track"],
        )

    def test_transfer_mode_summary_decodes_blend_mode_candidates(self):
        layer = {
            "name": "Screened Layer",
            "compositingHints": {
                "postLayerStreams": [
                    {
                        "index": 1,
                        "streamId": 201,
                        "typeName": "Blend Mode",
                        "flags": {"fbmd": 9},
                    }
                ]
            },
        }

        summary = ae_native_extract.transfer_mode_summary(layer)

        self.assertEqual(summary["blendMode"], "screen")
        self.assertEqual(summary["blendModeConfidence"], "raw-transfer-stream-code-candidate")
        self.assertEqual(summary["decodedBlendModeCandidates"][0]["code"], 9)
        self.assertEqual(summary["decodedBlendModeCandidates"][0]["name"], "screen")
        self.assertEqual(
            summary["transferStreamHintSummary"]["decodedTransferCandidateCounts"],
            {"blendMode": 1},
        )

    def test_keyframe_easing_summary_preserves_temporal_and_spatial_candidates(self):
        keyframes = [
            {
                "index": 0,
                "localFrame": 0,
                "value": [960.0, 540.0],
                "interpolationHint": "bezier",
                "recordProbe": {
                    "temporalEaseHeaderCandidates": {
                        "bytes4to32": "02020000000000000000000000000000000000000000000000000000",
                        "u16": [514, 0, 0, 0],
                        "u32": [33685504, 0],
                    },
                    "spatialTangentCandidates": {
                        "preValuePairs": [{"offset": 24, "value": -12.0}, {"offset": 32, "value": 0.0}],
                        "postValuePairs": [{"offset": 80, "value": 12.0}, {"offset": 88, "value": 0.0}],
                        "confidence": "raw-slot-candidate",
                    },
                },
            },
            {
                "index": 1,
                "localFrame": 12,
                "value": [1200.0, 540.0],
                "interpolationHint": "linear",
                "recordProbe": {
                    "temporalEaseHeaderCandidates": {
                        "bytes4to32": "01010000000000000000000000000000000000000000000000000000",
                        "u16": [257, 0, 0, 0],
                        "u32": [16842752, 0],
                    },
                },
            },
        ]

        summary = ae_native_extract.keyframe_easing_summary(keyframes)

        self.assertEqual(summary["decodeStatus"], "best-effort-easing-candidates")
        self.assertEqual(summary["candidateCount"], 2)
        self.assertEqual(summary["temporalEaseHeaderCandidateCount"], 2)
        self.assertEqual(summary["spatialTangentCandidateCount"], 1)
        self.assertTrue(summary["hasTemporalEaseHeaderCandidates"])
        self.assertTrue(summary["hasSpatialTangentCandidates"])
        self.assertEqual(summary["interpolationHints"], {"bezier": 1, "linear": 1})
        self.assertIn("temporalEaseHeaderCandidate", summary["candidateKeyframes"][0])
        self.assertIn("spatialTangentCandidate", summary["candidateKeyframes"][0])

    def test_normalized_keyframes_keep_easing_candidates(self):
        entry = {
            "matchName": "ADBE Position",
            "animation": {
                "keyframes": [
                    {
                        "index": 0,
                        "localFrame": 0,
                        "value": [960.0, 540.0],
                        "interpolationHint": "bezier",
                        "recordProbe": {
                            "temporalEaseHeaderCandidates": {
                                "bytes4to32": "02020000000000000000000000000000000000000000000000000000",
                                "u16": [514, 0, 0, 0],
                                "u32": [33685504, 0],
                            }
                        },
                    }
                ]
            },
        }
        parent_comp = {"composition": {"width": 1920, "height": 1080}}

        keyframes = ae_native_extract.normalized_keyframes_for_entry(entry, parent_comp)

        self.assertEqual(keyframes[0]["value"], [0.5, 0.5])
        self.assertEqual(keyframes[0]["interpolationHint"], "bezier")
        self.assertEqual(keyframes[0]["easingCandidate"]["interpolationHint"], "bezier")
        self.assertIn("temporalEaseHeaderCandidate", keyframes[0]["easingCandidate"])

    def test_mask_summary_decodes_common_mask_fields(self):
        properties = group(
            "ADBE Root",
            [
                group(
                    "ADBE Mask Atom",
                    [
                        prop("ADBE Mask Shape", [[0, 0], [100, 0], [100, 100], [0, 100]], "Mask Path"),
                        prop("ADBE Mask Feather", [8, 8], "Mask Feather"),
                        prop("ADBE Mask Opacity", [75], "Mask Opacity"),
                        prop("ADBE Mask Offset", [12], "Mask Expansion"),
                        prop("ADBE Mask Mode", [1], "Mask Mode"),
                        prop("ADBE Mask Inverted", [1], "Inverted"),
                    ],
                    "Mask 1",
                )
            ],
        )

        masks = ae_native_extract.mask_summaries(properties)

        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0]["name"], "Mask 1")
        self.assertEqual(masks[0]["shape"]["value"], [[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertEqual(masks[0]["shape"]["pathMetadata"]["vertexCount"], 4)
        self.assertFalse(masks[0]["shape"]["pathMetadata"]["hasTangents"])
        self.assertEqual(masks[0]["feather"]["value"], [8, 8])
        self.assertEqual(masks[0]["opacity"]["value"], [75])
        self.assertEqual(masks[0]["pathDecodeStatus"], "bezier-path-decoded")

    def test_mask_summary_preserves_bezier_tangents(self):
        path = {
            "vertices": [[0, 0], [100, 0], [100, 100]],
            "inTangents": [[0, 0], [-20, 0], [0, 20]],
            "outTangents": [[20, 0], [0, 20], [0, 0]],
            "closed": "false",
        }
        properties = group(
            "ADBE Root",
            [
                group(
                    "ADBE Mask Atom",
                    [prop("ADBE Mask Shape", path, "Mask Path")],
                    "Bezier Mask",
                )
            ],
        )

        masks = ae_native_extract.mask_summaries(properties)

        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0]["shape"]["value"], path)
        self.assertEqual(masks[0]["shape"]["pathMetadata"]["vertexCount"], 3)
        self.assertTrue(masks[0]["shape"]["pathMetadata"]["hasTangents"])
        self.assertFalse(masks[0]["shape"]["pathMetadata"]["closed"])
        self.assertEqual(masks[0]["shape"]["pathMetadata"]["inTangentCount"], 3)
        self.assertEqual(masks[0]["shape"]["pathMetadata"]["outTangentCount"], 3)
        self.assertEqual(masks[0]["pathDecodeStatus"], "bezier-path-with-tangents-decoded")

    def test_vector_shape_summary_preserves_custom_bezier_path(self):
        properties = group(
            "ADBE Root",
            [
                group(
                    "ADBE Vector Group",
                    [
                        group("ADBE Vector Shape - Group", [], "Path 1"),
                        prop("ADBE Vector Shape", [[0, 0], [50, 100], [100, 0]], "Path"),
                        prop("ADBE Vector Fill Color", [0.1, 0.2, 0.3, 1.0], "Fill Color"),
                    ],
                    "Custom Shape",
                )
            ],
        )

        shapes = ae_native_extract.vector_shape_summaries(properties)

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["type"], "path")
        self.assertEqual(shapes[0]["path"]["value"], [[0, 0], [50, 100], [100, 0]])
        self.assertEqual(shapes[0]["path"]["pathMetadata"]["vertexCount"], 3)
        self.assertFalse(shapes[0]["path"]["pathMetadata"]["hasTangents"])
        self.assertEqual(shapes[0]["pathDecodeStatus"], "bezier-path-decoded")
        self.assertEqual(shapes[0]["fill"]["value"], [0.1, 0.2, 0.3, 1.0])

    def test_vector_shape_summary_preserves_custom_bezier_path_tangents(self):
        path = {
            "vertices": [[0, 0], [50, 100], [100, 0]],
            "in_tangents": [[0, 0], [-18, 0], [0, 18]],
            "out_tangents": [[18, 0], [0, 18], [0, 0]],
            "closed": True,
        }
        properties = group(
            "ADBE Root",
            [
                group(
                    "ADBE Vector Group",
                    [
                        group("ADBE Vector Shape - Group", [], "Path 1"),
                        prop("ADBE Vector Shape", path, "Path"),
                    ],
                    "Bezier Shape",
                )
            ],
        )

        shapes = ae_native_extract.vector_shape_summaries(properties)

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["path"]["value"], path)
        self.assertEqual(shapes[0]["path"]["pathMetadata"]["vertexCount"], 3)
        self.assertTrue(shapes[0]["path"]["pathMetadata"]["hasTangents"])
        self.assertTrue(shapes[0]["path"]["pathMetadata"]["closed"])
        self.assertEqual(shapes[0]["path"]["pathMetadata"]["inTangentCount"], 3)
        self.assertEqual(shapes[0]["path"]["pathMetadata"]["outTangentCount"], 3)
        self.assertEqual(shapes[0]["pathDecodeStatus"], "bezier-path-with-tangents-decoded")

    def test_text_paragraph_hints_normalize_style_fields(self):
        payload = "/Justification 2 /Leading 42 /Tracking -20 /Baseline 3 /BoxText true"

        hints = ae_native_extract.text_paragraph_hints(payload)

        self.assertEqual(hints["justificationCandidates"][0]["labelCandidate"], "center")
        self.assertEqual(hints["paragraphStyle"]["justification"]["code"], 2)
        self.assertEqual(hints["paragraphStyle"]["justification"]["label"], "center")
        self.assertEqual(hints["paragraphStyle"]["leading"]["value"], 42)
        self.assertEqual(hints["paragraphStyle"]["tracking"]["value"], -20)
        self.assertEqual(hints["paragraphStyle"]["baselineShift"]["value"], 3)
        self.assertTrue(hints["paragraphStyle"]["boxText"])

    def test_text_conversion_summary_exposes_paragraph_style(self):
        visual = {
            "kind": "text",
            "textDocuments": [
                {
                    "sourceText": "Lower Third",
                    "style": {
                        "fontSize": 72,
                        "paragraphStyle": {
                            "justification": {"code": 0, "label": "left"},
                            "tracking": {"value": 50, "unit": "ae-tracking-units", "candidates": [50]},
                        },
                        "paragraphHints": {
                            "paragraphStyle": {
                                "justification": {"code": 0, "label": "left"},
                                "tracking": {"value": 50, "unit": "ae-tracking-units", "candidates": [50]},
                            }
                        },
                    },
                }
            ],
        }

        summary = ae_native_extract.text_conversion_summary(visual)

        self.assertEqual(summary["paragraphStyle"]["justification"]["label"], "left")
        self.assertEqual(summary["paragraphStyle"]["tracking"]["value"], 50)
        self.assertIn("paragraphHints", summary)

    def test_effect_specific_summary_maps_drop_shadow_and_fill(self):
        shadow = {
            "type": "effect",
            "name": "Drop Shadow",
            "matchName": "ADBE Drop Shadow",
            "children": [
                prop("ADBE Shadow Color", [0, 0, 0, 1], "Color"),
                prop("ADBE Shadow Opacity", [66], "Opacity"),
                prop("ADBE Shadow Direction", [135], "Direction"),
                prop("ADBE Shadow Distance", [18], "Distance"),
                prop("ADBE Shadow Softness", [24], "Softness"),
            ],
        }
        fill = {
            "type": "effect",
            "name": "Fill",
            "matchName": "ADBE Fill",
            "children": [
                prop("ADBE Fill Color", [1, 0, 0, 1], "Color"),
                prop("ADBE Fill Opacity", [80], "Opacity"),
            ],
        }

        shadow_summary = ae_native_extract.effect_summaries(group("ADBE Root", [shadow]))[0]
        fill_summary = ae_native_extract.effect_summaries(group("ADBE Root", [fill]))[0]

        self.assertEqual(shadow_summary["kind"], "dropShadow")
        self.assertIn("distance", shadow_summary["specific"]["dropShadow"])
        self.assertEqual(shadow_summary["specific"]["dropShadow"]["distance"]["value"], 18)
        self.assertEqual(fill_summary["kind"], "fill")
        self.assertEqual(fill_summary["specific"]["fill"]["color"]["value"], [1, 0, 0, 1])

    def test_effect_specific_summary_maps_common_visual_effects(self):
        blur = {
            "type": "effect",
            "name": "Gaussian Blur",
            "matchName": "ADBE Gaussian Blur 2",
            "children": [
                prop("ADBE Gaussian Blur-0001", [18], "Blurriness"),
                prop("ADBE Gaussian Blur-0003", [1], "Repeat Edge Pixels"),
            ],
        }
        glow = {
            "type": "effect",
            "name": "Glow",
            "matchName": "ADBE Glo2",
            "children": [
                prop("ADBE Glow-0002", [60], "Glow Threshold"),
                prop("ADBE Glow-0003", [24], "Glow Radius"),
                prop("ADBE Glow-0004", [1.5], "Glow Intensity"),
            ],
        }
        tint = {
            "type": "effect",
            "name": "Tint",
            "matchName": "ADBE Tint",
            "children": [
                prop("ADBE Tint-0001", [0, 0, 0, 1], "Map Black To"),
                prop("ADBE Tint-0002", [1, 1, 1, 1], "Map White To"),
                prop("ADBE Tint-0003", [80], "Amount to Tint"),
            ],
        }
        wipe = {
            "type": "effect",
            "name": "Linear Wipe",
            "matchName": "ADBE Linear Wipe",
            "children": [
                prop("ADBE Linear Wipe-0001", [45], "Transition Completion"),
                prop("ADBE Linear Wipe-0002", [90], "Wipe Angle"),
                prop("ADBE Linear Wipe-0003", [12], "Feather"),
            ],
        }
        stroke = {
            "type": "effect",
            "name": "Stroke",
            "matchName": "ADBE Stroke",
            "children": [
                prop("ADBE Stroke-0002", [1, 0, 0, 1], "Color"),
                prop("ADBE Stroke-0003", [8], "Brush Size"),
                prop("ADBE Stroke-0007", [100], "End"),
            ],
        }

        summaries = ae_native_extract.effect_summaries(group("ADBE Root", [blur, glow, tint, wipe, stroke]))
        by_kind = {summary["kind"]: summary for summary in summaries}

        self.assertEqual(by_kind["gaussianBlur"]["specific"]["gaussianBlur"]["blurriness"]["value"], 18)
        self.assertEqual(by_kind["glow"]["specific"]["glow"]["radius"]["value"], 24)
        self.assertEqual(by_kind["tint"]["specific"]["tint"]["amount"]["value"], 80)
        self.assertEqual(by_kind["linearWipe"]["specific"]["linearWipe"]["angle"]["value"], 90)
        self.assertEqual(by_kind["strokeEffect"]["specific"]["stroke"]["brushSize"]["value"], 8)

        hints = ae_native_extract.fusion_node_hints("text", summaries)

        self.assertIn("Blur", hints)
        self.assertIn("Glow", hints)
        self.assertIn("TintOrColorCorrector", hints)
        self.assertIn("WipeMask", hints)
        self.assertIn("PaintOrStroke", hints)

    def test_three_d_summary_reports_layer_flags_and_streams(self):
        properties = group(
            "ADBE Root",
            [
                prop("ADBE Position_0", [10], "X Position"),
                prop("ADBE Position_1", [20], "Y Position"),
                prop("ADBE Rotate Z", [45], "Z Rotation"),
                prop("ADBE Casts Shadows", [1], "Casts Shadows"),
                prop("ADBE Accepts Lights", [1], "Accepts Lights"),
            ],
        )
        raw_flags = (1 << 11) | (1 << 19)

        summary = ae_native_extract.three_d_summary(properties, "Layr", raw_flags)

        self.assertTrue(summary["layerIs3DFlagCandidate"])
        self.assertTrue(summary["environmentLayerFlagCandidate"])
        self.assertEqual(summary["objectTypeCandidate"], "avLayer")
        self.assertEqual(summary["transform3DStreams"]["position0"]["value"], [10])
        self.assertEqual(summary["materialOptions"]["castsShadows"]["value"], [1])

    def test_three_d_summary_reports_camera_options(self):
        properties = group(
            "ADBE Root",
            [
                prop("ADBE Camera Zoom", [2666.6667], "Zoom"),
                prop("ADBE Camera Depth of Field", [1], "Depth of Field"),
                prop("ADBE Camera Focus Distance", [500], "Focus Distance"),
                prop("ADBE Camera Aperture", [36], "Aperture"),
                prop("ADBE Camera Blur Level", [100], "Blur Level"),
            ],
        )

        summary = ae_native_extract.three_d_summary(properties, "CLay", None)

        self.assertEqual(summary["objectTypeCandidate"], "camera")
        self.assertEqual(summary["cameraOptions"]["zoom"]["value"], [2666.6667])
        self.assertEqual(summary["cameraOptions"]["depthOfField"]["value"], [1])
        self.assertEqual(summary["cameraOptions"]["focusDistance"]["value"], [500])

    def test_three_d_summary_reports_light_options(self):
        properties = group(
            "ADBE Root",
            [
                prop("ADBE Light Intensity", [75], "Intensity"),
                prop("ADBE Light Color", [1, 0.8, 0.6, 1], "Color"),
                prop("ADBE Light Cone Angle", [90], "Cone Angle"),
                prop("ADBE Light Shadow Darkness", [50], "Shadow Darkness"),
            ],
        )

        summary = ae_native_extract.three_d_summary(properties, "Layr", None)

        self.assertEqual(summary["objectTypeCandidate"], "light")
        self.assertEqual(summary["lightOptions"]["intensity"]["value"], [75])
        self.assertEqual(summary["lightOptions"]["coneAngle"]["value"], [90])
        self.assertEqual(summary["lightOptions"]["color"]["value"], [1, 0.8, 0.6, 1])

    def test_expression_evaluator_resolves_layer_bounds_effects_and_final_array(self):
        layers = [
            {
                "name": "Text",
                "visual": {
                    "kind": "text",
                    "transform": {"position": {"value": [157.25, 220.0]}},
                    "textDocuments": [{"style": {"sourceRect": {"width": 383.84, "height": 88.0}}}],
                },
            },
            {
                "name": "EXPRESSIONS",
                "effects": [
                    {
                        "name": "Text Space",
                        "controlName": "Text Space",
                        "kind": "sliderControl",
                        "parameters": [{"name": "Slider", "value": 44.0}],
                        "value": 44.0,
                    }
                ],
            },
            {"name": "Keyword POS", "visual": {"kind": "layer"}},
        ]
        layers_by_name = {layer["name"]: layer for layer in layers}
        expression = """ref1 = thisComp.layer("Text");
refX1 = ref1.transform.position[0];
refWidth1 = ref1.sourceRectAtTime(time, false).width;
xOffset = thisComp.layer("EXPRESSIONS").effect("Text Space")("Slider").value;
newX = refX1 + refWidth1 + xOffset;
[newX, value[1]]"""

        result = ae_native_extract.evaluate_expression(
            expression,
            [1920.0, 2056.0],
            layers_by_name["Keyword POS"],
            layers_by_name,
        )

        self.assertAlmostEqual(result["assignments"]["newX"], 585.09)
        self.assertAlmostEqual(result["finalValueCandidate"][0], 585.09)
        self.assertAlmostEqual(result["finalValueCandidate"][1], 2056.0)
        self.assertNotIn("unresolvedTerms", result)

    def test_expression_evaluator_resolves_scalar_ternary(self):
        layers = [
            {
                "name": "EXPRESSIONS",
                "effects": [
                    {
                        "name": "Dropdown Menu Control",
                        "controlName": "Dropdown Menu Control",
                        "kind": "dropdownControl",
                        "parameters": [{"name": "Menu", "value": 2.0}],
                        "value": 2.0,
                    },
                    {
                        "name": "BG Alpha",
                        "controlName": "BG Alpha",
                        "kind": "sliderControl",
                        "parameters": [{"name": "Slider", "value": 64.0}],
                        "value": 64.0,
                    },
                ],
            },
            {"name": "Background", "visual": {"kind": "shape"}},
        ]
        layers_by_name = {layer["name"]: layer for layer in layers}
        expression = """menu = thisComp.layer("EXPRESSIONS").effect("Dropdown Menu Control")("Menu").value; // 1 = Off, 2 = On
(menu == 2) ? thisComp.layer("EXPRESSIONS").effect("BG Alpha")("Slider") : 0;"""

        result = ae_native_extract.evaluate_expression(
            expression,
            100.0,
            layers_by_name["Background"],
            layers_by_name,
        )

        self.assertEqual(result["assignments"]["menu"], 2.0)
        self.assertEqual(result["finalValueCandidate"], 64.0)
        self.assertNotIn("unresolvedTerms", result)

    def test_expression_evaluator_resolves_linear_clamp_and_math_helpers(self):
        layers = [
            {
                "name": "CTRL",
                "effects": [
                    {
                        "name": "Progress",
                        "controlName": "Progress",
                        "kind": "sliderControl",
                        "parameters": [{"name": "Slider", "value": 75.0}],
                        "value": 75.0,
                    }
                ],
            },
            {"name": "Target", "visual": {"kind": "shape"}},
        ]
        layers_by_name = {layer["name"]: layer for layer in layers}
        expression = """raw = thisComp.layer("CTRL").effect("Progress")("Slider").value;
mapped = linear(raw, 0, 100, 20, 220);
clamped = clamp(mapped, 50, 200);
[Math.max(clamped, value[0]), Math.floor(value[1] + 0.8)]"""

        result = ae_native_extract.evaluate_expression(
            expression,
            [10.0, 30.0],
            layers_by_name["Target"],
            layers_by_name,
        )

        self.assertIn("linearFunction", result["patterns"])
        self.assertIn("clampFunction", result["patterns"])
        self.assertIn("mathFunction", result["patterns"])
        self.assertEqual(result["assignments"]["raw"], 75.0)
        self.assertEqual(result["assignments"]["mapped"], 170.0)
        self.assertEqual(result["assignments"]["clamped"], 170.0)
        self.assertEqual(result["finalValueCandidate"], [170.0, 30.0])
        self.assertNotIn("unresolvedTerms", result)

    def test_expression_evaluator_resolves_comp_and_current_layer_references(self):
        layer = {
            "name": "Current Text",
            "ordinal": 4,
            "parentComp": {
                "width": 1920,
                "height": 1080,
                "frameRate": 25,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 1.5, "outSeconds": 4.5, "startSeconds": 0.25},
            "visual": {
                "kind": "text",
                "transform": {"position": {"value": [300.0, 420.0, 0]}},
                "textDocuments": [
                    {
                        "style": {
                            "sourceRect": {
                                "left": -10.0,
                                "top": -20.0,
                                "width": 240.0,
                                "height": 80.0,
                            }
                        }
                    }
                ],
            },
        }
        expression = """rectLeft = sourceRectAtTime(time, false).left;
rectWidth = thisLayer.sourceRectAtTime(time, false).width;
center = transform.position[0] + rectLeft + rectWidth / 2;
right = thisComp.width - thisComp.height / 4;
frames = thisComp.duration / thisComp.frameDuration;
[center + thisLayer.index + thisLayer.inPoint, right + frames]"""

        result = ae_native_extract.evaluate_expression(
            expression,
            [0.0, 0.0],
            layer,
            {"Current Text": layer},
        )

        self.assertIn("compPropertyReference", result["patterns"])
        self.assertIn("currentLayerReference", result["patterns"])
        self.assertIn("timeDriven", result["patterns"])
        self.assertEqual(result["assignments"]["rectLeft"], -10.0)
        self.assertEqual(result["assignments"]["rectWidth"], 240.0)
        self.assertEqual(result["assignments"]["center"], 410.0)
        self.assertEqual(result["assignments"]["right"], 1650.0)
        self.assertEqual(result["assignments"]["frames"], 125.0)
        self.assertEqual(result["finalValueCandidate"], [415.5, 1775.0])
        self.assertNotIn("unresolvedTerms", result)

    def test_expression_evaluator_resolves_ease_and_angle_helpers(self):
        layer = {"name": "Target", "visual": {"kind": "shape"}}
        expression = """mid = ease(0.5, 0, 1, 0, 100);
easedIn = easeIn(0.5, 0, 1, 0, 100);
easedOut = easeOut(0.5, 0, 1, 0, 100);
angle = radiansToDegrees(degreesToRadians(90));
[mid + easedOut - easedIn, Math.round(angle)]"""

        result = ae_native_extract.evaluate_expression(
            expression,
            [0.0, 0.0],
            layer,
            {"Target": layer},
        )

        self.assertIn("easeFunction", result["patterns"])
        self.assertIn("angleConversionFunction", result["patterns"])
        self.assertIn("mathFunction", result["patterns"])
        self.assertEqual(result["assignments"]["mid"], 50.0)
        self.assertEqual(result["assignments"]["easedIn"], 25.0)
        self.assertEqual(result["assignments"]["easedOut"], 75.0)
        self.assertEqual(result["assignments"]["angle"], 90.0)
        self.assertEqual(result["finalValueCandidate"], [100.0, 90.0])
        self.assertNotIn("unresolvedTerms", result)


if __name__ == "__main__":
    unittest.main()
