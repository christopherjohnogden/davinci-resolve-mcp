"""Tests for native AE semantic JSON conversion into Fusion settings."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AEP_TO_SETTINGS_PATH = PROJECT_ROOT / "scripts" / "aep_to_settings.py"

spec = importlib.util.spec_from_file_location("aep_to_settings", AEP_TO_SETTINGS_PATH)
assert spec and spec.loader
aep_to_settings = importlib.util.module_from_spec(spec)
sys.modules["aep_to_settings"] = aep_to_settings
spec.loader.exec_module(aep_to_settings)


SEMANTIC_FIXTURE = {
    "format": "semantic-ae-project-v1",
    "items": [],
    "layers": [
        {
            "ordinal": 0,
            "name": "EXPRESSIONS",
            "parentComp": {
                "name": "Main",
                "width": 3840,
                "height": 2160,
                "frameRate": 30,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "visual": {"kind": "layer", "transform": {}},
            "effects": [
                {
                    "name": "Color Control",
                    "controlName": "Text Color",
                    "kind": "colorControl",
                    "parameters": [
                        {
                            "name": "Color",
                            "value": [255, 51, 77, 102],
                            "color": {"rgba": [0.2, 0.3, 0.4, 1.0]},
                        }
                    ],
                },
                {
                    "name": "Slider Control",
                    "controlName": "Shadow Alpha",
                    "kind": "sliderControl",
                    "parameters": [{"name": "Slider", "value": 128}],
                },
                {
                    "name": "Slider Control",
                    "controlName": "Shadow Distance",
                    "kind": "sliderControl",
                    "parameters": [{"name": "Slider", "value": 20}],
                },
            ],
        },
        {
            "ordinal": 1,
            "name": "Title",
            "parentComp": {
                "name": "Main",
                "width": 3840,
                "height": 2160,
                "frameRate": 30,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "visual": {
                "kind": "text",
                "textDocuments": [
                    {
                        "sourceText": "HELLO",
                        "primaryFont": "BrandonGrotesque-Bold",
                        "style": {
                            "fontSize": 72,
                            "fillColor": {"rgba": [0.9, 0.8, 0.1, 1]},
                            "sourceRect": {"width": 300, "height": 100},
                        },
                    }
                ],
                "transform": {
                    "position": {
                        "keyframes": {
                            "keyframes": [
                                {"compFrame": 0, "value": [0.1, 0.2, 0]},
                                {"compFrame": 30, "value": [0.2, 0.2, 0]},
                            ]
                        }
                    }
                },
                "masks": [
                    {
                        "name": "Title Crop",
                        "shape": {"value": [[100, 200], [500, 200], [500, 320], [100, 320]]},
                        "opacity": {"value": 100},
                        "pathDecodeStatus": "bezier-path-decoded",
                    }
                ],
            },
            "effects": [
                {
                    "name": "Fill",
                    "kind": "fill",
                    "specific": {
                        "fill": {
                            "color": {
                                "name": "Color",
                                "value": [255, 255, 255, 255],
                                "color": {"rgba": [1, 1, 1, 1]},
                                "expressions": [
                                    'thisComp.layer("EXPRESSIONS").effect("Text Color")("Color")'
                                ],
                            }
                        }
                    },
                    "parameters": [
                        {
                            "name": "Color",
                            "value": [255, 255, 255, 255],
                            "color": {"rgba": [1, 1, 1, 1]},
                            "expressions": [
                                'thisComp.layer("EXPRESSIONS").effect("Text Color")("Color")'
                            ],
                        }
                    ],
                },
                {
                    "name": "Drop Shadow",
                    "kind": "dropShadow",
                    "specific": {
                        "dropShadow": {
                            "opacity": {
                                "name": "Opacity",
                                "value": 127.5,
                                "expressions": [
                                    'thisComp.layer("EXPRESSIONS").effect("Shadow Alpha")("Slider")'
                                ],
                            },
                            "distance": {
                                "name": "Distance",
                                "value": 22,
                                "expressions": [
                                    'thisComp.layer("EXPRESSIONS").effect("Shadow Distance")("Slider")'
                                ],
                            },
                            "softness": {"name": "Softness", "value": 25},
                        }
                    },
                },
            ],
        },
        {
            "ordinal": 2,
            "name": "Box",
            "parentComp": {
                "name": "Main",
                "width": 3840,
                "height": 2160,
                "frameRate": 30,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "visual": {
                "kind": "shape",
                "transform": {
                    "position": {"value": [0.5, 0.75, 0]},
                    "opacity": {"value": [80]},
                },
                "shapes": [
                    {
                        "name": "Rectangle 1",
                        "type": "rectangle",
                        "size": {"value": [960, 240]},
                        "fill": {
                            "value": [0.1, 0.2, 0.3, 1],
                            "expressions": [
                                'thisComp.layer("EXPRESSIONS").effect("Text Color")("Color")'
                            ],
                        },
                    }
                ],
            },
            "expressionEvaluations": [
                {
                    "matchName": "ADBE Vector Rect Size",
                    "finalValueCandidate": [1234, 250],
                },
                {
                    "matchName": "ADBE Position",
                    "finalValueCandidate": [1440, 900, 0],
                },
                {
                    "matchName": "ADBE Opacity",
                    "finalValueCandidate": 42,
                },
            ],
        },
        {
            "ordinal": 3,
            "name": "Box MATTE",
            "parentComp": {
                "name": "Main",
                "width": 3840,
                "height": 2160,
                "frameRate": 30,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "transfer": {"usedAsTrackMatteCandidateFor": {"layerName": "Box"}},
            "visual": {
                "kind": "shape",
                "transform": {"position": {"value": [0.5, 0.75, 0]}},
                "shapes": [{"name": "Matte", "type": "rectangle", "size": {"value": [960, 240]}}],
            },
        },
    ],
}


def fixture_with_footage(source_path: Path | str) -> dict:
    fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
    fixture["layers"].append(
        {
            "ordinal": 4,
            "name": "BG Image",
            "parentComp": {
                "name": "Main",
                "width": 3840,
                "height": 2160,
                "frameRate": 30,
                "durationSeconds": 5,
            },
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "sourceItem": {
                "itemId": 99,
                "name": "bg-image",
                "kind": "footage-or-solid",
                "candidateFileReferences": [
                    {
                        "path": str(source_path),
                        "sourceType": "image",
                        "extension": ".png",
                        "isStillCandidate": True,
                    }
                ],
            },
            "visual": {
                "kind": "footage",
                "candidateFileReferences": [
                    {
                        "path": str(source_path),
                        "sourceType": "image",
                        "extension": ".png",
                        "isStillCandidate": True,
                    }
                ],
                "transform": {
                    "position": {"value": [0.25, 0.4, 0]},
                    "scale": {"value": [1.5, 1.5, 1]},
                    "opacity": {"value": 75},
                },
            },
        }
    )
    return fixture


class NativeAEPToSettingsTest(unittest.TestCase):
    def test_semantic_position_conversion_detects_pixels_vs_normalized(self):
        self.assertEqual(
            aep_to_settings.semantic_position_to_ae_pixels([0.5, 0.25, 0], 3840, 2160),
            [1920.0, 540.0, 0.0],
        )
        self.assertEqual(
            aep_to_settings.semantic_position_to_ae_pixels([1920, 540, 0], 3840, 2160),
            [1920.0, 540.0, 0.0],
        )

    def test_static_position_value_payload_sets_layer_center(self):
        layer = {
            "transform": {
                "children": [
                    {
                        "matchName": "ADBE Position",
                        "value": {"value": [1920.0, 1620.0, 0.0]},
                    }
                ]
            }
        }

        self.assertEqual(
            aep_to_settings.layer_center_from_position(layer, 3840, 2160),
            (0.5, 0.25),
        )

    def test_position_expression_tracks_reference_layer_offset_and_width(self):
        comp = {
            "name": "Main",
            "width": 1920,
            "height": 1080,
            "duration": 5,
            "frameRate": 30,
            "layers": [
                {
                    "index": 5,
                    "name": "Notes",
                    "type": "text",
                    "transform": {
                        "children": [
                            {
                                "matchName": "ADBE Position",
                                "value": {"value": [1000.0, 400.0, 0.0]},
                            }
                        ]
                    },
                },
                {
                    "index": 6,
                    "name": "Citation",
                    "type": "text",
                    "text": {"sourceRect": {"width": 400.0}},
                    "transform": {
                        "children": [
                            {
                                "matchName": "ADBE Position",
                                "value": {"value": [2000.0, 900.0, 0.0]},
                            }
                        ]
                    },
                    "nativeExpressionEvaluations": [
                        {
                            "matchName": "ADBE Position",
                            "expression": (
                                'ref = thisComp.layer("Notes");\n'
                                'offset = effect("Offset")("Slider");\n'
                                "refPos = ref.transform.position;\n"
                                "newY = refPos[1] + offset;\n"
                                'A = thisComp.layer("Notes");\n'
                                "Acenter = A.toComp(A.anchorPoint)[0];\n"
                                "Bw = thisLayer.sourceRectAtTime(time, false).width;\n"
                                "totalW = Bw + 30;\n"
                                "leftX = Acenter - totalW/2;\n"
                                "[leftX + Bw/2, newY];"
                            ),
                            "assignments": {
                                "offset": 132.0,
                                "Bw": 400.0,
                                "totalW": 430.0,
                            },
                            "resolvedTerms": [
                                {
                                    "kind": "sourceRect",
                                    "layer": "Notes",
                                    "property": "height",
                                    "value": 120.0,
                                }
                            ],
                        }
                    ],
                },
            ],
        }
        builder = aep_to_settings.FusionBuilder(comp, 1920, 1080)

        self.assertEqual(builder.ae_position_at_frame(comp["layers"][1], 0), (985.0, 532.0))

    def test_text_reference_height_expression_gets_fusion_center_compensation(self):
        comp = {
            "name": "Main",
            "width": 1920,
            "height": 1080,
            "duration": 5,
            "frameRate": 30,
            "layers": [
                {
                    "index": 5,
                    "name": "Notes",
                    "type": "text",
                    "text": {"sourceRect": {"height": 120.0}},
                    "transform": {
                        "children": [
                            {
                                "matchName": "ADBE Position",
                                "value": {"value": [1000.0, 400.0, 0.0]},
                            }
                        ]
                    },
                },
                {
                    "index": 6,
                    "name": "Citation",
                    "type": "text",
                    "text": {"sourceRect": {"width": 400.0}},
                    "transform": {
                        "children": [
                            {
                                "matchName": "ADBE Position",
                                "value": {"value": [1000.0, 500.0, 0.0]},
                            }
                        ]
                    },
                    "nativeExpressionEvaluations": [
                        {
                            "matchName": "ADBE Position",
                            "expression": (
                                'ref = thisComp.layer("Notes");\n'
                                'offset = effect("Offset")("Slider");\n'
                                "refPos = ref.transform.position;\n"
                                "refHeight = ref.sourceRectAtTime(time, false).height;\n"
                                "newY = refPos[1] + offset;\n"
                                "[refPos[0], newY];"
                            ),
                            "assignments": {
                                "offset": 132.0,
                                "refHeight": 120.0,
                            },
                        }
                    ],
                },
            ],
        }
        builder = aep_to_settings.FusionBuilder(comp, 1920, 1080)

        self.assertEqual(builder.ae_position_at_frame(comp["layers"][1], 0), (1000.0, 592.0))

    def test_preserve_point_plateaus_keeps_hold_before_reverse_animation(self):
        samples = [
            (0, (0.5, 0.1)),
            (38, (0.5, 0.2)),
            (262, (0.5, 0.2)),
            (300, (0.5, 0.1)),
        ]

        self.assertEqual(aep_to_settings.preserve_point_plateaus(samples), samples)

    def test_instance_inputs_sort_text_tab_notes_before_citation(self):
        builder = aep_to_settings.FusionBuilder(
            {"name": "Main", "width": 1920, "height": 1080, "duration": 5, "layers": []},
            1920,
            1080,
        )
        items = [
            {"id": "BG_Color_Red", "source_op": "AE_Shape_9_BG", "source": "TopLeftRed", "page": "BG"},
            {"id": "Position", "source_op": "AE_Position", "source": "Center", "page": "Position"},
            {"id": "AE_Text_6_Citation_StyledText", "source_op": "AE_Text_6_Citation", "source": "StyledText", "page": "Text"},
            {"id": "TextDropShadowAlpha", "source_op": "AE_Text_6_Citation_DropShadow", "source": "shadowStrength", "page": "Shadow"},
            {"id": "AE_Text_5_Notes_HighlightWriteOn_End", "source_op": "AE_Text_5_Notes_Highlight", "source": "End", "page": "Text"},
            {"id": "AE_Text_5_Notes_Color_Red", "source_op": "AE_Text_5_Notes", "source": "Red1", "page": "Text"},
            {"id": "AE_Text_5_Notes_HighlightColor_Red", "source_op": "AE_Text_5_Notes_Highlight", "source": "Red1", "page": "Text"},
            {"id": "AE_Text_5_Notes_StyledText", "source_op": "AE_Text_5_Notes", "source": "StyledText", "page": "Text"},
            {"id": "AE_Text_5_Notes_HighlightWriteOn_Start", "source_op": "AE_Text_5_Notes_Highlight", "source": "Start", "page": "Text"},
            {"id": "Text_AllCaps", "source_op": "AE_Text_AllCaps", "source": "Value", "page": "Text"},
        ]

        ordered = [item["id"] for item in sorted(items, key=builder.instance_input_sort_key)]

        self.assertEqual(
            ordered,
            [
                "AE_Text_5_Notes_StyledText",
                "Text_AllCaps",
                "AE_Text_5_Notes_Color_Red",
                "AE_Text_5_Notes_HighlightColor_Red",
                "AE_Text_5_Notes_HighlightWriteOn_Start",
                "AE_Text_5_Notes_HighlightWriteOn_End",
                "AE_Text_6_Citation_StyledText",
                "BG_Color_Red",
                "TextDropShadowAlpha",
                "Position",
            ],
        )

    def test_extract_aegraphic_project_extracts_embedded_aep_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            package = tmpdir / "test.aegraphic"
            extract_dir = tmpdir / "assets"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("25-MasterScreenGraphics.aep", b"not a real project")
                archive.writestr("(Footage)/Assets/art.png", b"png")

            project = aep_to_settings.extract_aegraphic_project(package, extract_dir)

            self.assertEqual(project, (extract_dir / "25-MasterScreenGraphics.aep").resolve())
            self.assertTrue(project.exists())
            self.assertTrue((extract_dir / "(Footage)" / "Assets" / "art.png").exists())

    def test_resolve_font_family_style_maps_ae_postscript_names(self):
        self.assertEqual(
            aep_to_settings.resolve_font_family_style("BebasNeueBold"),
            ("Bebas Neue", "Bold"),
        )
        self.assertEqual(
            aep_to_settings.resolve_font_family_style("BrandonGrotesque-Bold"),
            ("Brandon Grotesque", "Bold"),
        )

    def test_resolve_font_family_style_can_fallback_when_installed_font_is_missing(self):
        original = aep_to_settings.font_family_available
        try:
            aep_to_settings.font_family_available = lambda family: {
                "Bebas Neue": False,
                "Brandon Grotesque": True,
            }.get(family, None)

            self.assertEqual(
                aep_to_settings.resolve_font_family_style(
                    "BebasNeueBold",
                    validate_installed=True,
                ),
                ("Brandon Grotesque", "Bold"),
            )
        finally:
            aep_to_settings.font_family_available = original

    def test_semantic_to_legacy_export_maps_text_shape_and_skips_matte(self):
        export = aep_to_settings.semantic_to_legacy_export(
            fixture_with_footage("/tmp/native-bg-image.png"),
            None,
            1920,
            1080,
        )
        comp = export["comps"][0]
        layers = {layer["name"]: layer for layer in comp["layers"]}

        self.assertTrue(export["nativeSemantic"])
        self.assertEqual(comp["name"], "Main")
        self.assertEqual(comp["width"], 3840.0)
        self.assertEqual(layers["Title"]["type"], "text")
        self.assertEqual(layers["Title"]["text"]["sourceText"], "HELLO")
        self.assertEqual(layers["Title"]["text"]["fillColor"], [0.2, 0.3, 0.4, 1.0])
        self.assertEqual(layers["Title"]["effects"]["dropShadow"]["opacity"], 128)
        self.assertEqual(layers["Title"]["effects"]["dropShadow"]["distance"], 20)
        self.assertEqual(layers["Title"]["transform"]["children"][0]["keyframes"][0]["value"], [384.0, 432.0, 0.0])
        self.assertEqual(layers["Title"]["masks"][0]["rect"]["center"], [300.0, 260.0])
        self.assertEqual(layers["Title"]["masks"][0]["rect"]["w"], 400.0)
        self.assertEqual(layers["Title"]["masks"][0]["rect"]["h"], 120.0)
        self.assertEqual(layers["Title"]["masks"][0]["approximation"], "axis-aligned-rectangle")
        self.assertEqual(layers["Box"]["shape"]["rect"]["center"], [1440.0, 900.0])
        self.assertEqual(layers["Box"]["shape"]["rect"]["w"], 1234.0)
        self.assertEqual(layers["Box"]["shape"]["rect"]["h"], 250.0)
        self.assertEqual(layers["Box"]["shape"]["fillColor"], [0.2, 0.3, 0.4, 1.0])
        self.assertEqual(layers["Box"]["shape"]["opacity"], 42)
        self.assertFalse(layers["Box MATTE"]["enabled"])
        self.assertEqual(layers["BG Image"]["type"], "footage")
        self.assertEqual(layers["BG Image"]["footage"]["sourcePath"], "/tmp/native-bg-image.png")
        self.assertEqual(layers["EXPRESSIONS"]["role"], "controlLayer")
        self.assertEqual(layers["EXPRESSIONS"]["controls"][0]["name"], "Text Color")

    def test_semantic_layer_to_legacy_preserves_parent_relationship(self):
        layer = {
            "ordinal": 2,
            "name": "Child Text",
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "visual": {
                "kind": "text",
                "textDocuments": [{"sourceText": "Child"}],
                "transform": {"position": {"value": [960, 540, 0]}},
            },
            "parentIndex": 1,
            "parentLayer": {
                "parentIndex": 1,
                "parentName": "Parent Null",
                "relationshipConfidence": "ae-layer-index-candidate",
            },
        }

        legacy = aep_to_settings.semantic_layer_to_legacy(layer, 1920, 1080, {})

        self.assertEqual(legacy["parentIndex"], 1)
        self.assertEqual(legacy["nativeParentLayer"]["parentName"], "Parent Null")

    def test_semantic_layer_to_legacy_preserves_transfer_modes(self):
        layer = {
            "ordinal": 5,
            "name": "Blend Layer",
            "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
            "visual": {
                "kind": "shape",
                "transform": {"position": {"value": [0.5, 0.5, 0]}},
                "shapes": [
                    {
                        "name": "Rect",
                        "type": "rectangle",
                        "size": {"value": [500, 200]},
                        "fill": {"value": [1, 1, 1, 1]},
                    }
                ],
            },
            "transfer": {
                "blendMode": "screen",
                "blendModeConfidence": "raw-transfer-stream-code-candidate",
                "decodedBlendModeCandidates": [
                    {
                        "kind": "blendMode",
                        "flag": "fbmd",
                        "code": 9,
                        "name": "screen",
                    }
                ],
            },
        }

        legacy = aep_to_settings.semantic_layer_to_legacy(layer, 1920, 1080, {})

        self.assertEqual(legacy["blendMode"], "screen")
        self.assertEqual(legacy["blendModeConfidence"], "raw-transfer-stream-code-candidate")
        self.assertEqual(legacy["nativeTransfer"]["decodedBlendModeCandidates"][0]["name"], "screen")

    def test_semantic_text_layer_uses_source_rect_height_when_font_size_missing(self):
        layer = {
            "name": "Text",
            "visual": {
                "textDocuments": [
                    {
                        "sourceText": "Large text",
                        "style": {
                            "sourceRect": {"width": 800, "height": 130},
                            "fillColor": {"rgba": [1, 1, 1, 1]},
                        },
                    }
                ]
            },
        }

        text = aep_to_settings.semantic_text_layer(layer, 3840, 2160, {})

        self.assertEqual(text["fontSize"], 104)

    def test_semantic_text_layer_uses_resolved_text_animator_fill_color(self):
        layer = {
            "name": "Notes",
            "visual": {
                "textDocuments": [
                    {
                        "sourceText": "The consequence of sin is always later and greater.",
                        "primaryFont": "BrandonGrotesque-Bold",
                        "style": {
                            "fontSize": 72,
                            "fillColor": {"rgba": [0, 0, 0, 1]},
                        },
                    }
                ]
            },
            "expressionEvaluations": [
                {
                    "path": [
                        "ADBE Text Properties",
                        "ADBE Text Animators",
                        "Animator 2",
                        "ADBE Text Animator Properties",
                        "ADBE Text Fill Color",
                    ],
                    "matchName": "ADBE Text Fill Color",
                    "finalValueCandidate": None,
                    "resolvedTerms": [
                        {
                            "kind": "effectParameter",
                            "layer": "Expression",
                            "effect": "Notes Color",
                            "property": "Color",
                            "value": {"rgba": [1, 1, 1, 1]},
                        }
                    ],
                },
                {
                    "path": [
                        "ADBE Text Properties",
                        "ADBE Text Animators",
                        "Animator 1",
                        "ADBE Text Animator Properties",
                        "ADBE Text Fill Color",
                    ],
                    "matchName": "ADBE Text Fill Color",
                    "finalValueCandidate": None,
                    "resolvedTerms": [
                        {
                            "kind": "effectParameter",
                            "layer": "Expression",
                            "effect": "Notes Highlight Color",
                            "property": "Color",
                            "value": {"rgba": [0.076, 0.731, 0.943, 1]},
                        }
                    ],
                },
                {
                    "matchName": "ADBE Text Percent Start",
                    "finalValueCandidate": 6,
                },
                {
                    "matchName": "ADBE Text Percent End",
                    "finalValueCandidate": 30,
                },
            ],
        }

        text = aep_to_settings.semantic_text_layer(layer, 3840, 2160, {})

        self.assertEqual(text["fillColor"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(text["highlightColor"], [0.076, 0.731, 0.943, 1.0])
        self.assertEqual(text["highlightStart"], 6)
        self.assertEqual(text["highlightEnd"], 30)

    def test_semantic_text_style_sidecar_all_caps_updates_rendered_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["visual"]["textDocuments"][0]["sourceText"] = "Hello world"
            title["visual"]["textDocuments"][0]["style"].pop("fontSize", None)
            fixture["auxiliaryLayers"] = [
                {
                    "name": "Markers",
                    "parentComp": {"name": "Main", "width": 3840, "height": 2160},
                    "context": {
                        "postLayerStreams": {
                            "extras": [
                                {
                                    "kind": "CIFO",
                                    "textStyleOverride": {
                                        "compName": "Main",
                                        "layerName": "Title",
                                        "sourceText": "Hello world",
                                        "style": {
                                            "fontEditValue": "BrandonGrotesque-Bold",
                                            "fontFSAllCapsValue": True,
                                            "fontSizeEditValue": 106,
                                        },
                                    },
                                }
                            ]
                        }
                    },
                }
            ]
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            legacy = aep_to_settings.semantic_to_legacy_export(fixture, None, 1920, 1080)
            legacy_title = next(layer for layer in legacy["comps"][0]["layers"] if layer["name"] == "Title")
            self.assertTrue(legacy_title["text"]["allCaps"])
            self.assertEqual(legacy_title["text"]["fontSize"], 106)

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn('Name = "All Caps"', setting)
            self.assertIn('Source = "AllCaps"', setting)
            self.assertIn('INPID_InputControl = "CheckboxControl"', setting)
            self.assertNotIn("AE_Text_AllCaps = PublishNumber", setting)
            self.assertIn('StyledText = Input { Value = "Hello world", },', setting)
            self.assertIn("iif(AllCaps > 0.5, string.upper(tostring(AE_Text_", setting)
            self.assertIn("_Title_TextSource.StyledText.Value)), tostring(AE_Text_", setting)

    def test_convert_native_semantic_json_to_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            asset_path = tmpdir / "native-bg-image.png"
            asset_path.write_bytes(b"not a real png, only path existence matters for Loader wiring")
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture_with_footage(asset_path)), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("GroupOperator", setting)
            self.assertIn("MainOutput1 = InstanceOutput", setting)
            self.assertIn("InstanceInput", setting)
            self.assertIn('Source = "StyledText"', setting)
            self.assertIn('Name = "Title"', setting)
            self.assertIn('Name = "Title Color"', setting)
            self.assertIn('Name = "BG Color"', setting)
            self.assertIn('Name = "BG Box Width"', setting)
            self.assertIn('Name = "Position"', setting)
            self.assertIn('Page = "Text"', setting)
            self.assertIn('Page = "BG"', setting)
            self.assertIn('Page = "Position"', setting)
            self.assertIn("TextPlus", setting)
            self.assertIn("RectangleMask", setting)
            self.assertIn("DropShadow", setting)
            self.assertIn("Loader", setting)
            self.assertIn("Transform", setting)
            self.assertIn("BezierSpline", setting)
            self.assertIn("StyledText = Input { Value = \"HELLO\"", setting)
            self.assertIn("AE_Text_1_Title_LayerMask", setting)
            self.assertIn(f"Filename = \"{asset_path}\"", setting)
            self.assertIn("Red1 = Input { Value = 0.2", setting)
            self.assertIn("Font = Input { Value = \"Brandon Grotesque\"", setting)
            self.assertIn("Style = Input { Value = \"Bold\"", setting)
            self.assertIn("media_out", report)
            self.assertIn("group", report)
            self.assertEqual(report["source_size"], [3840.0, 2160.0])
            self.assertGreaterEqual(len(report["converted_layers"]), 3)
            title_reports = [
                layer for layer in report["converted_layers"] if layer.get("name") == "Title"
            ]
            self.assertEqual(title_reports[0]["mask_count"], 1)
            self.assertEqual(title_reports[0]["layer_mask_approximation"], "axis-aligned-rectangle")
            self.assertEqual(title_reports[0]["layer_mask_tool_type"], "RectangleMask")
            footage_reports = [
                layer for layer in report["converted_layers"] if layer.get("type") == "footage"
            ]
            self.assertEqual(footage_reports[0]["source_path"], str(asset_path))
            self.assertEqual(report["controls"][0]["name"], "EXPRESSIONS")
            self.assertGreaterEqual(len(report["controls"][0]["controls"]), 3)
            unsupported_names = {layer.get("name") for layer in report["unsupported_layers"]}
            self.assertNotIn("EXPRESSIONS", unsupported_names)
            skipped_names = {layer.get("name") for layer in report["skipped_layers"]}
            self.assertIn("EXPRESSIONS", skipped_names)

    def test_convert_native_semantic_json_to_flat_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            asset_path = tmpdir / "native-bg-image.png"
            asset_path.write_bytes(b"not a real png, only path existence matters for Loader wiring")
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture_with_footage(asset_path)), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
                wrap_group=False,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn("GroupOperator", setting)
            self.assertNotIn("MainOutput1 = InstanceOutput", setting)
            self.assertIn("MediaOut1 = MediaOut", setting)
            self.assertIn('SourceOp = "AE_Merge_', setting)
            self.assertIn("ViewInfo = OperatorInfo", setting)
            self.assertTrue(report["flat_output"])
            self.assertIn("media_out", report)
            self.assertNotIn("group", report)

    def test_convert_native_semantic_target_fps_retimes_keyframes_by_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            keyframes = title["visual"]["transform"]["position"]["keyframes"]["keyframes"]
            keyframes[0]["compSeconds"] = 0.0
            keyframes[1]["compSeconds"] = 1.0
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
                target_fps=24,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("[24] = { 0.2", setting)
            self.assertNotIn("[30] = { 0.2", setting)
            self.assertIn("GlobalOut = Input { Value = 119", setting)
            self.assertEqual(report["source_frame_rate"], 30.0)
            self.assertEqual(report["target_frame_rate"], 24.0)

    def test_convert_native_semantic_text_animator_highlight_to_overlay_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["expressionEvaluations"] = [
                {
                    "path": ["ADBE Text Properties", "ADBE Text Animators", "Animator 2", "ADBE Text Fill Color"],
                    "matchName": "ADBE Text Fill Color",
                    "resolvedTerms": [
                        {
                            "kind": "effectParameter",
                            "layer": "EXPRESSIONS",
                            "effect": "Text Color",
                            "property": "Color",
                            "value": {"rgba": [1, 1, 1, 1]},
                        }
                    ],
                },
                {
                    "path": ["ADBE Text Properties", "ADBE Text Animators", "Animator 1", "ADBE Text Fill Color"],
                    "matchName": "ADBE Text Fill Color",
                    "resolvedTerms": [
                        {
                            "kind": "effectParameter",
                            "layer": "EXPRESSIONS",
                            "effect": "Highlight Color",
                            "property": "Color",
                            "value": {"rgba": [0.076, 0.731, 0.943, 1]},
                        }
                    ],
                },
                {"matchName": "ADBE Text Percent Start", "finalValueCandidate": 6},
                {"matchName": "ADBE Text Percent End", "finalValueCandidate": 30},
            ]
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn("AE_Text_1_Title_Highlight = TextPlus", setting)
            self.assertIn("AE_Text_1_Title_HighlightMerge = Merge", setting)
            self.assertIn("Start = Input { Value = 0.06", setting)
            self.assertIn("End = Input { Value = 0.3", setting)
            self.assertIn("Red1 = Input { Value = 0.076", setting)
            self.assertIn("Green1 = Input { Value = 0.731", setting)
            self.assertIn("Blue1 = Input { Value = 0.943", setting)
            self.assertIn('Source = "Start"', setting)
            self.assertIn('Name = "Highlight Start"', setting)
            self.assertIn('Source = "End"', setting)
            self.assertIn('Name = "Highlight End"', setting)
            self.assertNotIn("HighlightStart", setting)
            self.assertNotIn("HighlightEnd", setting)

    def test_convert_can_add_12stone_brand_palette_to_color_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            json_path = tmpdir / "fixture.json"
            plain_setting_path = tmpdir / "plain.setting"
            branded_setting_path = tmpdir / "branded.setting"
            json_path.write_text(json.dumps(SEMANTIC_FIXTURE), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=plain_setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=None,
            )
            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=branded_setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=None,
                brand_palette="12stone",
            )

            plain = plain_setting_path.read_text(encoding="utf-8")
            branded = branded_setting_path.read_text(encoding="utf-8")
            self.assertNotIn('INPID_InputControl = "ComboControl"', plain)
            self.assertIn('INPID_InputControl = "ComboControl"', branded)
            self.assertIn('{ CCS_AddString = "Seafoam" }', branded)
            self.assertIn('{ CCS_AddString = "Dark Charcoal" }', branded)
            self.assertIn('{ CCS_AddString = "Blue" }', branded)
            self.assertIn('Name = "Title Color"', branded)
            self.assertIn('Name = "Custom Title Color"', branded)
            self.assertIn('Expression = "iif(AE_Text_1_Title_Color_Preset < 0.5', branded)
            self.assertIn('AE_Text_1_Title_Color_Custom.TopLeftRed', branded)
            self.assertIn('Expression = "AE_Text_1_Title_Color_Custom.TopLeftAlpha"', branded)
            self.assertIn('Expression = "0.42 * (AE_Shape_2_Box_Color_Custom.TopLeftAlpha)"', branded)
            self.assertIn('Source = "AE_Text_1_Title_Color_Preset"', branded)

    def test_convert_warns_for_unapplied_non_normal_blend_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["transfer"] = {
                "blendMode": "screen",
                "blendModeConfidence": "raw-transfer-stream-code-candidate",
                "decodedBlendModeCandidates": [
                    {
                        "kind": "blendMode",
                        "flag": "fbmd",
                        "code": 9,
                        "name": "screen",
                    }
                ],
            }
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            blend_warnings = [
                item
                for item in report["warnings"]
                if item.get("blend_mode") == "screen"
            ]
            self.assertEqual(len(blend_warnings), 1)
            self.assertEqual(blend_warnings[0]["name"], "Title")

    def test_convert_native_semantic_polygon_mask_to_polyline_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            asset_path = tmpdir / "native-bg-image.png"
            asset_path.write_bytes(b"not a real png, only path existence matters for Loader wiring")
            fixture = fixture_with_footage(asset_path)
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["visual"]["masks"] = [
                {
                    "name": "Triangle Crop",
                    "shape": {"value": [[100, 200], [500, 260], [120, 420]]},
                    "opacity": {"value": 100},
                    "pathDecodeStatus": "bezier-path-decoded",
                }
            ]
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("PolylineMask", setting)
            self.assertIn("Value = Polyline", setting)
            self.assertIn("Closed = true", setting)
            self.assertIn("X = -0.473958", setting)
            self.assertIn("Y = 0.407407", setting)
            title_reports = [
                layer for layer in report["converted_layers"] if layer.get("name") == "Title"
            ]
            self.assertEqual(title_reports[0]["layer_mask_approximation"], "polygon-vertices")
            self.assertEqual(title_reports[0]["layer_mask_tool_type"], "PolylineMask")

    def test_convert_native_semantic_shape_mask_uses_transform_position_and_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            matte = next(layer for layer in fixture["layers"] if layer["name"] == "Box MATTE")
            matte["visual"]["transform"]["position"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0.5, 0.5, 0]},
                        {"compFrame": 30, "value": [0.5, 0.75, 0]},
                    ]
                }
            }
            matte["visual"]["transform"]["scale"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0, 0, 0]},
                        {"compFrame": 30, "value": [0.8, 0.8, 1]},
                    ]
                }
            }
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn("AE_Matte_3_Box_MATTE_MaskCenterX", setting)
            self.assertIn("AE_Matte_3_Box_MATTE_MaskWidth", setting)
            self.assertIn("AE_Matte_3_Box_MATTE_MaskHeight", setting)
            self.assertIn("[30] = { 0.2", setting)
            self.assertIn("[30] = { 0.088889", setting)
            self.assertIn("[30] = { 0.25", setting)

    def test_convert_native_semantic_reveal_scale_mask_animates_size_and_holds_plateau(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            matte = next(layer for layer in fixture["layers"] if layer["name"] == "Box MATTE")
            matte["visual"]["transform"]["position"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0.5, 0.6, 0]},
                        {"compFrame": 30, "value": [0.5, 0.75, 0]},
                        {"compFrame": 90, "value": [0.5, 0.6, 0]},
                    ]
                }
            }
            matte["visual"]["transform"]["scale"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0, 0, 0]},
                        {"compFrame": 30, "value": [0.8, 0.8, 1]},
                        {"compFrame": 90, "value": [0, 0, 0]},
                    ]
                }
            }
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn("AE_Matte_3_Box_MATTE_MaskCenterY", setting)
            self.assertNotIn("AE_Matte_3_Box_MATTE_MaskWidth = BezierSpline", setting)
            self.assertIn("AE_Matte_3_Box_MATTE_MaskHeight = BezierSpline", setting)
            self.assertIn("Width = Input { Value = 0.25", setting)
            self.assertIn("[0] = { 0, Flags = { Linear = true } }", setting)
            self.assertIn("[30] = { 0.088889", setting)
            self.assertIn("[90] = { 0, Flags = { Linear = true } }", setting)

    def test_convert_native_semantic_reveal_mask_uses_evaluated_rect_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            matte = next(layer for layer in fixture["layers"] if layer["name"] == "Box MATTE")
            matte["visual"]["transform"]["scale"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0, 0, 0]},
                        {"compFrame": 30, "value": [0.8, 0.8, 1]},
                        {"compFrame": 90, "value": [0, 0, 0]},
                    ]
                }
            }
            matte["expressionEvaluations"] = [
                {
                    "matchName": "ADBE Vector Rect Size",
                    "assignments": {
                        "newWidth": 1500.0,
                        "newHeight": 300.0,
                    },
                    "resolvedTerms": [
                        {
                            "kind": "sourceRect",
                            "layer": "Title",
                            "field": "width",
                            "value": 1200.0,
                        },
                        {
                            "kind": "effectParameter",
                            "layer": "Box MATTE",
                            "effect": "Width Adjust",
                            "property": "Slider",
                            "value": 300.0,
                        },
                    ],
                }
            ]
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn("Width = Input { Value = 0.390625", setting)
            self.assertIn("[30] = { 0.111111", setting)

    def test_convert_native_semantic_bezier_keyframes_emit_fusion_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            matte = next(layer for layer in fixture["layers"] if layer["name"] == "Box MATTE")
            matte["visual"]["transform"]["scale"] = {
                "keyframes": {
                    "keyframes": [
                        {"compFrame": 0, "value": [0, 0, 0], "interpolationHint": "linear"},
                        {"compFrame": 30, "value": [0.8, 0.8, 1], "interpolationHint": "bezier"},
                        {"compFrame": 90, "value": [0, 0, 0], "interpolationHint": "linear"},
                    ]
                }
            }
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            self.assertIn("[0] = { 0, RH = { 10, 0 } }", setting)
            self.assertIn(
                "[30] = { 0.088889, LH = { 20, 0.088889 }, RH = { 50, 0.088889 } }",
                setting,
            )
            self.assertIn("[90] = { 0, LH = { 70, 0 } }", setting)

    def test_convert_native_semantic_companion_move_layer_animates_text_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fixture = json.loads(json.dumps(SEMANTIC_FIXTURE))
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["visual"]["transform"]["position"] = {"value": [0.2, 0.2, 0]}
            fixture["layers"].append(
                {
                    "ordinal": 10,
                    "name": "Title Move",
                    "parentComp": {
                        "name": "Main",
                        "width": 3840,
                        "height": 2160,
                        "frameRate": 30,
                        "durationSeconds": 5,
                    },
                    "timing": {"inSeconds": 0, "outSeconds": 5, "startSeconds": 0},
                    "conversionSummary": {"role": "precompOrFootage"},
                    "visual": {
                        "kind": "layer",
                        "transform": {
                            "position": {
                                "keyframes": {
                                    "keyframes": [
                                        {"compFrame": 0, "value": [0, 120, 0]},
                                        {"compFrame": 30, "value": [0, 0, 0]},
                                        {"compFrame": 90, "value": [0, 120, 0]},
                                    ]
                                }
                            }
                        },
                    },
                }
            )
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("AE_Text_1_TitleCenterY", setting)
            self.assertIn("[0] = { 0.744444", setting)
            self.assertIn("[30] = { 0.8", setting)
            skipped_names = {layer.get("name") for layer in report["skipped_layers"]}
            self.assertIn("Title Move", skipped_names)

    def test_convert_native_semantic_bezier_mask_tangents_to_polyline_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            asset_path = tmpdir / "native-bg-image.png"
            asset_path.write_bytes(b"not a real png, only path existence matters for Loader wiring")
            fixture = fixture_with_footage(asset_path)
            title = next(layer for layer in fixture["layers"] if layer["name"] == "Title")
            title["visual"]["masks"] = [
                {
                    "name": "Curved Crop",
                    "shape": {
                        "value": {
                            "vertices": [[100, 200], [500, 260], [120, 420]],
                            "inTangents": [[0, 0], [-60, 20], [20, 40]],
                            "outTangents": [[80, -30], [0, 0], [-20, -40]],
                            "closed": True,
                        }
                    },
                    "opacity": {"value": 100},
                    "pathDecodeStatus": "bezier-path-decoded",
                }
            ]
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            setting = setting_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("PolylineMask", setting)
            self.assertIn("LX = -0.015625", setting)
            self.assertIn("LY = -0.009259", setting)
            self.assertIn("RX = 0.020833", setting)
            self.assertIn("RY = 0.013889", setting)
            title_reports = [
                layer for layer in report["converted_layers"] if layer.get("name") == "Title"
            ]
            self.assertEqual(title_reports[0]["layer_mask_approximation"], "bezier-tangents")
            self.assertEqual(title_reports[0]["layer_mask_tool_type"], "PolylineMask")

    def test_convert_repairs_windows_collect_project_footage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            asset_path = tmpdir / "(Footage)" / "Assets" / "native-bg-image.png"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_bytes(b"placeholder")
            fixture = fixture_with_footage(
                r"C:\Users\editor\AppData\Local\Temp\aegraphic\abc\collectprj\(Footage)\Assets\native-bg-image.png"
            )
            fixture["path"] = str(tmpdir / "Project.aep")
            json_path = tmpdir / "fixture.json"
            setting_path = tmpdir / "fixture.setting"
            report_path = tmpdir / "fixture.report.json"
            json_path.write_text(json.dumps(fixture), encoding="utf-8")

            aep_to_settings.convert_json_to_settings(
                json_path=json_path,
                output_path=setting_path,
                comp_name=None,
                target_w=1920,
                target_h=1080,
                report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            footage_reports = [
                layer for layer in report["converted_layers"] if layer.get("type") == "footage"
            ]
            self.assertEqual(footage_reports[0]["source_path"], str(asset_path))


if __name__ == "__main__":
    unittest.main()
