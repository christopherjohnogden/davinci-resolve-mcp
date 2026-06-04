#!/usr/bin/env python3
"""Validate native AE decoder output against a fixture manifest.

The manifest points at local .aep/.aepx files and declares expected semantic
metrics or required layer relationships. It intentionally keeps fixture files
out of the repo so real AE projects can stay in a user's scratch area.

Example:
{
  "fixtures": [
    {
      "name": "core_text_third",
      "path": "/path/to/project.aep",
      "expect": {
        "layerCount": 21,
        "auxiliaryLayerCount": 11,
        "mattePairs": [
          {"matte": "Background MATTE", "target": "Background"}
        ],
        "requiredLayerNames": ["Text", "Background"]
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AE_NATIVE_EXTRACT_PATH = SCRIPT_DIR / "ae_native_extract.py"


def load_ae_native_extract() -> Any:
    spec = importlib.util.spec_from_file_location("ae_native_extract", AE_NATIVE_EXTRACT_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load {AE_NATIVE_EXTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layers_by_name(layers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(layer.get("name")): layer for layer in layers if layer.get("name")}


def collect_metrics(semantic: dict[str, Any]) -> dict[str, Any]:
    layers = semantic.get("layers") or []
    auxiliary_layers = semantic.get("auxiliaryLayers") or []
    all_layers = layers + auxiliary_layers

    matte_pairs = []
    decoded_blend_modes = []
    decoded_track_modes = []
    parent_layer_edges = []
    for layer in layers:
        transfer = layer.get("transfer") or {}
        candidate = transfer.get("trackMatteCandidate")
        if candidate:
            matte_pairs.append(
                {
                    "matte": candidate.get("matteLayerName"),
                    "target": layer.get("name"),
                    "type": candidate.get("matteTypeCandidate"),
                }
            )
        if transfer.get("decodedBlendModeCandidates"):
            decoded_blend_modes.append(
                {
                    "layer": layer.get("name"),
                    "blendMode": transfer.get("blendMode"),
                    "candidates": transfer.get("decodedBlendModeCandidates"),
                }
            )
        if transfer.get("decodedTrackMatteCandidates"):
            decoded_track_modes.append(
                {
                    "layer": layer.get("name"),
                    "trackMatte": transfer.get("trackMatte"),
                    "candidates": transfer.get("decodedTrackMatteCandidates"),
                }
            )
        parent_layer = layer.get("parentLayer") or {}
        if parent_layer.get("parentIndex") is not None:
            parent_layer_edges.append(
                {
                    "child": layer.get("name"),
                    "parent": parent_layer.get("parentName"),
                    "parentIndex": parent_layer.get("parentIndex"),
                    "status": parent_layer.get("resolutionStatus"),
                }
            )

    expression_final_candidates = []
    for layer in layers:
        for evaluation in layer.get("expressionEvaluations") or []:
            if evaluation.get("finalValueCandidate") is not None:
                expression_final_candidates.append(
                    {
                        "layer": layer.get("name"),
                        "path": evaluation.get("path"),
                        "value": evaluation.get("finalValueCandidate"),
                    }
                )

    return {
        "layerCount": len(layers),
        "auxiliaryLayerCount": len(auxiliary_layers),
        "expressionLayerCount": sum(1 for layer in layers if layer.get("expressionEvaluations")),
        "finalExpressionCandidateCount": len(expression_final_candidates),
        "matteRelationshipCount": len(matte_pairs),
        "decodedBlendModeLayerCount": len(decoded_blend_modes),
        "decodedTrackMatteModeLayerCount": len(decoded_track_modes),
        "parentLayerRelationshipCount": len(parent_layer_edges),
        "effectSpecificCount": sum(1 for layer in layers for effect in (layer.get("effects") or []) if effect.get("specific")),
        "switchFlagLayerCount": sum(1 for layer in all_layers if (layer.get("switches") or {}).get("aegpLayerFlagNameCandidates")),
        "maskLayerCount": sum(1 for layer in layers if (layer.get("visual") or {}).get("masks")),
        "customPathLayerCount": sum(
            1
            for layer in layers
            if any(shape.get("path") for shape in ((layer.get("visual") or {}).get("shapes") or []))
        ),
        "threeDLayerCount": sum(1 for layer in all_layers if layer.get("threeD")),
        "layerNames": [layer.get("name") for layer in layers],
        "auxiliaryLayerNames": [layer.get("name") for layer in auxiliary_layers],
        "mattePairs": matte_pairs,
        "decodedBlendModes": decoded_blend_modes,
        "decodedTrackMatteModes": decoded_track_modes,
        "parentLayerEdges": parent_layer_edges,
        "expressionFinalCandidates": expression_final_candidates,
    }


def compare_number(name: str, actual: int | float, expectation: Any, failures: list[str]) -> None:
    if isinstance(expectation, dict):
        if "min" in expectation and actual < expectation["min"]:
            failures.append(f"{name}: expected >= {expectation['min']}, got {actual}")
        if "max" in expectation and actual > expectation["max"]:
            failures.append(f"{name}: expected <= {expectation['max']}, got {actual}")
        if "equals" in expectation and actual != expectation["equals"]:
            failures.append(f"{name}: expected {expectation['equals']}, got {actual}")
        return
    if actual != expectation:
        failures.append(f"{name}: expected {expectation}, got {actual}")


def compare_expectations(metrics: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    numeric_keys = {
        "layerCount",
        "auxiliaryLayerCount",
        "expressionLayerCount",
        "finalExpressionCandidateCount",
        "matteRelationshipCount",
        "decodedBlendModeLayerCount",
        "decodedTrackMatteModeLayerCount",
        "parentLayerRelationshipCount",
        "effectSpecificCount",
        "switchFlagLayerCount",
        "maskLayerCount",
        "customPathLayerCount",
        "threeDLayerCount",
    }
    for key in numeric_keys:
        if key in expect:
            compare_number(key, metrics.get(key, 0), expect[key], failures)

    layer_names = set(metrics.get("layerNames") or [])
    for name in expect.get("requiredLayerNames") or []:
        if name not in layer_names:
            failures.append(f"requiredLayerNames: missing {name!r}")

    auxiliary_names = set(metrics.get("auxiliaryLayerNames") or [])
    for name in expect.get("requiredAuxiliaryLayerNames") or []:
        if name not in auxiliary_names:
            failures.append(f"requiredAuxiliaryLayerNames: missing {name!r}")

    actual_pairs = {
        (pair.get("matte"), pair.get("target"))
        for pair in metrics.get("mattePairs") or []
    }
    for pair in expect.get("mattePairs") or []:
        expected_pair = (pair.get("matte"), pair.get("target"))
        if expected_pair not in actual_pairs:
            failures.append(f"mattePairs: missing matte={expected_pair[0]!r} target={expected_pair[1]!r}")

    return failures


def validate_fixture(module: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    path = Path(fixture["path"]).expanduser()
    parsed = module.parse_project(path, include_bdata=True)
    semantic = module.semantic_project(parsed)
    metrics = collect_metrics(semantic)
    failures = compare_expectations(metrics, fixture.get("expect") or {})
    return {
        "name": fixture.get("name") or path.name,
        "path": str(path),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "metrics": metrics,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate native AE decoder fixture manifests.")
    parser.add_argument("manifest", help="JSON manifest with fixture paths and expectations.")
    parser.add_argument("-o", "--output", help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module = load_ae_native_extract()
    reports = [validate_fixture(module, fixture) for fixture in manifest.get("fixtures") or []]
    output = {"status": "pass" if all(item["status"] == "pass" for item in reports) else "fail", "fixtures": reports}
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).expanduser().write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
