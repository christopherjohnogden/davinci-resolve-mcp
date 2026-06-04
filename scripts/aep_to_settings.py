#!/usr/bin/env python3
"""Convert After Effects projects or AE-export JSON into Fusion .setting files.

This converter supports two extraction paths:

1. Native .aep/.aepx parsing through scripts/ae_native_extract.py. This does
   not launch After Effects and is the default for project files.
2. Legacy After Effects JSX export through /Users/cogden/Desktop/ae_to_json3.jsx,
   still available with --use-after-effects.

The Fusion builder intentionally starts with the reliably mappable subset:
text, simple shape rectangles, solids, opacity, and basic position animation.
Unmapped data is preserved in the JSON and summarized in a sidecar report so
the mapper can be expanded without changing the AE extraction path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_EXPORTER = Path("/Users/cogden/Desktop/ae_to_json3.jsx")
DEFAULT_AE = Path(
    "/Applications/Adobe After Effects 2026/"
    "Adobe After Effects 2026.app/Contents/MacOS/After Effects"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NATIVE_EXTRACTOR = SCRIPT_DIR / "ae_native_extract.py"


_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


BRAND_PALETTES: dict[str, list[dict[str, Any]]] = {
    "12stone": [
        {"name": "Seafoam", "hex": "#93D3CF", "rgb": (0.576471, 0.827451, 0.811765)},
        {"name": "Silver Gray", "hex": "#979B9E", "rgb": (0.592157, 0.607843, 0.619608)},
        {"name": "Slate Gray", "hex": "#65666A", "rgb": (0.396078, 0.4, 0.415686)},
        {"name": "Deep Teal", "hex": "#113639", "rgb": (0.066667, 0.211765, 0.223529)},
        {"name": "Teal", "hex": "#257981", "rgb": (0.145098, 0.47451, 0.505882)},
        {"name": "Charcoal", "hex": "#333436", "rgb": (0.2, 0.203922, 0.211765)},
        {"name": "Dark Charcoal", "hex": "#282828", "rgb": (0.156863, 0.156863, 0.156863)},
        {"name": "Burgundy", "hex": "#761318", "rgb": (0.462745, 0.07451, 0.094118)},
        {"name": "Rust", "hex": "#A8402A", "rgb": (0.658824, 0.25098, 0.164706)},
        {"name": "Sage", "hex": "#BFD17D", "rgb": (0.74902, 0.819608, 0.490196)},
        {"name": "Light Gray", "hex": "#CECED0", "rgb": (0.807843, 0.807843, 0.815686)},
        {"name": "Orange", "hex": "#DF5930", "rgb": (0.87451, 0.34902, 0.188235)},
        {"name": "Pale Green", "hex": "#E2EDC0", "rgb": (0.886275, 0.929412, 0.752941)},
        {"name": "Soft White", "hex": "#EBEBEB", "rgb": (0.921569, 0.921569, 0.921569)},
        {"name": "Blue", "hex": "#0F72E3", "rgb": (0.058824, 0.447059, 0.890196)},
    ],
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_num(value: float, digits: int = 6) -> float:
    factor = 10**digits
    return round(float(value) * factor) / factor


def clean_tool_name(value: str, fallback: str = "Layer") -> str:
    name = _NAME_RE.sub("_", value.strip())
    name = name.strip("_") or fallback
    if name[0].isdigit():
        name = f"AE_{name}"
    return name[:80]


def lua_string(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return f'"{text}"'


def lua_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def lua_number(value: Any, default: float = 0.0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isclose(number, round(number), abs_tol=1e-9):
        return str(int(round(number)))
    return f"{round_num(number):.6f}".rstrip("0").rstrip(".")


def lua_point(values: list[float]) -> str:
    return "{ " + ", ".join(lua_number(v) for v in values) + " }"


def input_value(value: Any) -> str:
    if isinstance(value, bool):
        rendered = lua_bool(value)
    elif isinstance(value, (int, float)):
        rendered = lua_number(value)
    elif isinstance(value, list):
        rendered = lua_point(value)
    else:
        rendered = lua_string(value)
    return f"Input {{ Value = {rendered}, }}"


def input_value_expression(value: Any, expression: str) -> str:
    if isinstance(value, bool):
        rendered = lua_bool(value)
    elif isinstance(value, (int, float)):
        rendered = lua_number(value)
    elif isinstance(value, list):
        rendered = lua_point(value)
    else:
        rendered = lua_string(value)
    return f"Input {{ Value = {rendered}, Expression = {lua_string(expression)}, }}"


def input_link(source_op: str, source: str = "Output") -> str:
    return (
        "Input {\n"
        f"\t\t\t\t\tSourceOp = {lua_string(source_op)},\n"
        f"\t\t\t\t\tSource = {lua_string(source)},\n"
        "\t\t\t\t}"
    )


def input_link_inline(source_op: str, source: str = "Output") -> str:
    return f"Input {{ SourceOp = {lua_string(source_op)}, Source = {lua_string(source)}, }}"


def color3(value: Any, default: tuple[float, float, float] = (1, 1, 1)) -> tuple[float, float, float]:
    if isinstance(value, dict):
        rgba = value.get("rgba") or value.get("rgb")
        if isinstance(rgba, list):
            value = rgba
    if isinstance(value, list) and len(value) >= 3:
        return (
            clamp(float(value[0]), 0, 1),
            clamp(float(value[1]), 0, 1),
            clamp(float(value[2]), 0, 1),
        )
    return default


def alpha_value(value: Any, default: float = 1.0) -> float:
    if isinstance(value, dict):
        rgba = value.get("rgba")
        if isinstance(rgba, list) and len(rgba) >= 4:
            return clamp(float(rgba[3]), 0, 1)
    if isinstance(value, list) and len(value) >= 4:
        return clamp(float(value[3]), 0, 1)
    return default


def scalar_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fusion_format_id(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".png": "PNGFormat",
        ".jpg": "JpegFormat",
        ".jpeg": "JpegFormat",
        ".tif": "TiffFormat",
        ".tiff": "TiffFormat",
        ".exr": "OpenEXRFormat",
        ".dpx": "DPXFormat",
        ".mov": "QuickTimeMovies",
        ".mp4": "QuickTimeMovies",
        ".m4v": "QuickTimeMovies",
        ".avi": "AVIFormat",
    }.get(ext, "")


def first_value(prop: dict[str, Any] | None, default: Any = None) -> Any:
    if not prop:
        return default
    if "value" in prop:
        value = prop["value"]
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value
    keyframes = prop.get("keyframes") or prop.get("bakedSamples") or []
    if keyframes:
        return keyframes[0].get("value", default)
    return default


def key_samples(prop: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not prop:
        return []
    samples = prop.get("bakedSamples") or prop.get("keyframes") or []
    return [s for s in samples if isinstance(s, dict) and "frame" in s and "value" in s]


def keyframe_interpolation_hints(prop: dict[str, Any] | None) -> dict[int, str]:
    hints: dict[int, str] = {}
    for sample in key_samples(prop):
        hint = sample.get("interpolationHint")
        if hint:
            hints[int(sample["frame"])] = str(hint)
    return hints


def merge_interpolation_hints(*hint_maps: dict[int, str]) -> dict[int, str]:
    merged: dict[int, str] = {}
    for hint_map in hint_maps:
        for frame, hint in hint_map.items():
            if hint == "bezier" or frame not in merged:
                merged[frame] = hint
    return merged


def simplify_scalar_samples(samples: list[tuple[int, float]], tolerance: float = 1e-5) -> list[tuple[int, float]]:
    if not samples:
        return []
    simplified = [samples[0]]
    for frame, value in samples[1:]:
        if abs(float(value) - float(simplified[-1][1])) > tolerance:
            simplified.append((frame, value))
    if len(simplified) > 1 and simplified[-1][0] != samples[-1][0]:
        simplified.append(samples[-1])
    return simplified


def preserve_scalar_plateaus(samples: list[tuple[int, float]], tolerance: float = 1e-5) -> list[tuple[int, float]]:
    if not samples:
        return []
    preserved = [samples[0]]
    for index, (frame, value) in enumerate(samples[1:], start=1):
        previous_value = samples[index - 1][1]
        next_value = samples[index + 1][1] if index + 1 < len(samples) else None
        value_changed = abs(float(value) - float(preserved[-1][1])) > tolerance
        plateau_end = (
            next_value is not None
            and abs(float(value) - float(previous_value)) <= tolerance
            and abs(float(next_value) - float(value)) > tolerance
        )
        if value_changed or plateau_end:
            preserved.append((frame, value))
    if preserved[-1][0] != samples[-1][0]:
        preserved.append(samples[-1])
    return preserved


def simplify_point_samples(
    samples: list[tuple[int, tuple[float, float]]],
    tolerance: float = 1e-05,
) -> list[tuple[int, tuple[float, float]]]:
    if not samples:
        return []
    simplified = [samples[0]]
    for frame, point in samples[1:]:
        prev = simplified[-1][1]
        if abs(point[0] - prev[0]) > tolerance or abs(point[1] - prev[1]) > tolerance:
            simplified.append((frame, point))
    if len(simplified) > 1 and simplified[-1][0] != samples[-1][0]:
        simplified.append(samples[-1])
    return simplified


def preserve_point_plateaus(
    samples: list[tuple[int, tuple[float, float]]],
    tolerance: float = 1e-05,
) -> list[tuple[int, tuple[float, float]]]:
    if not samples:
        return []
    preserved = [samples[0]]
    for index, (frame, point) in enumerate(samples[1:], start=1):
        previous_point = samples[index - 1][1]
        next_point = samples[index + 1][1] if index + 1 < len(samples) else None
        last_point = preserved[-1][1]
        value_changed = (
            abs(point[0] - last_point[0]) > tolerance
            or abs(point[1] - last_point[1]) > tolerance
        )
        plateau_end = False
        if next_point is not None:
            same_as_previous = (
                abs(point[0] - previous_point[0]) <= tolerance
                and abs(point[1] - previous_point[1]) <= tolerance
            )
            different_from_next = (
                abs(next_point[0] - point[0]) > tolerance
                or abs(next_point[1] - point[1]) > tolerance
            )
            plateau_end = same_as_previous and different_from_next
        if value_changed or plateau_end:
            preserved.append((frame, point))
    if preserved[-1][0] != samples[-1][0]:
        preserved.append(samples[-1])
    return preserved


def find_prop(nodes: Any, match_name: str) -> dict[str, Any] | None:
    if isinstance(nodes, dict):
        if nodes.get("matchName") == match_name:
            return nodes
        children = nodes.get("children") or nodes.get("props") or nodes.get("properties")
        found = find_prop(children, match_name)
        if found:
            return found
    elif isinstance(nodes, list):
        for node in nodes:
            found = find_prop(node, match_name)
            if found:
                return found
    return None


def load_native_extractor(path: Path = DEFAULT_NATIVE_EXTRACTOR) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Native AE extractor not found: {path}")
    spec = importlib.util.spec_from_file_location("ae_native_extract", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load native AE extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_native_semantic_json(data: dict[str, Any]) -> bool:
    return data.get("format") == "semantic-ae-project-v1" or (
        isinstance(data.get("layers"), list) and isinstance(data.get("items"), list) and "comps" not in data
    )


def semantic_comp_groups(data: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for layer in data.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        parent = layer.get("parentComp") or {}
        name = parent.get("name") or "AE Comp"
        group_data = groups.setdefault(name, {"name": name, "parentComp": parent, "layers": []})
        group_data["layers"].append(layer)
        if parent and not group_data.get("parentComp"):
            group_data["parentComp"] = parent
    return list(groups.values())


def semantic_text_style_overrides(data: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = [item for item in data.get("textStyleOverrides") or [] if isinstance(item, dict)]
    if overrides:
        return overrides

    seen = set()
    for layer in (data.get("layers") or []) + (data.get("auxiliaryLayers") or []):
        extras = (((layer.get("context") or {}).get("postLayerStreams") or {}).get("extras") or [])
        for extra in extras:
            override = extra.get("textStyleOverride")
            if not isinstance(override, dict):
                continue
            key = (
                override.get("compName"),
                override.get("layerName"),
                json.dumps(override.get("style") or {}, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            overrides.append(override)
    return overrides


def semantic_text_style_override_map(data: dict[str, Any], group_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comp_name = group_data.get("name")
    out: dict[str, dict[str, Any]] = {}
    for override in semantic_text_style_overrides(data):
        layer_name = override.get("layerName")
        if not layer_name:
            continue
        override_comp = override.get("compName")
        if override_comp and comp_name and override_comp != comp_name:
            continue
        out.setdefault(str(layer_name), override)
    return out


def select_semantic_comp(data: dict[str, Any], comp_name: str | None) -> dict[str, Any]:
    groups = semantic_comp_groups(data)
    if not groups:
        raise ValueError("No semantic AE layers found.")
    if comp_name:
        for group_data in groups:
            if group_data.get("name") == comp_name:
                return group_data
        names = ", ".join(str(group_data.get("name")) for group_data in groups)
        raise ValueError(f"Comp {comp_name!r} not found. Available comps: {names}")
    return max(groups, key=lambda item: len(item.get("layers") or []))


def semantic_comp_dimensions(group_data: dict[str, Any], target_w: int, target_h: int) -> tuple[float, float]:
    parent = group_data.get("parentComp") or {}
    width = float(parent.get("width") or target_w)
    height = float(parent.get("height") or target_h)
    return width, height


def semantic_value(summary: dict[str, Any] | None) -> Any:
    if not isinstance(summary, dict):
        return None
    return summary.get("value")


EFFECT_EXPR_RE = re.compile(
    r'(?:thisComp\.layer\("(?P<layer>[^"]+)"\)\.)?effect\("(?P<effect>[^"]+)"\)\("(?P<property>[^"]+)"\)(?:\.value)?'
)


def normalized_parameter_value(parameter: dict[str, Any]) -> Any:
    color = parameter.get("color")
    if isinstance(color, dict) and isinstance(color.get("rgba"), list):
        return color["rgba"]
    if "value" in parameter:
        return parameter["value"]
    return None


def normalized_color_value(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        rgba = value.get("rgba")
        if isinstance(rgba, list) and len(rgba) >= 3:
            return [float(channel) for channel in rgba[:4]]
        return None
    if isinstance(value, list) and len(value) >= 3:
        try:
            channels = [float(channel) for channel in value[:4]]
        except (TypeError, ValueError):
            return None
        if any(abs(channel) > 1 for channel in channels[:3]):
            channels[:3] = [channel / 255 for channel in channels[:3]]
        if len(channels) >= 4 and abs(channels[3]) > 1:
            channels[3] = channels[3] / 255
        return channels
    return None


def semantic_control_map(layers: list[dict[str, Any]]) -> dict[tuple[str, str, str], Any]:
    controls: dict[tuple[str, str, str], Any] = {}
    for layer in layers:
        layer_name = layer.get("name")
        if not layer_name:
            continue
        for effect in layer.get("effects") or []:
            effect_names = [effect.get("controlName"), effect.get("name")]
            effect_names = [str(name) for name in effect_names if name]
            parameters = effect.get("parameters") or []
            if not parameters and ("value" in effect or "color" in effect):
                parameters = [{"name": "Value", "value": effect.get("value"), "color": effect.get("color")}]
            for parameter in parameters:
                parameter_name = parameter.get("name") or parameter.get("matchName") or "Value"
                value = normalized_parameter_value(parameter)
                if value is None:
                    continue
                for effect_name in effect_names:
                    controls[(str(layer_name), effect_name, str(parameter_name))] = value
                    if parameter_name in {"Slider", "Color", "Menu", "Value"}:
                        controls[(str(layer_name), effect_name, "Value")] = value
    return controls


def semantic_control_entries(layer: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for effect in layer.get("effects") or []:
        kind = effect.get("kind")
        if kind not in {"sliderControl", "colorControl", "dropdownControl"}:
            continue
        parameters = effect.get("parameters") or []
        primary = None
        for parameter in parameters:
            value = normalized_parameter_value(parameter)
            if value is not None:
                primary = {
                    "name": parameter.get("name") or parameter.get("matchName") or "Value",
                    "value": value,
                }
                if parameter.get("min") is not None:
                    primary["min"] = parameter.get("min")
                if parameter.get("max") is not None:
                    primary["max"] = parameter.get("max")
                break
        if primary is None and ("value" in effect or "color" in effect):
            primary = {"name": "Value", "value": normalized_parameter_value(effect)}
        entry = {
            "name": effect.get("controlName") or effect.get("name"),
            "effectName": effect.get("name"),
            "kind": kind,
        }
        if primary:
            entry["parameter"] = primary
            entry["value"] = primary.get("value")
        if effect.get("options"):
            entry["options"] = effect.get("options")
        entries.append(entry)
    return entries


def semantic_layer_role(layer: dict[str, Any], visual_kind: str | None = None) -> str | None:
    summary = layer.get("conversionSummary") or {}
    if summary.get("role"):
        return summary.get("role")
    if semantic_control_entries(layer):
        return "controlLayer"
    return visual_kind


def resolve_effect_expression_value(value_owner: dict[str, Any], controls: dict[tuple[str, str, str], Any]) -> Any:
    for expression in value_owner.get("expressions") or []:
        for match in EFFECT_EXPR_RE.finditer(expression):
            layer_name = match.group("layer")
            effect_name = match.group("effect")
            property_name = match.group("property")
            if not layer_name:
                continue
            for key in (
                (layer_name, effect_name, property_name),
                (layer_name, effect_name, "Value"),
            ):
                if key in controls:
                    return controls[key]
    return None


def semantic_effect_parameter_value(parameter: dict[str, Any], controls: dict[tuple[str, str, str], Any]) -> Any:
    resolved = resolve_effect_expression_value(parameter, controls)
    if resolved is not None:
        return resolved
    return normalized_parameter_value(parameter)


def first_effect_by_kind(layer: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for effect in layer.get("effects") or []:
        if effect.get("kind") == kind:
            return effect
    return None


def semantic_fill_override(layer: dict[str, Any], controls: dict[tuple[str, str, str], Any]) -> Any:
    effect = first_effect_by_kind(layer, "fill")
    if not effect:
        return None
    fill = ((effect.get("specific") or {}).get("fill") or {})
    parameter = fill.get("color")
    if isinstance(parameter, dict):
        return semantic_effect_parameter_value(parameter, controls)
    for parameter in effect.get("parameters") or []:
        if parameter.get("name") == "Color":
            return semantic_effect_parameter_value(parameter, controls)
    if isinstance(effect.get("color"), dict):
        return effect["color"].get("rgba")
    return effect.get("value")


def semantic_evaluation_resolved_value(evaluation: dict[str, Any]) -> Any:
    if evaluation.get("finalValueCandidate") is not None:
        return evaluation.get("finalValueCandidate")
    for term in evaluation.get("resolvedTerms") or []:
        if not isinstance(term, dict) or "value" not in term:
            continue
        return term.get("value")
    return None


def semantic_text_animator_colors(layer: dict[str, Any]) -> dict[str, Any]:
    colors: dict[str, Any] = {}
    for evaluation in layer.get("expressionEvaluations") or []:
        if evaluation.get("matchName") != "ADBE Text Fill Color":
            continue
        color = normalized_color_value(semantic_evaluation_resolved_value(evaluation))
        if not color:
            continue
        resolved_terms = evaluation.get("resolvedTerms") or []
        resolved_names = " ".join(
            str(term.get("effect") or "") for term in resolved_terms if isinstance(term, dict)
        ).lower()
        path = " ".join(str(part) for part in evaluation.get("path") or []).lower()
        if "highlight" in resolved_names or "highlight" in path:
            colors["highlightColor"] = color
        elif "baseColor" not in colors:
            colors["baseColor"] = color
    for match_name, key in (
        ("ADBE Text Percent Start", "highlightStart"),
        ("ADBE Text Percent End", "highlightEnd"),
    ):
        value = semantic_eval_final_value(layer, match_name)
        if value is not None:
            colors[key] = value
    return colors


def semantic_drop_shadow(layer: dict[str, Any], controls: dict[tuple[str, str, str], Any]) -> dict[str, Any] | None:
    effect = first_effect_by_kind(layer, "dropShadow")
    if not effect:
        return None
    shadow = ((effect.get("specific") or {}).get("dropShadow") or {})
    out: dict[str, Any] = {}
    for key in ("color", "opacity", "direction", "distance", "softness"):
        parameter = shadow.get(key)
        if isinstance(parameter, dict):
            out[key] = semantic_effect_parameter_value(parameter, controls)
    if "color" not in out:
        out["color"] = [0, 0, 0, 1]
    return out or None


def semantic_keyframes(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    keyframes = summary.get("keyframes")
    if isinstance(keyframes, dict):
        return [item for item in keyframes.get("keyframes") or [] if isinstance(item, dict)]
    if isinstance(keyframes, list):
        return [item for item in keyframes if isinstance(item, dict)]
    return []


def semantic_position_to_ae_pixels(value: Any, comp_w: float, comp_h: float) -> Any:
    if isinstance(value, list) and len(value) >= 2:
        x = float(value[0])
        y = float(value[1])
        if -2 <= x <= 2 and -2 <= y <= 2:
            out = [x * comp_w, y * comp_h]
        else:
            out = [x, y]
        if len(value) >= 3:
            out.append(float(value[2]))
        return out
    return value


def semantic_mask_shape_value(mask: dict[str, Any]) -> Any:
    shape = mask.get("shape")
    value = semantic_value(shape) if isinstance(shape, dict) else None
    if value is not None:
        return value
    keyframes = semantic_keyframes(shape)
    if keyframes:
        return keyframes[0].get("value")
    return None


def semantic_mask_points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        value = value.get("vertices") or value.get("points") or value.get("value")
    points = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points


def semantic_mask_tangents(value: Any, *keys: str) -> list[tuple[float, float]]:
    if not isinstance(value, dict):
        return []
    raw = None
    for key in keys:
        if key in value:
            raw = value.get(key)
            break
    if raw is None:
        return []
    return semantic_mask_points(raw)


def semantic_mask_has_tangents(
    in_tangents: list[tuple[float, float]],
    out_tangents: list[tuple[float, float]],
    tolerance: float = 1e-7,
) -> bool:
    for tangent in [*in_tangents, *out_tangents]:
        if abs(tangent[0]) > tolerance or abs(tangent[1]) > tolerance:
            return True
    return False


def semantic_mask_closed(value: Any) -> bool:
    if isinstance(value, dict) and "closed" in value:
        return semantic_bool(value.get("closed"))
    return True


def semantic_mask_is_axis_aligned_rect(points: list[tuple[float, float]]) -> bool:
    if len(points) < 4:
        return False
    xs = {round_num(point[0], 4) for point in points}
    ys = {round_num(point[1], 4) for point in points}
    if len(xs) != 2 or len(ys) != 2:
        return False
    return all(round_num(point[0], 4) in xs and round_num(point[1], 4) in ys for point in points)


def semantic_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalized_mask_point(point: tuple[float, float], comp_w: float, comp_h: float) -> dict[str, float]:
    return {
        "x": round_num(point[0] / comp_w),
        "y": round_num(1 - (point[1] / comp_h)),
    }


def normalized_mask_tangent(tangent: tuple[float, float] | None, comp_w: float, comp_h: float) -> dict[str, float]:
    if tangent is None:
        tangent = (0.0, 0.0)
    return {
        "x": round_num(tangent[0] / comp_w),
        "y": round_num(-(tangent[1] / comp_h)),
    }


def semantic_mask_polygon(
    points: list[tuple[float, float]],
    shape_value: Any,
    comp_w: float,
    comp_h: float,
) -> tuple[list[dict[str, Any]], bool]:
    in_tangents = semantic_mask_tangents(shape_value, "inTangents", "in_tangents", "inTangent")
    out_tangents = semantic_mask_tangents(shape_value, "outTangents", "out_tangents", "outTangent")
    has_tangents = semantic_mask_has_tangents(in_tangents, out_tangents)
    polygon = []
    for point_index, point in enumerate(points):
        normalized = normalized_mask_point(point, comp_w, comp_h)
        in_tangent = normalized_mask_tangent(
            in_tangents[point_index] if point_index < len(in_tangents) else None,
            comp_w,
            comp_h,
        )
        out_tangent = normalized_mask_tangent(
            out_tangents[point_index] if point_index < len(out_tangents) else None,
            comp_w,
            comp_h,
        )
        polygon.append(
            {
                "x": normalized["x"],
                "y": normalized["y"],
                "lx": in_tangent["x"],
                "ly": in_tangent["y"],
                "rx": out_tangent["x"],
                "ry": out_tangent["y"],
                "hasTangents": has_tangents,
            }
        )
    return polygon, has_tangents


def semantic_layer_masks(
    masks: Any,
    comp_w: float,
    comp_h: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted = []
    unsupported = []
    if not isinstance(masks, list):
        return converted, unsupported
    for index, mask in enumerate(masks):
        if not isinstance(mask, dict):
            continue
        shape_value = semantic_mask_shape_value(mask)
        points = semantic_mask_points(shape_value)
        if len(points) < 2:
            unsupported.append(
                {
                    "index": index,
                    "name": mask.get("name"),
                    "reason": "Mask path was not decoded to point vertices.",
                    "pathDecodeStatus": mask.get("pathDecodeStatus"),
                }
            )
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        expansion = semantic_scalar_or_list(semantic_value(mask.get("expansion")))
        try:
            expansion_px = float(expansion or 0)
        except (TypeError, ValueError):
            expansion_px = 0
        min_x = min(xs) - expansion_px
        max_x = max(xs) + expansion_px
        min_y = min(ys) - expansion_px
        max_y = max(ys) + expansion_px
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            unsupported.append(
                {
                    "index": index,
                    "name": mask.get("name"),
                    "reason": "Mask path bounds are empty.",
                }
            )
            continue
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        polygon_points, has_tangents = semantic_mask_polygon(points, shape_value, comp_w, comp_h)
        is_axis_rect = semantic_mask_is_axis_aligned_rect(points) and not has_tangents
        approximation = (
            "axis-aligned-rectangle"
            if is_axis_rect
            else "bezier-tangents"
            if has_tangents
            else "polygon-vertices"
        )
        converted.append(
            {
                "index": index,
                "name": mask.get("name") or f"Mask {index + 1}",
                "mode": semantic_scalar_or_list(semantic_value(mask.get("mode"))),
                "inverted": semantic_bool(semantic_scalar_or_list(semantic_value(mask.get("inverted")))),
                "opacity": semantic_scalar_or_list(semantic_value(mask.get("opacity"))),
                "feather": semantic_scalar_or_list(semantic_value(mask.get("feather"))),
                "expansion": expansion_px,
                "pathDecodeStatus": mask.get("pathDecodeStatus"),
                "approximation": approximation,
                "rect": {
                    "center": [center_x, center_y],
                    "w": width,
                    "h": height,
                    "normalizedCenter": [center_x / comp_w, 1 - (center_y / comp_h)],
                    "normalizedWidth": width / comp_w,
                    "normalizedHeight": height / comp_h,
                },
                "polygon": {
                    "closed": semantic_mask_closed(shape_value),
                    "points": polygon_points,
                    "vertexCount": len(polygon_points),
                    "hasTangents": has_tangents,
                },
            }
        )
    return converted, unsupported


def semantic_scalar_or_list(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def semantic_eval_final_value(layer: dict[str, Any] | None, match_name: str) -> Any:
    if not isinstance(layer, dict):
        return None
    for evaluation in layer.get("expressionEvaluations") or []:
        if evaluation.get("matchName") == match_name and "finalValueCandidate" in evaluation:
            return evaluation.get("finalValueCandidate")
    return None


def legacy_transform_prop(
    match_name: str,
    summary: dict[str, Any] | None,
    comp_w: float,
    comp_h: float,
    *,
    position: bool = False,
    override_value: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        summary = {}
    value = override_value if override_value is not None else semantic_value(summary)
    if position:
        value = semantic_position_to_ae_pixels(value, comp_w, comp_h)
    else:
        value = semantic_scalar_or_list(value)
    out: dict[str, Any] = {"matchName": match_name}
    if value is not None:
        out["value"] = {"value": value}
    samples = []
    for keyframe in semantic_keyframes(summary):
        sample_value = keyframe.get("value")
        if position:
            sample_value = semantic_position_to_ae_pixels(sample_value, comp_w, comp_h)
        else:
            sample_value = semantic_scalar_or_list(sample_value)
        frame = keyframe.get("compFrame", keyframe.get("localFrame"))
        if frame is None or sample_value is None:
            continue
        sample: dict[str, Any] = {"frame": int(round(float(frame))), "value": sample_value}
        seconds = keyframe.get("compSeconds", keyframe.get("localSeconds"))
        if seconds is not None:
            try:
                sample["seconds"] = float(seconds)
            except (TypeError, ValueError):
                pass
        for metadata_key in ("interpolationHint", "easingCandidate"):
            if keyframe.get(metadata_key) is not None:
                sample[metadata_key] = keyframe[metadata_key]
        samples.append(sample)
    if samples:
        out["keyframes"] = samples
    return out if out.get("value") is not None or out.get("keyframes") else None


def legacy_transform_from_semantic(
    visual: dict[str, Any],
    comp_w: float,
    comp_h: float,
    layer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transform = visual.get("transform") or {}
    children = []
    for key, match_name, is_position in [
        ("anchorPoint", "ADBE Anchor Point", True),
        ("position", "ADBE Position", True),
        ("scale", "ADBE Scale", False),
        ("opacity", "ADBE Opacity", False),
        ("rotateZ", "ADBE Rotate Z", False),
    ]:
        override = semantic_eval_final_value(layer, match_name)
        child = legacy_transform_prop(
            match_name,
            transform.get(key),
            comp_w,
            comp_h,
            position=is_position,
            override_value=override,
        )
        if child:
            children.append(child)
    return {"children": children}


def semantic_text_layer(
    layer: dict[str, Any],
    comp_w: float,
    comp_h: float,
    controls: dict[tuple[str, str, str], Any],
    text_style_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    visual = layer.get("visual") or {}
    docs = visual.get("textDocuments") or []
    if not docs:
        return None
    doc = docs[0]
    style = doc.get("style") or {}
    animator_colors = semantic_text_animator_colors(layer)
    fill = animator_colors.get("baseColor") or semantic_fill_override(layer, controls) or style.get("fillColor")
    source_rect = style.get("sourceRect") or {}
    font_size = style.get("fontSize")
    override_style = (text_style_override or {}).get("style") or {}
    if font_size is None and override_style.get("fontSizeEditValue") is not None:
        font_size = override_style.get("fontSizeEditValue")
    if font_size is None and isinstance(source_rect, dict):
        height = scalar_number(source_rect.get("height"), 0)
        if height > 0:
            font_size = height * 0.8
    return {
        "sourceText": doc.get("sourceText") or doc.get("text") or layer.get("name") or "",
        "font": override_style.get("fontEditValue")
        or doc.get("primaryFont")
        or (doc.get("fontCandidates") or [None])[0]
        or "Arial",
        "fontSize": font_size or 48,
        "fillColor": (fill.get("rgba") if isinstance(fill, dict) else fill),
        "highlightColor": animator_colors.get("highlightColor"),
        "highlightStart": animator_colors.get("highlightStart"),
        "highlightEnd": animator_colors.get("highlightEnd"),
        "tracking": (style.get("paragraphHints") or {}).get("trackingCandidate"),
        "sourceRect": source_rect,
        "allCaps": bool(override_style.get("fontFSAllCapsValue")),
        "smallCaps": bool(override_style.get("fontFSSmallCapsValue")),
    }


def first_semantic_shape(layer: dict[str, Any]) -> dict[str, Any] | None:
    visual = layer.get("visual") or {}
    for shape in visual.get("shapes") or []:
        if shape.get("type") == "rectangle":
            return shape
    shapes = visual.get("shapes") or []
    return shapes[0] if shapes else None


def semantic_shape_layer(
    layer: dict[str, Any],
    comp_w: float,
    comp_h: float,
    controls: dict[tuple[str, str, str], Any],
) -> dict[str, Any] | None:
    visual = layer.get("visual") or {}
    shape = first_semantic_shape(layer)
    if not shape:
        return None
    transform = visual.get("transform") or {}
    position = semantic_eval_final_value(layer, "ADBE Position")
    if position is None:
        position = semantic_value(transform.get("position")) or [0.5, 0.5]
    center = semantic_position_to_ae_pixels(position, comp_w, comp_h)
    size = semantic_eval_final_value(layer, "ADBE Vector Rect Size")
    if size is None:
        size = semantic_value(shape.get("size")) or [comp_w, comp_h]
    rect_position = semantic_value(shape.get("rectPosition"))
    if isinstance(rect_position, list) and isinstance(center, list) and len(rect_position) >= 2:
        center = [center[0] + float(rect_position[0]), center[1] + float(rect_position[1])]
    fill_summary = shape.get("fill") or {}
    fill = resolve_effect_expression_value(fill_summary, controls)
    if fill is None:
        fill = fill_summary.get("color") or fill_summary.get("value")
    fill = semantic_fill_override(layer, controls) or fill
    opacity = semantic_eval_final_value(layer, "ADBE Opacity")
    if opacity is None:
        opacity = semantic_scalar_or_list(semantic_value(transform.get("opacity"))) or 100
    return {
        "rect": {
            "center": center[:2] if isinstance(center, list) else [comp_w / 2, comp_h / 2],
            "w": float(size[0]) if isinstance(size, list) and len(size) >= 1 else comp_w,
            "h": float(size[1]) if isinstance(size, list) and len(size) >= 2 else comp_h,
            "roundness": semantic_scalar_or_list(semantic_value(shape.get("roundness"))) or 0,
        },
        "fillColor": (fill.get("rgba") if isinstance(fill, dict) else fill),
        "opacity": semantic_scalar_or_list(opacity),
        "shapeType": shape.get("type"),
    }


def source_candidates_from_semantic(layer: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    visual = layer.get("visual") or {}
    source_item = layer.get("sourceItem") or {}
    for owner in (visual, source_item):
        for candidate in owner.get("candidateFileReferences") or []:
            if isinstance(candidate, dict) and candidate.get("path"):
                candidates.append(candidate)
    seen = set()
    out = []
    for candidate in candidates:
        path = str(candidate.get("path"))
        if path in seen:
            continue
        seen.add(path)
        out.append(candidate)
    return out


def semantic_footage_layer(layer: dict[str, Any]) -> dict[str, Any] | None:
    candidates = source_candidates_from_semantic(layer)
    source_item = layer.get("sourceItem") or {}
    visual = layer.get("visual") or {}
    if not candidates and not source_item:
        return None
    out: dict[str, Any] = {
        "sourceItem": source_item,
        "sourceMetadata": source_item.get("sourceMetadata") or visual.get("sourceMetadata"),
        "sourcePathCandidates": candidates,
    }
    if candidates:
        out["sourcePath"] = candidates[0].get("path")
    return out


def semantic_layer_enabled(layer: dict[str, Any]) -> bool:
    transfer = layer.get("transfer") or {}
    if transfer.get("usedAsTrackMatteCandidateFor"):
        return False
    return True


def semantic_layer_to_legacy(
    layer: dict[str, Any],
    comp_w: float,
    comp_h: float,
    controls: dict[tuple[str, str, str], Any],
    text_style_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    visual = layer.get("visual") or {}
    visual_kind = visual.get("kind")
    role = semantic_layer_role(layer, visual_kind)
    timing = layer.get("timing") or {}
    legacy: dict[str, Any] = {
        "index": layer.get("ordinal") or layer.get("treeIndex") or 0,
        "name": layer.get("name"),
        "type": visual_kind if visual_kind in {"text", "shape", "footage"} else "unsupported",
        "role": role,
        "enabled": semantic_layer_enabled(layer),
        "inPoint": float(timing.get("inSeconds") or 0),
        "outPoint": float(timing.get("outSeconds") or 0),
        "startTime": float(timing.get("startSeconds") or 0),
        "transform": legacy_transform_from_semantic(visual, comp_w, comp_h, layer),
        "nativeTransfer": layer.get("transfer"),
        "nativeExpressionEvaluations": layer.get("expressionEvaluations") or [],
        "nativeConversionSummary": layer.get("conversionSummary"),
    }
    if layer.get("parentIndex") is not None:
        legacy["parentIndex"] = layer["parentIndex"]
    if layer.get("parentLayer"):
        legacy["nativeParentLayer"] = layer["parentLayer"]
    transfer = layer.get("transfer") or {}
    if transfer.get("blendMode"):
        legacy["blendMode"] = transfer["blendMode"]
    if transfer.get("blendModeConfidence"):
        legacy["blendModeConfidence"] = transfer["blendModeConfidence"]
    if transfer.get("trackMatte"):
        legacy["trackMatte"] = transfer["trackMatte"]
    if transfer.get("trackMatteConfidence"):
        legacy["trackMatteConfidence"] = transfer["trackMatteConfidence"]
    effects: dict[str, Any] = {}
    drop_shadow = semantic_drop_shadow(layer, controls)
    if drop_shadow:
        effects["dropShadow"] = drop_shadow
    fill_override = semantic_fill_override(layer, controls)
    if fill_override is not None:
        effects["fillColor"] = fill_override
    if effects:
        legacy["effects"] = effects
    controls_out = semantic_control_entries(layer)
    if controls_out:
        legacy["controls"] = controls_out
    masks, unsupported_masks = semantic_layer_masks(visual.get("masks"), comp_w, comp_h)
    if masks:
        legacy["masks"] = masks
    if unsupported_masks:
        legacy["unsupportedMasks"] = unsupported_masks
    if visual_kind == "text":
        text_override = (text_style_overrides or {}).get(str(layer.get("name")))
        text = semantic_text_layer(layer, comp_w, comp_h, controls, text_override)
        if text:
            legacy["text"] = text
    elif visual_kind == "shape":
        shape = semantic_shape_layer(layer, comp_w, comp_h, controls)
        if shape:
            legacy["shape"] = shape
    elif visual_kind == "footage":
        footage = semantic_footage_layer(layer)
        if footage:
            legacy["footage"] = footage
    return legacy


def semantic_to_legacy_export(data: dict[str, Any], comp_name: str | None, target_w: int, target_h: int) -> dict[str, Any]:
    group_data = select_semantic_comp(data, comp_name)
    parent = group_data.get("parentComp") or {}
    comp_w, comp_h = semantic_comp_dimensions(group_data, target_w, target_h)
    fps = float(parent.get("frameRate") or 30)
    controls = semantic_control_map(group_data.get("layers") or [])
    text_style_overrides = semantic_text_style_override_map(data, group_data)
    duration_seconds = parent.get("durationSeconds")
    if duration_seconds is None:
        max_out = max((float((layer.get("timing") or {}).get("outSeconds") or 0) for layer in group_data.get("layers") or []), default=5)
        duration_seconds = max(max_out, 5)
    comp = {
        "name": group_data.get("name") or "AE Comp",
        "width": comp_w,
        "height": comp_h,
        "frameRate": fps,
        "duration": float(duration_seconds),
        "layers": [
            semantic_layer_to_legacy(layer, comp_w, comp_h, controls, text_style_overrides)
            for layer in group_data.get("layers") or []
        ],
        "nativeSemantic": True,
        "projectPath": data.get("path"),
    }
    return {"comps": [comp], "nativeSemantic": True}


def native_project_to_semantic(input_path: Path, extractor_path: Path = DEFAULT_NATIVE_EXTRACTOR) -> dict[str, Any]:
    extractor = load_native_extractor(extractor_path)
    parsed = extractor.parse_project(input_path, include_bdata=True)
    return extractor.semantic_project(parsed)


def extract_aegraphic_project(input_path: Path, extract_dir: Path) -> Path:
    extract_root = extract_dir.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path) as archive:
        project_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".aep", ".aepx")) and not name.endswith("/")
        ]
        if not project_members:
            raise ValueError(f"No embedded .aep/.aepx project found in {input_path}")
        for member in archive.infolist():
            target = (extract_root / member.filename).resolve()
            try:
                target.relative_to(extract_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in .aegraphic archive: {member.filename}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as dest:
                dest.write(source.read())
    return (extract_root / project_members[0]).resolve()


def layer_center_from_bounds(layer: dict[str, Any], comp_w: float, comp_h: float) -> tuple[float, float] | None:
    for bounds in layer.get("bounds") or []:
        if bounds.get("sample") not in ("midPoint", "inPoint"):
            continue
        corners = bounds.get("compSpaceCorners") or {}
        tl = corners.get("topLeft")
        br = corners.get("bottomRight")
        if isinstance(tl, list) and isinstance(br, list) and len(tl) >= 2 and len(br) >= 2:
            x = (float(tl[0]) + float(br[0])) / 2
            y = (float(tl[1]) + float(br[1])) / 2
            return x / comp_w, 1 - (y / comp_h)
    return None


def layer_center_from_position(layer: dict[str, Any], comp_w: float, comp_h: float) -> tuple[float, float]:
    pos = first_value(find_prop(layer.get("transform"), "ADBE Position"), [comp_w / 2, comp_h / 2])
    if isinstance(pos, list) and len(pos) >= 2:
        return float(pos[0]) / comp_w, 1 - (float(pos[1]) / comp_h)
    return 0.5, 0.5


def layer_center(layer: dict[str, Any], comp_w: float, comp_h: float) -> tuple[float, float]:
    return layer_center_from_bounds(layer, comp_w, comp_h) or layer_center_from_position(layer, comp_w, comp_h)


def fusion_position_from_ae(value: Any, comp_w: float, comp_h: float) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) >= 2:
        return float(value[0]) / comp_w, 1 - (float(value[1]) / comp_h)
    return None


def font_size_to_fusion(font_size: Any, comp_h: float) -> float:
    try:
        return round_num(float(font_size) / comp_h)
    except (TypeError, ValueError):
        return 0.06


def text_range_to_fusion(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1:
        number /= 100
    return clamp(number, 0, 1)


FONT_NAME_OVERRIDES = {
    "BebasNeueBold": ("Bebas Neue", "Bold"),
    "BebasNeue-Regular": ("Bebas Neue", "Regular"),
    "BebasNeue": ("Bebas Neue", "Regular"),
    "BrandonGrotesque-Bold": ("Brandon Grotesque", "Bold"),
    "BrandonGrotesqueBold": ("Brandon Grotesque", "Bold"),
    "BrandonGrotesque-Regular": ("Brandon Grotesque", "Regular"),
    "BrandonGrotesque": ("Brandon Grotesque", "Regular"),
}


@lru_cache(maxsize=128)
def font_family_available(family: str) -> bool | None:
    if not family:
        return None
    try:
        proc = subprocess.run(
            ["fc-match", family],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    output = (proc.stdout or "").lower()
    words = [word for word in re.split(r"\s+", family.lower()) if word]
    if not words:
        return None
    return all(word in output for word in words)


def fallback_installed_font(family: str, style: str) -> tuple[str, str]:
    available = font_family_available(family)
    if available is True or available is None:
        return family, style
    for candidate in ("Brandon Grotesque", "Helvetica", "Arial"):
        if font_family_available(candidate) is not False:
            fallback_style = "Bold" if style.lower().startswith("bold") else "Regular"
            return candidate, fallback_style
    return "Arial", "Regular"


def resolve_font_family_style(font_name: Any, *, validate_installed: bool = False) -> tuple[str, str]:
    raw = str(font_name or "").strip()
    if not raw or raw == "AdobeInvisFont":
        return "Arial", "Regular"
    if raw in FONT_NAME_OVERRIDES:
        family, style = FONT_NAME_OVERRIDES[raw]
        return fallback_installed_font(family, style) if validate_installed else (family, style)

    style = "Regular"
    family = raw
    if "-" in raw:
        family, style_candidate = raw.rsplit("-", 1)
        style = style_candidate or style
    else:
        for suffix, style_name in (
            ("BoldItalic", "Bold Italic"),
            ("BoldOblique", "Bold Italic"),
            ("Italic", "Italic"),
            ("Oblique", "Italic"),
            ("Bold", "Bold"),
            ("Regular", "Regular"),
        ):
            if raw.endswith(suffix) and len(raw) > len(suffix):
                family = raw[: -len(suffix)]
                style = style_name
                break

    family = FONT_NAME_OVERRIDES.get(family, (family, style))[0]
    family = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", family).strip() or "Arial"
    if validate_installed:
        family, style = fallback_installed_font(family, style)
    return family, style


def comp_duration_frames(comp: dict[str, Any], fps: float | None = None) -> int:
    duration = float(comp.get("duration") or 5)
    resolved_fps = float(fps or comp.get("frameRate") or 30)
    return max(0, int(round(duration * resolved_fps)) - 1)


def layer_frame(layer: dict[str, Any], key: str, fps: float) -> int:
    try:
        return max(0, int(round(float(layer.get(key, 0)) * fps)))
    except (TypeError, ValueError):
        return 0


class FusionBuilder:
    def __init__(
        self,
        comp: dict[str, Any],
        target_w: int,
        target_h: int,
        *,
        wrap_group: bool = True,
        target_fps: float | None = None,
        brand_palette: str | None = None,
    ):
        self.comp = comp
        self.source_w = float(comp.get("width") or target_w)
        self.source_h = float(comp.get("height") or target_h)
        self.target_w = target_w
        self.target_h = target_h
        self.wrap_group = wrap_group
        self.brand_palette_name = brand_palette.lower() if brand_palette else None
        self.brand_palette = BRAND_PALETTES.get(self.brand_palette_name or "")
        if self.brand_palette_name and not self.brand_palette:
            names = ", ".join(sorted(BRAND_PALETTES))
            raise ValueError(f"Unknown brand palette {brand_palette!r}. Available palettes: {names}")
        self.source_fps = float(comp.get("frameRate") or 30)
        self.target_fps = float(target_fps or self.source_fps)
        self.end_frame = comp_duration_frames(comp, self.target_fps)
        self.project_path = Path(comp["projectPath"]).expanduser() if comp.get("projectPath") else None
        self.project_dir = self.project_path.parent if self.project_path else None
        self.layers_by_index = {
            int(layer.get("index")): layer
            for layer in comp.get("layers", [])
            if isinstance(layer, dict) and layer.get("index") is not None
        }
        self.layers_by_name = {
            str(layer.get("name")): layer
            for layer in comp.get("layers", [])
            if isinstance(layer, dict) and layer.get("name") is not None
        }
        self.companion_move_layers: dict[str, dict[str, Any]] = {}
        for layer in comp.get("layers", []):
            if not isinstance(layer, dict):
                continue
            name = str(layer.get("name") or "").strip()
            if not name.endswith(" Move"):
                continue
            self.companion_move_layers[name[:-5].strip()] = layer
        self.tools: list[tuple[str, str]] = []
        self.user_controls_by_tool: dict[str, list[str]] = {}
        self.merge_blend_expression_by_foreground: dict[str, str] = {}
        self.matte_masks_by_target: dict[str, str] = {}
        self.mask_dimensions: dict[str, tuple[float, float]] = {}
        self.shape_text_y_offsets: dict[int, float] = {}
        self.all_caps_control_tool: str | None = None
        self.published_inputs: list[dict[str, Any]] = []
        self.published_input_ids: set[str] = set()
        self.published_input_keys: set[tuple[str, str, str]] = set()
        self.next_control_group = 1
        self.primary_shadow_tool: str | None = None
        self.reported_control_layers: set[tuple[Any, Any]] = set()
        self.reported_layer_mask_layers: set[tuple[Any, Any]] = set()
        self.reported_transfer_layers: set[tuple[Any, Any]] = set()
        self.visible_tool_count = 0
        self.report: dict[str, Any] = {
            "comp": comp.get("name"),
            "source_size": [self.source_w, self.source_h],
            "target_size": [target_w, target_h],
            "source_frame_rate": self.source_fps,
            "target_frame_rate": self.target_fps,
            "converted_layers": [],
            "skipped_layers": [],
            "unsupported_layers": [],
            "controls": [],
            "warnings": [],
        }

    def add_default_view_info(self, body: str) -> str:
        stripped = body.lstrip()
        if "ViewInfo" in body or stripped.startswith(("BezierSpline", "XYPath")):
            return body
        index = self.visible_tool_count
        self.visible_tool_count += 1
        x = 80 + (index % 5) * 210
        y = -160 + (index // 5) * 115
        return body.replace(
            "{\n",
            "{\n\t\t\tViewInfo = OperatorInfo { Pos = { "
            f"{lua_number(x)}, {lua_number(y)}, "
            "}, },\n",
            1,
        )

    def sample_frame(self, sample: dict[str, Any]) -> int:
        seconds = sample.get("seconds")
        if seconds is not None:
            try:
                return max(0, int(round(float(seconds) * self.target_fps)))
            except (TypeError, ValueError):
                pass
        try:
            source_frame = float(sample.get("frame") or 0)
        except (TypeError, ValueError):
            source_frame = 0.0
        if self.source_fps > 0 and abs(self.target_fps - self.source_fps) > 1e-6:
            source_frame = source_frame * self.target_fps / self.source_fps
        return max(0, int(round(source_frame)))

    def layer_timeline_frame(self, layer: dict[str, Any], key: str) -> int:
        return layer_frame(layer, key, self.target_fps)

    def layer_global_out(self, layer: dict[str, Any]) -> int:
        return max(self.layer_timeline_frame(layer, "outPoint") - 1, 0)

    def add_tool(self, name: str, body: str) -> str:
        body = self.add_default_view_info(body)
        self.tools.append((name, body))
        return name

    def update_tool_body(self, name: str, body: str) -> None:
        for index, (tool_name, _tool_body) in enumerate(self.tools):
            if tool_name == name:
                self.tools[index] = (tool_name, body)
                return
        raise KeyError(f"Tool {name!r} has not been added.")

    def set_tool_input_expression(self, tool: str, source: str, value: Any, expression: str) -> None:
        rendered_value = input_value_expression(value, expression)
        pattern = re.compile(rf"^(\t+){re.escape(source)} = Input \{{ Value = .*?, \}},$", re.MULTILINE)
        for tool_name, body in self.tools:
            if tool_name != tool:
                continue
            replacement = rf"\1{source} = {rendered_value},"
            updated, count = pattern.subn(replacement, body, count=1)
            if count:
                self.update_tool_body(tool, updated)
            else:
                self.report["warnings"].append(
                    {
                        "tool": tool,
                        "input": source,
                        "warning": "Could not attach brand palette expression; input line was not found.",
                    }
                )
            return
        self.report["warnings"].append(
            {
                "tool": tool,
                "input": source,
                "warning": "Could not attach brand palette expression; tool was not found.",
            }
        )

    def set_tool_input_value(self, tool: str, source: str, value: Any) -> None:
        rendered_value = input_value(value)
        pattern = re.compile(rf"^(\t+){re.escape(source)} = Input \{{ Value = .*?(?:, Expression = .*?)?, \}},$", re.MULTILINE)
        for tool_name, body in self.tools:
            if tool_name != tool:
                continue
            replacement = rf"\1{source} = {rendered_value},"
            updated, count = pattern.subn(replacement, body, count=1)
            if count:
                self.update_tool_body(tool, updated)
            else:
                self.report["warnings"].append(
                    {
                        "tool": tool,
                        "input": source,
                        "warning": "Could not set generated input value; input line was not found.",
                    }
                )
            return
        self.report["warnings"].append(
            {
                "tool": tool,
                "input": source,
                "warning": "Could not set generated input value; tool was not found.",
            }
        )

    def add_tool_user_controls(self, tool: str, controls: list[str]) -> None:
        if not controls:
            return
        self.user_controls_by_tool.setdefault(tool, []).extend(controls)

    def render_tool_user_controls(self, tool: str) -> str:
        controls = self.user_controls_by_tool.get(tool) or []
        if not controls:
            return ""
        lines = ["\t\t\tUserControls = ordered() {"]
        lines.extend(controls)
        lines.append("\t\t\t},")
        return "\n".join(lines)

    def body_with_user_controls(self, tool: str, body: str) -> str:
        controls = self.render_tool_user_controls(tool)
        if not controls:
            return body
        if "UserControls = ordered()" in body:
            self.report["warnings"].append(
                {
                    "tool": tool,
                    "warning": "Skipped generated brand palette controls because the tool already has UserControls.",
                }
            )
            return body
        return body.replace("{\n", "{\n" + controls + "\n", 1)

    def palette_rgb_expression(self, preset_control: str, custom_tool: str, custom_source: str, channel: int) -> str:
        assert self.brand_palette
        expr = lua_number(self.brand_palette[-1]["rgb"][channel])
        for index in range(len(self.brand_palette) - 1, 0, -1):
            value = lua_number(self.brand_palette[index - 1]["rgb"][channel])
            expr = f"iif({preset_control} < {lua_number(index + 0.5)}, {value}, {expr})"
        return f"iif({preset_control} < 0.5, {custom_tool}.{custom_source}, {expr})"

    def palette_alpha_expression(self, preset_control: str, custom_tool: str, custom_source: str, alpha: float) -> str:
        return f"{custom_tool}.{custom_source}"

    def add_custom_color_source(
        self,
        name: str,
        color: tuple[float, float, float],
        alpha: float,
    ) -> str:
        tool = self.unique_name(name)
        inputs = [
            ("Type", 'Input { Value = FuID { "Solid" }, }'),
            ("TopLeftRed", input_value(color[0])),
            ("TopLeftGreen", input_value(color[1])),
            ("TopLeftBlue", input_value(color[2])),
            ("TopLeftAlpha", input_value(alpha)),
            ("Width", input_value(256)),
            ("Height", input_value(256)),
        ]
        body = "Background {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def render_palette_user_controls(
        self,
        header_id: str,
        preset_id: str,
        label: str,
        page: str,
    ) -> list[str]:
        assert self.brand_palette
        lines = [
            f"\t\t\t\t{header_id} = {{",
            f"\t\t\t\t\tLINKS_Name = {lua_string(label)},",
            '\t\t\t\t\tLINKID_DataType = "Number",',
            '\t\t\t\t\tINPID_InputControl = "LabelControl",',
            "\t\t\t\t\tLBLC_DropDownButton = true,",
            "\t\t\t\t\tLBLC_NumInputs = 4,",
            "\t\t\t\t\tINP_Default = 1,",
            f"\t\t\t\t\tICS_ControlPage = {lua_string(page)},",
            "\t\t\t\t},",
            f"\t\t\t\t{preset_id} = {{",
            '\t\t\t\t\t{ CCS_AddString = "Custom" },',
        ]
        for item in self.brand_palette:
            lines.append(f"\t\t\t\t\t{{ CCS_AddString = {lua_string(item['name'])} }},")
        lines.extend(
            [
                "\t\t\t\t\tINP_Integer = true,",
                '\t\t\t\t\tLINKID_DataType = "Number",',
                '\t\t\t\t\tINPID_InputControl = "ComboControl",',
                "\t\t\t\t\tINP_Default = 0,",
                f"\t\t\t\t\tLINKS_Name = {lua_string(label)},",
                f"\t\t\t\t\tICS_ControlPage = {lua_string(page)},",
                "\t\t\t\t},",
            ]
        )
        return lines

    def unique_input_id(self, base: str) -> str:
        clean = clean_tool_name(base, "Control")
        if clean not in self.published_input_ids:
            self.published_input_ids.add(clean)
            return clean
        index = 2
        while f"{clean}_{index}" in self.published_input_ids:
            index += 1
        name = f"{clean}_{index}"
        self.published_input_ids.add(name)
        return name

    def publish_instance_input(
        self,
        source_op: str,
        source: str,
        name: str | None,
        page: str,
        *,
        default: Any | None = None,
        default_x: float | None = None,
        default_y: float | None = None,
        control_group: int | None = None,
        input_id: str | None = None,
    ) -> None:
        if not self.wrap_group:
            return
        key = (source_op, source, input_id or name or page)
        if key in self.published_input_keys:
            return
        self.published_input_keys.add(key)
        self.published_inputs.append(
            {
                "id": self.unique_input_id(input_id or f"{page}_{name or source}_{source_op}_{source}"),
                "source_op": source_op,
                "source": source,
                "name": name,
                "page": page,
                "default": default,
                "default_x": default_x,
                "default_y": default_y,
                "control_group": control_group,
            }
        )

    def publish_rgba_controls(
        self,
        key_prefix: str,
        source_op: str,
        sources: tuple[str, str, str, str],
        label: str,
        page: str,
        color: tuple[float, float, float],
        alpha: float = 1.0,
    ) -> None:
        if self.brand_palette:
            self.publish_brand_rgba_controls(key_prefix, source_op, sources, label, page, color, alpha)
            return
        group = self.next_control_group
        self.next_control_group += 1
        defaults = (color[0], color[1], color[2], alpha)
        suffixes = ("Red", "Green", "Blue", "Alpha")
        for index, source in enumerate(sources):
            self.publish_instance_input(
                source_op,
                source,
                label if index == 0 else None,
                page,
                default=defaults[index],
                control_group=group,
                input_id=f"{key_prefix}_{suffixes[index]}",
            )

    def publish_brand_rgba_controls(
        self,
        key_prefix: str,
        source_op: str,
        sources: tuple[str, str, str, str],
        label: str,
        page: str,
        color: tuple[float, float, float],
        alpha: float = 1.0,
    ) -> None:
        safe_prefix = clean_tool_name(key_prefix, "Color")
        preset_id = clean_tool_name(f"{safe_prefix}_Preset", "ColorPreset")
        header_id = clean_tool_name(f"{safe_prefix}_PresetHdr", "ColorPresetHdr")
        custom_tool = self.add_custom_color_source(f"{safe_prefix}_Custom", color, alpha)

        self.add_tool_user_controls(
            source_op,
            self.render_palette_user_controls(header_id, preset_id, label, page),
        )

        self.publish_instance_input(
            source_op,
            header_id,
            label,
            page,
            input_id=f"{safe_prefix}_PresetHdr",
        )
        self.publish_instance_input(
            source_op,
            preset_id,
            label,
            page,
            default=0,
            input_id=f"{safe_prefix}_Preset",
        )

        custom_group = self.next_control_group
        self.next_control_group += 1
        custom_sources = ("TopLeftRed", "TopLeftGreen", "TopLeftBlue", "TopLeftAlpha")
        defaults = (color[0], color[1], color[2], alpha)
        suffixes = ("Red", "Green", "Blue", "Alpha")
        for index, custom_source in enumerate(custom_sources):
            self.publish_instance_input(
                custom_tool,
                custom_source,
                f"Custom {label}" if index == 0 else None,
                page,
                default=defaults[index],
                control_group=custom_group,
                input_id=f"{key_prefix}_{suffixes[index]}",
            )

        for index, source in enumerate(sources[:3]):
            expression = self.palette_rgb_expression(preset_id, custom_tool, custom_sources[index], index)
            self.set_tool_input_expression(source_op, source, color[index], expression)
        alpha_expression = self.palette_alpha_expression(preset_id, custom_tool, custom_sources[3], alpha)
        if sources[3] == "TopLeftAlpha":
            self.set_tool_input_value(source_op, sources[3], 1.0)
            self.merge_blend_expression_by_foreground[source_op] = alpha_expression
        else:
            self.set_tool_input_expression(source_op, sources[3], alpha, alpha_expression)

    def publish_number_control(
        self,
        key_prefix: str,
        source_op: str,
        source: str,
        label: str,
        page: str,
        default: float,
    ) -> None:
        self.publish_instance_input(
            source_op,
            source,
            label,
            page,
            default=default,
            input_id=key_prefix,
        )

    def default_all_caps_enabled(self) -> int:
        for layer in self.comp.get("layers", []):
            if isinstance(layer, dict) and (layer.get("text") or {}).get("allCaps"):
                return 1
        return 0

    def all_caps_control(self, host_tool: str) -> str:
        if self.all_caps_control_tool:
            return self.all_caps_control_tool
        default = self.default_all_caps_enabled()
        self.add_tool_user_controls(
            host_tool,
            [
                "\t\t\t\tAllCaps = {",
                "\t\t\t\t\tLINKS_Name = \"All Caps\",",
                "\t\t\t\t\tLINKID_DataType = \"Number\",",
                "\t\t\t\t\tINPID_InputControl = \"CheckboxControl\",",
                f"\t\t\t\t\tINP_Default = {lua_number(default)},",
                "\t\t\t\t\tINP_Integer = false,",
                "\t\t\t\t\tINP_MinScale = 0,",
                "\t\t\t\t\tINP_MaxScale = 1,",
                "\t\t\t\t\tINP_MinAllowed = 0,",
                "\t\t\t\t\tINP_MaxAllowed = 1,",
                "\t\t\t\t\tCBC_TriState = false,",
                "\t\t\t\t\tICS_ControlPage = \"Text\",",
                "\t\t\t\t},",
            ],
        )
        self.publish_instance_input(
            host_tool,
            "AllCaps",
            "All Caps",
            "Text",
            default=default,
            input_id="Text_AllCaps",
        )
        self.all_caps_control_tool = host_tool
        return host_tool

    def add_text_source_control(self, base: str, source_text: str, label: str) -> str:
        source_tool = self.unique_name(f"{base}_TextSource")
        body = "TextPlus {\n" + self.render_inputs([("StyledText", input_value(source_text))]) + "\n\t\t}"
        self.add_tool(source_tool, body)
        self.publish_instance_input(
            source_tool,
            "StyledText",
            label,
            "Text",
            default=0,
            input_id=f"{base}_StyledText",
        )
        return source_tool

    def all_caps_text_expression(self, source_tool: str, host_tool: str) -> str:
        all_caps_tool = self.all_caps_control(host_tool)
        control = "AllCaps" if all_caps_tool == host_tool else f"{all_caps_tool}.AllCaps"
        raw_text = f"tostring({source_tool}.StyledText.Value)"
        return f"iif({control} > 0.5, string.upper({raw_text}), {raw_text})"

    def publish_range_control(
        self,
        key_prefix: str,
        source_op: str,
        start_source: str,
        end_source: str,
        page: str,
        start_default: float,
        end_default: float,
        *,
        start_label: str = "Start",
        end_label: str = "End",
    ) -> None:
        group = self.next_control_group
        self.next_control_group += 1
        self.publish_instance_input(
            source_op,
            start_source,
            start_label,
            page,
            default=start_default,
            control_group=group,
            input_id=f"{key_prefix}_Start",
        )
        self.publish_instance_input(
            source_op,
            end_source,
            end_label,
            page,
            default=end_default,
            control_group=group,
            input_id=f"{key_prefix}_End",
        )

    def display_layer_label(self, layer: dict[str, Any], fallback: str = "Text") -> str:
        name = str(layer.get("name") or fallback).strip() or fallback
        if name.lower().endswith(" move"):
            name = name[:-5].strip()
        return name

    def instance_input_sort_key(self, item: dict[str, Any]) -> tuple[int, int, int, int, str]:
        page = str(item.get("page") or "")
        source_op = str(item.get("source_op") or "")
        input_id = str(item.get("id") or "")
        source = str(item.get("source") or "")
        page_order = {
            "Text": 0,
            "BG": 1,
            "Shadow": 2,
            "Position": 3,
        }.get(page, 50)

        if page == "Text":
            if input_id == "Text_AllCaps":
                layer_order = 0
            elif "Notes" in source_op or "_Notes_" in input_id:
                layer_order = 0
            elif "Citation" in source_op or "_Citation_" in input_id:
                layer_order = 1
            else:
                layer_order = 2

            if source == "StyledText":
                control_order = 0
            elif input_id == "Text_AllCaps":
                control_order = 1
            elif input_id.endswith("_PresetHdr") or input_id.endswith("_Preset"):
                control_order = 9 if "HighlightColor" not in input_id else 19
            elif "HighlightColor" not in input_id and (
                input_id.endswith("_Color_Red")
                or input_id.endswith("_Color_Green")
                or input_id.endswith("_Color_Blue")
                or input_id.endswith("_Color_Alpha")
            ):
                control_order = 10
            elif "HighlightColor" in input_id:
                control_order = 20
            elif "HighlightWriteOn" in input_id:
                control_order = 30
            else:
                control_order = 90
        elif page == "BG":
            layer_order = 0
            if input_id.endswith("_PresetHdr") or input_id.endswith("_Preset"):
                control_order = 0
            elif "_Color_" in input_id:
                control_order = 0
            elif source == "Width" or "Width" in input_id:
                control_order = 10
            else:
                control_order = 90
        elif page == "Shadow":
            layer_order = 0
            if source == "shadowStrength":
                control_order = 0
            elif source == "ShadowDistance":
                control_order = 1
            elif source == "shadowBlur":
                control_order = 2
            else:
                control_order = 90
        else:
            layer_order = 0
            control_order = 0

        channel_order = {
            "Red1": 0,
            "TopLeftRed": 0,
            "Green1": 1,
            "TopLeftGreen": 1,
            "Blue1": 2,
            "TopLeftBlue": 2,
            "Alpha1": 3,
            "TopLeftAlpha": 3,
            "Start": 0,
            "End": 1,
        }.get(source, 0)
        if input_id.endswith("_PresetHdr"):
            channel_order = -2
        elif input_id.endswith("_Preset"):
            channel_order = -1
        return (page_order, layer_order, control_order, channel_order, input_id)

    def unique_name(self, base: str) -> str:
        base = clean_tool_name(base)
        existing = {name for name, _ in self.tools}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def render_inputs(self, inputs: list[tuple[str, str]]) -> str:
        lines = ["\t\t\tInputs = {"]
        for key, value in inputs:
            lines.append(f"\t\t\t\t{key} = {value},")
        lines.append("\t\t\t}")
        return "\n".join(lines)

    def add_background(
        self,
        name: str,
        color: tuple[float, float, float],
        alpha: float,
        mask: str | None = None,
    ) -> str:
        tool = self.unique_name(name)
        inputs = [
            ("GlobalOut", input_value(self.end_frame)),
            ("Width", input_value(self.target_w)),
            ("Height", input_value(self.target_h)),
            ("UseFrameFormatSettings", input_value(1)),
        ]
        if mask:
            inputs.append(("EffectMask", input_link(mask, "Mask")))
        inputs.extend(
            [
                ("TopLeftRed", input_value(color[0])),
                ("TopLeftGreen", input_value(color[1])),
                ("TopLeftBlue", input_value(color[2])),
                ("TopLeftAlpha", input_value(alpha)),
            ]
        )
        body = "Background {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def add_rectangle_mask(
        self,
        name: str,
        center: tuple[float, float] | str,
        width: float | str,
        height: float | str,
        roundness: float = 0,
    ) -> str:
        tool = self.unique_name(name)
        center_input = (
            center
            if isinstance(center, str)
            else input_value([round_num(center[0]), round_num(center[1])])
        )
        width_input = width if isinstance(width, str) else input_value(width)
        height_input = height if isinstance(height, str) else input_value(height)
        inputs = [
            ("MaskWidth", input_value(self.target_w)),
            ("MaskHeight", input_value(self.target_h)),
            ("PixelAspect", input_value([1, 1])),
            ("Center", center_input),
            ("Width", width_input),
            ("Height", height_input),
        ]
        if roundness:
            inputs.append(("CornerRadius", input_value(roundness / max(self.source_w, self.source_h))))
        body = "RectangleMask {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def render_polyline_value(self, points: list[Any], closed: bool) -> str:
        lines = [
            "Polyline {",
            f"\t\t\t\t\t\tClosed = {lua_bool(closed)},",
            "\t\t\t\t\t\tPoints = {",
        ]
        has_any_tangent = any(
            isinstance(point, dict)
            and (
                point.get("hasTangents")
                or any(abs(float(point.get(key) or 0)) > 1e-9 for key in ("lx", "ly", "rx", "ry"))
            )
            for point in points
        )
        for point in points:
            if isinstance(point, dict):
                x = float(point.get("x") or 0) - 0.5
                y = float(point.get("y") or 0) - 0.5
                fields = [
                    "Linear = true",
                    f"X = {lua_number(x)}",
                    f"Y = {lua_number(y)}",
                ]
                if has_any_tangent:
                    fields.extend(
                        [
                            f"LX = {lua_number(point.get('lx') or 0)}",
                            f"LY = {lua_number(point.get('ly') or 0)}",
                            f"RX = {lua_number(point.get('rx') or 0)}",
                            f"RY = {lua_number(point.get('ry') or 0)}",
                        ]
                    )
            else:
                x = float(point[0]) - 0.5
                y = float(point[1]) - 0.5
                fields = [
                    "Linear = true",
                    f"X = {lua_number(x)}",
                    f"Y = {lua_number(y)}",
                ]
            lines.append("\t\t\t\t\t\t\t{ " + ", ".join(fields) + " },")
        lines.extend(["\t\t\t\t\t\t}", "\t\t\t\t\t}"])
        return "\n".join(lines)

    def add_polyline_value_spline(
        self,
        name: str,
        points: list[Any],
        closed: bool,
    ) -> str:
        tool = self.unique_name(name)
        polyline = self.render_polyline_value(points, closed)
        body = (
            "BezierSpline {\n"
            "\t\t\tSplineColor = { Red = 173, Green = 255, Blue = 47 },\n"
            "\t\t\tCtrlWZoom = false,\n"
            "\t\t\tKeyFrames = {\n"
            "\t\t\t\t[0] = { 0, Flags = { Linear = true, LockedY = true }, Value = "
            + polyline
            + " },\n"
            "\t\t\t}\n"
            "\t\t}"
        )
        self.add_tool(tool, body)
        return tool

    def add_polyline_mask(
        self,
        name: str,
        points: list[Any],
        closed: bool = True,
    ) -> str:
        tool = self.unique_name(name)
        spline = self.add_polyline_value_spline(f"{tool}Polyline", points, closed)
        disabled_polyline = (
            "Input {\n"
            "\t\t\t\t\tValue = Polyline {\n"
            "\t\t\t\t\t},\n"
            "\t\t\t\t\tDisabled = true,\n"
            "\t\t\t\t}"
        )
        inputs = [
            ("MaskWidth", input_value(self.target_w)),
            ("MaskHeight", input_value(self.target_h)),
            ("PixelAspect", input_value([1, 1])),
            ("Polyline", input_link(spline, "Value")),
            ("Polyline2", disabled_polyline),
        ]
        body = (
            "PolylineMask {\n"
            "\t\t\tDrawMode = \"InsertAndModify\",\n"
            "\t\t\tDrawMode2 = \"InsertAndModify\",\n"
            + self.render_inputs(inputs)
            + "\n\t\t}"
        )
        self.add_tool(tool, body)
        return tool

    def add_drop_shadow(
        self,
        name: str,
        source: str,
        shadow: dict[str, Any],
        control_label: str | None = None,
    ) -> str:
        tool = self.unique_name(name)
        opacity = scalar_number(shadow.get("opacity"), 0)
        if opacity > 1:
            opacity = opacity / 255 if opacity > 100 else opacity / 100
        opacity = clamp(opacity, 0, 1)
        distance = scalar_number(shadow.get("distance"), 0) / max(self.source_h, 1)
        softness = clamp(scalar_number(shadow.get("softness"), 0) / 250, 0, 1)
        angle = scalar_number(shadow.get("direction"), 135)
        color = color3(shadow.get("color"), (0, 0, 0))
        linked_shadow_tool = self.primary_shadow_tool
        strength_input = input_value(opacity)
        distance_input = input_value(distance)
        blur_input = input_value(softness)
        if linked_shadow_tool:
            strength_input = input_value_expression(opacity, f"{linked_shadow_tool}.shadowStrength")
            distance_input = input_value_expression(distance, f"{linked_shadow_tool}.ShadowDistance")
            blur_input = input_value_expression(softness, f"{linked_shadow_tool}.shadowBlur")
        inputs = [
            ("Source", input_link(source)),
            ("shadowStrength", strength_input),
            ("shadowAngle", input_value(angle)),
            ("ShadowDistance", distance_input),
            ("shadowBlur", blur_input),
            ("shadowColorRed", input_value(color[0])),
            ("shadowColorGreen", input_value(color[1])),
            ("shadowColorBlue", input_value(color[2])),
            ("isLegacyComp", input_value(0)),
            ("blendGroup", input_value(0)),
            ("blendIn", input_value(1)),
            ("blend", input_value(0)),
            ("ignoreContentShape", input_value(0)),
            ("legacyIsProcessRGBOnly", input_value(0)),
            ("IsNoTemporalFramesReqd", input_value(0)),
            ("refreshTrigger", input_value(1)),
            ("srcProcessingAlphaMode", input_value(2)),
            ("dstProcessingAlphaMode", input_value(2)),
            ("resolvefxVersion", input_value("2.0")),
        ]
        body = "ofx.com.blackmagicdesign.resolvefx.DropShadow {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        if not linked_shadow_tool:
            self.primary_shadow_tool = tool
            self.publish_number_control(
                "TextDropShadowAlpha",
                tool,
                "shadowStrength",
                "Text Drop Shadow Alpha",
                "Shadow",
                opacity,
            )
            self.publish_number_control(
                "TextDropShadowDistance",
                tool,
                "ShadowDistance",
                "Text Drop Shadow Distance",
                "Shadow",
                distance,
            )
            self.publish_number_control(
                "TextDropShadowBlur",
                tool,
                "shadowBlur",
                "Text Drop Shadow Blur",
                "Shadow",
                softness,
            )
        return tool

    def source_path_candidates(self, layer: dict[str, Any]) -> list[str]:
        footage = layer.get("footage") or {}
        candidates = []
        if footage.get("sourcePath"):
            candidates.append(str(footage["sourcePath"]))
        for candidate in footage.get("sourcePathCandidates") or []:
            if isinstance(candidate, dict) and candidate.get("path"):
                candidates.append(str(candidate["path"]))
            elif isinstance(candidate, str):
                candidates.append(candidate)
        source = layer.get("source") or {}
        for key in ("file", "path", "sourcePath", "filename"):
            if source.get(key):
                candidates.append(str(source[key]))
        seen = set()
        out = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def project_relative_source_candidates(self, candidate: str) -> list[str]:
        if not self.project_dir:
            return []
        project_dir = self.project_dir
        parts = [part for part in re.split(r"[\\/]+", candidate) if part and not part.endswith(":")]
        tails: list[Path] = []
        for marker in ("collectprj", "(Footage)", "Footage"):
            if marker in parts:
                index = parts.index(marker)
                tail_parts = parts[index + 1 :] if marker == "collectprj" else parts[index:]
                if tail_parts:
                    tails.append(Path(*tail_parts))
        if parts:
            tails.append(Path(parts[-1]))

        candidates = []
        seen = set()
        for tail in tails:
            path = project_dir / tail
            text = str(path)
            if text not in seen:
                seen.add(text)
                candidates.append(text)

        return candidates

    def existing_source_path(self, layer: dict[str, Any]) -> tuple[str | None, list[str]]:
        candidates = self.source_path_candidates(layer)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists() and path.is_file():
                return str(path), candidates
        seen = set(candidates)
        for candidate in list(candidates):
            for repaired in self.project_relative_source_candidates(candidate):
                if repaired not in seen:
                    seen.add(repaired)
                    candidates.append(repaired)
                path = Path(repaired).expanduser()
                if path.exists() and path.is_file():
                    return str(path), candidates
        return None, candidates

    def add_loader(self, name: str, source_path: str, layer: dict[str, Any]) -> str:
        tool = self.unique_name(name)
        format_id = fusion_format_id(source_path)
        global_in = self.layer_timeline_frame(layer, "inPoint")
        global_out = self.layer_global_out(layer)
        clip_lines = [
            "\t\t\tClips = {",
            "\t\t\t\tClip {",
            "\t\t\t\t\tID = \"Clip1\",",
            f"\t\t\t\t\tFilename = {lua_string(source_path)},",
        ]
        if format_id:
            clip_lines.append(f"\t\t\t\t\tFormatID = {lua_string(format_id)},")
        clip_lines.extend(
            [
                "\t\t\t\t\tStartFrame = 0,",
                "\t\t\t\t\tLength = 1,",
                "\t\t\t\t\tMultiframe = true,",
                "\t\t\t\t\tTrimIn = 0,",
                "\t\t\t\t\tTrimOut = 0,",
                "\t\t\t\t\tExtendFirst = 0,",
                "\t\t\t\t\tExtendLast = 0,",
                "\t\t\t\t\tLoop = 1,",
                f"\t\t\t\t\tGlobalStart = {int(global_in)},",
                f"\t\t\t\t\tGlobalEnd = {int(global_out)},",
                "\t\t\t\t}",
                "\t\t\t},",
            ]
        )
        inputs = [
            ("GlobalIn", input_value(global_in)),
            ("GlobalOut", input_value(global_out)),
        ]
        body = "Loader {\n" + "\n".join(clip_lines) + "\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def layer_size_value(self, layer: dict[str, Any]) -> float:
        scale_prop = find_prop(layer.get("transform"), "ADBE Scale")
        value = first_value(scale_prop)
        scale = self.scale_factor(value, 1.0)
        return scale if scale > 0 else 1.0

    def scale_factor(self, value: Any, default: float = 1.0) -> float:
        if isinstance(value, list) and value:
            values = [abs(float(item)) for item in value[:2] if item is not None]
            if not values:
                return default
            scale = sum(values) / len(values)
        else:
            scale = scalar_number(value, default)
        if scale > 10:
            scale /= 100
        return scale

    def layer_angle_value(self, layer: dict[str, Any]) -> float:
        rotate_prop = find_prop(layer.get("transform"), "ADBE Rotate Z")
        return scalar_number(first_value(rotate_prop), 0)

    def add_transform_layer(
        self,
        name: str,
        source: str,
        layer: dict[str, Any],
        *,
        matte_mask: str | None = None,
    ) -> str:
        tool = self.unique_name(name)
        center_input, _ = self.layer_center_input(tool, layer)
        inputs = [
            ("Center", center_input),
            ("Size", input_value(self.layer_size_value(layer))),
        ]
        angle = self.layer_angle_value(layer)
        if angle:
            inputs.append(("Angle", input_value(angle)))
        if matte_mask:
            inputs.append(("EffectMask", input_link(matte_mask, "Mask")))
        inputs.append(("Input", input_link(source)))
        body = "Transform {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def apply_layer_effects(self, tool: str, layer: dict[str, Any]) -> str:
        effects = layer.get("effects") or {}
        shadow = effects.get("dropShadow") if isinstance(effects, dict) else None
        if isinstance(shadow, dict):
            return self.add_drop_shadow(
                f"{tool}_DropShadow",
                tool,
                shadow,
                control_label=self.display_layer_label(layer),
            )
        return tool

    def bezier_spline(
        self,
        name: str,
        samples: list[tuple[int, float]],
        interpolation_hints: dict[int, str] | None = None,
    ) -> str:
        tool = self.unique_name(name)
        interpolation_hints = interpolation_hints or {}
        handles: dict[int, dict[str, tuple[float, float]]] = {}
        for index, (frame, value) in enumerate(samples[:-1]):
            next_frame, next_value = samples[index + 1]
            if next_frame <= frame or abs(float(next_value) - float(value)) <= 1e-6:
                continue
            if interpolation_hints.get(int(frame)) != "bezier" and interpolation_hints.get(int(next_frame)) != "bezier":
                continue
            handle_dt = (float(next_frame) - float(frame)) / 3
            handles.setdefault(int(frame), {})["RH"] = (float(frame) + handle_dt, float(value))
            handles.setdefault(int(next_frame), {})["LH"] = (float(next_frame) - handle_dt, float(next_value))

        lines = [
            "BezierSpline {",
            "\t\t\tSplineColor = { Red = 205, Green = 205, Blue = 205 },",
            "\t\t\tNameSet = true,",
            "\t\t\tKeyFrames = {",
        ]
        for frame, value in samples:
            frame = int(frame)
            fields = [str(lua_number(value))]
            key_handles = handles.get(frame) or {}
            if key_handles.get("LH"):
                lh_frame, lh_value = key_handles["LH"]
                fields.append(f"LH = {{ {lua_number(lh_frame)}, {lua_number(lh_value)} }}")
            if key_handles.get("RH"):
                rh_frame, rh_value = key_handles["RH"]
                fields.append(f"RH = {{ {lua_number(rh_frame)}, {lua_number(rh_value)} }}")
            if not key_handles:
                fields.append("Flags = { Linear = true }")
            lines.append(f"\t\t\t\t[{frame}] = {{ " + ", ".join(fields) + " },")
        lines.extend(["\t\t\t}", "\t\t}"])
        self.add_tool(tool, "\n".join(lines))
        return tool

    def xy_path(
        self,
        name: str,
        samples: list[tuple[int, tuple[float, float]]],
        interpolation_hints: dict[int, str] | None = None,
    ) -> str:
        path = self.unique_name(name)
        x = self.bezier_spline(f"{path}X", [(f, p[0]) for f, p in samples], interpolation_hints)
        y = self.bezier_spline(f"{path}Y", [(f, p[1]) for f, p in samples], interpolation_hints)
        body = (
            "XYPath {\n"
            "\t\t\tShowKeyPoints = false,\n"
            "\t\t\tDrawMode = \"ModifyOnly\",\n"
            "\t\t\tInputs = {\n"
            f"\t\t\t\tX = {input_link(x, 'Value')},\n"
            f"\t\t\t\tY = {input_link(y, 'Value')},\n"
            "\t\t\t}\n"
            "\t\t}"
        )
        self.add_tool(path, body)
        return path

    def layer_center_input(self, tool_name: str, layer: dict[str, Any]) -> tuple[str, list[str]]:
        has_position_samples = bool(self.prop_sample_frames(layer, "ADBE Position"))
        for parent in self.parent_chain(layer):
            has_position_samples = has_position_samples or bool(self.prop_sample_frames(parent, "ADBE Position"))
        for driver in self.motion_driver_layers(layer):
            has_position_samples = has_position_samples or bool(self.prop_sample_frames(driver, "ADBE Position"))
        if not has_position_samples:
            position_prop = find_prop(layer.get("transform"), "ADBE Position")
            if first_value(position_prop) is not None:
                position_center = layer_center_from_position(layer, self.source_w, self.source_h)
                return input_value([round_num(position_center[0]), round_num(position_center[1])]), []
            bounds_center = layer_center_from_bounds(layer, self.source_w, self.source_h)
            if bounds_center:
                return input_value([round_num(bounds_center[0]), round_num(bounds_center[1])]), []
        samples = preserve_point_plateaus(self.accumulated_position_samples(layer))
        if len(samples) > 1:
            path = self.xy_path(
                f"{tool_name}Center",
                samples,
                self.accumulated_position_interpolation_hints(layer),
            )
            return input_link(path, "Value"), [path]
        if len(samples) == 1:
            return input_value([round_num(samples[0][1][0]), round_num(samples[0][1][1])]), []
        center = layer_center(layer, self.source_w, self.source_h)
        return input_value([round_num(center[0]), round_num(center[1])]), []

    def property_interpolation_hints(self, layer: dict[str, Any], match_name: str) -> dict[int, str]:
        hints: dict[int, str] = {}
        for sample in key_samples(find_prop(layer.get("transform"), match_name)):
            hint = sample.get("interpolationHint")
            if hint:
                hints[self.sample_frame(sample)] = str(hint)
        return hints

    def accumulated_position_interpolation_hints(self, layer: dict[str, Any]) -> dict[int, str]:
        hints = self.property_interpolation_hints(layer, "ADBE Position")
        for parent in self.parent_chain(layer):
            hints = merge_interpolation_hints(hints, self.property_interpolation_hints(parent, "ADBE Position"))
        for driver in self.motion_driver_layers(layer):
            hints = merge_interpolation_hints(hints, self.property_interpolation_hints(driver, "ADBE Position"))
        return hints

    def layer_scale_samples(self, layer: dict[str, Any]) -> list[tuple[int, float]]:
        scale_prop = find_prop(layer.get("transform"), "ADBE Scale")
        samples = []
        for sample in key_samples(scale_prop):
            frame = self.sample_frame(sample)
            scale = max(self.scale_factor(sample.get("value"), 1.0), 0.0)
            samples.append((frame, scale))
        return preserve_scalar_plateaus(samples)

    def rectangle_mask_reveal_center_input(
        self,
        tool_name: str,
        layer: dict[str, Any],
        final_height: float,
        reveal_samples: list[tuple[int, float]],
    ) -> str:
        peak = max((scale for _, scale in reveal_samples), default=0.0)
        if peak <= 1e-6:
            return self.layer_center_input(tool_name, layer)[0]
        scale_by_frame = {frame: scale for frame, scale in reveal_samples}
        frames = sorted(set(self.prop_sample_frames(layer, "ADBE Position")) | set(scale_by_frame))
        if not frames:
            frames = [0]
        samples = []
        scale_prop = find_prop(layer.get("transform"), "ADBE Scale")
        for frame in frames:
            x, y = self.accumulated_ae_position_at_frame(layer, frame)
            scale = scale_by_frame.get(frame)
            if scale is None:
                scale = max(self.scale_factor(self.prop_value_at_frame(scale_prop, frame, [peak, peak, peak]), peak), 0.0)
            center_x = x / self.source_w
            base_center_y = 1 - (y / self.source_h)
            animated_height = final_height * (scale / peak)
            bottom_anchored_y = base_center_y - ((final_height - animated_height) / 2)
            samples.append((frame, (center_x, bottom_anchored_y)))
        samples = preserve_point_plateaus(samples)
        if len(samples) > 1:
            path = self.xy_path(
                f"{tool_name}Center",
                samples,
                merge_interpolation_hints(
                    self.property_interpolation_hints(layer, "ADBE Position"),
                    self.property_interpolation_hints(layer, "ADBE Scale"),
                ),
            )
            return input_link(path, "Value")
        return input_value([round_num(samples[0][1][0]), round_num(samples[0][1][1])])

    def rectangle_mask_size_inputs(
        self,
        base: str,
        layer: dict[str, Any],
        width: float,
        height: float,
        reveal_samples: list[tuple[int, float]] | None = None,
    ) -> tuple[str, str]:
        samples = reveal_samples if reveal_samples is not None else self.layer_scale_samples(layer)
        if reveal_samples:
            width_input = input_value(width)
            height_spline = self.bezier_spline(
                f"{base}_MaskHeight",
                [(frame, height * scale) for frame, scale in reveal_samples],
                self.property_interpolation_hints(layer, "ADBE Scale"),
            )
            return width_input, input_link(height_spline, "Value")

        if len(samples) > 1:
            scale_hints = self.property_interpolation_hints(layer, "ADBE Scale")
            width_spline = self.bezier_spline(
                f"{base}_MaskWidth",
                [(frame, width * scale) for frame, scale in samples],
                scale_hints,
            )
            height_spline = self.bezier_spline(
                f"{base}_MaskHeight",
                [(frame, height * scale) for frame, scale in samples],
                scale_hints,
            )
            return input_link(width_spline, "Value"), input_link(height_spline, "Value")

        scale_prop = find_prop(layer.get("transform"), "ADBE Scale")
        scale = self.scale_factor(first_value(scale_prop), 1.0)
        scale = max(scale, 0.0)
        return input_value(width * scale), input_value(height * scale)

    def layer_reveal_scale_samples(self, layer: dict[str, Any]) -> list[tuple[int, float]]:
        samples = self.layer_scale_samples(layer)
        return samples if self.is_reveal_scale_pattern(samples) else []

    def is_reveal_scale_pattern(self, samples: list[tuple[int, float]]) -> bool:
        if len(samples) < 3:
            return False
        values = [max(float(value), 0.0) for _, value in samples]
        peak = max(values)
        if peak <= 1e-6:
            return False
        return values[0] <= peak * 0.01 and values[-1] <= peak * 0.01

    def is_inverted_reveal_offset_pattern(self, samples: list[tuple[int, float]]) -> bool:
        if len(samples) < 3:
            return False
        values = [max(float(value), 0.0) for _, value in samples]
        peak = max(values)
        valley = min(values)
        if peak <= 1e-6:
            return False
        return values[0] >= peak * 0.99 and values[-1] >= peak * 0.99 and valley <= peak * 0.01

    def reveal_reference_frame(self, layer: dict[str, Any]) -> int:
        reveal_samples = self.layer_reveal_scale_samples(layer)
        if reveal_samples:
            peak = max(value for _, value in reveal_samples)
            for frame, value in reveal_samples:
                if abs(value - peak) <= max(peak * 0.01, 1e-6):
                    return frame
        position_samples = self.prop_sample_frames(layer, "ADBE Position")
        if len(position_samples) > 1:
            return sorted(position_samples)[1]
        return 0

    def referenced_text_layers_for_shape(self, layer: dict[str, Any]) -> list[dict[str, Any]]:
        names: list[str] = []
        for evaluation in layer.get("nativeExpressionEvaluations") or layer.get("expressionEvaluations") or []:
            for term in evaluation.get("resolvedTerms") or []:
                if term.get("kind") == "sourceRect" and term.get("layer"):
                    names.append(str(term["layer"]))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ref = self.layers_by_name.get(name)
            if isinstance(ref, dict) and (ref.get("type") == "text" or ref.get("text")):
                out.append(ref)
        return out

    def rectangle_size_from_expression_evaluations(
        self,
        layer: dict[str, Any],
        fallback_width_px: float,
        fallback_height_px: float,
    ) -> tuple[float, float]:
        width_px = float(fallback_width_px)
        height_px = float(fallback_height_px)
        evaluations = layer.get("nativeExpressionEvaluations") or layer.get("expressionEvaluations") or []
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            match_name = str(evaluation.get("matchName") or "")
            expression = str(evaluation.get("expression") or "")
            assignments = evaluation.get("assignments") or {}
            final_value = evaluation.get("finalValueCandidate")
            is_rect_size = (
                match_name == "ADBE Vector Rect Size"
                or "Vector Rect Size" in match_name
                or "newWidth" in assignments
                or "sourceRectAtTime" in expression
            )
            if not is_rect_size:
                continue
            if isinstance(final_value, list) and len(final_value) >= 2:
                width_px = max(width_px, scalar_number(final_value[0], width_px))
                height_px = max(height_px, scalar_number(final_value[1], height_px))
            if isinstance(assignments, dict):
                new_width = scalar_number(assignments.get("newWidth"), 0)
                new_height = scalar_number(assignments.get("newHeight"), 0)
                if new_width > 0:
                    width_px = max(width_px, new_width)
                if new_height > 0:
                    height_px = max(height_px, new_height)

            source_widths: list[float] = []
            source_heights: list[float] = []
            width_adjust = 0.0
            height_adjust = 0.0
            for term in evaluation.get("resolvedTerms") or []:
                if not isinstance(term, dict):
                    continue
                value = term.get("value")
                if term.get("kind") == "sourceRect":
                    field = str(term.get("field") or term.get("property") or "").lower()
                    if field == "width":
                        resolved_width = scalar_number(value, 0)
                        if resolved_width > 0:
                            source_widths.append(resolved_width)
                    elif field == "height":
                        resolved_height = scalar_number(value, 0)
                        if resolved_height > 0:
                            source_heights.append(resolved_height)
                elif term.get("kind") == "effectParameter":
                    effect = str(term.get("effect") or "").lower()
                    resolved = scalar_number(value, 0)
                    if "width" in effect:
                        width_adjust += resolved
                    elif "height" in effect or "offset" in effect:
                        height_adjust += resolved
            if source_widths:
                width_px = max(width_px, max(source_widths) + width_adjust)
            if source_heights:
                height_px = max(height_px, sum(source_heights) + height_adjust)
        return width_px, height_px

    def shape_text_y_offset(self, layer: dict[str, Any]) -> float:
        key = id(layer)
        if key in self.shape_text_y_offsets:
            return self.shape_text_y_offsets[key]
        self.shape_text_y_offsets[key] = 0.0
        if not (layer.get("type") == "shape" or layer.get("shape")):
            return 0.0
        refs = self.referenced_text_layers_for_shape(layer)
        if len(refs) < 2:
            return 0.0
        frame = self.reveal_reference_frame(layer)
        ref_positions = [self.accumulated_ae_position_at_frame(ref, frame)[1] for ref in refs]
        target_y = sum(ref_positions) / len(ref_positions)
        _, current_y = self.ae_position_at_frame(layer, frame)
        offset = target_y - current_y
        self.shape_text_y_offsets[key] = offset
        return offset

    def rectangle_mask_from_shape_layer(self, layer: dict[str, Any], base: str) -> str | None:
        shape = layer.get("shape") or {}
        rect = shape.get("rect") or {}
        if not rect:
            return None
        width_px, height_px = self.rectangle_size_from_expression_evaluations(
            layer,
            float(rect.get("w") or 0),
            float(rect.get("h") or 0),
        )
        width = width_px / self.source_w
        height = height_px / self.source_h
        reveal_samples = self.layer_reveal_scale_samples(layer)
        if reveal_samples:
            peak = max((scale for _, scale in reveal_samples), default=0.0)
            center_input = self.rectangle_mask_reveal_center_input(
                f"{base}_Mask",
                layer,
                height * max(peak, 0.0),
                reveal_samples,
            )
        else:
            center_input, _ = self.layer_center_input(f"{base}_Mask", layer)
        width_input, height_input = self.rectangle_mask_size_inputs(
            base,
            layer,
            width,
            height,
            reveal_samples=reveal_samples or None,
        )
        mask = self.add_rectangle_mask(
            f"{base}_Mask",
            center_input,
            width_input,
            height_input,
            float(rect.get("roundness") or 0),
        )
        peak = max((scale for _, scale in reveal_samples), default=1.0) if reveal_samples else 1.0
        self.mask_dimensions[mask] = (width * max(peak, 0.0), height * max(peak, 0.0))
        return mask

    def fusion_mask_from_layer_mask(self, layer: dict[str, Any], base: str) -> tuple[str | None, dict[str, Any]]:
        info: dict[str, Any] = {
            "mask_count": len(layer.get("masks") or []),
            "unsupported_mask_count": len(layer.get("unsupportedMasks") or []),
        }
        masks = [mask for mask in layer.get("masks") or [] if isinstance(mask, dict) and mask.get("rect")]
        if not masks:
            return None, info

        key = (layer.get("index"), layer.get("name"))
        first = masks[0]
        rect = first.get("rect") or {}
        center = rect.get("normalizedCenter") or [0.5, 0.5]
        width = float(rect.get("normalizedWidth") or 0)
        height = float(rect.get("normalizedHeight") or 0)
        if width <= 0 or height <= 0:
            return None, info

        if key not in self.reported_layer_mask_layers:
            self.reported_layer_mask_layers.add(key)
            if len(masks) > 1:
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "Only the first AE layer mask is applied. Additional masks are preserved in the report.",
                        "mask_count": len(masks),
                    }
                )
            if first.get("inverted"):
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "AE mask inversion was decoded but is not emitted to Fusion yet.",
                        "mask": first.get("name"),
                    }
                )
            if first.get("feather") not in (None, 0, [0, 0]):
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "AE mask feather was decoded but is not emitted to Fusion yet.",
                        "mask": first.get("name"),
                    }
                )
            if first.get("opacity") not in (None, 100, [100]):
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "AE mask opacity was decoded but is not emitted to Fusion yet.",
                        "mask": first.get("name"),
                    }
                )
            if first.get("approximation") in {"polygon-vertices", "bezier-tangents"} and first.get("expansion") not in (None, 0, 0.0):
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "AE mask expansion on polygon masks was decoded but is not emitted to Fusion yet.",
                        "mask": first.get("name"),
                    }
                )
            for unsupported in layer.get("unsupportedMasks") or []:
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "AE layer mask could not be converted.",
                        "mask": unsupported,
                    }
                )

        polygon = first.get("polygon") or {}
        polygon_points = polygon.get("points") if isinstance(polygon, dict) else None
        if first.get("approximation") in {"polygon-vertices", "bezier-tangents"} and isinstance(polygon_points, list) and len(polygon_points) >= 3:
            tool = self.add_polyline_mask(
                f"{base}_LayerMask",
                polygon_points,
                closed=semantic_bool(polygon.get("closed", True)),
            )
            tool_type = "PolylineMask"
        else:
            tool = self.add_rectangle_mask(
                f"{base}_LayerMask",
                (float(center[0]), float(center[1])),
                width,
                height,
            )
            tool_type = "RectangleMask"
        info.update(
            {
                "layer_mask": tool,
                "layer_mask_name": first.get("name"),
                "layer_mask_approximation": first.get("approximation"),
                "layer_mask_tool_type": tool_type,
            }
        )
        return tool, info

    def effect_mask_for_layer(self, layer: dict[str, Any], base: str) -> tuple[str | None, dict[str, Any]]:
        matte_mask = self.matte_masks_by_target.get(str(layer.get("name")))
        if matte_mask:
            info = {
                "matte_mask": matte_mask,
                "mask_count": len(layer.get("masks") or []),
                "unsupported_mask_count": len(layer.get("unsupportedMasks") or []),
            }
            if layer.get("masks") or layer.get("unsupportedMasks"):
                self.report["warnings"].append(
                    {
                        "index": layer.get("index"),
                        "name": layer.get("name"),
                        "warning": "Layer has both a track matte and AE layer masks. The track matte is applied; layer masks are preserved in the report.",
                    }
                )
            return matte_mask, info
        if (
            layer.get("type") == "text"
            and self.motion_driver_layers(layer)
            and not layer.get("masks")
            and not layer.get("unsupportedMasks")
            and len(self.matte_masks_by_target) == 1
        ):
            mask = next(iter(self.matte_masks_by_target.values()))
            return mask, {
                "reveal_mask": mask,
                "mask_count": 0,
                "unsupported_mask_count": 0,
            }
        layer_mask, info = self.fusion_mask_from_layer_mask(layer, base)
        return layer_mask, info

    def prepare_track_matte_masks(self, layers: list[dict[str, Any]]) -> None:
        for layer in layers:
            transfer = layer.get("nativeTransfer") or {}
            target = (transfer.get("usedAsTrackMatteCandidateFor") or {}).get("layerName")
            if not target or layer.get("type") != "shape":
                continue
            base = clean_tool_name(f"AE_Matte_{layer.get('index', '')}_{layer.get('name', 'Matte')}")
            mask = self.rectangle_mask_from_shape_layer(layer, base)
            if mask:
                self.matte_masks_by_target[str(target)] = mask

    def layer_opacity_samples(self, layer: dict[str, Any]) -> list[tuple[int, float]]:
        opacity_prop = find_prop(layer.get("transform"), "ADBE Opacity")
        samples = []
        for sample in key_samples(opacity_prop):
            try:
                samples.append((self.sample_frame(sample), clamp(float(sample["value"]) / 100, 0, 1)))
            except (TypeError, ValueError):
                continue
        samples = simplify_scalar_samples(samples)
        if samples:
            return samples
        return []

    def driver_reveal_opacity_samples(self, layer: dict[str, Any]) -> list[tuple[int, float]]:
        drivers = self.motion_driver_layers(layer)
        if not drivers:
            return []
        driver = drivers[0]
        position_prop = find_prop(driver.get("transform"), "ADBE Position")
        samples = []
        for sample in key_samples(position_prop):
            value = sample.get("value")
            if not isinstance(value, list) or len(value) < 2:
                continue
            samples.append((self.sample_frame(sample), max(abs(float(value[0])), abs(float(value[1])))))
        samples = preserve_scalar_plateaus(samples)
        if not self.is_inverted_reveal_offset_pattern(samples):
            return []
        peak = max(value for _, value in samples)
        if peak <= 1e-6:
            return []
        return [(frame, clamp(1 - (value / peak), 0, 1)) for frame, value in samples]

    def prop_sample_frames(self, layer: dict[str, Any], match_name: str) -> list[int]:
        prop = find_prop(layer.get("transform"), match_name)
        return [self.sample_frame(sample) for sample in key_samples(prop)]

    def companion_move_layer(self, layer: dict[str, Any]) -> dict[str, Any] | None:
        name = str(layer.get("name") or "").strip()
        if not name:
            return None
        driver = self.companion_move_layers.get(name)
        if driver is layer:
            return None
        return driver

    def motion_driver_layers(self, layer: dict[str, Any]) -> list[dict[str, Any]]:
        driver = self.companion_move_layer(layer)
        return [driver] if driver else []

    def parent_chain(self, layer: dict[str, Any]) -> list[dict[str, Any]]:
        chain = []
        seen = set()
        parent_index = layer.get("parentIndex")
        while parent_index is not None:
            try:
                idx = int(parent_index)
            except (TypeError, ValueError):
                break
            if idx in seen:
                break
            seen.add(idx)
            parent = self.layers_by_index.get(idx)
            if not parent:
                break
            chain.append(parent)
            parent_index = parent.get("parentIndex")
        return chain

    def prop_value_at_frame(self, prop: dict[str, Any] | None, frame: int, default: Any) -> Any:
        if not prop:
            return default
        samples = key_samples(prop)
        if not samples:
            return first_value(prop, default)
        previous = samples[0]
        for sample in samples:
            sample_frame = self.sample_frame(sample)
            if sample_frame == frame:
                return sample.get("value", default)
            if sample_frame > frame:
                return previous.get("value", default)
            previous = sample
        return previous.get("value", default)

    def expression_position_at_frame(self, layer: dict[str, Any], frame: int) -> tuple[float, float] | None:
        for evaluation in layer.get("nativeExpressionEvaluations") or layer.get("expressionEvaluations") or []:
            if evaluation.get("matchName") != "ADBE Position":
                continue
            expression = str(evaluation.get("expression") or "")
            if "ref.transform.position" not in expression or "newY = refPos[1] + offset" not in expression:
                continue
            ref_name = None
            for term in evaluation.get("resolvedTerms") or []:
                if isinstance(term, dict) and term.get("kind") == "sourceRect" and term.get("layer") != layer.get("name"):
                    ref_name = str(term.get("layer"))
                    break
            if not ref_name:
                match = re.search(r'ref\s*=\s*thisComp\.layer\("([^"]+)"\)', expression)
                if match:
                    ref_name = match.group(1)
            ref = self.layers_by_name.get(ref_name or "")
            if not ref:
                continue
            ref_x, ref_y = self.ae_position_at_frame(ref, frame)
            assignments = evaluation.get("assignments") or {}
            offset = assignments.get("offset")
            if offset is None:
                for term in evaluation.get("resolvedTerms") or []:
                    if isinstance(term, dict) and term.get("kind") == "effectParameter" and term.get("effect") == "Offset":
                        offset = term.get("value")
                        break
            try:
                y = ref_y + float(offset or 0)
            except (TypeError, ValueError):
                y = ref_y
            ref_height = scalar_number(assignments.get("refHeight"), 0)
            if (
                ref_height > 0
                and "refHeight" in expression
                and "newY = refPos[1] + offset" in expression
                and (layer.get("type") == "text" or layer.get("text"))
            ):
                y += ref_height / 2

            x = ref_x
            bw = assignments.get("Bw")
            total_w = assignments.get("totalW")
            try:
                if bw is not None and total_w is not None:
                    x = ref_x - (float(total_w) - float(bw)) / 2
                elif "leftX = Acenter - totalW/2" in expression:
                    spacing_match = re.search(r"totalW\s*=\s*Bw\s*\+\s*([0-9.]+)", expression)
                    spacing = float(spacing_match.group(1)) if spacing_match else 0
                    x = ref_x - spacing / 2
            except (TypeError, ValueError):
                x = ref_x
            return x, y
        return None

    def ae_position_at_frame(
        self,
        layer: dict[str, Any],
        frame: int,
        default: list[float] | tuple[float, float, float] | None = None,
    ) -> tuple[float, float]:
        expression_position = self.expression_position_at_frame(layer, frame)
        if expression_position is not None:
            return expression_position
        position_prop = find_prop(layer.get("transform"), "ADBE Position")
        if default is None:
            default = [self.source_w / 2, self.source_h / 2, 0]
        value = self.prop_value_at_frame(position_prop, frame, default)
        if isinstance(value, list) and len(value) >= 2:
            return float(value[0]), float(value[1])
        return float(default[0]), float(default[1])

    def accumulated_ae_position_at_frame(self, layer: dict[str, Any], frame: int) -> tuple[float, float]:
        x, y = self.ae_position_at_frame(layer, frame)
        for parent in self.parent_chain(layer):
            px, py = self.ae_position_at_frame(parent, frame)
            x += px
            y += py
        for driver in self.motion_driver_layers(layer):
            dx, dy = self.ae_position_at_frame(driver, frame, [0, 0, 0])
            x += dx
            y += dy
        y += self.shape_text_y_offset(layer)
        return x, y

    def accumulated_position_samples(self, layer: dict[str, Any]) -> list[tuple[int, tuple[float, float]]]:
        frames = set(self.prop_sample_frames(layer, "ADBE Position"))
        for parent in self.parent_chain(layer):
            frames.update(self.prop_sample_frames(parent, "ADBE Position"))
        for driver in self.motion_driver_layers(layer):
            frames.update(self.prop_sample_frames(driver, "ADBE Position"))
        if not frames:
            x, y = self.accumulated_ae_position_at_frame(layer, 0)
            return [(0, (x / self.source_w, 1 - (y / self.source_h)))]
        out = []
        for frame in sorted(frames):
            x, y = self.accumulated_ae_position_at_frame(layer, frame)
            out.append((frame, (x / self.source_w, 1 - (y / self.source_h))))
        return out

    def layer_static_opacity(self, layer: dict[str, Any]) -> float:
        opacity_prop = find_prop(layer.get("transform"), "ADBE Opacity")
        try:
            return clamp(float(first_value(opacity_prop, 100)) / 100, 0, 1)
        except (TypeError, ValueError):
            return 1.0

    def add_text_layer(self, layer: dict[str, Any]) -> str | None:
        text = layer.get("text") or {}
        source_text = str(text.get("sourceText") or text.get("text") or "")
        if not source_text:
            return None
        base = clean_tool_name(f"AE_Text_{layer.get('index', '')}_{layer.get('name', 'Text')}")
        tool = self.unique_name(base)
        layer_label = self.display_layer_label(layer)
        source_tool = self.add_text_source_control(base, source_text, layer_label)
        visible_styled_text_input = input_value_expression(source_text, self.all_caps_text_expression(source_tool, tool))
        fill = color3(text.get("fillColor"), (1, 1, 1))
        requested_font_family, requested_font_style = resolve_font_family_style(text.get("font"))
        font_family, font_style = resolve_font_family_style(text.get("font"), validate_installed=True)
        center_input, _ = self.layer_center_input(tool, layer)
        effect_mask, mask_info = self.effect_mask_for_layer(layer, base)

        def text_inputs(
            fill_color: tuple[float, float, float],
            extra: list[tuple[str, str]] | None = None,
            styled_text_input: str | None = None,
        ) -> list[tuple[str, str]]:
            inputs = [
                ("GlobalIn", input_value(self.layer_timeline_frame(layer, "inPoint"))),
                ("GlobalOut", input_value(self.layer_global_out(layer))),
                ("Width", input_value(self.target_w)),
                ("Height", input_value(self.target_h)),
                ("UseFrameFormatSettings", input_value(1)),
                ("Center", center_input),
                ("Red1", input_value(fill_color[0])),
                ("Green1", input_value(fill_color[1])),
                ("Blue1", input_value(fill_color[2])),
                ("Alpha1", input_value(1)),
                ("StyledText", styled_text_input or input_value(source_text)),
                ("Font", input_value(font_family)),
                ("Style", input_value(font_style)),
                ("Size", input_value(font_size_to_fusion(text.get("fontSize"), self.source_h))),
                ("VerticalJustificationNew", input_value(3)),
                ("HorizontalJustificationNew", input_value(3)),
            ]
            if effect_mask:
                inputs.insert(5, ("EffectMask", input_link(effect_mask, "Mask")))
            tracking = text.get("tracking")
            if tracking is not None:
                inputs.append(("CharacterSpacing", input_value(float(tracking) / 1000)))
            if extra:
                inputs.extend(extra)
            return inputs

        inputs = text_inputs(fill, styled_text_input=visible_styled_text_input)
        body = "TextPlus {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        self.publish_rgba_controls(
            f"{base}_Color",
            tool,
            ("Red1", "Green1", "Blue1", "Alpha1"),
            f"{layer_label} Color",
            "Text",
            fill,
            1.0,
        )

        highlight_color = text.get("highlightColor")
        highlight_start = text_range_to_fusion(text.get("highlightStart"))
        highlight_end = text_range_to_fusion(text.get("highlightEnd"))
        if highlight_color and highlight_start is not None and highlight_end is not None:
            highlight_tool = self.unique_name(f"{base}_Highlight")
            highlight_fill = color3(highlight_color, fill)
            highlight_inputs = text_inputs(
                highlight_fill,
                [
                    ("Start", input_value(highlight_start)),
                    ("End", input_value(highlight_end)),
                ],
                styled_text_input=input_value_expression(
                    source_text,
                    f"tostring({tool}.StyledText.Value)",
                ),
            )
            body = "TextPlus {\n" + self.render_inputs(highlight_inputs) + "\n\t\t}"
            self.add_tool(highlight_tool, body)
            self.publish_rgba_controls(
                f"{base}_HighlightColor",
                highlight_tool,
                ("Red1", "Green1", "Blue1", "Alpha1"),
                f"{layer_label} Highlight Color",
                "Text",
                highlight_fill,
                1.0,
            )
            self.publish_range_control(
                f"{base}_HighlightWriteOn",
                highlight_tool,
                "Start",
                "End",
                "Text",
                highlight_start,
                highlight_end,
                start_label="Highlight Start",
                end_label="Highlight End",
            )
            merge = self.unique_name(f"{base}_HighlightMerge")
            merge_inputs = [
                ("Background", input_link(tool)),
                ("Foreground", input_link(highlight_tool)),
                ("PerformDepthMerge", input_value(0)),
            ]
            body = "Merge {\n" + self.render_inputs(merge_inputs) + "\n\t\t}"
            self.add_tool(merge, body)
            tool = merge
        tool = self.apply_layer_effects(tool, layer)
        self.report["converted_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": "text",
                "tool": tool,
                "source_font": text.get("font"),
                "requested_font": requested_font_family,
                "requested_font_style": requested_font_style,
                "font": font_family,
                "font_style": font_style,
                "font_fallback": (
                    font_family != requested_font_family or font_style != requested_font_style
                ),
                "effects": sorted((layer.get("effects") or {}).keys()) if isinstance(layer.get("effects"), dict) else [],
                **mask_info,
            }
        )
        return tool

    def add_shape_layer(self, layer: dict[str, Any]) -> str | None:
        shape = layer.get("shape") or {}
        rect = shape.get("rect") or {}
        if not rect:
            self.report["unsupported_layers"].append(
                {
                    "index": layer.get("index"),
                    "name": layer.get("name"),
                    "type": "shape",
                    "reason": "No simple rectangle geometry was exported for this shape layer.",
                }
            )
            return None
        fill = color3(shape.get("fillColor"), (0, 0, 0))
        alpha = clamp(float(shape.get("opacity", 100)) / 100, 0, 1)
        base = clean_tool_name(f"AE_Shape_{layer.get('index', '')}_{layer.get('name', 'Shape')}")
        matte_mask = self.matte_masks_by_target.get(str(layer.get("name")))
        if (layer.get("masks") or layer.get("unsupportedMasks")) and not matte_mask:
            self.report["warnings"].append(
                {
                    "index": layer.get("index"),
                    "name": layer.get("name"),
                    "warning": "AE layer masks on shape layers are decoded but not combined with shape geometry yet.",
                }
            )
        mask = matte_mask or self.rectangle_mask_from_shape_layer(layer, base)
        if not mask:
            return None
        bg = self.add_background(base, fill, alpha, mask=mask)
        layer_label = self.display_layer_label(layer, "BG")
        if "bg" in layer_label.lower() or "box" in layer_label.lower():
            color_label = "BG Color"
            width_label = "BG Box Width"
        else:
            color_label = f"{layer_label} Color"
            width_label = f"{layer_label} Width"
        self.publish_rgba_controls(
            f"{base}_Color",
            bg,
            ("TopLeftRed", "TopLeftGreen", "TopLeftBlue", "TopLeftAlpha"),
            color_label,
            "BG",
            fill,
            alpha,
        )
        if mask in self.mask_dimensions:
            mask_width, _mask_height = self.mask_dimensions[mask]
            self.publish_number_control(
                f"{base}_Width",
                mask,
                "Width",
                width_label,
                "BG",
                mask_width,
            )
        bg = self.apply_layer_effects(bg, layer)
        self.report["converted_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": "shape",
                "tool": bg,
                "matte_mask": matte_mask,
                "mask_count": len(layer.get("masks") or []),
                "unsupported_mask_count": len(layer.get("unsupportedMasks") or []),
                "effects": sorted((layer.get("effects") or {}).keys()) if isinstance(layer.get("effects"), dict) else [],
            }
        )
        return bg

    def add_solid_layer(self, layer: dict[str, Any]) -> str | None:
        source = layer.get("source") or {}
        main = source.get("mainSource") or {}
        fill = color3(layer.get("solidColor") or main.get("color"), (0, 0, 0))
        base = clean_tool_name(f"AE_Solid_{layer.get('index', '')}_{layer.get('name', 'Solid')}")
        effect_mask, mask_info = self.effect_mask_for_layer(layer, base)
        tool = self.add_background(
            base,
            fill,
            self.layer_static_opacity(layer),
            mask=effect_mask,
        )
        self.report["converted_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": "solid",
                "tool": tool,
                **mask_info,
            }
        )
        return tool

    def add_footage_layer(self, layer: dict[str, Any]) -> str | None:
        source_path, candidates = self.existing_source_path(layer)
        if not source_path:
            self.report["unsupported_layers"].append(
                {
                    "index": layer.get("index"),
                    "name": layer.get("name"),
                    "type": "footage",
                    "reason": "No existing local source file was found for this footage layer.",
                    "source_path_candidates": candidates,
                }
            )
            return None
        base = clean_tool_name(f"AE_Footage_{layer.get('index', '')}_{layer.get('name', 'Footage')}")
        loader = self.add_loader(base, source_path, layer)
        effect_mask, mask_info = self.effect_mask_for_layer(layer, base)
        transformed = self.add_transform_layer(f"{base}_Transform", loader, layer, matte_mask=effect_mask)
        transformed = self.apply_layer_effects(transformed, layer)
        self.report["converted_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": "footage",
                "tool": transformed,
                "source_path": source_path,
                "effects": sorted((layer.get("effects") or {}).keys()) if isinstance(layer.get("effects"), dict) else [],
                **mask_info,
            }
        )
        return transformed

    def record_layer_controls(self, layer: dict[str, Any]) -> None:
        controls = layer.get("controls") or []
        if not controls:
            return
        key = (layer.get("index"), layer.get("name"))
        if key in self.reported_control_layers:
            return
        self.reported_control_layers.add(key)
        self.report["controls"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "role": layer.get("role"),
                "controls": controls,
            }
        )

    def is_consumed_helper_layer(self, layer: dict[str, Any]) -> bool:
        role = layer.get("role")
        if role in {"controlLayer", "utilityLayer"}:
            return True
        if role == "precompOrFootage" and layer.get("type") == "unsupported":
            source_item = ((layer.get("nativeConversionSummary") or {}).get("sourceItem") or {})
            if source_item.get("kind") == "composition":
                return False
            return True
        return False

    def skip_consumed_helper_layer(self, layer: dict[str, Any]) -> None:
        self.report["skipped_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": layer.get("type"),
                "role": layer.get("role"),
                "reason": "AE helper/control layer was consumed for expressions or controls and does not emit a renderable Fusion node.",
                "controls": [control.get("name") for control in layer.get("controls") or []],
            }
        )

    def add_unsupported_placeholder(self, layer: dict[str, Any]) -> None:
        self.report["unsupported_layers"].append(
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "type": layer.get("type"),
                "role": layer.get("role"),
                "reason": "Layer type is not mapped to Fusion tools yet.",
            }
        )

    def add_layer(self, layer: dict[str, Any]) -> str | None:
        self.record_layer_controls(layer)
        if not layer.get("enabled", True):
            self.report["skipped_layers"].append(
                {
                    "index": layer.get("index"),
                    "name": layer.get("name"),
                    "type": layer.get("type"),
                    "reason": "Layer is disabled or used only as a track matte/helper in the native semantic decode.",
                }
            )
            return None
        if self.is_consumed_helper_layer(layer):
            self.skip_consumed_helper_layer(layer)
            return None
        layer_type = layer.get("type")
        if layer_type == "text" or layer.get("text"):
            return self.add_text_layer(layer)
        if layer_type == "shape" or layer.get("shape"):
            return self.add_shape_layer(layer)
        if layer_type == "solid":
            return self.add_solid_layer(layer)
        if layer_type == "footage" or layer.get("footage"):
            return self.add_footage_layer(layer)
        self.add_unsupported_placeholder(layer)
        return None

    def add_merge(self, background: str, foreground: str, layer: dict[str, Any]) -> str:
        tool = self.unique_name(f"AE_Merge_{layer.get('index', '')}_{layer.get('name', 'Layer')}")
        self.report_transfer_limitations(layer)
        opacity_samples = self.layer_opacity_samples(layer)
        static_opacity = self.layer_static_opacity(layer)
        blend_input = input_value(static_opacity)
        opacity_expression = self.merge_blend_expression_by_foreground.get(foreground)
        if len(opacity_samples) > 1:
            blend = self.bezier_spline(
                f"{tool}Blend",
                opacity_samples,
                self.property_interpolation_hints(layer, "ADBE Opacity"),
            )
            blend_input = input_link(blend, "Value")
            if opacity_expression:
                blend_input = input_value_expression(static_opacity, f"{blend}.Value * ({opacity_expression})")
        elif opacity_expression:
            if math.isclose(static_opacity, 1.0, abs_tol=1e-9):
                blend_input = input_value_expression(static_opacity, opacity_expression)
            else:
                blend_input = input_value_expression(static_opacity, f"{lua_number(static_opacity)} * ({opacity_expression})")
        inputs = [
            ("Background", input_link(background)),
            ("Foreground", input_link(foreground)),
            ("Blend", blend_input),
            ("PerformDepthMerge", input_value(0)),
        ]
        body = "Merge {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        return tool

    def report_transfer_limitations(self, layer: dict[str, Any]) -> None:
        key = (layer.get("index"), layer.get("name"))
        if key in self.reported_transfer_layers:
            return
        self.reported_transfer_layers.add(key)
        transfer = layer.get("nativeTransfer") or {}
        blend_mode = layer.get("blendMode") or transfer.get("blendMode")
        if blend_mode and blend_mode not in {"normal", "none"}:
            self.report["warnings"].append(
                {
                    "index": layer.get("index"),
                    "name": layer.get("name"),
                    "warning": "AE blend mode candidate was decoded but is not applied to the Fusion Merge yet.",
                    "blend_mode": blend_mode,
                    "confidence": layer.get("blendModeConfidence") or transfer.get("blendModeConfidence"),
                    "decoded_candidates": transfer.get("decodedBlendModeCandidates"),
                }
            )

    def media_out_body(self, current: str) -> str:
        body = (
            "MediaOut {\n"
            "\t\t\tCtrlWShown = false,\n"
            + self.render_inputs(
                [
                    ("Index", input_value("0")),
                    ("Input", input_link(current)),
                ]
            )
            + ",\n\t\t\tViewInfo = OperatorInfo { Pos = { 1320, 120, }, },\n"
            "\t\t}"
        )
        return body

    def add_global_position_transform(self, current: str) -> str:
        tool = self.unique_name("AE_Position")
        inputs = [
            ("Center", input_value([0.5, 0.5])),
            ("Input", input_link(current)),
        ]
        body = "Transform {\n" + self.render_inputs(inputs) + "\n\t\t}"
        self.add_tool(tool, body)
        self.publish_instance_input(
            tool,
            "Center",
            "Position",
            "Position",
            default_x=0.5,
            default_y=0.5,
            input_id="Position",
        )
        return tool

    def render_instance_input_lines(self) -> list[str]:
        lines: list[str] = []
        for item in sorted(self.published_inputs, key=self.instance_input_sort_key):
            lines.append(f"\t\t\t\t{item['id']} = InstanceInput {{")
            lines.append(f"\t\t\t\t\tSourceOp = {lua_string(item['source_op'])},")
            lines.append(f"\t\t\t\t\tSource = {lua_string(item['source'])},")
            if item.get("name"):
                lines.append(f"\t\t\t\t\tName = {lua_string(item['name'])},")
            if item.get("control_group") is not None:
                lines.append(f"\t\t\t\t\tControlGroup = {int(item['control_group'])},")
            lines.append(f"\t\t\t\t\tPage = {lua_string(item['page'])},")
            if item.get("default_x") is not None or item.get("default_y") is not None:
                lines.append(f"\t\t\t\t\tDefaultX = {lua_number(item.get('default_x'), 0.5)},")
                lines.append(f"\t\t\t\t\tDefaultY = {lua_number(item.get('default_y'), 0.5)},")
            elif item.get("default") is not None:
                lines.append(f"\t\t\t\t\tDefault = {lua_number(item.get('default'), 0)},")
            lines.append("\t\t\t\t},")
        return lines

    def render_grouped_setting(self, current: str) -> str:
        group_name = clean_tool_name(f"AE_{self.comp.get('name') or 'Converted_Title'}")
        media_out = self.unique_name("MediaOut1")
        self.add_tool(media_out, self.media_out_body(current))
        self.report["internal_media_out"] = media_out

        def indented(text: str, prefix: str) -> str:
            return "\n".join(prefix + line if line else line for line in text.splitlines())

        lines = [
            "{",
            "\tTools = ordered() {",
            f"\t\t{group_name} = GroupOperator {{",
            "\t\t\tCtrlWZoom = false,",
            "\t\t\tInputs = ordered() {",
        ]
        lines.extend(self.render_instance_input_lines())
        lines.extend([
            "\t\t\t},",
            "\t\t\tOutputs = {",
            "\t\t\t\tMainOutput1 = InstanceOutput {",
            f"\t\t\t\t\tSourceOp = {lua_string(current)},",
            "\t\t\t\t\tSource = \"Output\",",
            "\t\t\t\t},",
            "\t\t\t},",
            "\t\t\tViewInfo = GroupInfo {",
            "\t\t\t\tPos = { 220, 49.5 },",
            "\t\t\t\tFlags = {",
            "\t\t\t\t\tExpanded = true,",
            "\t\t\t\t\tAllowPan = false,",
            "\t\t\t\t\tAutoSnap = true,",
            "\t\t\t\t\tRemoveRouters = true,",
            "\t\t\t\t},",
            "\t\t\t\tDirection = \"Horizontal\",",
            "\t\t\t\tPipeStyle = \"Direct\",",
            "\t\t\t},",
            "\t\t\tTools = ordered() {",
        ])
        for name, body in self.tools:
            lines.append(indented(f"{name} = {self.body_with_user_controls(name, body)},", "\t\t\t\t"))
        lines.extend(["\t\t\t},", "\t\t},", "\t}", "}"])
        self.report["group"] = group_name
        return "\n".join(lines) + "\n"

    def render_flat_setting(self, current: str) -> str:
        lines = ["{", "\tTools = ordered() {"]
        for name, body in self.tools:
            lines.append(f"\t\t{name} = {self.body_with_user_controls(name, body)},")
        lines.extend(["\t},", f"\tActiveTool = {lua_string(current)},", "}"])
        self.report["flat_output"] = True
        return "\n".join(lines) + "\n"

    def build(self) -> tuple[str, dict[str, Any]]:
        layers = [layer for layer in self.comp.get("layers", []) if isinstance(layer, dict)]
        self.prepare_track_matte_masks(layers)
        current = self.add_background("AE_Transparent_Background", (0, 0, 0), 0)
        layers.sort(key=lambda item: int(item.get("index") or 0), reverse=True)
        for layer in layers:
            foreground = self.add_layer(layer)
            if foreground:
                current = self.add_merge(current, foreground, layer)

        current = self.add_global_position_transform(current)
        self.report["final_tool"] = current
        if self.wrap_group:
            self.report["media_out"] = "Internal group MediaOut1"
            return self.render_grouped_setting(current), self.report
        media_out = self.unique_name("MediaOut1")
        self.add_tool(media_out, self.media_out_body(current))
        self.report["media_out"] = media_out
        return self.render_flat_setting(media_out), self.report


def select_comp(data: dict[str, Any], comp_name: str | None) -> dict[str, Any]:
    comps = data.get("comps") or []
    if not comps:
        raise ValueError("No comps found in AE JSON.")
    if comp_name:
        for comp in comps:
            if comp.get("name") == comp_name:
                return comp
        names = ", ".join(str(c.get("name")) for c in comps)
        raise ValueError(f"Comp {comp_name!r} not found. Available comps: {names}")
    return comps[0]


def convert_json_to_settings(
    json_path: Path,
    output_path: Path,
    comp_name: str | None,
    target_w: int,
    target_h: int,
    report_path: Path | None,
    *,
    wrap_group: bool = True,
    target_fps: float | None = None,
    brand_palette: str | None = None,
) -> None:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if is_native_semantic_json(data):
        data = semantic_to_legacy_export(data, comp_name, target_w, target_h)
        comp_name = None
    comp = select_comp(data, comp_name)
    builder = FusionBuilder(
        comp,
        target_w,
        target_h,
        wrap_group=wrap_group,
        target_fps=target_fps,
        brand_palette=brand_palette,
    )
    setting_text, report = builder.build()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(setting_text, encoding="utf-8")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def find_after_effects(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"After Effects executable not found: {path}")
        return path
    if DEFAULT_AE.exists():
        return DEFAULT_AE
    apps = sorted(Path("/Applications").glob("Adobe After Effects */Adobe After Effects *.app/Contents/MacOS/After Effects"))
    if apps:
        return apps[-1]
    raise FileNotFoundError("Could not locate After Effects. Pass --after-effects /path/to/After\\ Effects.")


def jsx_literal(value: Any) -> str:
    return json.dumps(value).replace("</", "<\\/")


def write_headless_wrapper(
    wrapper_path: Path,
    exporter_path: Path,
    input_path: Path,
    json_path: Path,
    status_path: Path,
    comp_name: str | None,
    target_w: int,
    target_h: int,
    bake: bool,
    bake_step: int,
) -> None:
    script = f"""
function writeStatus(payload) {{
    var f = File({jsx_literal(str(status_path))});
    f.encoding = "UTF-8";
    f.open("w");
    f.write(payload);
    f.close();
}}

try {{
    var AE_TO_JSON_HEADLESS = {{
        inputFile: {jsx_literal(str(input_path))},
        outputFile: {jsx_literal(str(json_path))},
        scope: "all",
        compName: {jsx_literal(comp_name) if comp_name else "null"},
        targetWidth: {int(target_w)},
        targetHeight: {int(target_h)},
        bake: {str(bool(bake)).lower()},
        bakeStep: {int(bake_step)},
        includeFullLayerPropertyTree: true,
        includeProjectItemManifest: true
    }};
    $.evalFile(File({jsx_literal(str(exporter_path))}));
    writeStatus('{{"ok":true}}');
}} catch (err) {{
    var message = "";
    try {{ message = err.toString(); }} catch (e) {{ message = "Unknown After Effects export error"; }}
    message = message.replace(/\\\\/g, "\\\\\\\\");
    message = message.replace(/"/g, "\\\\" + '"');
    message = message.replace(/\\r/g, "\\\\r");
    message = message.replace(/\\n/g, "\\\\n");
    writeStatus('{{"ok":false,"error":"' + message + '"}}');
    throw err;
}}
"""
    wrapper_path.write_text(script, encoding="utf-8")


def run_after_effects_export(
    input_path: Path,
    json_path: Path,
    comp_name: str | None,
    target_w: int,
    target_h: int,
    exporter_path: Path,
    after_effects: str | None,
    timeout_seconds: int,
    bake: bool,
    bake_step: int,
) -> None:
    if not exporter_path.exists():
        raise FileNotFoundError(f"AE exporter JSX not found: {exporter_path}")
    ae = find_after_effects(after_effects)
    with tempfile.TemporaryDirectory(prefix="aep_to_settings_") as tmp:
        tmpdir = Path(tmp)
        wrapper = tmpdir / "run_ae_export.jsx"
        status = tmpdir / "status.json"
        write_headless_wrapper(
            wrapper,
            exporter_path,
            input_path,
            json_path,
            status,
            comp_name,
            target_w,
            target_h,
            bake,
            bake_step,
        )
        proc = subprocess.Popen([str(ae), "-r", str(wrapper)])
        deadline = time.monotonic() + timeout_seconds
        while True:
            if status.exists():
                try:
                    status_data = json.loads(status.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    status_data = {"ok": False, "error": status.read_text(encoding="utf-8")}
                if not status_data.get("ok"):
                    raise RuntimeError(status_data.get("error") or "After Effects export failed.")
                break
            if json_path.exists() and json_path.stat().st_size > 0:
                break
            if proc.poll() is not None:
                if proc.returncode not in (0, None) and not json_path.exists():
                    raise RuntimeError(f"After Effects exited with code {proc.returncode} before writing JSON.")
                if json_path.exists():
                    break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for After Effects export after {timeout_seconds}s.")
            time.sleep(1.5)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .aep/.aepx/.aegraphic/.json into a Fusion .setting file."
    )
    parser.add_argument("input", help="Input .aep, .aepx, .aegraphic, or AE-export .json")
    parser.add_argument("-o", "--output", help="Output .setting path")
    parser.add_argument("--comp", help="Comp name to convert. Defaults to first exported comp.")
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Resolve timeline frame rate for retiming AE keyframes by seconds. Defaults to the AE comp frame rate.",
    )
    parser.add_argument("--json-out", help="Keep/write intermediate JSON to this path")
    parser.add_argument("--report", help="Write conversion report JSON to this path")
    parser.add_argument("--exporter", default=str(DEFAULT_EXPORTER), help="Headless AE exporter JSX")
    parser.add_argument("--after-effects", help="After Effects executable path")
    parser.add_argument(
        "--use-after-effects",
        action="store_true",
        help="Open .aep/.aepx with After Effects and the JSX exporter instead of native parsing.",
    )
    parser.add_argument(
        "--native-extractor",
        default=str(DEFAULT_NATIVE_EXTRACTOR),
        help="Path to ae_native_extract.py for native .aep/.aepx parsing.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-bake", action="store_true", help="Do not bake AE animated/expression values")
    parser.add_argument("--bake-step", type=int, default=1)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write a visible top-level Fusion node graph instead of wrapping tools in a GroupOperator.",
    )
    parser.add_argument(
        "--brand-palette",
        choices=sorted(BRAND_PALETTES),
        help="Add brand-color preset dropdowns to published color controls.",
    )
    parser.add_argument(
        "--12stone",
        action="store_true",
        dest="use_12stone_palette",
        help="Shortcut for --brand-palette 12stone.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".setting")
    report_path = Path(args.report).expanduser().resolve() if args.report else output_path.with_suffix(".report.json")
    brand_palette = "12stone" if args.use_12stone_palette else args.brand_palette

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        json_path = input_path
    elif suffix in (".aep", ".aepx", ".aegraphic"):
        native_input_path = input_path
        if suffix == ".aegraphic":
            if args.use_after_effects:
                raise ValueError(".aegraphic input is supported through native parsing, not --use-after-effects.")
            asset_dir = output_path.with_suffix(".aegraphic_assets")
            native_input_path = extract_aegraphic_project(input_path, asset_dir)
        if args.json_out:
            json_path = Path(args.json_out).expanduser().resolve()
        else:
            json_path = output_path.with_suffix(".ae-native.json" if not args.use_after_effects else ".ae-export.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if args.use_after_effects:
            run_after_effects_export(
                input_path=input_path,
                json_path=json_path,
                comp_name=args.comp,
                target_w=args.target_width,
                target_h=args.target_height,
                exporter_path=Path(args.exporter).expanduser().resolve(),
                after_effects=args.after_effects,
                timeout_seconds=args.timeout,
                bake=not args.no_bake,
                bake_step=args.bake_step,
            )
        else:
            semantic = native_project_to_semantic(native_input_path, Path(args.native_extractor).expanduser().resolve())
            json_path.write_text(json.dumps(semantic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise ValueError("Input must be .aep, .aepx, .aegraphic, or .json")

    convert_json_to_settings(
        json_path=json_path,
        output_path=output_path,
        comp_name=args.comp,
        target_w=args.target_width,
        target_h=args.target_height,
        report_path=report_path,
        wrap_group=not args.flat_output,
        target_fps=args.target_fps,
        brand_palette=brand_palette,
    )
    print(f"Wrote setting: {output_path}")
    print(f"Wrote report:  {report_path}")
    if suffix in (".aep", ".aepx", ".aegraphic"):
        print(f"Wrote JSON:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
