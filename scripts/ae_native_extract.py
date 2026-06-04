#!/usr/bin/env python3
"""Read After Effects .aep/.aepx project containers without launching AE.

This is a native extractor, not yet a full semantic AE layer decoder.

After Effects .aep files are big-endian RIFF/RIFX containers. .aepx files are
XML wrappers around the same chunk vocabulary, with most data stored in bdata
hex attributes. This script normalizes both formats into a JSON chunk tree and
extracts searchable strings so the binary project format can be decoded
incrementally.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


PRINTABLE_RE = re.compile(rb"[\x20-\x7e]{4,}")
EXPR_LAYER_CALL_RE = re.compile(
    r'(?:(?:thisComp)|(?:comp\("(?P<comp>[^"]+)"\)))\.layer\("(?P<layer>[^"]+)"\)'
)
EXPR_EFFECT_CHAIN_RE = re.compile(
    r'(?:(?:(?:thisComp)|(?:comp\("(?P<comp>[^"]+)"\)))\.layer\("(?P<layer>[^"]+)"\)\.)?'
    r'effect\("(?P<effect>[^"]+)"\)\("(?P<property>[^"]+)"\)(?:\.value)?'
)
EXPR_LAYER_ALIAS_RE = re.compile(
    r'\b(?P<alias>[A-Za-z_]\w*)\s*=\s*(?:(?:thisComp)|(?:comp\("(?P<comp>[^"]+)"\)))'
    r'\.layer\("(?P<layer>[^"]+)"\)(?=\s*(?:;|\r?\n|$))'
)
EXPR_ALIAS_READ_RE = re.compile(
    r'\b(?P<alias>[A-Za-z_]\w*)\.(?P<path>transform\.[A-Za-z]+|sourceRectAtTime|effect\("[^"]+"\)\("[^"]+"\))'
)
AE_TICKS_PER = 1024
AE_TICKS_PER_SECOND = 30 * AE_TICKS_PER
RAW_LIST_TYPES = {"btdk"}
RENDER_LAYER_KINDS = {"Layr"}
COMP_LAYER_KINDS = {"Layr", "DLay", "SLay", "CLay", "SecL"}
IMAGE_EXTENSIONS = {".ai", ".bmp", ".exr", ".gif", ".jpeg", ".jpg", ".png", ".psd", ".svg", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".mxf", ".webm"}
AUDIO_EXTENSIONS = {".aif", ".aiff", ".m4a", ".mp3", ".wav"}
AEGP_LAYER_FLAG_NAMES_BY_BIT = {
    0: "AEGP_LayerFlag_VIDEO_ACTIVE",
    1: "AEGP_LayerFlag_AUDIO_ACTIVE",
    2: "AEGP_LayerFlag_EFFECTS_ACTIVE",
    3: "AEGP_LayerFlag_MOTION_BLUR",
    4: "AEGP_LayerFlag_FRAME_BLENDING",
    5: "AEGP_LayerFlag_LOCKED",
    6: "AEGP_LayerFlag_SHY",
    7: "AEGP_LayerFlag_COLLAPSE",
    8: "AEGP_LayerFlag_AUTO_ORIENT_ROTATION",
    9: "AEGP_LayerFlag_ADJUSTMENT_LAYER",
    10: "AEGP_LayerFlag_TIME_REMAPPING",
    11: "AEGP_LayerFlag_LAYER_IS_3D",
    12: "AEGP_LayerFlag_LOOK_AT_CAMERA",
    13: "AEGP_LayerFlag_LOOK_AT_POI",
    14: "AEGP_LayerFlag_SOLO",
    15: "AEGP_LayerFlag_MARKERS_LOCKED",
    16: "AEGP_LayerFlag_NULL_LAYER",
    17: "AEGP_LayerFlag_HIDE_LOCKED_MASKS",
    18: "AEGP_LayerFlag_GUIDE_LAYER",
    19: "AEGP_LayerFlag_ENVIRONMENT_LAYER",
    20: "AEGP_LayerFlag_ADVANCED_FRAME_BLENDING",
    21: "AEGP_LayerFlag_SUBLAYERS_RENDER_SEPARATELY",
}
AEGP_LAYER_FLAG_FRIENDLY_KEYS_BY_BIT = {
    0: "videoActive",
    1: "audioActive",
    2: "effectsActive",
    3: "motionBlur",
    4: "frameBlending",
    5: "locked",
    6: "shy",
    7: "collapseTransformationsOrContinuouslyRasterize",
    8: "autoOrientRotation",
    9: "adjustmentLayer",
    10: "timeRemapping",
    11: "layerIs3D",
    12: "lookAtCamera",
    13: "lookAtPointOfInterest",
    14: "solo",
    15: "markersLocked",
    16: "nullLayer",
    17: "hideLockedMasks",
    18: "guideLayer",
    19: "environmentLayer",
    20: "advancedFrameBlending",
    21: "sublayersRenderSeparately",
}
COMP_LAYER_KIND_LABELS = {
    "Layr": "renderLayer",
    "DLay": "defaultCameraOrViewLayer",
    "SLay": "standardViewLayer",
    "CLay": "customViewOrCameraLayer",
    "SecL": "markerOrSectionLayer",
}
KNOWN_TEXT_PARAGRAPH_TOKENS = {
    0: "left",
    1: "right",
    2: "center",
    3: "justified",
}
AE_TRACK_MATTE_CANDIDATE_BY_CODE = {
    0: "none",
    1: "alpha",
    2: "alphaInverted",
    3: "luma",
    4: "lumaInverted",
}
AE_BLEND_MODE_CANDIDATE_BY_CODE = {
    0: "none",
    1: "normal",
    2: "dissolve",
    3: "dancingDissolve",
    4: "darken",
    5: "multiply",
    6: "colorBurn",
    7: "linearBurn",
    8: "lighten",
    9: "screen",
    10: "colorDodge",
    11: "add",
    12: "overlay",
    13: "softLight",
    14: "hardLight",
    15: "linearLight",
    16: "vividLight",
    17: "pinLight",
    18: "hardMix",
    19: "difference",
    20: "classicDifference",
    21: "exclusion",
    22: "hue",
    23: "saturation",
    24: "color",
    25: "luminosity",
    26: "stencilAlpha",
    27: "stencilLuma",
    28: "silhouetteAlpha",
    29: "silhouetteLuma",
    30: "alphaAdd",
    31: "luminescentPremul",
}
TRACK_MATTE_FLAG_KEYS = {"fmat", "fmtm", "ftmt", "tmtt"}
BLEND_MODE_FLAG_KEYS = {"fbmd", "fbmo", "fblm", "fmod", "fmde", "bmod"}
NOISY_TRANSFER_FLAG_KEYS = {"fvdv", "ftts", "fiop", "foac", "fots", "fott", "fovc", "fiac", "fits", "fitt", "fivc", "fipc"}


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def node_kind(node: dict[str, Any]) -> str:
    if node.get("id") == "LIST" and node.get("listType"):
        return str(node["listType"])
    return str(node.get("id"))


def bdata_bytes(node: dict[str, Any]) -> bytes:
    raw = node.get("bdata")
    if not isinstance(raw, str):
        return b""
    try:
        return bytes.fromhex(raw)
    except ValueError:
        return b""


def decode_bdata_text(node: dict[str, Any]) -> str | None:
    data = bdata_bytes(node)
    if not data:
        return None
    text = data.rstrip(b"\x00").decode("utf-8", "replace")
    return text or None


def direct_text(node: dict[str, Any]) -> str | None:
    if "string" in node:
        return str(node["string"])
    strings = node.get("strings") or []
    if strings:
        for value in strings:
            if value != "Utf8":
                return str(value)
        return str(strings[0])
    for child in node.get("children", []) or []:
        text = direct_text(child)
        if text is not None:
            return text
    return None


def direct_child_strings(node: dict[str, Any]) -> list[str]:
    strings = []
    for child in node.get("children", []) or []:
        kind = node_kind(child)
        if kind == "string" and "string" in child:
            strings.append(str(child["string"]))
        elif kind == "Utf8":
            strings.extend(str(value) for value in child.get("strings", []) or [])
        else:
            strings.extend(str(value) for value in child.get("strings", []) or [])
    return [value for value in strings if value]


def first_child(node: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for child in node.get("children", []) or []:
        if node_kind(child) == kind:
            return child
    return None


def decode_ascii_strings(data: bytes) -> list[str]:
    out = []
    for match in PRINTABLE_RE.finditer(data):
        try:
            text = match.group(0).decode("utf-8", "replace").strip("\x00")
        except UnicodeDecodeError:
            continue
        if text:
            out.append(text)
    return out


def decode_utf16be_strings(data: bytes) -> list[str]:
    out = []
    # AE/RIFX is big-endian. Scan even-aligned UTF-16BE printable runs.
    current = bytearray()
    for i in range(0, len(data) - 1, 2):
        code = int.from_bytes(data[i : i + 2], "big")
        if code in (9, 10, 13) or 32 <= code <= 0x7E:
            current.extend(data[i : i + 2])
        else:
            if len(current) >= 8:
                text = current.decode("utf-16-be", "replace").strip()
                if text:
                    out.append(text)
            current = bytearray()
    if len(current) >= 8:
        text = current.decode("utf-16-be", "replace").strip()
        if text:
            out.append(text)
    return out


def extract_strings(data: bytes) -> list[str]:
    seen = set()
    out = []
    for text in decode_ascii_strings(data) + decode_utf16be_strings(data):
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def read_u32_be(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def read_i32_be(data: bytes, offset: int) -> int:
    return struct.unpack_from(">i", data, offset)[0]


def read_u32_values_be(data: bytes) -> list[int]:
    return [struct.unpack_from(">I", data, offset)[0] for offset in range(0, len(data) - 3, 4)]


def read_u16_values_be(data: bytes) -> list[int]:
    return [struct.unpack_from(">H", data, offset)[0] for offset in range(0, len(data) - 1, 2)]


def read_u32_values_le(data: bytes) -> list[int]:
    return [struct.unpack_from("<I", data, offset)[0] for offset in range(0, len(data) - 3, 4)]


def read_i32_values_be(data: bytes) -> list[int]:
    return [struct.unpack_from(">i", data, offset)[0] for offset in range(0, len(data) - 3, 4)]


def read_doubles_be(data: bytes) -> list[float]:
    if not data or len(data) % 8 != 0:
        return []
    return [struct.unpack_from(">d", data, offset)[0] for offset in range(0, len(data), 8)]


def fourcc(data: bytes) -> str:
    return data.decode("latin-1", "replace")


def parse_rifx_children(data: bytes, start: int, end: int, depth: int) -> list[dict[str, Any]]:
    children = []
    offset = start
    while offset + 8 <= end:
        chunk_id = fourcc(data[offset : offset + 4])
        size = read_u32_be(data, offset + 4)
        payload_start = offset + 8
        payload_end = payload_start + size
        if payload_end > end:
            children.append(
                {
                    "id": chunk_id,
                    "offset": offset,
                    "size": size,
                    "error": f"Chunk overruns parent: payload_end={payload_end}, parent_end={end}",
                }
            )
            break

        node: dict[str, Any] = {
            "id": chunk_id,
            "offset": offset,
            "size": size,
        }
        payload = data[payload_start:payload_end]
        if chunk_id == "LIST" and size >= 4:
            list_type = fourcc(payload[:4])
            node["listType"] = list_type
            if list_type in RAW_LIST_TYPES:
                raw_payload = payload[4:]
                strings = extract_strings(raw_payload)
                if strings:
                    node["strings"] = strings
                node["bdata"] = raw_payload.hex()
            else:
                node["children"] = parse_rifx_children(data, payload_start + 4, payload_end, depth + 1)
        else:
            strings = extract_strings(payload)
            if chunk_id == "Utf8":
                utf8_text = payload.rstrip(b"\x00").decode("utf-8", "replace")
                if utf8_text and utf8_text not in strings:
                    strings.insert(0, utf8_text)
            if strings:
                node["strings"] = strings
            node["bdata"] = payload.hex()
        children.append(node)

        offset = payload_end + (size % 2)
    return children


def parse_aep(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFX":
        raise ValueError(f"{path} is not a big-endian RIFX .aep file.")
    size = read_u32_be(data, 4)
    form_type = fourcc(data[8:12])
    root = {
        "format": "aep-rifx",
        "path": str(path),
        "size": len(data),
        "riffSize": size,
        "formType": form_type,
        "children": parse_rifx_children(data, 12, min(len(data), 8 + size), 0),
    }
    return root


def xml_node_to_dict(elem: ET.Element, include_bdata: bool) -> dict[str, Any]:
    tag = strip_ns(elem.tag)
    node: dict[str, Any] = {"id": tag}

    attrs = dict(elem.attrib)
    bdata_hex = attrs.pop("bdata", None)
    if attrs:
        node["attrs"] = attrs

    if bdata_hex is not None:
        node["bdataSize"] = len(bdata_hex) // 2
        if include_bdata:
            node["bdata"] = bdata_hex
        try:
            strings = extract_strings(bytes.fromhex(bdata_hex))
            if strings:
                node["strings"] = strings
        except ValueError:
            node["bdataError"] = "Invalid bdata hex."

    text = (elem.text or "").strip()
    if text and tag != "string":
        node["text"] = text
    if tag == "string":
        node["string"] = elem.text or ""

    children = [xml_node_to_dict(child, include_bdata) for child in list(elem)]
    if children:
        node["children"] = children
    return node


def parse_aepx(path: Path, include_bdata: bool) -> dict[str, Any]:
    tree = ET.parse(path)
    root_elem = tree.getroot()
    root = xml_node_to_dict(root_elem, include_bdata)
    root["format"] = "aepx-xml"
    root["path"] = str(path)
    root["size"] = path.stat().st_size
    return root


def walk_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        out.extend(walk_nodes(child))
    return out


def node_strings(node: dict[str, Any]) -> list[str]:
    out = []
    if "string" in node:
        out.append(str(node["string"]))
    out.extend(str(value) for value in node.get("strings", []) or [])
    for child in node.get("children", []) or []:
        out.extend(node_strings(child))
    return out


def is_layer_node(node: dict[str, Any]) -> bool:
    return node_kind(node) in RENDER_LAYER_KINDS


def is_comp_layer_node(node: dict[str, Any]) -> bool:
    return node_kind(node) in COMP_LAYER_KINDS


def layer_kind_label(kind: str) -> str:
    return COMP_LAYER_KIND_LABELS.get(kind, "unknownCompLayer")


def semantic_layers(root: dict[str, Any]) -> list[dict[str, Any]]:
    layers = []
    for tree_index, node in enumerate(walk_nodes(root), start=1):
        if not is_layer_node(node):
            continue
        strings = node_strings(node)
        visible = [value for value in strings if value and value not in ("Utf8", "-_0_/-")]
        layer_name = visible[0] if visible else None
        match_names = [
            value
            for value in visible
            if value.startswith("ADBE ") or value.startswith("Pseudo/")
        ]
        expressions = [
            value
            for value in visible
            if "thisComp." in value or "effect(" in value or "sourceRectAtTime" in value
        ]
        layers.append(
            {
                "ordinal": len(layers) + 1,
                "treeIndex": tree_index,
                "name": layer_name,
                "stringsSample": visible[:25],
                "matchNamesSample": match_names[:50],
                "expressionsSample": expressions[:10],
            }
        )
    return layers


def file_reference_metadata(path: str) -> dict[str, Any]:
    basename = re.split(r"[\\/]", path)[-1]
    stem = re.sub(r"\.[^.]+$", "", basename)
    extension = f".{basename.rsplit('.', 1)[1].lower()}" if "." in basename else ""
    if extension in IMAGE_EXTENSIONS:
        source_type = "image"
    elif extension in VIDEO_EXTENSIONS:
        source_type = "video"
    elif extension in AUDIO_EXTENSIONS:
        source_type = "audio"
    else:
        source_type = "unknown"
    out: dict[str, Any] = {
        "basename": basename,
        "stem": stem,
        "sourceType": source_type,
        "isStillCandidate": source_type == "image",
    }
    if extension:
        out["extension"] = extension
    return out


def enriched_file_reference(ref: dict[str, Any]) -> dict[str, Any]:
    out = dict(ref)
    path = str(out.get("path") or "")
    if path:
        out.update(file_reference_metadata(path))
    return out


def semantic_file_references(root: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    seen = set()

    def add_ref(path: Any, platform: Any = None, target_is_folder: Any = None) -> None:
        if not path:
            return
        normalized = str(path)
        if normalized in seen:
            return
        seen.add(normalized)
        refs.append(
            enriched_file_reference(
                {
                    "path": normalized,
                    "platform": platform,
                    "targetIsFolder": target_is_folder,
                }
            )
        )

    for node in walk_nodes(root):
        attrs = node.get("attrs") or {}
        if node.get("id") == "fileReference" or "fullpath" in attrs:
            add_ref(attrs.get("fullpath"), attrs.get("platform"), attrs.get("target_is_folder"))
        for value in node.get("strings") or []:
            text = str(value).strip()
            if not text:
                continue
            if text.startswith("{") and "fullpath" in text:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    add_ref(payload.get("fullpath"), payload.get("platform"), payload.get("target_is_folder"))
            elif re.search(r"\.(png|jpe?g|tiff?|mov|mp4|mxf|exr|psd|ai)$", text, re.IGNORECASE) and (
                "/" in text or "\\" in text
            ):
                add_ref(text)
    return refs


def chunk_data_summary(node: dict[str, Any] | None, max_values: int = 24) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if not data:
        return None
    out: dict[str, Any] = {
        "byteLength": len(data),
        "hexHead": data[: min(32, len(data))].hex(),
    }
    if len(data) >= 2:
        out["u16BEHead"] = read_u16_values_be(data)[:max_values]
    if len(data) >= 4:
        out["u32BEHead"] = read_u32_values_be(data)[:max_values]
        out["i32BEHead"] = read_i32_values_be(data)[:max_values]
    if len(data) % 8 == 0:
        doubles = read_doubles_be(data)
        if doubles:
            out["doubleBEHead"] = doubles[:max_values]
    return out


def chunk_scalar(node: dict[str, Any] | None) -> int | str | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) == 1:
        return data[0]
    if len(data) == 2:
        return struct.unpack_from(">H", data, 0)[0]
    if len(data) == 4:
        return read_u32_values_be(data)[0]
    text = direct_text(node)
    if text is not None:
        return text
    return None


def direct_child_texts(node: dict[str, Any], child_kinds: set[str] | None = None) -> list[str]:
    texts = []
    for child in node.get("children", []) or []:
        kind = node_kind(child)
        if child_kinds is not None and kind not in child_kinds:
            continue
        text = direct_text(child)
        if text is not None:
            texts.append(text)
    return texts


def item_name(node: dict[str, Any]) -> str | None:
    for text in direct_child_texts(node, {"string", "Utf8"}):
        if text and text not in ("Utf8", "-_0_/-"):
            return text
    for value in node_strings(node):
        if value and value not in ("Utf8", "-_0_/-"):
            return str(value)
    return None


def decode_iide(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 4:
        return None
    little = read_u32_values_le(data)[0]
    big = read_u32_values_be(data)[0]
    return {
        "id": little,
        "byteLength": len(data),
        "rawU32BE": big,
    }


def decode_idta(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 24:
        return None
    u16 = read_u16_values_be(data)
    u32 = read_u32_values_be(data)
    item_type = u16[0] if u16 else None
    type_labels = {
        1: "folder",
        4: "composition",
        7: "footage-or-solid",
    }
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-idta",
        "byteLength": len(data),
        "itemTypeCode": item_type,
        "itemType": type_labels.get(item_type, "unknown"),
        "rawU16Head": u16[:24],
        "rawU32Head": u32[:21],
    }
    if len(u32) > 4:
        out["itemId"] = u32[4]
    if len(u32) > 5:
        out["rawFlagOrSubtype"] = u32[5]
    if item_type == 4 and len(u32) > 14 and u32[14] > 0:
        # In tested AE projects this carries the comp width. The height lives in cdta.
        out["widthHint"] = u32[14]
    return out


def decode_cdta(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 24:
        return None
    u16 = read_u16_values_be(data)
    u32 = read_u32_values_be(data)
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-cdta",
        "byteLength": len(data),
        "rawU16Head": u16[:40],
        "rawU32Head": u32[:60],
    }
    if len(u32) > 4:
        ticks_per_frame = u32[1] or AE_TICKS_PER
        ticks_per_second = u32[2] or AE_TICKS_PER_SECOND
        duration_frames = u32[4]
        out.update(
            {
                "ticksPerFrame": ticks_per_frame,
                "ticksPerSecond": ticks_per_second,
                "frameRate": ticks_per_second / ticks_per_frame if ticks_per_frame else None,
                "durationFrames": duration_frames,
            }
        )
        if ticks_per_second:
            out["durationSeconds"] = duration_frames / (ticks_per_second / ticks_per_frame)
    if len(u32) > 11:
        out["workAreaDurationFrames"] = clean_number(ticks_to_frames(u32[11]))
    if len(u16) > 71:
        out["width"] = u16[70]
        out["height"] = u16[71]
    return out


def decode_ftgi(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 16:
        return None
    u32 = read_u32_values_be(data)
    return {
        "decodeStatus": "best-effort-ftgi",
        "byteLength": len(data),
        "rawU32": u32,
        "durationOrFrameHint": u32[3] if len(u32) > 3 else None,
    }


def item_child_kind_counts(node: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(node_kind(child) for child in node.get("children", []) or []))


def file_refs_for_item(name: str | None, node: dict[str, Any], file_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not file_refs:
        return []
    item_strings = {
        value
        for value in node_strings(node)
        if value and value not in ("Utf8", "-_0_/-", "{}") and len(value) < 260
    }
    if name:
        item_strings.add(name)
    matches = []
    seen = set()
    for ref in file_refs:
        path = str(ref.get("path") or "")
        basename = re.split(r"[\\/]", path)[-1]
        stem = re.sub(r"\.[^.]+$", "", basename)
        if not path:
            continue
        is_match = False
        for value in item_strings:
            if value and (value == basename or value == stem or value in path):
                is_match = True
                break
        if is_match and path not in seen:
            seen.add(path)
            matches.append(ref)
    return matches


def source_metadata_for_refs(file_refs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not file_refs:
        return None
    type_counts = Counter(ref.get("sourceType") or "unknown" for ref in file_refs)
    extensions = sorted({ref.get("extension") for ref in file_refs if ref.get("extension")})
    metadata: dict[str, Any] = {
        "sourceTypes": dict(type_counts),
        "extensions": extensions,
        "isStillCandidate": any(ref.get("isStillCandidate") for ref in file_refs),
    }
    if len(type_counts) == 1:
        metadata["primarySourceType"] = next(iter(type_counts))
    return metadata


def direct_child(node: dict[str, Any], kinds: set[str]) -> dict[str, Any] | None:
    for child in node.get("children", []) or []:
        if node_kind(child) in kinds:
            return child
    return None


def semantic_items(root: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    file_refs = semantic_file_references(root)
    item_nodes = [node for node in walk_nodes(root) if node_kind(node) == "Item"]
    for ordinal, node in enumerate(item_nodes, start=1):
        idta = decode_idta(first_child(node, "idta"))
        iide = decode_iide(first_child(node, "iide"))
        name = item_name(node)
        child_counts = item_child_kind_counts(node)
        kind = (idta or {}).get("itemType") or "unknown"
        if "cdta" in child_counts:
            kind = "composition"
        elif "sfdt" in child_counts:
            kind = "folder"
        elif "Pin" in child_counts or "Pin " in child_counts:
            kind = "footage-or-solid"

        layer_kinds = {"Layr", "DLay", "SLay", "CLay", "SecL"}
        direct_layers = [child for child in node.get("children", []) or [] if node_kind(child) in layer_kinds]
        item: dict[str, Any] = {
            "ordinal": ordinal,
            "name": name,
            "kind": kind,
            "itemId": (idta or {}).get("itemId") or (iide or {}).get("id"),
            "iide": iide,
            "idta": idta,
            "childKindCounts": child_counts,
        }
        cdta = decode_cdta(first_child(node, "cdta"))
        if cdta:
            item["composition"] = cdta
        ftgi = decode_ftgi(first_child(node, "ftgi"))
        if ftgi:
            item["footage"] = ftgi
        if kind == "footage-or-solid":
            item["stringsSample"] = [
                value for value in node_strings(node) if value and value not in ("Utf8", "-_0_/-")
            ][:20]
            matching_file_refs = file_refs_for_item(name, node, file_refs)
            if matching_file_refs:
                item["candidateFileReferences"] = matching_file_refs
                source_metadata = source_metadata_for_refs(matching_file_refs)
                if source_metadata:
                    item["sourceMetadata"] = source_metadata
        if direct_layers:
            item["directLayerCount"] = len(direct_layers)
            item["directLayerNames"] = [item_name(layer) for layer in direct_layers]
        items.append(item)
    return items


def ticks_to_frames(ticks: int | float, ticks_per_frame: int = AE_TICKS_PER) -> float:
    return ticks / ticks_per_frame


def ticks_to_seconds(ticks: int | float, ticks_per_second: int = AE_TICKS_PER_SECOND) -> float:
    return ticks / ticks_per_second if ticks_per_second else 0.0


def clean_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def unique_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def bit_positions(value: int, width: int = 32) -> list[int]:
    return [index for index in range(width) if value & (1 << index)]


def sdk_flag_name_candidates(bit_numbers: list[int]) -> list[dict[str, Any]]:
    candidates = []
    for bit in bit_numbers:
        name = AEGP_LAYER_FLAG_NAMES_BY_BIT.get(bit)
        if not name:
            continue
        candidates.append(
            {
                "bit": bit,
                "name": name,
                "confidence": "candidate",
            }
        )
    return candidates


def friendly_layer_switch_candidates(raw_flags: int) -> dict[str, bool]:
    return {
        name: bool(raw_flags & (1 << bit))
        for bit, name in AEGP_LAYER_FLAG_FRIENDLY_KEYS_BY_BIT.items()
    }


def enabled_friendly_layer_switches(raw_flags: int) -> list[str]:
    switches = friendly_layer_switch_candidates(raw_flags)
    return [name for name, enabled in switches.items() if enabled]


def unknown_layer_flag_bits(bit_numbers: list[int]) -> list[int]:
    return [
        bit
        for bit in bit_numbers
        if bit not in AEGP_LAYER_FLAG_NAMES_BY_BIT and bit not in AEGP_LAYER_FLAG_FRIENDLY_KEYS_BY_BIT
    ]


def stream_flag_summary(stream: dict[str, Any]) -> dict[str, Any]:
    chunks = stream.get("chunks") or []
    flags = dict(stream.get("flags") or {})
    for chunk in chunks:
        kind = chunk.get("kind")
        if isinstance(kind, str) and kind.startswith("f"):
            scalar = chunk.get("scalar")
            if scalar is not None:
                flags[kind] = scalar
    return {
        "index": stream.get("index"),
        "streamId": stream.get("streamId"),
        "typeName": stream.get("typeName"),
        "flags": flags,
    }


def transfer_stream_hints(streams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints = []
    for stream in streams:
        flags = stream.get("flags") or {}
        non_zero_flags = {
            key: value
            for key, value in flags.items()
            if value not in (0, None)
        }
        type_name = str(stream.get("typeName") or "")
        semantic_signals = [
            token
            for token in ("matte", "track", "blend", "transfer", "mode", "opacity")
            if token in type_name.lower()
        ]
        if not non_zero_flags and not semantic_signals:
            continue
        hint: dict[str, Any] = {
            "index": stream.get("index"),
            "streamId": stream.get("streamId"),
            "typeName": stream.get("typeName"),
            "confidence": "raw-post-layer-stream-evidence",
        }
        if non_zero_flags:
            hint["nonZeroFlags"] = non_zero_flags
            hint["nonZeroFlagKinds"] = sorted(non_zero_flags)
        if semantic_signals:
            hint["semanticNameSignals"] = semantic_signals
        candidates = decoded_transfer_stream_candidates(stream)
        if candidates:
            hint["decodedTransferCandidates"] = candidates
        hints.append(hint)
    return hints


def decoded_transfer_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def decoded_transfer_stream_candidates(stream: dict[str, Any]) -> list[dict[str, Any]]:
    flags = stream.get("flags") or {}
    type_name = str(stream.get("typeName") or "")
    type_lower = type_name.lower()
    candidates = []
    decodable_flag_keys = [
        key
        for key, value in flags.items()
        if decoded_transfer_code(value) is not None and str(key).lower() not in NOISY_TRANSFER_FLAG_KEYS
    ]
    use_type_name_fallback = len(decodable_flag_keys) == 1
    for key, value in flags.items():
        code = decoded_transfer_code(value)
        if code is None:
            continue
        lower_key = str(key).lower()
        is_track_matte_name = "track matte" in type_lower or ("matte" in type_lower and "mode" in type_lower)
        if lower_key in TRACK_MATTE_FLAG_KEYS or (
            is_track_matte_name and use_type_name_fallback and lower_key not in NOISY_TRANSFER_FLAG_KEYS
        ):
            candidates.append(
                {
                    "kind": "trackMatte",
                    "flag": key,
                    "code": code,
                    "name": AE_TRACK_MATTE_CANDIDATE_BY_CODE.get(code, f"unknown-{code}"),
                    "confidence": "raw-transfer-stream-code-candidate",
                }
            )
            continue
        is_blend_mode_name = "blend mode" in type_lower or "transfer mode" in type_lower
        if lower_key in BLEND_MODE_FLAG_KEYS or (
            is_blend_mode_name and use_type_name_fallback and lower_key not in NOISY_TRANSFER_FLAG_KEYS
        ):
            candidates.append(
                {
                    "kind": "blendMode",
                    "flag": key,
                    "code": code,
                    "name": AE_BLEND_MODE_CANDIDATE_BY_CODE.get(code, f"unknown-{code}"),
                    "confidence": "raw-transfer-stream-code-candidate",
                }
            )
    return candidates


def transfer_stream_hint_summary(stream_hints: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not stream_hints:
        return None
    flag_kinds = sorted(
        {
            flag
            for stream in stream_hints
            for flag in stream.get("nonZeroFlagKinds") or []
        }
    )
    semantic_signals = sorted(
        {
            signal
            for stream in stream_hints
            for signal in stream.get("semanticNameSignals") or []
        }
    )
    out: dict[str, Any] = {
        "nonDefaultStreamCount": len(stream_hints),
    }
    if flag_kinds:
        out["nonZeroFlagKinds"] = flag_kinds
    if semantic_signals:
        out["semanticNameSignals"] = semantic_signals
    decoded = [
        candidate
        for stream in stream_hints
        for candidate in stream.get("decodedTransferCandidates") or []
    ]
    if decoded:
        by_kind = Counter(candidate.get("kind") for candidate in decoded)
        out["decodedTransferCandidateCounts"] = dict(sorted(by_kind.items()))
    return out


def layer_switch_summary(
    timing: dict[str, Any] | None,
    node_kind_value: str,
    visual: dict[str, Any] | None,
    effects: list[dict[str, Any]],
    source_item: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    hints: dict[str, Any] = {
        "decodeStatus": "best-effort-layer-switches",
        "nodeKind": node_kind_value,
        "layerKind": layer_kind_label(node_kind_value),
        "visualKind": (visual or {}).get("kind"),
        "hasSourceItem": bool(source_item),
        "effectKinds": [effect.get("kind") for effect in effects if effect.get("kind")],
    }
    raw_flags = (timing or {}).get("rawLayerFlags")
    if isinstance(raw_flags, int):
        hints["rawLayerFlags"] = raw_flags
        hints["rawLayerFlagsHex"] = f"0x{raw_flags:08X}"
        bit_numbers = bit_positions(raw_flags)
        hints["setBitPositions"] = bit_numbers
        hints["friendlyLayerSwitchCandidates"] = friendly_layer_switch_candidates(raw_flags)
        hints["enabledFriendlyLayerSwitchCandidates"] = enabled_friendly_layer_switches(raw_flags)
        unknown_bits = unknown_layer_flag_bits(bit_numbers)
        if unknown_bits:
            hints["unknownSetBitPositions"] = unknown_bits
        candidates = sdk_flag_name_candidates(bit_numbers)
        if candidates:
            hints["aegpLayerFlagNameCandidates"] = candidates
            hints["aegpFlagMappingNote"] = (
                "Names follow the Adobe SDK AEGP_LayerFlags order; AEP file storage bit positions are retained as raw truth "
                "because direct numeric equivalence is not yet fully proven for every flag."
            )
        hints["strongSwitchHints"] = {
            "videoActiveCandidate": bool(raw_flags & 0x1),
            "effectsActiveCandidate": bool(raw_flags & 0x4),
            "hiddenOrGuideCandidate": not bool(raw_flags & 0x1),
        }
        hints["lowWord"] = raw_flags & 0xFFFF
        hints["highWord"] = (raw_flags >> 16) & 0xFFFF
    post_streams = ((context or {}).get("postLayerStreams") or {}).get("streams") or []
    if post_streams:
        stream_summaries = [stream_flag_summary(stream) for stream in post_streams]
        hints["postLayerStreams"] = stream_summaries
        stream_hints = transfer_stream_hints(stream_summaries)
        if stream_hints:
            hints["postLayerStreamHints"] = stream_hints
            summary = transfer_stream_hint_summary(stream_hints)
            if summary:
                hints["postLayerStreamHintSummary"] = summary
    extras = ((context or {}).get("postLayerStreams") or {}).get("extras") or []
    if extras:
        hints["sidecarChunkKinds"] = [item.get("kind") for item in extras if item.get("kind")]
    return hints if len(hints) > 5 or raw_flags is not None or post_streams or extras else None


def expression_uses(expression: str) -> dict[str, bool]:
    uses = {}
    for token in [
        "time",
        "value",
        "sourceRectAtTime",
        "thisComp",
        "thisLayer",
        "parent",
        "toComp",
        "transform",
        "linear",
        "ease",
        "easeIn",
        "easeOut",
        "clamp",
        "degreesToRadians",
        "radiansToDegrees",
        "Math",
    ]:
        if re.search(rf"\b{re.escape(token)}\b", expression):
            uses[token] = True
    return uses


def expression_patterns(expression: str) -> list[str]:
    patterns = []
    if "sourceRectAtTime" in expression:
        patterns.append("textBoundsRead")
        if re.search(r"\b(transform\.position|position|newX|newY)\b", expression, re.IGNORECASE):
            patterns.append("textBoundsDrivenLayout")
    if "Dropdown Menu Control" in expression or re.search(r'effect\("[^"]+"\)\("Menu"\)', expression):
        patterns.append("dropdownControlRead")
    if re.search(r'effect\("[^"]+"\)\("Color"\)', expression):
        patterns.append("controlLinkedColor")
    if re.search(r'effect\("[^"]+"\)\("Slider"\)', expression):
        patterns.append("controlLinkedSlider")
    if re.search(r'effect\("[^"]+"\)\("Checkbox"\)', expression):
        patterns.append("controlLinkedCheckbox")
    if "effect(" in expression and "?" in expression:
        patterns.append("conditionalExpression")
    if re.search(r"\[\s*newX\s*,\s*value\s*\[\s*1\s*\]\s*\]", expression):
        patterns.append("xPositionOverride")
    if re.search(r"\[\s*value\s*\[\s*0\s*\]\s*,\s*newY\s*\]", expression):
        patterns.append("yPositionOverride")
    if re.search(r"\btime\b", expression):
        patterns.append("timeDriven")
    if re.search(r"\blinear\s*\(", expression):
        patterns.append("linearFunction")
    if re.search(r"\bease(?:In|Out)?\s*\(", expression):
        patterns.append("easeFunction")
    if re.search(r"\bclamp\s*\(", expression):
        patterns.append("clampFunction")
    if re.search(r"\b(?:degreesToRadians|radiansToDegrees)\s*\(", expression):
        patterns.append("angleConversionFunction")
    if re.search(r"\bMath\.", expression):
        patterns.append("mathFunction")
    if re.search(r"\bthisComp\.(?:width|height|duration|frameDuration|frameRate)\b", expression):
        patterns.append("compPropertyReference")
    if re.search(r"\bthisLayer\.|(?<!\.)\bsourceRectAtTime\s*\(|(?<!\.)\btransform\.position", expression):
        patterns.append("currentLayerReference")
    return list(dict.fromkeys(patterns))


def parse_expression(expression: str) -> dict[str, Any]:
    aliases = {}
    layer_refs = []
    effect_refs = []
    property_reads = []

    for match in EXPR_LAYER_ALIAS_RE.finditer(expression):
        alias = match.group("alias")
        layer = match.group("layer")
        comp = match.group("comp")
        aliases[alias] = {"layer": layer}
        if comp:
            aliases[alias]["comp"] = comp

    for match in EXPR_LAYER_CALL_RE.finditer(expression):
        entry = {"name": match.group("layer")}
        if match.group("comp"):
            entry["comp"] = match.group("comp")
        layer_refs.append(entry)

    for match in EXPR_EFFECT_CHAIN_RE.finditer(expression):
        entry = {
            "effect": match.group("effect"),
            "property": match.group("property"),
        }
        if match.group("layer"):
            entry["scope"] = "layer"
            entry["layer"] = match.group("layer")
            if match.group("comp"):
                entry["comp"] = match.group("comp")
        else:
            entry["scope"] = "local"
        effect_refs.append(entry)

    for match in EXPR_ALIAS_READ_RE.finditer(expression):
        alias = match.group("alias")
        alias_target = aliases.get(alias)
        if not alias_target:
            continue
        path = match.group("path")
        entry = {
            "layer": alias_target["layer"],
            "path": path,
        }
        if alias_target.get("comp"):
            entry["comp"] = alias_target["comp"]
        property_reads.append(entry)

    if re.search(r"(?<!\.)\bsourceRectAtTime\s*\(", expression):
        property_reads.append({"scope": "local", "path": "sourceRectAtTime"})

    out: dict[str, Any] = {}
    if layer_refs:
        out["layers"] = unique_dicts(layer_refs)
    if effect_refs:
        out["effects"] = unique_dicts(effect_refs)
    if aliases:
        out["aliases"] = aliases
    if property_reads:
        out["propertyReads"] = unique_dicts(property_reads)
    uses = expression_uses(expression)
    if uses:
        out["uses"] = uses
    patterns = expression_patterns(expression)
    if patterns:
        out["patterns"] = patterns
    return out


def merge_expression_analysis(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("layers", "effects", "propertyReads"):
        values = []
        for analysis in analyses:
            values.extend(analysis.get(key) or [])
        if values:
            merged[key] = unique_dicts(values)

    patterns = []
    for analysis in analyses:
        patterns.extend(analysis.get("patterns") or [])
    if patterns:
        merged["patterns"] = list(dict.fromkeys(patterns))

    aliases = {}
    for analysis in analyses:
        for name, target in (analysis.get("aliases") or {}).items():
            aliases[name] = target
    if aliases:
        merged["aliases"] = aliases

    uses = {}
    for analysis in analyses:
        uses.update(analysis.get("uses") or {})
    if uses:
        merged["uses"] = uses
    return merged


def analyze_expressions(expressions: list[str]) -> dict[str, Any] | None:
    unique_expressions = list(dict.fromkeys(expression.strip() for expression in expressions if expression.strip()))
    unique_expressions = [
        expression
        for expression in unique_expressions
        if not any(expression != other and expression in other for other in unique_expressions)
    ]
    analyses = [parse_expression(expression) for expression in unique_expressions]
    analyses = [analysis for analysis in analyses if analysis]
    if not analyses:
        return None
    merged = merge_expression_analysis(analyses)
    merged["expressionCount"] = len(unique_expressions)
    return merged


def decode_ldta(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 36:
        return None

    u32 = read_u32_values_be(data)
    i32 = read_i32_values_be(data)
    if len(i32) < 9:
        return None

    ticks_per_second = u32[4] if len(u32) > 4 and u32[4] else AE_TICKS_PER_SECOND
    start_ticks = i32[3]
    local_in_ticks = i32[5]
    local_out_ticks = i32[7]
    comp_in_ticks = start_ticks + local_in_ticks
    comp_out_ticks = start_ticks + local_out_ticks

    out = {
        "decodeStatus": "best-effort-ldta",
        "byteLength": len(data),
        "id": u32[0] if u32 else None,
        "rawU32Head": u32[:20],
        "ticksPerFrame": AE_TICKS_PER,
        "ticksPerSecond": ticks_per_second,
        "frameRate": ticks_per_second / AE_TICKS_PER,
        "startTicks": start_ticks,
        "localInTicks": local_in_ticks,
        "localOutTicks": local_out_ticks,
        "compInTicks": comp_in_ticks,
        "compOutTicks": comp_out_ticks,
        "startFrame": clean_number(ticks_to_frames(start_ticks)),
        "localInFrame": clean_number(ticks_to_frames(local_in_ticks)),
        "localOutFrame": clean_number(ticks_to_frames(local_out_ticks)),
        "inFrame": clean_number(ticks_to_frames(comp_in_ticks)),
        "outFrame": clean_number(ticks_to_frames(comp_out_ticks)),
        "startSeconds": ticks_to_seconds(start_ticks, ticks_per_second),
        "inSeconds": ticks_to_seconds(comp_in_ticks, ticks_per_second),
        "outSeconds": ticks_to_seconds(comp_out_ticks, ticks_per_second),
    }
    if len(u32) > 9:
        # These fields are stable across the tested binary .aep and XML .aepx
        # pair. sourceItemId points to the Item id for footage/solid-backed
        # layers; text/shape layers generally leave it at 0.
        out["rawLayerFlags"] = u32[9]
    if len(u32) > 10 and u32[10]:
        out["sourceItemId"] = u32[10]
    return out


def signed_u16(value: int) -> int:
    return value - 0x10000 if value >= 0x8000 else value


def decode_tdb4(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if not data:
        return None
    u16 = [struct.unpack_from(">H", data, offset)[0] for offset in range(0, len(data) - 1, 2)]
    u32 = [struct.unpack_from(">I", data, offset)[0] for offset in range(0, len(data) - 3, 4)]
    out: dict[str, Any] = {
        "byteLength": len(data),
        "rawU16Head": u16[:24],
        "rawU32Head": u32[:16],
    }
    if len(u16) >= 8:
        out.update(
            {
                "magic": hex(u16[0]),
                # This consistently matches scalar/vector/color component count
                # in the local AE projects.
                "componentCount": u16[1],
                "storageCode": u16[2],
                "dimensionCode": u16[3],
                "flagsA": signed_u16(u16[4]),
                "flagsB": signed_u16(u16[5]),
            }
        )
    return out


def decoded_doubles_for_child(node: dict[str, Any], child_kind: str) -> list[float]:
    child = first_child(node, child_kind)
    return read_doubles_be(bdata_bytes(child)) if child else []


def normalized_color(values: list[float]) -> dict[str, Any] | None:
    if len(values) >= 4:
        alpha, red, green, blue = values[:4]
        divisor = 255.0 if max(abs(red), abs(green), abs(blue), abs(alpha)) > 1.0 else 1.0
        return {
            "rgba": [red / divisor, green / divisor, blue / divisor, alpha / divisor],
            "sourceOrder": "alpha, red, green, blue",
        }
    if len(values) >= 3:
        red, green, blue = values[:3]
        divisor = 255.0 if max(abs(red), abs(green), abs(blue)) > 1.0 else 1.0
        return {"rgb": [red / divisor, green / divisor, blue / divisor]}
    return None


def decode_pdf_literal(raw: bytes) -> str:
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace")
    return raw.decode("utf-8", "replace")


def pdf_literal_strings(data: bytes) -> list[str]:
    strings = []
    offset = 0
    while offset < len(data):
        if data[offset] != 0x28:
            offset += 1
            continue

        offset += 1
        depth = 1
        raw = bytearray()
        while offset < len(data) and depth > 0:
            byte = data[offset]
            if byte == 0x5C and offset + 1 < len(data):
                offset += 1
                raw.append(data[offset])
                offset += 1
                continue
            if byte == 0x28:
                depth += 1
                raw.append(byte)
                offset += 1
                continue
            if byte == 0x29:
                depth -= 1
                if depth == 0:
                    offset += 1
                    break
                raw.append(byte)
                offset += 1
                continue
            raw.append(byte)
            offset += 1

        strings.append(decode_pdf_literal(bytes(raw)))
    return strings


def looks_like_font_name(value: str) -> bool:
    if len(value) < 3 or len(value) > 80:
        return False
    if value.startswith("Version ") or value in {"DVA", "Hard", "Soft", "Normal RGB"}:
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    return bool(
        "-" in value
        or "Font" in value
        or "Bold" in value
        or "Regular" in value
        or "Roman" in value
        or "Myriad" in value
        or "Bebas" in value
        or "Brandon" in value
    )


PDF_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)")


def parse_pdf_number_list(value: str) -> list[float]:
    return [float(match.group(0)) for match in PDF_NUMBER_RE.finditer(value)]


def source_literal_window(data: bytes, source_text: str | None, width: int = 3200) -> str:
    if source_text:
        literal = b"(\xfe\xff" + (source_text + "\r").encode("utf-16-be") + b")"
        index = data.find(literal)
        if index >= 0:
            return data[index : index + width].decode("latin-1", "replace")
    return data.decode("latin-1", "replace")


def text_paint_slots(payload: str) -> list[dict[str, Any]]:
    paints = []
    seen = set()
    pattern = re.compile(
        r"/(\d+)\s+<<\s+/99\s+/SimplePaint\s+/0\s+<<\s+/0\s+\d+\s+/1\s+\[([^\]]+)\]"
    )
    for match in pattern.finditer(payload):
        values = parse_pdf_number_list(match.group(2))
        if len(values) < 4:
            continue
        key = (match.group(1), tuple(round(value, 6) for value in values[:4]))
        if key in seen:
            continue
        seen.add(key)
        color = normalized_color(values[:4])
        paints.append(
            {
                "slot": int(match.group(1)),
                "rawValues": values[:4],
                "color": color,
            }
        )
    return paints


def text_size_candidates(payload: str) -> list[float]:
    sizes = []
    for match in re.finditer(r"/(?:14|31)\s+([-+]?(?:\d+\.\d+|\.\d+|\d+))", payload):
        value = float(match.group(1))
        if 1 <= value <= 500:
            sizes.append(value)
    return sizes


def text_bounds_candidates(payload: str) -> list[dict[str, Any]]:
    boxes = []
    for match in re.finditer(r"/1\s+\[\s+([^\]]+)\]", payload):
        values = parse_pdf_number_list(match.group(1))
        if len(values) != 4:
            continue
        left, top, right, bottom = values
        if abs(left) > 1e-6 or top >= 0 or right <= left or bottom <= top:
            continue
        boxes.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
            }
        )
    return boxes


def text_glyph_runs(payload: str) -> list[dict[str, Any]]:
    runs = []
    pattern = re.compile(
        r"/21\s+<<\s+/0\s+\[([^\]]*)\]\s+/1\s+\[([^\]]*)\]\s+/2\s+\[([^\]]*)\]"
    )
    for match in pattern.finditer(payload):
        runs.append(
            {
                "codeOrWidthValues": parse_pdf_number_list(match.group(1)),
                "positions": parse_pdf_number_list(match.group(2)),
                "baselines": parse_pdf_number_list(match.group(3)),
            }
        )
    return runs


def text_glyph_run_metrics(glyph_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not glyph_runs:
        return None
    run_metrics = []
    total_positions = 0
    total_advance = 0.0
    for index, run in enumerate(glyph_runs, start=1):
        positions = run.get("positions") or []
        code_values = run.get("codeOrWidthValues") or []
        baselines = run.get("baselines") or []
        glyph_count = max(len(code_values), len(positions) // 2, len(baselines))
        metric: dict[str, Any] = {
            "run": index,
            "glyphCountEstimate": glyph_count,
            "positionValueCount": len(positions),
            "baselineValueCount": len(baselines),
        }
        if len(positions) >= 2:
            xs = positions[0::2]
            ys = positions[1::2]
            metric["xRange"] = [min(xs), max(xs)]
            metric["yRange"] = [min(ys), max(ys)]
            advance = max(xs) - min(xs)
            metric["xAdvanceEstimate"] = advance
            total_advance += advance
            total_positions += len(xs)
        run_metrics.append(metric)
    out: dict[str, Any] = {
        "runs": run_metrics,
        "totalGlyphCountEstimate": sum(item["glyphCountEstimate"] for item in run_metrics),
    }
    if total_positions:
        out["totalXAdvanceEstimate"] = total_advance
        out["averageXAdvancePerPosition"] = total_advance / total_positions
    return out


def text_case_summary(source_text: str | None) -> dict[str, Any] | None:
    if not source_text:
        return None
    letters = [char for char in source_text if char.isalpha()]
    if not letters:
        return {"length": len(source_text), "letterCount": 0}
    uppercase_count = sum(1 for char in letters if char.upper() == char)
    return {
        "length": len(source_text),
        "letterCount": len(letters),
        "uppercaseLetterCount": uppercase_count,
        "isAllCaps": uppercase_count == len(letters),
        "uppercaseRatio": uppercase_count / len(letters),
    }


def unique_numbers(values: list[float]) -> list[float]:
    out = []
    seen = set()
    for value in values:
        cleaned = clean_number(value)
        key = repr(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def dominant_number(values: list[float]) -> int | float | None:
    if not values:
        return None
    counts = Counter(clean_number(value) for value in values)
    return counts.most_common(1)[0][0]


def text_numeric_style_summary(values: list[float], unit: str) -> dict[str, Any] | None:
    dominant = dominant_number(values)
    if dominant is None:
        return None
    return {
        "value": dominant,
        "unit": unit,
        "candidates": unique_numbers(values),
    }


def text_paragraph_style_summary(hints: dict[str, Any]) -> dict[str, Any] | None:
    style: dict[str, Any] = {}
    justifications = hints.get("justificationCandidates") or []
    if justifications:
        counts = Counter(item.get("code") for item in justifications if item.get("code") is not None)
        code = counts.most_common(1)[0][0] if counts else justifications[-1].get("code")
        label = KNOWN_TEXT_PARAGRAPH_TOKENS.get(code)
        style["justification"] = {
            "code": code,
            "label": label,
            "candidates": justifications,
        }
    for label, key, unit in [
        ("leading", "leadingCandidates", "pixels-or-points"),
        ("tracking", "trackingCandidates", "ae-tracking-units"),
        ("baselineShift", "baselineShiftCandidates", "pixels-or-points"),
    ]:
        summary = text_numeric_style_summary(hints.get(key) or [], unit)
        if summary:
            style[label] = summary
    if hints.get("boxTextCandidate"):
        style["boxText"] = True
    return style or None


def text_paragraph_hints(payload: str) -> dict[str, Any] | None:
    hints: dict[str, Any] = {}
    for label in ("Justification", "Alignment", "Paragraph", "Leading", "Tracking", "Baseline"):
        if label in payload:
            hints.setdefault("tokens", []).append(label)
    justification_candidates = []
    for match in re.finditer(r"/(?:Justification|alignment|Alignment)\s+([-+]?\d+)", payload):
        code = int(match.group(1))
        justification_candidates.append(
            {
                "code": code,
                "labelCandidate": KNOWN_TEXT_PARAGRAPH_TOKENS.get(code),
            }
        )
    if justification_candidates:
        hints["justificationCandidates"] = justification_candidates
    leading_candidates = [
        value
        for match in re.finditer(r"/(?:Leading|leading)\s+([-+]?(?:\d+\.\d+|\.\d+|\d+))", payload)
        for value in [float(match.group(1))]
        if -1000 <= value <= 1000
    ]
    if leading_candidates:
        hints["leadingCandidates"] = leading_candidates
    tracking_candidates = [
        value
        for match in re.finditer(r"/(?:Tracking|tracking)\s+([-+]?(?:\d+\.\d+|\.\d+|\d+))", payload)
        for value in [float(match.group(1))]
        if -10000 <= value <= 10000
    ]
    if tracking_candidates:
        hints["trackingCandidates"] = tracking_candidates
    baseline_candidates = [
        value
        for match in re.finditer(r"/(?:Baseline|baseline)\s+([-+]?(?:\d+\.\d+|\.\d+|\d+))", payload)
        for value in [float(match.group(1))]
        if -1000 <= value <= 1000
    ]
    if baseline_candidates:
        hints["baselineShiftCandidates"] = baseline_candidates
    if re.search(r"(BoxText|TextBox|boxText|box text)", payload):
        hints["boxTextCandidate"] = True
    paragraph_style = text_paragraph_style_summary(hints)
    if paragraph_style:
        hints["paragraphStyle"] = paragraph_style
    return hints or None


def decode_text_style(data: bytes, source_text: str | None) -> dict[str, Any] | None:
    payload = source_literal_window(data, source_text)
    paints = text_paint_slots(payload)
    sizes = text_size_candidates(payload)
    boxes = text_bounds_candidates(payload)
    glyph_runs = text_glyph_runs(payload)

    out: dict[str, Any] = {"decodeStatus": "best-effort-text-style"}
    if paints:
        out["paintSlots"] = paints
        for slot_name, slot in [("strokeOrPrimaryColor", 53), ("fillColor", 54), ("backgroundOrAuxColor", 79)]:
            for paint in paints:
                if paint.get("slot") == slot:
                    out[slot_name] = paint.get("color")
                    break
    if sizes:
        out["fontSizeCandidates"] = sizes
        out["fontSize"] = Counter(sizes).most_common(1)[0][0]
    if boxes:
        out["sourceRectCandidates"] = boxes
        out["sourceRect"] = max(boxes, key=lambda item: item["width"] * item["height"])
    if glyph_runs:
        out["glyphRuns"] = glyph_runs[:8]
        metrics = text_glyph_run_metrics(glyph_runs)
        if metrics:
            out["glyphMetrics"] = metrics
    paragraph_hints = text_paragraph_hints(payload)
    if paragraph_hints:
        out["paragraphHints"] = paragraph_hints
        if paragraph_hints.get("paragraphStyle"):
            out["paragraphStyle"] = paragraph_hints["paragraphStyle"]
    return out if len(out) > 1 else None


def decode_text_document(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if not data:
        return None
    literals = pdf_literal_strings(data)
    source_candidates = [
        value.rstrip("\r\n")
        for value in literals
        if value.endswith("\r") and value.strip("\r\n")
    ]
    fonts = []
    seen_fonts = set()
    for literal in literals:
        value = literal.strip("\r\n")
        if looks_like_font_name(value) and value not in seen_fonts:
            seen_fonts.add(value)
            fonts.append(value)

    out: dict[str, Any] = {
        "byteLength": len(data),
        "decodeStatus": "pdf-literal-strings",
        "literalCount": len(literals),
        "literalStrings": literals,
        "sourceTextCandidates": source_candidates,
        "fontCandidates": fonts,
    }
    if source_candidates:
        out["sourceText"] = source_candidates[-1]
        out["sourceTextLength"] = len(out["sourceText"])
        case_summary = text_case_summary(out["sourceText"])
        if case_summary:
            out["case"] = case_summary
    if fonts:
        out["primaryFont"] = fonts[0]
    style = decode_text_style(data, out.get("sourceText"))
    if style:
        out["style"] = style
    return out


def decode_property_value(match_name: str | None, tdb4: dict[str, Any] | None, cdat: list[float], tdu_min: list[float], tdu_max: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    component_count = int((tdb4 or {}).get("componentCount") or 0)
    if cdat:
        out["rawDoubles"] = cdat
        if component_count > 0:
            out["value"] = cdat[:component_count]
        else:
            out["value"] = cdat

    if tdu_min:
        out["min"] = tdu_min[: max(1, component_count)]
    if tdu_max:
        out["max"] = tdu_max[: max(1, component_count)]

    match = match_name or ""
    if cdat and ("Color" in match or "Colour" in match):
        color = normalized_color(cdat)
        if color:
            out["color"] = color
    if cdat and "Opacity" in match and tdu_max and tdu_max[0] == 100 and abs(cdat[0]) <= 1:
        out["displayValue"] = cdat[0] * 100

    if not cdat and tdb4:
        storage_code = tdb4.get("storageCode")
        if storage_code in (14, 15):
            out["status"] = "external-or-expression-backed-value"
            out["note"] = "No cdat payload on this property body; values/keyframes are referenced through AE's native animated/expression storage."

    return out


def decode_lhd3(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    data = bdata_bytes(node)
    if len(data) < 20:
        return None
    u32 = read_u32_values_be(data)
    out: dict[str, Any] = {
        "byteLength": len(data),
        "magic": hex(u32[0]) if u32 else None,
        "rawU32Head": u32[:16],
    }
    if len(u32) >= 5:
        out.update(
            {
                "keyframeCount": u32[2],
                "recordSize": u32[4],
                "recordCountMirror": u32[5] if len(u32) > 5 else None,
            }
        )
    return out


def keyframe_value_offset(record_size: int, component_count: int) -> int | None:
    if record_size >= 128 and component_count > 1:
        return 56
    if record_size >= 48 and component_count == 1:
        return 8
    return None


def interpolation_hint(record: bytes) -> str | None:
    header = record[4:12]
    if header.startswith(b"\x01\x01"):
        return "linear"
    if b"\x02\x02" in header[:4]:
        return "bezier"
    return None


def finite_reasonable_float(value: float) -> bool:
    return value == value and abs(value) != float("inf") and abs(value) < 1e12


def keyframe_record_probe(record: bytes, component_count: int, value_offset: int | None) -> dict[str, Any]:
    doubles = []
    for offset in range(8, len(record) - 7, 8):
        value = struct.unpack_from(">d", record, offset)[0]
        if finite_reasonable_float(value):
            doubles.append({"offset": offset, "value": value})

    out: dict[str, Any] = {
        "byteLength": len(record),
        "headerU16": read_u16_values_be(record[: min(32, len(record))]),
        "headerU32": read_u32_values_be(record[: min(32, len(record))]),
        "doubleSlots": doubles[:24],
    }
    if value_offset is not None:
        before = [item for item in doubles if item["offset"] < value_offset]
        after = [item for item in doubles if item["offset"] >= value_offset + component_count * 8]
        if before:
            out["preValueDoubleSlots"] = before[-12:]
        if after:
            out["postValueDoubleSlots"] = after[:12]
        if component_count >= 2:
            out["spatialTangentCandidates"] = {
                "preValuePairs": before[-4:],
                "postValuePairs": after[:4],
                "confidence": "raw-slot-candidate",
            }
    if len(record) >= 32:
        out["temporalEaseHeaderCandidates"] = {
            "bytes4to32": record[4:32].hex(),
            "u16": read_u16_values_be(record[4:32]),
            "u32": read_u32_values_be(record[4:32]),
        }
    return out


def keyframe_temporal_summary(keyframes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not keyframes:
        return None
    hints = Counter(item.get("interpolationHint") or "unknown" for item in keyframes)
    decoded_values = [item.get("value") for item in keyframes if "value" in item]
    unknown_value_count = sum(1 for item in keyframes if item.get("valueDecodeStatus"))
    frame_values = [
        item.get("compFrame", item.get("localFrame"))
        for item in keyframes
        if item.get("compFrame", item.get("localFrame")) is not None
    ]
    out: dict[str, Any] = {
        "interpolationHints": dict(hints),
        "hasLinear": bool(hints.get("linear")),
        "hasBezier": bool(hints.get("bezier")),
        "hasUnknownInterpolation": bool(hints.get("unknown")),
        "decodedValueCount": len(decoded_values),
        "unknownValueLayoutCount": unknown_value_count,
    }
    if frame_values:
        out["frameSpan"] = [min(frame_values), max(frame_values)]
    if decoded_values:
        unique_values = {json.dumps(value, sort_keys=True) for value in decoded_values}
        out["uniqueValueCount"] = len(unique_values)
        scalar_values = [value for value in decoded_values if isinstance(value, (int, float))]
        if scalar_values:
            out["scalarValueRange"] = [min(scalar_values), max(scalar_values)]
    return out


def keyframe_easing_candidate(entry: dict[str, Any]) -> dict[str, Any] | None:
    probe = entry.get("recordProbe") or {}
    temporal = probe.get("temporalEaseHeaderCandidates")
    spatial = probe.get("spatialTangentCandidates")
    if not temporal and not spatial and not entry.get("interpolationHint"):
        return None

    out: dict[str, Any] = {
        "index": entry.get("index"),
        "confidence": "raw-record-slot-candidate",
    }
    if entry.get("interpolationHint"):
        out["interpolationHint"] = entry["interpolationHint"]
    if temporal:
        out["temporalEaseHeaderCandidate"] = {
            key: temporal.get(key)
            for key in ("bytes4to32", "u16", "u32")
            if temporal.get(key) is not None
        }
    if spatial:
        out["spatialTangentCandidate"] = spatial
    return out


def keyframe_easing_summary(keyframes: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        candidate
        for keyframe in keyframes
        for candidate in [keyframe_easing_candidate(keyframe)]
        if candidate
    ]
    if not candidates:
        return None

    temporal_count = sum(1 for item in candidates if item.get("temporalEaseHeaderCandidate"))
    spatial_count = sum(1 for item in candidates if item.get("spatialTangentCandidate"))
    hints = Counter(item.get("interpolationHint") or "unknown" for item in candidates)
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-easing-candidates",
        "candidateCount": len(candidates),
        "hasTemporalEaseHeaderCandidates": temporal_count > 0,
        "hasSpatialTangentCandidates": spatial_count > 0,
        "temporalEaseHeaderCandidateCount": temporal_count,
        "spatialTangentCandidateCount": spatial_count,
        "interpolationHints": dict(hints),
        "candidateKeyframes": candidates,
    }
    return out


def decode_keyframe_records(
    list_node: dict[str, Any] | None,
    match_name: str | None,
    tdb4: dict[str, Any] | None,
    tdu_max: list[float],
    layer_timing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not list_node:
        return None
    header = decode_lhd3(first_child(list_node, "lhd3"))
    ldat = first_child(list_node, "ldat")
    data = bdata_bytes(ldat) if ldat else b""
    if not header or not data:
        return None

    record_size = int(header.get("recordSize") or 0)
    if record_size <= 0:
        return {"header": header, "byteLength": len(data), "decodeStatus": "missing-record-size"}

    available_count = len(data) // record_size
    expected_count = int(header.get("keyframeCount") or available_count)
    record_count = min(expected_count, available_count)
    component_count = int((tdb4 or {}).get("componentCount") or 1) or 1
    value_offset = keyframe_value_offset(record_size, component_count)
    ticks_per_second = int((layer_timing or {}).get("ticksPerSecond") or AE_TICKS_PER_SECOND)
    layer_start_ticks = int((layer_timing or {}).get("startTicks") or 0)

    keyframes = []
    for index in range(record_count):
        start = index * record_size
        record = data[start : start + record_size]
        if len(record) < 8:
            continue
        local_ticks = read_i32_be(record, 0)
        comp_ticks = local_ticks + layer_start_ticks
        entry: dict[str, Any] = {
            "index": index,
            "localTicks": local_ticks,
            "localFrame": clean_number(ticks_to_frames(local_ticks)),
            "localSeconds": ticks_to_seconds(local_ticks, ticks_per_second),
            "rawHeader": record[: min(16, len(record))].hex(),
        }
        if layer_timing:
            entry.update(
                {
                    "compTicks": comp_ticks,
                    "compFrame": clean_number(ticks_to_frames(comp_ticks)),
                    "compSeconds": ticks_to_seconds(comp_ticks, ticks_per_second),
                }
            )
        hint = interpolation_hint(record)
        if hint:
            entry["interpolationHint"] = hint
        entry["recordProbe"] = keyframe_record_probe(record, component_count, value_offset)

        if value_offset is not None and value_offset + component_count * 8 <= len(record):
            values = [
                struct.unpack_from(">d", record, value_offset + component_index * 8)[0]
                for component_index in range(component_count)
            ]
            if component_count == 1:
                value: float | list[float] = values[0]
            else:
                value = values
            entry["value"] = value
            if (
                component_count == 1
                and match_name
                and "Opacity" in match_name
                and tdu_max
                and tdu_max[0] == 100
                and abs(values[0]) <= 1
            ):
                entry["displayValue"] = values[0] * 100
        else:
            entry["valueDecodeStatus"] = "unknown-record-layout"
        keyframes.append(entry)

    out = {
        "header": header,
        "byteLength": len(data),
        "recordSize": record_size,
        "decodeStatus": "decoded-known-layout" if value_offset is not None else "unknown-record-layout",
        "keyframes": keyframes,
    }
    temporal_summary = keyframe_temporal_summary(keyframes)
    if temporal_summary:
        out["temporalSummary"] = temporal_summary
    easing_summary = keyframe_easing_summary(keyframes)
    if easing_summary:
        out["easingSummary"] = easing_summary
    return out


def parse_key_data(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"chunks": []}
    for child in node.get("children", []) or []:
        kind = node_kind(child)
        data = bdata_bytes(child)
        entry: dict[str, Any] = {"kind": kind, "byteLength": len(data)}
        doubles = read_doubles_be(data)
        if doubles:
            entry["doubles"] = doubles
        strings = child.get("strings") or []
        if strings:
            entry["strings"] = strings
        out["chunks"].append(entry)
    return out


def parse_property_body(
    node: dict[str, Any],
    match_name: str | None,
    animated: bool = False,
    layer_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    display_name = direct_text(first_child(node, "tdsn") or {})
    tdb4 = decode_tdb4(first_child(node, "tdb4"))
    cdat = decoded_doubles_for_child(node, "cdat")
    tdu_min = decoded_doubles_for_child(node, "tdum")
    tdu_max = decoded_doubles_for_child(node, "tduM")
    clean_display_name = display_name if display_name not in ("-_0_/-", "Utf8") else None
    prop: dict[str, Any] = {
        "type": "property",
        "name": clean_display_name,
        "matchName": match_name,
        "animatedContainer": animated,
        "tdb4": tdb4,
        "value": decode_property_value(match_name, tdb4, cdat, tdu_min, tdu_max),
        "chunks": {
            node_kind(child): len(bdata_bytes(child))
            for child in node.get("children", []) or []
            if bdata_bytes(child)
        },
    }
    expressions = [
        value
        for value in direct_child_strings(node)
        if "thisComp." in value or "effect(" in value or "sourceRectAtTime" in value
    ]
    if expressions:
        prop["expressions"] = expressions
        expression_analysis = analyze_expressions(expressions)
        if expression_analysis:
            prop["expressionAnalysis"] = expression_analysis
    keyframe_data = decode_keyframe_records(
        first_child(node, "list"),
        match_name,
        tdb4,
        tdu_max,
        layer_timing,
    )
    if keyframe_data:
        prop["keyframes"] = keyframe_data
        if prop["value"].get("status") == "external-or-expression-backed-value":
            prop["value"] = {"status": "animated-keyframes-decoded"}
    return prop


def parse_otst(node: dict[str, Any], match_name: str | None, layer_timing: dict[str, Any] | None = None) -> dict[str, Any]:
    body = first_child(node, "tdbs")
    prop = parse_property_body(body or {}, match_name, animated=True, layer_timing=layer_timing)
    prop["animation"] = {
        "kind": "otst",
        "keyContainers": [parse_key_data(child) for child in node.get("children", []) or [] if node_kind(child) == "otky"],
    }
    return prop


def parse_text_document(node: dict[str, Any], match_name: str | None) -> dict[str, Any]:
    body = first_child(node, "tdbs")
    base = parse_property_body(body or {}, match_name)
    return {
        "type": "textDocument",
        "name": base.get("name"),
        "matchName": match_name,
        "value": base.get("value"),
        "tdb4": base.get("tdb4"),
        "textDocument": decode_text_document(first_child(node, "btdk")),
        "chunks": {
            node_kind(child): len(bdata_bytes(child))
            for child in node.get("children", []) or []
            if bdata_bytes(child)
        },
    }


def parse_effect_param_definitions(node: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = []
    current: dict[str, Any] | None = None

    for child in node.get("children", []) or []:
        kind = node_kind(child)
        if kind == "tdmn":
            match_name = decode_bdata_text(child)
            current = {"matchName": match_name}
            definitions.append(current)
        elif kind == "pard" and current is not None:
            strings = [
                value
                for value in node_strings(child)
                if value and value not in ("Utf8", "-_0_/-")
            ]
            if strings:
                current["name"] = strings[0]
        elif kind == "pdnm" and current is not None:
            options = []
            for value in node_strings(child):
                for label in value.split("|"):
                    label = label.strip()
                    if label and label not in ("Utf8", "-_0_/-"):
                        options.append(label)
            if options:
                current["options"] = options

    return [item for item in definitions if item.get("matchName")]


def parse_property_group(
    node: dict[str, Any],
    match_name: str | None = None,
    layer_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_name = direct_text(first_child(node, "tdsn") or {})
    group: dict[str, Any] = {
        "type": "group",
        "name": group_name if group_name != "-_0_/-" else None,
        "matchName": match_name,
        "children": [],
    }
    current_match = None
    for child in node.get("children", []) or []:
        kind = node_kind(child)
        if kind == "tdmn":
            current_match = decode_bdata_text(child)
        elif kind == "tdbs":
            group["children"].append(parse_property_body(child, current_match, layer_timing=layer_timing))
        elif kind == "otst":
            group["children"].append(parse_otst(child, current_match, layer_timing=layer_timing))
        elif kind == "btds":
            group["children"].append(parse_text_document(child, current_match))
        elif kind == "tdgp":
            group["children"].append(parse_property_group(child, current_match, layer_timing=layer_timing))
        elif kind == "sspc":
            effect: dict[str, Any] = {
                "type": "effect",
                "matchName": current_match,
                "name": direct_text(first_child(child, "fnam") or {}) or direct_text(child),
                "children": [],
            }
            for effect_child in child.get("children", []) or []:
                effect_kind = node_kind(effect_child)
                if effect_kind == "parT":
                    effect["paramDefinitions"] = parse_effect_param_definitions(effect_child)
                elif effect_kind == "tdgp":
                    effect["children"].append(
                        parse_property_group(effect_child, current_match, layer_timing=layer_timing)
                    )
            group["children"].append(effect)
    return group


def walk_semantic(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    out = [node]
    for child in node.get("children") or []:
        out.extend(walk_semantic(child))
    return out


def semantic_nodes_by_match(node: dict[str, Any] | None, match_name: str) -> list[dict[str, Any]]:
    return [item for item in walk_semantic(node) if item.get("matchName") == match_name]


def first_semantic_by_match(node: dict[str, Any] | None, match_name: str) -> dict[str, Any] | None:
    matches = semantic_nodes_by_match(node, match_name)
    return matches[0] if matches else None


def property_value(node: dict[str, Any] | None) -> Any:
    if not node:
        return None
    value = node.get("value")
    if isinstance(value, dict):
        if "displayValue" in value:
            return value["displayValue"]
        return value.get("value")
    return value


def property_color(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    value = node.get("value")
    if isinstance(value, dict) and isinstance(value.get("color"), dict):
        return value["color"]
    return None


def path_payload_vertices(value: Any) -> list[Any]:
    if isinstance(value, dict):
        vertices = value.get("vertices") or value.get("points") or value.get("value")
        return vertices if isinstance(vertices, list) else []
    return value if isinstance(value, list) else []


def path_payload_tangents(value: Any, *keys: str) -> list[Any]:
    if not isinstance(value, dict):
        return []
    for key in keys:
        tangents = value.get(key)
        if isinstance(tangents, list):
            return tangents
    return []


def path_payload_has_tangents(value: Any) -> bool:
    in_tangents = path_payload_tangents(value, "inTangents", "in_tangents", "inTangent")
    out_tangents = path_payload_tangents(value, "outTangents", "out_tangents", "outTangent")
    return bool(in_tangents or out_tangents)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0", ""}:
            return False
    return bool(value)


def path_payload_metadata(value: Any) -> dict[str, Any] | None:
    vertices = path_payload_vertices(value)
    if not vertices:
        return None
    metadata: dict[str, Any] = {
        "vertexCount": len(vertices),
        "hasTangents": path_payload_has_tangents(value),
    }
    if isinstance(value, dict):
        if "closed" in value:
            metadata["closed"] = boolish(value.get("closed"))
        in_tangents = path_payload_tangents(value, "inTangents", "in_tangents", "inTangent")
        out_tangents = path_payload_tangents(value, "outTangents", "out_tangents", "outTangent")
        if in_tangents:
            metadata["inTangentCount"] = len(in_tangents)
        if out_tangents:
            metadata["outTangentCount"] = len(out_tangents)
    return metadata


def path_decode_status(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return "raw-or-expression-backed"
    metadata = path_payload_metadata(summary.get("value"))
    if not metadata:
        return "raw-or-expression-backed"
    if metadata.get("hasTangents"):
        return "bezier-path-with-tangents-decoded"
    return "bezier-path-decoded"


def property_summary(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    value_payload = node.get("value") if isinstance(node.get("value"), dict) else {}
    out: dict[str, Any] = {
        "value": property_value(node),
    }
    path_metadata = path_payload_metadata(out.get("value"))
    if path_metadata:
        out["pathMetadata"] = path_metadata
    if isinstance(value_payload, dict):
        for key in ("status", "note"):
            if value_payload.get(key):
                out[key] = value_payload[key]
    color = property_color(node)
    if color:
        out["color"] = color
    tdb4 = node.get("tdb4") or {}
    if tdb4:
        out["tdb4"] = {
            key: tdb4.get(key)
            for key in ("componentCount", "storageCode", "dimensionCode", "flagsA", "flagsB")
            if tdb4.get(key) is not None
        }
    if node.get("chunks") and out.get("value") is None:
        out["chunks"] = node["chunks"]
    if node.get("expressions"):
        out["expressions"] = node["expressions"]
    if node.get("expressionAnalysis"):
        out["expressionAnalysis"] = node["expressionAnalysis"]
    if node.get("keyframes"):
        out["keyframes"] = node["keyframes"]
    return out


def value_as_scalar(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def named_child_group(effect: dict[str, Any]) -> str | None:
    for child in effect.get("children") or []:
        name = child.get("name")
        match_name = child.get("matchName")
        if name and name != "Compositing Options" and match_name != "ADBE Effect Built In Params":
            return name
    return None


def effect_kind(effect: dict[str, Any]) -> str:
    match_name = effect.get("matchName") or ""
    name = effect.get("name") or ""
    normalized_name = name.lower()
    if match_name.startswith("Pseudo/") or "Dropdown" in name:
        return "dropdownControl"
    if match_name == "ADBE Slider Control":
        return "sliderControl"
    if match_name == "ADBE Color Control":
        return "colorControl"
    if match_name == "ADBE Fill":
        return "fill"
    if match_name == "ADBE Drop Shadow":
        return "dropShadow"
    if match_name in {"ADBE Gaussian Blur", "ADBE Gaussian Blur 2"} or "gaussian blur" in normalized_name:
        return "gaussianBlur"
    if match_name in {"ADBE Box Blur", "ADBE Box Blur2", "ADBE Fast Box Blur"} or "box blur" in normalized_name:
        return "boxBlur"
    if match_name in {"ADBE Glo2", "ADBE Glow"} or normalized_name == "glow":
        return "glow"
    if match_name == "ADBE Tint" or normalized_name == "tint":
        return "tint"
    if match_name == "ADBE Linear Wipe" or "linear wipe" in normalized_name:
        return "linearWipe"
    if match_name == "ADBE Stroke" or normalized_name == "stroke":
        return "strokeEffect"
    return "effect"


def effect_parameter_definitions(effect: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = {}
    for item in effect.get("paramDefinitions") or []:
        match_name = item.get("matchName")
        if match_name:
            definitions[match_name] = item
    return definitions


def effect_options(effect: dict[str, Any]) -> list[str]:
    options = []
    for item in effect.get("paramDefinitions") or []:
        for label in item.get("options") or []:
            if label not in options:
                options.append(label)
    return options


def effect_property_nodes(effect: dict[str, Any]) -> list[dict[str, Any]]:
    properties = []

    def walk(node: dict[str, Any], in_builtin: bool = False) -> None:
        node_type = node.get("type")
        builtin = (
            in_builtin
            or node.get("matchName") == "ADBE Effect Built In Params"
            or node.get("name") == "Compositing Options"
        )
        if node_type == "property" and not builtin:
            properties.append(node)
            return
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, builtin)

    walk(effect)
    named = [item for item in properties if item.get("name")]
    if named:
        properties = [
            item
            for item in properties
            if item.get("name") or not str(item.get("matchName") or "").endswith("-0000")
        ]
    return properties


def effect_property_name(prop: dict[str, Any], effect: dict[str, Any]) -> str:
    name = prop.get("name")
    if name:
        return name

    definitions = effect_parameter_definitions(effect)
    definition = definitions.get(prop.get("matchName") or "")
    if definition and definition.get("name"):
        return definition["name"]

    kind = effect_kind(effect)
    if kind == "dropdownControl":
        return "Menu"
    if kind == "sliderControl":
        return "Slider"
    if kind == "colorControl":
        return "Color"
    return "Value"


def value_looks_like_color(prop: dict[str, Any], effect: dict[str, Any]) -> bool:
    name = prop.get("name") or ""
    match_name = prop.get("matchName") or ""
    kind = effect_kind(effect)
    return (
        "Color" in name
        or "Colour" in name
        or "Color" in match_name
        or "Colour" in match_name
        or (kind == "fill" and name == "Color")
        or (kind == "colorControl" and effect_property_name(prop, effect) == "Color")
    )


def effect_parameter_summary(prop: dict[str, Any], effect: dict[str, Any]) -> dict[str, Any]:
    value = prop.get("value") if isinstance(prop.get("value"), dict) else {}
    summary: dict[str, Any] = {
        "name": effect_property_name(prop, effect),
        "matchName": prop.get("matchName"),
    }
    raw_value = value.get("displayValue", value.get("value"))
    if raw_value is not None:
        summary["value"] = value_as_scalar(raw_value)

    color = value.get("color")
    if not color and isinstance(raw_value, list) and value_looks_like_color(prop, effect):
        color = normalized_color(raw_value)
    if color:
        summary["color"] = color

    for key in ("min", "max"):
        if key in value:
            summary[key] = value_as_scalar(value[key])
    if prop.get("expressions"):
        summary["expressions"] = prop["expressions"]
    if prop.get("expressionAnalysis"):
        summary["expressionAnalysis"] = prop["expressionAnalysis"]
    if prop.get("keyframes"):
        summary["keyframes"] = prop["keyframes"]
    return summary


def effect_specific_summary(effect: dict[str, Any], parameters: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind = effect_kind(effect)
    by_name = {str(parameter.get("name") or ""): parameter for parameter in parameters}
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-effect-specific",
        "kind": kind,
    }
    if parameters:
        out["parameterMap"] = {
            str(parameter.get("name") or parameter.get("matchName") or index): {
                key: parameter.get(key)
                for key in ("value", "color", "min", "max", "matchName")
                if parameter.get(key) is not None
            }
            for index, parameter in enumerate(parameters, start=1)
        }

    def map_parameters(specs: dict[str, tuple[str, ...]]) -> dict[str, Any]:
        mapped = {}
        for label, names in specs.items():
            for name in names:
                if name in by_name:
                    mapped[label] = by_name[name]
                    break
        return mapped

    if kind == "dropShadow":
        mapped = map_parameters(
            {
                "color": ("Color", "Shadow Color"),
                "opacity": ("Opacity", "Shadow Opacity"),
                "direction": ("Direction", "Angle"),
                "distance": ("Distance",),
                "softness": ("Softness", "Blur"),
            }
        )
        if mapped:
            out["dropShadow"] = mapped
    elif kind == "fill":
        mapped = map_parameters(
            {
                "color": ("Color",),
                "opacity": ("Opacity",),
                "mask": ("Fill Mask", "Mask"),
            }
        )
        if mapped:
            out["fill"] = mapped
    elif kind in {"gaussianBlur", "boxBlur"}:
        mapped = map_parameters(
            {
                "blurriness": ("Blurriness", "Blur Radius", "Blur"),
                "blurDimensions": ("Blur Dimensions", "Dimensions"),
                "repeatEdgePixels": ("Repeat Edge Pixels",),
                "iterations": ("Iterations",),
            }
        )
        if mapped:
            out[kind] = mapped
    elif kind == "glow":
        mapped = map_parameters(
            {
                "threshold": ("Glow Threshold", "Threshold"),
                "radius": ("Glow Radius", "Radius"),
                "intensity": ("Glow Intensity", "Intensity"),
                "operation": ("Glow Operation", "Operation"),
                "compositeOriginal": ("Composite Original",),
                "colors": ("Glow Colors", "Colors"),
            }
        )
        if mapped:
            out["glow"] = mapped
    elif kind == "tint":
        mapped = map_parameters(
            {
                "mapBlackTo": ("Map Black To", "Black To"),
                "mapWhiteTo": ("Map White To", "White To"),
                "amount": ("Amount to Tint", "Amount"),
            }
        )
        if mapped:
            out["tint"] = mapped
    elif kind == "linearWipe":
        mapped = map_parameters(
            {
                "completion": ("Transition Completion", "Completion"),
                "angle": ("Wipe Angle", "Angle"),
                "feather": ("Feather",),
            }
        )
        if mapped:
            out["linearWipe"] = mapped
    elif kind == "strokeEffect":
        mapped = map_parameters(
            {
                "path": ("Path",),
                "color": ("Color",),
                "brushSize": ("Brush Size", "Size"),
                "brushHardness": ("Brush Hardness", "Hardness"),
                "opacity": ("Opacity",),
                "start": ("Start",),
                "end": ("End",),
                "spacing": ("Spacing",),
                "paintStyle": ("Paint Style",),
            }
        )
        if mapped:
            out["stroke"] = mapped
    elif kind in {"sliderControl", "colorControl", "dropdownControl"}:
        out["control"] = {
            "controlType": kind,
            "value": parameters[0] if len(parameters) == 1 else parameters,
        }
    return out if len(out) > 2 else None


def effect_summaries(properties: dict[str, Any] | None) -> list[dict[str, Any]]:
    effects = [
        item
        for item in walk_semantic(properties)
        if item.get("type") == "effect"
    ]
    summaries = []
    for effect in effects:
        parameters = [
            effect_parameter_summary(prop, effect)
            for prop in effect_property_nodes(effect)
        ]
        kind = effect_kind(effect)
        options = effect_options(effect)
        summary: dict[str, Any] = {
            "name": effect.get("name"),
            "matchName": effect.get("matchName"),
            "kind": kind,
        }
        control_name = named_child_group(effect)
        if control_name:
            summary["controlName"] = control_name
        if options and kind == "dropdownControl":
            summary["options"] = options
        if parameters:
            summary["parameters"] = parameters
            specific = effect_specific_summary(effect, parameters)
            if specific:
                summary["specific"] = specific
            if len(parameters) == 1:
                parameter = parameters[0]
                if "value" in parameter:
                    summary["value"] = parameter["value"]
                if "color" in parameter:
                    summary["color"] = parameter["color"]
        if parameters or (options and kind == "dropdownControl"):
            summaries.append(summary)
    return summaries


def layer_expression_dependencies(properties: dict[str, Any] | None) -> dict[str, Any] | None:
    expressions = [
        expression
        for item in walk_semantic(properties)
        for expression in (item.get("expressions") or [])
    ]
    return analyze_expressions(expressions)


def semantic_path_label(node: dict[str, Any]) -> str | None:
    name = node.get("name")
    match_name = node.get("matchName")
    node_type = node.get("type")
    if name and name not in ("Utf8", "-_0_/-"):
        return str(name)
    if match_name and match_name not in ("Utf8", "-_0_/-"):
        return str(match_name)
    if node_type and node_type not in {"group", "property"}:
        return str(node_type)
    return None


def keyframe_frame_range(keyframes: list[dict[str, Any]]) -> dict[str, Any] | None:
    frames = [
        item.get("compFrame", item.get("localFrame"))
        for item in keyframes
        if item.get("compFrame", item.get("localFrame")) is not None
    ]
    seconds = [
        item.get("compSeconds", item.get("localSeconds"))
        for item in keyframes
        if item.get("compSeconds", item.get("localSeconds")) is not None
    ]
    if not frames and not seconds:
        return None
    out: dict[str, Any] = {}
    if frames:
        out["frames"] = [min(frames), max(frames)]
    if seconds:
        out["seconds"] = [min(seconds), max(seconds)]
    return out


def animation_keyframes_summary(prop: dict[str, Any]) -> dict[str, Any]:
    keyframes = prop.get("keyframes") or {}
    entries = keyframes.get("keyframes") or []
    summary: dict[str, Any] = {
        "decodeStatus": keyframes.get("decodeStatus"),
        "recordSize": keyframes.get("recordSize"),
        "count": len(entries),
        "keyframes": entries,
    }
    header = keyframes.get("header")
    if header:
        summary["header"] = header
    if keyframes.get("temporalSummary"):
        summary["temporalSummary"] = keyframes["temporalSummary"]
    if keyframes.get("easingSummary"):
        summary["easingSummary"] = keyframes["easingSummary"]
    frame_range = keyframe_frame_range(entries)
    if frame_range:
        summary["range"] = frame_range
    return summary


def animated_property_entry(
    prop: dict[str, Any],
    path: list[str],
    effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path,
        "name": prop.get("name"),
        "matchName": prop.get("matchName"),
        "type": prop.get("type"),
        "value": property_value(prop),
        "animation": animation_keyframes_summary(prop),
    }
    if prop.get("expressionAnalysis"):
        entry["expressionAnalysis"] = prop["expressionAnalysis"]
    if effect:
        entry["effect"] = effect
    return entry


def animated_properties(properties: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not properties:
        return []

    entries = []

    def walk(
        node: dict[str, Any],
        path: list[str],
        effect_context: dict[str, Any] | None = None,
    ) -> None:
        label = semantic_path_label(node)
        next_path = path + [label] if label else path
        next_effect = effect_context
        if node.get("type") == "effect":
            next_effect = {
                "name": node.get("name"),
                "matchName": node.get("matchName"),
                "kind": effect_kind(node),
            }
            control_name = named_child_group(node)
            if control_name:
                next_effect["controlName"] = control_name

        if node.get("keyframes"):
            entries.append(animated_property_entry(node, next_path, next_effect))

        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, next_path, next_effect)

    walk(properties, [])
    return entries


def comp_dimensions(parent_comp: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not parent_comp:
        return None, None
    composition = parent_comp.get("composition") or {}
    width = composition.get("width")
    height = composition.get("height")
    return width, height


def comp_coordinate_space(parent_comp: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parent_comp:
        return None
    composition = parent_comp.get("composition") or {}
    width, height = comp_dimensions(parent_comp)
    out: dict[str, Any] = {
        "itemId": parent_comp.get("itemId"),
        "name": parent_comp.get("name"),
        "width": width,
        "height": height,
        "frameRate": composition.get("frameRate"),
        "durationFrames": composition.get("durationFrames"),
        "durationSeconds": composition.get("durationSeconds"),
    }
    return {key: value for key, value in out.items() if value is not None}


def value_as_float_list(value: Any) -> list[float] | None:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return None


def normalize_comp_value(match_name: str | None, value: Any, parent_comp: dict[str, Any] | None) -> Any:
    values = value_as_float_list(value)
    if not values:
        return None
    width, height = comp_dimensions(parent_comp)
    if not width or not height:
        return None

    match = match_name or ""
    two_axis_matches = {
        "ADBE Position",
        "ADBE Anchor Point",
        "ADBE Vector Position",
        "ADBE Vector Rect Position",
        "ADBE Vector Rect Size",
        "ADBE Text Position",
    }
    if match in two_axis_matches or match.endswith("Position") or match.endswith("Size"):
        if len(values) < 2:
            return None
        normalized = [values[0] / width, values[1] / height]
        if len(values) > 2:
            normalized.extend(values[2:])
        return [clean_number(item) for item in normalized]
    return None


def normalized_keyframes_for_entry(entry: dict[str, Any], parent_comp: dict[str, Any] | None) -> list[dict[str, Any]]:
    match_name = entry.get("matchName")
    out = []
    for keyframe in ((entry.get("animation") or {}).get("keyframes") or []):
        normalized = normalize_comp_value(match_name, keyframe.get("value"), parent_comp)
        if normalized is None:
            continue
        normalized_keyframe = {
            "index": keyframe.get("index"),
            "localFrame": keyframe.get("localFrame"),
            "compFrame": keyframe.get("compFrame"),
            "localSeconds": keyframe.get("localSeconds"),
            "compSeconds": keyframe.get("compSeconds"),
            "value": normalized,
            "interpolationHint": keyframe.get("interpolationHint"),
        }
        easing = keyframe_easing_candidate(keyframe)
        if easing:
            normalized_keyframe["easingCandidate"] = easing
        out.append(normalized_keyframe)
    return out


def normalized_transform(properties: dict[str, Any] | None, parent_comp: dict[str, Any] | None) -> dict[str, Any]:
    transform = {}
    if not properties:
        return transform
    for label, match_name in [
        ("anchorPoint", "ADBE Anchor Point"),
        ("position", "ADBE Position"),
    ]:
        prop = first_semantic_by_match(properties, match_name)
        if not prop:
            continue
        value = property_value(prop)
        normalized = normalize_comp_value(match_name, value, parent_comp)
        summary: dict[str, Any] = {}
        if normalized is not None:
            summary["value"] = normalized
        if prop.get("keyframes"):
            temp_entry = animated_property_entry(prop, [match_name])
            keyframes = normalized_keyframes_for_entry(temp_entry, parent_comp)
            if keyframes:
                summary["keyframes"] = keyframes
        if summary:
            transform[label] = summary
    return transform


def coordinate_mapping(
    properties: dict[str, Any] | None,
    animations: list[dict[str, Any]],
    parent_comp: dict[str, Any] | None,
) -> dict[str, Any] | None:
    coordinate_space = comp_coordinate_space(parent_comp)
    if not coordinate_space:
        return None
    out: dict[str, Any] = {
        "sourceComp": coordinate_space,
        "positionUnit": "comp-normalized",
        "notes": [
            "x and y are normalized against the AE parent comp width/height; y remains AE's top-down coordinate direction."
        ],
    }
    transform = normalized_transform(properties, parent_comp)
    if transform:
        out["transform"] = transform
    normalized_animations = []
    for entry in animations:
        keyframes = normalized_keyframes_for_entry(entry, parent_comp)
        if not keyframes:
            continue
        normalized_animations.append(
            {
                "path": entry.get("path"),
                "matchName": entry.get("matchName"),
                "keyframes": keyframes,
            }
        )
    if normalized_animations:
        out["animatedProperties"] = normalized_animations
    return out


def compact_animation_summary(animations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not animations:
        return []
    out = []
    for entry in animations:
        animation = entry.get("animation") or {}
        item: dict[str, Any] = {
            "path": entry.get("path"),
            "matchName": entry.get("matchName"),
            "count": animation.get("count"),
            "range": animation.get("range"),
            "decodeStatus": animation.get("decodeStatus"),
        }
        if animation.get("temporalSummary"):
            item["temporalSummary"] = animation["temporalSummary"]
        if animation.get("easingSummary"):
            item["easingSummary"] = animation["easingSummary"]
        if entry.get("effect"):
            item["effect"] = entry["effect"]
        out.append(item)
    return out


def compact_controls(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = []
    for effect in effects:
        if effect.get("kind") not in {"dropdownControl", "colorControl", "sliderControl"}:
            continue
        control: dict[str, Any] = {
            "name": effect.get("controlName") or effect.get("name"),
            "kind": effect.get("kind"),
            "matchName": effect.get("matchName"),
        }
        if "value" in effect:
            control["value"] = effect["value"]
        if "color" in effect:
            control["color"] = effect["color"]
        if effect.get("options"):
            control["options"] = effect["options"]
        if effect.get("parameters"):
            control["parameters"] = effect["parameters"]
        controls.append(control)
    return controls


def compact_appearance(effects: list[dict[str, Any]], visual: dict[str, Any] | None) -> dict[str, Any]:
    appearance: dict[str, Any] = {}
    fills = [effect for effect in effects if effect.get("kind") == "fill"]
    if fills:
        appearance["fills"] = fills
    shadows = [effect for effect in effects if effect.get("kind") == "dropShadow"]
    if shadows:
        appearance["dropShadows"] = shadows
    blurs = [effect for effect in effects if effect.get("kind") in {"gaussianBlur", "boxBlur"}]
    if blurs:
        appearance["blurs"] = blurs
    glows = [effect for effect in effects if effect.get("kind") == "glow"]
    if glows:
        appearance["glows"] = glows
    tints = [effect for effect in effects if effect.get("kind") == "tint"]
    if tints:
        appearance["tints"] = tints
    wipes = [effect for effect in effects if effect.get("kind") == "linearWipe"]
    if wipes:
        appearance["linearWipes"] = wipes
    strokes = [effect for effect in effects if effect.get("kind") == "strokeEffect"]
    if strokes:
        appearance["strokes"] = strokes
    if visual and visual.get("kind") == "shape":
        shape_fills = []
        for shape in visual.get("shapes") or []:
            if shape.get("fill"):
                shape_fills.append({"shape": shape.get("name"), "fill": shape["fill"]})
        if shape_fills:
            appearance["shapeFills"] = shape_fills
    return appearance


def text_conversion_summary(visual: dict[str, Any] | None) -> dict[str, Any] | None:
    if not visual or visual.get("kind") != "text":
        return None
    docs = visual.get("textDocuments") or []
    if not docs:
        return None
    primary = docs[0]
    out: dict[str, Any] = {}
    for key in ("sourceText", "sourceTextLength", "case", "primaryFont", "fontCandidates"):
        if primary.get(key):
            out[key] = primary[key]
    style = primary.get("style") or {}
    if style:
        for key in (
            "fontSize",
            "fillColor",
            "sourceRect",
            "glyphRuns",
            "glyphMetrics",
            "paragraphHints",
            "paragraphStyle",
        ):
            if style.get(key):
                out[key] = style[key]
    return out or None


def layer_role(layer: dict[str, Any]) -> str:
    visual_kind = (layer.get("visual") or {}).get("kind")
    effects = layer.get("effects") or []
    control_effects = [
        effect
        for effect in effects
        if effect.get("kind") in {"dropdownControl", "colorControl", "sliderControl"}
    ]
    if visual_kind in {"text", "shape", "footage"}:
        return visual_kind
    if control_effects:
        return "controlLayer"
    if layer.get("sourceItem"):
        return "precompOrFootage"
    if layer.get("name") and re.search(r"\b(POS|Move|Control|Expression)", layer["name"], re.IGNORECASE):
        return "utilityLayer"
    return visual_kind or "layer"


def fusion_node_hints(role: str, effects: list[dict[str, Any]]) -> list[str]:
    nodes = []
    if role == "text":
        nodes.extend(["TextPlus", "Transform"])
    elif role == "shape":
        nodes.extend(["Background", "RectangleMask", "Transform"])
    elif role == "footage":
        nodes.extend(["MediaInOrLoader", "Transform"])
    elif role == "controlLayer":
        nodes.append("PublishControls")
    elif role in {"utilityLayer", "precompOrFootage"}:
        nodes.append("Transform")
    else:
        nodes.append("MergeInput")

    if any(effect.get("kind") == "dropShadow" for effect in effects):
        nodes.append("DropShadow")
    if any(effect.get("kind") == "fill" for effect in effects):
        nodes.append("ColorOrFillExpression")
    if any(effect.get("kind") in {"gaussianBlur", "boxBlur"} for effect in effects):
        nodes.append("Blur")
    if any(effect.get("kind") == "glow" for effect in effects):
        nodes.append("Glow")
    if any(effect.get("kind") == "tint" for effect in effects):
        nodes.append("TintOrColorCorrector")
    if any(effect.get("kind") == "linearWipe" for effect in effects):
        nodes.append("WipeMask")
    if any(effect.get("kind") == "strokeEffect" for effect in effects):
        nodes.append("PaintOrStroke")
    return list(dict.fromkeys(nodes))


def layer_conversion_summary(layer: dict[str, Any]) -> dict[str, Any]:
    effects = layer.get("effects") or []
    visual = layer.get("visual") or {}
    role = layer_role(layer)
    summary: dict[str, Any] = {
        "role": role,
        "name": layer.get("name"),
        "suggestedFusionNodes": fusion_node_hints(role, effects),
    }
    if layer.get("parentComp"):
        summary["sourceComp"] = layer["parentComp"]
    if layer.get("timing"):
        timing = layer["timing"]
        summary["timing"] = {
            key: timing.get(key)
            for key in ("inFrame", "outFrame", "inSeconds", "outSeconds", "startFrame")
            if timing.get(key) is not None
        }
    if layer.get("sourceItem"):
        summary["sourceItem"] = layer["sourceItem"]
    if layer.get("parentLayer"):
        parent_layer = layer["parentLayer"]
        compact_parent = {
            key: parent_layer.get(key)
            for key in (
                "parentIndex",
                "parentName",
                "parentOrdinal",
                "relationshipConfidence",
                "resolutionStatus",
            )
            if parent_layer.get(key) is not None
        }
        if compact_parent:
            summary["parentLayer"] = compact_parent
    if layer.get("transfer"):
        summary["transfer"] = layer["transfer"]
    if layer.get("threeD"):
        three_d = layer["threeD"]
        compact_three_d = {
            key: three_d.get(key)
            for key in (
                "objectTypeCandidate",
                "layerIs3DFlagCandidate",
                "environmentLayerFlagCandidate",
                "lookAtCameraFlagCandidate",
                "lookAtPointOfInterestFlagCandidate",
            )
            if three_d.get(key) is not None
        }
        if three_d.get("materialOptions"):
            compact_three_d["materialOptions"] = three_d["materialOptions"]
        if three_d.get("cameraOptions"):
            compact_three_d["cameraOptions"] = three_d["cameraOptions"]
        if three_d.get("lightOptions"):
            compact_three_d["lightOptions"] = three_d["lightOptions"]
        if compact_three_d:
            summary["threeD"] = compact_three_d
    if layer.get("compositingHints"):
        hints = layer["compositingHints"]
        compact_hints = {
            key: hints.get(key)
            for key in ("rawLayerFlagsHex", "setBitPositions", "visualKind", "hasSourceItem")
            if hints.get(key) is not None
        }
        if hints.get("postLayerStreams"):
            compact_hints["postLayerStreams"] = [
                {
                    key: stream.get(key)
                    for key in ("index", "streamId", "typeName")
                    if stream.get(key) is not None
                }
                for stream in hints["postLayerStreams"]
            ]
        if hints.get("postLayerStreamHintSummary"):
            compact_hints["postLayerStreamHintSummary"] = hints["postLayerStreamHintSummary"]
        if compact_hints:
            summary["compositingHints"] = compact_hints
    if layer.get("coordinateMapping"):
        mapping = layer["coordinateMapping"]
        compact_mapping = {
            key: mapping.get(key)
            for key in ("positionUnit", "transform", "animatedProperties")
            if mapping.get(key)
        }
        if compact_mapping:
            summary["coordinateMapping"] = compact_mapping
    text_summary = text_conversion_summary(visual)
    if text_summary:
        summary["text"] = text_summary
    if visual.get("kind") == "shape" and visual.get("shapes"):
        summary["shapes"] = visual["shapes"]
    if visual.get("masks"):
        summary["masks"] = visual["masks"]
    controls = compact_controls(effects)
    if controls:
        summary["controls"] = controls
    appearance = compact_appearance(effects, visual)
    if appearance:
        summary["appearance"] = appearance
    animations = compact_animation_summary(layer.get("animatedProperties") or [])
    if animations:
        summary["animations"] = animations
    if layer.get("expressionDependencies"):
        summary["dependencies"] = layer["expressionDependencies"]
    if layer.get("expressionEvaluations"):
        summary["expressionEvaluations"] = [
            {
                key: evaluation.get(key)
                for key in ("path", "matchName", "patterns", "assignments", "finalValueCandidate", "resolvedTerms", "unresolvedTerms")
                if evaluation.get(key)
            }
            for evaluation in layer["expressionEvaluations"]
        ]
    return summary


def project_graph(items: list[dict[str, Any]], layers: list[dict[str, Any]]) -> dict[str, Any]:
    compositions = []
    for item in items:
        if item.get("kind") != "composition":
            continue
        comp = item.get("composition") or {}
        compositions.append(
            {
                "itemId": item.get("itemId"),
                "name": item.get("name"),
                "width": comp.get("width"),
                "height": comp.get("height"),
                "durationFrames": comp.get("durationFrames"),
                "frameRate": comp.get("frameRate"),
                "directLayerCount": item.get("directLayerCount"),
                "directLayerNames": item.get("directLayerNames"),
            }
        )

    layer_source_edges = []
    expression_edges = []
    parent_layer_edges = []
    for layer in layers:
        parent = layer.get("parentComp") or {}
        source = layer.get("sourceItem") or {}
        if source:
            layer_source_edges.append(
                {
                    "fromCompId": parent.get("itemId"),
                    "fromCompName": parent.get("name"),
                    "layerOrdinal": layer.get("ordinal"),
                    "layerName": layer.get("name"),
                    "toItemId": source.get("itemId"),
                    "toItemName": source.get("name"),
                    "toItemKind": source.get("kind"),
                    "toSourceMetadata": source.get("sourceMetadata"),
                }
            )

        dependencies = layer.get("expressionDependencies") or {}
        for ref in dependencies.get("layers") or []:
            expression_edges.append(
                {
                    "kind": "layerReference",
                    "fromLayerOrdinal": layer.get("ordinal"),
                    "fromLayerName": layer.get("name"),
                    "fromCompId": parent.get("itemId"),
                    "fromCompName": parent.get("name"),
                    "toCompName": ref.get("comp") or parent.get("name"),
                    "toLayerName": ref.get("name"),
                }
            )
        for ref in dependencies.get("effects") or []:
            expression_edges.append(
                {
                    "kind": "effectReference",
                    "fromLayerOrdinal": layer.get("ordinal"),
                    "fromLayerName": layer.get("name"),
                    "fromCompId": parent.get("itemId"),
                    "fromCompName": parent.get("name"),
                    "toCompName": ref.get("comp") or parent.get("name"),
                    "toLayerName": ref.get("layer") or layer.get("name"),
                    "effect": ref.get("effect"),
                    "property": ref.get("property"),
                    "scope": ref.get("scope"),
                }
            )
        parent_layer = layer.get("parentLayer") or {}
        if parent_layer.get("parentIndex") is not None:
            parent_layer_edges.append(
                {
                    "kind": "parentLayer",
                    "fromCompId": parent.get("itemId"),
                    "fromCompName": parent.get("name"),
                    "fromLayerOrdinal": layer.get("ordinal"),
                    "fromLayerName": layer.get("name"),
                    "toLayerOrdinal": parent_layer.get("parentOrdinal") or parent_layer.get("parentIndex"),
                    "toLayerName": parent_layer.get("parentName"),
                    "confidence": parent_layer.get("relationshipConfidence"),
                    "resolutionStatus": parent_layer.get("resolutionStatus"),
                }
            )

    out: dict[str, Any] = {
        "compositionCount": len(compositions),
        "layerCount": len(layers),
        "compositions": compositions,
    }
    if layer_source_edges:
        out["layerSourceEdges"] = layer_source_edges
    if expression_edges:
        out["expressionEdges"] = unique_dicts(expression_edges)
    if parent_layer_edges:
        out["parentLayerEdges"] = unique_dicts(parent_layer_edges)
    return out


def first_text_document(layer: dict[str, Any]) -> dict[str, Any] | None:
    docs = ((layer.get("visual") or {}).get("textDocuments") or [])
    return docs[0] if docs else None


def layer_source_rect(layer: dict[str, Any]) -> dict[str, Any] | None:
    doc = first_text_document(layer)
    return ((doc or {}).get("style") or {}).get("sourceRect")


def layer_timing_value(layer: dict[str, Any], key: str) -> float | None:
    timing = layer.get("timing") or {}
    value = timing.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def layer_parent_comp_value(layer: dict[str, Any], key: str) -> float | None:
    comp = layer.get("parentComp") or {}
    value = comp.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def expression_context_variables(layer: dict[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    in_seconds = layer_timing_value(layer, "inSeconds")
    start_seconds = layer_timing_value(layer, "startSeconds")
    frame_rate = layer_parent_comp_value(layer, "frameRate")
    width = layer_parent_comp_value(layer, "width")
    height = layer_parent_comp_value(layer, "height")
    duration = layer_parent_comp_value(layer, "durationSeconds")
    layer_index = layer.get("ordinal") or layer.get("treeIndex")

    variables["time"] = in_seconds if in_seconds is not None else start_seconds if start_seconds is not None else 0.0
    if frame_rate:
        variables["frameDuration"] = 1.0 / frame_rate
        variables["frameRate"] = frame_rate
    if width is not None:
        variables["compWidth"] = width
    if height is not None:
        variables["compHeight"] = height
    if duration is not None:
        variables["compDuration"] = duration
    if isinstance(layer_index, (int, float)):
        variables["index"] = float(layer_index)
    return variables


def representative_keyframed_value(summary: dict[str, Any]) -> list[float] | None:
    entries = ((summary.get("keyframes") or {}).get("keyframes") or [])
    values = []
    for entry in entries:
        value = value_as_float_list(entry.get("value"))
        if value:
            values.append(tuple(value))
    if not values:
        return None

    counts = Counter(values)
    best_count = counts.most_common(1)[0][1]
    tied = [value for value, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return list(tied[0])
    return list(values[len(values) // 2])


def layer_transform_position(layer: dict[str, Any]) -> list[float] | None:
    summary = (((layer.get("visual") or {}).get("transform") or {}).get("position") or {})
    position = value_as_float_list(summary.get("value"))
    if position:
        return position
    return representative_keyframed_value(summary)


def layer_effect_parameter(layer: dict[str, Any], effect_name: str, property_name: str) -> Any:
    for effect in layer.get("effects") or []:
        names = {effect.get("name"), effect.get("controlName")}
        if effect_name not in names:
            continue
        for parameter in effect.get("parameters") or []:
            if parameter.get("name") == property_name or parameter.get("matchName") == property_name:
                if "color" in parameter:
                    return parameter["color"]
                if "value" in parameter:
                    return parameter["value"]
        if property_name in {"Slider", "Menu", "Color"}:
            if "color" in effect:
                return effect["color"]
            if "value" in effect:
                return effect["value"]
    return None


def ae_clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ae_interpolation_progress(value: float, in_min: float | None = None, in_max: float | None = None) -> float:
    if in_min is None or in_max is None:
        return ae_clamp(value, 0.0, 1.0)
    if in_max == in_min:
        return 0.0
    return ae_clamp((value - in_min) / (in_max - in_min), 0.0, 1.0)


def ae_lerp(progress: float, out_min: float, out_max: float) -> float:
    return out_min + progress * (out_max - out_min)


def ae_linear(*args: float) -> float | None:
    if len(args) == 3:
        value, out_min, out_max = args
        return ae_lerp(ae_interpolation_progress(value), out_min, out_max)
    if len(args) == 5:
        value, in_min, in_max, out_min, out_max = args
        return ae_lerp(ae_interpolation_progress(value, in_min, in_max), out_min, out_max)
    return None


def ae_ease_progress(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def ae_ease_in_progress(progress: float) -> float:
    return progress * progress


def ae_ease_out_progress(progress: float) -> float:
    return 1.0 - (1.0 - progress) * (1.0 - progress)


def ae_ease_with_curve(curve: Any, *args: float) -> float | None:
    if len(args) == 3:
        value, out_min, out_max = args
        progress = ae_interpolation_progress(value)
        return ae_lerp(curve(progress), out_min, out_max)
    if len(args) == 5:
        value, in_min, in_max, out_min, out_max = args
        progress = ae_interpolation_progress(value, in_min, in_max)
        return ae_lerp(curve(progress), out_min, out_max)
    return None


def ae_ease(*args: float) -> float | None:
    return ae_ease_with_curve(ae_ease_progress, *args)


def ae_ease_in(*args: float) -> float | None:
    return ae_ease_with_curve(ae_ease_in_progress, *args)


def ae_ease_out(*args: float) -> float | None:
    return ae_ease_with_curve(ae_ease_out_progress, *args)


def ae_expression_eval_env(variables: dict[str, Any]) -> dict[str, Any]:
    env = {name: value for name, value in variables.items() if isinstance(value, (int, float, list))}
    env.update(
        {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
            "clamp": ae_clamp,
            "linear": ae_linear,
            "ease": ae_ease,
            "easeIn": ae_ease_in,
            "easeOut": ae_ease_out,
            "degreesToRadians": math.radians,
            "radiansToDegrees": math.degrees,
        }
    )
    return env


def normalize_ae_expression_functions(expression: str) -> str:
    replacements = {
        "Math.abs": "abs",
        "Math.min": "min",
        "Math.max": "max",
        "Math.round": "round",
        "Math.floor": "floor",
        "Math.ceil": "ceil",
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression


def safe_arithmetic_eval(expression: str, variables: dict[str, Any]) -> float | list[float] | None:
    expression = normalize_ae_expression_functions(expression.strip())
    if not expression:
        return None
    if not re.fullmatch(r"[0-9eE+\-*/().,\sA-Za-z_\[\]<>!=]+", expression):
        return None
    env = ae_expression_eval_env(variables)
    try:
        value = eval(expression, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(value, (int, float)) and finite_reasonable_float(float(value)):
        return float(value)
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return None


def strip_js_line_comments(expression: str) -> str:
    lines = []
    for line in expression.splitlines():
        lines.append(line.split("//", 1)[0])
    return "\n".join(lines)


def js_ternary_to_python(expression: str) -> str:
    expression = expression.strip().rstrip(";")
    if "?" not in expression or ":" not in expression:
        return expression
    question = expression.find("?")
    depth = 0
    colon = -1
    for index, char in enumerate(expression[question + 1 :], start=question + 1):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            colon = index
            break
    if colon < 0:
        return expression
    condition = expression[:question].strip()
    true_expr = expression[question + 1 : colon].strip()
    false_expr = expression[colon + 1 :].strip()
    if condition.startswith("(") and condition.endswith(")"):
        condition = condition[1:-1].strip()
    return f"({true_expr}) if ({condition}) else ({false_expr})"


def final_expression_line(expression: str) -> str | None:
    cleaned = strip_js_line_comments(expression)
    statements = []
    for line in cleaned.splitlines():
        for statement in line.split(";"):
            statement = statement.strip()
            if statement:
                statements.append(statement)
    for statement in reversed(statements):
        if re.match(r"^[A-Za-z_]\w*\s*=", statement):
            continue
        return statement
    return None


def substitute_expression_terms(
    expression: str,
    current_value: Any,
    current_layer: dict[str, Any],
    layers_by_name: dict[str, dict[str, Any]],
    aliases: dict[str, dict[str, Any]],
    variables: dict[str, Any],
    resolved: list[dict[str, Any]],
    unresolved: list[str],
) -> str:
    def replace_alias_position(match: re.Match[str]) -> str:
        alias = match.group("alias")
        axis = int(match.group("axis"))
        target_name = (aliases.get(alias) or {}).get("layer")
        target = layers_by_name.get(target_name or "")
        position = layer_transform_position(target or {})
        if position and len(position) > axis:
            value = position[axis]
            resolved.append({"kind": "layerPosition", "alias": alias, "layer": target_name, "axis": axis, "value": value})
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    def replace_alias_rect(match: re.Match[str]) -> str:
        alias = match.group("alias")
        field = match.group("field")
        target_name = (aliases.get(alias) or {}).get("layer")
        rect = layer_source_rect(layers_by_name.get(target_name or "") or {})
        if rect and field in rect:
            value = rect[field]
            resolved.append({"kind": "sourceRect", "alias": alias, "layer": target_name, "field": field, "value": value})
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    def replace_current_rect(match: re.Match[str]) -> str:
        field = match.group("field")
        rect = layer_source_rect(current_layer)
        if rect and field in rect:
            value = rect[field]
            resolved.append({"kind": "sourceRect", "layer": current_layer.get("name"), "field": field, "value": value})
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    def replace_current_position(match: re.Match[str]) -> str:
        axis = int(match.group("axis"))
        position = layer_transform_position(current_layer)
        if position and len(position) > axis:
            value = position[axis]
            resolved.append({"kind": "layerPosition", "layer": current_layer.get("name"), "axis": axis, "value": value})
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    def replace_effect(match: re.Match[str]) -> str:
        layer_name = match.group("layer")
        effect_name = match.group("effect")
        property_name = match.group("property")
        target_layer = layers_by_name.get(layer_name) if layer_name else current_layer
        value = layer_effect_parameter(target_layer or {}, effect_name, property_name)
        if value is not None:
            resolved.append(
                {
                    "kind": "effectParameter",
                    "layer": layer_name or current_layer.get("name"),
                    "effect": effect_name,
                    "property": property_name,
                    "value": value,
                }
            )
            if isinstance(value, list):
                variables[f"__effect_{len(resolved)}"] = value
                return f"__effect_{len(resolved)}"
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    def replace_value(match: re.Match[str]) -> str:
        axis = int(match.group("axis"))
        values = value_as_float_list(current_value)
        if values and len(values) > axis:
            value = values[axis]
            resolved.append({"kind": "currentValue", "axis": axis, "value": value})
            return repr(value)
        unresolved.append(match.group(0))
        return match.group(0)

    expression = re.sub(
        r"\b(?P<alias>(?!thisLayer\b|thisComp\b|thisProperty\b)[A-Za-z_]\w*)\.transform\.position\s*\[\s*(?P<axis>[01])\s*\]",
        replace_alias_position,
        expression,
    )
    expression = re.sub(
        r"\bthisLayer\.transform\.position\s*\[\s*(?P<axis>[01])\s*\]",
        replace_current_position,
        expression,
    )
    expression = re.sub(
        r"(?<!\.)\btransform\.position\s*\[\s*(?P<axis>[01])\s*\]",
        replace_current_position,
        expression,
    )
    expression = re.sub(
        r"\b(?P<alias>(?!thisLayer\b|thisComp\b|thisProperty\b)[A-Za-z_]\w*)\.sourceRectAtTime\s*\([^)]*\)\.(?P<field>width|height|left|top)",
        replace_alias_rect,
        expression,
    )
    expression = re.sub(
        r"\bthisLayer\.sourceRectAtTime\s*\([^)]*\)\.(?P<field>width|height|left|top)",
        replace_current_rect,
        expression,
    )
    expression = re.sub(
        r"(?<!\.)\bsourceRectAtTime\s*\([^)]*\)\.(?P<field>width|height|left|top)",
        replace_current_rect,
        expression,
    )
    expression = re.sub(
        r'(?:thisComp\.layer\("(?P<layer>[^"]+)"\)\.)?effect\("(?P<effect>[^"]+)"\)\("(?P<property>[^"]+)"\)(?:\.value)?',
        replace_effect,
        expression,
    )
    comp_replacements = {
        r"\bthisComp\.width\b": "compWidth",
        r"\bthisComp\.height\b": "compHeight",
        r"\bthisComp\.duration\b": "compDuration",
        r"\bthisComp\.frameDuration\b": "frameDuration",
        r"\bthisComp\.frameRate\b": "frameRate",
        r"\bthisLayer\.index\b": "index",
        r"\bthisLayer\.inPoint\b": repr(layer_timing_value(current_layer, "inSeconds") or 0.0),
        r"\bthisLayer\.outPoint\b": repr(layer_timing_value(current_layer, "outSeconds") or 0.0),
        r"\bthisLayer\.startTime\b": repr(layer_timing_value(current_layer, "startSeconds") or 0.0),
    }
    for pattern, replacement in comp_replacements.items():
        expression = re.sub(pattern, replacement, expression)
    expression = re.sub(r"\bvalue\s*\[\s*(?P<axis>[012])\s*\]", replace_value, expression)
    current_values = value_as_float_list(current_value)
    if current_values and len(current_values) == 1:
        expression = re.sub(r"\bvalue\b", repr(current_values[0]), expression)
    elif isinstance(current_value, (int, float)):
        expression = re.sub(r"\bvalue\b", repr(float(current_value)), expression)
    for name, value in variables.items():
        if isinstance(value, (int, float)):
            expression = re.sub(rf"\b{re.escape(name)}\b", repr(value), expression)
    return expression


def split_top_level_args(value: str) -> list[str]:
    args = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            args.append(value[start:index].strip())
            start = index + 1
    args.append(value[start:].strip())
    return args


def final_array_parts(expression: str) -> list[str] | None:
    expression = expression.strip().rstrip(";")
    if not expression:
        return None
    last_line = expression.splitlines()[-1].strip().rstrip(";")
    if not (last_line.startswith("[") and last_line.endswith("]")):
        return None
    parts = split_top_level_args(last_line[1:-1])
    return parts if len(parts) == 2 else None


def evaluate_expression(
    expression: str,
    current_value: Any,
    current_layer: dict[str, Any],
    layers_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analysis = parse_expression(expression)
    aliases = analysis.get("aliases") or {}
    variables: dict[str, Any] = expression_context_variables(current_layer)
    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    assignments = {}

    for match in re.finditer(r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;\r\n]+)", expression, re.MULTILINE):
        if EXPR_LAYER_ALIAS_RE.fullmatch(match.group(0).strip()):
            continue
        name = match.group("name")
        rhs = substitute_expression_terms(
            match.group("rhs"),
            current_value,
            current_layer,
            layers_by_name,
            aliases,
            variables,
            resolved,
            unresolved,
        )
        value = safe_arithmetic_eval(rhs, variables)
        if value is not None:
            variables[name] = value
            assignments[name] = value
        else:
            unresolved.append(match.group(0))

    final_value = None
    final_parts = final_array_parts(expression)
    if final_parts:
        parts = []
        for part in final_parts:
            rhs = substitute_expression_terms(
                part,
                current_value,
                current_layer,
                layers_by_name,
                aliases,
                variables,
                resolved,
                unresolved,
            )
            value = safe_arithmetic_eval(rhs, variables)
            parts.append(value)
        if all(isinstance(value, (int, float)) for value in parts):
            final_value = parts
    else:
        final_line = final_expression_line(expression)
        if final_line:
            rhs = substitute_expression_terms(
                final_line,
                current_value,
                current_layer,
                layers_by_name,
                aliases,
                variables,
                resolved,
                unresolved,
            )
            value = safe_arithmetic_eval(js_ternary_to_python(rhs), variables)
            if isinstance(value, (int, float)):
                final_value = value
            elif isinstance(value, list):
                final_value = value

    out: dict[str, Any] = {
        "decodeStatus": "best-effort-expression-evaluation",
        "patterns": analysis.get("patterns") or [],
    }
    if assignments:
        out["assignments"] = assignments
    if final_value is not None:
        out["finalValueCandidate"] = final_value
    if resolved:
        out["resolvedTerms"] = unique_dicts(resolved)
    if unresolved:
        out["unresolvedTerms"] = list(dict.fromkeys(unresolved))[:20]
    return out


def expression_current_value(node: dict[str, Any]) -> Any:
    value = property_value(node)
    if value is not None:
        return value
    if node.get("keyframes"):
        return representative_keyframed_value({"keyframes": node.get("keyframes")})
    return None


def collect_layer_expression_records(layer: dict[str, Any]) -> list[dict[str, Any]]:
    records = []

    def walk(node: dict[str, Any], path: list[str]) -> None:
        label = semantic_path_label(node)
        next_path = path + [label] if label else path
        for expression in node.get("expressions") or []:
            records.append(
                {
                    "path": next_path,
                    "matchName": node.get("matchName"),
                    "name": node.get("name"),
                    "value": expression_current_value(node),
                    "expression": expression,
                }
            )
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, next_path)

    walk(layer.get("properties") or {}, [])
    return records


def apply_expression_evaluations(layers: list[dict[str, Any]]) -> None:
    layers_by_name = {layer.get("name"): layer for layer in layers if layer.get("name")}
    for layer in layers:
        evaluations = []
        for record in collect_layer_expression_records(layer):
            evaluation = evaluate_expression(record["expression"], record.get("value"), layer, layers_by_name)
            evaluation.update(
                {
                    "path": record.get("path"),
                    "matchName": record.get("matchName"),
                    "expression": record.get("expression"),
                }
            )
            evaluations.append(evaluation)
        if evaluations:
            layer["expressionEvaluations"] = evaluations


def apply_layer_compositing_relationships(layers: list[dict[str, Any]]) -> None:
    for layer in layers:
        layer["transfer"] = transfer_mode_summary(layer)

    for index, layer in enumerate(layers):
        name = str(layer.get("name") or "")
        visual_kind = ((layer.get("visual") or {}).get("kind") or "")
        raw_flags = ((layer.get("timing") or {}).get("rawLayerFlags"))
        hidden_candidate = isinstance(raw_flags, int) and not bool(raw_flags & 0x1)
        looks_like_matte = bool(re.search(r"\b(MATTE|ALPHA|LUMA)\b", name, re.IGNORECASE))
        if not (looks_like_matte or (hidden_candidate and visual_kind == "shape")):
            continue
        if index + 1 >= len(layers):
            continue
        target = layers[index + 1]
        matte_type = "alpha" if "ALPHA" in name.upper() else "luma" if "LUMA" in name.upper() else "unknown"
        layer["transfer"]["usedAsTrackMatteCandidateFor"] = {
            "layerOrdinal": target.get("ordinal"),
            "layerName": target.get("name"),
            "matteTypeCandidate": matte_type,
            "confidence": "name-and-stack-heuristic",
        }
        target["transfer"]["trackMatte"] = "candidate"
        target["transfer"]["trackMatteCandidate"] = {
            "matteLayerOrdinal": layer.get("ordinal"),
            "matteLayerName": layer.get("name"),
            "matteTypeCandidate": matte_type,
            "confidence": "name-and-stack-heuristic",
        }


def scalar_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, list) and value:
        return scalar_numeric_value(value[0])
    if isinstance(value, dict):
        for key in ("value", "index", "parentIndex"):
            if key in value:
                result = scalar_numeric_value(value.get(key))
                if result is not None:
                    return result
    return None


def parent_property_candidate(properties: dict[str, Any] | None) -> dict[str, Any] | None:
    if not properties:
        return None
    for match_name in (
        "ADBE Parent",
        "ADBE Parent Layer",
        "ADBE Layer Parent",
        "ADBE Layer Parent ID",
        "ADBE Layer Parent Index",
    ):
        candidate = first_semantic_by_match(properties, match_name)
        if candidate:
            return candidate
    for node in walk_semantic(properties):
        if node.get("type") != "property":
            continue
        name = str(node.get("name") or "")
        match_name = str(node.get("matchName") or "")
        if name.strip().lower() == "parent" or "parent" in match_name.lower():
            return node
    return None


def layer_parent_summary(properties: dict[str, Any] | None) -> dict[str, Any] | None:
    parent_prop = parent_property_candidate(properties)
    if not parent_prop:
        return None
    summary = property_summary(parent_prop) or {}
    raw_value = summary.get("value")
    numeric_value = scalar_numeric_value(raw_value)
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-parent-layer",
        "matchName": parent_prop.get("matchName"),
        "name": parent_prop.get("name"),
        "raw": summary,
    }
    if numeric_value is not None:
        parent_index = int(numeric_value)
        if parent_index > 0:
            out["parentIndex"] = parent_index
            out["parentIndexBase"] = "ae-layer-index-candidate"
        else:
            out["resolutionStatus"] = "unparented-or-disabled"
    else:
        out["resolutionStatus"] = "raw-parent-value-not-decoded"
    return out


def apply_layer_parent_relationships(layers: list[dict[str, Any]]) -> None:
    layers_by_ordinal = {
        int(layer["ordinal"]): layer
        for layer in layers
        if layer.get("ordinal") is not None
    }
    for layer in layers:
        parent_index = layer.get("parentIndex")
        parent_layer = layer.get("parentLayer")
        if parent_index is None and isinstance(parent_layer, dict):
            parent_index = parent_layer.get("parentIndex")
        if parent_index is None:
            continue
        if not isinstance(parent_layer, dict):
            parent_layer = {
                "decodeStatus": "best-effort-parent-layer",
                "parentIndex": parent_index,
                "parentIndexBase": "ae-layer-index-candidate",
            }
            layer["parentLayer"] = parent_layer
        try:
            resolved_index = int(parent_index)
        except (TypeError, ValueError):
            parent_layer["resolutionStatus"] = "invalid-parent-index"
            continue
        parent = layers_by_ordinal.get(resolved_index)
        if not parent:
            parent_layer["resolutionStatus"] = "parent-layer-not-found"
            continue
        parent_layer.update(
            {
                "parentOrdinal": parent.get("ordinal"),
                "parentName": parent.get("name"),
                "relationshipConfidence": "ae-layer-index-candidate",
                "resolutionStatus": "resolved",
            }
        )
        child_entry = {
            "ordinal": layer.get("ordinal"),
            "name": layer.get("name"),
        }
        children = parent.setdefault("childLayers", [])
        if child_entry not in children:
            children.append(child_entry)


def layer_transform_summary(properties: dict[str, Any] | None) -> dict[str, Any]:
    transform = {}
    for label, match_name in [
        ("anchorPoint", "ADBE Anchor Point"),
        ("position", "ADBE Position"),
        ("scale", "ADBE Scale"),
        ("orientation", "ADBE Orientation"),
        ("rotateX", "ADBE Rotate X"),
        ("rotateY", "ADBE Rotate Y"),
        ("rotateZ", "ADBE Rotate Z"),
        ("opacity", "ADBE Opacity"),
        ("timeRemap", "ADBE Time Remapping"),
    ]:
        prop = first_semantic_by_match(properties, match_name)
        summary = property_summary(prop)
        if summary:
            transform[label] = summary
    return transform


def property_summary_map(properties: dict[str, Any] | None, specs: list[tuple[str, str]]) -> dict[str, Any]:
    out = {}
    for label, match_name in specs:
        value = property_summary(first_semantic_by_match(properties, match_name))
        if value:
            out[label] = value
    return out


def three_d_summary(properties: dict[str, Any] | None, node_kind_value: str, raw_flags: int | None) -> dict[str, Any] | None:
    if not properties:
        return None
    summary: dict[str, Any] = {
        "layerKind": layer_kind_label(node_kind_value),
    }
    if isinstance(raw_flags, int):
        summary["layerIs3DFlagCandidate"] = bool(raw_flags & (1 << 11))
        summary["environmentLayerFlagCandidate"] = bool(raw_flags & (1 << 19))
        summary["lookAtCameraFlagCandidate"] = bool(raw_flags & (1 << 12))
        summary["lookAtPointOfInterestFlagCandidate"] = bool(raw_flags & (1 << 13))

    transform = {}
    for label, match_name in [
        ("position0", "ADBE Position_0"),
        ("position1", "ADBE Position_1"),
        ("orientation", "ADBE Orientation"),
        ("rotateX", "ADBE Rotate X"),
        ("rotateY", "ADBE Rotate Y"),
        ("rotateZ", "ADBE Rotate Z"),
    ]:
        value = property_summary(first_semantic_by_match(properties, match_name))
        if value:
            transform[label] = value
    if transform:
        summary["transform3DStreams"] = transform

    material = {}
    for label, match_name in [
        ("castsShadows", "ADBE Casts Shadows"),
        ("lightTransmission", "ADBE Light Transmission"),
        ("acceptsShadows", "ADBE Accepts Shadows"),
        ("acceptsLights", "ADBE Accepts Lights"),
        ("shadowColor", "ADBE Shadow Color"),
        ("appearsInReflections", "ADBE Appears in Reflections"),
        ("ambient", "ADBE Ambient Coefficient"),
        ("diffuse", "ADBE Diffuse Coefficient"),
        ("specular", "ADBE Specular Coefficient"),
        ("shininess", "ADBE Shininess Coefficient"),
        ("metal", "ADBE Metal Coefficient"),
        ("reflection", "ADBE Reflection Coefficient"),
        ("glossiness", "ADBE Glossiness Coefficient"),
        ("fresnel", "ADBE Fresnel Coefficient"),
        ("transparency", "ADBE Transparency Coefficient"),
        ("transparencyRolloff", "ADBE Transp Rolloff"),
        ("indexOfRefraction", "ADBE Index of Refraction"),
    ]:
        value = property_summary(first_semantic_by_match(properties, match_name))
        if value:
            material[label] = value
    if material:
        summary["materialOptions"] = material

    camera_options = property_summary_map(
        properties,
        [
            ("zoom", "ADBE Camera Zoom"),
            ("depthOfField", "ADBE Camera Depth of Field"),
            ("focusDistance", "ADBE Camera Focus Distance"),
            ("aperture", "ADBE Camera Aperture"),
            ("blurLevel", "ADBE Camera Blur Level"),
            ("irisShape", "ADBE Iris Shape"),
            ("irisRotation", "ADBE Iris Rotation"),
            ("irisRoundness", "ADBE Iris Roundness"),
            ("irisAspectRatio", "ADBE Iris Aspect Ratio"),
            ("irisDiffractionFringe", "ADBE Iris Diffraction Fringe"),
            ("highlightGain", "ADBE Iris Highlight Gain"),
            ("highlightThreshold", "ADBE Iris Highlight Threshold"),
            ("highlightSaturation", "ADBE Iris Highlight Saturation"),
        ],
    )
    if camera_options:
        summary["cameraOptions"] = camera_options

    light_options = property_summary_map(
        properties,
        [
            ("intensity", "ADBE Light Intensity"),
            ("color", "ADBE Light Color"),
            ("coneAngle", "ADBE Light Cone Angle"),
            ("coneFeather", "ADBE Light Cone Feather"),
            ("shadowDarkness", "ADBE Light Shadow Darkness"),
            ("shadowDiffusion", "ADBE Light Shadow Diffusion"),
            ("falloffType", "ADBE Light Falloff Type"),
            ("falloffStart", "ADBE Light Falloff Start"),
            ("falloffDistance", "ADBE Light Falloff Distance"),
        ],
    )
    if light_options:
        summary["lightOptions"] = light_options

    geometry = {}
    for label, match_name in [
        ("bevelDirection", "ADBE Bevel Direction"),
        ("extrusionOptions", "ADBE Extrsn Options Group"),
        ("planeOptions", "ADBE Plane Options Group"),
    ]:
        node = first_semantic_by_match(properties, match_name)
        value = property_summary(node)
        if value:
            geometry[label] = value
        elif node:
            geometry[label] = {"present": True}
    if geometry:
        summary["geometryOptions"] = geometry

    object_type = None
    if camera_options:
        object_type = "camera"
    elif light_options:
        object_type = "light"
    elif node_kind_value == "CLay":
        object_type = "cameraOrCustomView"
    elif node_kind_value == "SLay":
        object_type = "standardView"
    elif node_kind_value == "DLay":
        object_type = "defaultViewOrCamera"
    elif node_kind_value == "Layr":
        object_type = "avLayer"
    elif node_kind_value == "SecL":
        object_type = "markerSection"
    if object_type:
        summary["objectTypeCandidate"] = object_type
    return summary if len(summary) > 1 else None


def mask_summaries(properties: dict[str, Any] | None) -> list[dict[str, Any]]:
    masks = []
    for group in walk_semantic(properties):
        if group.get("type") != "group" or group.get("matchName") != "ADBE Mask Atom":
            continue
        mask: dict[str, Any] = {
            "name": group.get("name"),
            "matchName": group.get("matchName"),
        }
        for label, match_name in [
            ("shape", "ADBE Mask Shape"),
            ("feather", "ADBE Mask Feather"),
            ("opacity", "ADBE Mask Opacity"),
            ("expansion", "ADBE Mask Offset"),
            ("mode", "ADBE Mask Mode"),
            ("inverted", "ADBE Mask Inverted"),
        ]:
            value = property_summary(first_semantic_by_match(group, match_name))
            if value:
                mask[label] = value
        mask["pathDecodeStatus"] = path_decode_status(mask.get("shape"))
        masks.append(mask)
    return masks


def transfer_mode_summary(layer: dict[str, Any]) -> dict[str, Any]:
    hints = layer.get("compositingHints") or {}
    streams = hints.get("postLayerStreams") or []
    stream_flags = [stream.get("flags") or {} for stream in streams]
    stream_hints = transfer_stream_hints(streams)
    non_default_streams = [
        {
            "index": stream.get("index"),
            "streamId": stream.get("streamId"),
            "typeName": stream.get("typeName"),
            "flags": stream.get("flags"),
        }
        for stream in streams
        if any(value not in (0, None) for value in (stream.get("flags") or {}).values())
    ]
    out: dict[str, Any] = {
        "decodeStatus": "best-effort-transfer-mode",
        "blendMode": "normal",
        "blendModeConfidence": "assumed-default-unless-transfer-streams-indicate-otherwise",
        "trackMatte": "none",
    }
    if non_default_streams:
        out["nonDefaultStreamHints"] = non_default_streams
    if stream_hints:
        out["transferStreamHints"] = stream_hints
        summary = transfer_stream_hint_summary(stream_hints)
        if summary:
            out["transferStreamHintSummary"] = summary
        if any("matte" in (hint.get("semanticNameSignals") or []) for hint in stream_hints):
            out["trackMatteStreamCandidate"] = True
        if any(
            signal in {"blend", "transfer", "mode", "opacity"}
            for hint in stream_hints
            for signal in hint.get("semanticNameSignals") or []
        ):
            out["blendModeStreamCandidate"] = True
        decoded_candidates = [
            candidate
            for hint in stream_hints
            for candidate in hint.get("decodedTransferCandidates") or []
        ]
        track_matte_candidates = [
            candidate for candidate in decoded_candidates if candidate.get("kind") == "trackMatte"
        ]
        blend_mode_candidates = [
            candidate for candidate in decoded_candidates if candidate.get("kind") == "blendMode"
        ]
        if track_matte_candidates:
            out["decodedTrackMatteCandidates"] = track_matte_candidates
            non_none_track = [
                candidate for candidate in track_matte_candidates if candidate.get("name") != "none"
            ]
            if non_none_track:
                out["trackMatte"] = non_none_track[0].get("name")
                out["trackMatteConfidence"] = non_none_track[0].get("confidence")
        if blend_mode_candidates:
            out["decodedBlendModeCandidates"] = blend_mode_candidates
            non_normal_blend = [
                candidate
                for candidate in blend_mode_candidates
                if candidate.get("name") not in {"none", "normal"}
            ]
            if non_normal_blend:
                out["blendMode"] = non_normal_blend[0].get("name")
                out["blendModeConfidence"] = non_normal_blend[0].get("confidence")
    if any(flags.get("fifl") for flags in stream_flags):
        out["frameBlendingStreamFlagCandidate"] = True
    return out


def vector_group_transform_summary(group: dict[str, Any]) -> dict[str, Any]:
    transform = {}
    for label, match_name in [
        ("anchorPoint", "ADBE Vector Anchor"),
        ("position", "ADBE Vector Position"),
        ("scale", "ADBE Vector Scale"),
        ("skew", "ADBE Vector Skew"),
        ("skewAxis", "ADBE Vector Skew Axis"),
        ("rotation", "ADBE Vector Rotation"),
        ("opacity", "ADBE Vector Group Opacity"),
    ]:
        summary = property_summary(first_semantic_by_match(group, match_name))
        if summary:
            transform[label] = summary
    return transform


def vector_stroke_summary(group: dict[str, Any]) -> dict[str, Any] | None:
    if not first_semantic_by_match(group, "ADBE Vector Graphic - Stroke"):
        return None
    stroke = {}
    for label, match_name in [
        ("color", "ADBE Vector Stroke Color"),
        ("opacity", "ADBE Vector Stroke Opacity"),
        ("width", "ADBE Vector Stroke Width"),
        ("lineCap", "ADBE Vector Stroke Line Cap"),
        ("lineJoin", "ADBE Vector Stroke Line Join"),
        ("miterLimit", "ADBE Vector Stroke Miter Limit"),
    ]:
        summary = property_summary(first_semantic_by_match(group, match_name))
        if summary:
            stroke[label] = summary
    return stroke or {"status": "stroke-present-properties-not-decoded"}


def vector_match_names(group: dict[str, Any]) -> list[str]:
    names = []
    for node in walk_semantic(group):
        match_name = node.get("matchName")
        if isinstance(match_name, str) and match_name.startswith("ADBE Vector"):
            names.append(match_name)
    return list(dict.fromkeys(names))


def vector_shape_summaries(properties: dict[str, Any] | None) -> list[dict[str, Any]]:
    groups = [
        item
        for item in walk_semantic(properties)
        if item.get("type") == "group" and item.get("matchName") == "ADBE Vector Group"
    ]
    shapes = []
    for group in groups:
        rect_size = first_semantic_by_match(group, "ADBE Vector Rect Size")
        rect_position = first_semantic_by_match(group, "ADBE Vector Rect Position")
        rect_roundness = first_semantic_by_match(group, "ADBE Vector Rect Roundness")
        fill_color = first_semantic_by_match(group, "ADBE Vector Fill Color")
        vector_path = first_semantic_by_match(group, "ADBE Vector Shape")
        bezier_path = first_semantic_by_match(group, "ADBE Vector Shape - Group")
        vector_group = (
            first_semantic_by_match(group, "ADBE Vector Shape - Rect")
            or first_semantic_by_match(group, "ADBE Vector Shape - Ellipse")
            or first_semantic_by_match(group, "ADBE Vector Shape - Star")
            or bezier_path
        )
        if first_semantic_by_match(group, "ADBE Vector Shape - Rect"):
            shape_type = "rectangle"
        elif first_semantic_by_match(group, "ADBE Vector Shape - Ellipse"):
            shape_type = "ellipse"
        elif first_semantic_by_match(group, "ADBE Vector Shape - Star"):
            shape_type = "star"
        elif bezier_path:
            shape_type = "path"
        else:
            shape_type = "unknown-vector"
        shape: dict[str, Any] = {
            "name": group.get("name"),
            "type": shape_type,
        }
        if vector_group:
            shape["shapeMatchName"] = vector_group.get("matchName")
        size = property_summary(rect_size)
        if size:
            shape["size"] = size
        position = property_summary(rect_position)
        if position:
            shape["rectPosition"] = position
        roundness = property_summary(rect_roundness)
        if roundness:
            shape["roundness"] = roundness
        path = property_summary(vector_path or bezier_path)
        if path:
            shape["path"] = path
            shape["pathDecodeStatus"] = path_decode_status(path)
        fill = property_summary(fill_color)
        if fill:
            shape["fill"] = fill
        stroke = vector_stroke_summary(group)
        if stroke:
            shape["stroke"] = stroke
        vector_transform = vector_group_transform_summary(group)
        if vector_transform:
            shape["vectorTransform"] = vector_transform
        match_names = vector_match_names(group)
        if match_names:
            shape["vectorMatchNames"] = match_names
        shapes.append(shape)
    return shapes


def layer_visual_summary(
    layer_name: str | None,
    properties: dict[str, Any] | None,
    file_refs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not properties:
        return None

    text_docs = [node for node in walk_semantic(properties) if node.get("type") == "textDocument"]
    vector_root = first_semantic_by_match(properties, "ADBE Root Vectors Group")
    transform = layer_transform_summary(properties)
    masks = mask_summaries(properties)

    if vector_root:
        out = {
            "kind": "shape",
            "shapes": vector_shape_summaries(properties),
            "transform": transform,
        }
        if masks:
            out["masks"] = masks
        return out

    if text_docs:
        docs = [node.get("textDocument") for node in text_docs if node.get("textDocument")]
        out = {
            "kind": "text",
            "textDocuments": docs,
            "transform": transform,
        }
        if masks:
            out["masks"] = masks
        return out

    has_media_signal = bool(
        layer_name and re.search(r"(image|footage|video|photo|media)", layer_name, re.IGNORECASE)
    )
    if file_refs and has_media_signal:
        out = {
            "kind": "footage",
            "candidateFileReferences": file_refs,
            "transform": transform,
        }
        if masks:
            out["masks"] = masks
        return out

    if transform:
        out = {
            "kind": "layer",
            "name": layer_name,
            "transform": transform,
        }
        if masks:
            out["masks"] = masks
        return out
    if masks:
        return {"kind": "layer", "name": layer_name, "masks": masks}
    return None


def layer_sidecar_chunk_summary(node: dict[str, Any]) -> dict[str, Any]:
    data = bdata_bytes(node)
    entry: dict[str, Any] = {
        "kind": node_kind(node),
        "byteLength": len(data),
    }
    scalar = chunk_scalar(node)
    if scalar is not None:
        entry["scalar"] = scalar
    strings = [value for value in node_strings(node) if value and value not in ("Utf8", "-_0_/-")]
    if strings:
        entry["strings"] = strings[:10]
        if node_kind(node) in {"CIFO", "CIF2"}:
            override = text_style_override_from_sidecar_strings(strings)
            if override:
                entry["textStyleOverride"] = override
    if data and "scalar" not in entry:
        entry["data"] = chunk_data_summary(node, max_values=12)
    return entry


def text_style_override_from_sidecar_strings(strings: list[str]) -> dict[str, Any] | None:
    """Decode Essential Graphics text style metadata from CIFO/CIF sidecars."""
    if len(strings) < 2:
        return None
    settings: dict[str, Any] | None = None
    for value in strings:
        if not isinstance(value, str) or "fontFS" not in value:
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            settings = decoded
            break
    if not settings:
        return None

    out: dict[str, Any] = {
        "style": settings,
        "decodeStatus": "essential-graphics-text-style-sidecar",
    }
    if strings:
        out["compName"] = strings[2] if len(strings) > 2 else strings[0]
    if len(strings) > 5:
        out["layerName"] = strings[5] or strings[3]
    elif len(strings) > 3:
        out["layerName"] = strings[3]
    if len(strings) > 7:
        out["sourceText"] = strings[7]
    if len(strings) > 8:
        out["displayText"] = strings[8]
    return out


def decode_layer_post_streams(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not chunks:
        return None
    extras = []
    streams = []
    current: dict[str, Any] | None = None

    for child in chunks:
        kind = node_kind(child)
        entry = layer_sidecar_chunk_summary(child)
        if kind == "fvdv":
            if current:
                streams.append(current)
            current = {
                "index": len(streams) + 1,
                "videoVersion": entry.get("scalar"),
                "chunks": [entry],
            }
            continue
        if current is not None and kind.startswith("f"):
            current["chunks"].append(entry)
            if kind == "ftts":
                current["streamId"] = entry.get("scalar")
            elif kind in {"fitt", "fott"} and entry.get("strings"):
                current["typeName"] = entry["strings"][0]
            continue
        extras.append(entry)

    if current:
        streams.append(current)

    out: dict[str, Any] = {}
    if extras:
        out["extras"] = extras
    if streams:
        out["streams"] = streams
    return out or None


def text_style_overrides_from_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overrides = []
    seen = set()
    for layer in layers:
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


def layer_contexts_by_node(root: dict[str, Any]) -> dict[int, dict[str, Any]]:
    contexts: dict[int, dict[str, Any]] = {}
    for item in [node for node in walk_nodes(root) if node_kind(node) == "Item"]:
        parent_item_name = item_name(item)
        parent_idta = decode_idta(first_child(item, "idta"))
        parent_iide = decode_iide(first_child(item, "iide"))
        parent_cdta = decode_cdta(first_child(item, "cdta"))
        parent_item: dict[str, Any] = {
            "itemId": (parent_idta or {}).get("itemId") or (parent_iide or {}).get("id"),
            "name": parent_item_name,
            "kind": "composition" if parent_cdta else (parent_idta or {}).get("itemType", "unknown"),
        }
        if parent_cdta:
            parent_item["composition"] = parent_cdta
        children = item.get("children", []) or []
        for index, child in enumerate(children):
            if not is_comp_layer_node(child):
                continue
            sidecars = []
            cursor = index + 1
            while cursor < len(children) and not is_comp_layer_node(children[cursor]):
                sidecars.append(children[cursor])
                cursor += 1
            contexts[id(child)] = {
                "layerNodeKind": node_kind(child),
                "parentItemName": parent_item_name,
                "parentItem": parent_item,
                "childIndex": index,
                "postLayerStreams": decode_layer_post_streams(sidecars),
            }
    return contexts


def slim_item_reference(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    out: dict[str, Any] = {
        "itemId": item.get("itemId"),
        "name": item.get("name"),
        "kind": item.get("kind"),
    }
    if item.get("candidateFileReferences"):
        out["candidateFileReferences"] = item["candidateFileReferences"]
    if item.get("sourceMetadata"):
        out["sourceMetadata"] = item["sourceMetadata"]
    if item.get("footage"):
        out["footage"] = item["footage"]
    return out


def semantic_project(root: dict[str, Any]) -> dict[str, Any]:
    layers = []
    auxiliary_layers = []
    file_refs = semantic_file_references(root)
    items = semantic_items(root)
    items_by_id = {item.get("itemId"): item for item in items if item.get("itemId") is not None}
    layer_contexts = layer_contexts_by_node(root)

    def decode_layer(node: dict[str, Any], tree_index: int, ordinal: int) -> dict[str, Any]:
        strings = [value for value in node_strings(node) if value and value not in ("Utf8", "-_0_/-")]
        layer_name = strings[0] if strings else None
        layer_timing = decode_ldta(first_child(node, "ldta"))
        property_root = first_child(node, "tdgp")
        properties = parse_property_group(property_root, layer_timing=layer_timing) if property_root else None
        context = layer_contexts.get(id(node))
        parent_comp = (context or {}).get("parentItem")
        node_kind_value = node_kind(node)
        layer: dict[str, Any] = {
            "ordinal": ordinal,
            "treeIndex": tree_index,
            "name": layer_name,
            "nodeKind": node_kind_value,
            "layerKind": layer_kind_label(node_kind_value),
            "isRenderLayer": is_layer_node(node),
            "timing": layer_timing,
            "visual": layer_visual_summary(layer_name, properties, file_refs),
            "properties": properties,
        }
        effects = effect_summaries(properties)
        if effects:
            layer["effects"] = effects
        expression_dependencies = layer_expression_dependencies(properties)
        if expression_dependencies:
            layer["expressionDependencies"] = expression_dependencies
        layer_animations = animated_properties(properties)
        if layer_animations:
            layer["animatedProperties"] = layer_animations
        mapping = coordinate_mapping(properties, layer_animations, parent_comp)
        if mapping:
            layer["coordinateMapping"] = mapping
        if parent_comp:
            layer["parentComp"] = comp_coordinate_space(parent_comp)
        parent_layer = layer_parent_summary(properties)
        if parent_layer:
            layer["parentLayer"] = parent_layer
            if parent_layer.get("parentIndex") is not None:
                layer["parentIndex"] = parent_layer["parentIndex"]
        source_item = slim_item_reference(items_by_id.get((layer_timing or {}).get("sourceItemId")))
        if source_item:
            layer["sourceItem"] = source_item
        three_d = three_d_summary(properties, node_kind_value, (layer_timing or {}).get("rawLayerFlags"))
        if three_d:
            layer["threeD"] = three_d
        if context:
            layer["context"] = context
        compositing_hints = layer_switch_summary(
            layer_timing,
            node_kind_value,
            layer.get("visual"),
            effects,
            source_item,
            context,
        )
        if compositing_hints:
            layer["compositingHints"] = compositing_hints
            layer["switches"] = compositing_hints
        return layer

    for index, node in enumerate(walk_nodes(root), start=1):
        if not is_comp_layer_node(node):
            continue
        if is_layer_node(node):
            layers.append(decode_layer(node, index, len(layers) + 1))
        else:
            auxiliary_layers.append(decode_layer(node, index, len(auxiliary_layers) + 1))

    apply_expression_evaluations(layers)
    apply_layer_compositing_relationships(layers)
    apply_layer_parent_relationships(layers)
    for layer in layers:
        layer["conversionSummary"] = layer_conversion_summary(layer)
    for layer in auxiliary_layers:
        layer["conversionSummary"] = layer_conversion_summary(layer)

    text_style_overrides = text_style_overrides_from_layers(layers + auxiliary_layers)

    out = {
        "format": root.get("format"),
        "path": root.get("path"),
        "items": items,
        "layers": layers,
        "auxiliaryLayers": auxiliary_layers,
        "graph": project_graph(items, layers),
        "fileReferences": file_refs,
    }
    if text_style_overrides:
        out["textStyleOverrides"] = text_style_overrides
    return out


def summarize(root: dict[str, Any]) -> dict[str, Any]:
    nodes = walk_nodes(root)
    chunk_counts = Counter(str(node.get("id")) for node in nodes if node.get("id"))
    list_counts = Counter(str(node.get("listType")) for node in nodes if node.get("listType"))
    strings = []
    seen = set()
    for node in nodes:
        for value in node.get("strings", []) or []:
            if value not in seen:
                seen.add(value)
                strings.append(value)
        if "string" in node and node["string"] not in seen:
            seen.add(node["string"])
            strings.append(node["string"])
    return {
        "nodeCount": len(nodes),
        "chunkCounts": dict(chunk_counts.most_common()),
        "listCounts": dict(list_counts.most_common()),
        "stringCount": len(strings),
        "strings": strings,
        "layers": semantic_layers(root),
        "items": semantic_items(root),
        "fileReferences": semantic_file_references(root),
    }


def parse_project(path: Path, include_bdata: bool) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".aepx":
        root = parse_aepx(path, include_bdata)
    elif suffix == ".aep":
        root = parse_aep(path)
    else:
        raise ValueError("Input must be .aep or .aepx")
    root["summary"] = summarize(root)
    return root


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native After Effects .aep/.aepx chunk extractor."
    )
    parser.add_argument("input", help="Input .aep or .aepx project")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument(
        "--no-bdata",
        action="store_true",
        help="For .aepx, omit raw bdata hex from output while keeping sizes and decoded strings.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the summary, not the full chunk tree.",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Write decoded layer/property structure instead of the raw chunk tree.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    parsed = parse_project(input_path, include_bdata=(not args.no_bdata or args.semantic_only))
    if args.semantic_only:
        output = semantic_project(parsed)
    elif args.summary_only:
        output = parsed["summary"]
    else:
        output = parsed
    text = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote native extraction: {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
