"""MediaPoolItem operations and metadata helpers."""

from src.granular.common import *  # noqa: F401,F403
from src.utils.media_analysis import mark_registry_stale_for_clip as _mark_analysis_registry_stale

resolve = ResolveProxy()


def _invalidate_analysis_registry_for_clip(project, clip, *, reason: str) -> Optional[Dict[str, Any]]:
    """Best-effort: mark cached analysis stale after a Resolve clip replace.

    Called from replace_clip / replace_media_pool_clip / preserve_sub_clip
    after the Resolve API confirms the mutation. Failures here must NEVER
    block the Resolve mutation — we return the result for diagnostics but
    swallow exceptions so registry trouble does not corrupt the user's edit.
    """
    if project is None or clip is None:
        return None
    try:
        project_name = project.GetName() if hasattr(project, "GetName") else None
        project_id = project.GetUniqueId() if hasattr(project, "GetUniqueId") else None
        clip_id = clip.GetUniqueId() if hasattr(clip, "GetUniqueId") else None
        media_id = clip.GetMediaId() if hasattr(clip, "GetMediaId") else None
        source_file = None
        try:
            source_file = clip.GetClipProperty("File Path") if hasattr(clip, "GetClipProperty") else None
        except Exception:
            source_file = None
        return _mark_analysis_registry_stale(
            project_name=project_name,
            project_id=project_id,
            clip_id=clip_id,
            media_id=media_id,
            source_file=source_file,
            reason=reason,
        )
    except Exception as exc:  # noqa: BLE001 — never break a Resolve mutation
        return {"success": False, "error": f"registry invalidation skipped: {type(exc).__name__}: {exc}"}

@mcp.tool(annotations=EXTERNAL_DESTRUCTIVE_TOOL)
def link_proxy_media(clip_name: str, proxy_file_path: str) -> str:
    """Link a proxy media file to a clip.
    
    Args:
        clip_name: Name of the clip to link proxy to
        proxy_file_path: Path to the proxy media file
    """
    pm, current_project = get_current_project()
    if not current_project:
        return "Error: No project currently open"
    
    media_pool = current_project.GetMediaPool()
    if not media_pool:
        return "Error: Failed to get Media Pool"
    
    # Find the clip by name
    clips = get_all_media_pool_clips(media_pool)
    target_clip = None
    
    for clip in clips:
        if clip.GetName() == clip_name:
            target_clip = clip
            break
    
    if not target_clip:
        return f"Error: Clip '{clip_name}' not found in Media Pool"
    
    # Check if file exists
    if not os.path.exists(proxy_file_path):
        return f"Error: Proxy file '{proxy_file_path}' does not exist"
    
    try:
        result = target_clip.LinkProxyMedia(proxy_file_path)
        if result:
            return f"Successfully linked proxy media '{proxy_file_path}' to clip '{clip_name}'"
        else:
            return f"Failed to link proxy media to clip '{clip_name}'"
    except Exception as e:
        return f"Error linking proxy media: {str(e)}"


@mcp.tool()
def unlink_proxy_media(clip_name: str) -> str:
    """Unlink proxy media from a clip.
    
    Args:
        clip_name: Name of the clip to unlink proxy from
    """
    pm, current_project = get_current_project()
    if not current_project:
        return "Error: No project currently open"
    
    media_pool = current_project.GetMediaPool()
    if not media_pool:
        return "Error: Failed to get Media Pool"
    
    # Find the clip by name
    clips = get_all_media_pool_clips(media_pool)
    target_clip = None
    
    for clip in clips:
        if clip.GetName() == clip_name:
            target_clip = clip
            break
    
    if not target_clip:
        return f"Error: Clip '{clip_name}' not found in Media Pool"
    
    try:
        result = target_clip.UnlinkProxyMedia()
        if result:
            return f"Successfully unlinked proxy media from clip '{clip_name}'"
        else:
            return f"Failed to unlink proxy media from clip '{clip_name}'"
    except Exception as e:
        return f"Error unlinking proxy media: {str(e)}"


@mcp.tool()
def replace_clip(clip_name: str, replacement_path: str) -> str:
    """Replace a clip with another media file.
    
    Args:
        clip_name: Name of the clip to be replaced
        replacement_path: Path to the replacement media file
    """
    pm, current_project = get_current_project()
    if not current_project:
        return "Error: No project currently open"
    
    media_pool = current_project.GetMediaPool()
    if not media_pool:
        return "Error: Failed to get Media Pool"
    
    # Find the clip by name
    clips = get_all_media_pool_clips(media_pool)
    target_clip = None
    
    for clip in clips:
        if clip.GetName() == clip_name:
            target_clip = clip
            break
    
    if not target_clip:
        return f"Error: Clip '{clip_name}' not found in Media Pool"
    
    # Check if file exists
    if not os.path.exists(replacement_path):
        return f"Error: Replacement file '{replacement_path}' does not exist"
    
    try:
        result = target_clip.ReplaceClip(replacement_path)
        if result:
            _invalidate_analysis_registry_for_clip(
                current_project, target_clip, reason="replace_clip"
            )
            return f"Successfully replaced clip '{clip_name}' with '{replacement_path}'"
        else:
            return f"Failed to replace clip '{clip_name}'"
    except Exception as e:
        return f"Error replacing clip: {str(e)}"


@mcp.tool()
def transcribe_audio(clip_name: str, language: str = "en-US") -> str:
    """Transcribe audio for a clip.
    
    Args:
        clip_name: Name of the clip to transcribe
        language: Language code for transcription (default: en-US)
    """
    pm, current_project = get_current_project()
    if not current_project:
        return "Error: No project currently open"
    
    media_pool = current_project.GetMediaPool()
    if not media_pool:
        return "Error: Failed to get Media Pool"
    
    # Find the clip by name
    clips = get_all_media_pool_clips(media_pool)
    target_clip = None
    
    for clip in clips:
        if clip.GetName() == clip_name:
            target_clip = clip
            break
    
    if not target_clip:
        return f"Error: Clip '{clip_name}' not found in Media Pool"
    
    try:
        # Resolve's TranscribeAudio takes an optional speaker-detection bool, not
        # a language; language is governed by project settings. Call with no arg.
        result = target_clip.TranscribeAudio()
        if result:
            return f"Successfully started audio transcription for clip '{clip_name}'"
        else:
            return f"Failed to start audio transcription for clip '{clip_name}'"
    except Exception as e:
        return f"Error during audio transcription: {str(e)}"


@mcp.tool()
def clear_transcription(clip_name: str) -> str:
    """Clear audio transcription for a clip.
    
    Args:
        clip_name: Name of the clip to clear transcription from
    """
    pm, current_project = get_current_project()
    if not current_project:
        return "Error: No project currently open"
    
    media_pool = current_project.GetMediaPool()
    if not media_pool:
        return "Error: Failed to get Media Pool"
    
    # Find the clip by name
    clips = get_all_media_pool_clips(media_pool)
    target_clip = None
    
    for clip in clips:
        if clip.GetName() == clip_name:
            target_clip = clip
            break
    
    if not target_clip:
        return f"Error: Clip '{clip_name}' not found in Media Pool"
    
    try:
        result = target_clip.ClearTranscription()
        if result:
            return f"Successfully cleared audio transcription for clip '{clip_name}'"
        else:
            return f"Failed to clear audio transcription for clip '{clip_name}'"
    except Exception as e:
        return f"Error clearing audio transcription: {str(e)}"


@mcp.tool()
def get_clip_metadata(clip_id: str, metadata_type: str = "") -> Dict[str, Any]:
    """Get metadata for a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        metadata_type: Specific metadata key, or empty for all metadata.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    if metadata_type:
        result = clip.GetMetadata(metadata_type)
    else:
        result = clip.GetMetadata()
    return {"clip_id": clip_id, "metadata": result if result else {}}


@mcp.tool()
def set_clip_metadata(clip_id: str, metadata: Dict[str, str]) -> Dict[str, Any]:
    """Set metadata on a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        metadata: Dict of metadata key-value pairs to set.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.SetMetadata(metadata)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_third_party_metadata(clip_id: str, metadata_key: str = "") -> Dict[str, Any]:
    """Get third-party metadata for a clip.

    Args:
        clip_id: Unique ID of the clip.
        metadata_key: Specific key, or empty for all.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    if metadata_key:
        result = clip.GetThirdPartyMetadata(metadata_key)
    else:
        result = clip.GetThirdPartyMetadata()
    return {"clip_id": clip_id, "third_party_metadata": result if result else {}}


@mcp.tool()
def set_clip_third_party_metadata(clip_id: str, metadata: Dict[str, str]) -> Dict[str, Any]:
    """Set third-party metadata on a clip.

    Args:
        clip_id: Unique ID of the clip.
        metadata: Dict of metadata key-value pairs.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.SetThirdPartyMetadata(metadata)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_media_id(clip_id: str) -> Dict[str, Any]:
    """Get the media ID for a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    media_id = clip.GetMediaId()
    return {"clip_id": clip_id, "media_id": media_id}


@mcp.tool()
def add_clip_marker(clip_id: str, frame_id: int, color: str, name: str, note: str = "", duration: int = 1, custom_data: str = "") -> Dict[str, Any]:
    """Add a marker to a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        frame_id: Frame number for the marker.
        color: Marker color (Blue, Cyan, Green, Yellow, Red, Pink, Purple, Fuchsia, Rose, Lavender, Sky, Mint, Lemon, Sand, Cocoa, Cream).
        name: Marker name.
        note: Marker note. Default: empty.
        duration: Marker duration in frames. Default: 1.
        custom_data: Custom data string. Default: empty.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.AddMarker(frame_id, color, name, note, duration, custom_data)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_markers(clip_id: str) -> Dict[str, Any]:
    """Get all markers on a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    markers = clip.GetMarkers()
    return {"clip_id": clip_id, "markers": markers if markers else {}}


@mcp.tool()
def get_clip_marker_by_custom_data(clip_id: str, custom_data: str) -> Dict[str, Any]:
    """Get a marker by its custom data string.

    Args:
        clip_id: Unique ID of the clip.
        custom_data: Custom data string to search for.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    marker = clip.GetMarkerByCustomData(custom_data)
    return {"marker": marker if marker else {}}


@mcp.tool()
def update_clip_marker_custom_data(clip_id: str, frame_id: int, custom_data: str) -> Dict[str, Any]:
    """Update the custom data of a clip marker.

    Args:
        clip_id: Unique ID of the clip.
        frame_id: Frame number of the marker.
        custom_data: New custom data string.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.UpdateMarkerCustomData(frame_id, custom_data)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_marker_custom_data(clip_id: str, frame_id: int) -> Dict[str, Any]:
    """Get the custom data of a clip marker at a specific frame.

    Args:
        clip_id: Unique ID of the clip.
        frame_id: Frame number of the marker.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    data = clip.GetMarkerCustomData(frame_id)
    return {"frame_id": frame_id, "custom_data": data if data else ""}


@mcp.tool()
def delete_clip_markers_by_color(clip_id: str, color: str) -> Dict[str, Any]:
    """Delete all markers of a specific color on a clip.

    Args:
        clip_id: Unique ID of the clip.
        color: Color of markers to delete. Use '' to delete all.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.DeleteMarkersByColor(color)
    return {"success": bool(result)}


@mcp.tool()
def delete_clip_marker_at_frame(clip_id: str, frame_id: int) -> Dict[str, Any]:
    """Delete a marker at a specific frame on a clip.

    Args:
        clip_id: Unique ID of the clip.
        frame_id: Frame number of the marker to delete.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.DeleteMarkerAtFrame(frame_id)
    return {"success": bool(result)}


@mcp.tool()
def delete_clip_marker_by_custom_data(clip_id: str, custom_data: str) -> Dict[str, Any]:
    """Delete a marker by its custom data string.

    Args:
        clip_id: Unique ID of the clip.
        custom_data: Custom data string of the marker to delete.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.DeleteMarkerByCustomData(custom_data)
    return {"success": bool(result)}


@mcp.tool()
def add_clip_flag(clip_id: str, color: str) -> Dict[str, Any]:
    """Add a flag to a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        color: Flag color (Blue, Cyan, Green, Yellow, Red, Pink, Purple, Fuchsia, Rose, Lavender, Sky, Mint, Lemon, Sand, Cocoa, Cream).
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.AddFlag(color)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_flag_list(clip_id: str) -> Dict[str, Any]:
    """Get list of flags on a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    flags = clip.GetFlagList()
    return {"clip_id": clip_id, "flags": flags if flags else []}


@mcp.tool()
def clear_clip_flags(clip_id: str, color: str = "") -> Dict[str, Any]:
    """Clear flags on a clip.

    Args:
        clip_id: Unique ID of the clip.
        color: Specific color to clear, or empty for all colors.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.ClearFlags(color)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_color(clip_id: str) -> Dict[str, Any]:
    """Get the clip color of a Media Pool item.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    color = clip.GetClipColor()
    return {"clip_id": clip_id, "clip_color": color if color else ""}


@mcp.tool()
def set_clip_color(clip_id: str, color: str) -> Dict[str, Any]:
    """Set the clip color of a Media Pool item.

    Args:
        clip_id: Unique ID of the clip.
        color: Color name (Orange, Apricot, Yellow, Lime, Olive, Green, Teal, Navy, Blue, Purple, Violet, Pink, Tan, Beige, Brown, Chocolate).
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.SetClipColor(color)
    return {"success": bool(result)}


@mcp.tool()
def clear_clip_color(clip_id: str) -> Dict[str, Any]:
    """Clear the clip color of a Media Pool item.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.ClearClipColor()
    return {"success": bool(result)}


@mcp.tool()
def set_clip_property(clip_id: str, property_name: str, property_value: str) -> Dict[str, Any]:
    """Set a property on a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        property_name: Property name (e.g. 'Clip Name', 'Comments', 'Description').
        property_value: Value to set.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.SetClipProperty(property_name, property_value)
    return {"success": bool(result)}


@mcp.tool()
def get_clip_property(clip_id: str, property_name: str = "") -> Dict[str, Any]:
    """Get a property of a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        property_name: Property name, or empty for all properties.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    if property_name:
        result = clip.GetClipProperty(property_name)
    else:
        result = clip.GetClipProperty()
    return {"clip_id": clip_id, "property": result if result else {}}


@mcp.tool()
def set_media_pool_clip_name(clip_id: str, new_name: str) -> Dict[str, Any]:
    """Rename a Media Pool clip.

    Args:
        clip_id: Unique ID of the clip.
        new_name: New clip name.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    missing = _requires_method(clip, "SetName", "20.2")
    if missing:
        return missing
    result = clip.SetName(new_name)
    return {"success": bool(result), "name": new_name}


@mcp.tool()
def link_clip_proxy_media(clip_id: str, proxy_path: str) -> Dict[str, Any]:
    """Link proxy media to a clip.

    Args:
        clip_id: Unique ID of the clip.
        proxy_path: Absolute path to the proxy media file.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.LinkProxyMedia(proxy_path)
    return {"success": bool(result)}


@mcp.tool()
def link_clip_full_resolution_media(clip_id: str, full_res_media_path: str) -> Dict[str, Any]:
    """Link full resolution media to a proxy clip.

    Args:
        clip_id: Unique ID of the clip.
        full_res_media_path: Absolute path to the full resolution media file.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    missing = _requires_method(clip, "LinkFullResolutionMedia", "20.0")
    if missing:
        return missing
    result = clip.LinkFullResolutionMedia(full_res_media_path)
    return {"success": bool(result)}


@mcp.tool()
def unlink_clip_proxy_media(clip_id: str) -> Dict[str, Any]:
    """Unlink proxy media from a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.UnlinkProxyMedia()
    return {"success": bool(result)}


@mcp.tool()
def replace_media_pool_clip(clip_id: str, new_file_path: str) -> Dict[str, Any]:
    """Replace a clip with a new media file.

    Args:
        clip_id: Unique ID of the clip to replace.
        new_file_path: Absolute path to the new media file.
    """
    project, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.ReplaceClip(new_file_path)
    response: Dict[str, Any] = {"success": bool(result)}
    if result:
        registry = _invalidate_analysis_registry_for_clip(
            project, clip, reason="replace_media_pool_clip"
        )
        if registry is not None:
            response["analysis_registry_invalidation"] = registry
    return response


@mcp.tool()
def replace_media_pool_clip_preserve_sub_clip(clip_id: str, file_path: str) -> Dict[str, Any]:
    """Replace a clip's underlying media while preserving subclip extents.

    Args:
        clip_id: Unique ID of the clip to replace.
        file_path: Absolute path to the replacement media file.
    """
    project, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    missing = _requires_method(clip, "ReplaceClipPreserveSubClip", "20.0")
    if missing:
        return missing
    result = clip.ReplaceClipPreserveSubClip(file_path)
    response: Dict[str, Any] = {"success": bool(result)}
    if result:
        registry = _invalidate_analysis_registry_for_clip(
            project, clip, reason="replace_media_pool_clip_preserve_sub_clip"
        )
        if registry is not None:
            response["analysis_registry_invalidation"] = registry
    return response


@mcp.tool()
def monitor_clip_growing_file(clip_id: str) -> Dict[str, Any]:
    """Monitor a growing media file for the given Media Pool clip."""
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    missing = _requires_method(clip, "MonitorGrowingFile", "20.0")
    if missing:
        return missing
    result = clip.MonitorGrowingFile()
    return {"success": bool(result)}


@mcp.tool()
def get_clip_unique_id_by_name(clip_name: str) -> Dict[str, Any]:
    """Find a clip by name and return its unique ID.

    Args:
        clip_name: Name of the clip to find.
    """
    _, mp, err = _get_mp()
    if err:
        return err

    def search(folder):
        for clip in (folder.GetClipList() or []):
            if clip.GetName() == clip_name:
                return clip
        for sub in (folder.GetSubFolderList() or []):
            found = search(sub)
            if found:
                return found
        return None

    clip = search(mp.GetRootFolder())
    if clip:
        return {"name": clip.GetName(), "unique_id": clip.GetUniqueId()}
    return {"error": f"Clip '{clip_name}' not found"}


@mcp.tool()
def transcribe_clip_audio(clip_id: str) -> Dict[str, Any]:
    """Transcribe audio for a specific clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.TranscribeAudio()
    return {"success": bool(result)}


@mcp.tool()
def clear_clip_transcription(clip_id: str) -> Dict[str, Any]:
    """Clear transcription for a specific clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.ClearTranscription()
    return {"success": bool(result)}


def _transcription_status(clip) -> str:
    """Best-effort transcription status string ('' if unavailable)."""
    try:
        return clip.GetClipProperty("Transcription Status") or ""
    except Exception:
        return ""


def _transcription_text(clip) -> str:
    """Best-effort transcript text ('' if unavailable)."""
    try:
        return clip.GetClipProperty("Transcription") or ""
    except Exception:
        return ""


# Resolve's "Transcription" clip property is a preview that is cut off with a
# trailing ellipsis for long transcripts; the full text lives in the subtitle
# track. The ellipsis is the reliable truncation signal (the ~699-char cap is
# undocumented and may vary by version).
def _is_truncated(text) -> bool:
    if not text:
        return False
    stripped = text.rstrip()
    return stripped.endswith("…") or stripped.endswith("...")


def _frames_to_tc(frame, fps) -> str:
    """Format a timeline frame number as HH:MM:SS:FF using fps (rounded)."""
    try:
        f = int(round(float(frame)))
        rate = int(round(float(fps))) or 24
    except Exception:
        return ""
    s, ff = divmod(f, rate)
    return "%02d:%02d:%02d:%02d" % (s // 3600, (s // 60) % 60, s % 60, ff)


def _transcript_timecode_lines(project, clip, *, wait_seconds: int = 30) -> Dict[str, Any]:
    """Assemble caption-chunk transcript lines (text + frame-accurate TC).

    Timecodes are timeline-scoped: they come from the current timeline's subtitle
    track, which only covers clips actually on that timeline. Returns a dict with
    either {"lines": [...], "subtitle_track_created": bool, "granularity": ...}
    or {"note": "..."} when timecodes are unavailable. Never raises.
    """
    try:
        tl = project.GetCurrentTimeline() if project else None
    except Exception:
        tl = None
    if not tl:
        return {"note": "No current timeline; timecodes require the clip to be on an open timeline."}

    fps = tl.GetSetting("timelineFrameRate")

    def read_lines():
        out = []
        try:
            count = tl.GetTrackCount("subtitle")
        except Exception:
            count = 0
        for idx in range(1, (count or 0) + 1):
            for item in (tl.GetItemListInTrack("subtitle", idx) or []):
                try:
                    out.append({
                        "text": item.GetName(),
                        "start_tc": _frames_to_tc(item.GetStart(), fps),
                        "end_tc": _frames_to_tc(item.GetEnd(), fps),
                        "start_frame": item.GetStart(),
                    })
                except Exception:
                    continue
        return out

    # If captions already exist, read them without mutating anything.
    existing = read_lines()
    created = False
    if not existing:
        try:
            before = tl.GetTrackCount("subtitle")
        except Exception:
            before = 0
        try:
            ok = tl.CreateSubtitlesFromAudio({})
        except Exception as exc:
            return {"note": f"Could not generate captions for timecodes: {exc}"}
        if not ok:
            return {"note": "CreateSubtitlesFromAudio returned failure; timecodes unavailable."}
        # CreateSubtitlesFromAudio is async — poll until items populate.
        deadline = max(1, int(wait_seconds))
        lines = []
        for _ in range(deadline * 2):  # poll twice/second
            lines = read_lines()
            if lines:
                break
            time.sleep(0.5)
        try:
            after = tl.GetTrackCount("subtitle")
            created = after > before
        except Exception:
            created = True
        if not lines:
            return {
                "note": "Timecode generation pending; try again.",
                "subtitle_track_created": created,
            }
        return {
            "lines": lines,
            "granularity": "caption-chunk",
            "subtitle_track_created": created,
        }

    return {
        "lines": existing,
        "granularity": "caption-chunk",
        "subtitle_track_created": False,
    }


def _join_subtitle_text(tc: Dict[str, Any]) -> str:
    """Join the text of subtitle caption lines (from _transcript_timecode_lines)
    into the full transcript. Returns '' when no lines are present."""
    lines = (tc or {}).get("lines") or []
    parts = [str(ln.get("text", "")).strip() for ln in lines]
    return "\n".join(p for p in parts if p)


def _build_transcript_payload(project, clip, *, with_timecodes: bool, wait_seconds: int) -> Dict[str, Any]:
    """Shared get/get_all body for one clip. Always returns text; replaces a
    truncated property preview with the full subtitle-track text when possible.
    Adds caption lines only when with_timecodes and they are available."""
    status = _transcription_status(clip)
    name = clip.GetName()
    if status != "Transcribed":
        return {
            "name": name,
            "status": status,
            "text": None,
            "note": "Clip is not transcribed. Run transcribe_clip_audio first.",
        }

    text = _transcription_text(clip)        # fast path: clip property
    source = "property"
    truncated = _is_truncated(text)
    tc = None
    lines = None
    note = None

    if truncated or with_timecodes:
        tc = _transcript_timecode_lines(project, clip, wait_seconds=wait_seconds)
        lines = tc.get("lines")
        full = _join_subtitle_text(tc)
        if truncated:
            if full:
                text, source = full, "subtitles"
            else:
                note = tc.get("note", "Full transcript unavailable; showing truncated preview.")

    payload = {"name": name, "status": status, "text": text,
               "source": source, "truncated": truncated}
    if note:
        payload["note"] = note
    if with_timecodes and tc is not None:
        # Spread timecode-level fields: lines, granularity, subtitle_track_created, note.
        if lines is not None:
            payload["lines"] = lines
        for key in ("granularity", "subtitle_track_created"):
            if key in tc:
                payload[key] = tc[key]
        # If tc only has a "note" (e.g. no timeline), propagate it (may override
        # the truncation note set above, but with_timecodes note is more specific).
        if "note" in tc and not note:
            payload["note"] = tc["note"]
        elif "note" in tc and note:
            # Both truncation and timecode have notes — timecode note wins since
            # it directly answers the with_timecodes request.
            payload["note"] = tc["note"]
    return payload


def _resolve_target_clips(project, mp, p: Dict[str, Any]):
    """Resolve a set of media-pool clips from params.

    Selection (first match wins):
      clip_ids: explicit list of media-pool clip UniqueIds
      scope="mediapool": every media-pool clip
      scope="timeline": clips on the current timeline, mapped to their media pool
                        items via GetMediaPoolItem (deduped)
      scope="folder" + folder_name: clips in a named folder (recursive)

    Returns (clips, error_dict_or_None).
    """
    clip_ids = p.get("clip_ids")
    if clip_ids:
        root = mp.GetRootFolder()
        clips = []
        for cid in clip_ids:
            c = _find_clip_by_id(root, cid)
            if c:
                clips.append(c)
        return clips, None

    scope = (p.get("scope") or "").lower()
    if scope in ("mediapool", "media_pool", ""):
        # Only real media (skip timelines/fusion comps) — they have a File Path.
        clips = [c for c in get_all_media_pool_clips(mp)
                 if _safe_clip_property(c, "File Path")]
        return clips, None

    if scope == "timeline":
        tl = project.GetCurrentTimeline() if project else None
        if not tl:
            return None, {"error": "No current timeline for scope='timeline'"}
        seen, clips = set(), []
        for ttype in ("video", "audio"):
            try:
                count = tl.GetTrackCount(ttype)
            except Exception:
                count = 0
            for idx in range(1, (count or 0) + 1):
                for item in (tl.GetItemListInTrack(ttype, idx) or []):
                    try:
                        c = item.GetMediaPoolItem() if hasattr(item, "GetMediaPoolItem") else None
                    except Exception:
                        c = None
                    if not c:
                        continue
                    try:
                        uid = c.GetUniqueId()
                    except Exception:
                        uid = id(c)
                    if uid in seen:
                        continue
                    seen.add(uid)
                    clips.append(c)
        return clips, None

    if scope == "folder":
        folder_name = p.get("folder_name")
        if not folder_name:
            return None, {"error": "folder_name is required for scope='folder'"}
        target = None
        if folder_name.lower() in ("root", "master"):
            target = mp.GetRootFolder()
        else:
            for folder in get_all_media_pool_folders(mp):
                if folder.GetName() == folder_name:
                    target = folder
                    break
        if not target:
            return None, {"error": f"Folder '{folder_name}' not found"}

        def collect(folder):
            out = list(folder.GetClipList() or [])
            for sub in (folder.GetSubFolderList() or []):
                out += collect(sub)
            return out

        clips = [c for c in collect(target) if _safe_clip_property(c, "File Path")]
        return clips, None

    return None, {"error": f"Unknown scope '{scope}'. Use mediapool, timeline, or folder."}


def _safe_clip_property(clip, key):
    try:
        return clip.GetClipProperty(key) or ""
    except Exception:
        return ""


@mcp.tool()
def clip_transcript(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read audio transcriptions from clips.

    The MCP can trigger transcription (transcribe_clip_audio) but previously
    could not read the result. This tool reads it: text by default, plus
    frame-accurate caption-chunk timecodes on request.

    Text comes from the clip property (clip-scoped). Timecodes come from the
    current timeline's subtitle track (timeline-scoped) and only cover clips on
    the open timeline.

    Actions:
      list() -> {clips: [{clip_id, name, status, char_count}], count}
        Every clip whose transcription status is 'Transcribed'.
      get(clip_id, with_timecodes=False, wait_seconds=30)
        -> {name, status, text, lines?, granularity?, subtitle_track_created?}
        with_timecodes=True generates captions on the current timeline if none
        exist (mutates the timeline; async — polls up to wait_seconds).
      get_all(with_timecodes=False, wait_seconds=30)
        -> {transcripts: [{clip_id, name, status, text, ...}], count}
        Pulls text for all transcribed clips. with_timecodes only populates
        lines for clips on the current timeline.
      transcribe(clip_ids=[...] | scope, folder_name?, skip_existing=True,
                 use_speaker_detection?)
        -> {started, skipped, failed, count_started, note}
        Starts transcription (async) on a set of clips. Target by clip_ids or
        scope: 'mediapool' (all clips), 'timeline' (clips on current timeline,
        mapped to media pool items), or 'folder' (+folder_name). skip_existing
        skips already-transcribed clips. Does NOT wait — poll 'status'.
      status(clip_ids=[...] | scope, folder_name?)
        -> {clips: [{clip_id, name, status}], count, transcribed, all_done}
        Transcription status for a set of clips, for polling completion.
    """
    p = params or {}
    project, mp, err = _get_mp()
    if err:
        return err

    if action == "list":
        clips = []
        for clip in get_all_media_pool_clips(mp):
            status = _transcription_status(clip)
            if status == "Transcribed":
                clips.append({
                    "clip_id": clip.GetUniqueId(),
                    "name": clip.GetName(),
                    "status": status,
                    "char_count": len(_transcription_text(clip)),
                })
        return {"clips": clips, "count": len(clips)}

    if action == "get":
        clip_id = p.get("clip_id")
        if not clip_id:
            return {"error": "clip_id is required for action 'get'"}
        clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
        if not clip:
            return {"error": f"Clip {clip_id} not found"}
        return _build_transcript_payload(
            project, clip,
            with_timecodes=bool(p.get("with_timecodes", False)),
            wait_seconds=int(p.get("wait_seconds", 30)),
        )

    if action == "get_all":
        with_tc = bool(p.get("with_timecodes", False))
        wait_seconds = int(p.get("wait_seconds", 30))
        transcripts = []
        for clip in get_all_media_pool_clips(mp):
            if _transcription_status(clip) != "Transcribed":
                continue
            entry = _build_transcript_payload(
                project, clip, with_timecodes=with_tc, wait_seconds=wait_seconds
            )
            entry["clip_id"] = clip.GetUniqueId()
            transcripts.append(entry)
        return {"transcripts": transcripts, "count": len(transcripts)}

    if action == "transcribe":
        clips, terr = _resolve_target_clips(project, mp, p)
        if terr:
            return terr
        skip_existing = bool(p.get("skip_existing", True))
        # TranscribeAudio takes an optional speaker-detection bool (NOT language).
        use_sd = p.get("use_speaker_detection")
        started, skipped, failed = [], [], []
        for clip in clips:
            try:
                name = clip.GetName()
                uid = clip.GetUniqueId()
            except Exception:
                continue
            if skip_existing and _transcription_status(clip) == "Transcribed":
                skipped.append({"clip_id": uid, "name": name, "reason": "already transcribed"})
                continue
            try:
                ok = clip.TranscribeAudio(use_sd) if use_sd is not None else clip.TranscribeAudio()
            except Exception as exc:
                failed.append({"clip_id": uid, "name": name, "error": str(exc)})
                continue
            (started if ok else failed).append(
                {"clip_id": uid, "name": name} if ok
                else {"clip_id": uid, "name": name, "error": "TranscribeAudio returned False"}
            )
        return {
            "started": started,
            "skipped": skipped,
            "failed": failed,
            "count_started": len(started),
            "note": "Transcription runs asynchronously. Poll clip_transcript 'status' until clips are 'Transcribed'.",
        }

    if action == "status":
        clips, terr = _resolve_target_clips(project, mp, p)
        if terr:
            return terr
        out = []
        for clip in clips:
            try:
                out.append({
                    "clip_id": clip.GetUniqueId(),
                    "name": clip.GetName(),
                    "status": _transcription_status(clip),
                })
            except Exception:
                continue
        done = sum(1 for c in out if c["status"] == "Transcribed")
        return {"clips": out, "count": len(out), "transcribed": done,
                "all_done": done == len(out) and len(out) > 0}

    return {"error": f"Unknown action '{action}'. Use list, get, get_all, transcribe, or status."}


@mcp.tool()
def get_clip_audio_mapping(clip_id: str) -> Dict[str, Any]:
    """Get audio mapping for a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    mapping = clip.GetAudioMapping()
    return {"clip_id": clip_id, "audio_mapping": mapping if mapping else ""}


@mcp.tool()
def get_clip_mark_in_out(clip_id: str) -> Dict[str, Any]:
    """Get mark in/out points for a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.GetMarkInOut()
    return {"clip_id": clip_id, "mark_in_out": result if result else {}}


@mcp.tool()
def set_clip_mark_in_out(clip_id: str, mark_in: int, mark_out: int) -> Dict[str, Any]:
    """Set mark in/out points for a clip.

    Args:
        clip_id: Unique ID of the clip.
        mark_in: Mark in frame number.
        mark_out: Mark out frame number.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.SetMarkInOut(mark_in, mark_out)
    return {"success": bool(result)}


@mcp.tool()
def clear_clip_mark_in_out(clip_id: str) -> Dict[str, Any]:
    """Clear mark in/out points for a clip.

    Args:
        clip_id: Unique ID of the clip.
    """
    _, mp, err = _get_mp()
    if err:
        return err
    clip = _find_clip_by_id(mp.GetRootFolder(), clip_id)
    if not clip:
        return {"error": f"Clip {clip_id} not found"}
    result = clip.ClearMarkInOut()
    return {"success": bool(result)}
