"""Optional RunPod dispatch helpers for remote media analysis.

The Resolve MCP keeps local analysis as the default path. This module only
builds/submits remote jobs when a tool explicitly asks for the RunPod backend.
When configured, it may copy source media to a RunPod network volume through the
S3-compatible API so workers can read it; it never transcodes, proxies, relinks,
or modifies the source file.
"""

from __future__ import annotations

import json
import mimetypes
import os
import ssl
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


RUNPOD_API_BASE = "https://api.runpod.ai/v2"
RUNPOD_REST_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
DEFAULT_RUNPOD_ENDPOINT_ENV = "RUNPOD_ANALYSIS_ENDPOINT_ID"
DEFAULT_RUNPOD_NETWORK_VOLUME_ID_ENV = "RUNPOD_NETWORK_VOLUME_ID"
DEFAULT_RUNPOD_NETWORK_VOLUME_DATACENTER_ENV = "RUNPOD_NETWORK_VOLUME_DATACENTER_ID"
DEFAULT_RUNPOD_NETWORK_VOLUME_ENDPOINT_ENV = "RUNPOD_NETWORK_VOLUME_ENDPOINT_URL"
DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX_ENV = "RUNPOD_VOLUME_REMOTE_PREFIX"
DEFAULT_RUNPOD_VOLUME_MOUNT_PATH_ENV = "RUNPOD_VOLUME_MOUNT_PATH"
DEFAULT_RUNPOD_STAGING_RETENTION_DAYS_ENV = "RUNPOD_STAGING_RETENTION_DAYS"
DEFAULT_RUNPOD_STAGING_RETENTION_DAYS = 14
DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX = "resolve-mcp-staging"
DEFAULT_RUNPOD_VOLUME_MOUNT_PATH = "/runpod-volume"
RUNPOD_TERMINAL_SUCCESS = {"COMPLETED"}
RUNPOD_TERMINAL_FAILURE = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodConfigError(RuntimeError):
    """Raised when the optional RunPod backend is requested but not configured."""


class RunPodJobError(RuntimeError):
    """Raised when a RunPod job fails or returns an unexpected payload."""


def build_runpod_endpoint_base(endpoint_id: Optional[str] = None, endpoint_url: Optional[str] = None) -> str:
    """Return the base endpoint URL without a trailing slash."""

    if endpoint_url:
        base = str(endpoint_url).rstrip("/")
        if base.endswith("/run"):
            base = base[:-4]
        return base
    endpoint = endpoint_id or os.environ.get(DEFAULT_RUNPOD_ENDPOINT_ENV) or os.environ.get("RUNPOD_ENDPOINT_ID")
    if not endpoint:
        raise RunPodConfigError(
            "RunPod analysis requires runpod_endpoint_id or RUNPOD_ANALYSIS_ENDPOINT_ID."
        )
    return f"{RUNPOD_API_BASE}/{endpoint}"


def runpod_api_key(api_key: Optional[str] = None, api_key_env: str = DEFAULT_RUNPOD_API_KEY_ENV) -> str:
    key = api_key or os.environ.get(api_key_env or DEFAULT_RUNPOD_API_KEY_ENV)
    if not key:
        raise RunPodConfigError(
            f"RunPod analysis requires an API key in {api_key_env or DEFAULT_RUNPOD_API_KEY_ENV}."
        )
    return key


def get_runpod_network_volume(
    network_volume_id: str,
    api_key_value: Optional[str] = None,
    *,
    api_key_env: str = DEFAULT_RUNPOD_API_KEY_ENV,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Read RunPod network-volume metadata from the REST API."""

    key = api_key_value or runpod_api_key(api_key_env=api_key_env)
    return _runpod_json_request(
        "GET",
        f"{RUNPOD_REST_API_BASE}/networkvolumes/{network_volume_id}",
        key,
        timeout=timeout,
    )


def runpod_s3_endpoint_for_datacenter(data_center_id: str) -> str:
    data_center = str(data_center_id or "").strip()
    if not data_center:
        raise RunPodConfigError("RunPod network volume upload requires a datacenter id.")
    return f"https://s3api-{data_center.lower()}.runpod.io/"


def resolve_runpod_network_volume_config(
    *,
    network_volume_id: Optional[str] = None,
    data_center_id: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    api_key_value: Optional[str] = None,
    api_key_env: str = DEFAULT_RUNPOD_API_KEY_ENV,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    volume_id = network_volume_id or os.environ.get(DEFAULT_RUNPOD_NETWORK_VOLUME_ID_ENV) or os.environ.get("RUNPOD_VOLUME_ID")
    if not volume_id:
        return None
    data_center = data_center_id or os.environ.get(DEFAULT_RUNPOD_NETWORK_VOLUME_DATACENTER_ENV)
    endpoint = endpoint_url or os.environ.get(DEFAULT_RUNPOD_NETWORK_VOLUME_ENDPOINT_ENV)
    if not data_center and endpoint:
        data_center = _datacenter_from_s3_endpoint(str(endpoint))
    if not data_center:
        volume = get_runpod_network_volume(
            str(volume_id),
            api_key_value=api_key_value,
            api_key_env=api_key_env,
            timeout=timeout,
        )
        data_center = volume.get("dataCenterId") or volume.get("data_center_id")
    if not data_center:
        raise RunPodConfigError(
            "RunPod network volume upload requires RUNPOD_NETWORK_VOLUME_DATACENTER_ID "
            "or a discoverable datacenter from the volume metadata."
        )
    if not endpoint:
        endpoint = runpod_s3_endpoint_for_datacenter(str(data_center))
    return {
        "network_volume_id": str(volume_id),
        "data_center_id": str(data_center),
        "s3_endpoint_url": str(endpoint).rstrip("/") + "/",
        "mount_path": os.environ.get(DEFAULT_RUNPOD_VOLUME_MOUNT_PATH_ENV, DEFAULT_RUNPOD_VOLUME_MOUNT_PATH).rstrip("/"),
        "remote_prefix": os.environ.get(DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX_ENV, DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX).strip("/"),
    }


def runpod_staging_retention_days(value: Optional[Any] = None) -> int:
    raw = value if value not in {None, ""} else os.environ.get(DEFAULT_RUNPOD_STAGING_RETENTION_DAYS_ENV)
    try:
        days = int(raw) if raw not in {None, ""} else DEFAULT_RUNPOD_STAGING_RETENTION_DAYS
    except (TypeError, ValueError):
        days = DEFAULT_RUNPOD_STAGING_RETENTION_DAYS
    return max(0, days)


def runpod_staging_object_key(
    *,
    file_path: str,
    project_name: str,
    media_id: str,
    remote_prefix: Optional[str] = None,
) -> str:
    prefix = (remote_prefix or os.environ.get(DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX_ENV) or DEFAULT_RUNPOD_VOLUME_REMOTE_PREFIX).strip("/")
    project = _safe_component(project_name or "Project")
    media = _safe_component(media_id or Path(file_path).stem)
    filename = _safe_filename(Path(file_path).name)
    return "/".join(part for part in [prefix, project, media, filename] if part)


def runpod_volume_path(object_key: str, mount_path: Optional[str] = None) -> str:
    mount = (mount_path or os.environ.get(DEFAULT_RUNPOD_VOLUME_MOUNT_PATH_ENV) or DEFAULT_RUNPOD_VOLUME_MOUNT_PATH).rstrip("/")
    return f"{mount}/{str(object_key).lstrip('/')}"


def upload_file_to_runpod_network_volume(
    *,
    file_path: str,
    object_key: str,
    network_volume_id: str,
    data_center_id: str,
    endpoint_url: str,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Upload a local file to a RunPod network volume via its S3-compatible API."""

    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except Exception as exc:
        raise RunPodConfigError(
            "RunPod network-volume upload requires boto3 in the MCP Python environment. "
            "Install with: venv/bin/python -m pip install boto3"
        ) from exc

    path = Path(file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise RunPodConfigError(f"Cannot upload missing media file: {file_path}")
    key = str(object_key).lstrip("/")
    access_key = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RunPodConfigError(
            "RunPod network-volume upload requires S3 API credentials: "
            "AWS_ACCESS_KEY_ID/RUNPOD_S3_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY/RUNPOD_S3_SECRET_ACCESS_KEY."
        )
    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=str(data_center_id),
        endpoint_url=str(endpoint_url),
        config=Config(
            retries={"max_attempts": int(os.environ.get("AWS_MAX_ATTEMPTS", "10") or 10), "mode": "standard"},
            read_timeout=int(os.environ.get("RUNPOD_S3_READ_TIMEOUT", "7200") or 7200),
            connect_timeout=int(os.environ.get("RUNPOD_S3_CONNECT_TIMEOUT", "60") or 60),
        ),
    )
    size = path.stat().st_size
    if skip_existing:
        try:
            head = client.head_object(Bucket=network_volume_id, Key=key)
            if int(head.get("ContentLength") or -1) == size:
                return _staging_upload_result(
                    uploaded=False,
                    skipped=True,
                    file_path=str(path),
                    object_key=key,
                    network_volume_id=network_volume_id,
                    endpoint_url=endpoint_url,
                    data_center_id=data_center_id,
                    size=size,
                )
        except ClientError as exc:
            code = str((exc.response or {}).get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
    extra_args: Dict[str, Any] = {}
    content_type, _ = mimetypes.guess_type(str(path))
    if content_type:
        extra_args["ContentType"] = content_type
    if extra_args:
        client.upload_file(str(path), network_volume_id, key, ExtraArgs=extra_args)
    else:
        client.upload_file(str(path), network_volume_id, key)
    return _staging_upload_result(
        uploaded=True,
        skipped=False,
        file_path=str(path),
        object_key=key,
        network_volume_id=network_volume_id,
        endpoint_url=endpoint_url,
        data_center_id=data_center_id,
        size=size,
    )


def stage_file_to_runpod_network_volume(
    *,
    file_path: str,
    project_name: str,
    media_id: str,
    api_key_value: Optional[str] = None,
    network_volume_id: Optional[str] = None,
    data_center_id: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    remote_prefix: Optional[str] = None,
    dry_run: bool = False,
    api_key_env: str = DEFAULT_RUNPOD_API_KEY_ENV,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Stage a local source copy onto a configured RunPod network volume."""

    config = resolve_runpod_network_volume_config(
        network_volume_id=network_volume_id,
        data_center_id=data_center_id,
        endpoint_url=endpoint_url,
        api_key_value=api_key_value,
        api_key_env=api_key_env,
        timeout=timeout,
    )
    if not config:
        return None
    object_key = runpod_staging_object_key(
        file_path=file_path,
        project_name=project_name,
        media_id=media_id,
        remote_prefix=remote_prefix or config.get("remote_prefix"),
    )
    staged_path = runpod_volume_path(object_key, config.get("mount_path"))
    if dry_run:
        return {
            "dry_run": True,
            "would_upload": True,
            "network_volume_id": config["network_volume_id"],
            "data_center_id": config["data_center_id"],
            "s3_endpoint_url": config["s3_endpoint_url"],
            "object_key": object_key,
            "s3_uri": f"s3://{config['network_volume_id']}/{object_key}",
            "volume_path": staged_path,
            "file_url": f"runpod-volume://{object_key}",
        }
    result = upload_file_to_runpod_network_volume(
        file_path=file_path,
        object_key=object_key,
        network_volume_id=config["network_volume_id"],
        data_center_id=config["data_center_id"],
        endpoint_url=config["s3_endpoint_url"],
    )
    result["volume_path"] = staged_path
    result["file_url"] = f"runpod-volume://{object_key}"
    return result


def build_runpod_file_url(
    file_path: str,
    *,
    file_url: Optional[str] = None,
    local_prefix: Optional[str] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Resolve the remote URL a worker can use to read a local media path.

    The explicit URL wins. Otherwise a local-prefix/url-prefix mapping is used,
    either from params or RUNPOD_MEDIA_LOCAL_PREFIX/RUNPOD_MEDIA_URL_PREFIX.
    """

    if file_url:
        return str(file_url)

    local = local_prefix or os.environ.get("RUNPOD_MEDIA_LOCAL_PREFIX")
    remote = url_prefix or os.environ.get("RUNPOD_MEDIA_URL_PREFIX")
    if not local or not remote:
        raise RunPodConfigError(
            "RunPod analysis cannot read local files directly. Provide runpod_file_url, "
            "or set RUNPOD_MEDIA_LOCAL_PREFIX and RUNPOD_MEDIA_URL_PREFIX."
        )

    source = os.path.abspath(os.path.expanduser(str(file_path)))
    local_root = os.path.abspath(os.path.expanduser(str(local)))
    try:
        common = os.path.commonpath([source, local_root])
    except ValueError as exc:
        raise RunPodConfigError(f"Media path is not under RunPod local prefix: {file_path}") from exc
    if common != local_root:
        raise RunPodConfigError(f"Media path is not under RunPod local prefix: {file_path}")

    relative = os.path.relpath(source, local_root)
    encoded_relative = urlparse.quote(relative.replace(os.sep, "/"), safe="/")
    return f"{str(remote).rstrip('/')}/{encoded_relative}"


def build_runpod_visual_input(
    *,
    file_url: str,
    file_path: str,
    media_id: str,
    clip_name: str,
    fps: float,
    duration_frames: int,
    tier: str,
    sample_every_n: int,
    object_every_n: Optional[int],
    object_every_seconds: float,
    batch_size: int,
    proxy_width: int,
    pose_model: str,
    object_model: str,
    expression: bool,
    expression_every_n: int,
    vlm_model: Optional[str],
    vlm_max_keyframes: int,
    staging_retention_days: Optional[int] = None,
    staged_volume_path: Optional[str] = None,
    network_volume_id: Optional[str] = None,
    staging_object_key: Optional[str] = None,
    analysis_proxy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the portable worker input for one source clip."""

    return {
        "job_type": "resolve_clip_visual_analysis",
        "schema_version": 1,
        "media": {
            "media_id": str(media_id),
            "clip_name": str(clip_name),
            "file_url": str(file_url),
            "volume_path": staged_volume_path,
            "network_volume_id": network_volume_id,
            "staging_object_key": staging_object_key,
            "source_file_path": str(file_path),
            "analysis_proxy": analysis_proxy or None,
            "fps": float(fps or 0.0),
            "duration_frames": int(duration_frames or 0),
        },
        "analysis": {
            "tier": str(tier or "fast"),
            "sample_every_n": int(sample_every_n or 10),
            "object_every_n": int(object_every_n) if object_every_n else None,
            "object_every_seconds": float(object_every_seconds or 5.0),
            "batch_size": int(batch_size or 1),
            "proxy_width": int(proxy_width or 640),
            "pose_model": str(pose_model or "yolo11n-pose"),
            "object_model": str(object_model or "yolo11n"),
            "expression": bool(expression),
            "expression_every_n": int(expression_every_n or 2),
            "vlm_model": vlm_model,
            "vlm_max_keyframes": int(vlm_max_keyframes or 0),
        },
        "output": {
            "return_visual_sidecar": True,
            "allow_visual_sidecar_url": True,
        },
        "staging": {
            "retention_days": runpod_staging_retention_days(staging_retention_days),
            "delete_after_analysis": False,
            "note": "Retain staged source copy for short-term re-analysis; cleanup is enforced by the staging backend.",
        },
    }


def submit_runpod_job(
    endpoint_base: str,
    api_key_value: str,
    job_input: Dict[str, Any],
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    payload = {"input": job_input}
    return _runpod_json_request(
        "POST",
        f"{endpoint_base.rstrip('/')}/run",
        api_key_value,
        payload=payload,
        timeout=timeout,
    )


def get_runpod_job_status(
    endpoint_base: str,
    api_key_value: str,
    job_id: str,
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    return _runpod_json_request(
        "GET",
        f"{endpoint_base.rstrip('/')}/status/{job_id}",
        api_key_value,
        timeout=timeout,
    )


def wait_for_runpod_job(
    endpoint_base: str,
    api_key_value: str,
    job_id: str,
    *,
    poll_interval: float = 2.0,
    max_wait_seconds: float = 1800.0,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(max_wait_seconds or 1800.0))
    interval = max(0.5, float(poll_interval or 2.0))
    last_status: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_status = get_runpod_job_status(endpoint_base, api_key_value, job_id, timeout=timeout)
        state = str(last_status.get("status") or "").upper()
        if state in RUNPOD_TERMINAL_SUCCESS:
            return last_status
        if state in RUNPOD_TERMINAL_FAILURE:
            raise RunPodJobError(_runpod_failure_message(last_status))
        time.sleep(interval)
    raise RunPodJobError(f"RunPod job {job_id} did not finish within {int(max_wait_seconds)} seconds.")


def extract_visual_payload_from_runpod(
    status_payload: Dict[str, Any],
    *,
    expected_media_id: Optional[str] = None,
    api_key_value: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Extract a visual sidecar payload from common RunPod worker response shapes."""

    output = status_payload.get("output", status_payload)
    if not isinstance(output, dict):
        raise RunPodJobError("RunPod output was not a JSON object.")
    if output.get("error"):
        raise RunPodJobError(str(output.get("error")))

    payload = _first_dict(
        output.get("visual_sidecar"),
        output.get("visual"),
        (output.get("sidecars") or {}).get("visual") if isinstance(output.get("sidecars"), dict) else None,
    )
    sidecar_url = (
        output.get("visual_sidecar_url")
        or output.get("visual_url")
        or ((output.get("sidecar_urls") or {}).get("visual") if isinstance(output.get("sidecar_urls"), dict) else None)
    )
    if payload is None and sidecar_url:
        payload = fetch_json_url(str(sidecar_url), api_key_value=api_key_value, timeout=timeout)
    if payload is None and _looks_like_visual_payload(output):
        payload = output
    if payload is None:
        raise RunPodJobError("RunPod output did not include a visual sidecar payload or URL.")

    if expected_media_id and str(payload.get("media_id") or "") != str(expected_media_id):
        raise RunPodJobError(
            f"RunPod visual payload media_id mismatch: expected {expected_media_id}, got {payload.get('media_id')}"
        )
    payload.setdefault("analysis_backend", {})
    if isinstance(payload["analysis_backend"], dict):
        payload["analysis_backend"].update({
            "name": "runpod",
            "job_id": status_payload.get("id") or status_payload.get("job_id"),
            "status": status_payload.get("status"),
        })
    return payload


def fetch_json_url(url: str, *, api_key_value: Optional[str] = None, timeout: float = 30.0) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    parsed = urlparse.urlparse(url)
    if api_key_value and parsed.netloc.endswith("runpod.ai"):
        headers["Authorization"] = f"Bearer {api_key_value}"
    request = urlrequest.Request(url, headers=headers, method="GET")
    try:
        with urlrequest.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RunPodJobError(f"Failed to fetch RunPod sidecar JSON: {exc}") from exc


def _runpod_json_request(
    method: str,
    url: str,
    api_key_value: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urlrequest.Request(url, data=body, headers=headers, method=method)
    try:
        with urlrequest.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            raw = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RunPodJobError(f"RunPod HTTP {exc.code}: {detail}") from exc
    except (urlerror.URLError, TimeoutError) as exc:
        raise RunPodJobError(f"RunPod request failed: {exc}") from exc
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RunPodJobError(f"RunPod returned non-JSON response: {raw[:200]}") from exc
    if not isinstance(data, dict):
        raise RunPodJobError("RunPod returned a non-object JSON response.")
    return data


def _first_dict(*values: Any) -> Optional[Dict[str, Any]]:
    for value in values:
        if isinstance(value, dict):
            return value
    return None


def _looks_like_visual_payload(value: Dict[str, Any]) -> bool:
    return bool(value.get("media_id") and ("metadata_rollup" in value or "events" in value or "objects" in value))


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _staging_upload_result(
    *,
    uploaded: bool,
    skipped: bool,
    file_path: str,
    object_key: str,
    network_volume_id: str,
    endpoint_url: str,
    data_center_id: str,
    size: int,
) -> Dict[str, Any]:
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "file_path": file_path,
        "object_key": object_key,
        "network_volume_id": network_volume_id,
        "data_center_id": data_center_id,
        "s3_endpoint_url": endpoint_url,
        "s3_uri": f"s3://{network_volume_id}/{object_key}",
        "size_bytes": int(size),
    }


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value).strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:96] or "item"


def _safe_filename(value: str) -> str:
    name = Path(value).name or "media"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " "} else "_" for ch in name).strip()
    return cleaned or "media"


def _datacenter_from_s3_endpoint(endpoint_url: str) -> Optional[str]:
    parsed = urlparse.urlparse(endpoint_url if "://" in endpoint_url else f"https://{endpoint_url}")
    host = parsed.netloc or parsed.path
    prefix = "s3api-"
    suffix = ".runpod.io"
    if host.startswith(prefix) and host.endswith(suffix):
        return host[len(prefix):-len(suffix)].upper()
    return None


def _runpod_failure_message(payload: Dict[str, Any]) -> str:
    state = payload.get("status") or "FAILED"
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    error = payload.get("error") or output.get("error")
    return f"RunPod job {payload.get('id') or payload.get('job_id') or ''} {state}: {error or 'no error detail'}"
