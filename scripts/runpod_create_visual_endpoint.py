#!/usr/bin/env python3
"""Create a RunPod Serverless endpoint for Resolve MCP visual analysis.

This script assumes the worker image has already been built and pushed to a
container registry that RunPod can pull. It creates a Serverless template,
creates an endpoint attached to the configured network volume, then writes the
endpoint id back into .env.local.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env.local"
REST_BASE = "https://rest.runpod.io/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ENV_PATH), help="Env file to read/update.")
    parser.add_argument("--image", default=None, help="Worker image, e.g. docker.io/user/resolve-mcp-worker:latest.")
    parser.add_argument("--name", default="Resolve MCP Visual Analysis", help="RunPod endpoint/template name.")
    parser.add_argument("--gpu", action="append", dest="gpus", help="GPU type id. Can be repeated.")
    parser.add_argument("--workers-max", type=int, default=3)
    parser.add_argument("--workers-min", type=int, default=0)
    parser.add_argument("--idle-timeout", type=int, default=5)
    parser.add_argument("--execution-timeout-ms", type=int, default=3_600_000)
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--vcpu-count", type=int, default=8)
    parser.add_argument("--scaler-value", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser()
    env = read_env_file(env_file)
    api_key = env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    volume_id = env.get("RUNPOD_NETWORK_VOLUME_ID") or os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    datacenter = env.get("RUNPOD_NETWORK_VOLUME_DATACENTER_ID") or os.environ.get("RUNPOD_NETWORK_VOLUME_DATACENTER_ID")
    image = args.image or env.get("RUNPOD_WORKER_IMAGE") or os.environ.get("RUNPOD_WORKER_IMAGE")

    missing = [
        name
        for name, value in {
            "RUNPOD_API_KEY": api_key,
            "RUNPOD_NETWORK_VOLUME_ID": volume_id,
            "RUNPOD_NETWORK_VOLUME_DATACENTER_ID": datacenter,
            "RUNPOD_WORKER_IMAGE or --image": image,
        }.items()
        if not value
    ]
    if missing:
        print("Missing required config: " + ", ".join(missing), file=sys.stderr)
        return 2

    gpus = args.gpus or split_csv(env.get("RUNPOD_ENDPOINT_GPU_TYPES")) or [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA GeForce RTX 4090",
    ]

    template_payload = {
        "name": f"{args.name} Template",
        "imageName": image,
        "category": "NVIDIA",
        "containerDiskInGb": args.container_disk_gb,
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "env": {
            "RUNPOD_VOLUME_MOUNT_PATH": env.get("RUNPOD_VOLUME_MOUNT_PATH", "/runpod-volume"),
            "RESOLVE_MCP_OLLAMA_NUM_CTX": env.get("RESOLVE_MCP_OLLAMA_NUM_CTX", "4096"),
        },
        "isPublic": False,
        "isServerless": True,
        "ports": [],
        "readme": "Resolve MCP visual-analysis worker.",
        "volumeInGb": 0,
        "volumeMountPath": env.get("RUNPOD_VOLUME_MOUNT_PATH", "/runpod-volume"),
    }

    endpoint_payload = {
        "templateId": "<created-template-id>",
        "computeType": "GPU",
        "dataCenterIds": [datacenter],
        "executionTimeoutMs": args.execution_timeout_ms,
        "flashboot": True,
        "gpuCount": 1,
        "gpuTypeIds": gpus,
        "idleTimeout": args.idle_timeout,
        "name": args.name,
        "networkVolumeId": volume_id,
        "networkVolumeIds": [volume_id],
        "scalerType": "QUEUE_DELAY",
        "scalerValue": args.scaler_value,
        "vcpuCount": args.vcpu_count,
        "workersMax": args.workers_max,
        "workersMin": args.workers_min,
    }

    if args.dry_run:
        print(json.dumps({"template": template_payload, "endpoint": endpoint_payload}, indent=2))
        return 0

    template = runpod_request("POST", "/templates", api_key, template_payload)
    template_id = template.get("id")
    if not template_id:
        raise SystemExit(f"Template creation returned no id: {template}")
    endpoint_payload["templateId"] = template_id
    endpoint = runpod_request("POST", "/endpoints", api_key, endpoint_payload)
    endpoint_id = endpoint.get("id")
    if not endpoint_id:
        raise SystemExit(f"Endpoint creation returned no id: {endpoint}")

    update_env_file(env_file, {"RUNPOD_ANALYSIS_ENDPOINT_ID": str(endpoint_id)})
    print(json.dumps({"template_id": template_id, "endpoint_id": endpoint_id}, indent=2))
    return 0


def runpod_request(method: str, path: str, api_key: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = [
        "curl",
        "-sS",
        "--fail",
        "-X",
        method,
        f"{REST_BASE}{path}",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Accept: application/json",
    ]
    if payload is not None:
        cmd.extend(["-H", "Content-Type: application/json", "--data", json.dumps(payload)])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise SystemExit(proc.stderr.strip() or proc.stdout[:500] or f"RunPod request failed: {method} {path}")
    data = json.loads(proc.stdout or "{}")
    if not isinstance(data, dict):
        raise SystemExit(f"RunPod returned a non-object response: {proc.stdout[:500]}")
    return data


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def update_env_file(path: Path, values: Dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    updated: List[str] = []
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            updated.append(line)
            continue
        key, _ = line.split("=", 1)
        stripped = key.strip()
        if stripped in values:
            updated.append(f"{stripped}={values[stripped]}")
            seen.add(stripped)
        else:
            updated.append(line)
    for key, value in values.items():
        if key not in seen:
            updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n")


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
