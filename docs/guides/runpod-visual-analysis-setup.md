# RunPod Visual Analysis Setup

Goal: create a RunPod Serverless endpoint that runs the Resolve MCP visual-analysis worker, attaches the configured RunPod network volume, and writes the final endpoint ID into `.env.local` as `RUNPOD_ANALYSIS_ENDPOINT_ID`.

## Current State

- `RUNPOD_API_KEY` is set in `.env.local`.
- `RUNPOD_NETWORK_VOLUME_ID` is set in `.env.local`.
- Network volume S3 endpoint: `https://s3api-us-nc-1.runpod.io`.
- Datacenter: `US-NC-1`.
- RunPod MCP is installed in Claude Code.
- Existing RunPod endpoints are unrelated image-generation endpoints. Do not reuse them.

## 1. Confirm RunPod Resources

Using the RunPod MCP:

- List network volumes.
- Confirm the volume ID from `.env.local` exists.
- Confirm that volume's datacenter is `US-NC-1`.
- List Serverless endpoints.
- Do not use the existing SDXL or Qwen image-edit endpoints.

## 2. Build And Push Worker Image

The endpoint needs a Docker image that runs:

```bash
python -u examples/runpod_visual_worker.py
```

This repo already includes:

```text
Dockerfile.runpod
examples/runpod_visual_worker.py
```

Build and push the image to a registry RunPod can pull:

```bash
docker build --platform linux/amd64 -f Dockerfile.runpod -t <registry>/resolve-mcp-worker:latest .
docker push <registry>/resolve-mcp-worker:latest
```

Use Docker Hub, GHCR, RunPod Container Registry, or another registry accessible to RunPod.

## 3. Create Serverless Template

Create a RunPod Serverless template:

```json
{
  "name": "Resolve MCP Visual Analysis Template",
  "imageName": "<registry>/resolve-mcp-worker:latest",
  "category": "NVIDIA",
  "containerDiskInGb": 80,
  "dockerEntrypoint": [],
  "dockerStartCmd": [],
  "env": {
    "RUNPOD_VOLUME_MOUNT_PATH": "/runpod-volume",
    "RESOLVE_MCP_OLLAMA_NUM_CTX": "4096"
  },
  "isPublic": false,
  "isServerless": true,
  "ports": [],
  "volumeInGb": 0,
  "volumeMountPath": "/runpod-volume"
}
```

Save the returned `template_id`.

## 4. Create Serverless Endpoint

Create a RunPod Serverless endpoint:

```json
{
  "name": "Resolve MCP Visual Analysis",
  "templateId": "<template_id>",
  "computeType": "GPU",
  "dataCenterIds": ["US-NC-1"],
  "networkVolumeId": "<RUNPOD_NETWORK_VOLUME_ID>",
  "networkVolumeIds": ["<RUNPOD_NETWORK_VOLUME_ID>"],
  "gpuCount": 1,
  "gpuTypeIds": [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA GeForce RTX 4090"
  ],
  "workersMin": 0,
  "workersMax": 5,
  "idleTimeout": 5,
  "executionTimeoutMs": 3600000,
  "flashboot": true,
  "scalerType": "QUEUE_DELAY",
  "scalerValue": 4,
  "vcpuCount": 8
}
```

Save the returned endpoint `id`.

## 5. Update `.env.local`

Set:

```bash
RUNPOD_ANALYSIS_ENDPOINT_ID=<endpoint_id>
RUNPOD_WORKER_IMAGE=<registry>/resolve-mcp-worker:latest
```

Keep these blank or delete them when using network-volume staging:

```bash
RUNPOD_MEDIA_LOCAL_PREFIX=
RUNPOD_MEDIA_URL_PREFIX=
```

## 6. Optional Helper Script

After the worker image is pushed, this repo can create the template and endpoint automatically:

```bash
venv/bin/python scripts/runpod_create_visual_endpoint.py --image <registry>/resolve-mcp-worker:latest
```

Dry-run the payload first:

```bash
venv/bin/python scripts/runpod_create_visual_endpoint.py --dry-run --image <registry>/resolve-mcp-worker:latest
```

The non-dry-run command writes `RUNPOD_ANALYSIS_ENDPOINT_ID=<endpoint_id>` into `.env.local`.

## 7. Test Staging Only

With Resolve open and a media-pool clip available, test the upload path first:

```text
media_analysis(action="runpod_stage_clip", params={"clip_id":"<clip_id>", "dry_run":true})
```

If the paths look correct, run:

```text
media_analysis(action="runpod_stage_clip", params={"clip_id":"<clip_id>", "dry_run":false})
```

Expected staged paths:

```text
s3://<RUNPOD_NETWORK_VOLUME_ID>/resolve-mcp-staging/<project>/<media_id>/<filename>
/runpod-volume/resolve-mcp-staging/<project>/<media_id>/<filename>
```

## 8. Test Visual Analysis

Use one short video clip first:

```text
analyze_clip_visual(
  clip_id="<clip_id>",
  analysis_backend="runpod",
  tier="fast",
  runpod_wait=false
)
```

For faster uploads, especially with large camera originals, use an analysis-only
proxy. This creates a low-resolution MP4 under `~/Resolve_Analysis`, uploads
that proxy to RunPod, and still writes all returned sidecar data and Resolve
metadata to the original media-pool clip:

```text
analyze_clip_visual(
  clip_id="<clip_id>",
  analysis_backend="runpod",
  tier="deep",
  vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
  runpod_use_proxy=true,
  runpod_proxy_width=960,
  runpod_wait=false
)
```

The proxy is not relinked into Resolve and does not replace source media. It is
only an analysis input. The proxy keeps one proxy frame per source frame, so
source-frame event numbers still map back to the original clip.

Then poll and commit:

```text
media_analysis(
  action="runpod_visual_status",
  params={"clip_id":"<clip_id>", "job_id":"<job_id>", "wait":true}
)
```

### Split Fast Video + Deep Keyframes

For the lowest upload cost, run the dense fast tier from an analysis-only 1080p
proxy once, then submit only selected JPEG keyframes for the VLM deep tier:

```text
analyze_clip_visual(
  clip_id="<clip_id>",
  analysis_backend="runpod",
  tier="fast",
  sample_every_n=10,
  object_every_seconds=5,
  runpod_use_proxy=true,
  runpod_proxy_width=1920,
  runpod_wait=false
)
```

Poll the fast job with `runpod_visual_status` until it commits the sidecar. Then
submit the VLM-only pass:

```text
media_analysis(
  action="runpod_visual_deep_keyframes",
  params={
    "clip_id":"<clip_id>",
    "vlm_model":"Qwen/Qwen2.5-VL-7B-Instruct",
    "image_width":1920
  }
)
```

If `max_keyframes` is omitted or `0`, the MCP chooses an adaptive VLM frame
budget:

| Clip length | VLM frames | Purpose |
|---:|---:|---|
| under 90s | 24 | Enough to see gesture/emotion changes without wasting calls |
| 90s-2 min | 32 | Transition band between short and medium clips |
| 2-4 min | 48 | Better full-video summary coverage |
| 5-10 min | 72 | One frame every roughly 8-12s plus action peaks |
| 10+ min | adaptive, capped | Long-form clips should be summarized in sections; current fallback samples roughly every 10s with a cap |
 
You can still override it explicitly:

```text
media_analysis(
  action="runpod_visual_deep_keyframes",
  params={
    "clip_id":"<clip_id>",
    "vlm_model":"Qwen/Qwen2.5-VL-7B-Instruct",
    "max_keyframes":36,
    "image_width":1920
  }
)
```

Poll and merge the VLM result:

```text
media_analysis(
  action="runpod_visual_keyframes_status",
  params={"clip_id":"<clip_id>", "job_id":"<job_id>", "wait":true}
)
```

This second pass uploads only the chosen JPEG frames. JPEGs are extracted with
timestamp seeks from the analysis proxy so long clips do not require decoding
from the beginning of the proxy just to reach later VLM frames. After the
per-frame VLM pass, the worker runs a clip-level summary prompt over all
timestamped observations and writes it to `vlm.clip_summary`:

```json
{
  "clip_summary": "Rich paragraph describing what happens across the clip...",
  "visual_timeline": [
    {"start_seconds": 0, "end_seconds": 12, "summary": "...", "editorial_value": "..."}
  ],
  "important_moments": [
    {"time_seconds": 39.6, "frame": 950, "moment": "...", "why_it_matters": "..."}
  ],
  "editorial_opportunities": ["..."],
  "search_keywords": ["..."],
  "confidence_notes": "..."
}
```

The metadata rollup uses `clip_summary.clip_summary` as the searchable
Description when available, while the sidecar keeps the full timeline and
important moments for AI-edit decisions.

For clips longer than 10 minutes, the VLM summary path switches to a sectioned
summary strategy: summarize roughly 2-minute sections first, then build the
whole-clip summary from those section summaries. This avoids one giant prompt
that loses the middle of a long clip.

## Edit Intelligence Tools

After transcript and visual sidecars exist, the MCP can turn analysis into edit
decisions instead of making the assistant reason over raw JSON.

Build transcript meaning vectors:

```text
media_analysis(
  action="analyze_transcript_embeddings",
  params={"clip_id":"<clip_id>"}
)
```

Search by meaning:

```text
media_analysis(
  action="query_transcript_semantic",
  params={
    "clip_ids":["<clip_id>"],
    "query":"faith under pressure",
    "limit":10
  }
)
```

The initial embedding backend is `hashing-v1`: fast, local, no downloads, and
persisted to `<media_id>_transcript_embeddings.json`. It is intentionally shaped
so a neural sentence-embedding backend can replace it later without changing the
MCP tool contract.

Create a scored AI cut plan:

```text
media_analysis(
  action="plan_ai_cut",
  params={
    "clip_ids":["<clip_id_a>", "<clip_id_b>"],
    "goal":"short punchy talking-head cut",
    "style":"clean punchy talking-head"
  }
)
```

The plan includes:

- `plan_semantics`: declares this is ranked candidate data, not a keep/drop
  decision. The MCP does deterministic scoring; the assistant-editor skill owns
  thresholds, taste, approval rules, and the under-cut bias.
- `analysis_depth` and `inputs_used`: whether the plan came from transcript-only,
  transcript+fast visuals, transcript+deep VLM, and whether embeddings were
  available.
- `clip_roles`: `main_take`, `main_take_with_context`, `establishing`,
  `reaction`, `cutaway`, or `supporting`.
- `selected_ranges`: transcript-backed source-frame ranges worth keeping.
- `punch_ins`: gesture/expression/motion-backed punch-in frames.
- `cutaways`: VLM/timeline-backed context or transition ranges.
- `candidate_counts`: how many candidates existed before the returned top-N
  limits. Low returned counts mean weak evidence, not an editorial rejection.

The second-stage action currently returns an application preview:

```text
media_analysis(
  action="apply_ai_cut_plan",
  params={"plan_id":"<plan_id>"}
)
```

This keeps planning separate from destructive timeline mutation. The next layer
can wire the persisted plan into timeline assembly once the edit behavior is
confirmed.

The verification contract is already present:

```text
media_analysis(
  action="verify_ai_cut_plan",
  params={"plan_id":"<plan_id>"}
)
```

Today it returns a preview because `apply_ai_cut_plan` does not mutate the
timeline yet. When application is implemented, this action should diff the
actual timeline against the selected ranges, punch-ins, and cutaways in the
plan so the assistant can stop/report instead of trusting Resolve blindly.

## Success Criteria

- `.env.local` has a non-empty `RUNPOD_ANALYSIS_ENDPOINT_ID`.
- The RunPod endpoint is attached to the configured network volume.
- The worker reads staged media from `/runpod-volume/...`.
- The MCP receives visual sidecar JSON and writes local sidecar/metadata.
- Workers scale to zero when idle because `workersMin=0`.

## Do Not Do

- Do not use the existing SDXL or Qwen image-edit endpoints.
- Do not set `RUNPOD_MEDIA_LOCAL_PREFIX` / `RUNPOD_MEDIA_URL_PREFIX` for this network-volume path.
- Do not upload source media anywhere except the configured RunPod network volume.
- Do not transcode, proxy, relink, replace, or modify source media.
