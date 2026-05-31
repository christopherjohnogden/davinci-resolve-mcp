# MCP Tool Call Integrity Audit - 2026-05-30

## Purpose

Audit the compound MCP tool surface and Resolve API wrapper code for bugs that
method-coverage checks do not catch: hidden action branches, stale advertised
actions, incorrect client parameter coercion, unsafe test discovery behavior,
and workflow actions that chat clients were likely to call incorrectly.

## Scope

Checked:

- compound MCP tool registration in `src/server.py`
- compound action dispatch branches
- Resolve scripting API parity via `scripts/audit_api_parity.py`
- high-risk boolean parameters passed by MCP clients as JSON/string values
- audio-sync workflow actions
- clip-level enable/disable implementation
- unit-test discovery safety for live Resolve scripts
- visible version markers used by ResolveChat and installer surfaces

This was a code-integrity audit, not a full live validation matrix across every
Resolve page, media type, codec, Studio-only feature, cloud project mode, and
platform-specific dependency.

## Findings Fixed

1. Hidden `media_analysis` actions.
   Several implemented state/correction actions were inside an enclosing action
   set that did not include them, so they were effectively unreachable. The
   action guard now includes those branches, the unknown-action help list is
   current, and a static regression test checks this bug class.

2. Boolean string coercion.
   Multiple actions used Python truthiness for client parameters, so `"false"`
   could be treated as true. Added a shared `_param_bool` helper and replaced
   high-risk call sites for dry-run flags, selected scopes, readback toggles,
   recursive scans, render guards, project/archive guards, extension lifecycle
   probes, Fusion readback, color/DRX guards, timeline import/export guards,
   Media Pool ingest helpers, and transcript options.

3. Clip-level disable/readback path.
   Timeline item enable/disable now uses Resolve's clip-level
   `SetClipEnabled()`/`GetClipEnabled()` methods instead of routing `enabled`
   through generic property writes. Bulk item edits also accept both top-level
   `enabled` and `properties.enabled`.

4. Deterministic Media Pool audio sync.
   Added a one-call sync path that collects video/audio candidates from the
   Media Pool, batches one camera clip at a time, normalizes Resolve's enum
   settings, and reports actual Resolve readback fields instead of letting chat
   clients invent a brittle multi-step sequence.

5. Live-test discovery safety.
   Several `test_*.py` scripts were live Resolve probes and were being picked up
   by `unittest discover`. They now skip when imported by discovery and still
   run when executed directly.

6. Unit-test reliability.
   Fixed missing test imports, UTF-8 file reading in static tests, and a marker
   unit test that was accidentally invoking the live destructive archive hook.

7. Visible version markers.
   Updated the manifest/server/install/readme version markers so ResolveChat can
   show whether the corrected MCP build is actually the one being launched.

8. Positioned audio append readback.
   `media_pool.append_to_timeline` no longer reports failure solely because
   Resolve returns a thin audio TimelineItem without a readable UniqueId. The
   wrapper now scans the requested target track/range to recover the actual
   timeline item id, and reports when a matching item was already present before
   the append. `create_timeline_from_clips` also reports video/audio item counts
   and warns when Resolve already included linked/source audio from clip infos.

## Verification

```bash
venv/bin/python -m py_compile src/server.py src/granular/media_pool_item.py tests/test_append_clip_infos_result_handling.py tests/test_compound_action_integrity.py
venv/bin/python -m unittest tests.test_append_clip_infos_result_handling tests.test_compound_action_integrity tests.test_project_lifecycle_probe tests.test_marker_params tests.test_import
venv/bin/python scripts/audit_api_parity.py
git diff --check
```

Observed result: all commands passed.

The API parity audit still reports advisory undocumented calls in Fusion/script
support (`Execute`, `RunScript`, `LoadSettings`, `SaveSettings`,
`GetExpression`, `SetExpression`, `GetData`, `SetData`, `AddKeyword`). Those
are not public Resolve API coverage gaps; they are Fusion/script surfaces that
need live validation when touched.

## Remaining Risk

The repo now has stronger static and unit coverage for the bug classes found in
this audit, but "everything possible" cannot be proven by static inspection.
The remaining honest target is live validation on disposable Resolve projects
for workflows that mutate projects, render, archive, relink, grade, run Fusion
graphs, or depend on Studio/page-specific behavior.
