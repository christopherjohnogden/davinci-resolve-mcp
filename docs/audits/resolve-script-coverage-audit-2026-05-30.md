# Resolve Script Coverage Audit - 2026-05-30

## Purpose

Audit the MCP server against:

- the bundled DaVinci Resolve scripting API reference
- the installed Blackmagic example scripts
- the repository's maintained API coverage table
- representative public DaVinci Resolve script and MCP repositories

The goal is to distinguish three different kinds of coverage:

1. Public Resolve scripting API method coverage.
2. Higher-level script/workflow coverage.
3. Non-API automation patterns that scripts can do but should not necessarily
   become first-class MCP tools.

## Sources Checked

Local primary sources:

- `docs/reference/resolve_scripting_api.txt`
- `docs/reference/api-coverage.md`
- `src/server.py`
- `src/granular/`
- `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Examples`

Public source repositories cloned into `/tmp/resolve-mcp-audit` for this audit:

- `X-Raym/DaVinci-Resolve-Scripts`
- `diop/davinci-resolve-api`
- `apvlv/davinci-resolve-mcp`
- `DigitalWorkflowCompany/resolve-mcp`
- `barckley75/resolve-mcp` / `resolve-claude-mcp`
- `GuillaumeHullin/davinci-resolve-scripts`
- `jashanmak/Davinci-Resolve-Scripts`
- `Water-zi/DaVinciResolve-Script`
- `ebibibi/DavinciResolveScripts`
- `EricSeastrand/davinci-resolve-scripts`
- `eshaan-mehta/B-Roller`
- `NUROKU/DavinciResolve_VoiceAutoTools_python`
- `thesleepingsage/Davinci-Resolve-Scripting-API`
- `thomjiji/resolve-toolkits`
- `FusionPixelStudio/Davinci-Resolve-Functions-Toolkit`
- `czukowski/fusionscript-stubs`

## Method Coverage Result

The maintained coverage table contains 336 API rows, representing 284 unique
method names after methods shared across classes are deduplicated.

Machine extraction found:

- Local MCP source references all 284 unique public method names from
  `docs/reference/api-coverage.md`.
- External public scripts and installed Blackmagic examples referenced 154 of
  those public API method names.
- No public API method observed in external scripts was absent from the MCP
  source.

Conclusion: there is no evidence from the official coverage table or audited
public script sample that the MCP is missing a public Resolve API method.

## Script Workflow Coverage Result

Public scripts cluster into these workflow families:

- Project/database lifecycle and tree reporting.
- Media import, sorted stringouts, subclips, bin organization, metadata CSV.
- Timeline assembly, range edits, markers, flags, subtitles, and export.
- Render queue setup, render-all-timelines, progress monitoring.
- Grade operations: DRX application, stills, LUTs, node graph toggles.
- Fusion composition import/export, node creation, template editing.
- Audio sync, transcription, subtitle generation, and local Whisper workflows.
- UI wrappers around scripts using Fusion UIManager or Tkinter.
- Platform-specific helpers such as screenshots, keypresses, clipboard, SMB
  mounting, file watchers, Slack notifications, and ffmpeg/ffprobe analysis.

The MCP already covers the underlying public API methods for these families.
However, the audio-sync incident shows that method coverage alone is not enough:
some workflows need deterministic one-call actions so clients such as Claude app
do not reason through the wrong sequence.

## Non-API Patterns Found

These are outside plain Resolve public API coverage:

- Arbitrary code escape hatches: other MCPs expose `execute_python` and
  `execute_lua`.
- Fusion UIManager windows: common in Lua utility scripts, but not a good
  default for headless MCP chat clients.
- Screen capture and UI inspection: macOS-specific and permission-sensitive.
- Keyboard/mouse automation: brittle and page/layout dependent.
- Local transcription stacks: `ffmpeg`, `ffprobe`, Whisper, `mlx-whisper`.
- External integrations: Slack, file watchers, SMB mounting, OpenAI/Gemini
  analysis scripts.
- Source-media transformations: scripts may transcode/extract/convert media;
  this repository must keep source media safe unless explicitly asked.

The MCP intentionally supports some of these through guarded surfaces:

- `script_plugin` can author, install, execute, and run inline Resolve-page
  Lua/Python scripts.
- `media_analysis` uses source-safe ffmpeg/ffprobe/transcription workflows that
  persist sidecar artifacts and write Resolve metadata/markers by default for
  Resolve targets.
- Fusion composition and graph tools expose `AddTool`, connections, settings,
  and timeline-item comp targeting.

## Gaps Are Workflow Gaps, Not API Gaps

Highest-value workflow actions to add or harden next:

1. `timeline.sync_media_pool_audio` - done in v2.27.2.
   - Deterministic whole-Media-Pool audio sync.
   - Recursively collects camera/audio clips, batches one camera at a time, and
     reports Resolve readback.

2. `media_pool.organize_by_type`
   - Deterministic bin layout action for Camera, Audio, Timelines, Stills,
     Graphics, Synced/Compound, and Unused.
   - Should report before/after tree and never delete clips by default.

3. `timeline.create_sorted_stringout`
   - Equivalent to the official sorted-timeline example, with reusable sort
     keys: name, start TC, creation date, clip metadata, folder order.

4. `timeline.create_subclip_stringout`
   - Uses clipInfo dictionaries to append subranges from media pool clips.
   - Should preserve source metadata and report exact source ranges.

5. `render.render_all_timelines_checked`
   - Load preset, set format/codec/settings, add jobs for each timeline, start
     render, poll progress, optionally clear jobs.
   - Must use safe output path policy.

6. `color.apply_drx_to_timeline_checked`
   - Official example applies DRX to all timeline clips. MCP version must
     inspect representative frames first, preserve recoverable grade versions,
     and respect existing creative grades.

7. `markers.import_export_csv`
   - Public scripts often exchange markers/metadata through CSV. A guarded
     import/export pair should cover Media Pool, timeline, and timeline items.

8. `fusion.apply_template_to_clips`
   - Import `.comp` or create node graph from template across selected clips,
     with parameter mapping and readback.

9. `transcription.local_to_subtitles`
   - The MCP has media analysis transcription and Resolve subtitle probes; a
     deterministic one-call action would match public Whisper-to-SRT workflows.

10. `app.screenshot_resolve_window`
    - Useful for Claude-style visual inspection, but macOS permission-sensitive.
      Keep optional and clearly platform-gated.

## What "Absolutely Everything Possible" Cannot Mean

The MCP cannot honestly guarantee coverage of every possible Resolve automation:

- private scripts are not discoverable
- Blackmagic exposes undocumented and version-specific behavior
- some scripts rely on GUI state, keyboard shortcuts, or UIManager windows
- some workflows mutate source media or external files, which this repo must
  refuse unless explicitly requested
- some features require Studio, cloud projects, Dolby Vision/HDR assets, or
  platform-specific dependencies

The defensible target is:

- 100% public API method coverage
- live-tested guardrails around dangerous calls
- deterministic workflow actions for common multi-step tasks
- explicit escape hatches for advanced scripts with clear safety boundaries

## Commands Used

Focused verification after the audio-sync action addition:

```bash
venv/bin/python -m unittest tests.test_audio_fairlight_probe tests.test_v233_helpers
```

Result:

```text
Ran 34 tests in 0.006s
OK
```

Dry-run verification of the new deterministic action against the current
organized Media Pool:

```python
from src import server
server.timeline("sync_media_pool_audio", {"dry_run": True})
```

Result summary:

```json
{
  "success": true,
  "action": "sync_media_pool_audio",
  "batch_by_video": true,
  "plan_count": 3,
  "video_count": 3,
  "audio_candidate_count": 4
}
```

