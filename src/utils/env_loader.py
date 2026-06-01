"""Small .env loader for local Resolve MCP configuration.

This avoids adding python-dotenv as a runtime dependency. Existing process
environment values win, so shell exports and MCP client env blocks remain the
highest-priority configuration source.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional


def load_local_env(
    *,
    start: Optional[Path] = None,
    filenames: Iterable[str] = (".env", ".env.local"),
    override: bool = False,
) -> Dict[str, str]:
    root = _project_root(start)
    loaded: Dict[str, str] = {}
    explicit = os.environ.get("RESOLVE_MCP_ENV_FILE")
    paths = [Path(explicit).expanduser()] if explicit else [root / name for name in filenames]
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for key, value in _parse_env_file(path).items():
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    return loaded


def _project_root(start: Optional[Path]) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "README.md").exists():
            return candidate
    return current


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        values[key] = _clean_value(value)
    return values


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if "#" in value:
        value = value.split("#", 1)[0].rstrip()
    return value
