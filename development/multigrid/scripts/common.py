"""Small provenance/serialization helpers for the standalone study."""

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "development/multigrid"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float):
        from math import isfinite

        return value if isfinite(value) else None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def identity(value):
    return sha256(
        json.dumps(
            clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def snapshot():
    paths = [
        p
        for p in (ROOT / "src/nozzle_hyperbolic").rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml")
    ]
    src_files = {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    names = (
        "common.py",
        "nozzle_case.py",
        "core.py",
        "aggregation.py",
        "transfers.py",
        "audit.py",
        "run.py",
        "startup.py",
    )
    numerical = [STUDY / "scripts" / name for name in names]
    numerical += [
        STUDY / "settings.json",
        STUDY / "settings_implicit.json",
        STUDY / "settings_fmg.json",
    ]
    study_files = {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in numerical
        if p.exists()
    }
    return {
        "src": {"sha256": identity(src_files), "files": src_files},
        "numerical_study": {"sha256": identity(study_files), "files": study_files},
    }


def archive_snapshot(path, fingerprints):
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for group in fingerprints.values():
            for name in group["files"]:
                archive.write(ROOT / name, name)


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "created_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "platform": os.name,
        "source": snapshot(),
    }


def environment_probe():
    from development.time_benchmarking.scripts.environment import inspect_environment

    state = inspect_environment()
    parent = int(os.environ.get("NOZZLE_MULTIGRID_CONTROLLER_PID", "0"))
    ancestors = {
        int(value)
        for value in os.environ.get("NOZZLE_MULTIGRID_ANCESTOR_PIDS", "").split(",")
        if value
    }
    ancestors.add(parent)
    if "competing_processes" in state:
        state["competing_processes"] = [
            p for p in state["competing_processes"] if p["pid"] not in ancestors
        ]
        state["status"] = (
            "contended"
            if state["competing_processes"]
            else "unverified" if state.get("unreadable_python_pids") else "clean"
        )
        state["excluded_controller_pid"] = parent
    return state
