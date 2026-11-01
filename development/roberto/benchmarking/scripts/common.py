"""Serialization and reproducible identities for benchmark campaigns."""

from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from zoneinfo import ZoneInfo

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
SNAPSHOT = ROOT / "legacy/pre_generic_fvm"


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def identity(value):
    return sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_snapshot():
    directory = ROOT / "src/nozzle_hyperbolic"
    files = {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
             for path in sorted(directory.rglob("*"))
             if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts}
    return dict(sha256=identity(files), files=files)


def harness_snapshot():
    names = ("scripts/common.py", "scripts/run.py", "scripts/worker.py", "scripts/legacy_adapter.py", "scripts/environment.py", "settings.yaml")
    files = {name: sha256((STUDY/name).read_bytes()).hexdigest() for name in names}
    return dict(sha256=identity(files), files=files)


def provenance():
    import jax
    versions = {}
    for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "nurbspy", "pyyaml"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    machine = dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                   processor=platform.processor(), python=platform.python_version(), packages=versions,
                   jax_backend=jax.default_backend(), jax_devices=[str(device) for device in jax.devices()],
                   threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    return dict(created_at=datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                machine=machine, machine_sha256=identity(machine), source=source_snapshot(), harness=harness_snapshot(),
                legacy_manifest_sha256=sha256((SNAPSHOT/"source_manifest.json").read_bytes()).hexdigest())
