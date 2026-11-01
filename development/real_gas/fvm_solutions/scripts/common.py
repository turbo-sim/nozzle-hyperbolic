"""Study-local serialization and source provenance; no solver imports."""
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
BASELINE = STUDY.parent / "co2_nozzle_reference"
OUTPUT = STUDY / "outputs"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(value):
    return sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_manifest():
    directory = REPO / "src"
    return {p.relative_to(REPO).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and
            "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def harness_manifest():
    files = [STUDY / "settings.json", *sorted((STUDY / "scripts").glob("*.py"))]
    return {p.relative_to(STUDY).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}


def provenance():
    return dict(
        created_at=datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in
                  ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "matplotlib", "CoolProp")},
        property_backend="JAXProp FluidBicubic, equilibrium HEOS-generated tables, 100x100",
        source_before=source_manifest(), harness=harness_manifest(),
        initialization="Production solve_grid_sequence: 50→100→200→400→800; numerical prolongation only",
        repetitions=1,
        timing="Single accuracy/robustness campaign. Stage elapsed includes JIT and all retained warm/cold solve attempts, excludes post-solve audits; no speedup claim.",
    )
