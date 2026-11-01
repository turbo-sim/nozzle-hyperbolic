"""Provenance and serialization for source-backed flux studies."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from importlib.resources import files

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
CASES = ("euler_nozzle/shock_35", "euler_nozzle/shock_70",
         "euler_annular/shock35", "euler_annular/shock70")


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return clean(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def case_resources():
    """The exact installed resources used by the canonical case loader."""
    resources = {}
    for case in CASES:
        model, name = case.split("/")
        resources[case] = files("nozzle_hyperbolic").joinpath("models",model,"cases",name+".yaml")
    return resources


def case_hashes():
    return {case: hashlib.sha256(resource.read_bytes()).hexdigest() for case,resource in case_resources().items()}


def client_hashes():
    return {p.name: digest(p) for p in (FAMILY / "shared").glob("*.py")
            if p.name in ("common.py", "assessment.py", "branch_metrics.py", "run.py", "audit.py", "migration.py")}


def git_context():
    return dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                status=subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).splitlines())


def study_path(name):
    candidate=(FAMILY/name).resolve()
    if candidate.parent!=FAMILY.resolve() or not (candidate/"verification_profile.yaml").is_file():
        raise ValueError("Choose a study directory with a verification profile")
    return candidate
