"""Study-local serialization and source provenance; no solver imports."""
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
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


def source_for_flux(metadata, flux):
    """Return the immutable source actually assigned to a method's results."""
    source_id = metadata["source_by_flux"][flux]
    files = metadata["sources"][source_id]["files"]
    if identity(files) != source_id:
        raise RuntimeError(f"Corrupt source manifest for {flux}")
    return files


def execution_source(metadata):
    """Return the production source pinned for new solves."""
    source_id = metadata["execution_source_sha256"]
    files = metadata["sources"][source_id]["files"]
    if identity(files) != source_id:
        raise RuntimeError("Corrupt execution source manifest")
    return files


def require_current_source(metadata, flux=None):
    expected = execution_source(metadata) if flux is None else source_for_flux(metadata, flux)
    if source_manifest() != expected:
        raise RuntimeError(
            "Production source differs from the pinned "
            + (f"{flux} source manifest" if flux else "execution source manifest")
        )


def harness_manifest():
    files = [STUDY / "settings.json", *sorted((STUDY / "scripts").glob("*.py"))]
    return {p.relative_to(STUDY).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}


def provenance():
    files = source_manifest()
    source_id = identity(files)
    snapshot = OUTPUT / "source_A"
    for name, expected in files.items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
        if sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Production source changed while saving its snapshot")
    save(snapshot / "manifest.json", dict(source_sha256=source_id, files=files))
    profile = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    return dict(
        created_at=datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in
                  ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "matplotlib", "CoolProp")},
        property_backend="JAXProp FluidBicubic, equilibrium HEOS-generated tables, 100x100",
        sources={source_id: dict(files=files, snapshot="source_A")},
        source_by_flux={flux: source_id for flux in profile["fluxes"]},
        execution_source_sha256=source_id,
        settings_sha256=identity(profile), harness=harness_manifest(),
        initialization="Production solve_grid_sequence: 50→100→200→400→800; numerical prolongation only",
        repetitions=1,
        timing="Single accuracy/robustness campaign. Stage elapsed includes JIT and all retained warm/cold solve attempts, excludes post-solve audits; no speedup claim.",
    )
