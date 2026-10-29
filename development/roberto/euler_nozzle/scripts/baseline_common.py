"""Case configuration, source fingerprints and strict saved-result serialization."""

from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import yaml

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/nozzle_hyperbolic").rglob("*.py"))}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), allow_nan=False, indent=2), encoding="utf-8")


def config(kind):
    path = STUDY / "cases" / f"{kind}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported baseline schema")
    return data


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    inputs = [*sorted((STUDY / "cases").glob("*.yaml")),
              *sorted((STUDY / "scripts").glob("*.py")),
              STUDY / "verify_euler_nozzle_steady.py", STUDY / "verify_euler_nozzle_sod.py",
              STUDY.parent / "backpressure_sensitivity/scripts/reference.py",
              STUDY.parent / "backpressure_sensitivity/scripts/run_sweep.py"]
    return dict(created=datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                working_tree_status=git("status", "--short"), source_hashes=source_hashes(),
                input_hashes={p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs},
                versions={name: importlib.metadata.version(name) for name in
                          ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "pyyaml")})


def validate_saved_campaign(report, cfg):
    """Do not overwrite a baseline after changing source or its physical setup."""
    if report["provenance"]["source_hashes"] != source_hashes() or report["config"] != cfg:
        raise RuntimeError("Saved baseline source/config differs; choose a new output directory")
    dependencies=[STUDY/f"verify_euler_nozzle_{report['kind']}.py"]
    if report["kind"]=="steady":
        dependencies += [STUDY.parent/"backpressure_sensitivity/scripts"/name
                         for name in ("reference.py","run_sweep.py")]
    for path in dependencies:
        relative=path.relative_to(ROOT).as_posix()
        if sha256(path.read_bytes()).hexdigest()!=report["provenance"]["input_hashes"][relative]:
            raise RuntimeError(f"Baseline dependency changed: {relative}; choose a new output directory")


def steady_cases(reference, definitions):
    """Use the established independent analytical operating map."""
    for definition in definitions:
        item = dict(definition)
        mode, value = item["condition"], item.get("value")
        if mode == "throat_mach":
            astar = reference.At / reference.area_mach(value)
            ratio = reference.pressure_ratio(reference.mach(reference.Ae / astar, "sub"))
        elif mode == "shock_fraction":
            ratio = reference.shock_data(reference.length * (1 + value) / 2)[0]
        elif mode == "external_midpoint":
            ratio = (reference.shock_exit + reference.adapted) / 2
        elif mode == "adapted_factor":
            ratio = reference.adapted * value
        else:
            ratio = getattr(reference, mode)
        item["back_pressure_ratio"] = float(ratio)
        yield item
