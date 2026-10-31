"""Strict evidence I/O and provenance; no numerical implementation."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "__array__"):
        return clean(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(data), allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprints():
    return {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/nozzle_hyperbolic").rglob("*.py"))
    }


def client_fingerprints():
    paths = [STUDY / "settings.yaml"]
    paths += [
        STUDY / "scripts" / f"{name}.py"
        for name in ("common", "fixtures", "run", "audit")
    ]
    # The canonical Sod reference is included in the production source hashes.
    return {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.exists()
    }


def provenance():
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        platform=platform.platform(),
        precision="float64; jax_enable_x64=True",
        versions={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy")
        },
        source_before=fingerprints(),
        client_before=client_fingerprints(),
    )


def finish(data):
    p = data["provenance"]
    p["source_after"] = fingerprints()
    p["client_after"] = client_fingerprints()
    p["source_unchanged"] = p["source_before"] == p["source_after"]
    p["client_unchanged"] = p["client_before"] == p["client_after"]
