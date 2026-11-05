"""Small serialization and provenance helpers; no solver code."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "development/two_fluid_test_v2"
VARIANTS = ("baseline", "inverse_old_sonic", "sonic_only", "both")
CASES = ("subsonic", "supersonic_shock", "supersonic_adapted")
GRIDS = (50, 100, 200, 400)
RECONSTRUCTIONS = ("constant", "muscl_van_albada")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sources():
    physical = [p for p in (ROOT / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    cases = ROOT / "cases/NICFD2026"
    physical += [cases / "scripts" / f for f in ("case.py", "metrics.py")]
    physical += [cases / "protocol.json", cases / "provenance.json"]
    physical += [cases / "two_component" / regime / f for regime in CASES for f in ("settings.yaml", "reference.csv")]
    numerical = [Path(__file__), STUDY / "scripts/numerics.py", STUDY / "scripts/run.py"]
    result = {}
    for name, paths in (("physical", physical), ("study", numerical)):
        files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
        result[name] = dict(sha256=identity(files), files=files)
    return result


def archive(path, source):
    with ZipFile(path, "w", ZIP_DEFLATED) as out:
        for group in source.values():
            for name in group["files"]:
                out.write(ROOT / name, name)


def group_name(regime, reconstruction, cells):
    return f"{regime}__{reconstruction}__{cells}"
