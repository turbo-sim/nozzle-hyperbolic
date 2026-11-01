"""Atomic evidence, immutable execution inputs and study-local cache policy."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import gzip
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
OUTPUT = STUDY / "outputs"
NUMERICAL_CLIENTS = ("common.py", "problem.py", "audit.py", "execution.py", "worker.py", "run.py", "direct.py")


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if hasattr(value, "__array__"):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def identity(value):
    return sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Stream serialization rather than materializing a second giant JSON string.
    opener = gzip.open if path.suffix == ".gz" else open
    compression = {"compresslevel": 3} if path.suffix == ".gz" else {}
    with opener(temporary, "wt", encoding="utf-8", **compression) as stream:
        json.dump(clean(value), stream, allow_nan=False, separators=(",", ":") if path.suffix == ".gz" else None,
                  indent=None if path.suffix == ".gz" else 2)
        stream.write("\n")
    temporary.replace(path)


def load(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def record_paths(folder):
    """Current records, excluding temporary writes and archived epochs."""
    folder = Path(folder)
    return sorted([*folder.glob("*.json"), *folder.glob("*.json.gz")])


def profile():
    return load(STUDY / "settings.json")


def source_manifest():
    return {path.relative_to(REPO).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted((REPO / "src").rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def client_manifest():
    paths = [STUDY / "settings.json", *(STUDY / "scripts" / name for name in NUMERICAL_CLIENTS)]
    return {path.relative_to(REPO).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}


def pin_source():
    """Freeze actual working-tree bytes; Git HEAD alone is not the source identity."""
    source, clients, settings = source_manifest(), client_manifest(), profile()
    path = OUTPUT / "provenance.json"
    if path.exists():
        old = load(path)
        require_source(old)
        return old
    snapshot = OUTPUT / "source"
    for name, expected in {**source, **clients}.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
        if sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Input changed while taking execution snapshot")
    if source_manifest() != source or client_manifest() != clients:
        raise RuntimeError("Execution input changed while freezing campaign")
    versions = {}
    for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "pysolver-view", "pysolver_view", "plotly"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=identity(source),
                  client_sha256=identity(clients), settings_sha256=identity(settings), source_files=source,
                  study_files=clients, snapshot="source", versions=versions, python=platform.python_version(),
                  platform=platform.platform(),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  execution=settings["execution"],
                  timing_note="Single compile-inclusive trajectories; up to two independent workers. Durations are diagnostic, not a speed ranking.")
    save(path, result)
    return result


def require_source(metadata):
    if source_manifest() != metadata["source_files"]:
        raise RuntimeError("Production source differs from pinned source; preserve this campaign before starting a new epoch")
    if client_manifest() != metadata["study_files"] or identity(profile()) != metadata["settings_sha256"]:
        raise RuntimeError("Numerical client/settings differ from pinned execution inputs")
    for name, expected in {**metadata["source_files"], **metadata["study_files"]}.items():
        original = OUTPUT / metadata["snapshot"] / name
        if not original.is_file() or sha256(original.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen execution snapshot changed: {name}")


def configure_compilation_cache():
    import jax
    controls = profile()["execution"]["compilation_cache"]
    directory = (OUTPUT / controls["relative_directory"]).resolve()
    if controls["policy"] != "study_local_persistent" or not directory.is_relative_to(OUTPUT.resolve()) or directory == OUTPUT.resolve():
        raise ValueError("Cache must remain a trusted subdirectory of this study's outputs")
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", controls["minimum_compile_time_seconds"])
    return controls


def run_id(case, outlet, flux, reconstruction, start, strategy):
    return "__".join((case, outlet, flux, reconstruction, start, strategy))


def seed_id(flux, reconstruction):
    return "__".join((flux, reconstruction))
