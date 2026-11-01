"""Configuration, atomic evidence storage and immutable execution identities."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess

import numpy as np

from development.real_gas.scripts.common import (
    clean as _clean, identity as _identity, save as _save, source_manifest,
)

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
OUTPUT = STUDY / "outputs"


def clean(value):
    """Move concrete array diagnostics to the host before JSON conversion."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if hasattr(value, "__array__") and hasattr(value, "shape"):
        value = np.asarray(value)
    return _clean(value)


def save(path, value):
    """Use atomic shared storage after study-local concrete-array conversion."""
    return _save(path, clean(value))


def identity(value):
    return _identity(clean(value))


def profile():
    return json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))


def configure_compilation_cache():
    """Configure this process to reuse only the study's trusted executables.

    This changes compilation/storage policy, never fluid properties, numerical
    states, nonlinear equations or residual stopping rules.
    """
    import jax
    cache = profile()["execution"]["compilation_cache"]
    if cache["policy"] != "study_local_persistent":
        raise ValueError("Expected the declared study-local compilation cache policy")
    directory = (OUTPUT / cache["relative_directory"]).resolve()
    if not directory.is_relative_to(OUTPUT.resolve()) or directory == OUTPUT.resolve():
        raise ValueError("Compilation cache must remain in a subdirectory of this study's outputs")
    threshold = cache["minimum_compile_time_seconds"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Compilation-cache minimum time must be finite and nonnegative")
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", threshold)
    return dict(policy=cache["policy"], directory=str(directory), minimum_compile_time_seconds=threshold)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pin_source():
    """Freeze the actual production and study inputs before executing cases."""
    settings = profile()
    manifest = source_manifest()
    study_files = {p.relative_to(REPO).as_posix(): sha256(p.read_bytes()).hexdigest()
                   for p in [STUDY / "settings.json", *sorted((STUDY / "scripts").glob("*.py"))]
                   if p.name not in ("report.py", "audit_report.py")}
    for relative in (
        "development/real_gas/cases.json", "development/real_gas/settings.json",
        "development/real_gas/scripts/problem.py", "development/real_gas/scripts/thermodynamics.py",
        "development/real_gas/scripts/geometry.py", "development/real_gas/scripts/audit.py",
        "development/real_gas/scripts/common.py", "development/real_gas/scripts/prepare_reference.py",
    ):
        study_files[relative] = sha256((REPO/relative).read_bytes()).hexdigest()
    provenance_file = OUTPUT / "provenance.json"
    if provenance_file.exists():
        previous = load(provenance_file)
        if previous["source_sha256"] != identity(manifest) or previous["settings_sha256"] != identity(settings):
            raise RuntimeError("Source/settings changed; preserve this campaign and select a new output epoch")
        require_source(previous)
        return previous
    snapshot = OUTPUT / "source"
    for name, expected in {**manifest, **study_files}.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, path)
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source changed while freezing execution inputs")
    if source_manifest() != manifest:
        raise RuntimeError("Source changed while freezing execution inputs")
    versions = {}
    for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "plotly"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=identity(manifest),
        settings_sha256=identity(settings), source_files=manifest,
        study_files=study_files, snapshot="source", versions=versions,
        python=platform.python_version(), platform=platform.platform(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        execution=settings["execution"],
        timing_note="Single trajectories including solve/JIT; independent chains may overlap. No flux-speed ranking.",
    )
    save(provenance_file, metadata)
    return metadata


def _input_path(root, relative):
    """Resolve a manifest input without accepting aliases outside its root."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise RuntimeError(f"Invalid execution input path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise RuntimeError(f"Execution input escapes its root: {relative}")
    return path


def _study_relocation(metadata):
    """Allow only the documented directory spelling correction, with byte proof.

    The executed snapshots and their original hashes remain authoritative. The
    three allowed client changes concern launch paths and provenance validation;
    numerical inputs must retain their original bytes at the corrected path.
    """
    recorded = "development/BC_baclpressure"
    if not any(name.startswith(recorded + "/") for name in metadata["study_files"]):
        return None
    path = STUDY / "relocation.json"
    if not path.is_file():
        raise RuntimeError("Historical campaign requires its explicit directory relocation manifest")
    relocation = load(path)
    current = STUDY.relative_to(REPO).as_posix()
    allowed = {"scripts/common.py", "scripts/run.py", "scripts/audit_campaign.py"}
    if (relocation.get("schema_version") != 1
            or relocation.get("recorded_directory") != recorded
            or relocation.get("current_directory") != current
            or current != "development/BC_backpressure"
            or set(relocation.get("changed_files", {})) != allowed):
        raise RuntimeError("Directory relocation must declare only the registered spelling correction and client changes")
    for relative, digests in relocation["changed_files"].items():
        _input_path(STUDY, relative)
        if (set(digests) != {"recorded_sha256", "current_sha256"}
                or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                       for digest in digests.values())
                or metadata["study_files"].get(recorded + "/" + relative) != digests["recorded_sha256"]):
            raise RuntimeError(f"Invalid recorded/current relocation digest pair: {relative}")
    return relocation


def require_source(metadata):
    if source_manifest() != metadata["source_files"]:
        raise RuntimeError("Current production source differs from the pinned campaign source")
    if identity(profile()) != metadata["settings_sha256"]:
        raise RuntimeError("Current study settings differ from the pinned campaign settings")
    snapshot = _input_path(OUTPUT, metadata["snapshot"])
    relocation = _study_relocation(metadata)
    for name, expected in metadata["study_files"].items():
        original = _input_path(snapshot, name)
        if not original.is_file() or sha256(original.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Recorded execution snapshot changed after source freeze: {name}")
        current_name, current_expected = name, expected
        if relocation and name.startswith(relocation["recorded_directory"] + "/"):
            relative = name[len(relocation["recorded_directory"]) + 1:]
            current_name = relocation["current_directory"] + "/" + relative
            if relative in relocation["changed_files"]:
                current_expected = relocation["changed_files"][relative]["current_sha256"]
        current = _input_path(REPO, current_name)
        if not current.is_file() or sha256(current.read_bytes()).hexdigest() != current_expected:
            raise RuntimeError(f"Execution input changed after source freeze: {current_name}")


def chain_id(case, outlet, flux, reconstruction, start):
    return "__".join((case, outlet, flux, reconstruction, start))


def seed_id(family, flux, reconstruction):
    return "__".join((family, flux, reconstruction, "adapted"))


def archive_epoch(name):
    """Preserve a provisional campaign before starting with changed source.

    Both resolved paths must remain beneath this named study. This deliberately
    moves the complete evidence directory rather than deleting failed attempts.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use a simple explicit epoch name")
    source = OUTPUT.resolve()
    destination = (STUDY / "outputs_epochs" / name).resolve()
    study = STUDY.resolve()
    if source != (study/"outputs") or not source.is_relative_to(study) or not destination.is_relative_to(study/"outputs_epochs"):
        raise RuntimeError("Epoch archive paths must remain in the study")
    if not source.is_dir() or destination.exists():
        raise RuntimeError("Expected existing output directory and a new archive destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination
