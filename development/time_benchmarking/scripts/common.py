"""Production campaign identities and exact numerical-runtime inventory."""

from hashlib import sha256
from pathlib import Path
from datetime import datetime
import importlib.metadata
import os
import platform
import subprocess
from zoneinfo import ZoneInfo

from .runtime_support import ROOT, clean, identity, save, source_snapshot

STUDY=Path(__file__).resolve().parents[1]
RUNTIME=(
    "development/time_benchmarking/settings.yaml",
    "development/time_benchmarking/scripts/common.py",
    "development/time_benchmarking/scripts/run.py",
    "development/time_benchmarking/scripts/worker.py",
    "development/time_benchmarking/scripts/runtime_support.py",
    "development/time_benchmarking/scripts/environment.py",
)


def runtime_snapshot():
    files={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in RUNTIME}
    return dict(files=files,sha256=identity(files))


def provenance():
    import jax
    versions={}
    for name in ("jax","jaxlib","jaxprop","numpy","scipy","nurbspy","pyyaml"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    machine=dict(system=platform.system(),release=platform.release(),machine=platform.machine(),
        processor=platform.processor(),python=platform.python_version(),packages=versions,
        jax_backend=jax.default_backend(),jax_devices=[str(d) for d in jax.devices()],
        threads={name:os.environ.get(name) for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    return dict(created_at=datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        machine=machine,machine_sha256=identity(machine),source=source_snapshot(),harness=runtime_snapshot())
