"""Snapshot current read-only source and run entirely inside this study."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]


def snapshot(output):
    target = output / "source_snapshot" / "src"
    manifest = output / "source_snapshot" / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        for name, digest in recorded.items():
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        return target, recorded
    recorded = {}
    for original in sorted((ROOT / "src" / "nozzle_hyperbolic").rglob("*.py")):
        relative = original.relative_to(ROOT / "src")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = original.read_bytes()
        destination.write_bytes(blob)
        recorded[relative.as_posix()] = hashlib.sha256(blob).hexdigest()
    manifest.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    return target, recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs" / "recommended_cfl"
    )
    args, forwarded = parser.parse_known_args()
    output = args.output.resolve()
    if not output.is_relative_to(STUDY.resolve()):
        raise ValueError("All study outputs must remain inside BC_backpressure")
    output.mkdir(parents=True, exist_ok=True)
    source, hashes = snapshot(output)
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        MPLBACKEND="Agg",
    )
    environment["PYTHONPATH"] = str(source) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    print(
        f"Using {len(hashes)} snapshotted source modules; live source is read-only.",
        flush=True,
    )
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "development.roberto.BC_backpressure.scripts.run_study",
        "--output",
        str(output),
        *forwarded,
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
