"""Run the adapted-initialization study against the same frozen Euler baseline."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from development.roberto.historical.newton_solve.scripts.source_snapshot import (
    BASELINE_REF,
    ROOT,
    STUDY,
    freeze_source,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs" / "adapted_initialization"
    )
    args, forwarded = parser.parse_known_args()
    source, commit = freeze_source(args.output, args.baseline_ref)
    environment = dict(os.environ, NEWTON_BASELINE_COMMIT=commit)
    environment["PYTHONPATH"] = str(source) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "development.roberto.historical.newton_solve.scripts.investigate_adapted",
        "--output",
        str(args.output.resolve()),
        *forwarded,
    ]
    print(f"Frozen Euler baseline: {commit}; no live source changes.", flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
