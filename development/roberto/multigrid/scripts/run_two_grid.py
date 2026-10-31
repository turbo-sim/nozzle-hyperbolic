"""Two-grid hypothesis: 40 -> 400 intervals, both spatial RMS targets 1e-8.

Uses the unchanged benchmark logic: source nodal convention gives 41/401
sampled points and 39/399 interior unknown cells. The fine mesh matches the
original four-grid benchmark exactly. No FAS or source edits are introduced.
"""

import argparse
import hashlib
import json
from pathlib import Path

from . import run_demo

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]


def main(
    argv=None,
    *,
    intervals=(40, 400),
    folder="two_grid_tight",
    experiment="Two grids with equally tight endpoint residuals"
):
    """Run a separate tight-tolerance hierarchy through the common benchmark."""
    parser = argparse.ArgumentParser(description=experiment)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs" / folder)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=5000)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(STUDY / "outputs"):
        raise ValueError("Keep experimental records under multigrid/outputs")
    if (output / "summary.json").exists():
        raise FileExistsError(
            "Choose a new output directory to preserve the completed dataset"
        )
    shared = [
        ROOT / "development/roberto/historical/newton_solve/scripts/problem.py",
        ROOT / "development/roberto/historical/newton_solve/scripts/pseudo_time.py",
        ROOT / "development/roberto/historical/euler_nozzle/verify_euler_nozzle_steady.py",
    ]
    hashes = lambda: {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in shared
    }
    before = hashes()
    points = [n + 1 for n in intervals]
    run_demo.main(
        [
            "--points",
            *map(str, points),
            "--coarse-tolerance",
            "1e-8",
            "--repeats",
            str(args.repeats),
            "--max-iterations",
            str(args.max_iterations),
            "--output",
            str(output),
        ]
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["metadata"].update(
        experiment=experiment,
        mesh_intervals=list(intervals),
        mesh_count_convention="Intervals between sampled source points; two point slots are algebraic ghosts. Interior unknown cells: "
        + "/".join(str(n - 2) for n in points),
        shared_import_hashes_before=before,
        shared_import_hashes_after=hashes(),
    )
    assert (
        summary["metadata"]["shared_import_hashes_before"]
        == summary["metadata"]["shared_import_hashes_after"]
    )
    run_demo.dump(output / "summary.json", summary)


if __name__ == "__main__":
    main()
