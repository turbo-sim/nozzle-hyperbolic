"""Execute the five published Rusanov grid hierarchies in fresh processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main():
    definitions = json.loads(
        (ROOT / "case_definitions.json").read_text(encoding="utf-8")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", help="study case ID or all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    studies = [
        s for s in definitions["mesh_studies"] if args.case in ("all", s["case_id"])
    ]
    if not studies:
        parser.error("No matching mesh study")
    for study in studies:
        grids = sorted(set(study["cells"]))
        command = [
            sys.executable,
            "-B",
            "-m",
            "cases.TurboExpo2025.scripts.run",
            "--case",
            study["case_id"],
            "--flux",
            study["flux"],
            "--cells",
            str(grids[-1]),
            "--grids",
            ",".join(map(str, grids)),
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
