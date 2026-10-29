"""Reassess analytical accuracy from saved fields without changing solve records.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.reassess_accuracy
"""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from .accuracy import ACCURACY_DEFINITION, accuracy_target_met, assessed_errors, target_definition


STUDY = Path(__file__).resolve().parents[1]
CASE_FILE = re.compile(r"^.+?_(?:N)?[1-9][0-9]*\.json$")


def reassess(folder):
    """Write a versioned overlay; parent/child hashes and raw histories stay intact."""
    rows, hashes = [], {}
    for path in sorted(folder.glob("*.json")):
        if not CASE_FILE.fullmatch(path.name):
            continue
        raw = path.read_bytes()
        saved = json.loads(raw)
        hashes[path.name] = sha256(raw).hexdigest()
        for method, run in saved["methods"].items():
            errors = assessed_errors(saved, run)
            rows.append(dict(file=path.name, input_sha256=hashes[path.name],
                case=saved["case"], cells=saved["cells"], method=method,
                definition=target_definition(saved.get("shock_m") is not None),
                recorded_accuracy_target_met=run["analytical_target_met"],
                accuracy_target_met=accuracy_target_met(saved, run),
                Cp_mean_absolute=errors["Cp_mean_absolute"], Cp_max=errors["Cp_max"],
                conservative_L1=errors["conservative_L1"],
                shock_error_cells=errors.get("shock_error_cells")))
    if not rows:
        raise ValueError(f"No saved case records found in {folder}")
    finest = {name:max(row["cells"] for row in rows if row["case"]==name)
              for name in {row["case"] for row in rows}}
    final = [row for row in rows if row["cells"]==finest[row["case"]]]
    report = dict(definition=ACCURACY_DEFINITION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        postprocessing_only=True, numerical_iterations=0,
        original_result_files_unchanged=True, original_stopping_rules_unchanged=True,
        summary=dict(numerical_solutions=len(rows), finest_assessments=len(final),
            finest_passes=sum(row["accuracy_target_met"] for row in final),
            finest_failures=[dict(case=row["case"],method=row["method"],
                Cp_mean_absolute=row["Cp_mean_absolute"], conservative_L1=row["conservative_L1"])
                for row in final if not row["accuracy_target_met"]]),
        cases=rows)
    for name, signature in hashes.items():
        if sha256((folder/name).read_bytes()).hexdigest()!=signature:
            raise RuntimeError(f"Saved case changed during postprocessing: {name}")
    (folder/"accuracy_reassessment.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (folder/"accuracy_reassessment.csv").open("w",newline="",encoding="utf-8") as stream:
        keys=("case","cells","method","recorded_accuracy_target_met","accuracy_target_met",
              "Cp_mean_absolute","Cp_max","conservative_L1","shock_error_cells")
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"],indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=STUDY/"outputs/grid_sequence")
    args=parser.parse_args()
    if not args.folder.is_dir():
        parser.error("Use an existing saved-results directory")
    reassess(args.folder.resolve())


if __name__=="__main__":
    main()
