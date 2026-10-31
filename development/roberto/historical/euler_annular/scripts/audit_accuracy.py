"""Independently audit the saved-field Cp reassessment without solver imports."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


STUDY=Path(__file__).resolve().parents[1]


def mean_error(faces,actual,reference):
    length=faces[-1]-faces[0]
    return math.fsum((faces[i+1]-faces[i])*abs(actual[i]-reference[i])
                     for i in range(len(actual)))/length


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=STUDY/"outputs/grid_sequence")
    args=parser.parse_args()
    overlay=json.loads((args.folder/"accuracy_reassessment.json").read_text(encoding="utf-8"))
    # Opposite signs must not cancel; graded widths must change the average.
    assert mean_error([0,1,2],[1,-1],[0,0])==1
    assert mean_error([0,1,4],[2,0],[0,0])==0.5
    assert mean_error([0,4],[0.01],[0])==0.01
    seen=set()
    for row in overlay["cases"]:
        identity=(row["file"],row["method"])
        assert identity not in seen,identity
        seen.add(identity)
        path=args.folder/row["file"]
        raw=path.read_bytes()
        assert sha256(raw).hexdigest()==row["input_sha256"],path
        saved=json.loads(raw)
        run=saved["methods"][row["method"]]
        cp=mean_error(saved["m_faces"],run["fields"]["Cp"],saved["reference"]["Cp"])
        assert math.isclose(cp,row["Cp_mean_absolute"],rel_tol=2e-15,abs_tol=1e-18),identity
        passed=(run["errors"]["shock_error_cells"]<=2 if saved.get("shock_m") is not None else
                cp<=0.01 and max(run["errors"]["conservative_L1"])<0.001)
        assert bool(passed)==row["accuracy_target_met"],identity
        assert row["recorded_accuracy_target_met"]==run["analytical_target_met"],identity
    report=dict(passed=True,assessments=len(seen),original_input_hashes_match=True,
        independent_cp_quadrature=True,sign_cancellation_check=True,
        nonuniform_length_weighting_check=True,inclusive_cp_limit_check=True,
        numerical_iterations=0)
    (args.folder/"accuracy_reassessment.audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
