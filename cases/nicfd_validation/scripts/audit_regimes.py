"""Add independent intended-regime checks to existing saved-state audits."""

import json
import numpy as np

from .case import ROOT
from .run import regime_audit, write_json


def main():
    protocol = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
    path = ROOT / protocol["verification"]
    verification = json.loads(path.read_text(encoding="utf-8"))
    rows = {(item["run"], item["regime"], item["cells"]): item for item in verification["stages"]}
    mismatches = 0
    for result in sorted((ROOT / "outputs").glob("run_*/result.json")):
        record = json.loads(result.read_text(encoding="utf-8"))
        for stage in record["stages"]:
            if "fields" not in stage:
                continue
            with np.load(result.parent / stage["fields"]) as data:
                np.testing.assert_allclose(data["Ma_mix"], abs(data["u_mix"]) / data["a_mix"], rtol=2e-15)
                audit = regime_audit(data, stage["regime"])
            key = (str(result.parent.relative_to(ROOT)),stage["regime"],stage["cells"])
            if key not in rows:
                continue
            row = rows[key]
            row["regime_audit"] = audit
            row["fresh_case_success"] = bool(row["fresh_success"] and audit["passed"])
            if row["fresh_success"] and not audit["passed"]:
                mismatches += 1
                print("Numerical root on unintended regime:", row["run"], audit, flush=True)
    verification["numerically_converged_regime_mismatches"] = mismatches
    write_json(path, verification)
    print("Intended-regime audits completed:",len(rows),"states;",mismatches,"converged mismatch(es)")


if __name__ == "__main__":
    main()
