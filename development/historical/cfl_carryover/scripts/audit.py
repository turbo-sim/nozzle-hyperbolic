"""Audit saved policy wiring, counts, convergence, provenance and report assets."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
OUT = STUDY / "outputs/comparison"


def main():
    names = json.loads((OUT/"index.json").read_text())
    rows = [json.loads((OUT/name).read_text()) for name in names]
    assert len(rows) == 32
    assert sum(r["phase"] == "measured" for r in rows) == 24
    assert sum(r["phase"] == "warmup" for r in rows) == 8
    for row in rows:
        assert row["success"] and row["audits_passed"]
        assert [s["cells"] for s in row["stages"]] == [50,100,200,400]
        previous = None
        for stage in row["stages"]:
            assert stage["audit"]["passed"]
            expected = previous if row["policy"] == "carry" and previous is not None else .2
            assert stage["initial_cfl"] == expected
            assert stage["history"][0]["cfl"] == expected
            accepted = [h for h in stage["history"] if h.get("phase") == "accepted"]
            assert stage["last_accepted_cfl"] == accepted[-1]["cfl"]
            previous = stage["last_accepted_cfl"]
        for key, count in row["total_work"].items():
            assert count == sum(s["work"][key] for s in row["stages"])
    summary = json.loads((OUT/"report_summary.json").read_text())
    assert len(summary) == 4 and all(g["timing_eligible"] for g in summary)
    for case in ("euler_nozzle/shock_35","euler_nozzle/shock_70"):
        for recon in ("constant","muscl_van_leer"):
            selected = [r for r in rows if (r["case_id"],r["reconstruction"]) == (case,recon)]
            assert len(selected) == 8
            initial = selected[0]["stages"][0]["history"][0]["equation_norm"]
            for row in selected:
                np.testing.assert_array_equal(row["stages"][0]["history"][0]["equation_norm"],initial)
                assert row["fallback_count"] == 0
    metadata = json.loads((OUT/"metadata.json").read_text())
    snapshot = OUT/"source_at_measurement.zip"
    changed = []
    with zipfile.ZipFile(snapshot) if snapshot.exists() else nullcontext() as source:
        for path, digest in metadata["source_hashes"].items():
            if path.startswith("src/") or path.endswith("/run.py"):
                content = source.read(path) if snapshot.exists() else (ROOT/path).read_bytes()
                assert hashlib.sha256(content).hexdigest() == digest, path
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
                    changed.append(path)
    from .audit_report import audit_html
    audit_html()
    for path in STUDY.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "\\(" not in text and "\\[" not in text
        for link in re.findall(r'\]\(([^)]+)\)',text):
            assert "://" in link or (path.parent/link).exists(), (path,link)
    result = dict(passed=True, measured_hierarchies=24, excluded_warmups=8,
        audited_mesh_stages=sum(len(r["stages"]) for r in rows), policy_wiring=True,
        identical_cold_initialization=True, numerical_source_snapshot_verified=True,
        current_source_changes_since_measurement=changed,
        complete_embedded_evidence=True)
    (OUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
