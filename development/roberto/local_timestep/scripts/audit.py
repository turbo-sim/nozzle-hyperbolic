"""Audit retained samples, initial conditions, source identity and standalone assets."""

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]


def main():
    out=STUDY/"outputs/comparison"
    metadata=json.loads((out/"metadata.json").read_text())
    records=[json.loads((out/name).read_text()) for name in json.loads((out/"index.json").read_text())]
    assert len(records)==64
    assert sum(r["phase"]=="measured" for r in records)==48
    assert sum(r["phase"]=="warmup" for r in records)==16
    for case in metadata["case_ids"]:
        for recon in metadata["reconstructions"]:
            for mode in metadata["modes"]:
                selected=[r for r in records if (r["case_id"],r["reconstruction"],r["mode"])==(case,recon,mode)]
                assert len(selected)==8
                for method in metadata["methods"]:
                    rows=[r for r in selected if r["method"]==method]
                    assert sorted(r["repeat"] for r in rows if r["phase"]=="measured")==[1,2,3]
                initial=None
                for row in selected:
                    if "stages" not in row: continue
                    assert [s["cells"] for s in row["stages"]]==([50,100,200,400] if mode=="sequence" else [400])
                    norms=row["stages"][0]["history"][0]["equation_norm"]
                    if initial is None: initial=norms
                    np.testing.assert_array_equal(norms,initial)
                    previous=None
                    for stage in row["stages"]:
                        assert stage["initial_cfl"] == (.2 if previous is None or stage["attempts"] else previous)
                        previous=stage["last_accepted_cfl"] if stage["success"] else None
                        multipliers=np.asarray(stage["final_cell_step_multipliers"])
                        assert len(multipliers)==stage["cells"] and np.min(multipliers)==1.
                        if row["method"]=="global": np.testing.assert_array_equal(multipliers,1.)
                    for key,value in row["total_work"].items(): assert value==sum(s["work"][key] for s in row["stages"])
    current_changes=[]
    with zipfile.ZipFile(out/"source_at_measurement.zip") as archive:
        for path,digest in metadata["source_hashes"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest()==digest
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                current_changes.append(path)
        for path,digest in metadata["study_hashes"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest()==digest
            if Path(path).name in ("run.py","adapter.py","common.py"):
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert json.loads((out/"source_verification.json").read_text())["src_unchanged"]
    assert metadata["implementation_checks"]["passed"]
    from .audit_report import audit_html
    audit_html()
    for path in STUDY.glob("*.md"):
        text=path.read_text(encoding="utf-8")
        assert "\\(" not in text and "\\[" not in text
        for link in re.findall(r'\]\(([^)]+)\)',text):
            assert "://" in link or (path.parent/link).exists(),(path,link)
    result=dict(passed=True,measured_solves=48,excluded_warmups=16,
        recorded_stages=sum(len(r.get("stages",[])) for r in records),
        successful_solves=sum(r["success"] for r in records),
        audited_stages=sum(s["audit"]["passed"] for r in records for s in r.get("stages",[])),
        src_unchanged_during_campaign=True,measurement_source_snapshot_verified=True,
        current_src_changes_since_campaign=current_changes,complete_embedded_evidence=True)
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
