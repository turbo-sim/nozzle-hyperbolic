"""Check offline packaging and exact saved-data mappings without flow imports."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import statistics
from zipfile import ZipFile

from development.flux_methods.shared.offline_report import check_inputs
from development.flux_methods.shared.audit_offline_report import Document

STUDY=Path(__file__).resolve().parents[1]


def audit_html():
    manifest=check_inputs(STUDY)
    document=Document()
    document.feed((STUDY/"REPORT.html").read_text(encoding="utf-8"))
    assert not document.external
    assert len(document.ids)==len(set(document.ids))
    required={"report-theme","case","strategy","reconstruction","cells","sample","profile-range","residual-mode",
        "methods-all","methods-none","page-overview","page-explorer","page-discussion","page-protocol",
        "timing-plot","strategy-plot","selected-timing","stage-time","pressure-plot","velocity-plot",
        "residual-plot","cfl-plot","work-plot","local-step-plot","payload","plotly-runtime","report-app"}
    assert required<=set(document.ids),required-set(document.ids)
    data=json.loads(''.join(document.payload))
    assert data["study"]=="local_timestep" and len(data["records"])==64
    assert sum(r["phase"]=="measured" for r in data["records"])==48
    assert data["presentation"]["numerical_cases_rerun"] is False
    assert data["presentation"]["inputs_sha256"]==manifest
    assert json.loads(document.downloads["plotted_data.json"])==data
    for row in data["records"]:
        original=json.loads((STUDY/row["input_file"]).read_text())
        assert {k:v for k,v in row.items() if k not in ("id","input_file")}==original,row["id"]
        for stage in row["stages"]:
            assert len(stage["state"])==stage["cells"]+2
            assert all(len(w)==3 for w in stage["state"])
    for group in data["comparisons"]:
        for method,timing in group["timing"].items():
            values=[next(r for r in data["records"] if r["id"]==id)["elapsed_s"] for id in timing["sample_ids"]]
            assert values==timing["samples_s"]
            assert statistics.mean(values)==timing["mean_s"]
            assert min(values)==timing["min_s"] and max(values)==timing["max_s"]
            ordered=sorted(timing["sample_ids"],key=lambda id:next(r for r in data["records"] if r["id"]==id)["elapsed_s"])
            assert timing["representative_id"]==ordered[1]
    with ZipFile(BytesIO(document.downloads["report_data_and_evidence.zip"])) as archive:
        assert json.loads(archive.read("plotted_data.json"))==data
        for name,digest in manifest.items():
            assert sha256(archive.read("original/"+name)).hexdigest()==digest,name
    metadata=json.loads((STUDY/"outputs/comparison/metadata.json").read_text())
    assert data["metadata"]==metadata
    result=dict(passed=True,records=64,measured_solves=48,numerical_inputs_unchanged=len(manifest),
        external_assets=[],exact_saved_records_preserved=True,representative_histories_verified=True,
        embedded_downloads=list(document.downloads),numerical_cases_rerun=False)
    (STUDY/"outputs/report_presentation/structural_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__": audit_html()
