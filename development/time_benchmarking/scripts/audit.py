"""Audit saved production measurements, source archive and offline report."""

import argparse
import base64
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from zipfile import ZipFile

STUDY=Path(__file__).resolve().parents[1]


class HTMLAudit(HTMLParser):
    def __init__(self):
        super().__init__();self.pages=set();self.targets=set();self.images=0;self.downloads=[]
        self.ids=[];self.payload=[];self.in_payload=False

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if "id" in attrs:self.ids.append(attrs["id"])
        if tag=="section" and attrs.get("id","").startswith("page-"):self.pages.add(attrs["id"].replace("page-",""))
        if tag=="button" and "data-page" in attrs:self.targets.add(attrs["data-page"])
        if tag=="script" and attrs.get("id")=="payload":self.in_payload=True
        if tag=="img":
            assert attrs["src"].startswith("data:image/png;base64,")
            assert base64.b64decode(attrs["src"].split(",",1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
            self.images+=1
        if "download" in attrs:
            data=base64.b64decode(attrs["href"].split(",",1)[1])
            assert sha256(data).hexdigest()==attrs["data-sha256"]
            self.downloads.append(attrs["download"])
        if tag in ("script","link"):assert "src" not in attrs and "href" not in attrs
        if tag=="a":assert attrs.get("href","").startswith(("data:","#"))

    def handle_data(self,value):
        if self.in_payload:self.payload.append(value)

    def handle_endtag(self,tag):
        if tag=="script":self.in_payload=False


def audit_cfl(stages, protocol):
    """Verify actual controller history, including failed transfers and cold retries."""
    carry = protocol["grid_sequence"]["carry_cfl"]
    initial = protocol["steady"]["cfl"]
    cap = protocol["steady"]["cfl_max"]
    previous = None
    for stage in stages:
        expected = initial
        if previous and previous["success"] and carry:
            expected = min(previous["last_accepted_cfl"] or initial, cap)
        assert stage["cfl_policy"] == ("carry" if carry else "reset")
        for attempt in stage["attempts"]:
            assert math.isclose(attempt["initial_cfl"], expected, rel_tol=1e-12)
            if attempt["history"]:
                assert math.isclose(attempt["history"][0]["cfl"], expected, rel_tol=1e-12)
            else:
                assert not attempt["success"]
        if stage["attempts"]:
            expected = initial
        assert math.isclose(stage["initial_cfl"], expected, rel_tol=1e-12)
        accepted = [h for h in stage["history"] if h["phase"] in ("initial", "accepted")]
        if stage["history"]:
            assert stage["initial_cfl"] == stage["history"][0]["cfl"]
        else:
            assert not stage["success"]
        assert stage["last_accepted_cfl"] == (accepted[-1]["cfl"] if accepted else None)
        assert stage["residual_normalization"] == "fresh per-equation initial norm on this grid"
        previous = stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=STUDY/"outputs/cfl_carryover_20261103/summary.json")
    parser.add_argument("--skip-report",action="store_true")
    args=parser.parse_args();path=args.input.resolve();s=json.loads(path.read_text(encoding="utf-8"));p=s["protocol"]
    assert s["completed"]
    expected=len(p["cases"])*len(p["reconstructions"])*len(p["methods"])
    assert len(s["results"])==expected
    assert len({r["id"] for r in s["results"]})==expected
    archive=path.parent/s["source_archive"]["file"]
    assert sha256(archive.read_bytes()).hexdigest()==s["source_archive"]["sha256"]
    with ZipFile(archive) as z:
        for category in ("source","harness"):
            for name,digest in s["provenance"][category]["files"].items():assert sha256(z.read(name)).hexdigest()==digest,name
        for case in s["cases"]:assert sha256(z.read(case["path"])).hexdigest()==case["sha256"]
    samples=0;kernels=0;cfl_stages=0;warnings=[]
    for r in s["results"]:
        assert r["source_unchanged"],r["id"]
        assert r["timing_environment"]["status"]=="clean",r["id"]
        intended=p["repeats"][r["case_id"].split("/")[0]]
        if len(r["samples"])!=intended:warnings.append(r["id"]+": incomplete repetitions")
        for sample in r["samples"]:
            data=(path.parent/sample["file"]).read_bytes()
            assert sha256(data).hexdigest()==sample["sha256"],sample["file"]
            raw=json.loads(data)
            if "grid_sequence" in p:
                audit_cfl(raw["stages"],p)
                cfl_stages+=len(raw["stages"])
            assert raw["elapsed_s"]==sample["elapsed_s"]>0
            assert [stage["cells"] for stage in raw["stages"]]==r["method"]["grids"]
            assert raw["fine"]["cells"]==400
            assert raw["fine"]["state"]==raw["stages"][-1]["state"]
            for stage in raw["stages"]:
                assert stage["stage_total_s"]==stage["elapsed_s"]+sum(a["elapsed_s"] for a in stage["attempts"])
                if stage["success"]:
                    metrics=stage["metrics"];policy=p["residual"]
                    assert metrics["boundary_max"]<=policy["boundary_tolerance"]
                    assert max(metrics["global_defect"])<=policy["global_tolerance"]
                    initial=next(h["equation_norm"] for h in stage["history"] if h["phase"]=="initial")
                    for current,start in zip(metrics["equation_norm"],initial):
                        assert math.isfinite(current) and current>=0 and math.isfinite(start) and start>=0
                        absolute=current<=policy["absolute_tolerance"]
                        relative=start>policy["initial_floor"] and current/start<=policy["relative_tolerance"]
                        assert policy["criterion"]=="either" and (absolute or relative)
            assert raw["fallback_count"]==sum(bool(stage["attempts"]) for stage in raw["stages"])
            samples+=1
        assert [sample["elapsed_s"] for sample in r["samples"]]==r["timing"]["samples_s"]
        for warm in r["warmups"]:
            warm_path=path.parent/warm["file"]
            assert warm_path.is_file()
            if "grid_sequence" in p:
                stages=json.loads(warm_path.read_text(encoding="utf-8"))["stages"]
                audit_cfl(stages,p)
                cfl_stages+=len(stages)
        if r.get("kernel_timing"):
            assert r["kernel_metadata"]["fixed_corrections"]==20
            assert r["kernel_metadata"]["inner_admissibility"]
            assert r["kernel_metadata"]["local_derivative_reuse"]=="always enabled"
            assert len(r["kernel_samples"])==p["kernel_repeats"]
            for key,stats in r["kernel_timing"].items():
                assert stats["samples_s"]==[sample[key] for sample in r["kernel_samples"]]
                assert min(stats["samples_s"])>0
            kernels+=1
        else:warnings.append(r["id"]+": missing component measurements")
        if r.get("success"):
            a=r["assessment"];m=a["metrics"]
            assert m["all_equations_passed"] and m["boundary_passed"] and m["global_passed"] and m["finite"] and m["admissible"]
            assert a["branch_passed"]
        if not r.get("assessment",{}).get("analytical_passed"):warnings.append(r["id"]+": original analytical target missed")
        if r.get("fallback_count"):warnings.append(r["id"]+": retained cold retry")
    report=None
    validation=path.parent/"api_removal_validation.json"
    if validation.is_file():
        evidence=json.loads(validation.read_text(encoding="utf-8"))
        assert evidence["source_sha256"]==s["provenance"]["source"]["sha256"]
        assert all(c["bitwise_equal"] and c["accuracy_unchanged"] for c in evidence["state_checks"])
    if not args.skip_report:
        report=HTMLAudit();report.feed((STUDY/"REPORT.html").read_text(encoding="utf-8"))
        assert report.pages==report.targets=={"overview","timing","fields","protocol"}
        assert set(report.downloads)=={"production_results.zip","source_and_cases.zip","timings.csv","PLOTLY_LICENSE.txt"}
        assert len(report.ids)==len(set(report.ids)),"Duplicate HTML IDs"
        controls={"model","case","reconstruction","metric","methods","report-theme","profile-range","residual-mode",
                  "coverage-plot","operation-plot","total-plot","samples-plot","stages-plot","pressure-plot","mach-plot","residual-plot","cfl-plot"}
        assert controls<=set(report.ids)
        plotted=json.loads("".join(report.payload))
        assert plotted["measurement_summary_sha256"]==sha256(path.read_bytes()).hexdigest()
        assert len(plotted["cases"])==len(p["cases"])
        assert len(plotted["rows"])==len(s["results"])
        index={r["id"]:r for r in s["results"]}
        for row in plotted["rows"]:
            original=index[row["id"]]
            for key in ("timing","fine_timing","kernel_timing"):assert row[key]==original[key],(row["id"],key)
            assert row["fields"]["x"]==original["assessment"]["state"]["x"]
            assert row["fields"]["values"]==original["assessment"]["state"]["fields"]
            assert len(row["history"])==len(original["fine_history"])
            for actual,raw in zip(row["history"],original["fine_history"]):
                assert all(raw[k]==v for k,v in actual.items())
    manifest=path.parent/"presentation/numerical_input_manifest.json"
    if manifest.is_file():
        preserved=json.loads(manifest.read_text(encoding="utf-8"))
        for name,digest in preserved.items():assert sha256((path.parent/name).read_bytes()).hexdigest()==digest,name
    result=dict(configurations=expected,measured_solves=samples,kernel_blocks=kernels,
        cfl_stage_records_checked=cfl_stages,
        offline_images=report.images if report else None,embedded_downloads=report.downloads if report else None,
        warnings=warnings,measurement_artifacts_unchanged=len(preserved) if manifest.is_file() else None,
        browser_interaction="See presentation/browser_audit.json when available; structural and application audits are separate")
    (path.parent/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
