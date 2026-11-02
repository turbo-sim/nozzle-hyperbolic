"""Audit saved evidence, conservation, accuracy and standalone HTML packaging."""

import base64
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

STUDY=Path(__file__).resolve().parents[1]
OUT=STUDY/"outputs"


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=[]
        self.images=[]
        self.links=[]
        self.pages=[]
        self.payload=[]
        self.collect=False
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if "id" in attrs:self.ids.append(attrs["id"])
        if tag=="img":self.images.append(attrs["src"])
        if tag=="a":self.links.append(attrs)
        if tag=="section" and "page" in attrs.get("class","").split():self.pages.append(attrs["id"])
        if tag=="script" and attrs.get("id")=="report-data":self.collect=True
        if tag in ("script","link"):
            assert "src" not in attrs and "href" not in attrs,"External dependency"

    def handle_data(self,data):
        if self.collect:self.payload.append(data)

    def handle_endtag(self,tag):
        if tag=="script":self.collect=False


def audit():
    result=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    checks={}
    checks["all_steady_converged"]=result["all_solved"]
    checks["geometry_mapping"]=result["geometry_audit"]["passed"]
    comparisons=[]
    for name,variant in result["variants"].items():
        values=[]
        for grid in variant["grids"]:
            record=json.loads((OUT/name/f"hyperbolic_n{grid['cells']}.json").read_text())
            flux=np.asarray(record["face_flux"])
            source=np.asarray(record["sources"])
            budget=flux[-1]-flux[0]-source.sum(axis=0)
            np.testing.assert_allclose(budget,record["budget"],atol=1e-12,rtol=1e-11)
            metrics=record["metrics"]
            assert max(metrics["equation_norm"])<=1e-9
            assert metrics["boundary_max"]<=1e-9 and max(metrics["global_defect"])<=1e-9
            fields=record["fields"]
            assert min(fields["M_m"])>0 and max(fields["M_m"])<1
            weights=np.asarray(record["volumes"])/sum(record["volumes"])
            for key,ref in record["reference_fields"].items():
                np.testing.assert_allclose(weights@abs(np.asarray(fields[key])-ref),
                    record["errors"][key]["l1"],rtol=2e-13,atol=1e-13)
            values.append(max(record["errors"]["conservative_normalized_l1"]))
        assert len(variant["grids"])==len(result["config"]["numerics"]["grids"])
        assert variant["references"][-1]["audit"]["passed"]
        assert all(b<a for a,b in zip(values[:-1],values[1:]))
        assert values[-1]<1e-3
        comparisons.append(dict(variant=name,errors=values,
                                last_reference_change=variant["reference_last_refinement_change"]))
    checks["refinement_accuracy_and_budgets"]=True
    # Preserve all source hashes; an unexpected change remains a visible failure.
    checks["sources_fixed_during_campaign"]=not result["sources_changed_during_run"]
    document=(STUDY/"REPORT.html").read_text(encoding="utf-8")
    parser=ReportParser()
    parser.feed(document)
    assert len(set(parser.ids))==len(parser.ids)
    assert set(parser.pages)=={"solution","geometry","convergence","model","evidence"}
    required={"case","grid","method","display","skip","field-group","field-plots",
              "download-fields","download-history","download-data","theme","report-app","plotly-library"}
    assert required.issubset(parser.ids)
    assert len(parser.images)==1
    for source in parser.images:
        assert source.startswith("data:image/png;base64,")
        assert base64.b64decode(source.split(",",1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
    download=next(a for a in parser.links if a.get("download")=="annular_verification_results.json")
    assert base64.b64decode(download["href"].split(",",1)[1])==(OUT/"results.json").read_bytes()
    math=re.findall(r"<math\b.*?</math>",document,re.S)
    assert len(math)==1
    equation=ET.fromstring(math[0])
    assert len(equation.findall(".//mtr"))==4
    checks["standalone_report_structure_and_download"]=True
    payload=json.loads("".join(parser.payload))
    assert payload["presentation"]["numerical_cases_rerun"] is False
    for path,digest in payload["presentation"]["input_sha256"].items():
        assert sha256((STUDY/path).read_bytes()).hexdigest()==digest
    for variant in payload["variants"]:
        for row in variant["runs"]:
            saved=json.loads((OUT/variant["id"]/f"hyperbolic_n{row['cells']}.json").read_text())
            assert row["fields"]==saved["fields"]
            assert row["reference_fields"]==saved["reference_fields"]
            assert row["m"]==saved["m"] and row["scheme"]==saved["scheme"]
        reference=json.loads((OUT/variant["id"]/"nozzlex_1e-12.json").read_text())["fields"]
        for key,source in dict(p="p",rho="d",v_m="v_m",v_theta="v_t",T="T",M_m="Ma_m",
                               h="h",h0="h0",s="s",alpha_deg="alpha",M="Ma").items():
            assert variant["reference"]["fields"][key]==reference[source]
    checks["interactive_payload_matches_saved_values"]=True
    before=json.loads((OUT/"presentation_input_manifest.json").read_text())
    checks["numerical_inputs_unchanged"]=all(sha256((STUDY/path).read_bytes()).hexdigest()==digest for path,digest in before.items())
    browser_file=OUT/"browser_audit/audit.json"
    browser=json.loads(browser_file.read_text()) if browser_file.exists() else {}
    checks["live_browser_review"]=bool(browser.get("passed") and browser.get("html_sha256")==sha256((STUDY/"REPORT.html").read_bytes()).hexdigest())
    evidence=dict(checks=checks,comparisons=comparisons,
        report_sha256=sha256((STUDY/"REPORT.html").read_bytes()).hexdigest(),
        limitation="Live browser evidence must match the current HTML SHA-256; static and numerical checks are recorded separately.")
    evidence["passed"]=all(value for key,value in checks.items() if key!="live_browser_review")
    (OUT/"audit.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    print(json.dumps(evidence,indent=2))
    assert evidence["passed"]


if __name__=="__main__":audit()
