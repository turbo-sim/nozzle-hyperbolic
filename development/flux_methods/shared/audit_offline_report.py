"""Saved-data and offline packaging checks, without physical-model imports."""

import base64
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .offline_report import check_inputs, clean


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=[];self.downloads={};self.payload=[];self.collect=False;self.external=[]

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a:self.ids.append(a["id"])
        if tag=="script" and a.get("id")=="payload":self.collect=True
        if tag in ("script","link","img"):
            for key in ("src","href"):
                if key in a and not a[key].startswith(("data:","blob:","#")):self.external.append(a[key])
        if "download" in a:
            data=base64.b64decode(a["href"].split(",",1)[1],validate=True)
            assert sha256(data).hexdigest()==a["data-sha256"],a["download"]
            self.downloads[a["download"]]=data

    def handle_data(self,value):
        if self.collect:self.payload.append(value)

    def handle_endtag(self,tag):
        if tag=="script":self.collect=False


def audit(study):
    study=Path(study).resolve();manifest=check_inputs(study)
    page=Document();page.feed((study/"REPORT.html").read_text(encoding="utf-8"))
    assert not page.external,page.external
    assert len(page.ids)==len(set(page.ids)),"Duplicate HTML IDs"
    assert {"report-theme","case","cells","page-overview","page-explorer","page-discussion","page-protocol","plotly-runtime","study-runtime","report-app"}<=set(page.ids)
    data=json.loads("".join(page.payload))
    assert json.loads(page.downloads["plotted_data.json"])==data
    with ZipFile(BytesIO(page.downloads["report_data_and_evidence.zip"])) as z:
        assert json.loads(z.read("plotted_data.json"))==data
        for name,digest in manifest.items():assert sha256(z.read("original/"+name)).hexdigest()==digest,name
    assert data["presentation"]["numerical_cases_rerun"] is False
    assert data["presentation"]["inputs_sha256"]==manifest
    for row in data["records"]:
        original=clean(json.loads((study/row["input_file"]).read_text(encoding="utf-8")))
        for key in ("success","reason","history","metrics"):
            assert row.get(key)==original.get(key),(row.get("id"),key)
        if data["study"]=="limiters":
            for key in ("pressure","mach","coordinate","reference_pressure","analytical_coordinate","analytical_pressure","pressure_normalized_l1","conservative_normalized_l1","shock_error_m"):
                assert row.get(key)==original.get(key),(row["id"],key)
        else:
            assert row["comparison"]==original["comparison"]
            npz=(study/row["input_file"]).with_name("solution.npz")
            if npz.is_file():
                with np.load(npz) as fields:
                    for key in fields.files:assert row["fields"][key]==clean(fields[key]),(row["id"],key)
                    q=np.asarray(fields["quality"],float);valid=np.isfinite(q)&(q>=0)&(q<=1)
                    expected=clean(np.where(valid,1-q,np.nan))
                    assert row["displayed_liquid_fraction"]==expected,row["id"]
                    assert row["quality_diagnostic"]["masked_samples"]==int(np.count_nonzero(~valid))
            verification=(study/row["input_file"]).with_name("verification.json")
            if verification.is_file():assert row["saved_verification"]==clean(json.loads(verification.read_text()))
    if data["study"]=="lettieri":
        assert len(data["cases"])==5
        for case in data["cases"]:
            assert len(case["experiment"]["pressure_pa"])==13
            assert bool(case["upstream_cfd"])==(case["id"] in ("L1","L5"))
    result=dict(passed=True,study=data["study"],records=len(data["records"]),original_input_files_unchanged=len(manifest),
                external_assets=page.external,embedded_downloads=list(page.downloads),numerical_cases_rerun=False,
                browser_preview="Separate check; application audit does not assert browser layout")
    (study/"outputs/report_presentation/structural_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result
