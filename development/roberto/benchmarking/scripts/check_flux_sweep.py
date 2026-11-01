"""Exercise campaign dispatch and standalone reporting with synthetic results.

No physical operator is built and no numerical solve is run. The synthetic
fixture is deliberately labeled and remains under outputs/ for inspection.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import yaml

from . import run
from .audit import expected_configurations
from .common import STUDY
from .report import render, audit_html


def main():
    root_profile=STUDY.parents[1]/"roe_method/benchmark_profile.yaml"
    config=yaml.safe_load(root_profile.read_text(encoding="utf-8"))
    fluxes=run.resolved_fluxes(config)
    catalog=list(run.available_cases())
    protocol=dict(config,cases=catalog)
    expected=expected_configurations(protocol)
    assert len(expected)==640
    assert sum(v["expected_samples"] for v in expected.values())==2080
    assert all("__flux_" in key for key in expected)
    assert len({run.config_key(v["case_id"],v["reconstruction"],v["cells"],v["cap"],v["backend"],v["flux"]) for v in expected.values()})==640
    old=yaml.safe_load((STUDY/"settings.yaml").read_text(encoding="utf-8"))
    old_protocol=dict(old,cases=catalog)
    old_expected=expected_configurations(old_protocol)
    assert len(old_expected)==160 and all("__flux_" not in key for key in old_expected)
    assert run.resolved_fluxes(old)==["characteristic"]
    assert run.config_key("euler_nozzle/subsonic","constant",50,10,"current")=="euler_nozzle__subsonic__constant__n50__cap10__current"
    profile=run.flux_profile(config,"roe")
    profile["scheme"]["delta_factor"]=123
    assert config["scheme"]["delta_factor"]==.1
    assert "roe_options" not in run.flux_profile(config,"hll")
    methods=[("current",flux) for flux in fluxes]
    assert {tuple(run.rotating_order(methods,i)) for i in range(4)}=={tuple(methods[i:]+methods[:i]) for i in range(4)}
    invalid=deepcopy(config);invalid["fluxes"]=["roe","roe"]
    try:run.resolved_fluxes(invalid)
    except ValueError:pass
    else:raise AssertionError("Duplicate fluxes accepted")

    folder=STUDY/"outputs"/("synthetic_flux_contract_"+uuid4().hex)
    folder.mkdir(parents=True)
    config.update(cases=["euler_nozzle/subsonic"],grids=[4,6],reconstructions=["constant"],repeats=2,output=str(folder))
    config_path=folder/"profile.yaml"
    config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    definition=SimpleNamespace(case_id=config["cases"][0],label="SYNTHETIC dispatch fixture",model_name="euler_nozzle",
                               data={},source_name="synthetic",source_sha256="fixture",to_dict=lambda:{})
    physical=SimpleNamespace(definition=definition,legacy_payload={"initial_state":[[1.,2.,3.]]},cell_reference=lambda:None)
    bound=[];solved=[]

    class FakeWorker:
        def __init__(self,backend,folder,timeout):self.backend=backend
        def ask(self,command):
            if command["command"]=="bind":
                self.profile=command["profile"];self.flux=self.profile["scheme"]["flux"]
                bound.append((command["cells"],self.flux,deepcopy(self.profile)))
                return {"metadata":{"backend":self.backend,"flux":self.flux,"roe":self.profile.get("roe_options")}}
            solved.append(self.flux)
            return dict(elapsed_s=1.,success=True,reason="synthetic",state=[[1.,2.,3.]],fields={},x=[.5],volumes=[1.],
                        history=[],work={"accepted_steps":1},metrics={},roe_diagnostics=None)
        def close(self):pass

    assessment=dict(errors={},conservative_errors={},metrics={},analytical_passed=True,accuracy_assessed=False,
                    branch_passed=True,state={"x":[.5],"fields":{"p":[1.]},"full":[[1.,2.,3.]]},reference={})
    provenance=dict(created_at="synthetic",source={"sha256":"fixture"},harness={"sha256":"fixture"},legacy_manifest_sha256="fixture",machine_sha256="fixture")
    with patch.object(run.sys,"argv",["synthetic_dispatch","--config",str(config_path)]),\
         patch.object(run,"available_cases",lambda:tuple(config["cases"])),\
         patch.object(run,"load_case",lambda case:definition),\
         patch.object(run,"build_case",lambda case,cells:physical),\
         patch.object(run,"inspect_environment",lambda:{"status":"clean"}),\
         patch.object(run,"provenance",lambda:provenance),\
         patch.object(run,"source_snapshot",lambda:{"sha256":"fixture"}),\
         patch.object(run,"harness_snapshot",lambda:{"sha256":"fixture"}),\
         patch.object(run,"assess",lambda *args:deepcopy(assessment)),\
         patch.object(run,"Worker",FakeWorker):
        run.main()
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    assert len(bound)==8 and len(solved)==20
    assert len(summary["results"])==len(expected_configurations(summary["protocol"]))==8
    for row in summary["results"]:
        assert row["flux"]==row["metadata"]["flux"]==row["settings"]["scheme"]["flux"]
        assert row["id"].endswith("__flux_"+row["flux"])
        assert len(row["samples"])==(2 if row["cells"]==6 else 1)
    # Failure retention and mixed successful/failed fluxes are presentation inputs.
    summary["results"][0].update(status="failed",success=False,timing_valid=False,reason="SYNTHETIC failure fixture")
    summary["title"]="SYNTHETIC harness/report fixture; no simulation"
    html=render(summary,title=summary["title"])
    assert audit_html(html)["passed"]
    for control in ("summary-flux","case-flux","case-reconstruction"):
        assert f'id="{control}"' in html
    (folder/"REPORT.html").write_text(html,encoding="utf-8")
    print("Flux contract checks passed; 640 unique configurations, 2080 measured solves, 640 excluded solves.")
    print("Synthetic dispatch/report fixture:",folder)


if __name__=="__main__":
    main()
