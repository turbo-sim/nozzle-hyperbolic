"""Read-only production naming and study-dependency checks; no flow solves."""
from dataclasses import asdict
import ast
from pathlib import Path
import yaml
from nozzle_hyperbolic.solvers.fvm import Scheme
from .common import ROOT,FAMILY,save,utc,source_hashes,case_hashes


def main():
    methods=("stadtke","rusanov","hll","roe","osher")
    schemes={name:asdict(Scheme(flux=name)) for name in methods}
    try:
        Scheme(flux="characteristic")
    except ValueError:
        retired_alias_rejected=True
    else:
        retired_alias_rejected=False
    old_imports=[]
    checked=[]
    for path in [*(ROOT/"src").rglob("*.py"),*(FAMILY/"shared").glob("*.py")]:
        tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):
                names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom):
                names=[node.module or ""]
            for name in names:
                if "shock_spikers" in name:
                    old_imports.append(dict(file=path.relative_to(ROOT).as_posix(),line=node.lineno,module=name))
        checked.append(path.relative_to(ROOT).as_posix())
    baseline_profiles={}
    for name in ("development/historical/benchmarking/settings.yaml","development/flux_methods/roe/benchmark_profile.yaml"):
        config=yaml.safe_load((ROOT/name).read_text(encoding="utf-8-sig"))
        baseline_profiles[name]=dict(flux=config["scheme"].get("flux"),fluxes=config.get("fluxes"),
             custom_jump_correction=config["scheme"]["custom_jump_correction"],
             custom_endpoint_envelope=config["scheme"]["custom_endpoint_envelope"])
    passed=bool(retired_alias_rejected and not old_imports
                and Scheme().flux=="stadtke" and Scheme().custom_jump_correction and Scheme().custom_endpoint_envelope
                and all(not r["custom_jump_correction"] and not r["custom_endpoint_envelope"] for r in baseline_profiles.values()))
    save(FAMILY/"shared/evidence/production_api_audit.json",dict(utc=utc(),passed=passed,
         schemes=schemes,retired_alias_rejected=retired_alias_rejected,old_runtime_imports=old_imports,
         checked_files=checked,original_baseline_profiles=baseline_profiles,
         source_sha256=source_hashes(),case_sha256=case_hashes(),
         interpretation="Public method names and static imports only; full production tests and numerical campaigns are separate evidence. Historical private baseline IDs and mathematical characteristic names are intentionally retained."))
    print("Production API/dependency audit:",passed)
    if not passed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
