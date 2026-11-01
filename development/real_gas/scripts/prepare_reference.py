"""Prepare continuous Euler references with the benchmark's bicubic tables."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json

import numpy as np

from .common import STUDY, REPO, OUTPUT, identity, save
from .thermodynamics import BicubicCO2
from .geometry import Nozzle
from nozzle_hyperbolic.models.euler_nozzle.analytical.density_eos import (
    SmoothPath, solve_shock_location, sample_profile,
)
from .reference_audit import audit_case, refine_check


def reference_fingerprint(profile=None):
    """Reference identity excludes solver settings/source, which it does not use."""
    if profile is None:
        profile = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    files = {
        "src/nozzle_hyperbolic/models/euler_nozzle/analytical/density_eos.py":
            REPO / "src/nozzle_hyperbolic/models/euler_nozzle/analytical/density_eos.py",
        **{f"development/real_gas/scripts/{name}": STUDY / "scripts" / name
           for name in ("thermodynamics.py", "geometry.py", "reference_audit.py")},
    }
    specification = dict(
        physical_cases=json.loads((STUDY / "cases.json").read_text(encoding="utf-8")),
        table_overrides=profile.get("table_overrides", {}),
        reference_code_sha256={name: sha256(path.read_bytes()).hexdigest()
                               for name, path in files.items()},
        refinement_code_sha256=sha256(inspect.getsource(refine_check).encode("utf-8")).hexdigest(),
        preparation_algorithm="bicubic_density_isentropic_rh_v1",
        stations=1001,
    )
    return identity(specification), specification


def _write_csv(path, profile):
    import csv
    keys = [k for k in ("x", "A", "side") if k in profile]
    keys += [k for k in profile if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(profile[k] for k in keys)))


def prepare_reference(force=False):
    """Build both fluid references using their actual FVM table settings."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    fingerprint, specification = reference_fingerprint(profile)
    summary_path = OUTPUT / "reference_summary.json"
    if summary_path.exists() and not force:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("metadata", {}).get("reference_fingerprint") == fingerprint:
            complete = all(
                (OUTPUT / f"reference_{case['id']}_{regime}.csv").exists()
                for case in cached["cases"] for regime in ("adapted", "shock")
            )
            hashes_match = all(
                (OUTPUT / "fluid_tables" / f"{case['settings']['table']['name']}.pkl").exists()
                and sha256((OUTPUT / "fluid_tables" / f"{case['settings']['table']['name']}.pkl").read_bytes()).hexdigest()
                == case["table_sha256"] for case in cached["cases"]
            )
            complete_case_ids = {case["id"] for case in cached["cases"]} == {
                case["id"] for case in specification["physical_cases"]["cases"]}
            if complete_case_ids and complete and hashes_match and all(case["audit"]["passed"] for case in cached["cases"]):
                print("Matching FVM references are already prepared", flush=True)
                return cached
    definitions = specification["physical_cases"]
    summary = dict(
        geometry=definitions["geometry"], cases=[],
        metadata=dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_fingerprint=fingerprint, specification=specification,
            generation_code_sha256={
                **specification["reference_code_sha256"],
                "development/real_gas/scripts/prepare_reference.py":
                    sha256((STUDY / "scripts/prepare_reference.py").read_bytes()).hexdigest(),
            },
            method="Continuous bicubic-density Euler ODE with nested RH/backpressure roots",
            property_backend="JAXProp FluidBicubic; equilibrium HEOS-generated 100x100 tables",
            table_domain_reason="The table domains cover reservoir, reference and numerical startup states",
            reference_kind="continuous_euler",
        ),
    )
    for definition in definitions["cases"]:
        case = deepcopy(definition)
        case["table"].update(profile.get("table_overrides", {}).get(case["id"], {}))
        print(f"Preparing matching reference {case['id']}: {case['table']}", flush=True)
        eos = BicubicCO2(case["table"], OUTPUT)
        nozzle = Nozzle(**(definitions["geometry"] | case.get("geometry_override", {})))
        reservoir = eos.reservoir(case["p0"], case["T0"])
        upstream = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                              eos.settings["p_min"] * 1.001)
        throat = upstream.choke()
        mdot = nozzle.At * throat["G"]
        adapted = sample_profile(upstream, nozzle, mdot, stations=1001)
        shock = solve_shock_location(eos, upstream, nozzle, mdot, case["backpressure"])
        shocked = sample_profile(upstream, nozzle, mdot, shock=shock, stations=1001)
        audit = audit_case(eos, upstream, nozzle, mdot, adapted, shocked, shock, case["backpressure"])
        audit["refinement"] = refine_check(eos, reservoir, upstream, nozzle, mdot,
                                           case["backpressure"], shock, adapted)
        pc, Tc = float(eos.fluid.critical_point["p"]), float(eos.fluid.critical_point["T"])
        if case["id"] == "supercritical":
            setup_passed = all(np.all(q["p"] > pc) and np.all(q["T"] > Tc) for q in (adapted, shocked))
        else:
            setup_passed = all(np.any((q["Q"] > 0.05) & (q["Q"] < 0.95) & (q["phase_flag"] > 0.9))
                               for q in (adapted, shocked))
        audit["physical_regime_passed"] = bool(setup_passed)
        audit["passed"] = bool(audit["passed"] and setup_passed and audit["refinement"]["passed"])
        record = dict(
            id=case["id"], label=case["label"], settings=case, geometry=asdict(nozzle),
            reservoir=reservoir, throat=throat, mdot=mdot,
            adapted_exit={k: v[-1] for k, v in adapted.items() if k != "side"},
            backpressure=case["backpressure"],
            shock={k: v for k, v in shock.items() if k != "downstream"},
            table_sha256=eos.table_sha256, audit=audit,
        )
        _write_csv(OUTPUT / f"reference_{case['id']}_adapted.csv", adapted)
        _write_csv(OUTPUT / f"reference_{case['id']}_shock.csv", shocked)
        summary["cases"].append(record)
        save(summary_path, summary)
        print(f"Reference {case['id']}: audits={audit['passed']}", flush=True)
        if not audit["passed"]:
            raise RuntimeError(f"Matching reference {case['id']} failed its required audit")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_reference(force=args.force)


if __name__ == "__main__":
    main()
