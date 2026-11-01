"""Prepare matching continuous references for the FVM property-table domains.

Only study-local outputs are written. The established analytical study remains
unchanged, including its original narrower supercritical property table.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from .common import BASELINE, STUDY, OUTPUT, identity, save
from development.real_gas.co2_nozzle_reference.scripts.thermodynamics import BicubicCO2
from development.real_gas.co2_nozzle_reference.scripts.reference import (
    Nozzle, SmoothPath, solve_shock_location, sample_profile,
)
from development.real_gas.co2_nozzle_reference.scripts.audit import audit_case
from development.real_gas.co2_nozzle_reference.scripts.run import refine_check


def reference_fingerprint(profile=None):
    """Reference identity excludes solver settings/source, which it does not use."""
    if profile is None:
        profile = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    names = ("scripts/reference.py", "scripts/thermodynamics.py", "scripts/audit.py", "scripts/run.py")
    specification = dict(
        physical_cases=json.loads((BASELINE / "cases.json").read_text(encoding="utf-8")),
        table_overrides=profile.get("table_overrides", {}),
        reference_code_sha256={name: sha256((BASELINE / name).read_bytes()).hexdigest()
                               for name in names},
        preparation_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
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


def _matching_tables(settings, original, saved):
    """Copy existing original bytes only when the full table settings match."""
    if settings != original:
        return
    cache = OUTPUT / "fluid_tables"
    cache.mkdir(parents=True, exist_ok=True)
    table = f"{settings['name']}.pkl"
    destination = cache / table
    if not destination.exists():
        source = BASELINE / "outputs/fluid_tables" / table
        if not source.exists():
            raise FileNotFoundError("Regenerate the original baseline tables first")
        shutil.copyfile(source, destination)
    if sha256(destination.read_bytes()).hexdigest() != saved["table_sha256"]:
        raise RuntimeError("Unmodified table does not match its original reference hash")
    source_log = BASELINE / "outputs" / f"{settings['name']}_table.log"
    if source_log.exists() and not (OUTPUT / source_log.name).exists():
        shutil.copyfile(source_log, OUTPUT / source_log.name)


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
    originals = json.loads((BASELINE / "outputs/summary.json").read_text(encoding="utf-8"))
    original_by_id = {case["id"]: case for case in originals["cases"]}
    summary = dict(
        geometry=definitions["geometry"], cases=[],
        metadata=dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_fingerprint=fingerprint, specification=specification,
            method="Same continuous bicubic-density Euler ODE and nested RH/backpressure roots as original baseline",
            property_backend="JAXProp FluidBicubic; equilibrium HEOS-generated 100x100 tables",
            table_domain_reason="Supercritical upper-pressure extension permits transient startup overshoots; all comparisons use a recomputed reference from the same table",
            original_baseline_unchanged=True, fvm_solutions=False,
        ),
    )
    for original_case in definitions["cases"]:
        case = deepcopy(original_case)
        original = original_by_id[case["id"]]
        original_settings = deepcopy(case["table"])
        case["table"].update(profile.get("table_overrides", {}).get(case["id"], {}))
        print(f"Preparing matching reference {case['id']}: {case['table']}", flush=True)
        _matching_tables(case["table"], original_settings, original)
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
        drift = dict(
            table_settings_changed=case["table"] != original_settings,
            original_table=original_settings,
            original_table_sha256=original["table_sha256"],
            mass_flow_relative=mdot / original["mdot"] - 1,
            reservoir_enthalpy_difference_J_kg=reservoir["h"] - original["reservoir"]["h"],
            throat_pressure_relative=throat["p"] / original["throat"]["p"] - 1,
            adapted_exit_pressure_relative=adapted["p"][-1] / original["adapted_exit"]["p"] - 1,
            shock_position_difference_m=shock["x"] - original["shock"]["x"],
            shock_pre_pressure_relative=shock["pre"]["p"] / original["shock"]["pre"]["p"] - 1,
            shock_post_pressure_relative=shock["post"]["p"] / original["shock"]["post"]["p"] - 1,
        )
        record = dict(
            id=case["id"], label=case["label"], settings=case, geometry=asdict(nozzle),
            reservoir=reservoir, throat=throat, mdot=mdot,
            adapted_exit={k: v[-1] for k, v in adapted.items() if k != "side"},
            backpressure=case["backpressure"],
            shock={k: v for k, v in shock.items() if k != "downstream"},
            table_sha256=eos.table_sha256, audit=audit, original_reference_drift=drift,
        )
        _write_csv(OUTPUT / f"reference_{case['id']}_adapted.csv", adapted)
        _write_csv(OUTPUT / f"reference_{case['id']}_shock.csv", shocked)
        summary["cases"].append(record)
        save(summary_path, summary)
        print(f"Reference {case['id']}: audits={audit['passed']}; original-reference drift={drift}", flush=True)
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
