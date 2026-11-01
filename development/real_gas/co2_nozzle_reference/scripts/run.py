"""Build all four continuous reference profiles and execute their audits."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import traceback

import jax
import numpy as np

from .thermodynamics import BicubicCO2
from .reference import Nozzle, SmoothPath, solve_shock_location, sample_profile
from .audit import audit_case


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def fingerprints(directory):
    return {str(p.relative_to(directory)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and
            "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, profile):
    keys = [k for k in ("x", "A", "side") if k in profile]
    keys += [k for k in profile if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(profile[k] for k in keys)))


def refine_check(eos, reservoir, upstream, nozzle, mdot, backpressure, shock, adapted):
    """Repeat the ODE primitive at tighter tolerance and smaller maximum steps."""
    tighter = dict(rtol=2e-12, max_step=0.003)
    path = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                      upstream.p_low, **tighter)
    star = path.choke()
    mdot_fine = nozzle.At * star["G"]
    shock_fine = solve_shock_location(eos, path, nozzle, mdot_fine, backpressure, **tighter)
    adapted_fine = sample_profile(path, nozzle, mdot_fine, stations=151)
    checks = dict(
        mdot_relative=abs(mdot_fine / mdot - 1),
        throat_pressure_relative=abs(star["p"] / upstream.p_star - 1),
        adapted_exit_pressure_relative=abs(adapted_fine["p"][-1] / adapted["p"][-1] - 1),
        shock_position_over_divergent_length=abs(shock_fine["x"] - shock["x"]) / nozzle.x_out,
        shock_pre_pressure_relative=abs(shock_fine["pre"]["p"] / shock["pre"]["p"] - 1),
        shock_post_pressure_relative=abs(shock_fine["post"]["p"] / shock["post"]["p"] - 1),
    )
    checks["threshold"] = 1e-6
    checks["passed"] = bool(max(v for k, v in checks.items() if k != "threshold") < 1e-6)
    checks["settings"] = tighter
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=int, default=501)
    parser.add_argument("--skip-refinement", action="store_true",
                        help="Diagnostic runs only; default includes a tighter-ODE repeat")
    args = parser.parse_args()
    if args.stations < 51:
        parser.error("at least 51 plot stations are required")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    definitions = json.loads((ROOT / "cases.json").read_text())
    nozzle = Nozzle(**definitions["geometry"])
    source_before = fingerprints(REPO / "src")
    save_json(output / "source_before.json", source_before)
    summary = dict(
        geometry=asdict(nozzle), cases=[],
        metadata=dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            method="Continuous quasi-1D Euler: bicubic density ODE + nested bracketed RH/backpressure roots",
            property_backend="jaxprop.FluidBicubic; HEOS source for table generation; equilibrium",
            table_shape=[100, 100], dtype="float64", python=platform.python_version(),
            packages={p: importlib.metadata.version(p) for p in
                      ("jaxprop", "jax", "jaxlib", "numpy", "scipy", "matplotlib", "CoolProp")},
            jax_backend=jax.default_backend(),
            branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            ode=dict(method="DOP853", rtol=2e-11, atol_J_kg=2e-6, max_step_log_p=0.008),
            algebraic_roots=dict(method="scipy.optimize.brentq", pressure_xtol_Pa=1e-5,
                                 pressure_rtol=2e-13, position_xtol_m=2e-12),
            stations=args.stations, fvm_solutions=False,
        ),
    )
    try:
        for case in definitions["cases"]:
            print(f"Building {case['id']}", flush=True)
            nozzle = Nozzle(**(definitions["geometry"] | case.get("geometry_override", {})))
            eos = BicubicCO2(case["table"], output)
            reservoir = eos.reservoir(case["p0"], case["T0"])
            upstream = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                                  eos.settings["p_min"] * 1.001)
            throat = upstream.choke()
            mdot = nozzle.At * throat["G"]
            backpressure = case["backpressure"]
            if backpressure is None:
                raise ValueError("Set a fixed backpressure in cases.json after the optional feasibility probe")
            adapted = sample_profile(upstream, nozzle, mdot, stations=args.stations)
            shock = solve_shock_location(eos, upstream, nozzle, mdot, backpressure)
            shocked = sample_profile(upstream, nozzle, mdot, shock=shock, stations=args.stations)
            write_csv(output / f"{case['id']}_adapted.csv", adapted)
            write_csv(output / f"{case['id']}_shock.csv", shocked)
            # Pressure-parametrized mass-flux curve makes branch selection reviewable.
            pressure = np.geomspace(upstream.p_low, upstream.p_high, 1201)
            choking_curve = upstream.state(pressure)
            write_csv(output / f"{case['id']}_choking_curve.csv",
                      dict(p_over_p0=pressure / case["p0"],
                           G_over_Gstar=choking_curve["G"] / throat["G"], **choking_curve))
            audit = audit_case(eos, upstream, nozzle, mdot, adapted, shocked, shock, backpressure)
            pc = float(eos.fluid.critical_point["p"])
            Tc = float(eos.fluid.critical_point["T"])
            if case["id"] == "supercritical":
                setup_passed = all(np.all(profile["p"] > pc) and np.all(profile["T"] > Tc)
                                   for profile in (adapted, shocked))
                audit["setup"] = dict(passed=bool(setup_passed), requirement="Every sampled state above both critical pressure and temperature",
                                      critical_pressure_Pa=pc, critical_temperature_K=Tc)
            else:
                setup_passed = all(np.any((profile["Q"] > 0.05) & (profile["Q"] < 0.95) &
                                         (profile["phase_flag"] > 0.9)) for profile in (adapted, shocked))
                audit["setup"] = dict(passed=bool(setup_passed), requirement="Both profiles contain interior equilibrium two-phase states",
                                      critical_pressure_Pa=pc, critical_temperature_K=Tc)
            audit["passed"] = bool(audit["passed"] and setup_passed)
            if not args.skip_refinement:
                audit["refinement"] = refine_check(eos, reservoir, upstream, nozzle, mdot,
                                                   backpressure, shock, adapted)
                audit["passed"] = bool(audit["passed"] and audit["refinement"]["passed"])
            record = dict(
                id=case["id"], label=case["label"], settings=case, reservoir=reservoir,
                geometry=asdict(nozzle),
                throat=throat, mdot=mdot,
                adapted_exit={k: v[-1] for k, v in adapted.items() if k not in ("side",)},
                backpressure=backpressure,
                shock={k: v for k, v in shock.items() if k != "downstream"},
                table_sha256=eos.table_sha256, audit=audit,
            )
            summary["cases"].append(record)
            save_json(output / "summary.json", summary)
            print(f"{case['id']}: mdot={mdot:.9g} kg/s, adapted pb={adapted['p'][-1]/1e6:.9g} MPa, "
                  f"shock pb={backpressure/1e6:.6g} MPa, x_s={shock['x']*1e3:.9g} mm; audits={audit['passed']}", flush=True)
            if not audit["passed"]:
                raise RuntimeError(f"{case['id']} failed required audit; see outputs/summary.json")
    except Exception:
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        source_after = fingerprints(REPO / "src")
        summary["metadata"]["src_unchanged"] = source_before == source_after
        summary["metadata"]["src_file_count"] = len(source_before)
        summary["metadata"]["study_files_sha256"] = {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "cases.json", *sorted((ROOT / "scripts").glob("*.py"))]
        }
        save_json(output / "source_after.json", source_after)
        save_json(output / "summary.json", summary)
    if not summary["metadata"]["src_unchanged"]:
        raise RuntimeError("Source files changed during run; inspect source manifests")
    from .plot_results import build_report
    build_report()


if __name__ == "__main__":
    main()
