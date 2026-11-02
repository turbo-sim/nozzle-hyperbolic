"""Execute published nitrogen/water cases through the shared two-fluid solver."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import primitives
from nozzle_hyperbolic.solvers.fvm import (
    Operator, solve_single_grid, save_restart, load_restart, prepare_continuation_state,
)
from nozzle_hyperbolic.solvers.fvm.drivers import sampler
from nozzle_hyperbolic.solvers.convergence_output import strict_json

from .case import (ROOT, REGIMES, REFERENCE_SHA, specification, reference,
                   prepare_fluids, build_case, environment, source_hashes)


def write_json(path, value):
    path.write_text(strict_json(value), encoding="utf-8")


def profile(model, state):
    U = primitives(jnp.asarray(state))
    pg = model.fluid_g.get_state(jxp.HmassP_INPUTS, U[:, 4], U[:, 0])
    pl = model.fluid_l.get_state(jxp.HmassP_INPUTS, U[:, 5], U[:, 0])
    ev = jax.vmap(lambda u: model._wave_data(u)[0])(U)
    sound, velocity = .5 * (ev[:, 2] - ev[:, 3]), .5 * (ev[:, 2] + ev[:, 3])
    ug, ul, alpha = U[:, 1], U[:, 2], U[:, 3]
    mg, ml = alpha * pg["rho"], (1 - alpha) * pl["rho"]
    q = mg * ug / (mg * ug + ml * ul)
    area = np.r_[model.mesh.area_faces[0], model.mesh.area_centers, model.mesh.area_faces[-1]]
    result = dict(x=model.mesh.points, A=area, r=np.sqrt(area / np.pi), p=U[:, 0], alpha_g=alpha,
        u_g=ug, u_l=ul, T_g=pg["T"], T_l=pl["T"], h_g=U[:, 4], h_l=U[:, 5],
        rho_g=pg["rho"], rho_l=pl["rho"], a_mix=sound, u_mix=velocity,
        Ma_mix=jnp.abs(velocity) / sound, quality_static=mg / (mg + ml),
        quality_flow=q, slip=ug / ul, mdot=area * (mg * ug + ml * ul))
    return {key: np.asarray(value) for key, value in result.items()}


def reference_errors(data, regime):
    ref = reference(regime)
    x = data["x"][1:-1]
    result = {}
    scales = {"p": 2e6, "u_g": max(abs(ref["u_g"][1:-1])),
              "u_l": max(abs(ref["u_l"][1:-1])), "alpha_g": 1.,
              "Ma_mix": 1., "quality_flow": 1., "T_g": 295.15, "T_l": 295.15}
    for key, scale in scales.items():
        # Compare interior profiles only; do not use endpoint ghosts as samples.
        expected = np.interp(x, ref["x"][1:-1], ref[key][1:-1])
        delta = data[key][1:-1] - expected
        result[key] = dict(rmse=float(np.sqrt(np.mean(delta**2))),
                           max_abs=float(np.max(np.abs(delta))), scale=float(scale),
                           normalized_rmse=float(np.sqrt(np.mean(delta**2)) / scale))
    result["reference_interior_cells"] = len(ref) - 2
    result["comparison"] = "point-profile interpolation; coarse endpoint samples use bounded interpolation"
    return result


def regime_audit(data, regime):
    mach = data["Ma_mix"][1:-1]
    maximum, outlet = float(max(mach)), float(mach[-1])
    forward = bool(np.all(data["u_g"] > 0) and np.all(data["u_l"] > 0))
    correct = (maximum < 1 if regime == "subsonic" else
               maximum > 1 and outlet < 1 if regime == "supersonic_shock" else outlet > 1)
    return dict(passed=bool(forward and correct), forward_flow=forward,
                max_mach=maximum, outlet_mach=outlet,
                definition="subsonic: max M<1; shock: max M>1 and exit M<1; adapted: exit M>1; both velocities positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=(*REGIMES, "all"), nargs="+", default=["subsonic"])
    parser.add_argument("--cells", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--independent-meshes", action="store_true",
                        help="Initialize each mesh independently; do not prolong the previous mesh solution")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--newton-iterations", type=int)
    parser.add_argument("--cfl", type=float)
    parser.add_argument("--stretch", type=float, default=1.)
    parser.add_argument("--initialization", choices=("cold", "reference"), default="cold")
    parser.add_argument("--area-law", choices=("original", "polynomial"), default="original",
                        help="original reproduces the NICFD closure; polynomial is an explicit later-physics diagnostic")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / "outputs" / time.strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(reference_repository_revision=REFERENCE_SHA, environment=environment(),
                  source_hashes=source_hashes(),
                  arguments={key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}, stages=[],
                  timing_note="Compile-inclusive elapsed times; no repeated warmed benchmark claim.")
    write_json(out / "result.json", record)
    try:
        config = specification("subsonic")
        started = time.perf_counter()
        fluids, metadata = prepare_fluids(config)
        record.update(property_preparation_s=time.perf_counter() - started, properties=metadata)
        write_json(out / "result.json", record)
        if args.prepare_only:
            print("Property preparation complete:", out, flush=True)
            return 0
        if "all" in args.regime and len(args.regime) != 1:
            raise ValueError("Use all alone or list explicit regimes")
        regimes = REGIMES if args.regime == ["all"] else tuple(args.regime)
        for regime in regimes:
            config = specification(regime)
            previous = load_restart(args.restart) if args.restart else None
            for cells in args.cells:
                if args.independent_meshes:
                    previous = load_restart(args.restart) if args.restart else None
                started = time.perf_counter()
                model, scheme, settings, effective = build_case(config, fluids, cells,
                    initialization=args.initialization, stretch=args.stretch,
                    newton_iterations=args.newton_iterations, cfl=args.cfl, max_steps=args.steps,
                    area_law=args.area_law)
                stage = dict(regime=regime, cells=cells, source_configuration=config,
                    effective=effective, success=False, reason="preparing", history=[])
                stage["initialization"] = ("successful portable/mesh restart" if previous is not None
                                           else args.initialization)
                stage["model_scale_reference_seed"] = effective["initialization"]
                record["stages"].append(stage)
                op = Operator(model, scheme)
                if previous is None:
                    # Coupled Newton owns the physical trace equations. A local
                    # completion provides a consistent seed if it succeeds.
                    initial = model.initial_state()
                    completed, info = op.complete_boundaries(initial, 0.)
                    completed.block_until_ready()
                    stage["initial_boundary_completion"] = info._asdict()
                    if bool(info.converged & info.admissible):
                        initial = completed
                else:
                    initial, preparation = prepare_continuation_state(model, previous, scheme)
                    stage["continuation_preparation"] = preparation
                    if not preparation["success"]:
                        stage.update(reason="continuation_preparation_failed",
                                     preparation_s=time.perf_counter() - started)
                        write_json(out / "result.json", record)
                        previous = None
                        continue
                stage["preparation_s"] = time.perf_counter() - started
                solve_started = time.perf_counter()
                previous_event = solve_started
                def progress(row):
                    nonlocal previous_event
                    now = time.perf_counter()
                    stage["history"].append(dict(row, elapsed_s=now - solve_started,
                                                 event_elapsed_s=now - previous_event))
                    previous_event = now
                    if row["phase"] == "initial" or row["step"] % 25 == 0:
                        message = f"{regime} N={cells} {row['phase']} step={row['step']} CFL={row['cfl']:.3g}"
                        if "equation_norm" in row:
                            message += f" maxL2={max(row['equation_norm']):.3g}"
                        print(message, flush=True)
                        write_json(out / "result.json", record)
                result = solve_single_grid(model, scheme, settings, initial_state=initial, callback=progress)
                stage.update(success=result.success, reason=result.reason, metrics=result.metrics,
                             solve_s=time.perf_counter() - solve_started,
                             driver_elapsed_s=result.elapsed_s)
                # Re-evaluate the saved final state, without a further update.
                audit_started = time.perf_counter()
                stage["fresh_metrics"] = sampler(result.operator, settings.residual)(
                    jnp.asarray(result.state), np.asarray(stage["history"][0]["equation_norm"]))
                data = profile(model, result.state)
                stage["errors"] = reference_errors(data, regime)
                stage["regime_audit"] = regime_audit(data, regime)
                stage["case_success"] = bool(result.success and stage["regime_audit"]["passed"])
                stage["case_reason"] = (result.reason if stage["regime_audit"]["passed"]
                                        else "case_regime_mismatch")
                stage["field_diagnostics"] = dict(
                    outlet_mach=float(data["Ma_mix"][-2]), max_mach=float(max(data["Ma_mix"][1:-1])),
                    min_slip_velocity=float(min((data["u_g"]-data["u_l"])[1:-1])),
                    max_slip_velocity=float(max((data["u_g"]-data["u_l"])[1:-1])),
                    flow_quality_min=float(min(data["quality_flow"][1:-1])),
                    flow_quality_max=float(max(data["quality_flow"][1:-1])),
                    point_mdot_min=float(min(data["mdot"][1:-1])),
                    point_mdot_max=float(max(data["mdot"][1:-1])))
                stage["audit_s"] = time.perf_counter() - audit_started
                stem = f"{len(record['stages']):02d}_{regime}_N{cells}"
                np.savez_compressed(out / (stem + "_fields.npz"), state=result.state, **data)
                np.savetxt(out / (stem + ".csv"), np.column_stack(list(data.values())),
                           delimiter=",", header=",".join(data), comments="", fmt="%.13e")
                restart_result = replace(result, success=stage["case_success"],
                    reason=result.reason if stage["regime_audit"]["passed"] else "case_regime_mismatch")
                save_restart(out / (stem + "_restart.npz"), restart_result,
                             metadata={"regime": regime, "reference_sha": REFERENCE_SHA,
                                       "numerical_root_success": result.success,
                                       "initialization": stage["initialization"]})
                stage["fields"] = stem + "_fields.npz"
                stage["restart"] = stem + "_restart.npz"
                write_json(out / "result.json", record)
                print(regime, cells, stage["case_reason"], f"{stage['solve_s']:.2f}s", flush=True)
                previous = result if stage["case_success"] else None
        return int(any(not stage.get("case_success", stage["success"]) for stage in record["stages"]))
    except Exception as exc:
        record["failure"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        write_json(out / "result.json", record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
