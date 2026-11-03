"""Trace the existing implicit Newton algorithm without changing its policy.

This is diagnostic instrumentation, not a second nonlinear solver. JAX debug
callbacks observe the production Newton function and its production operators.
Wall times include compilation and callbacks and are not benchmark timings.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import primitives
from nozzle_hyperbolic.models.two_fluid.sources import evaporation_weight
from nozzle_hyperbolic.solvers.fvm import SteadyWorkspace
from nozzle_hyperbolic.solvers.fvm import Solution, solve_single_grid
from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver

from .case import ROOT, REGIMES, build_case, prepare_fluids, specification
from .run import numerical_sources, write_json
from .metrics import independent_audit, profile_kernel, reference_errors, regime_audit


def scan_steps(model, scheme, settings, workspace, initial, record, destination, args):
    """Compare existing implicit steps from the identical published seed."""
    extractor = profile_kernel(model)
    initial_metrics = workspace.sample(initial)
    initial_norm = max(initial_metrics["equation_norm"])
    rows = []
    for cfl in ((2.0, 0.02, 0.2, 20.0, 200.0, 1e4, 1e6) if args.scan else ()):
        dt = jnp.float64(workspace.operator.compute_dt(initial, cfl))
        started = time.perf_counter()
        candidate, info = workspace.operator.step(initial, dt, jnp.float64(0.0))
        candidate, info = jax.device_get((candidate, info))
        row = dict(
            cfl=cfl,
            dt=float(dt),
            newton=info._asdict(),
            elapsed_s=time.perf_counter() - started,
        )
        if bool(info.converged and info.admissible):
            metrics = workspace.sample(candidate, initial_metrics["equation_norm"])
            fields = {
                key: np.asarray(value) for key, value in extractor(candidate).items()
            }
            row.update(
                metrics=metrics,
                regime=regime_audit(fields, args.regime),
                spatial_ratio=float(max(metrics["equation_norm"]) / initial_norm),
            )
        rows.append(row)
        print(
            "CFL SCAN",
            cfl,
            "status",
            int(info.status),
            "inner",
            float(info.residual_norm),
            "spatial ratio",
            row.get("spatial_ratio"),
            flush=True,
        )
        record["scan"] = rows
        write_json(destination, record)
    eligible = [r for r in rows if r.get("regime", {}).get("passed")]
    if (args.solve_best and eligible) or args.solve_cfl is not None:
        chosen_cfl = (
            args.solve_cfl
            if args.solve_cfl is not None
            else min(eligible, key=lambda r: r["spatial_ratio"])["cfl"]
        )
        selected = replace(
            settings,
            cfl=chosen_cfl,
            max_steps=args.max_steps,
            cfl_growth=args.growth,
            cfl_max=args.cfl_max,
        )
        record["selected_settings"] = asdict(selected)
        started, last_print = time.perf_counter(), [0.0]
        history = []
        last = dict(state=initial, candidate=initial)
        original_step = workspace.operator.step

        def retained_step(*values):
            outcome = original_step(*values)
            last["candidate"] = outcome[0]
            return outcome

        workspace.operator.step = retained_step

        def progress(row):
            elapsed = time.perf_counter() - started
            history.append(dict(row, elapsed_s=elapsed))
            if row["phase"] != "initial" and row["accepted"]:
                last["state"] = last["candidate"]
            if row["phase"] == "initial" or elapsed - last_print[0] >= 20:
                last_print[0] = elapsed
                print(
                    "SOLVE",
                    row["step"],
                    row["phase"],
                    row.get("cfl"),
                    max(row.get("equation_norm", [0])),
                    f"{elapsed:.1f}s",
                    flush=True,
                )
            if elapsed > args.limit_s:
                raise TimeoutError("Diagnostic solve resource limit")

        try:
            result = solve_single_grid(
                model,
                scheme,
                selected,
                initial_state=initial,
                workspace=workspace,
                callback=progress,
            )
        except TimeoutError as error:
            record["solve_error"] = str(error)
            result = Solution(
                np.asarray(last["state"]),
                False,
                "resource_time_limit",
                history,
                workspace.sample(last["state"], initial_metrics["equation_norm"]),
            )
        finally:
            workspace.operator.step = original_step
        if result is not None:
            fields = {
                key: np.asarray(value) for key, value in extractor(result.state).items()
            }
            record["solve"] = dict(
                success=result.success,
                reason=result.reason,
                history=result.history,
                metrics=result.metrics,
                audit=independent_audit(
                    workspace,
                    result.state,
                    result.history[0]["equation_norm"],
                    result.metrics,
                ),
                regime=regime_audit(fields, args.regime),
                errors=reference_errors(
                    fields,
                    model,
                    "single_component",
                    args.regime,
                    specification("single_component", args.regime),
                ),
            )
            np.savez_compressed(
                destination.with_suffix(".npz"), state=result.state, **fields
            )
            print("SOLVE RESULT", result.reason, flush=True)
    record["source_after"] = numerical_sources()
    record["source_unchanged"] = record["source_before"] == record["source_after"]
    write_json(destination, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, default="subsonic")
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backtracks", type=int, default=20)
    parser.add_argument(
        "--reconstruction",
        choices=("constant", "muscl_van_albada"),
        default="muscl_van_albada",
    )
    parser.add_argument("--void-strength", type=float)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--solve-best", action="store_true")
    parser.add_argument("--solve-cfl", type=float)
    parser.add_argument("--growth", type=float, default=1.2)
    parser.add_argument("--cfl-max", type=float, default=1e6)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--label", default="cfl_scan")
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--point-file", type=Path)
    parser.add_argument("--limit-s", type=float, default=180.0)
    args = parser.parse_args()
    out = ROOT / "single_component/outputs/convergence_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{args.regime}_N{args.cells}_CFL{args.cfl:g}.json"
    config = specification("single_component", args.regime)
    if args.void_strength is not None:
        config["numerics"]["eps_void"] = args.void_strength
    fluids, metadata = prepare_fluids(config, "single_component")
    model, scheme, settings, effective = build_case(
        config,
        fluids,
        args.cells,
        family="single_component",
        initialization="reference",
        newton_iterations=args.iterations,
        reconstruction=args.reconstruction,
        cfl=args.cfl,
    )
    scheme = replace(scheme, max_backtracks=args.backtracks)
    effective["scheme"] = asdict(scheme)
    workspace = SteadyWorkspace(model, scheme, settings.residual)
    op = workspace.operator
    initial = model.initial_state()
    completed, info = op.complete_boundaries(initial, 0.0)
    if bool(info.converged and info.admissible):
        initial = completed
    old = jnp.asarray(initial).ravel()
    dt = jnp.float64(op.compute_dt(initial, args.cfl))
    clock = time.perf_counter()
    record = dict(
        regime=args.regime,
        cells=args.cells,
        cfl=args.cfl,
        dt=float(dt),
        effective=effective,
        properties=metadata,
        source_before=numerical_sources(),
        scope="Existing production Newton algorithm with debug callbacks; no CFD convergence claim or performance benchmark.",
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        events=[],
    )
    if args.fd_check:
        if args.point_file is None:
            point = old
        else:
            with np.load(args.point_file) as saved:
                old = jnp.asarray(saved["initial"]).ravel()
                point = jnp.asarray(saved["inner_final"]).ravel()
            record["point_file"] = str(args.point_file)
        dt = jnp.float64(op.compute_dt(old.reshape(op.shape), args.cfl))
        time_zero = jnp.float64(0.0)
        residual = np.asarray(op.residual(point, old, dt, time_zero))
        correction = op.direction(point, old, dt, time_zero)
        magnitude = float(
            jnp.max(
                abs(correction)
                / jnp.tile(jnp.asarray(model.variable_scales), args.cells + 2)
            )
        )
        vector = correction / magnitude
        action = jax.jit(
            lambda x, v: jax.jvp(
                lambda z: op.residual(z, old, dt, time_zero), (x,), (v,)
            )[1]
        )
        tangent = np.asarray(action(point, vector))
        rows = []
        for epsilon in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            plus, minus = point + epsilon * vector, point - epsilon * vector
            valid = bool(
                op.is_admissible(plus.reshape(op.shape))
                & op.is_admissible(minus.reshape(op.shape))
            )
            if valid:
                finite = (
                    np.asarray(op.residual(plus, old, dt, time_zero))
                    - np.asarray(op.residual(minus, old, dt, time_zero))
                ) / (2 * epsilon)
                row = dict(
                    epsilon=epsilon,
                    relative_jvp_error=float(
                        np.max(abs(finite - tangent)) / np.max(abs(tangent))
                    ),
                    merit_slope_relative_to_newton=float(
                        np.dot(residual, finite)
                        * magnitude
                        / np.dot(residual, residual)
                    ),
                )
            else:
                row = dict(epsilon=epsilon, admissible=False)
            rows.append(row)
            print("FINITE DIFFERENCE", row, flush=True)
        record.update(derivative_checks=rows, source_after=numerical_sources())
        record["source_unchanged"] = record["source_before"] == record["source_after"]
        write_json(out / f"{args.regime}_N{args.cells}_{args.label}.json", record)
        return
    if args.scan or args.solve_cfl is not None:
        destination = out / f"{args.regime}_N{args.cells}_{args.label}.json"
        scan_steps(
            model, scheme, settings, workspace, initial, record, destination, args
        )
        return
    context = dict(iteration=0, point=None, direction=None, gate=None)
    scale = np.tile(model.variable_scales, args.cells + 2)

    @jax.jit
    def gates(flat):
        u = primitives(flat.reshape(op.shape))[1:-1]
        liq = model.fluid_l.get_state(jxp.HmassP_INPUTS, u[:, 5], u[:, 0])
        sat = model.fluid_l.get_state_saturation(u[:, 0])
        return evaporation_weight(
            u[:, 5],
            liq["T"],
            sat["enthalpy"],
            sat["temperature"],
            u[:, 3],
            p=u[:, 0],
            p_max=model.pins.pressure_cutoff,
        )

    @jax.jit
    def action(flat, vector, previous, delta_time, current_time):
        return jax.jvp(
            lambda x: op.residual(x, previous, delta_time, current_time),
            (flat,),
            (vector,),
        )[1]

    def emit(kind, **values):
        event = dict(
            kind=kind,
            iteration=context["iteration"],
            elapsed_s=time.perf_counter() - clock,
            **values,
        )
        record["events"].append(event)
        if kind == "direction":
            print(json.dumps(event), flush=True)

    def observe_direction(point, vector, residual, linear, gate):
        point, vector, residual, linear, gate = map(
            np.asarray, (point, vector, residual, linear, gate)
        )
        context.update(
            iteration=context["iteration"] + 1, point=point, direction=vector, gate=gate
        )
        emit(
            "direction",
            scaled_max_update=float(np.max(abs(vector) / scale)),
            relative_linear_defect=float(
                np.max(abs(residual + linear)) / max(np.max(abs(residual)), 1e-300)
            ),
            residual_rms=float(np.sqrt(np.mean(residual**2))),
            free_cells=int(np.sum(gate)),
            pinned_cells=int(len(gate) - np.sum(gate)),
        )

    def observe_valid(point, valid):
        point = np.asarray(point)
        if context["point"] is None:
            alpha = None
        else:
            delta = context["direction"] / scale
            alpha = float(
                np.dot((point - context["point"]) / scale, delta) / np.dot(delta, delta)
            )
        emit("admissible", passed=bool(valid), alpha=alpha)

    def observe_residual(point, value, gate):
        point, value, gate = map(np.asarray, (point, value, gate))
        emit(
            "residual",
            rms=float(np.sqrt(np.mean(value**2))),
            maximum=float(np.max(abs(value))),
            worst_row=int(np.argmax(abs(value))),
            gate_changes=(
                0 if context["gate"] is None else int(np.sum(gate != context["gate"]))
            ),
        )

    def residual(point, previous, delta_time, current_time):
        value = op.residual(point, previous, delta_time, current_time)
        jax.debug.callback(observe_residual, point, value, gates(point), ordered=True)
        return value

    def direction(point, previous, delta_time, current_time):
        vector = op.direction(point, previous, delta_time, current_time)
        value = op.residual(point, previous, delta_time, current_time)
        linear = action(point, vector, previous, delta_time, current_time)
        jax.debug.callback(
            observe_direction, point, vector, value, linear, gates(point), ordered=True
        )
        return vector

    def admissible(point):
        valid = op.is_admissible(point.reshape(op.shape))
        jax.debug.callback(observe_valid, point, valid, ordered=True)
        return valid

    solve = jax.jit(
        make_newton_solver(
            residual,
            direction,
            lambda _: jnp.asarray(scale),
            max_iterations=scheme.newton_iterations,
            tolerance=scheme.newton_update_tolerance,
            residual_tolerance=scheme.newton_residual_tolerance,
            admissible=admissible,
            backtracking=scheme.newton_backtracking,
            max_backtracks=scheme.max_backtracks,
        )
    )
    final, result = solve(old, old, dt, jnp.float64(0.0))
    jax.block_until_ready(final)
    jax.effects_barrier()
    record["newton_info"] = result._asdict()
    record["source_after"] = numerical_sources()
    record["source_unchanged"] = record["source_before"] == record["source_after"]
    np.savez_compressed(
        destination.with_suffix(".npz"),
        initial=np.asarray(initial),
        inner_final=np.asarray(final).reshape(op.shape),
    )
    write_json(destination, record)
    print("Newton result:", result, flush=True)
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    main()
