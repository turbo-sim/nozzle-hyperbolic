"""Reproducible solver campaign; all writes stay under this study directory."""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter
from unittest.mock import patch
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from nozzle_hyperbolic.solvers.fvm import (
    Operator,
    Scheme,
    SteadySettings,
    solve_grid_sequence,
    solve_transient,
    sampler,
    ResidualPolicy,
)
from nozzle_hyperbolic.solvers.fvm import drivers
from .problem import geometry, model

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
METHODS = {
    "euler": ("euler-explicit", "segregated"),
    "rk3": ("runge-kutta-3", "segregated"),
    "implicit": ("euler-implicit", "coupled"),
}


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.ndarray, jax.Array)):
        return np.asarray(x).tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(path, data):
    path.write_text(
        json.dumps(plain(data), indent=2, allow_nan=False), encoding="utf-8"
    )


def hashes():
    return {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src").rglob("*.py")
    }


def record(sol):
    m = sol.operator.model
    fields = {k: np.asarray(v) for k, v in m.fields(jnp.asarray(sol.state)).items()}
    return dict(
        success=sol.success,
        reason=sol.reason,
        elapsed_s=sol.elapsed_s,
        time=sol.time,
        metrics=sol.metrics,
        history=sol.history,
        state=sol.state,
        fields=fields,
        x=m.native_mesh.x_center,
        cells=m.mesh.n_cells,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    ap.add_argument("--steady-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip independent audits and contour/GIF generation",
    )
    args = ap.parse_args()
    if args.report_only:
        from .report import build

        build()
        return
    c = yaml.safe_load((STUDY / "case.yaml").read_text())
    out = STUDY / "outputs"
    out.mkdir(exist_ok=True)
    before = hashes()
    save(out / "source_before.json", before)
    geom = geometry(c)
    mm = np.linspace(0, geom.length, 601)
    g = geom(mm)
    geom_data = dict(
        m=mm,
        x=g.x,
        r=g.r,
        b=g.b,
        A=g.A,
        tip_x=g.x - g.b * g.r_prime / 2,
        hub_x=g.x + g.b * g.r_prime / 2,
        tip_r=g.r + g.b * g.x_prime / 2,
        hub_r=g.r - g.b * g.x_prime / 2,
    )
    save(out / "geometry.json", geom_data)
    print(
        "Geometry built; max tip-radius error",
        np.max(np.abs(geom_data["tip_r"] - c["geometry"]["tip_radius_m"])),
        flush=True,
    )
    ncfg = c["numerics"]
    models = {n: model(c, geom, n) for n in ncfg["steady_grids"]}
    # Process-local constructor substitution only reuses identical immutable
    # production Operators. All driver/controller equations remain production.
    cache = {}

    def cached(m, s):
        key = (id(m), s)
        if key not in cache:
            cache[key] = Operator(m, s)
        return cache[key]

    count = c["benchmark"]["measured_repeats"] if args.repeats is None else args.repeats
    if count < 1:
        raise ValueError("At least one measured repetition is required")
    result = dict(
        config=c,
        environment=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            platform=platform.platform(),
            devices=str(jax.devices()),
        ),
        timing_protocol="One excluded complete solve, then repeated identical solves with cached production Operators. Driver elapsed includes monitoring and retries; excludes model/geometry construction, output and plotting. Warmup separately retained. Method order reverses on alternate repetitions.",
        steady={},
        transient={},
        source_before=before,
    )
    schemes = {
        name: Scheme(
            flux=ncfg["flux"],
            reconstruction=ncfg["reconstruction"],
            time_integrator=METHODS[name][0],
            boundary_treatment=METHODS[name][1],
        )
        for name in args.methods
    }

    def sequence(name):
        stages = []

        def stage(m, sol, metadata):
            sol.elapsed_s += sum(a["elapsed_s"] for a in metadata[-1]["attempts"])
            stages.append(sol)

        with patch.object(drivers, "Operator", side_effect=cached):
            solve_grid_sequence(
                lambda n: models[n],
                ncfg["steady_grids"],
                schemes[name],
                steady_settings[name],
                on_stage=stage,
            )
        return stages

    steady_settings = {
        name: SteadySettings(
            max_steps=ncfg["max_steady_steps"],
            cfl=0.2,
            cfl_max=1e6 if name == "implicit" else ncfg["explicit_cfl"],
        )
        for name in schemes
    }
    for name, s in schemes.items():
        print("STEADY WARMUP", name, flush=True)
        warm = sequence(name)
        warmrec = [record(sol) for sol in warm]
        save(out / f"steady_{name}_warmup.json", warmrec)
        result["steady"][name] = dict(
            scheme=asdict(s),
            settings=asdict(steady_settings[name]),
            warmup=[
                {k: r[k] for k in ("success", "reason", "elapsed_s", "cells")}
                for r in warmrec
            ],
            samples=[],
        )
    for repeat in range(count):
        order = list(schemes) if repeat % 2 == 0 else list(reversed(schemes))
        for name in order:
            sols = sequence(name)
            elapsed = sum(sol.elapsed_s for sol in sols)
            result["steady"][name]["samples"].append(
                dict(
                    elapsed_s=elapsed,
                    success=all(sol.success for sol in sols),
                    grids=[
                        dict(
                            cells=sol.operator.mesh.n_cells,
                            success=sol.success,
                            reason=sol.reason,
                            elapsed_s=sol.elapsed_s,
                            metrics=sol.metrics,
                            steps=len(sol.history) - 1,
                        )
                        for sol in sols
                    ],
                )
            )
            result["steady"][name]["representative"] = [record(sol) for sol in sols]
            print(
                f"STEADY {name} measured {repeat+1}/{count}: {elapsed:.3f}s success={[sol.success for sol in sols]}",
                flush=True,
            )
            save(out / "results.json", result)
    successful = [
        name
        for name in ("implicit", "rk3", "euler")
        if name in result["steady"]
        and result["steady"][name]["representative"][-1]["success"]
    ]
    if not successful:
        raise RuntimeError("No converged steady seed; retained failed runs in outputs")
    seed = result["steady"][successful[0]]["representative"][-1]["state"]
    if not args.steady_only:
        driven = model(c, geom, ncfg["transient_cells"], driven=True, initial=seed)
        times = np.linspace(
            0,
            c["forcing"]["cycles"] / c["forcing"]["frequency_Hz"],
            c["forcing"]["cycles"] * c["forcing"]["frames_per_cycle"] + 1,
        )[1:]
        initials = {}

        def transient_run(name):
            s = schemes[name]
            cfl = (
                ncfg["implicit_transient_cfl"]
                if name == "implicit"
                else ncfg["explicit_cfl"]
            )
            with patch.object(drivers, "Operator", side_effect=cached):
                return solve_transient(
                    driven,
                    times,
                    s,
                    cfl=cfl,
                    initial_state=initials[name],
                    max_steps=ncfg["max_transient_steps"],
                )

        for name, s in schemes.items():
            print("TRANSIENT WARMUP", name, flush=True)
            op = cached(driven, s)
            init, info = op.complete_boundaries(jnp.asarray(seed), 0.0)
            if not bool(info.converged & info.admissible):
                raise RuntimeError("Initial trace solve failed")
            initials[name] = init
            snaps = transient_run(name)
            result["transient"][name] = dict(
                scheme=asdict(s),
                cfl=(
                    ncfg["implicit_transient_cfl"]
                    if name == "implicit"
                    else ncfg["explicit_cfl"]
                ),
                warmup=dict(
                    success=all(sol.success for sol in snaps),
                    elapsed_s=snaps[-1].elapsed_s,
                ),
                samples=[],
            )
            print("  warmup", snaps[-1].reason, snaps[-1].time, flush=True)
        for repeat in range(count):
            order = list(schemes) if repeat % 2 == 0 else list(reversed(schemes))
            for name in order:
                snaps = transient_run(name)
                last = snaps[-1]
                result["transient"][name]["samples"].append(
                    dict(
                        success=all(sol.success for sol in snaps)
                        and len(snaps) == len(times),
                        elapsed_s=last.elapsed_s,
                        reason=last.reason,
                        accepted_steps=last.metrics["accepted_steps"],
                        rejected_steps=last.metrics["rejected_steps"],
                    )
                )
                print(
                    f"TRANSIENT {name} measured {repeat+1}/{count}: {last.elapsed_s:.3f}s success={last.success} t={last.time:.6g}",
                    flush=True,
                )
                fields = [
                    {
                        k: np.asarray(v)
                        for k, v in driven.fields(jnp.asarray(sol.state)).items()
                    }
                    for sol in snaps
                ]
                op = cached(driven, schemes[name])
                result["transient"][name].update(
                    times=[sol.time for sol in snaps],
                    fields={k: np.stack([f[k] for f in fields]) for k in fields[0]},
                    states=np.stack([sol.state for sol in snaps]),
                    inlet_interior_state=np.stack(
                        [
                            np.asarray(op.reconstruct(jnp.asarray(sol.state))[1][0])
                            for sol in snaps
                        ]
                    ),
                    metrics=[sol.metrics for sol in snaps],
                    history=snaps[-1].history,
                    x=driven.native_mesh.x_center,
                )
                save(out / "results.json", result)
        # Additional temporal refinement at fixed 200 cells, never grid sequencing.
        name = "rk3_reference"
        s = Scheme(
            flux="hll",
            reconstruction=ncfg["reconstruction"],
            time_integrator="runge-kutta-3",
            boundary_treatment="segregated",
        )
        op = cached(driven, s)
        init, info = op.complete_boundaries(jnp.asarray(seed), 0.0)
        with patch.object(drivers, "Operator", side_effect=cached):
            snaps = solve_transient(
                driven,
                times,
                s,
                cfl=ncfg["explicit_cfl"] / 2,
                initial_state=init,
                max_steps=ncfg["max_transient_steps"],
            )
        result["reference"] = dict(
            cfl=ncfg["explicit_cfl"] / 2,
            success=all(sol.success for sol in snaps) and len(snaps) == len(times),
            elapsed_s=snaps[-1].elapsed_s,
            times=[sol.time for sol in snaps],
            fields={
                k: np.stack(
                    [
                        np.asarray(driven.fields(jnp.asarray(sol.state))[k])
                        for sol in snaps
                    ]
                )
                for k in ("p", "M", "alpha_deg")
            },
        )
    after = hashes()
    result["source_unchanged"] = before == after
    save(out / "source_after.json", after)
    save(out / "results.json", result)
    if before != after:
        print(
            "Source manifest changed on disk during campaign; independent HLL residual and timestep compatibility audit is required.",
            flush=True,
        )
    from .report import build

    build()
    if not args.steady_only and not args.no_postprocess:
        from .audit import main as audit

        audit()
        if "rk3" in result["transient"]:
            from .visuals import main as visuals

            visuals()
        from .audit_html import main as audit_html

        audit_html()


if __name__ == "__main__":
    main()
