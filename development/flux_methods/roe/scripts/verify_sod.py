"""Untimed production-Roe Sod verification and conservative refinement audit.

Run from the repository root with the existing project environment:
    poetry run python -B -m development.flux_methods.roe.scripts.verify_sod

The physical setup is defined below, independently of retired development
studies. Reference cell averages are independently integrated from the installed
exact Sod sampler. Physical values and numerical defaults are unchanged.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, solve_transient

from .audit_roe import ROOT, STUDY, _fingerprints, _git, _json_safe, _version


# Keep the four integrator/boundary combinations used by this Roe verification.
METHODS = (
    dict(
        id="euler_explicit",
        label="Explicit Euler · segregated BC",
        integrator="euler-explicit",
        boundary_treatment="segregated",
    ),
    dict(
        id="rk3",
        label="SSPRK3 · segregated BC",
        integrator="runge-kutta-3",
        boundary_treatment="segregated",
    ),
    dict(
        id="implicit_coupled",
        label="Implicit Euler · coupled BC",
        integrator="euler-implicit",
        boundary_treatment="coupled",
    ),
    dict(
        id="implicit_segregated",
        label="Implicit Euler · segregated BC",
        integrator="euler-implicit",
        boundary_treatment="segregated",
    ),
)
PHYSICAL_SETTINGS = dict(
    sod=dict(
        length=1.0,
        diaphragm=0.5,
        left=dict(rho=1.0, u=0.0, p=100000.0),
        right=dict(rho=0.125, u=0.0, p=10000.0),
        end_acoustic_times=0.2,
    )
)


def build_sod_case(settings):
    """Build the unchanged Sod initial states and physical output times.

    Parameters
    ----------
    settings : dict
        Cell/frame counts and ``sod`` inputs in SI units. The end time is
        measured in acoustic transit times based on the left initial state.

    Returns
    -------
    model : NozzleModel
        Constant-area perfect-gas model with transmissive endpoint traces.
    case : dict
        Uniformly spaced physical output times under ``times``.
    """
    fluid = jxp.FluidPerfectGas("Air")
    config = settings["sod"]
    mesh = ConstantAreaGeometry(length=config["length"], area=1.0).build_mesh(
        settings["cells"]
    )
    left, right = config["left"], config["right"]
    mask = mesh.points < config["diaphragm"]
    pressure = np.where(mask, left["p"], right["p"])
    density = np.where(mask, left["rho"], right["rho"])
    velocity = np.where(mask, left["u"], right["u"])
    state = fluid.get_state(
        jxp.DmassP_INPUTS, jnp.asarray(density), jnp.asarray(pressure)
    )
    initial = jnp.stack(
        (jnp.asarray(pressure), jnp.asarray(velocity), state["h"]), axis=-1
    )
    left_state = fluid.get_state(jxp.DmassP_INPUTS, left["rho"], left["p"])
    model = NozzleModel(
        mesh,
        fluid,
        p0=left["p"],
        T0=float(left_state["T"]),
        transmissive=True,
        initial_state=initial,
    )
    time_scale = config["length"] / float(left_state["a"])
    times = np.linspace(
        0.0, config["end_acoustic_times"] * time_scale, settings["frames"]
    )
    return model, dict(times=times)


def fingerprints():
    result = _fingerprints()
    extra = [
        Path(__file__),
        ROOT / "src/nozzle_hyperbolic/models/euler_nozzle/analytical/shock_tube.py",
        ROOT / "src/nozzle_hyperbolic/solvers/fvm/drivers.py",
        ROOT / "src/nozzle_hyperbolic/solvers/time_marching/explicit.py",
        ROOT / "src/nozzle_hyperbolic/solvers/time_marching/newton.py",
        ROOT / "src/nozzle_hyperbolic/solvers/boundaries.py",
    ]
    for path in extra:
        if path.is_file():
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def save(path, data):
    def serializable(value):
        if isinstance(value, (np.ndarray, jax.Array)):
            return serializable(np.asarray(value).tolist())
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return _json_safe(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(data), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reference_conservative(model, physical, time, order):
    """Gauss cell averages of the independent installed exact solution.

    The reference uses DmassP to convert exact rho,p to h, and does not use
    numerical Roe states, matrices, fluxes, or model storage callbacks.
    """
    points, weights = np.polynomial.legendre.leggauss(order)
    faces = np.asarray(model.mesh.faces)
    x = (
        0.5 * (faces[:-1, None] + faces[1:, None])
        + 0.5 * np.diff(faces)[:, None] * points
    )
    left, right = physical["left"], physical["right"]
    rho, u, p = sod_exact(
        x,
        time,
        physical["diaphragm"],
        left["rho"],
        left["u"],
        left["p"],
        right["rho"],
        right["u"],
        right["p"],
        float(model.fluid.constants.gamma),
    )
    h = np.asarray(model.fluid.get_state(jxp.DmassP_INPUTS, rho, p)["h"])
    conservative = np.stack((rho, rho * u, rho * (h + 0.5 * u * u) - p), -1)
    return 0.5 * np.einsum("j,ijk->ik", weights, conservative)


def run_one(cells, method, physical_settings):
    settings = {**physical_settings, "cells": cells, "frames": 11}
    model, case = build_sod_case(settings)
    scheme = Scheme(
        flux="roe",
        reconstruction="constant",
        delta_factor=0.1,
        time_integrator=method["integrator"],
        boundary_treatment=method["boundary_treatment"],
        newton_iterations=10,
        newton_backtracking=False,
        newton_update_tolerance=1e-10,
        newton_residual_tolerance=1e-12,
        jacobian_mode="colored",
        linear_solver="banded",
        filter_strength=0.0,
    )
    setup = Operator(model, scheme)
    initial, info = setup.complete_boundaries(jnp.asarray(model.initial_state()), 0.0)
    info = jax.device_get(info)
    if not bool(info.converged and info.admissible):
        raise RuntimeError(f"Initial boundary completion failed: {info}")
    if not np.array_equal(initial[1:-1], model.initial_state()[1:-1]):
        raise AssertionError("Boundary completion changed initial physical cells")
    snapshots = solve_transient(
        model,
        case["times"][1:],
        scheme,
        cfl=0.5,
        initial_state=initial,
        max_steps=30000,
    )
    if not snapshots:
        raise RuntimeError("No transient output was returned")
    last = snapshots[-1]
    volumes, scales = np.asarray(model.mesh.volumes), np.asarray(
        model.conservative_scales
    )
    content_scale = volumes.sum() * scales
    states = [np.asarray(initial)] + [snapshot.state for snapshot in snapshots]
    times = [0.0] + [snapshot.time for snapshot in snapshots]
    initial_totals = np.asarray(setup.storage(initial[1:-1])).sum(0)
    final_totals = np.asarray(setup.storage(jnp.asarray(last.state[1:-1]))).sum(0)
    accepted = [row for row in last.history if row["accepted"]]
    summed_transfer = sum(
        (np.asarray(row["integrated_transfer"]) for row in accepted), np.zeros(3)
    )
    independent_budget = (
        final_totals - initial_totals - summed_transfer
    ) / content_scale
    physical = settings["sod"]
    # Before any wave reaches an end, mass/energy transfer is zero and the
    # constant pressure-force difference is the exact momentum transfer.
    analytic_transfer = (
        np.asarray([0.0, physical["left"]["p"] - physical["right"]["p"], 0.0])
        * last.time
    )
    analytic_budget = (
        final_totals - initial_totals - analytic_transfer
    ) / content_scale
    reference = reference_conservative(model, physical, last.time, 128)
    reference_coarse = reference_conservative(model, physical, last.time, 64)
    actual = np.asarray(model.cell_conserved(jnp.asarray(last.state[1:-1])))
    delta = (actual - reference) / scales
    errors = dict(
        L1=np.average(np.abs(delta), axis=0, weights=volumes).tolist(),
        L2=np.sqrt(np.average(delta**2, axis=0, weights=volumes)).tolist(),
        Linf=np.max(np.abs(delta), axis=0).tolist(),
    )
    quadrature_difference = np.average(
        np.abs(reference - reference_coarse) / scales, axis=0, weights=volumes
    )
    fields = [model.fields(jnp.asarray(state)) for state in states]
    minima = {
        key: min(float(np.min(value[key])) for value in fields)
        for key in ("p", "rho", "T")
    }
    max_boundary = max(
        float(np.max(np.abs(setup.boundary_residual(jnp.asarray(state), time))))
        for state, time in zip(states, times)
    )
    max_budget = max(
        float(np.max(np.abs(snapshot.metrics["conservation_budget"])))
        for snapshot in snapshots
    )
    success = bool(
        len(snapshots) == len(case["times"]) - 1
        and all(snapshot.success for snapshot in snapshots)
    )
    passed = bool(
        success
        and last.time == case["times"][-1]
        and max_boundary <= 1e-9
        and max_budget <= 1e-9
        and np.max(np.abs(independent_budget)) <= 1e-9
        and np.max(np.abs(analytic_budget)) <= 1e-8
        and all(value > 0 for value in minima.values())
        and max(errors["L1"]) < 0.08
    )
    return dict(
        cells=cells,
        method=method,
        scheme=asdict(scheme),
        passed=passed,
        success=success,
        reason=last.reason,
        time=last.time,
        target_time=float(case["times"][-1]),
        source_timing_claim="untimed verification; elapsed not used as a benchmark",
        accepted_steps=len(accepted),
        rejected_steps=len(last.history) - len(accepted),
        outer_iterations=sum(row["inner_iterations"] for row in accepted),
        maximum_reported_cfl=max((row["cfl"] for row in accepted), default=0.0),
        maximum_boundary_defect=max_boundary,
        maximum_reported_conservation_defect=max_budget,
        independent_summed_transfer_budget=independent_budget.tolist(),
        exact_pre_boundary_interaction_budget=analytic_budget.tolist(),
        initial_conserved_totals=initial_totals.tolist(),
        final_conserved_totals=final_totals.tolist(),
        summed_numerical_transfer=summed_transfer.tolist(),
        exact_transfer=analytic_transfer.tolist(),
        conservative_errors=errors,
        reference_quadrature_64_vs_128_normalized_L1=quadrature_difference.tolist(),
        minimum_over_11_output_frames=minima,
        output_times=times,
        final_state=last.state.tolist(),
        exact_final_conservative_cell_averages=reference.tolist(),
        physical_cell_centers=np.asarray(model.mesh.centers).tolist(),
        history=last.history,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs/roe_transient.json"
    )
    args = parser.parse_args()
    before = fingerprints()
    settings = PHYSICAL_SETTINGS
    data = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        case="Sod shock tube",
        numerical_method="production Roe, constant reconstruction",
        source=dict(
            git_head=_git("rev-parse", "HEAD"),
            branch=_git("branch", "--show-current"),
            fingerprints_sha256=before,
        ),
        versions={
            name: _version(name)
            for name in (
                "jax",
                "jaxlib",
                "jaxprop",
                "CoolProp",
                "numpy",
                "nozzle-hyperbolic",
            )
        },
        settings=dict(
            grids=args.grids,
            cfl=0.5,
            frames=11,
            max_steps=30000,
            physical=settings["sod"],
            initialization="same canonical left/right physical data on every grid; no continuation",
            boundary_conditions="transmissive independent algebraic traces",
            equation_order=["mass", "meridional momentum", "total energy"],
            normalized_error="volume-weighted L1/L2 and maxabs of conservative cell averages divided by fixed model conservative scales",
            reference="installed exact perfect-gas Sod sampler,128-point Gauss cell quadrature,64-point sensitivity reported",
            independent_budget="total content change minus summed accepted transfers; separately exact [0,(pL-pR)*t,0] transfer before boundary interaction",
        ),
        records=[],
        refinement=[],
        implicit_boundary_comparisons=[],
        completed=False,
        limitations=[
            "Perfect-gas analytical wave test validates time integration and discontinuous flow, not real-gas flow behavior.",
            "Reported minima cover11 saved physical output frames; every accepted stage/trial additionally passes production admissibility.",
            "This is untimed verification under concurrent workload; no solver-performance claim is made.",
            "Gauss quadrature samples exact shock/contact jumps; its64-vs128 sensitivity is retained and is much smaller than reported discretization error.",
        ],
    )
    save(args.output, data)
    for cells in args.grids:
        for method in METHODS:
            try:
                record = run_one(cells, method, settings)
            except Exception as error:
                record = dict(
                    cells=cells,
                    method=method,
                    passed=False,
                    success=False,
                    exception=repr(error),
                    traceback=traceback.format_exc(),
                )
            data["records"].append(record)
            save(args.output, data)
            print(
                json.dumps(
                    dict(
                        cells=cells,
                        method=method["id"],
                        passed=record["passed"],
                        steps=record.get("accepted_steps"),
                        rejections=record.get("rejected_steps"),
                        L1=record.get("conservative_errors", {}).get("L1"),
                        reason=record.get("reason", record.get("exception")),
                    )
                ),
                flush=True,
            )
    for method in METHODS:
        rows = sorted(
            (
                row
                for row in data["records"]
                if row["method"]["id"] == method["id"] and "conservative_errors" in row
            ),
            key=lambda r: r["cells"],
        )
        for coarse, fine in zip(rows, rows[1:]):
            a, b = np.asarray(coarse["conservative_errors"]["L1"]), np.asarray(
                fine["conservative_errors"]["L1"]
            )
            data["refinement"].append(
                dict(
                    method=method["id"],
                    coarse_cells=coarse["cells"],
                    fine_cells=fine["cells"],
                    L1_ratio=(a / b).tolist(),
                    observed_order=(
                        np.log(a / b) / np.log(fine["cells"] / coarse["cells"])
                    ).tolist(),
                    L1_reduced_in_all_equations=bool(np.all(b < a)),
                )
            )
    for cells in args.grids:
        rows = {
            row["method"]["id"]: row
            for row in data["records"]
            if row["cells"] == cells and "final_state" in row
        }
        if {"implicit_coupled", "implicit_segregated"}.issubset(rows):
            a, b = (rows[key] for key in ("implicit_coupled", "implicit_segregated"))
            # PH primitive difference is normalized by canonical initial scales.
            pscale = settings["sod"]["left"]["p"]
            fluid = jxp.FluidPerfectGas("Air")
            sound = float(
                fluid.get_state(
                    jxp.DmassP_INPUTS, settings["sod"]["left"]["rho"], pscale
                )["a"]
            )
            error = np.max(
                np.abs(np.asarray(a["final_state"]) - np.asarray(b["final_state"]))
                / np.asarray([pscale, sound, sound**2]),
                axis=0,
            )
            data["implicit_boundary_comparisons"].append(
                dict(
                    cells=cells,
                    normalized_max_primitive_difference=error.tolist(),
                    passed=bool(np.max(error) < 1e-8),
                )
            )
    after = fingerprints()
    data["source"]["files_changed_during_verification"] = [
        name for name in before if before[name] != after.get(name)
    ]
    data["completed"] = True
    data["passed"] = bool(
        all(row["passed"] for row in data["records"])
        and all(row["L1_reduced_in_all_equations"] for row in data["refinement"])
        and all(row["passed"] for row in data["implicit_boundary_comparisons"])
        and not data["source"]["files_changed_during_verification"]
    )
    save(args.output, data)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                passed=data["passed"],
                runs=len(data["records"]),
            ),
            indent=2,
        ),
        flush=True,
    )
    if not data["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
