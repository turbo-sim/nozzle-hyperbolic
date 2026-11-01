"""Untimed actual-CO2 steady Roe checks using installed production machinery.

Run from the repository root:
    poetry run python -B -m development.flux_methods.roe.scripts.verify_real_gas

This is a finite-volume convergence/conservation check, not an analytical flow
accuracy claim or a runtime comparison. Every failed attempt remains in output.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, solve_single_grid
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid

from .audit_roe import ROOT, STUDY, _git, _version


def plain(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return plain(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_manifest():
    paths = sorted((ROOT / "src/nozzle_hyperbolic").rglob("*.py"))
    result = {
        str(path.relative_to(ROOT))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in paths
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return dict(aggregate_sha256=hashlib.sha256(encoded).hexdigest(), files=result)


class AuditedCO2(jxp.Fluid):
    """Original configured host fluid behind the existing JAXProp adapter.

    Count runtime host calls and reject every non-PH numerical entry. PT is
    permitted only during model reservoir setup. No property relation is added.
    """

    def __init__(self):
        self.source = jxp.Fluid("CO2", backend="HEOS")
        self.numerical_phase = False
        self.calls = Counter()

    def __getattr__(self, key):
        return getattr(self.source, key)

    def get_state(self, pair, first, second):
        label = (
            "PH"
            if pair == jxp.HmassP_INPUTS
            else ("PT" if pair == jxp.PT_INPUTS else str(pair))
        )
        phase = "numerical" if self.numerical_phase else "setup"
        self.calls[f"{phase}:{label}"] += 1
        if self.numerical_phase and pair != jxp.HmassP_INPUTS:
            raise AssertionError(
                f"Numerical EOS query must use HmassP_INPUTS, received {pair}"
            )
        return self.source.get_state(pair, first, second)


def build(kind, cells):
    host = AuditedCO2()
    fluid = as_jax_fluid(host)
    physical = dict(
        fluid="CO2",
        backend="HEOS",
        reservoir_pressure=8e6,
        reservoir_temperature=320.0,
        back_pressure=7.98e6,
        inlet_regime="subsonic",
        outlet_regime="subsonic",
        outlet_formulation="acoustic",
        wall_sources=False,
    )
    if kind == "nozzle":
        geometry_data = dict(length=0.2, inlet_area=3e-4, throat_area=2.7e-4)
        mesh = SymmetricNozzleGeometry(**geometry_data).build_mesh(cells)
        model = NozzleModel(
            mesh,
            fluid,
            p0=physical["reservoir_pressure"],
            T0=physical["reservoir_temperature"],
            back_pressure=physical["back_pressure"],
            outlet_formulation="acoustic",
        )
    else:
        geometry_data = dict(
            length=0.1, r_in=0.2, b_in=0.005, phi=0.04, semi_divergence=0.0
        )
        mesh, native = build_annular_mesh(conical_channel(**geometry_data), cells)
        physical["inlet_alpha_radians"] = 0.15
        model = AnnularModel(
            mesh,
            native,
            fluid,
            inlet_data=dict(
                p0=physical["reservoir_pressure"],
                T0=physical["reservoir_temperature"],
                alpha_in=0.15,
            ),
            outlet_pressure=physical["back_pressure"],
            outlet_formulation="acoustic",
        )
    host.numerical_phase = True
    initial = jnp.asarray(model.initial_state())
    return model, host, initial, dict(**physical, geometry=geometry_data)


def audit(solution, model, host):
    """Original production rows, with fixed-scale host reductions/telescoping."""
    op, state = solution.operator, jnp.asarray(solution.state)
    raw = np.asarray(op.spatial_residual(state))
    bc = np.asarray(model.boundary_residual(state, 0.0))
    flux = np.asarray(op.face_flux(state))
    sources = np.asarray(model.integrated_source(state))
    volumes, scales = np.asarray(model.mesh.volumes), np.asarray(
        model.conservative_scales
    )
    normalized = raw * model.reference_time / (volumes[:, None] * scales)
    l2 = np.sqrt(np.sum((volumes / volumes.sum())[:, None] * normalized**2, axis=0))
    initial_l2 = np.asarray(solution.history[0]["equation_norm"])
    relative = l2 / np.where(initial_l2 > 0, initial_l2, 1.0)
    equation_passed = (l2 <= 1e-9) | ((initial_l2 > 1e-14) & (relative <= 1e-6))
    global_defect = np.abs(raw.sum(0) / model.global_flux_scales)
    telescoping = (
        raw.sum(0) - (flux[-1] - flux[0] - sources.sum(0))
    ) / model.global_flux_scales
    # Explicit PH host evaluation independently confirms positive properties.
    properties = [
        host.get_state(jxp.HmassP_INPUTS, float(row[-1]), float(row[0]))
        for row in np.asarray(state)
    ]
    values = {
        key: np.array([float(item[key]) for item in properties])
        for key in ("rho", "a", "T", "G")
    }
    roe_data = jax.device_get(op.roe_data(state))
    finite = bool(
        np.isfinite(raw).all()
        and np.isfinite(bc).all()
        and np.isfinite(state).all()
        and all(np.isfinite(v).all() for v in values.values())
    )
    positivity = bool(
        np.all(np.asarray(state)[:, 0] > 0)
        and all(np.all(values[key] > 0) for key in ("rho", "a", "T"))
    )
    boundary_max = float(np.max(np.abs(bc)))
    model_audit = bool(model.solution_audit(state))
    independent = dict(
        finite=finite,
        positive_pressure_density_temperature_sound_speed=positivity,
        equation_l2=l2,
        equation_relative=relative,
        equation_passed=equation_passed,
        all_equations_passed=bool(np.all(equation_passed)),
        boundary_max=boundary_max,
        global_defect=global_defect,
        boundary_passed=boundary_max <= 1e-9,
        global_passed=bool(np.max(global_defect) <= 1e-9),
        spatial_telescoping_scaled_defect=telescoping,
        scaled_l2_difference_from_driver=l2
        - np.asarray(solution.metrics["equation_l2"]),
        operator_admissible=bool(op.is_admissible(state)),
        model_audit_passed=model_audit,
        pressure_min=float(np.min(np.asarray(state)[:, 0])),
        velocity_min=float(np.min(np.asarray(state)[:, 1])),
        meridional_mach_max=float(
            np.max(np.abs(np.asarray(state)[:, 1]) / values["a"])
        ),
        rho_range=[float(values["rho"].min()), float(values["rho"].max())],
        sound_speed_range=[float(values["a"].min()), float(values["a"].max())],
        temperature_range=[float(values["T"].min()), float(values["T"].max())],
        gruneisen_range=[float(values["G"].min()), float(values["G"].max())],
        roe_all_faces_valid=bool(np.all(roe_data["valid"])),
        roe_sound_speed_squared_min=float(np.min(roe_data["sound_speed_squared"])),
        roe_pressure_secant_defect_max=float(
            np.max(np.abs(roe_data["pressure_secant_defect"]))
        ),
    )
    arrays = dict(
        state=np.asarray(state),
        raw_spatial_residual=raw,
        boundary_residual=bc,
        face_flux=flux,
        integrated_source=sources,
        points=np.asarray(model.mesh.points),
        faces=np.asarray(model.mesh.faces),
        volumes=volumes,
        **{f"property_{key}": value for key, value in values.items()},
    )
    return independent, arrays


def run(kind, args, destination, before):
    directory = destination / kind
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    result = dict(
        case=kind,
        cells=args.cells,
        success=False,
        reason="not_started",
        timed_comparison=False,
        analytical_reference=False,
    )
    model = host = None
    try:
        model, host, initial, physical = build(kind, args.cells)
        scheme = Scheme(
            flux="roe",
            reconstruction="constant",
            delta_factor=0.1,
            time_integrator="euler-implicit",
            boundary_treatment="coupled",
            newton_iterations=10,
            newton_backtracking=False,
        )
        settings = SteadySettings(
            residual=ResidualPolicy(),
            cfl=0.2,
            cfl_growth=1.2,
            cfl_max=1e6,
            max_steps=args.max_steps,
            max_retries=8,
        )
        result.update(
            physical=physical,
            scheme=asdict(scheme),
            settings=asdict(settings),
            initial_state="model default cold initialization; no analytical or numerical warm start",
            roe_metadata=model.roe_metadata,
        )
        save(directory / "case_setup.json", result)
        np.savez_compressed(directory / "initial.npz", state=np.asarray(initial))

        def callback(row):
            history.append(row)
            step = int(row["step"])
            if row["phase"] == "rejected" or step % 10 == 0 or step == 1:
                print(
                    f'{kind}: step={step} phase={row["phase"]} '
                    f'Rrel={row.get("overall_relative", float("nan")):.3e} '
                    f'CFL={row.get("requested_cfl", row["cfl"]):.3e} '
                    f'inner={row.get("inner_iterations", 0)}',
                    flush=True,
                )
                save(
                    directory / "checkpoint.json",
                    dict(
                        last_row=row,
                        history=history,
                        elapsed_s=time.perf_counter() - started,
                        runtime_eos_calls=dict(host.calls),
                    ),
                )
            if time.perf_counter() - started > args.max_wall_seconds:
                raise TimeoutError(
                    "Untimed verification wall budget exceeded; no convergence claimed"
                )

        solution = solve_single_grid(
            model, scheme, settings, initial_state=initial, callback=callback
        )
        independent, arrays = audit(solution, model, host)
        np.savez_compressed(directory / "final.npz", **arrays)
        independently_passed = all(
            independent[key]
            for key in (
                "finite",
                "positive_pressure_density_temperature_sound_speed",
                "all_equations_passed",
                "boundary_passed",
                "global_passed",
                "operator_admissible",
                "model_audit_passed",
                "roe_all_faces_valid",
            )
        )
        result.update(
            success=bool(solution.success and independently_passed),
            reason=solution.reason,
            driver_success=bool(solution.success),
            independent_audit_passed=independently_passed,
            driver_metrics=solution.metrics,
            original_residual_audit=independent,
            elapsed_s_including_compile_audit=time.perf_counter() - started,
            elapsed_note="Diagnostic cost only; external workloads present; not a timing comparison",
        )
    except Exception as error:
        result.update(
            reason="verification_exception",
            exception=repr(error),
            traceback=traceback.format_exc(),
            elapsed_s_including_compile_audit=time.perf_counter() - started,
        )
    rows = [row for row in history if row["phase"] != "initial"]
    result["work"] = dict(
        accepted_steps=sum(row["phase"] == "accepted" for row in rows),
        rejected_steps=sum(row["phase"] == "rejected" for row in rows),
        newton_corrections_all_trials=sum(
            row.get("inner_iterations", 0) for row in rows
        ),
        rejected_newton_trials=sum(
            row.get("rejected_newton_trials", 0) for row in rows
        ),
        runtime_eos_calls=dict(host.calls) if host is not None else {},
        eos_call_note="Actual host entries including setup, solver, monitor and final audit calls",
    )
    result["source_unchanged"] = source_manifest() == before
    if not result["source_unchanged"]:
        result["success"] = False
        result["source_mismatch"] = "Production source changed during verification"
    save(directory / "history.json", history)
    save(directory / "result.json", result)
    print(
        f'{kind}: SUCCESS={result["success"]} reason={result["reason"]} work={result["work"]}',
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=25)
    parser.add_argument("--case", choices=("nozzle", "annular", "both"), default="both")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs/real_gas_steady"
    )
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(STUDY.resolve()):
        raise ValueError("All verification outputs must stay inside the Roe study")
    destination.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    save(destination / "source_manifest.json", before)
    setup = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=_git("rev-parse", "HEAD"),
        versions={
            key: _version(key)
            for key in ("jax", "jaxlib", "jaxprop", "CoolProp", "numpy")
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_aggregate_sha256=before["aggregate_sha256"],
        purpose="Untimed actual-CO2 PH steady residual/conservation verification; no analytical accuracy or speed claim",
        external_workload_note="Concurrent external CPU load can affect elapsed cost, which is not ranked",
    )
    save(destination / "setup.json", setup)
    kinds = ("nozzle", "annular") if args.case == "both" else (args.case,)
    results = [run(kind, args, destination, before) for kind in kinds]
    final = dict(
        **setup,
        results=results,
        all_success=all(row["success"] for row in results),
        source_unchanged=source_manifest() == before,
    )
    save(destination / "summary.json", final)
    return 0 if final["all_success"] and final["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
