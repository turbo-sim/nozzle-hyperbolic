"""Three measured HLL grid hierarchies: reset CFL versus carry accepted CFL."""

from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, SteadySettings, drivers
from nozzle_hyperbolic.verification import build_case, load_case

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
GRIDS = (50, 100, 200, 400)
POLICIES = ("reset", "carry")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for base in (ROOT / "src/nozzle_hyperbolic", STUDY / "scripts")
            for p in sorted(base.rglob("*")) if p.suffix in (".py", ".yaml")}


def work(history):
    return dict(accepted=sum(r.get("phase") == "accepted" for r in history),
                rejected=sum(r.get("phase") == "rejected" for r in history),
                corrections=sum(r.get("inner_iterations", 0) for r in history),
                backtracks=sum(r.get("rejected_newton_trials", 0) for r in history))


def audit(solution, problem, policy):
    """Fresh original residual, reduced independently with NumPy after timing."""
    op = solution.operator
    raw = np.asarray(op.spatial_residual(jnp.asarray(solution.state)))
    volumes = np.asarray(problem.mesh.volumes)
    scaled = raw * problem.model.reference_time / (volumes[:, None] * problem.model.conservative_scales)
    norms = np.sqrt(np.sum(volumes[:, None] / volumes.sum() * scaled**2, axis=0))
    initial = np.asarray(solution.history[0]["equation_norm"])
    active = initial > policy.initial_floor
    relative = norms / np.where(initial > 0, initial, 1.)
    passed = (norms <= policy.absolute_tolerance) | (active & (relative <= policy.relative_tolerance))
    bc = np.asarray(op.boundary_residual(jnp.asarray(solution.state), 0.))
    global_defect = np.abs(raw.sum(axis=0) / problem.model.global_flux_scales)
    branch = bool(problem.model.solution_audit(solution.state))
    agrees = bool(np.allclose(norms, solution.metrics["equation_norm"], rtol=1e-9, atol=1e-15))
    return dict(passed=bool(solution.success and np.all(passed) and np.isfinite(scaled).all()
                           and np.max(np.abs(bc)) <= policy.boundary_tolerance
                           and max(global_defect) <= policy.global_tolerance and branch and agrees),
                equation_l2=norms, equation_relative=relative, boundary_max=np.max(np.abs(bc)),
                global_defect=global_defect, branch_passed=branch, reduction_agrees=agrees)


def hierarchy(physical, scheme, settings, operators, samplers, policy):
    original_solve = drivers.solve_single_grid
    carried = None
    captured, control = [], {}

    def selected_solve(model, selected_scheme, selected_settings, **kwargs):
        nonlocal carried
        initial_cfl = settings.cfl
        if policy == "carry" and kwargs.get("initial_state") is not None and carried is not None:
            initial_cfl = min(carried, settings.cfl_max)
        solution = original_solve(model, selected_scheme, replace(selected_settings, cfl=initial_cfl), **kwargs)
        accepted = [row for row in solution.history if row.get("phase") == "accepted"]
        last = float(accepted[-1]["cfl"]) if accepted else initial_cfl
        control[id(solution)] = dict(initial_cfl=initial_cfl, last_accepted_cfl=last)
        carried = last if solution.success else None
        return solution

    def on_stage(model, solution, metadata):
        captured.append((model, solution, metadata[-1], control[id(solution)]))

    jax.effects_barrier()
    started = perf_counter()
    with patch.object(drivers, "Operator", lambda model, selected: operators[model.mesh.n_cells]), \
         patch.object(drivers, "sampler", lambda op, residual: samplers[id(op)]), \
         patch.object(drivers, "solve_single_grid", selected_solve):
        final, metadata = drivers.solve_grid_sequence(
            lambda n: physical[n].model, GRIDS, scheme, settings,
            on_stage=on_stage, quantity_evaluator_factory=lambda model: None,
            carry_cfl=False)  # Policy is applied by this study's wrapper above.
    jax.block_until_ready(final.state)
    elapsed = perf_counter() - started
    stages = []
    for model, solution, stage, controls in captured:
        problem = physical[model.mesh.n_cells]
        histories = [solution.history] + [attempt["history"] for attempt in stage["attempts"]]
        counts = {key: sum(work(h)[key] for h in histories) for key in work(solution.history)}
        stages.append(dict(cells=model.mesh.n_cells, success=solution.success, reason=solution.reason,
            **controls, work=counts, stage_elapsed_s=stage["elapsed_s"], solve_elapsed_s=stage["solve_elapsed_s"],
            initialization=stage["initialization"], attempts=stage["attempts"],
            history=solution.history, state=solution.state, metrics=solution.metrics,
            audit=audit(solution, problem, settings.residual)))
    problem = physical[400]
    fields = problem.numerical_fields(final.state)
    reference = problem.cell_reference()
    weights = problem.mesh.volumes / sum(problem.mesh.volumes)
    p_error = float(np.sum(weights * np.abs(fields["p"] - reference["fields"]["p"])) / problem.model.p0)
    crossings = np.where((fields["M"][:-1] > 1) & (fields["M"][1:] <= 1))[0]
    shock = float(np.mean(problem.mesh.centers[crossings[0]:crossings[0]+2])) if len(crossings) else None
    return dict(policy=policy, elapsed_s=elapsed, success=bool(all(s["success"] for s in stages)),
        audits_passed=bool(all(s["audit"]["passed"] for s in stages)), stages=stages,
        total_work={key: sum(s["work"][key] for s in stages) for key in stages[0]["work"]},
        fallback_count=sum(bool(s["attempts"]) for s in stages),
        coordinate=problem.mesh.centers, pressure=fields["p"], mach=fields["M"],
        reference_pressure=reference["fields"]["p"], reference_shock_m=problem.shock_position,
        shock_m=shock, pressure_l1_over_p0=p_error, variable_scales=problem.model.variable_scales)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = SteadySettings()
    source = fingerprint()
    metadata = dict(grids=GRIDS, policies=POLICIES, measured_repetitions=3, excluded_warmups_per_policy=1,
        case_ids=["euler_nozzle/shock_35", "euler_nozzle/shock_70"],
        reconstructions=["constant", "muscl_van_leer"], settings=asdict(settings),
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=source, platform=platform.platform(), processor=platform.processor(), python=sys.version,
        packages={name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy")},
        backend=jax.default_backend(), devices=[str(v) for v in jax.devices()], x64=bool(jax.config.jax_enable_x64),
        threads={name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        timing_boundary="prebound kernels; whole hierarchy including interpolation/monitoring/rejections/cold retries; synchronized; excludes audits/output",
        cases={key: load_case(key).to_dict() for key in ("euler_nozzle/shock_35", "euler_nozzle/shock_70")})
    save(args.output / "metadata.json", metadata)
    index = []
    for case_id in metadata["case_ids"]:
        for recon in metadata["reconstructions"]:
            binding_start = perf_counter()
            physical = {n: build_case(case_id, cells=n) for n in GRIDS}
            scheme = Scheme(flux="hll", reconstruction=recon, newton_backtracking=True)
            operators = {n: Operator(p.model, scheme) for n, p in physical.items()}
            samplers = {id(op): drivers.sampler(op, settings.residual) for op in operators.values()}
            binding_s = perf_counter() - binding_start
            for phase, repeats in (("warmup", 1), ("measured", 3)):
                for repeat in range(repeats):
                    order = POLICIES if repeat % 2 == 0 else tuple(reversed(POLICIES))
                    for policy in order:
                        print(f"{case_id} {recon} {phase} {repeat+1} {policy}: starting", flush=True)
                        record = hierarchy(physical, scheme, settings, operators, samplers, policy)
                        record.update(case_id=case_id, reconstruction=recon, phase=phase, repeat=repeat+1,
                                      scheme=asdict(scheme), binding_s=binding_s)
                        name = f"{case_id.split('/')[-1]}_{recon}_{phase}_{repeat+1}_{policy}.json"
                        save(args.output / name, record)
                        index.append(name)
                        save(args.output / "index.json", index)
                        print(f"  {record['elapsed_s']:.6f}s success={record['success']} "
                              f"audit={record['audits_passed']} work={record['total_work']}", flush=True)
            del physical, operators, samplers
            jax.clear_caches()
    assert fingerprint() == source, "Source changed during measurement"
    print("Campaign complete: 24 measured hierarchies, 8 excluded warmups.", flush=True)


if __name__ == "__main__":
    main()
