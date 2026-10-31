"""Benchmark unchanged EulerNozzle implicit marching with grid sequencing.

Run from the repository root with poetry run python -m
development.roberto.historical.multigrid.scripts.run_demo. No FAS or source edits.
"""

import os

for _thread_name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from development.roberto.historical.newton_solve.scripts.problem import SteadyNozzleProblem
from development.roberto.historical.newton_solve.scripts.pseudo_time import solve_implicit

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[3]
CASES = {"shock35": baseline.CASES[1], "shock70": baseline.CASES[2],
         "subsonic": baseline.CASES[4]}
METHODS = ("single_grid", "grid_sequence")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/"src").rglob("*.py"))}


def fields(problem, full):
    state = np.asarray(full)
    props = problem.fluid.get_state(jxp.HmassP_INPUTS, state[:, 2], state[:, 0])
    return dict(x_m=problem.x, p=state[:, 0], v_m=state[:, 1], h=state[:, 2],
                rho=np.asarray(props["rho"]), M=state[:, 1]/np.asarray(props["a"]))


def audit_state(problem, full):
    """Evaluate the saved full field, not an additionally advanced state."""
    full = jnp.asarray(full)
    physical = np.asarray(problem.operators["physical_residual"](full, full, jnp.float64(1)))
    scaled = physical.ravel()/problem.residual_scale
    mapped = np.asarray(problem.operators["state_from_interior"](full[1:-1].ravel()))
    scale = np.array([problem.p0, problem.a0, problem.h0])
    boundary_max = float(np.max(np.abs((np.asarray(full)[[0, -1]]-mapped[[0, -1]])/scale)))
    face_mass = np.asarray(problem.face_mass_flux(full))
    global_scale = problem.face_area[0]*problem.rho0*problem.a0*np.array([1., problem.a0, problem.h0])
    balance = np.sum(physical.reshape(-1, 3), axis=0)/global_scale
    props = problem.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
    valid = bool(np.isfinite(np.asarray(full)).all() and np.all(np.asarray(full)>0)
                 and all(np.isfinite(np.asarray(props[k])).all() and np.all(np.asarray(props[k])>0)
                         for k in ("rho", "a", "T")))
    f = fields(problem, full)
    error = {k: float(np.mean(np.abs(f[k][1:-1]-problem.reference[k][1:-1]))/s)
             for k, s in (("p", problem.p0), ("rho", problem.rho0), ("M", 1.))}
    downstream = np.where((problem.x[1:-2]+problem.x[2:-1])/2 > baseline.NOZZLE["L"]/2)[0]
    gradient = np.diff(f["v_m"][1:-1])/problem.dx
    index = downstream[np.argmin(gradient[downstream])]
    shock = float((problem.x[1:-2][index]+problem.x[2:-1][index])/2) if gradient[index]<0 else None
    mass = float(abs(face_mass[0]-face_mass[-1])/max(abs(face_mass[0]), 1e-300))
    rms = float(np.sqrt(np.mean(scaled**2)))
    return dict(residual_rms=rms, boundary_max=boundary_max, integrated_balance=balance,
                integrated_balance_max=float(np.max(np.abs(balance))), mass_imbalance=mass,
                admissible=valid, normalized_L1=error, analytical_pass=max(error.values())<.03,
                shock_proxy_m=shock, outlet_M=float(f["M"][-1]),
                root_success=bool(valid and rms<=1e-8 and boundary_max<=1e-8 and mass<=1e-4
                                  and np.max(np.abs(balance))<=1e-5)), physical


def transfer(previous, source_x, target):
    """Bounded primitive interpolation; target source BC completion is reused."""
    primitive = np.column_stack([np.interp(target.x, source_x, previous[:, k]) for k in range(3)])
    result = target.operators["state_from_interior"](jnp.asarray(primitive[1:-1].ravel()))
    jax.block_until_ready(result)
    if not np.isfinite(np.asarray(result)).all() or np.any(np.asarray(result)<=0):
        raise ValueError("Transferred initialization is not a positive finite primitive state")
    return result


def solve_method(problems, method, *, coarse_tolerance, max_iterations):
    """Time all stage solves, interpolation and common monitoring, not final audits."""
    selected = problems[-1:] if method=="single_grid" else problems
    stages, previous, source_x = [], None, None
    started = perf_counter()
    for problem in selected:
        transfer_started = perf_counter()
        initial = (jnp.array(problem.initial_state) if previous is None
                   else transfer(previous, source_x, problem))
        jax.block_until_ready(initial)
        transfer_s = perf_counter()-transfer_started
        is_fine = problem is problems[-1]
        result = solve_implicit(problem, initial_state=initial,
                                residual_rms_tol=1e-8 if is_fine else coarse_tolerance,
                                max_iterations=max_iterations, record_controls=True)
        # Return z -> full is ordinary source boundary completion, not another solve.
        final = problem.full_state(result["x"])
        jax.block_until_ready(final)
        previous, source_x = np.asarray(final), problem.x
        stages.append(dict(n_points=len(problem.x), dx_m=problem.dx,
                           wall_s=result["wall_s"], transfer_s=transfer_s,
                           tolerance=1e-8 if is_fine else coarse_tolerance,
                           iterations=result["iterations"],
                           trials=result["backend"]["implicit_step_trials"],
                           rejected_trials=result["backend"]["rejected_trials"],
                           history=result["history"], trial_history=result["trial_history"],
                           initial_state=np.asarray(initial), final_state=previous,
                           native_success=result["native_success"], message=result["message"]))
        if not result["native_success"]:
            break
    wall_s = perf_counter()-started
    # All EOS diagnostics and serialization are outside the benchmark timer equally.
    for stage, problem in zip(stages, selected):
        stage["diagnostics"], stage["spatial_residual"] = audit_state(problem, stage["final_state"])
        stage["fields"] = fields(problem, stage["final_state"])
    finest = stages[-1]
    return dict(wall_s=wall_s, stages=stages,
                iterations=sum(s["iterations"] for s in stages),
                fine_iterations=finest["iterations"] if finest["n_points"]==len(problems[-1].x) else 0,
                trials=sum(s["trials"] for s in stages),
                newton_corrections=sum(s["trials"] for s in stages),
                rejected_trials=sum(s["rejected_trials"] for s in stages),
                equivalent_fine_updates=sum(s["trials"]*(s["n_points"]-2)/(len(problems[-1].x)-2) for s in stages),
                root_success=bool(finest["n_points"]==len(problems[-1].x)
                                  and finest["native_success"] and finest["diagnostics"]["root_success"]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--reconstructions", nargs="+", choices=["constant", "muscl_van_leer"],
                        default=["constant", "muscl_van_leer"])
    parser.add_argument("--points", nargs="+", type=int, default=[51,101,201,401])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--coarse-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs")
    args=parser.parse_args(argv)
    if args.repeats<1 or any(n<5 for n in args.points) or sorted(set(args.points))!=args.points:
        parser.error("Require positive repeats and strictly increasing mesh sizes >=5")
    output=args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary=dict(metadata=dict(date="2026-10-28", study="EulerNozzle grid sequencing, no FAS",
        hierarchy_points=args.points, physical_cells=[n-2 for n in args.points], repeats=args.repeats,
        coarse_tolerance=args.coarse_tolerance, fine_tolerance=1e-8,
        boundary_tolerance=1e-8, mass_imbalance_tolerance=1e-4, integrated_balance_tolerance=1e-5,
        same_root_scaled_state_tolerance=1e-4, shock_match_cells=2,
        source_hashes_before=source_hashes(), source_hashes_after=None,
        timing="Sequential alternating order; one excluded complete solve per method; all stages, transfers, rejected work and common monitoring included; setup/JIT, final diagnostics and output excluded",
        inner_policy="Exactly one Newton correction per attempted implicit pseudo-step; no claim of fully converged implicit timestep",
        boundary_policy="Unchanged source EulerNozzle eliminated ghost maps, identical for both methods; no coupled-BC retrofit",
        transfer="Linear primitive interpolation in x, followed by unchanged target boundary completion; nonconservative initialization",
        versions={k:version(k) for k in ("jax", "jaxlib", "jaxprop", "numpy")},
        precision="float64", setup=[]), setups=[], runs=[])
    dump(output/"summary.json", summary)
    for case_id in args.cases:
        for reconstruction in args.reconstructions:
            started=perf_counter()
            settings=dict(baseline.SOLVER, reconstruction=reconstruction)
            problems=[SteadyNozzleProblem(CASES[case_id], n_points=n, solver_settings=settings)
                      for n in args.points]
            construction_s=perf_counter()-started
            started=perf_counter()
            for problem in problems:
                cold=problem.initial_state
                dt=problem.compute_dt(cold)[0]
                jax.block_until_ready(problem.step(cold, dt))
                jax.block_until_ready(problem._residual(jnp.asarray(problem.x0)))
                jax.block_until_ready(problem.full_state(problem.x0))
            compile_s=perf_counter()-started
            fine=problems[-1]
            summary["metadata"]["setup"].append(dict(case_id=case_id,reconstruction=reconstruction,
                                                       construction_s=construction_s, compile_s=compile_s))
            summary["setups"].append(dict(case_id=case_id,label=CASES[case_id]["label"],
                p_back_ratio=fine.p_back/fine.p0, reconstruction=reconstruction,
                n_points=len(fine.x), x_m=fine.x, area_m2=fine.area, reference=fine.reference,
                reference_fine=dict(x_m=fine.x_fine, **fine.reference_fine), settings=settings))
            for repeat in range(-1,args.repeats):
                order=METHODS if repeat%2 else METHODS[::-1]
                for method in order:
                    result=solve_method(problems,method,coarse_tolerance=args.coarse_tolerance,
                                        max_iterations=args.max_iterations)
                    result.update(case_id=case_id,reconstruction=reconstruction,method=method,
                                  repeat=repeat,excluded=repeat<0)
                    filename=f"runs/{case_id}_{reconstruction}_{method}_{'excluded' if repeat<0 else repeat:}.json"
                    result["file"]=filename
                    dump(output/filename,result)
                    summary["runs"].append({k:v for k,v in result.items() if k!="stages"})
                    summary["metadata"]["source_hashes_after"]=source_hashes()
                    summary["metadata"]["source_unchanged"]=(summary["metadata"]["source_hashes_before"]
                                                              ==summary["metadata"]["source_hashes_after"])
                    dump(output/"summary.json",summary)
                    print(f"{case_id} {reconstruction} {method} repeat={repeat} "
                          f"success={result['root_success']} updates={result['iterations']} "
                          f"fine={result['fine_iterations']} trials={result['trials']} "
                          f"time={result['wall_s']:.4f}s",flush=True)
    failed=[r for r in summary["runs"] if not r["root_success"]]
    print(f"Saved {len(summary['runs'])} complete attempts; {len(failed)} failures; "
          f"source unchanged={summary['metadata']['source_unchanged']}",flush=True)


if __name__=="__main__":
    main()
