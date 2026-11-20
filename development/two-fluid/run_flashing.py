"""Super Moby Dick flashing-water cases (NICFD single_component) with the k_free/smooth switches.

Run from the repository root:
    poetry run python -B development/two-fluid/run_flashing.py <variant> <cells> [budget_s] [start_from]
variant: k_inverse | k_inverse_smooth | k_free | k_free_smooth. Budget is the wall limit per regime (default 900 s).
start_from (optional): a previous run folder in outputs/flashing (same cells), e.g. k_free_N400; its saved
<regime>.npz state replaces the published-profile initial guess. Results then go to <variant>_N<cells>_from_<start_from>.
The case is built by cases/NICFD2026/scripts/case.py from the unchanged published settings and the
published-profile initial guess; only CompatibilityConfig(k_free, smooth) changes. The steady settings are
the same for every variant: case CFL and growth, 20 Newton corrections, 20 backtracks, at most 2000 steps.
Results: outputs/flashing/<variant>_N<cells>/<regime>.{json,npz}.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
from nozzle_hyperbolic.solvers.compilation import configure_compilation_cache

HERE = Path(__file__).resolve().parent
configure_compilation_cache(HERE / "outputs" / "jax_cache")
import numpy as np

from nozzle_hyperbolic.models.two_fluid import TwoFluidModel
from nozzle_hyperbolic.solvers.fvm import Solution, SteadyWorkspace, solve_single_grid
from nozzle_hyperbolic.solvers.convergence_output import solution_summary, strict_json
from cases.NICFD2026.scripts.case import REGIMES, build_case, prepare_fluids, specification
from cases.NICFD2026.scripts.metrics import independent_audit, profile_kernel, reference_errors, regime_audit

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "k_inverse"
CELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
START_FROM = sys.argv[4] if len(sys.argv) > 4 else None
K_FREE, SMOOTH = {"k_inverse": (False, False), "k_inverse_smooth": (False, True),
                  "k_free": (True, False), "k_free_smooth": (True, True)}[VARIANT]
OUTPUT = HERE / "outputs" / "flashing" / (f"{VARIANT}_N{CELLS}" + (f"_from_{START_FROM}" if START_FROM else ""))
OUTPUT.mkdir(parents=True, exist_ok=True)

for regime in REGIMES:
    # 1. Published case, then the same model with only the compatibility switches changed
    config = specification("single_component", regime)
    fluids, _ = prepare_fluids(config, "single_component")
    base, scheme, settings, effective = build_case(config, fluids, CELLS, family="single_component",
                                                   initialization="reference", newton_iterations=20, max_steps=2000)
    compatibility = replace(base.compatibility, k_free=K_FREE, smooth=SMOOTH)
    start = base.initial_state()
    if START_FROM:
        start = np.load(HERE / "outputs" / "flashing" / START_FROM / f"{regime}.npz")["state"]
    model = TwoFluidModel(base.mesh, base.fluid_g, base.fluid_l, initial_state=start,
                          boundaries=base.boundaries, closures=base.closures, compatibility=compatibility,
                          stabilization=base.stabilization, pins=base.pins, virtual_mass=base.virtual_mass,
                          source_scale=base.source_scale, area_source=base.area_source)
    scheme = replace(scheme, max_backtracks=20)
    workspace = SteadyWorkspace(model, scheme, settings.residual)
    initial = model.initial_state()
    print(f"\n=== {VARIANT} N{CELLS} {regime}: CFL {settings.cfl}, growth {settings.cfl_growth}, "
          f"budget {BUDGET:.0f} s ===", flush=True)

    # 2. Solve with a wall budget; keep the last accepted state if the budget stops it
    last = dict(state=initial, candidate=initial)
    original_step = workspace.operator.step

    def retained_step(*values):
        outcome = original_step(*values)
        last["candidate"] = outcome[0]
        return outcome

    workspace.operator.step = retained_step
    started, printed, history = time.perf_counter(), [0.0], []

    def progress(row):
        elapsed = time.perf_counter() - started
        history.append(row)
        if row["phase"] != "initial" and row["accepted"]:
            last["state"] = last["candidate"]
        if elapsed - printed[0] >= 30:
            printed[0] = elapsed
            print(f"  step {row['step']} {row['phase']} cfl {row.get('cfl')} "
                  f"R {max(row.get('equation_norm', [np.nan])):.3e} {elapsed:.0f} s", flush=True)
        if elapsed > BUDGET:
            raise TimeoutError("wall budget")

    try:
        result = solve_single_grid(model, scheme, settings, initial_state=initial, workspace=workspace, callback=progress)
    except TimeoutError:
        result = Solution(np.asarray(last["state"]), False, "wall_budget", history,
                          workspace.sample(last["state"], history[0]["equation_norm"]),
                          elapsed_s=time.perf_counter() - started)
    finally:
        workspace.operator.step = original_step

    # 3. Case audits (regime, reference profile errors, independent residual audit) and output
    fields = {key: np.asarray(value) for key, value in profile_kernel(model)(result.state).items()}
    summary = solution_summary(result)
    summary["elapsed_s"] = time.perf_counter() - started
    record = dict(
        variant=VARIANT, cells=CELLS, regime=regime, budget_s=BUDGET, summary=summary,
        regime_audit=regime_audit(fields, regime),
        reference_errors=reference_errors(fields, model, "single_component", regime, config),
        audit=independent_audit(workspace, result.state, result.history[0]["equation_norm"], result.metrics),
        settings=asdict(settings), scheme=asdict(scheme), compatibility=asdict(compatibility),
        history=result.history,
    )
    (OUTPUT / f"{regime}.json").write_text(strict_json(record), encoding="utf-8")
    np.savez_compressed(OUTPUT / f"{regime}.npz", state=result.state, **fields)
    print(f"  -> {summary['reason']}: success {summary['success']}, accepted {summary['accepted_steps']}, "
          f"rejected {summary['rejected_steps']}, regime passed {record['regime_audit']['passed']}, "
          f"max Mach {record['regime_audit']['maximum_mach']:.3f}, {summary['elapsed_s']:.0f} s", flush=True)
