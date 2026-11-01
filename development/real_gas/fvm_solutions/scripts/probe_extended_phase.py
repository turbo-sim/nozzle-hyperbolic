"""Isolated Lettieri-adapted N50 Rusanov probe with a wider pressure table.

This diagnostic writes only its unique table/cache and diagnostic JSON. It
does not alter the active campaign profile, prepared reference, or run files.
"""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
import json
import traceback

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, ResidualPolicy
from development.real_gas.co2_nozzle_reference.scripts.thermodynamics import BicubicCO2
from development.real_gas.co2_nozzle_reference.scripts.reference import Nozzle, SmoothPath, sample_profile
from .problem import CaseContext
from .audit import evaluate_solution
from .execution import budgeted_solve, CONTROLS
from .common import BASELINE, STUDY, OUTPUT, identity, save, source_manifest


def main():
    start = perf_counter()
    profile_bytes_before = (STUDY / "settings.json").read_bytes()
    active_reference = OUTPUT / "reference_summary.json"
    reference_bytes_before = active_reference.read_bytes()
    profile = json.loads(profile_bytes_before)
    source_before = source_manifest()
    destination = OUTPUT / "diagnostics/extended_phase_probe.json"
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Controlled Lettieri adapted N50 Rusanov constant probe: p_max 9 -> 18 MPa",
        active_profile=profile, source_before=source_before,
        execution_controls=dict(CONTROLS), success=False, verified=False,
    )
    try:
        definitions = json.loads((BASELINE / "cases.json").read_text(encoding="utf-8"))
        original = next(case for case in definitions["cases"] if case["id"] == "lettieri_equilibrium")
        case = deepcopy(original)
        case["table"].update(name="co2_lettieri_fvm_100x100", p_max=18_000_000.0)
        eos = BicubicCO2(case["table"], OUTPUT)
        nozzle = Nozzle(**(definitions["geometry"] | case.get("geometry_override", {})))
        reservoir = eos.reservoir(case["p0"], case["T0"])
        upstream = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                              eos.settings["p_min"] * 1.001)
        throat = upstream.choke()
        mdot = nozzle.At * throat["G"]
        adapted = sample_profile(upstream, nozzle, mdot, stations=1001)
        back_pressure = float(adapted["p"][-1])
        reference_record = dict(
            id=case["id"], label=case["label"], settings=case, geometry=asdict(nozzle),
            reservoir=reservoir, throat=throat, mdot=mdot,
            adapted_exit={key: value[-1] for key, value in adapted.items() if key != "side"},
            table_sha256=eos.table_sha256,
        )
        context = CaseContext(case["id"], "adapted", case, reference_record, eos, nozzle,
                              upstream, None, mdot, back_pressure, None)
        context._rho_vector = jax.jit(jax.vmap(eos._rho))
        model = context.model(50, cold_start=profile.get("cold_start"))
        scheme = Scheme(flux="rusanov", reconstruction="constant", **profile["scheme"])
        policy = ResidualPolicy(**profile["residual"])
        settings = SteadySettings(residual=policy, **profile["steady"])
        initial = np.asarray(model.initial_state())
        result.update(
            candidate_table=case["table"], reference=reference_record, reference_fields=adapted,
            scheme=asdict(scheme), steady_settings=asdict(settings),
            initial_state=initial, initial_admissible=bool(model.cell_admissible(initial)),
        )
        print(f"Candidate table prepared; initial admissible={result['initial_admissible']}; "
              f"adapted pb={back_pressure/1e6:.9g} MPa; mdot={mdot:.9g} kg/s", flush=True)
        save(destination, result)
        solution = budgeted_solve(model, scheme, settings)
        evaluation = evaluate_solution(context, solution, policy)
        state = np.asarray(solution.state)
        result.update(
            success=solution.success, verified=bool(solution.success and evaluation["audit"]["passed"]),
            reason=solution.reason, state=state, history=solution.history, metrics=solution.metrics,
            elapsed_s=solution.elapsed_s, **evaluation,
            state_pressure_max_Pa=float(np.max(state[:, 0])),
            state_pressure_min_Pa=float(np.min(state[:, 0])),
            accepted_steps=sum(row.get("phase") == "accepted" for row in solution.history),
            rejected_steps=sum(row.get("phase") == "rejected" for row in solution.history),
        )
        print(f"Candidate outcome success={result['success']} verified={result['verified']} "
              f"reason={result['reason']} accepted={result['accepted_steps']} "
              f"pmax={result['state_pressure_max_Pa']/1e6:.9g} MPa", flush=True)
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
        raise
    finally:
        result["source_after"] = source_manifest()
        result["src_unchanged"] = source_before == result["source_after"]
        result["active_settings_unchanged"] = profile_bytes_before == (STUDY / "settings.json").read_bytes()
        result["active_reference_unchanged"] = reference_bytes_before == active_reference.read_bytes()
        result["active_profile_sha256"] = identity(profile)
        result["total_elapsed_s"] = perf_counter() - start
        result["verified"] = bool(result["verified"] and result["src_unchanged"])
        save(destination, result)


if __name__ == "__main__":
    main()
