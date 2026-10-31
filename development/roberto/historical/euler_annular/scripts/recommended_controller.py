"""Apply the measured CFL recommendation using shared study orchestration.

This is a steady pseudo-time controller, not a physical-time integration policy.
It preserves the native step's strict inner convergence and admissibility tests.
"""

from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.models.euler_annular.fvm import AnnularSolution
from .steady_march import COLUMNS, make_marcher


SETTINGS = dict(cfl=0.2, growth=1.2, shrink=0.5, cfl_max=1e6,
                max_retries=12, newton_iterations=8,
                inner_acceptance="strict", growth_policy="fixed")


def json_safe_attempt(row):
    """Keep rejected Newton infinity sentinels explicit in standard JSON.

    An exhausted line search can return an infinite update diagnostic. It is
    neither an accepted update nor a computed flow-field infinity. Preserve its
    classification beside a null value instead of emitting invalid JSON tokens.
    """
    for key, value in tuple(row.items()):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            if row["accepted"]:
                raise ValueError(f"Accepted attempt has nonfinite {key}")
            row[key] = None
            row[key+"_nonfinite"] = ("nan" if np.isnan(value) else
                                      "positive_infinity" if value > 0 else "negative_infinity")
    return row


def solve_recommended(solver, initial, *, max_steps=16000, tolerance=1e-8, cfl=0.2):
    """Cold solve with all chronological attempts and four balance histories."""
    if solver.options.newton_iterations != SETTINGS["newton_iterations"]:
        raise ValueError("The recommended strict controller requires a cap of eight")
    if not bool(solver.is_admissible(initial)):
        raise ValueError("Invalid initial state for the recommended controller")
    march = make_marcher(solver, max_steps, tolerance=tolerance,
                         max_retries=SETTINGS["max_retries"], cfl_max=SETTINGS["cfl_max"],
                         inner_acceptance="strict", record_balance=True)
    output = jax.device_get(march(jnp.asarray(initial), jnp.asarray(cfl),
        jnp.asarray(SETTINGS["growth"]), jnp.asarray(SETTINGS["shrink"]), jnp.asarray(False)))
    state, final_cfl, accepted, retry, attempts, pseudo_time, _, raw = output
    # Fresh public audit independently of the compiled loop's stopping flag.
    metrics = solver._audit(jnp.asarray(state))
    defect = max(metrics[key] for key in ("spatial_rms", "boundary_max", "conservation_max"))
    success = bool(metrics["admissible"] and np.isfinite(defect)
                   and defect < tolerance and solver._regimes_match(metrics))
    history = []
    integers = {"step", "retry", "inner_iterations", "inner_status", "rejected_newton_trials"}
    for values in raw[:int(attempts)]:
        row = dict(zip(COLUMNS, map(float, values[:len(COLUMNS)])))
        for key in integers:
            row[key] = int(row[key])
        row["accepted"] = bool(row["accepted"])
        row["admissible"] = True if row["accepted"] else None
        row["residual_only_accept"] = bool(row["residual_only_accept"])
        if not np.isfinite(row["shock_proxy_m"]):
            row["shock_proxy_m"] = None  # No detected compressive front, not missing solver data.
        row["integrated_balance_scaled"] = values[len(COLUMNS):].tolist()
        row["next_cfl_factor"] = SETTINGS["growth"] if row["accepted"] else SETTINGS["shrink"]
        if row["accepted"] and not np.isclose(max(abs(values[len(COLUMNS):])),
                row["conservation_max"], rtol=1e-9, atol=1e-14):
            raise ValueError("Per-equation balances do not reproduce the scalar audit")
        history.append(json_safe_attempt(row))
    if success:
        message = "Original spatial, boundary and conservation targets and selected regimes met"
    elif retry > SETTINGS["max_retries"]:
        message = "Implicit timestep retries exhausted"
    elif int(accepted) >= max_steps:
        message = "Steady update limit reached"
    elif defect < tolerance:
        message = "Residual target met but selected boundary regimes do not match"
    else:
        message = "Attempt budget reached"
    solution = AnnularSolution(np.asarray(state), success, message, history, metrics)
    return solution, dict(initial_dt=float(solver.compute_dt(initial, cfl)),
                         final_cfl=float(final_cfl), pseudo_time=float(pseudo_time),
                         accepted=int(accepted), attempts=int(attempts), rejected=int(attempts-accepted))


def driver_hashes():
    folder = Path(__file__).parent
    return {name: sha256((folder/name).read_bytes()).hexdigest()
            for name in ("recommended_controller.py", "steady_march.py", "verify_annular.py")}
