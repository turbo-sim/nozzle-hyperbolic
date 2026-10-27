"""Unsteady EulerNozzle baseline: the classical Sod problem, initially at rest.

Run: poetry run python development_new/euler_nozzle/verify_euler_nozzle_sod.py
Add --no-show to save the results without opening a figure window.

This uses the existing EulerNozzle HLL kernel in a small, explicit finite-volume
driver. One global physical timestep is clipped BEFORE the update to hit every
output time exactly. The existing application driver's local pseudo-time
stepping is unsuitable for this transient check. No production solver is edited.
"""

# %% Settings: one method, one mesh, three times from the same transient run.
from pathlib import Path

N_CELLS = 400  # Physical cells; two additional ghost cells impose open ends.
LENGTH = 1.0  # m
DIAPHRAGM = 0.5  # m, aligned with a face for this mesh.
LEFT = dict(rho=1.0, u=0.0, p=1.0e5)  # kg/m3, m/s, Pa
RIGHT = dict(rho=0.125, u=0.0, p=1.0e4)
CFL = 0.45
# Dimensionless time tau = t*a_left/L. All waves remain inside the domain.
OUTPUT_TAU = (0.10, 0.20, 0.30)
N_EXACT = 4001
MAX_NORMALIZED_L1 = 0.04
MAX_BUDGET_ERROR = 1.0e-10
OUTPUT_DIR = Path(__file__).resolve().parent / "results_euler_nozzle_sod"

# %% Imports and precision.
import argparse
import json
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.models.euler_nozzle.flux_jax import _jax_F_hll


def main(show=True):
    # 1. Fluid and initial conditions. Take gamma from the backend, so the
    # exact reference uses precisely the same perfect gas as the numerical flux.
    started = perf_counter()
    fluid = jxp.FluidPerfectGas(name="Air")
    gamma = float(fluid.constants.gamma)
    a_left = np.sqrt(gamma * LEFT["p"] / LEFT["rho"])
    dx = LENGTH / N_CELLS
    x = (np.arange(N_CELLS) + 0.5) * dx
    if not np.isclose(DIAPHRAGM / dx, round(DIAPHRAGM / dx)):
        raise ValueError("Place the diaphragm on a cell face to retain this baseline.")
    if not (0.0 < DIAPHRAGM < LENGTH) or sorted(OUTPUT_TAU) != list(OUTPUT_TAU) or min(OUTPUT_TAU) <= 0:
        raise ValueError("Use an interior diaphragm and increasing, positive output times.")
    rho = np.where(x < DIAPHRAGM, LEFT["rho"], RIGHT["rho"])
    velocity = np.where(x < DIAPHRAGM, LEFT["u"], RIGHT["u"])
    pressure = np.where(x < DIAPHRAGM, LEFT["p"], RIGHT["p"])
    initial = np.stack([rho, rho * velocity,
                        pressure / (gamma - 1.0) + 0.5 * rho * velocity**2])
    # Conservative state Q = [rho, rho*u, rho*E], shape (3, cells + 2 ghosts).
    q = jnp.asarray(np.pad(initial, ((0, 0), (1, 1)), mode="edge"))
    initial_totals = dx * initial.sum(axis=1)
    boundary_integral = np.zeros(3)
    budget_scales = LENGTH * np.array([LEFT["rho"], LEFT["rho"] * a_left,
                                       LEFT["p"] / (gamma - 1.0)])
    output_times = np.asarray(OUTPUT_TAU) * LENGTH / a_left
    colors = plt.colormaps["plasma"](np.linspace(0.2, 0.8, len(output_times)))
    x_exact = np.linspace(0.0, LENGTH, N_EXACT)
    exact_args = (DIAPHRAGM, LEFT["rho"], LEFT["u"], LEFT["p"],
                  RIGHT["rho"], RIGHT["u"], RIGHT["p"], gamma)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. First-order HLL + forward Euler. This intentionally small driver uses
    # the model's JAX flux, constant reconstruction, and transmissive ghosts.
    @jax.jit
    def step(q_old, remaining_time):
        q_bc = q_old.at[:, 0].set(q_old[:, 1])
        q_bc = q_bc.at[:, -1].set(q_bc[:, -2])
        rho_c = q_bc[0]
        u_c = q_bc[1] / rho_c
        p_c = (gamma - 1.0) * (q_bc[2] - 0.5 * rho_c * u_c**2)
        state = fluid.get_state(jxp.DmassP_INPUTS, rho_c, p_c)
        a_c = state["a"]
        # The SAME timestep advances every cell. Apply the output-time cap here,
        # before using dt in the conservative update, not just in bookkeeping.
        dt = jnp.minimum(CFL * dx / jnp.max(jnp.abs(u_c) + a_c), remaining_time)
        areas = jnp.ones(q_bc.shape[1])  # Constant unit area: no geometric source.
        flux = _jax_F_hll(q_bc[:, :-1], q_bc[:, 1:], p_c[:-1], a_c[:-1],
                         p_c[1:], a_c[1:], areas[:-1], areas[1:])
        q_new = q_bc.at[:, 1:-1].add(-dt / dx * (flux[:, 1:] - flux[:, :-1]))
        # Integrated boundary influx for independent mass/momentum/energy budgets.
        influx = dt * (flux[:, 0] - flux[:, -1])
        return q_new, dt, influx

    # 3. Advance ONCE through all requested times; never restart from exact data.
    time = 0.0
    steps = 0
    summaries, time_handles = [], []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for index, (target, color) in enumerate(zip(output_times, colors)):
        while time < target:
            q, dt, influx = step(q, jnp.float64(target - time))
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0:
                raise RuntimeError("Invalid physical timestep")
            boundary_integral += np.asarray(influx)
            time = min(float(target), time + dt)
            steps += 1

        # 4. Sample the exact Riemann solution and quantify errors and budgets.
        # This existing exact solver assumes the Sod left-fan/right-shock pattern.
        q_num = np.asarray(q[:, 1:-1])
        rho_num = q_num[0]
        u_num = q_num[1] / rho_num
        p_num = (gamma - 1.0) * (q_num[2] - 0.5 * rho_num * u_num**2)
        numerical = dict(rho=rho_num, u=u_num, p=p_num)
        if not np.isfinite(q_num).all() or rho_num.min() <= 0 or p_num.min() <= 0:
            raise RuntimeError("Non-finite or non-positive numerical state")
        exact = dict(zip(("rho", "u", "p"), sod_exact(x_exact, time, *exact_args)))
        reference = dict(zip(("rho", "u", "p"), sod_exact(x, time, *exact_args)))
        for key in ("rho", "u", "p"):
            # Exact boundary states must still be undisturbed: no reflected waves.
            np.testing.assert_allclose(exact[key][[0, -1]], [LEFT[key], RIGHT[key]],
                                       rtol=1e-10, atol=1e-10)
        scales = dict(rho=LEFT["rho"], u=a_left, p=LEFT["p"])
        errors = {key: float(np.mean(np.abs(numerical[key] - reference[key])) / scales[key])
                  for key in numerical}
        budget = (dx * q_num.sum(axis=1) - initial_totals - boundary_integral) / budget_scales
        passed = bool(max(errors.values()) < MAX_NORMALIZED_L1
                      and np.max(np.abs(budget)) < MAX_BUDGET_ERROR)
        summary = dict(
            physical_time_s=time, tau=time * a_left / LENGTH, steps=steps,
            normalized_L1=errors,
            normalized_conservation_error=dict(zip(("mass", "momentum", "energy"), budget.tolist())),
            passed=passed,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        np.savetxt(OUTPUT_DIR / f"time_{index + 1}.csv", np.column_stack(
            [x] + [values for key in numerical for values in (numerical[key], reference[key])]),
            delimiter=",", header="x_m,rho_num,rho_exact,u_num_m_s,u_exact_m_s,p_num_Pa,p_exact_Pa",
            comments="")

        # 5. One Plasma color per physical time; the exact and numerical curves
        # share the color and differ by line/marker style.
        for ax, key, scale, ylabel in zip(axes, numerical, (1., 1., 1.0e3),
                                         (r"$\rho$ [kg/m$^3$]", "u [m/s]", "p [kPa]")):
            ax.plot(x_exact, exact[key] / scale, color=color, lw=1.5)
            ax.plot(x, numerical[key] / scale, "o", color=color, mfc="none", ms=3.2,
                    markevery=max(1, N_CELLS // 55))
            ax.set(xlabel="x [m]", ylabel=ylabel)
            ax.grid(alpha=0.2)
        time_handles.append(Line2D([], [], color=color, label=f"t = {time * 1e3:.3f} ms"))

    axes[0].legend(handles=time_handles, title="Physical time", fontsize=9)
    axes[1].legend(handles=[Line2D([], [], color="0.25", label="Exact"),
                           Line2D([], [], color="0.25", marker="o", mfc="none",
                                  linestyle="none", label="FVM")], fontsize=9)
    fig.suptitle(f"EulerNozzle Sod baseline: HLL / constant reconstruction / forward Euler, {N_CELLS} cells")
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "comparison.pdf")
    report = dict(
        method="HLL / first-order finite volumes / forward Euler",
        fluid="jaxprop.FluidPerfectGas(Air)", gamma=gamma,
        physical_cells=N_CELLS, CFL=CFL, length_m=LENGTH, diaphragm_m=DIAPHRAGM,
        left=LEFT, right=RIGHT, output_tau=OUTPUT_TAU,
        thresholds=dict(normalized_L1=MAX_NORMALIZED_L1, conservation=MAX_BUDGET_ERROR),
        snapshots=summaries, elapsed_s=perf_counter() - started,
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved unsteady comparison to {OUTPUT_DIR}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if not all(snapshot["passed"] for snapshot in summaries):
        raise SystemExit("Sod baseline failed; inspect summary.json for errors and budgets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening a window.")
    main(show=not parser.parse_args().no_show)
