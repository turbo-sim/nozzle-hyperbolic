"""Independent saved-state residual, geometry, analytic and conservation audit."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import json
from copy import copy
import jax.numpy as jnp
from hashlib import sha256
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from .run import STUDY, hashes, save
from .problem import geometry, model
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, sampler, ResidualPolicy


def main():
    d = json.loads((STUDY / "outputs/results.json").read_text())
    c = d["config"]
    geom = geometry(c)
    records = []
    native = {}
    for name, v in d["steady"].items():
        for sol in v["representative"]:
            n = sol["cells"]
            if n != 200 and name != "implicit":
                continue
            if n not in native:
                native[n] = model(c, geom, n)
            m = native[n]
            op = Operator(m, Scheme(**v["scheme"]))
            audit = sampler(op, ResidualPolicy())(
                np.array(sol["state"]), np.array(sol["history"][0]["equation_norm"])
            )
            passed = bool(
                audit["all_equations_passed"]
                and audit["boundary_passed"]
                and audit["global_passed"]
                and audit["admissible"]
                and m.solution_audit(jnp.asarray(sol["state"]))
            )
            gamma = float(m.fluid.constants.gamma)
            R = float(m.fluid.constants.R)
            cp = gamma * R / (gamma - 1)
            z = np.array(sol["state"])
            zin = np.array(m.inlet_data["state"])
            Tin = zin[3] / cp
            ell = zin[2]
            h0 = zin[3] + 0.5 * (zin[1] ** 2 + (ell / m.native_mesh.r_face[0]) ** 2)
            rhoin = zin[0] / (R * Tin)
            mdot = rhoin * zin[1] * m.native_mesh.A_face[0]
            exact = {k: [] for k in ("p", "M", "alpha_deg")}
            for r, A in zip(m.native_mesh.r_center, m.native_mesh.A_center):
                vt = ell / r
                energy = h0 - 0.5 * vt**2
                usonic = np.sqrt(2 * (gamma - 1) * energy / (gamma + 1))
                umax = np.sqrt(2 * energy)

                def mass(u):
                    T = (energy - 0.5 * u * u) / cp
                    return rhoin * (T / Tin) ** (1 / (gamma - 1)) * u * A - mdot

                u = brentq(mass, usonic, umax * (1 - 1e-12), xtol=1e-10)
                T = (energy - 0.5 * u * u) / cp
                exact["p"].append(zin[0] * (T / Tin) ** (gamma / (gamma - 1)))
                exact["M"].append(np.hypot(u, vt) / np.sqrt(gamma * R * T))
                exact["alpha_deg"].append(np.rad2deg(np.arctan2(vt, u)))
            errors = {
                k: float(np.sqrt(np.mean((np.array(sol["fields"][k]) - exact[k]) ** 2)))
                for k in exact
            }
            records.append(
                dict(
                    method=name,
                    cells=n,
                    passed=passed,
                    recomputed_metrics=audit,
                    analytic=exact,
                    analytic_RMS_errors=errors,
                    x=m.native_mesh.x_center,
                )
            )
            print(
                name, n, "residual audit", passed, "analytic errors", errors, flush=True
            )
    transient = []
    replays = []
    for name, v in d["transient"].items():
        total = c["forcing"]["cycles"] / c["forcing"]["frequency_Hz"]
        # Independently recover the final budget from retained accumulated flux
        # transfers and storage, rather than trusting the stored budget label.
        if 200 not in native:
            native[200] = model(c, geom, 200)
        op = Operator(native[200], Scheme(**v["scheme"]))
        seed_name = next(
            k
            for k in ("implicit", "rk3", "euler")
            if k in d["steady"] and d["steady"][k]["representative"][-1]["success"]
        )
        seed = np.array(d["steady"][seed_name]["representative"][-1]["state"])
        initial = np.array(op.storage(seed[1:-1])).sum(axis=0)
        final = np.array(op.storage(np.array(v["states"])[-1, 1:-1])).sum(axis=0)
        transfer = sum(
            (
                np.array(row["integrated_transfer"])
                for row in v["history"]
                if row["accepted"]
            ),
            start=np.zeros(4),
        )
        budget = (final - initial - transfer) / (
            np.sum(native[200].mesh.volumes) * np.array(native[200].conservative_scales)
        )
        okay = (
            all(s["success"] for s in v["samples"])
            and abs(v["times"][-1] - total) < 1e-14
            and np.max(np.abs(budget)) < 1e-8
        )
        transient.append(
            dict(
                method=name,
                passed=okay,
                recomputed_final_budget=budget,
                steps=v["samples"][-1]["accepted_steps"],
                rejections=v["samples"][-1]["rejected_steps"],
            )
        )
    # Replay retained accepted timesteps under the current checkout. This
    # also checks numerical compatibility when another session changed files
    # after the benchmark process had already imported its solver modules.
    for name, v in d["transient"].items():
        physical = copy(native[200])
        physical.driven = True
        operator = Operator(physical, Scheme(**v["scheme"]))
        for index in (5, 17, 185):
            if index >= len(v["times"]):
                continue
            start, end = v["times"][index - 1 : index + 1]
            state = jnp.asarray(v["states"][index - 1])
            steps = [
                r
                for r in v["history"]
                if r["accepted"]
                and r["time"] > start + 1e-15
                and r["time"] <= end + 1e-15
            ]
            accepted = True
            for row in steps:
                state, info = operator.step(
                    state, jnp.float64(row["dt"]), jnp.float64(row["time"])
                )
                accepted = accepted and bool(info.converged and info.admissible)
            error = float(
                np.max(
                    np.abs(np.asarray(state) - np.asarray(v["states"][index]))
                    / physical.variable_scales
                )
            )
            passed = accepted and bool(steps) and error < 1e-9
            replays.append(
                dict(
                    method=name,
                    output_index=index,
                    steps=len(steps),
                    maximum_normalized_state_difference=error,
                    passed=passed,
                )
            )
            print("HLL replay", name, index, passed, error, flush=True)
    g = json.loads((STUDY / "outputs/geometry.json").read_text())
    tiperr = float(np.max(np.abs(np.array(g["tip_r"]) - c["geometry"]["tip_radius_m"])))
    source_ok = (
        hashes()
        == d["source_before"]
        == json.loads((STUDY / "outputs/source_after.json").read_text())
    )
    all_ok = (
        all(r["passed"] for r in records + transient)
        and all(r["passed"] for r in replays)
        and tiperr < 1e-9
        and d.get("reference", {}).get("success", False)
    )
    result = dict(
        results_sha256=sha256(
            (STUDY / "outputs/results.json").read_bytes()
        ).hexdigest(),
        passed=all_ok,
        source_unchanged=source_ok,
        current_hll_compatibility_passed=all(r["passed"] for r in replays),
        timestep_replays=replays,
        audited_source_manifest=hashes(),
        changed_source_paths=[
            path
            for path in d["source_before"]
            if d["source_before"].get(path)
            != json.loads((STUDY / "outputs/source_after.json").read_text()).get(path)
        ],
        tip_radius_error_m=tiperr,
        steady=records,
        transient=transient,
    )
    save(STUDY / "outputs/audit.json", result)
    # Export standard scientific figures independently of browser availability.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(g["tip_x"], g["tip_r"], color=plt.cm.plasma(0.8), label="Tip")
    axes[0].plot(g["hub_x"], g["hub_r"], color=plt.cm.plasma(0.2), label="Hub")
    axes[0].plot(g["x"], g["r"], "--", color=plt.cm.plasma(0.5), label="Midline")
    axes[0].set(
        xlabel="Axial coordinate [m]",
        ylabel="Radius [m]",
        title="Annular transition duct",
    )
    axes[0].legend()
    for i, (name, v) in enumerate(d["transient"].items()):
        axes[1].plot(
            np.array(v["times"]) * 1000,
            np.array(v["fields"]["p"])[:, -1] / 1e5,
            color=plt.cm.plasma(0.2 + 0.3 * i),
            label=name,
        )
    axes[1].set(
        xlabel="Time [ms]", ylabel="Outlet-cell pressure [bar]", title="Forced response"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(STUDY / "outputs/overview.png", dpi=180)
    fig.savefig(STUDY / "outputs/overview.pdf")
    print("AUDIT", all_ok, flush=True)
    from .report import build

    build()
    assert all_ok, "Audit failures retained in outputs/audit.json"


if __name__ == "__main__":
    main()
