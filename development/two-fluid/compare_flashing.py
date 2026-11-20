"""Grid study of the K-free flashing solutions and the K-inverse restart from them.

Run from the repository root: poetry run python -B development/two-fluid/compare_flashing.py
Reads outputs/flashing/<run>/<regime>.{json,npz}; writes outputs/flashing/comparison/.
Fields include both endpoint traces; comparisons use interior points (traces excluded).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "flashing"
FIG = OUT / "comparison"
FIG.mkdir(exist_ok=True)
REGIMES = ("subsonic", "supersonic_shock", "supersonic_adapted")
GRIDS = (100, 200, 400, 800)
FIELDS = ("p", "alpha_g", "u_g", "u_l", "T_l")
REFERENCE = HERE.parents[1] / "cases" / "NICFD2026" / "single_component"

# 1. K-free on each grid: convergence, mass flow and profile difference against the published solution
rows = []
data = {}
for regime in REGIMES:
    ref = pd.read_csv(REFERENCE / regime / "reference.csv")
    for cells in GRIDS:
        folder = OUT / f"k_free_N{cells}"
        if not (folder / f"{regime}.npz").exists():
            continue
        record = json.loads((folder / f"{regime}.json").read_text(encoding="utf-8"))
        d = dict(np.load(folder / f"{regime}.npz"))
        data[regime, cells] = d
        x = d["x"][1:-1]
        row = dict(regime=regime, cells=cells, reason=record["summary"]["reason"],
                   accepted=record["summary"]["accepted_steps"], rejected=record["summary"]["rejected_steps"],
                   audit=record["audit"]["passed"], regime_passed=record["regime_audit"]["passed"],
                   m_dot=float(np.mean(d["mdot"][1:-1])), m_dot_published=float(np.mean(ref["mdot"])))
        row["m_dot_vs_published_pct"] = 100 * (row["m_dot"] / row["m_dot_published"] - 1)
        for field in FIELDS:
            on_ref = np.interp(ref["x"], x, d[field][1:-1])
            row[f"{field}_vs_pub"] = float(np.mean(np.abs(on_ref - ref[field])) / np.ptp(ref[field]))
        rows.append(row)
grid = pd.DataFrame(rows)

# 2. Change between consecutive grids (on the finer interior points) and observed order of the mass flow
rows = []
for regime in REGIMES:
    available = [c for c in GRIDS if (regime, c) in data]
    for coarse, fine in zip(available[:-1], available[1:]):
        a, b = data[regime, coarse], data[regime, fine]
        x = b["x"][1:-1]
        row = dict(regime=regime, step=f"{coarse}->{fine}")
        for field in FIELDS:
            row[f"{field}_change"] = float(np.mean(np.abs(np.interp(x, a["x"][1:-1], a[field][1:-1]) - b[field][1:-1]))
                                           / np.ptp(b[field][1:-1]))
        rows.append(row)
    m = [float(np.mean(data[regime, c]["mdot"][1:-1])) for c in available]
    if len(m) >= 3:
        orders = [np.log(abs(m[i] - m[i + 1]) / abs(m[i + 1] - m[i + 2])) / np.log(2) for i in range(len(m) - 2)]
        print(f"{regime}: m_dot {['%.4f' % v for v in m]} kg/s on {available}; observed order {['%.2f' % o for o in orders]}")
changes = pd.DataFrame(rows)

# 3. K-inverse restarted from the converged K-free 400-cell solution
rows = []
for regime in REGIMES:
    folder = OUT / "k_inverse_N400_from_k_free_N400"
    if not (folder / f"{regime}.npz").exists():
        continue
    record = json.loads((folder / f"{regime}.json").read_text(encoding="utf-8"))
    start, end = data[regime, 400], dict(np.load(folder / f"{regime}.npz"))
    history = record["history"]
    row = dict(regime=regime, reason=record["summary"]["reason"], accepted=record["summary"]["accepted_steps"],
               rejected=record["summary"]["rejected_steps"], audit=record["audit"]["passed"],
               initial_max_norm=max(history[0]["equation_norm"]), final_max_norm=max(record["summary"]["equation_norm"]),
               m_dot_change_pct=100 * (np.mean(end["mdot"][1:-1]) / np.mean(start["mdot"][1:-1]) - 1))
    for field in FIELDS:
        row[f"{field}_drift"] = float(np.mean(np.abs(end[field][1:-1] - start[field][1:-1])) / np.ptp(start[field][1:-1]))
    rows.append(row)
restart = pd.DataFrame(rows)

with pd.option_context("display.width", 250, "display.float_format", "{:.3e}".format):
    print("\nK-free per grid (|field - published| mean / published range):")
    print(grid.to_string(index=False))
    print("\nChange between consecutive grids (mean |diff| / range):")
    print(changes.to_string(index=False))
    print("\nK-inverse restarted from K-free 400 (drift = mean |end - start| / range):")
    print(restart.to_string(index=False))
grid.to_csv(FIG / "kfree_grid.csv", index=False)
changes.to_csv(FIG / "kfree_grid_changes.csv", index=False)
restart.to_csv(FIG / "kinverse_restart.csv", index=False)

# 4. Profiles: K-free on every grid and the published solution
for regime in REGIMES:
    ref = pd.read_csv(REFERENCE / regime / "reference.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for axis, field in zip(axes.flat, ("p", "alpha_g", "u_g", "mdot")):
        axis.plot(ref["x"], ref[field], color="0.6", lw=3, label="published")
        for cells, color in zip(GRIDS, ("C3", "C1", "C0", "k")):
            if (regime, cells) in data:
                d = data[regime, cells]
                axis.plot(d["x"][1:-1], d[field][1:-1], color=color, lw=1.2, label=f"k_free N{cells}")
        axis.set_ylabel(field)
        axis.grid(True, color="0.88", linewidth=0.5)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("x [m]")
    figure.suptitle(f"Super Moby Dick, {regime}")
    figure.tight_layout()
    figure.savefig(FIG / f"kfree_grids_{regime}.png", dpi=170)
    plt.close(figure)
print(f"\nSaved to {FIG}")
