"""Compare the k_inverse / k_free / smooth runs of run.py, per case and grid.

Run from the repository root: poetry run python -B development/two-fluid/compare_runs.py
Reads outputs/<variant>_N<cells>/<case>.{json,npz}; missing runs are skipped.
Writes outputs/comparison/summary.csv, differences.csv and figures.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIGURES = OUT / "comparison"
FIGURES.mkdir(parents=True, exist_ok=True)
VARIANTS = ("k_inverse", "k_inverse_smooth", "k_free", "k_free_smooth")
CASES = ("subsonic", "shocked", "adapted")
GRIDS = (100, 200, 400, 800)
FIELDS = ("p", "alpha_g", "u_g", "u_l", "T_g", "T_l")

spec = importlib.util.spec_from_file_location("two_fluid_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

# 1. Load every available run; mass flow from the cell-centre areas of run.py's mesh
runs = {}
for variant in VARIANTS:
    for cells in GRIDS:
        area = np.asarray(run.mesh_for(cells).area_centers)
        for case in CASES:
            folder = OUT / f"{variant}_N{cells}"
            if not (folder / f"{case}.npz").exists():
                continue
            data = dict(np.load(folder / f"{case}.npz"))
            summary = json.loads((folder / f"{case}.json").read_text(encoding="utf-8"))["summary"]
            m_dot = area * (data["alpha_g"] * data["rho_g"] * data["u_g"] + (1 - data["alpha_g"]) * data["rho_l"] * data["u_l"])
            runs[variant, cells, case] = dict(data=data, summary=summary, m_dot=m_dot)

# 2. Run summary: convergence, cost and integral quantities
rows = []
for (variant, cells, case), r in runs.items():
    s = r["summary"]
    rows.append(dict(case=case, cells=cells, variant=variant, success=s["success"], reason=s["reason"],
                     accepted=s["accepted_steps"], rejected=s["rejected_steps"], elapsed_s=round(s["elapsed_s"], 1),
                     regime_passed=s.get("regime_passed"), max_mach=s.get("max_mach"), m_dot=float(np.mean(r["m_dot"])),
                     m_dot_spread=float(r["m_dot"].max() / r["m_dot"].min() - 1)))
summary = pd.DataFrame(rows).sort_values(["case", "cells", "variant"])
summary.to_csv(FIGURES / "summary.csv", index=False)
print(summary.to_string(index=False))

# 3. Differences against k_inverse on the same grid: mean and max |diff| relative to the field range
rows = []
for (variant, cells, case), r in runs.items():
    if variant == "k_inverse" or ("k_inverse", cells, case) not in runs:
        continue
    ref = runs["k_inverse", cells, case]
    row = dict(case=case, cells=cells, variant=variant,
               m_dot_pct=100 * (np.mean(r["m_dot"]) / np.mean(ref["m_dot"]) - 1))
    for field in FIELDS:
        span = np.ptp(ref["data"][field])
        diff = np.abs(r["data"][field] - ref["data"][field]) / span
        row[f"{field}_mean"], row[f"{field}_max"] = diff.mean(), diff.max()
    rows.append(row)
differences = pd.DataFrame(rows).sort_values(["case", "variant", "cells"])
differences.to_csv(FIGURES / "differences.csv", index=False)
with pd.option_context("display.float_format", "{:.2e}".format, "display.width", 250):
    print("\nDifference vs k_inverse on the same grid (|diff| / field range):")
    print(differences[["case", "variant", "cells", "m_dot_pct"] + [f"{f}_mean" for f in FIELDS] + ["p_max", "alpha_g_max"]].to_string(index=False))

# 4. Profiles on the finest available grid of each case, all variants
labels = dict(p="p [bar]", alpha_g=r"$\alpha_g$ [-]", u_g=r"$u_g$ [m/s]", T_g=r"$T_g$ [K]")
styles = dict(k_inverse=("k", "-"), k_inverse_smooth=("C1", "--"), k_free=("C0", "-"), k_free_smooth=("C3", ":"))
for case in CASES:
    available = [c for c in GRIDS if any((v, c, case) in runs for v in VARIANTS)]
    if not available:
        continue
    cells = max(available)
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for axis, field in zip(axes.flat, labels):
        for variant in VARIANTS:
            if (variant, cells, case) in runs:
                d = runs[variant, cells, case]["data"]
                scale = 1e-5 if field == "p" else 1.0
                color, line = styles[variant]
                axis.plot(d["x"], scale * d[field], color=color, linestyle=line, label=variant)
        axis.set_ylabel(labels[field])
        axis.grid(True, color="0.88", linewidth=0.5)
    for axis in axes[1]:
        axis.set_xlabel("x [m]")
    axes[0, 0].legend()
    figure.suptitle(f"{case}, N = {cells}")
    figure.tight_layout()
    figure.savefig(FIGURES / f"profiles_{case}_N{cells}.png", dpi=200)
    plt.close(figure)

# 5. Mean pressure / void-fraction difference vs grid (should shrink if all variants converge to one solution)
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, field in zip(axes, ("p", "alpha_g")):
    for (case, variant), group in differences.groupby(["case", "variant"]):
        marker = dict(subsonic="o", shocked="s", adapted="^")[case]
        color = styles[variant][0]
        axis.loglog(group["cells"], group[f"{field}_mean"], marker=marker, color=color, label=f"{variant}, {case}")
    axis.set_xlabel("cells")
    axis.set_ylabel(f"mean |{field} - {field}_k_inverse| / range")
    axis.grid(True, which="both", color="0.88", linewidth=0.5)
axes[1].legend(fontsize=7)
figure.tight_layout()
figure.savefig(FIGURES / "difference_vs_grid.png", dpi=200)
plt.close(figure)
print(f"\nSaved to {FIGURES}")
