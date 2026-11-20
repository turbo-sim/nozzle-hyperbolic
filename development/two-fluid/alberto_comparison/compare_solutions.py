"""Compare main and alberto_branch steady solutions, baseline and with run-time inputs.

Run: python -B compare_solutions.py [old_suffix]   (reads outputs/new_{base,varied}.npz and
outputs/old_{base,varied}<old_suffix>.npz, e.g. old_suffix _20000 for the longer alberto_branch runs)
Interior cells sit at the same positions in both layouts. Differences are max |diff| / field range.
The response (varied - baseline) of each code is compared too: it isolates the effect of the run-time
back pressure and correction fields from the known baseline differences between the two codes.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "outputs"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
NAMES = ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")

# 1. Interior primitives of the four runs
runs = {}
for code in ("new", "old"):
    for case in ("base", "varied"):
        d = np.load(OUT / f"{code}_{case}{SUFFIX if code == 'old' else ''}.npz")
        runs[code, case] = dict(x=np.asarray(d["x"])[1:-1], U=np.asarray(d["U"])[1:-1])
        extra = f"steps {int(d['steps'])}, SSR {float(d['ssr']):.1e}, reason {d['reason']}" if code == "old" else \
            f"success {bool(d['success'])}, reason {d['reason']}"
        print(f"{code:3s} {case:6s}: {extra}")
x = runs["new", "base"]["x"]
assert np.allclose(x, runs["old", "base"]["x"])

# 2. Same case, main vs alberto_branch
span = np.ptp(runs["old", "base"]["U"], axis=0)
for case in ("base", "varied"):
    error = np.max(np.abs(runs["new", case]["U"] - runs["old", case]["U"]), axis=0) / span
    print(f"main vs alberto_branch, {case:6s}: " + "  ".join(f"{n}={e:.1e}" for n, e in zip(NAMES, error)))

# 3. Response to the run-time inputs, and its agreement between the codes
response = {code: runs[code, "varied"]["U"] - runs[code, "base"]["U"] for code in ("new", "old")}
size = np.max(np.abs(response["old"]), axis=0)
print("response size (alberto_branch) / range: " + "  ".join(f"{n}={s / r:.1e}" for n, s, r in zip(NAMES, size, span)))
mismatch = np.max(np.abs(response["new"] - response["old"]), axis=0) / size
print("response main vs alberto_branch / response size: " + "  ".join(f"{n}={e:.1e}" for n, e in zip(NAMES, mismatch)))
print(f"outlet pressure (last cell) [bar]: main base {runs['new', 'base']['U'][-1, 0] / 1e5:.4f} varied "
      f"{runs['new', 'varied']['U'][-1, 0] / 1e5:.4f} | alberto_branch base {runs['old', 'base']['U'][-1, 0] / 1e5:.4f} "
      f"varied {runs['old', 'varied']['U'][-1, 0] / 1e5:.4f}")

# 4. Figure: profiles and responses
figure, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for axis, (j, name) in zip(axes[0], ((0, "p"), (3, "alpha_g"), (1, "u_g"))):
    for (code, case), style in zip(runs, ("-", "-", "--", "--")):
        axis.plot(x, runs[code, case]["U"][:, j], style, label=f"{'main' if code == 'new' else 'alberto_branch'} {case}")
    axis.set_ylabel(name)
for axis, (j, name) in zip(axes[1], ((0, "p"), (3, "alpha_g"), (1, "u_g"))):
    axis.plot(x, response["new"][:, j], "-", label="main")
    axis.plot(x, response["old"][:, j], "--", label="alberto_branch")
    axis.set_ylabel(f"response {name}")
    axis.set_xlabel("x [m]")
axes[0, 0].legend(fontsize=7)
axes[1, 0].legend(fontsize=7)
figure.tight_layout()
figure.savefig(OUT / f"solutions_main_vs_alberto{SUFFIX}.png", dpi=150)
print(f"Saved {OUT / f'solutions_main_vs_alberto{SUFFIX}.png'}")
