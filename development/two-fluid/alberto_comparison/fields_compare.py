"""Compare R_old.npz and R_new.npz: max difference per equation, relative to the conservative term.

Run: python -B fields_compare.py <out_dir>
Also prints how much the fields change each residual, so agreement is not trivial.
"""

import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
old, new = np.load(OUT / "R_old.npz"), np.load(OUT / "R_new.npz")
rows = ("mass_g", "mass_l", "mom_g", "mom_l", "E_g", "E_l")
for name in [k for k in old.files if not k.endswith("__fields")]:
    scale = np.max(np.abs(new[name + "__conservative"]), axis=0)
    for label in ("", "__fields"):
        if name + label not in old.files:
            continue
        error = np.max(np.abs(old[name + label] - new[name + label]), axis=0) / scale
        print(f"{name + label:30s} old vs new: " + "  ".join(f"{r}={e:.1e}" for r, e in zip(rows, error)))
    effect = np.max(np.abs(old[name + "__fields"] - old[name]), axis=0) / scale
    print(f"{'  effect of the fields (old)':30s}             " + "  ".join(f"{r}={e:.1e}" for r, e in zip(rows, effect)))
