"""
probe_oq10_limiter_interior.py

PROBE for OPEN_QUESTIONS.md section 10 -- NOT a paper artifact.

Question: two_component converges at the EXIT node on every triplet but fails on
every INTERIOR station and integral.  Five mechanisms were tested and ruled out
(shock jitter, rigid translation, checkerboard phase, machine/iterative noise,
interpolation).  The remaining cheap probe: is it the LIMITER?

Under va_eps_mode='magnitude' the van Albada eps floor is normalised by field
MAGNITUDE, so alpha -- a large offset with a small variation (0.978 +/- 0.019) --
keeps only 0.2-7% of its true slope at these stations, i.e. it is reconstructed
essentially piecewise CONSTANT.  Under va_eps_mode='smoothness' the gate keys on
stencil shape instead and keeps ~100% of the slope in smooth cells.

If the interior non-monotonicity disappears under 'smoothness', the cause is
limiter-related and section 10 connects to the va_eps_mode work.  If it
persists, the cause is in the scheme or the physics and the limiter is exonerated.

CAVEAT on the comparison: the two ladders are not grid-identical.  The magnitude
ladder's N=358 never converged (SSR 1.1e-06, a limit cycle), which is why the
study uses N=351 instead; the smoothness ladder has N=358.  h(351) and h(358)
differ by 2%, so the refinement ratios differ slightly (1.426/1.373 vs
1.398/1.400).  That is immaterial for the question being asked, which is whether
the interior converges AT ALL, not the precise value of p_obs.

Run:
    poetry run python probe_oq10_limiter_interior.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nicfd_figures"))

from grid_convergence_gci import gci_3grid                      # noqa: E402
from gci_nicfd_table import (exit_qois, interior_qois,          # noqa: E402
                             load_run, STATIONS)

L = 0.31

LADDERS = {
    "magnitude  (study ladder)": {
        500: HERE / "results_grid_sesitivity/two_component_N500_muscl_van_albada_20260904_081640",
        351: HERE / "results_grid_sesitivity/two_component_N351_muscl_van_albada_20260910_124300",
        256: HERE / "results_grid_sesitivity/two_component_N256_muscl_van_albada_20260904_113252",
    },
    "smoothness (probe ladder)": {
        500: HERE / "results_grid_limiterfix/two_component_N500_muscl_van_albada_20260909_165535",
        358: HERE / "results_grid_limiterfix/two_component_N358_muscl_van_albada_20260909_165408",
        256: HERE / "results_grid_limiterfix/two_component_N256_muscl_van_albada_20260909_165353",
    },
}


def slope_kept(d, frac):
    """Fraction of alpha's true slope the solver's limiter actually retains at
    this station -- the quantity the probe is about."""
    x, U = d["x"], d["alpha_g"]
    dU = np.diff(U)
    a, b = dU[:-1], dU[1:]
    eps = 1e-3 * (np.abs(U[1:-1]) + 0.1 * np.mean(np.abs(U)))
    s = np.where(a * b > 0.0, a * b * (a + b) / (a * a + b * b + eps * eps), 0.0)
    i = int(np.argmin(np.abs(x - frac * L)))
    true = (U[i + 1] - U[i - 1]) / 2 if 0 < i < len(U) - 1 else np.nan
    return s[i - 1] / true if true not in (0, np.nan) else np.nan


def analyse(runs):
    loaded = {N: load_run(p, N) for N, p in runs.items()}
    for r in loaded.values():
        r["h"] = L / (r["Nx"] - 1)
        ex = exit_qois(r["data"], L)
        inte = interior_qois(r["data"], L, STATIONS)
        r["qois"] = {**ex, **inte}
        r["groups"] = {**{k: "exit" for k in ex},
                       **{k: "interior" for k in inte}}
    o = sorted(loaded.values(), key=lambda r: r["h"])
    r21, r32 = o[1]["h"] / o[0]["h"], o[2]["h"] / o[1]["h"]
    res = {q: gci_3grid(o[0]["qois"][q], o[1]["qois"][q], o[2]["qois"][q],
                        r21, r32) for q in o[0]["qois"]}
    return o, (r21, r32), res


def cell(r):
    if r.get("conv") != "converging":
        return f"{r['conv'][:4].upper()} R={r['R']:+.2f}"
    if r["p_clamped"]:
        return f"p={r['p']:.1f} OUT-OF-BAND"
    return f"PASS p={r['p']:.2f} GCI={r['gci_fine'] * 100:.3f}%"


def main():
    print(__doc__.split("Run:")[0].strip())
    print()
    out = {}
    for name, runs in LADDERS.items():
        o, (r21, r32), res = analyse(runs)
        out[name] = (o, res)
        ssrs = ", ".join(f"{r['ssr']:.1e}" for r in o)
        print(f"{name}: grids {[r['Nx'] for r in o]}  "
              f"r21={r21:.3f} r32={r32:.3f}  SSR [{ssrs}]")
    print()

    names = list(out)
    qois = list(out[names[0]][1])
    print(f"  {'QoI':22s}" + "".join(f"{n:>30s}" for n in names))
    for grp in ("exit", "interior"):
        print(f"  -- {grp.upper()} --")
        for q in qois:
            if out[names[0]][0][0]["groups"][q] != grp:
                continue
            print(f"  {q:22s}" + "".join(f"{cell(out[n][1][q]):>30s}"
                                         for n in names))
    print()

    # How much slope does each limiter actually keep for alpha?
    print("  alpha slope retained by the limiter at each station (N=500):")
    print(f"  {'x/L':>6}" + "".join(f"{n:>30s}" for n in names))
    for f in STATIONS:
        row = f"  {f:6.2f}"
        for n in names:
            d = out[n][0][0]["data"]
            row += f"{slope_kept(d, f):30.4f}"
        print(row)
    print()
    print("  (magnitude column uses the eps ~ |U| floor the run actually used;")
    print("   the smoothness column is shown with the same formula for reference,")
    print("   so it indicates what the OLD limiter WOULD have kept on that field.)")


if __name__ == "__main__":
    main()
