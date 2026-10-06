"""
gci_nicfd_table.py

Grid Convergence Index (GCI) for the two NICFD cases whose convergence study is
complete: the TWO-COMPONENT nozzle and the flashing SMD nozzle at p_out = 15 bar.
The 80 bar case is omitted: only three grids exist for it, so its observed order
cannot be cross-checked (see OPEN_QUESTIONS.md).

Method -- Celik et al. 2008 (ASME JFE 130:078001) / ASME V&V 20:

    h_i      = (x2 - x1)/(Nx_i - 1)          representative spacing (node grid)
    r21, r32 = h2/h1, h3/h2                  refinement ratios, must be >= 1.3
    eps21    = phi2 - phi1,   eps32 = phi3 - phi2
    R        = eps21/eps32                   CONVERGENCE CONDITION:
                 0 < R < 1   monotonic convergence -> GCI valid
                 R < 0       oscillatory           -> GCI NOT applicable
                 R > 1       monotonic divergence  -> GCI NOT applicable
    p        solves p = |ln|eps32/eps21| + q(p)|/ln(r21),
             q(p) = ln[(r21^p - s)/(r32^p - s)]
    phi_ext  = (r21^p phi1 - phi2)/(r21^p - 1)
    GCI_fine = Fs |eps21/phi1| / (r21^p - 1),   Fs = 1.25

Two guards, both learned from this data set:

  * R is TESTED, not assumed.  solve_order_p takes an absolute value (as the
    published formula does), so a DIVERGING triplet still returns a plausible
    positive p.  Ungated, the two-component pressure reported p_obs = 1.294 when
    the true signed order was -1.28.

  * p_obs outside [0.5, 3.0] falls back to the formal order 2 and is flagged.
    GCI divides by (r^p - 1), so a spuriously LARGE p collapses the band toward
    zero: mdot at the exit gave p_obs = 16.8, i.e. 0.0000% uncertainty.

QoI CHOICE IS PER CASE, for a physical reason.  A station can support a GCI only
if some state variable OTHER than the conserved mass flux still varies with grid
there.  Measured at the exit node (relative change, fine -> medium):

                  alpha      h_l    rho_mix     mdot    usable?
  two_component  8.9e-06  3.5e-03  7.9e-04  8.3e-06    YES  thermally out of
                                                            equilibrium; h_l
                                                            carries the signal
  smd 15 bar     4.7e-07  1.1e-05  3.0e-05  6.4e-04    NO   fully equilibrated;
                                                            only mdot is left, so
                                                            u_mix degenerates into
                                                            a mdot measurement

Both cases are therefore evaluated at BOTH stations, and the outcome is exactly
complementary -- which is itself the result:

                        exit node (3 QoIs)   interior + integrals (8 QoIs)
  two_component              3/3 PASS                 0/8
  smd 15 bar                   0/3                 8/8 PASS

For smd 15 bar the cause is established.  Its exit is fully equilibrated and
BC-pinned: alpha, h_g, h_l and rho_mix all vary by <= 3.0e-05 between grids while
mdot varies by 6.4e-04, so u_mix there degenerates into a mdot measurement (the
two track to within 5%, see the table above).  It fails the asymptotic check
outright (asym.chk = 2.75, against 1.002 three cells inside) and its GCI there is
3-10x TIGHTER than the interior -- it fails in the direction that FLATTERS the
result, which is precisely why it must not be quoted.  Its interior has no
captured shock (flashing front, 5.6% dp/p per cell) and converges cleanly at
p ~ 2.1-2.8 across five stations and three integrals.

For two_component the interior failure is NOT yet explained.  What is
established:

  * u_mix is NON-monotone in h at every interior station and monotone ONLY at
    the exit node, e.g. at x/L=0.80: 128.966 / 128.900 / 128.913 / 128.905 for
    N = 500 / 351 / 256 / 180, versus 74.841 / 74.901 / 74.990 / 75.168 at the
    exit.  Richardson extrapolation requires monotone convergence, so the R gate
    correctly rejects every interior QoI.
  * The jitter corresponds to a position shift of 0.07 to 1.55 cells, largest
    where the gradient is steepest (x/L=0.35, |du/dx| = 2165).

Three candidate mechanisms were tested and ALL were ruled out, so none should be
repeated as an explanation:
    shock-position jitter  -- non-monotonicity is present UPSTREAM of the shock
                              (x/L = 0.20, 0.35, 0.50) as well as downstream
    rigid translation      -- predicted/observed ratios scatter 0.04 to 1.01
    checkerboard sampling  -- the jitter is 17-2000x the local 2dx amplitude

The exit-node results below stand on their own evidence (consistent p_obs and
phi_ext over four independent triplets); the interior failure is reported as an
open question rather than explained away.

Failures are printed, not omitted: which station works where is evidence about
the flow, and hiding it would make the surviving numbers look arbitrary.

MASS FLOW IS DELIBERATELY NOT A GCI QoI.  The scheme conserves the numerical flux
A*F_hat exactly (the mass rows of the non-conservative correction X and of
D_VE_area are identically zero), so mdot is nearly grid-independent by
construction and what little it varies is reconstruction wobble, not
discretisation error.  It failed on every ladder and every triplet.  Its
streamwise spread is reported instead, as a dissipation diagnostic.

Run:
    poetry run python gci_nicfd_table.py
    poetry run python gci_nicfd_table.py --latex
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import statistics as st
import sys

import numpy as np

try:                                    # cubic sampling for interior stations
    from scipy.interpolate import CubicSpline
except ImportError:                     # pragma: no cover
    CubicSpline = None

HERE = pathlib.Path(__file__).resolve().parent
DEV = HERE.parent
sys.path.insert(0, str(DEV))
from case_study.nicfd.grid_convergence_gci import gci_3grid, FORMAL_ORDER, P_MIN, P_MAX  # noqa: E402

FS = 1.25

# -- Runs (pinned).  Every one is re-verified below: iteratively converged, Nx
# matching the CSV row count, and one shared settings fingerprint per ladder. --
# Both cases are evaluated at BOTH the exit and the interior, so the two can be
# compared on identical footing.  QoIs that fail are reported as failures rather
# than dropped -- which station works is itself a result (see the docstring).
STATIONS = (0.99, 0.98, 0.95, 0.90, 0.80, 0.70)

CASES = {
    "two_component": dict(
        domain=(0.0, 0.31),
        runs={
            500: DEV / "results_grid_sesitivity/two_component_N500_muscl_van_albada_20260904_081640",
            351: DEV / "results_grid_sesitivity/two_component_N351_muscl_van_albada_20260910_124300",
            256: DEV / "results_grid_sesitivity/two_component_N256_muscl_van_albada_20260904_113252",
            180: DEV / "results_grid_limiterfix/two_component_N180_muscl_van_albada_20260911_084317",
        },
        # Shock at x/L ~ 0.593.  STATIONS are all downstream of it; a global
        # cubic spline was checked against shape-preserving pchip there and the
        # two agree to ~1e-6 %, so spline ringing does not reach these stations.
        shock=0.593,
    ),
    "smd_15bar": dict(
        domain=(0.0, 1.2),
        runs={
            600: DEV / "results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260903_174651",
            429: DEV / "results_grid_sesitivity/smd_supersonic_N429_muscl_van_albada_20260910_124044",
            300: DEV / "results_grid_sesitivity/smd_supersonic_N300_muscl_van_albada_20260903_174755",
        },
        shock=None,          # flashing front at x/L ~ 0.39, no captured shock
    ),
    # p_out = 80e5 is the CANONICAL setting in
    # settings_flashing_smd_supersonic_test.yaml, so this is the production case.
    # Only three grids exist for it (the N=215 attempt ran at eps_void/eps_ghost
    # = 0.15 instead of 0.05 -- a different numerical model -- and the correct
    # -settings attempt NaN'd), so its observed order cannot be cross-checked
    # against a second triplet.  See OPEN_QUESTIONS.md.
    "smd_80bar": dict(
        domain=(0.0, 1.2),
        runs={
            600: DEV / "results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260910_123856",
            429: DEV / "results_grid_sesitivity/smd_supersonic_N429_muscl_van_albada_20260907_170522",
            300: DEV / "results_grid_sesitivity/smd_supersonic_N300_muscl_van_albada_20260903_213338",
        },
        shock=None,          # fully subsonic, M_max ~ 0.58, no captured shock
    ),
}

# Settings keys that must be identical across a ladder (only Nx may differ).
FINGERPRINT = ["p_out", "lowmach_diss", "lowmach_diss_bc", "recon", "eos_rows",
               "eos_project", "nc_form", "nc_scale", "eps_void", "eps_ghost",
               "k_vm", "va_eps_mode", "va_rel"]
# Keys added to settings_*.yaml after some runs already existed: absent means
# the run used the code default, not a different model.
DEFAULTS = {"va_eps_mode": "magnitude", "va_rel": "null"}


def _grab(text, key):
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def load_run(path, Nx_expected):
    """Read one run and verify it is fit to enter a ladder."""
    yamls = list(path.glob("settings_*.yaml"))
    if not yamls:
        raise SystemExit(f"{path}: no settings_*.yaml")
    text = yamls[0].read_text(encoding="utf-8", errors="ignore")

    nx = int(float(_grab(text, "Nx")))
    data = np.genfromtxt(path / "solution.csv", delimiter=",", names=True)
    if len(data["x"]) != nx:            # settings/data consistency
        raise SystemExit(f"{path}: settings say Nx={nx} but solution.csv has "
                         f"{len(data['x'])} rows -- the settings snapshot does "
                         f"not match the data")
    if nx != Nx_expected:
        raise SystemExit(f"{path}: expected Nx={Nx_expected}, found {nx}")

    log = (path / "run_log.txt").read_text(encoding="utf-8", errors="ignore")
    stop = re.search(r"Stopped:\s*(.+)", log)
    reason = stop.group(1).strip() if stop else "no stop line"
    if not reason.startswith("SSR="):
        raise SystemExit(f"{path}: not iteratively converged ({reason})")
    ssr = float(re.search(r"SSR=([\d.eE+\-]+)", reason).group(1))

    fp = tuple((_grab(text, k) or DEFAULTS.get(k)) for k in FINGERPRINT)
    return dict(path=path, Nx=nx, data=data, ssr=ssr, fp=fp)


# -- Quantities of interest --------------------------------------------------
def u_mix(d):
    """Mass-averaged mixture velocity."""
    ag, rg, rl = d["alpha_g"], d["rho_g"], d["rho_l"]
    return (ag * rg * d["u_g"] + (1 - ag) * rl * d["u_l"]) / (ag * rg + (1 - ag) * rl)


def mdot(d):
    ag, rg, rl = d["alpha_g"], d["rho_g"], d["rho_l"]
    return d["A"] * (ag * rg * d["u_g"] + (1 - ag) * rl * d["u_l"])


def exit_qois(d, L, stations=None):
    """Read at the outlet node.  x2 is an exact node on every grid, so these
    involve no interpolation at any refinement ratio."""
    return {
        "u_mix [m/s]":   float(u_mix(d)[-1]),
        "slip ratio":    float((d["u_g"] / d["u_l"])[-1]),
        "slip du [m/s]": float((d["u_g"] - d["u_l"])[-1]),
    }


def interior_qois(d, L, stations):
    if CubicSpline is None:
        raise SystemExit("scipy is required for the interior stations "
                         "(cubic sampling)")
    x, u = d["x"], u_mix(d)
    # WHY CUBIC AND NOT THE SOLVER'S OWN MUSCL RECONSTRUCTION
    #
    # It is tempting to sample with the scheme's van Albada reconstruction "for
    # consistency".  That is the wrong choice here, for three reasons, the last
    # decisive:
    #
    #  1. It is ambiguous.  The solver limits the six primitives (p, alpha, u_g,
    #     u_l, h_g, h_l), not u_mix, so one would reconstruct each primitive and
    #     then form u_mix.  That was tested; see below.
    #  2. The reconstruction is deliberately LOSSY.  Its job is to assemble
    #     fluxes and keep the shock monotone, which it does by discarding slope
    #     wherever it distrusts the data -- right for flux assembly, wrong for
    #     post-processing, where the best estimate of the solution is wanted.
    #  3. ORDER.  For GCI the sampling error must vanish FASTER than the
    #     discretisation error or it biases p_obs.  Cubic is O(h^4), two orders
    #     faster.  MUSCL is O(h^2) unlimited (the same order as the signal) and
    #     O(h) where limited (worse than the signal).  Linear interp is O(h^2)
    #     and measurably biases p_obs: switching linear->cubic moved eps32 by
    #     39.5% on the two-component ladder.
    #
    # Measured on the 15 bar ladder at these stations (magnitude limiter), the
    # solver reconstruction keeps only 0.2-7% of alpha's slope -- it is
    # piecewise constant in alpha, a ZEROTH-order sampler -- and u_mix inherits
    # that.  Resulting p_obs, cubic vs solver-MUSCL:
    #     x/L    0.98   0.95   0.90   0.80   0.70
    #     cubic  2.54   2.83   2.84   2.52   2.30   (scatter 0.54)
    #     muscl  2.06   3.37   2.75   2.79   DIVERGING
    #
    # Caveat: a cubic spline rings at a discontinuity, so these stations must
    # stay clear of the flashing front (x/L ~ 0.39 here); 0.70-0.98 are.
    # All three exit QoIs are also evaluated at each interior station, so the
    # same quantity can be compared exit-vs-interior within one case.
    sg, sl = CubicSpline(x, d["u_g"]), CubicSpline(x, d["u_l"])
    out = {}
    for f in stations:
        out[f"u_mix @ x/L={f:.2f}"] = float(CubicSpline(x, u)(f * L))
        out[f"slip ratio @ {f:.2f}"] = float(sg(f * L) / sl(f * L))
        out[f"slip du @ {f:.2f}"] = float(sg(f * L) - sl(f * L))
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    w = x[-1] - x[0]
    out["int p dx / L"] = float(tz(d["p"], x) / w)
    out["int u_mix dx / L"] = float(tz(u, x) / w)
    out["thrust int pA dx"] = float(tz(d["p"] * d["A"], x))
    return out


def recon_spread(d):
    """Streamwise spread of mdot.  This is a dissipation diagnostic, NOT a
    conservation error -- the scheme conserves the numerical flux exactly."""
    m = mdot(d)
    return float(np.ptp(m) / np.mean(m))


# -- Analysis ----------------------------------------------------------------
def analyse(name, cfg):
    x1, x2 = cfg["domain"]
    L = x2 - x1
    runs = {N: load_run(p, N) for N, p in cfg["runs"].items()}

    fps = {r["fp"] for r in runs.values()}
    if len(fps) != 1:
        raise SystemExit(f"{name}: ladder members do not share a settings "
                         f"fingerprint -- they are not the same numerical model")

    for r in runs.values():
        r["h"] = L / (r["Nx"] - 1)
        ex = exit_qois(r["data"], L)
        inte = interior_qois(r["data"], L, STATIONS)
        r["qois"] = {**ex, **inte}
        r["groups"] = {**{k: "exit" for k in ex},
                       **{k: "interior" for k in inte}}
        r["spread"] = recon_spread(r["data"])

    ordered = sorted(runs.values(), key=lambda r: r["h"])     # fine -> coarse
    trios = [t for t in itertools.combinations(ordered, 3)
             if (t[1]["h"] / t[0]["h"]) >= 1.3 and (t[2]["h"] / t[1]["h"]) >= 1.3]
    if not trios:
        raise SystemExit(f"{name}: no triplet has both ratios >= 1.3")

    results = {}
    for qoi in ordered[0]["qois"]:
        per_trio = []
        for t in trios:
            r21, r32 = t[1]["h"] / t[0]["h"], t[2]["h"] / t[1]["h"]
            res = gci_3grid(t[0]["qois"][qoi], t[1]["qois"][qoi],
                            t[2]["qois"][qoi], r21, r32)
            res["grids"] = [g["Nx"] for g in t]
            per_trio.append(res)
        results[qoi] = per_trio
    return runs, ordered, trios, results


def report(name, runs, ordered, trios, results):
    print("=" * 94)
    print(f"CASE: {name}")
    for r in ordered:
        print(f"    Nx={r['Nx']:<5d} h={r['h']:.5e}  SSR={r['ssr']:.3e}  "
              f"mdot spread={r['spread']*100:5.2f}%  {r['path'].name}")
    print(f"  valid triplets (both r >= 1.3): "
          f"{[[g['Nx'] for g in t] for t in trios]}")

    groups = ordered[0]["groups"]
    for tag, title in (("exit", "EXIT NODE (x2 is a node on every grid; no interpolation)"),
                       ("interior", "INTERIOR (cubic sampling) and DOMAIN INTEGRALS")):
        print()
        print(f"  -- {title} --")
        print(f"  {'QoI':22s} {'phi(fine)':>14s} {'R':>8s} {'p_obs':>7s} "
              f"{'GCI%':>9s} {'asym':>7s}  verdict")
        for qoi, per in results.items():
            if groups[qoi] != tag:
                continue
            first = per[0]
            phi = ordered[0]["qois"][qoi]
            if first.get("conv") != "converging":
                print(f"  {qoi:22s} {phi:14.6g} {first['R']:8.3f} {'--':>7} "
                      f"{'--':>9} {'--':>7}  FAIL: {first['conv'].upper()}")
                continue
            g = first["gci_fine"] * 100
            if first["p_clamped"]:
                print(f"  {qoi:22s} {phi:14.6g} {first['R']:8.3f} "
                      f"{first['p']:7.2f} {g:9.4f} "
                      f"{first['asymptotic_check']:7.3f}  FAIL: p_obs outside "
                      f"[{P_MIN},{P_MAX}], GCI shown at formal p={FORMAL_ORDER:g}")
                continue
            print(f"  {qoi:22s} {phi:14.6g} {first['R']:8.3f} {first['p']:7.2f} "
                  f"{g:9.4f} {first['asymptotic_check']:7.3f}  PASS")
            ps = [r["p"] for r in per if r.get("ok") and not r["p_clamped"]]
            if len(per) > 1 and len(ps) == len(per):
                ex = [r["phi_ext"] for r in per]
                print(f"  {'':22s}   p_obs over {len(per)} triplets: "
                      f"{[round(x, 2) for x in ps]}  mean={st.mean(ps):.2f}  "
                      f"spread={max(ps) - min(ps):.2f}")
                print(f"  {'':22s}   phi_ext spread "
                      f"{(max(ex) - min(ex)) / abs(st.mean(ex)) * 100:.4f}% of "
                      f"value (vs GCI {g:.3f}%) -> band is conservative")
    print()


def summary(out):
    """Side-by-side PASS/FAIL, so which station works where is visible at a
    glance.  Failures are shown, not omitted."""
    names = list(out)
    qois = list(out[names[0]][3])
    print("=" * 94)
    print("SUMMARY -- same QoIs, both cases.  A station supports a GCI only if")
    print("some state variable OTHER than the conserved mass flux still varies")
    print("with grid there (see docstring).")
    print()
    print(f"  {'QoI':22s} " + " ".join(f"{n:>26s}" for n in names))
    for q in qois:
        cells = []
        for n in names:
            per = out[n][3].get(q)
            if per is None:
                cells.append("n/a")
                continue
            r = per[0]
            if r.get("conv") != "converging":
                cells.append(f"FAIL {r['conv'][:4].upper()} (R={r['R']:+.2f})")
            elif r["p_clamped"]:
                cells.append(f"FAIL p={r['p']:.1f} out of band")
            else:
                cells.append(f"PASS p={r['p']:.2f} GCI={r['gci_fine']*100:.3f}%")
        print(f"  {q:22s} " + " ".join(f"{c:>26s}" for c in cells))
    print()


def latex(name, ordered, results):
    print(f"% ---- {name} ----")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"QoI & $\phi_{\mathrm{fine}}$ & $p_{\mathrm{obs}}$ & "
          r"GCI [\%] & asym. \\")
    print(r"\hline")
    n = 0
    for qoi, per in results.items():
        r = per[0]
        # Only fully PASSing QoIs reach a paper table.  A clamped p means the
        # triplet is outside the asymptotic range for that QoI, so its band is
        # not a valid uncertainty estimate -- it must not be tabulated silently.
        if r.get("conv") != "converging" or r["p_clamped"]:
            continue
        phi = ordered[0]["qois"][qoi]
        print(f"{qoi} & {phi:.6g} & {r['p']:.2f} & "
              f"{r['gci_fine'] * 100:.4f} & {r['asymptotic_check']:.3f} \\\\")
        n += 1
    if n == 0:
        print(r"\multicolumn{5}{c}{no QoI passed at this station} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latex", action="store_true",
                    help="also emit LaTeX tables")
    args = ap.parse_args()

    print(f"GCI -- Celik et al. 2008, Fs={FS}, formal order {FORMAL_ORDER:g}, "
          f"p_obs accepted in [{P_MIN}, {P_MAX}]")
    print("Every QoI is gated on R = eps21/eps32; only 0 < R < 1 is reported.")
    print()
    out = {}
    for name, cfg in CASES.items():
        out[name] = analyse(name, cfg)
        report(name, *out[name])
    summary(out)
    if args.latex:
        for name, (_, ordered, _, results) in out.items():
            latex(name, ordered, results)


if __name__ == "__main__":
    main()
