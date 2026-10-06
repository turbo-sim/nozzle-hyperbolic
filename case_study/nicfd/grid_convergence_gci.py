"""
grid_convergence_gci.py — GCI (Grid Convergence Index) post-processing for the
1D nozzle_hyperbolic runs, for the NICFD 2026 grid-convergence study.

Read-only.  Does not touch any solver, config, or result file — it only reads
`settings_*.yaml` + `solution.csv` + `run_log.txt` from existing run folders
and prints/writes a report.

Method
------
Standard 3-grid Richardson/GCI procedure (Celik et al. 2008, ASME V&V 20 /
Roache 1998):

    h_i        = (x2 - x1) / (Nx_i - 1)                  representative spacing
    r21 = h2/h1, r32 = h3/h2                              refinement ratios (>=1.3)
    eps21 = phi2 - phi1,  eps32 = phi3 - phi2             (1=fine, 2=medium, 3=coarse)
    s     = sign(eps32/eps21)
    p     solves  p = |ln|eps32/eps21| + q(p)| / ln(r21),
                  q(p) = ln[(r21^p - s)/(r32^p - s)]       (fixed-point iteration)
    phi_ext = (r21^p * phi1 - phi2) / (r21^p - 1)          Richardson extrapolation
    GCI_fine = Fs * |eps21/phi1| / (r21^p - 1),  Fs = 1.25 for a verified 3-grid order

Grids are discovered automatically and grouped by a "fingerprint" of every
setting that should NOT vary within one GCI ladder (p_out, lowmach_diss_bc,
recon, eos_rows/eos_project, nc_form/nc_scale, eps_void, eps_ghost, k_vm,
x1, x2, and the settings-file family) — only Nx is allowed to differ inside a
group.  This is deliberate: a first pass over this project's runs found two
folders both named "*_N300_*" that turned out to differ in p_out (one is the
subsonic regime, one the shock-pressure regime), and a "lowmach_diss_bc"
difference between the published subsonic/adapted runs (=0.0) and the
published shock run (=1.0).  Fingerprinting on content rather than folder
name avoids silently mixing those.

Usage
-----
    python grid_convergence_gci.py
    python grid_convergence_gci.py --root results_grid_sesitivity --frac 0.99
    python grid_convergence_gci.py --report GCI_REPORT.md
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import math
import os
import re
import sys
from typing import Optional

import numpy as np

# ── Where to look ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOTS = [
    os.path.join(HERE, "results_grid_sesitivity"),
]
# Published reference runs to fold in when their settings fingerprint matches
# a group found under DEFAULT_ROOTS (if it doesn't match, it is just reported
# on its own and ignored for GCI — decided by the data, not assumed here).
DEFAULT_EXTRA_RUNS = [
    os.path.join(HERE, "results_two_phase",
                 "smd_supersonic_N600_muscl_van_albada_20260725_173528"),
    os.path.join(HERE, "results_two_phase",
                 "smd_supersonic_N420_muscl_van_albada_20260723_090657"),
]

# Settings keys that define "the same numerical model" — only Nx (and the
# derived h) is allowed to differ between runs of one GCI ladder.
FINGERPRINT_KEYS = [
    "p_out", "lowmach_diss", "lowmach_diss_bc", "recon", "eos_rows",
    "eos_project", "nc_form", "nc_scale", "eps_void", "eps_ghost", "k_vm",
    # van Albada eps normalisation (added 2026-09-09).  This changes the
    # reconstruction operator itself, so runs that differ here are NOT the same
    # numerical model and must never share a ladder.  Absent in older settings
    # files -> parses as None -> those runs group together as before.
    "va_eps_mode", "va_rel", "va_r0", "va_rm",
]
GRID_KEYS = ["Nx", "x1", "x2"]

# Evaluated at the OUTLET NODE.  x2 is exactly the last node on every grid
# regardless of Nx, so these need no interpolation on any refinement ratio --
# which removes the error that dominates the frac-sampled QoIs below (measured:
# switching linear->cubic interpolation moved eps32 by 39.5%).
# NOTE: exit pressure is deliberately NOT a QoI -- it is identically p_out
# (verified: p[-1] - p_out = 0.0000), i.e. imposed by the BC with zero grid
# dependence.  Velocities and mdot are extrapolated, so they are valid.
QOIS_EXIT = ["u_mix", "mdot_exit", "slip_ratio", "slip_du"]
# Interpolated at x1 + frac*(x2-x1).  Kept for continuity with the earlier
# report; the R gate below will refuse them when they do not converge.
QOIS_POINT = ["p", "alpha_g", "mdot"]
QOIS = QOIS_EXIT + QOIS_POINT
X_FRAC_DEFAULT = 0.99          # interpolate QoIs at x1 + frac*(x2-x1)
FS_3GRID = 1.25                # Celik safety factor, observed order verified
FORMAL_ORDER = 2.0             # MUSCL + van Albada
P_MIN, P_MAX = 0.5, 3.0        # plausible band for p_obs around FORMAL_ORDER
FS_2GRID = 3.0                 # Roache safety factor, order ASSUMED (2 grids only)


# ── Settings / log parsing (regex — no yaml dependency needed) ──────────────
def _grab(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", text, re.MULTILINE)
    return m.group(1) if m else None


# Keys added to settings_*.yaml AFTER runs already existed.  A run whose file
# predates the key is not a different numerical model -- it used the code
# default.  Normalise absent (and the YAML spelling of "use the default") to one
# canonical token, or every pre-existing run splits off into its own group and
# no ladder can ever be completed.
FINGERPRINT_DEFAULTS = {
    "va_eps_mode": "magnitude",   # solver default before the key existed
    "va_rel": "null",             # null => mode-calibrated default (1e-3)
    "va_r0": "0.5",
    "va_rm": "4",
}


def parse_settings(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    out = {}
    for key in GRID_KEYS + FINGERPRINT_KEYS:
        raw = _grab(text, key)
        if raw is None and key in FINGERPRINT_DEFAULTS:
            raw = FINGERPRINT_DEFAULTS[key]
        out[key] = raw
    for key in ("Nx",):
        if out[key] is not None:
            out[key] = int(float(out[key]))
    for key in ("x1", "x2", "p_out"):
        if out[key] is not None:
            out[key] = float(out[key])
    return out


def parse_run_log(path: str) -> dict:
    if not os.path.exists(path):
        return dict(ssr_final=None, ssr_tol=None, converged=None, stop_reason=None)
    text = open(path, encoding="utf-8", errors="ignore").read()

    tol_m = re.search(r"SSR_tol\s*=\s*([\d.eE+\-]+)", text)
    ssr_tol = float(tol_m.group(1)) if tol_m else None

    rows = re.findall(
        r"^\s*\d+\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+([\d.eE+\-]+)\s*\(", text, re.MULTILINE
    )
    ssr_final = float(rows[-1]) if rows else None

    stop_m = re.search(r"Stopped:\s*(.+)", text)
    stop_reason = stop_m.group(1).strip() if stop_m else None

    converged = bool(stop_m and stop_m.group(1).strip().startswith("SSR="))
    return dict(ssr_final=ssr_final, ssr_tol=ssr_tol, converged=converged,
                stop_reason=stop_reason)


# ── Run discovery ─────────────────────────────────────────────────────────
@dataclasses.dataclass
class Run:
    path: str
    settings_file: str
    Nx: int
    x1: float
    x2: float
    h: float
    fingerprint: tuple
    family: str
    ssr_final: Optional[float]
    ssr_tol: Optional[float]
    converged: Optional[bool]
    stop_reason: Optional[str]
    qois: dict


def find_run_dirs(root: str) -> list:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in glob.glob(os.path.join(root, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "solution.csv"))
    )


def load_qois(csv_path: str, x1: float, x2: float, frac: float) -> dict:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    interior = data[1:-1]                       # exclude ghost cells (first/last row)
    x = interior["x"]
    x_t = x1 + frac * (x2 - x1)
    if not (x[0] <= x_t <= x[-1]):
        x_t = x[-1]                              # fall back to last interior cell

    p = np.interp(x_t, x, interior["p"])
    alpha_g = np.interp(x_t, x, interior["alpha_g"])
    mdot_cell = interior["A"] * (
        interior["alpha_g"] * interior["rho_g"] * interior["u_g"]
        + (1.0 - interior["alpha_g"]) * interior["rho_l"] * interior["u_l"]
    )
    mdot = np.interp(x_t, x, mdot_cell)
    # NOT a conservation check (verified 2026-09-10).  The scheme conserves the
    # NUMERICAL FLUX A*F_hat at faces exactly: the mass rows of the
    # non-conservative correction X are identically zero (matrices.py:383,
    # "Rows: (zero, zero, gas_mom, liq_mom, gas_E, liq_E)") and the area source
    # D_VE_area is [0, 0, ...] in the mass rows too.  What varies here is the
    # CELL-CENTRED product A*(a*rho_g*u_g + (1-a)*rho_l*u_l), which differs from
    # F_hat by the FVS upwind/dissipation terms -- an O(h^p) reconstruction
    # artefact, not a lost-mass error.  Evidence: the deviation profile keeps the
    # same SHAPE on all three grids (corr +0.98 N429 vs N600) while its amplitude
    # scales down ~O(h^0.9).  It is still a useful diagnostic: it is largest where
    # the scheme is most dissipative, so it flags shock-dominated cases (the
    # two_component ladder peaks at +1.38% at x/L=0.595, exactly at its 26.7%/cell
    # shock, versus 0.77% for the shock-free smd 15 bar case).
    mdot_spread = float(np.ptp(mdot_cell) / np.mean(mdot_cell))

    # ── Exit-node QoIs (full array: the first/last rows are the domain
    # boundary NODES, not ghost cells -- x[0] == x1 and x[-1] == x2 exactly).
    ag, rg, rl = data["alpha_g"], data["rho_g"], data["rho_l"]
    ug, ul = data["u_g"], data["u_l"]
    rho_mix = ag * rg + (1.0 - ag) * rl
    u_mix = (ag * rg * ug + (1.0 - ag) * rl * ul) / rho_mix     # mass-averaged
    mdot_x = data["A"] * (ag * rg * ug + (1.0 - ag) * rl * ul)

    return dict(p=float(p), alpha_g=float(alpha_g), mdot=float(mdot),
                mdot_spread=mdot_spread,
                u_mix=float(u_mix[-1]),
                mdot_exit=float(mdot_x[-1]),
                slip_ratio=float(ug[-1] / ul[-1]),
                slip_du=float(ug[-1] - ul[-1]))


def build_run(path: str, frac: float) -> Optional[Run]:
    settings_files = glob.glob(os.path.join(path, "settings_*.yaml"))
    if not settings_files:
        return None
    settings_file = settings_files[0]
    s = parse_settings(settings_file)
    if s["Nx"] is None or s["x1"] is None or s["x2"] is None:
        return None

    # CONSISTENCY GUARD (2026-09-11): settings_*.yaml and solution.csv must
    # describe the SAME grid.  A folder was found whose yaml said Nx=183 while
    # solution.csv held 351 points -- the settings snapshot had been overwritten
    # after the run.  Read that way, h would be 0.31/182 instead of 0.31/350,
    # a 1.92x error in the representative spacing, silently corrupting any ladder
    # the run joined.  h is therefore taken from the DATA and cross-checked.
    _n_rows = int(np.genfromtxt(os.path.join(path, "solution.csv"),
                                delimiter=",", names=True).shape[0])
    if _n_rows != s["Nx"]:
        raise ValueError(
            f"grid mismatch: settings say Nx={s['Nx']} but solution.csv has "
            f"{_n_rows} rows -- settings snapshot does not match the data")
    h = (s["x2"] - s["x1"]) / (s["Nx"] - 1)
    fp = tuple(s[k] for k in FINGERPRINT_KEYS) + (s["x1"], s["x2"])
    family = os.path.basename(settings_file)

    log = parse_run_log(os.path.join(path, "run_log.txt"))
    qois = load_qois(os.path.join(path, "solution.csv"), s["x1"], s["x2"], frac)

    return Run(path=path, settings_file=family, Nx=s["Nx"], x1=s["x1"], x2=s["x2"],
               h=h, fingerprint=fp, family=family, qois=qois, **log)


def group_runs(runs: list) -> dict:
    groups: dict = {}
    for r in runs:
        groups.setdefault((r.family,) + r.fingerprint, []).append(r)
    # dedupe same-Nx entries within a group, keep the one with more recent
    # folder name (last one found, since find_run_dirs is sorted)
    deduped = {}
    for key, members in groups.items():
        by_nx = {}
        for m in members:
            prev = by_nx.get(m.Nx)
            if prev is None:
                by_nx[m.Nx] = m
                continue
            # Prefer the CONVERGED run, then the lower final SSR, then the later
            # folder.  Keeping "whatever sorted last" silently replaced a
            # converged N=256 (SSR 1.0e-08) with an interrupted one (1.6e-06)
            # merely because a second scan root sorted after the first.
            def _rank(r):
                return (1 if r.converged else 0,
                        -(r.ssr_final if r.ssr_final is not None else float("inf")),
                        os.path.basename(r.path))
            keep, drop = (m, prev) if _rank(m) > _rank(prev) else (prev, m)
            _s = lambda r: ("?" if r.ssr_final is None
                            else f"{r.ssr_final:.2e}")
            print(f"  [note] duplicate Nx={m.Nx} in group; keeping "
                  f"{os.path.basename(keep.path)} (SSR={_s(keep)}) over "
                  f"{os.path.basename(drop.path)} (SSR={_s(drop)})")
            by_nx[m.Nx] = keep
        deduped[key] = sorted(by_nx.values(), key=lambda r: -r.Nx)  # finest first
    return deduped


def select_triplet(members: list) -> list:
    """Pick the best 3 grids from a group, finest first.

    Taking the three finest blindly is wrong once a group holds a superseded
    grid: {500, 358, 351, 256} gives r32 = 1.02, far below the 1.3 minimum,
    while {500, 351, 256} gives 1.426/1.373.  Rank every 3-subset by, in order:
      1. all three iteratively converged   (an unconverged grid makes eps
         meaningless, so this outranks everything else)
      2. both refinement ratios >= 1.3     (Celik minimum)
      3. largest min(r21, r32)             (better separated => better
         conditioned p-solve, and eps well above the noise floor)
      4. finest available h1               (tie-break)
    """
    best, best_key = None, None
    for trio in itertools.combinations(sorted(members, key=lambda r: r.h), 3):
        g1, g2, g3 = trio                       # h ascending => fine, med, coarse
        r21, r32 = g2.h / g1.h, g3.h / g2.h
        key = (all(g.converged for g in trio),
               r21 >= 1.3 and r32 >= 1.3,
               min(r21, r32),
               -g1.h)
        if best_key is None or key > best_key:
            best, best_key = list(trio), key
    return best


# ── GCI math ─────────────────────────────────────────────────────────────
def solve_order_p(eps32: float, eps21: float, r21: float, r32: float,
                   max_iter: int = 200, tol: float = 1e-12) -> Optional[float]:
    if eps21 == 0 or eps32 == 0:
        return None
    ratio = eps32 / eps21
    s = 1.0 if ratio > 0 else -1.0
    ln_ratio = math.log(abs(ratio))
    p = max(abs(ln_ratio) / math.log(r21), 0.1)
    for _ in range(max_iter):
        num = r21 ** p - s
        den = r32 ** p - s
        if num <= 0 or den <= 0:
            return None
        q = math.log(num / den)
        p_new = abs(ln_ratio + q) / math.log(r21)
        if not math.isfinite(p_new):
            return None
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def gci_3grid(phi1: float, phi2: float, phi3: float,
              r21: float, r32: float) -> dict:
    eps21 = phi2 - phi1
    eps32 = phi3 - phi2
    p = solve_order_p(eps32, eps21, r21, r32)

    # ── Celik convergence condition.  R = eps21/eps32 classifies the triplet:
    #     0 < R < 1  monotonic convergence  -> GCI applies
    #     R < 0      oscillatory            -> GCI formula NOT applicable
    #     R > 1      monotonic DIVERGENCE   -> the solution moves further apart
    #                as the grid refines; no order exists to extrapolate
    # This MUST be tested before reporting anything.  solve_order_p() takes the
    # absolute value (as the published formula does), so a diverging triplet
    # still yields a plausible-looking positive p -- e.g. the two_component
    # ladder gave p_obs=1.294 for pressure when the true signed order was -1.28.
    R = (eps21 / eps32) if eps32 != 0 else float("nan")
    if R != R:                      # nan: eps32 == 0
        conv = "degenerate"
    elif R < 0.0:
        conv = "oscillatory"
    elif R < 1.0:
        conv = "converging"
    else:
        conv = "diverging"
    out = dict(eps21=eps21, eps32=eps32, p=p, R=R, conv=conv,
               monotonic=(eps32 * eps21 > 0))
    if p is None or phi1 == 0 or conv != "converging":
        out["ok"] = False
        return out
    # Guard against an implausible observed order.  GCI divides by (r^p - 1),
    # so a spuriously LARGE p collapses the band toward zero: the two_component
    # ladder gave p_obs = 16.8 for mdot_exit (the fine pair agreed to 8.3e-6
    # while the coarse grid sat 205x further away), which would have been
    # reported as 0.0000% uncertainty.  A p outside [P_MIN, P_MAX] means the
    # triplet is not in the asymptotic range for this QoI, so fall back to the
    # formal order -- the conservative reading -- and say so.
    p_used, p_clamped = p, False
    if not (P_MIN <= p <= P_MAX):
        p_used, p_clamped = FORMAL_ORDER, True
    phi_ext = (r21 ** p_used * phi1 - phi2) / (r21 ** p_used - 1)
    gci_fine = FS_3GRID * abs(eps21 / phi1) / (r21 ** p_used - 1)
    gci_coarse = FS_3GRID * abs(eps32 / phi2) / (r32 ** p_used - 1)
    asym_check = (gci_coarse / (r21 ** p_used * gci_fine)
                  if gci_fine != 0 else float("nan"))
    out.update(ok=True, p_used=p_used, p_clamped=p_clamped,
               phi_ext=phi_ext, gci_fine=gci_fine, gci_coarse=gci_coarse,
               e_ext=abs((phi_ext - phi1) / phi_ext) if phi_ext != 0 else float("nan"),
               asymptotic_check=asym_check)
    return out


def gci_2grid_assumed(phi1: float, phi2: float, r21: float, p_assumed: float) -> dict:
    eps21 = phi2 - phi1
    if phi1 == 0:
        return dict(ok=False)
    gci = FS_2GRID * abs(eps21 / phi1) / (r21 ** p_assumed - 1)
    return dict(ok=True, p_assumed=p_assumed, eps21=eps21, gci=gci)


# ── Reporting ────────────────────────────────────────────────────────────
def fmt_run_row(r: Run) -> str:
    conv = ("OK" if r.converged else
            "NOT CONVERGED" if r.converged is False else "?")
    tol_txt = f"tol={r.ssr_tol:.1e}" if r.ssr_tol else ""
    ssr_txt = f"SSR={r.ssr_final:.3e}" if r.ssr_final is not None else "SSR=?"
    # mdot_spread: cell-centred reconstruction spread, NOT a conservation error
    # (see load_qois).  Large values flag a dissipation/shock-dominated solution.
    _sp = r.qois.get('mdot_spread')
    _sp_txt = f'mdot_recon_spread={_sp*100:5.2f}%' if _sp is not None else ''
    return (f"    Nx={r.Nx:<5d} h={r.h:.5e}  {ssr_txt} {tol_txt:12s} "
            f"[{conv}]  {_sp_txt}  {os.path.basename(r.path)}")


def report_group(key: tuple, members: list, out_lines: list):
    family = key[0]
    fp_keys = FINGERPRINT_KEYS
    fp_vals = key[1:1 + len(fp_keys)]
    x1, x2 = key[-2], key[-1]
    tag = "  ".join(f"{k}={v}" for k, v in zip(fp_keys, fp_vals) if v is not None)

    out_lines.append("")
    out_lines.append("=" * 88)
    out_lines.append(f"GROUP: {family}")
    out_lines.append(f"  domain x=[{x1}, {x2}]   {tag}")
    out_lines.append(f"  {len(members)} grid(s): " +
                      ", ".join(str(m.Nx) for m in members))
    for m in members:
        out_lines.append(fmt_run_row(m))

    # Only grids in the SELECTED triplet matter -- a superseded run left in the
    # folder must not raise a warning about a ladder it is not part of.
    _used = select_triplet(members) if len(members) >= 3 else members
    not_converged = [m for m in _used if m.converged is False]
    if not_converged:
        out_lines.append("  *** WARNING: the following grid(s) did NOT reach SSR_tol "
                          "-- iterative error is not proven small compared to the "
                          "discretization error. GCI below is PROVISIONAL. ***")
        for m in not_converged:
            out_lines.append(f"      {os.path.basename(m.path)}: "
                              f"stop_reason='{m.stop_reason}'  "
                              f"final SSR={m.ssr_final:.3e}  tol={m.ssr_tol:.1e}  "
                              f"(off by {m.ssr_final / m.ssr_tol:.1f}x)")

    if len(members) < 2:
        out_lines.append("  Only one grid -- no GCI possible (need >= 2, "
                          ">= 3 to verify the observed order).")
        return

    grids = select_triplet(members) if len(members) >= 3 else members
    # grids[0] is finest (smallest h); a refinement ratio is coarser/finer, so > 1
    r_list = [grids[i + 1].h / grids[i].h for i in range(len(grids) - 1)]
    for i, r in enumerate(r_list):
        lbl = "r21 (medium/fine)" if i == 0 else "r32 (coarse/medium)"
        flag = "" if r >= 1.3 else "  <-- below the recommended 1.3 minimum"
        out_lines.append(f"  {lbl} = {r:.4f}{flag}")

    if len(grids) >= 3:
        g1, g2, g3 = grids[0], grids[1], grids[2]
        r21, r32 = g2.h / g1.h, g3.h / g2.h
        _NOTE = {
            "oscillatory": "OSCILLATORY (R<0): sign flip between eps21 and eps32 "
                           "-- Richardson/GCI not applicable",
            "diverging":   "DIVERGING (R>1): grids move APART under refinement "
                           "-- no observed order exists to report",
            "degenerate":  "DEGENERATE (eps32 == 0): grids 2 and 3 identical",
        }
        for _label, _qois in (
                ("EXIT-NODE QoIs (x2 is a node on every grid; no interpolation)",
                 QOIS_EXIT),
                (f"POINT QoIs interpolated at frac={X_FRAC_DEFAULT:g}",
                 QOIS_POINT)):
            out_lines.append("")
            out_lines.append(f"  -- {_label} --")
            out_lines.append(f"  {'QoI':11s} {'phi1(fine)':>14s} {'phi2(med)':>14s} "
                             f"{'phi3(coarse)':>14s} {'R':>7s} {'p_obs':>7s} "
                             f"{'GCI_fine%':>10s} {'GCI_coarse%':>12s} "
                             f"{'phi_extrap':>14s} {'asym.chk':>9s}")
            for qoi in _qois:
                phi1, phi2, phi3 = g1.qois[qoi], g2.qois[qoi], g3.qois[qoi]
                res = gci_3grid(phi1, phi2, phi3, r21, r32)
                _R = res.get("R")
                _rtxt = f"{_R:7.3f}" if _R is not None and _R == _R else "    n/a"
                if not res["ok"]:
                    note = _NOTE.get(res.get("conv"),
                                     "could not resolve an observed order p")
                    out_lines.append(f"  {qoi:11s} {phi1:14.6g} {phi2:14.6g} "
                                     f"{phi3:14.6g} {_rtxt}   -- {note}")
                    continue
                out_lines.append(
                    f"  {qoi:11s} {phi1:14.6g} {phi2:14.6g} {phi3:14.6g} "
                    f"{_rtxt} {res['p']:7.3f} {res['gci_fine']*100:10.4f} "
                    f"{res['gci_coarse']*100:12.4f} {res['phi_ext']:14.6g} "
                    f"{res['asymptotic_check']:9.3f}"
                    + ("   <- p_obs outside [%.1f,%.1f]; GCI uses formal p=%.0f"
                       % (P_MIN, P_MAX, FORMAL_ORDER) if res["p_clamped"] else "")
                )
    else:
        g1, g2 = grids[0], grids[1]
        r21 = g2.h / g1.h
        out_lines.append("\n  Only 2 grids: observed order CANNOT be computed. "
                          "Reporting GCI under an ASSUMED order (Fs=3), bracketed "
                          "between 1st and 2nd order -- run a 3rd grid to verify.")
        out_lines.append(f"  {'QoI':10s} {'phi1(fine)':>14s} {'phi2(coarse)':>14s} "
                          f"{'rel.diff%':>10s} {'GCI(p=1)%':>11s} {'GCI(p=2)%':>11s}")
        for qoi in QOIS:
            phi1, phi2 = g1.qois[qoi], g2.qois[qoi]
            rel = abs((phi2 - phi1) / phi1) * 100 if phi1 else float("nan")
            g_p1 = gci_2grid_assumed(phi1, phi2, r21, 1.0)
            g_p2 = gci_2grid_assumed(phi1, phi2, r21, 2.0)
            gp1 = g_p1["gci"] * 100 if g_p1["ok"] else float("nan")
            gp2 = g_p2["gci"] * 100 if g_p2["ok"] else float("nan")
            out_lines.append(f"  {qoi:10s} {phi1:14.6g} {phi2:14.6g} {rel:10.4f} "
                              f"{gp1:11.4f} {gp2:11.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="*", default=DEFAULT_ROOTS,
                     help="directories to scan for run folders")
    ap.add_argument("--extra-runs", nargs="*", default=DEFAULT_EXTRA_RUNS,
                     help="individual published run folders to fold in if their "
                          "settings fingerprint matches a group")
    ap.add_argument("--frac", type=float, default=X_FRAC_DEFAULT,
                     help="QoIs are interpolated at x1 + frac*(x2-x1)")
    ap.add_argument("--report", default=None,
                     help="also write the report to this markdown file")
    args = ap.parse_args()

    run_dirs = []
    for root in args.root:
        run_dirs.extend(find_run_dirs(root))
    for extra in args.extra_runs:
        if os.path.isdir(extra) and extra not in run_dirs:
            run_dirs.append(extra)

    runs = []
    for d in run_dirs:
        try:
            r = build_run(d, args.frac)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"  [skip] {d}: {exc}")
            continue
        if r is not None:
            runs.append(r)

    if not runs:
        print("No runs found (looked in:", args.root, ")")
        sys.exit(1)

    groups = group_runs(runs)

    out_lines = [
        f"GCI report -- {len(runs)} run(s) discovered, "
        f"{len(groups)} distinct settings group(s)",
        f"EXIT-NODE QoIs ({', '.join(QOIS_EXIT)}) are read at x2, which is an "
        f"exact node on every grid -- no interpolation, any refinement ratio.",
        f"POINT QoIs ({', '.join(QOIS_POINT)}) are interpolated at "
        f"x1 + {args.frac:g}*(x2-x1), excluding the first/last row.",
        f"Every triplet is gated on Celik R = eps21/eps32: GCI is reported ONLY "
        f"for 0 < R < 1 (monotonic convergence).",
    ]
    for key, members in groups.items():
        if len(members) >= 3:
            continue  # printed after the incomplete ones, see below
    # print complete (>=3 grid) groups first, then incomplete ones
    complete = {k: v for k, v in groups.items() if len(v) >= 3}
    incomplete = {k: v for k, v in groups.items() if len(v) < 3}
    for key, members in complete.items():
        report_group(key, members, out_lines)
    if incomplete:
        out_lines.append("\n\n" + "#" * 88)
        out_lines.append("# Groups with FEWER than 3 grids (no verified observed order)")
        out_lines.append("#" * 88)
        for key, members in incomplete.items():
            report_group(key, members, out_lines)

    text = "\n".join(out_lines)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
