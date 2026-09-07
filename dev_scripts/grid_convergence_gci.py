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
]
GRID_KEYS = ["Nx", "x1", "x2"]

QOIS = ["p", "alpha_g", "mdot"]
X_FRAC_DEFAULT = 0.99          # interpolate QoIs at x1 + frac*(x2-x1)
FS_3GRID = 1.25                # Celik safety factor, observed order verified
FS_2GRID = 3.0                 # Roache safety factor, order ASSUMED (2 grids only)


# ── Settings / log parsing (regex — no yaml dependency needed) ──────────────
def _grab(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def parse_settings(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    out = {}
    for key in GRID_KEYS + FINGERPRINT_KEYS:
        raw = _grab(text, key)
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
    mdot_spread = float(np.ptp(mdot_cell) / np.mean(mdot_cell))  # conservation check

    return dict(p=float(p), alpha_g=float(alpha_g), mdot=float(mdot),
                mdot_spread=mdot_spread)


def build_run(path: str, frac: float) -> Optional[Run]:
    settings_files = glob.glob(os.path.join(path, "settings_*.yaml"))
    if not settings_files:
        return None
    settings_file = settings_files[0]
    s = parse_settings(settings_file)
    if s["Nx"] is None or s["x1"] is None or s["x2"] is None:
        return None

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
            if m.Nx in by_nx:
                print(f"  [note] duplicate Nx={m.Nx} in group; keeping "
                      f"{os.path.basename(m.path)} over "
                      f"{os.path.basename(by_nx[m.Nx].path)}")
            by_nx[m.Nx] = m
        deduped[key] = sorted(by_nx.values(), key=lambda r: -r.Nx)  # finest first
    return deduped


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
    out = dict(eps21=eps21, eps32=eps32, p=p, monotonic=(eps32 * eps21 > 0))
    if p is None or phi1 == 0:
        out["ok"] = False
        return out
    phi_ext = (r21 ** p * phi1 - phi2) / (r21 ** p - 1)
    gci_fine = FS_3GRID * abs(eps21 / phi1) / (r21 ** p - 1)
    gci_coarse = FS_3GRID * abs(eps32 / phi2) / (r32 ** p - 1)
    asym_check = gci_coarse / (r21 ** p * gci_fine) if gci_fine != 0 else float("nan")
    out.update(ok=True, phi_ext=phi_ext, gci_fine=gci_fine, gci_coarse=gci_coarse,
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
    return (f"    Nx={r.Nx:<5d} h={r.h:.5e}  {ssr_txt} {tol_txt:12s} "
            f"[{conv}]  {os.path.basename(r.path)}")


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

    not_converged = [m for m in members if m.converged is False]
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

    grids = members[:3] if len(members) >= 3 else members
    # grids[0] is finest (smallest h); a refinement ratio is coarser/finer, so > 1
    r_list = [grids[i + 1].h / grids[i].h for i in range(len(grids) - 1)]
    for i, r in enumerate(r_list):
        lbl = "r21 (medium/fine)" if i == 0 else "r32 (coarse/medium)"
        flag = "" if r >= 1.3 else "  <-- below the recommended 1.3 minimum"
        out_lines.append(f"  {lbl} = {r:.4f}{flag}")

    if len(grids) >= 3:
        g1, g2, g3 = grids[0], grids[1], grids[2]
        r21, r32 = g2.h / g1.h, g3.h / g2.h
        out_lines.append(f"\n  {'QoI':10s} {'phi1(fine)':>14s} {'phi2(med)':>14s} "
                          f"{'phi3(coarse)':>14s} {'p_obs':>7s} {'GCI_fine%':>10s} "
                          f"{'GCI_coarse%':>12s} {'phi_extrap':>14s} {'asym.chk':>9s}")
        for qoi in QOIS:
            phi1, phi2, phi3 = g1.qois[qoi], g2.qois[qoi], g3.qois[qoi]
            res = gci_3grid(phi1, phi2, phi3, r21, r32)
            if not res["ok"]:
                note = ("non-monotonic convergence (oscillatory) -- GCI formula "
                         "not applicable; consider Fs=3 two-grid bound instead"
                         if not res["monotonic"] else
                         "could not resolve an observed order p")
                out_lines.append(f"  {qoi:10s} {phi1:14.6g} {phi2:14.6g} "
                                  f"{phi3:14.6g}   -- {note}")
                continue
            out_lines.append(
                f"  {qoi:10s} {phi1:14.6g} {phi2:14.6g} {phi3:14.6g} "
                f"{res['p']:7.3f} {res['gci_fine']*100:10.4f} "
                f"{res['gci_coarse']*100:12.4f} {res['phi_ext']:14.6g} "
                f"{res['asymptotic_check']:9.3f}"
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
        f"QoIs interpolated at x1 + {args.frac:g}*(x2-x1); "
        f"first/last CSV row (ghost cells) excluded.",
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
