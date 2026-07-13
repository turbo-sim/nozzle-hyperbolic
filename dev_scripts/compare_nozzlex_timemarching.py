"""
Overlay the nozzlex space-marching reference (pickle) against one or more
time-marching solutions (solution.csv written by main_two_fluid_single_phase.py).

Serves two validations at once:
  * PHYSICS   — does the time-march relax to the independent nozzlex steady state?
                (Wallis = viscous forces modelled, virtual mass NOT → expect a
                 minor offset from our k_vm term, but shapes should match.)
  * ACCURACY  — overlay several eps_void runs: if the SMOOTH profiles coincide and
                only the 2Δx noise differs, the 4th-order filter is ε-independent
                (physics-preserving).  If a profile MOVES with eps_void → over-smearing.

Usage
-----
Edit PICKLE below (or pass as argv[1]).  By default it AUTO-DISCOVERS every
results/*/solution.csv, reads that run's copied YAML for (eps_void, Nx, recon)
to label it, and overlays them.  Set CSVS manually to pick specific runs.

    poetry run python dev_scripts/compare_nozzlex_timemarching.py
"""
import sys
import pathlib
import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).parent
_RESULTS = _HERE / "results"

# ── Config ────────────────────────────────────────────────────────────────────
PICKLE = _RESULTS / "solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+13_nd1.00e+11.pkl"
if len(sys.argv) > 1:
    PICKLE = pathlib.Path(sys.argv[1])

# Manual list: [(label, csv_path), ...].  Leave empty [] to AUTO-DISCOVER.
CSVS: list = []
ONLY_NX = None        # e.g. 600 to only compare 600-node runs; None = all
MAX_RUNS = 8          # cap on auto-discovered runs (most recent first)


# ── nozzlex pickle → mapped reference profile ─────────────────────────────────
def load_pickle_profile(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    passes = [data[k] for k in ("pass1", "pass2", "pass3")
              if isinstance(data, dict) and data.get(k) is not None]
    if not passes:
        raise ValueError(f"no non-empty passes in {path}")

    def cat(key):
        return np.concatenate([np.asarray(p[key], float).ravel() for p in passes])

    x = cat("x")
    rho1, rho2 = cat("rho1"), cat("rho2")
    phase1_liquid = np.nanmean(rho1) >= np.nanmean(rho2)   # idx-1 = denser = liquid?
    # map to (gas, liquid)
    if phase1_liquid:      # fluid1=liquid, fluid2=gas  (Elliot two-component pickle)
        g = lambda a, b: cat(b)   # gas  = idx-2
        l = lambda a, b: cat(a)   # liq  = idx-1
    else:
        g = lambda a, b: cat(a)
        l = lambda a, b: cat(b)
    prof = dict(
        x=x, p=cat("p"),
        u_g=g("v1", "v2"),   u_l=l("v1", "v2"),
        alpha_g=g("alpha1", "alpha2"),
        h_g=g("h1", "h2"),   h_l=l("h1", "h2"),
        T_g=g("T1", "T2"),   T_l=l("T1", "T2"),
    )
    order = np.argsort(prof["x"])
    for k in prof:
        prof[k] = prof[k][order]
    return prof


# ── time-march CSV loader ─────────────────────────────────────────────────────
def load_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {n: np.asarray(arr[n], float) for n in arr.dtype.names}


def normalize_csv(path):
    """Return a common schema {x,p,alpha_g,[u_g,u_l,T_g,T_l]}.  Handles both the
    new full two-phase CSV and the old dominant-only CSV (u_dom/T_dom).  This
    two-component case is LIQUID-dominant, so u_dom→u_l, T_dom→T_l."""
    d = load_csv(path)
    out = {"x": d["x"], "p": d["p"], "alpha_g": d["alpha_g"]}
    if "u_g" in d and "u_l" in d:                     # new full two-phase
        out.update(u_g=d["u_g"], u_l=d["u_l"], T_g=d["T_g"], T_l=d["T_l"])
    elif "u_dom" in d:                                # old dominant-only (=liquid here)
        out.update(u_l=d["u_dom"], T_l=d["T_dom"])
    return out


def discover_runs():
    runs = []
    for csv in sorted(_RESULTS.glob("*/solution.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        ymls = list(csv.parent.glob("*.yaml"))
        eps, nx, recon = None, None, None
        meta = {}                       # BC targets for plotting (p_out, p0, inlet/outlet type)
        if ymls:
            try:
                y = yaml.safe_load(open(ymls[0], encoding="utf-8"))
                eps   = float(y["numerics"].get("eps_void", 0.0))
                nx    = int(y["grid"]["Nx"])
                recon = str(y["numerics"].get("recon", "?"))
                bc = y.get("bc", {}) or {}
                meta = {
                    "p_out":  (float(bc["p_out"]) if "p_out" in bc else None),
                    "p0":     (float(bc["p0"])    if "p0"    in bc else None),
                    "inlet":  str(bc.get("inlet",  "?")),
                    "outlet": str(bc.get("outlet", "?")),
                }
            except Exception:
                pass
        if ONLY_NX is not None and nx != ONLY_NX:
            continue
        label = f"eps={eps}  N={nx}  {recon}" if eps is not None else csv.parent.name
        runs.append((label, csv, meta))
        if len(runs) >= MAX_RUNS:
            break
    return runs


# ── L2 relative difference of a CSV field vs the pickle reference ─────────────
def l2_rel(x_csv, f_csv, x_ref, f_ref):
    f_ref_i = np.interp(x_csv, x_ref, f_ref)
    m = (x_csv >= x_ref.min()) & (x_csv <= x_ref.max())
    num = np.sqrt(np.mean((f_csv[m] - f_ref_i[m]) ** 2))
    den = np.sqrt(np.mean(f_ref_i[m] ** 2)) + 1e-300
    return num / den


def main():
    ref = load_pickle_profile(PICKLE)
    runs = CSVS if CSVS else discover_runs()
    if not runs:
        print("No time-march solution.csv found under results/.  Run the solver first.")
        return
    print(f"Reference pickle: {PICKLE.name}   x:{ref['x'].min():.3f}..{ref['x'].max():.3f} m")
    print(f"Overlaying {len(runs)} time-march run(s).\n")

    # Panels: p, (u_g,u_l), alpha_g, (T_g,T_l)
    fields = [("p", "Pressure [bar]", 1e5),
              ("vel", "Velocity [m/s]", 1.0),
              ("alpha_g", r"$\alpha_g$", 1.0),
              ("Tvel", "Temperature [K]", 1.0)]
    fig, axs = plt.subplots(len(fields), 1, figsize=(6, 10), sharex=True)

    # reference (black)
    axs[0].plot(ref["x"], ref["p"] / 1e5, "k-", lw=1.6, label="nozzlex (ref)")
    axs[1].plot(ref["x"], ref["u_g"], "k-", lw=1.6, label="nozzlex gas")
    axs[1].plot(ref["x"], ref["u_l"], "k--", lw=1.6, label="nozzlex liq")
    axs[2].plot(ref["x"], ref["alpha_g"], "k-", lw=1.6, label="nozzlex")
    axs[3].plot(ref["x"], ref["T_g"], "k-", lw=1.6, label="nozzlex gas")
    axs[3].plot(ref["x"], ref["T_l"], "k--", lw=1.6, label="nozzlex liq")

    # Normalize to (label, csv, meta) — manual CSVS entries carry no BC meta.
    runs = [(r + ({},))[:3] if len(r) < 3 else r for r in runs]

    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(runs)))
    print(f"{'run':<26}  {'p':>8}  {'u_g':>8}  {'u_l':>8}  {'alpha':>8}  {'T_l':>8}   (L2 rel vs ref)")
    def _e(xg, f, key):
        return l2_rel(xg, f, ref["x"], ref[key]) if f is not None else None
    def _fmt(v):
        return f"{v:8.2%}" if v is not None else f"{'n/a':>8}"
    _bc_seen = {}   # (kind,value)→True so each BC line/label is drawn once
    for (label, csv, meta), c in zip(runs, colors):
        d  = normalize_csv(csv)
        xg = d["x"]
        axs[0].plot(xg, d["p"] / 1e5, "-", color=c, lw=0.9, ms=2, marker="o", label=label)
        axs[2].plot(xg, d["alpha_g"], "-", color=c, lw=0.9, ms=2, marker="o")
        if "u_g" in d:
            axs[1].plot(xg, d["u_g"], "-",  color=c, lw=0.9)
            axs[3].plot(xg, d["T_g"], "-",  color=c, lw=0.9)
        if "u_l" in d:
            axs[1].plot(xg, d["u_l"], "--", color=c, lw=0.9)
            axs[3].plot(xg, d["T_l"], "--", color=c, lw=0.9)
        e_p  = _e(xg, d["p"],               "p")
        e_ug = _e(xg, d.get("u_g"),         "u_g")
        e_ul = _e(xg, d.get("u_l"),         "u_l")
        e_a  = _e(xg, d["alpha_g"],         "alpha_g")
        e_Tl = _e(xg, d.get("T_l"),         "T_l")
        print(f"{label:<26}  {_fmt(e_p)}  {_fmt(e_ug)}  {_fmt(e_ul)}  {_fmt(e_a)}  {_fmt(e_Tl)}")
        # ── BC target markers on the pressure panel — did we REACH the BC? ──
        # p_out: the outlet back-pressure target, drawn as a ✕ at the outlet x and a
        #        faint dashed line.  If the last CSV point sits on the ✕ → outlet BC
        #        satisfied.  If it floats above/below → outlet is NOT following p_out
        #        (e.g. characteristic acting transmissive, or a supersonic outlet
        #        legitimately ignoring back-pressure).
        # p0:    the reservoir STAGNATION pressure (a ceiling; the static inlet p is
        #        below it by the dynamic head) — drawn as a faint dotted line so you
        #        can see how much of p0 the inlet has expanded.
        p_out = meta.get("p_out"); p0 = meta.get("p0")
        if p_out is not None and ("p_out", round(p_out)) not in _bc_seen:
            _bc_seen[("p_out", round(p_out))] = True
            axs[0].axhline(p_out / 1e5, color="crimson", ls="--", lw=0.8, alpha=0.6,
                           label=f"p_out BC ({p_out/1e5:.3g} bar, {meta.get('outlet','?')})")
            axs[0].plot(xg[-1], p_out / 1e5, "x", color="crimson", ms=9, mew=2)
        if p0 is not None and ("p0", round(p0)) not in _bc_seen:
            _bc_seen[("p0", round(p0))] = True
            axs[0].axhline(p0 / 1e5, color="navy", ls=":", lw=0.8, alpha=0.5,
                           label=f"p0 reservoir ({p0/1e5:.3g} bar, {meta.get('inlet','?')})")

    for ax, (_, ylab, _s) in zip(axs, fields):
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].legend(fontsize=7, ncol=2)
    axs[-1].set_xlabel("x [m]")
    fig.suptitle("nozzlex reference vs time-marching  (overlay = eps-independence + physics check)")
    fig.tight_layout()
    out = _HERE / "comparison_nozzlex_vs_timemarching.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
