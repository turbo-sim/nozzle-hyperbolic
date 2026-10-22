"""
Interfacial area of several runs: along the nozzle, and as a function of alpha.
=================================================================================

Each run's area law is rebuilt from the run's own resolved settings
(sources.interfacial_area / interfacial_area_params / bubble_number / droplet_number,
called the way sources.py calls it) and evaluated on the run's converged alpha_g(x).
The FIRST run is the reference (e.g. the parabola "truth"); every run keeps the same
colour in both figures.

    python plot_interfacial_area_compare.py                    # the three default runs below
    python plot_interfacial_area_compare.py REF RUN2 [RUN3 ...] --labels "truth" "case 2" ...
    python plot_interfacial_area_compare.py ... --out cases_x.png --out-alpha cases_alpha.png --show

Figure 1 (--out), along x: a_i; a_i / a_i(reference) on the reference grid; alpha_g; p
(with a zoom on the two-phase part).
Figure 2 (--out-alpha): a_i(alpha) of each law over 0..1, with the control points of
'segments' laws and the alpha range the runs do not reach shaded.
Note: interfacial_area_params: null means the law's CURRENT code defaults - if they
changed after the run, the plot no longer shows what the run used.
"""

import argparse
import glob
import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from nozzle_hyperbolic.fvm_solver.source_two_fluid import get_interfacial_area  # noqa: E402

DEFAULT_RUNS = [
    ("results/rsA_parabola_S3_v1e-5_N400_*", "parabola (truth)"),
    # ("results/rsC_sgG_S3_v1e-5_N400_*", "segments, original points"),
    # ("results/cold_S4_v1e-5_N400_muscl_van_albada_20261021_193548", "polynomial-cold"),
    ("results/restart_S3_v1e-5_N400_muscl_van_albada_20261021_201402", "polynomial-restart"),
    # ("results/rsB500_S3_v1e-5_N500_*", "segments"),
]

# Reference palette (dataviz skill): categorical slots in fixed order, surface and ink tokens.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def resolve_run(spec):
    hits = sorted(glob.glob(str(HERE / spec)))
    if not hits:
        raise SystemExit(f"no run folder matches {spec}")
    return pathlib.Path(hits[-1])


def load_run(run_dir, label=None):
    """Solution, area law (callable on alpha arrays) and metadata of one run folder."""
    with open(run_dir / "run_info.yaml", encoding="utf-8") as f:
        name = (yaml.safe_load(f).get("settings") or {}).get("resolved_file")
    with open(run_dir / name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src = cfg["sources"]
    law_name = src["interfacial_area"]
    params = src.get("interfacial_area_params") or {}
    law = get_interfacial_area(law_name, params)
    Nb, Nd = float(src["bubble_number"]), float(src["droplet_number"])
    ev = jax.jit(jax.vmap(lambda a: law(a, alpha_b=0.3, alpha_d=0.7, Nb=Nb, Nd=Nd, d_min_b=5e-6)))

    sol = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    x, ag = np.asarray(sol["x"]), np.asarray(sol["alpha_g"])
    ai, d = (np.asarray(v) for v in ev(jnp.asarray(ag)))
    mdot = ((sol["alpha_g"] * sol["rho_g"] * sol["u_g"]
             + (1 - sol["alpha_g"]) * sol["rho_l"] * sol["u_l"]) * sol["A"])[1:-1].mean()
    pts = None
    if law_name == "segments":
        from nozzle_hyperbolic.fvm_solver.source_two_fluid import interfacial_area_segments
        import inspect
        dflt = inspect.signature(interfacial_area_segments).parameters
        pts = (np.asarray(params.get("alpha_pts", dflt["alpha_pts"].default), float),
               np.asarray(params.get("ai_pts", dflt["ai_pts"].default), float))
    return dict(x=x, ag=ag, ai=ai, d=d, p=np.asarray(sol["p"]) / 1e5, mdot=mdot, ev=ev,
                pts=pts, law=law_name, label=label or law_name, dir=run_dir,
                nx=int(cfg["grid"]["Nx"]))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", help="run folders (glob ok); the first is the reference")
    ap.add_argument("--labels", nargs="*", help="one label per run (default: law names)")
    ap.add_argument("--out", default="interfacial_area_cases_x.png",
                    help="figure along x (PNG, relative to this folder)")
    ap.add_argument("--out-alpha", default="interfacial_area_cases_alpha.png",
                    help="figure a_i(alpha) (PNG, relative to this folder)")
    ap.add_argument("--show", action="store_true", help="also open the figure windows")
    cli = ap.parse_args()
    if not cli.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cli.runs:
        specs = cli.runs
        labels = cli.labels or [None] * len(specs)
    else:
        specs = [s for s, _ in DEFAULT_RUNS]
        labels = cli.labels or [lab for _, lab in DEFAULT_RUNS]
    if len(specs) < 2 or len(labels) != len(specs) or len(specs) > len(SERIES):
        raise SystemExit(f"give 2..{len(SERIES)} runs and as many labels")
    runs = [load_run(resolve_run(s), lab) for s, lab in zip(specs, labels)]
    ref = runs[0]
    for k, r in enumerate(runs):
        r["color"] = SERIES[k]

    # ── Figure 1: along the nozzle ─────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0, 1.0, 1.3]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)

    ax = axes[0]
    for r in runs:
        ax.plot(r["x"], r["ai"], color=r["color"], linewidth=1.6, label=r["label"])
    ax.set_ylabel(r"$a_i$  [1/m]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("Interfacial area along the nozzle", loc="left", fontsize=11, color=INK)

    ax = axes[1]
    ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    for r in runs[1:]:
        ratio = np.interp(ref["x"], r["x"], r["ai"]) / ref["ai"]
        ax.plot(ref["x"], ratio, color=r["color"], linewidth=1.6)
    ax.set_ylabel(f"$a_i$ / $a_i$ of\n{ref['label']}")

    ax = axes[2]
    for r in runs:
        ax.plot(r["x"], r["ag"], color=r["color"], linewidth=1.6)
    ax.set_ylabel(r"$\alpha_g$  [-]")

    ax = axes[3]
    for r in runs:
        ax.plot(r["x"], r["p"], color=r["color"], linewidth=1.6)
    ax.set_ylabel(r"$p$  [bar]")
    ax.set_xlabel("x  [m]")
    ins = ax.inset_axes([0.53, 0.40, 0.45, 0.55])
    style_axes(ins)
    x0 = 0.5 * (ref["x"][0] + ref["x"][-1])
    pz = []
    for r in runs:
        m = r["x"] >= x0
        ins.plot(r["x"][m], r["p"][m], color=r["color"], linewidth=1.4)
        pz.append(r["p"][m])
    pz = np.concatenate(pz)
    ins.set_xlim(x0, ref["x"][-1])
    ins.set_ylim(pz.min() - 0.1 * np.ptp(pz), pz.max() + 0.1 * np.ptp(pz))
    ins.tick_params(labelsize=8)
    ins.set_title("x > %.1f m" % x0, fontsize=8.5, color=INK_2, loc="left")

    fig.tight_layout()
    out_x = HERE / cli.out
    fig.savefig(out_x, dpi=150, facecolor=SURFACE)

    # ── Figure 2: a_i(alpha) of the laws ───────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8.6, 5.0))
    fig2.patch.set_facecolor(SURFACE)
    style_axes(ax)
    a_grid = np.linspace(0.0, 1.0, 2001)
    a_reach = max(r["ag"].max() for r in runs)
    ax.axvspan(a_reach, 1.0, color=GRID, alpha=0.6, linewidth=0)
    for r in runs:
        ai_g = np.asarray(r["ev"](jnp.asarray(a_grid))[0])
        ax.plot(a_grid, ai_g, color=r["color"], linewidth=1.8, label=r["label"])
        if r["pts"] is not None:
            ax.plot(r["pts"][0], r["pts"][1], linestyle="none", marker="o", markersize=5.5,
                    markerfacecolor=r["color"], markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.text(a_reach + 0.01, ax.get_ylim()[1] * 0.97,
            "not reached in\nthe nozzle\n" + r"($\alpha_g$ > " + f"{a_reach:.2f})",
            fontsize=8.5, color=INK_2, va="top")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\alpha_g$  [-]")
    ax.set_ylabel(r"$a_i$  [1/m]")
    ax.set_title("Interfacial area laws", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower center")
    fig2.tight_layout()
    out_a = HERE / cli.out_alpha
    fig2.savefig(out_a, dpi=150, facecolor=SURFACE)

    # ── Console summary ────────────────────────────────────────────────────
    for k, r in enumerate(runs):
        tag = "reference" if k == 0 else f"vs ref: max |dp| {np.max(np.abs(np.interp(ref['x'], r['x'], r['p']) - ref['p'])):.2f} bar"
        print(f"[{k}] {r['label']:28s} {r['law']:9s} N={r['nx']:<4d} mdot={r['mdot']:.3f} kg/s  {tag}\n"
              f"    {r['dir'].name}")
    print("\n   x     " + "".join(f"| {'alpha':>6s} {'a_i':>7s} {'p':>6s}  " for _ in runs))
    for xq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00):
        j = int(np.argmin(np.abs(ref["x"] - xq)))
        xx = ref["x"][j]
        row = "".join(f"| {np.interp(xx, r['x'], r['ag']):6.3f} {np.interp(xx, r['x'], r['ai']):7.0f} "
                      f"{np.interp(xx, r['x'], r['p']):6.2f}  " for r in runs)
        print(f"  {xx:.3f}  {row}")
    print(f"\nfigures -> {out_x}\n           {out_a}")
    if cli.show:
        plt.show()


if __name__ == "__main__":
    main()
