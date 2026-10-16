"""
Interfacial area along the nozzle, two runs in one figure.
==========================================================

Each run's area law is rebuilt from the run's own resolved settings
(sources.interfacial_area / interfacial_area_params / bubble_number / droplet_number,
called the way sources.py calls it) and evaluated on the run's converged alpha_g(x).

    python plot_interfacial_area_compare.py                       # parabola vs segments (defaults below)
    python plot_interfacial_area_compare.py RUN_A RUN_B           # any two run folders
    python plot_interfacial_area_compare.py RUN_A RUN_B --out ai.png --show

Panels: a_i(x) of both runs; the ratio a_i(B) / a_i(A) on A's grid; alpha_g(x) of both
(the a_i difference comes from both the law and the different void profile); p(x) of both.
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

DEFAULT_A = "results/c4_cold_S4_v1e-05_N400_*"         # parabola, N400, inlet void 1e-5
DEFAULT_B = "results/segments_S4_v1e-5_N400_*"         # segments, N400, inlet void 1e-5

# Reference palette (dataviz skill): categorical slots 1-2, surface and ink tokens.
SERIES = ("#2a78d6", "#eb6834")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def resolve_run(spec):
    hits = sorted(glob.glob(str(HERE / spec)))
    if not hits:
        raise SystemExit(f"no run folder matches {spec}")
    return pathlib.Path(hits[-1])


def load_run(run_dir):
    """x, alpha_g, a_i, d and a label for one run folder."""
    with open(run_dir / "run_info.yaml", encoding="utf-8") as f:
        name = (yaml.safe_load(f).get("settings") or {}).get("resolved_file")
    with open(run_dir / name, encoding="utf-8") as f:
        src = yaml.safe_load(f)["sources"]
    law_name = src["interfacial_area"]
    params = src.get("interfacial_area_params") or {}
    law = get_interfacial_area(law_name, params)
    Nb, Nd = float(src["bubble_number"]), float(src["droplet_number"])
    ev = jax.jit(jax.vmap(lambda a: law(a, alpha_b=0.3, alpha_d=0.7, Nb=Nb, Nd=Nd, d_min_b=5e-6)))

    sol = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    x, ag = np.asarray(sol["x"]), np.asarray(sol["alpha_g"])
    ai, d = (np.asarray(v) for v in ev(jnp.asarray(ag)))
    label = law_name + ("" if params else " (code defaults)")
    return dict(x=x, ag=ag, ai=ai, d=d, p=np.asarray(sol["p"]) / 1e5,
                label=label, law=law_name, dir=run_dir)


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
    ap.add_argument("run_a", nargs="?", default=DEFAULT_A, help="reference run folder (glob ok)")
    ap.add_argument("run_b", nargs="?", default=DEFAULT_B, help="compared run folder (glob ok)")
    ap.add_argument("--out", default="interfacial_area_compare.png", help="output PNG (relative to this folder)")
    ap.add_argument("--show", action="store_true", help="also open the figure window")
    cli = ap.parse_args()
    if not cli.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B = load_run(resolve_run(cli.run_a)), load_run(resolve_run(cli.run_b))
    ai_b_on_a = np.interp(A["x"], B["x"], B["ai"])
    ratio = ai_b_on_a / A["ai"]

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0, 1.0, 1.3]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)

    # 1 - interfacial area
    ax = axes[0]
    for run, c in zip((A, B), SERIES):
        ax.plot(run["x"], run["ai"], color=c, linewidth=1.6, label=run["label"])
        ax.annotate(run["law"], xy=(run["x"][-1], run["ai"][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_ylabel(r"$a_i$  [1/m]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("Interfacial area along the nozzle", loc="left", fontsize=11, color=INK)

    # 2 - ratio
    ax = axes[1]
    ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.plot(A["x"], ratio, color=INK, linewidth=1.6)
    i_max, i_min = int(np.argmax(ratio)), int(np.argmin(ratio))
    lo, hi = min(ratio.min(), 1.0), max(ratio.max(), 1.0)
    pad = 0.15 * (hi - lo) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    for i, dy in ((i_max, 6), (i_min, -14)):
        right = A["x"][i] > A["x"][0] + 0.75 * (A["x"][-1] - A["x"][0])
        ax.annotate(f"{ratio[i]:.2f} at x = {A['x'][i]:.2f}", xy=(A["x"][i], ratio[i]),
                    xytext=(-6 if right else 6, dy), textcoords="offset points",
                    ha="right" if right else "left", fontsize=8.5, color=INK_2)
    ax.set_ylabel(f"$a_i$ ratio\n{B['law']} / {A['law']}")

    # 3 - void fraction (context)
    ax = axes[2]
    for run, c in zip((A, B), SERIES):
        ax.plot(run["x"], run["ag"], color=c, linewidth=1.6, label=run["label"])
    ax.set_ylabel(r"$\alpha_g$  [-]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    # 4 - pressure
    ax = axes[3]
    for run, c in zip((A, B), SERIES):
        ax.plot(run["x"], run["p"], color=c, linewidth=1.6, label=run["label"])
    ax.set_ylabel(r"$p$  [bar]")
    ax.set_xlabel("x  [m]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    # zoom on the two-phase part, where the runs differ
    ins = ax.inset_axes([0.53, 0.40, 0.45, 0.55])
    style_axes(ins)
    x0 = 0.5 * (A["x"][0] + A["x"][-1])
    for run, c in zip((A, B), SERIES):
        m = run["x"] >= x0
        ins.plot(run["x"][m], run["p"][m], color=c, linewidth=1.4)
    pz = np.concatenate([run["p"][run["x"] >= x0] for run in (A, B)])
    ins.set_xlim(x0, A["x"][-1])
    ins.set_ylim(pz.min() - 0.1 * np.ptp(pz), pz.max() + 0.1 * np.ptp(pz))
    ins.tick_params(labelsize=8)
    ins.set_title("x > %.1f m" % x0, fontsize=8.5, color=INK_2, loc="left")

    fig.tight_layout()
    out = HERE / cli.out
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"A: {A['dir'].name}  ({A['label']})")
    print(f"B: {B['dir'].name}  ({B['label']})")
    print(f"ratio a_i(B)/a_i(A): max {ratio[i_max]:.2f} at x = {A['x'][i_max]:.3f}, "
          f"min {ratio[i_min]:.2f} at x = {A['x'][i_min]:.3f}")
    p_b_on_a = np.interp(A["x"], B["x"], B["p"])
    dp = p_b_on_a - A["p"]; k = int(np.argmax(np.abs(dp)))
    print(f"pressure p(B) - p(A): max |dp| {abs(dp[k]):.2f} bar ({100 * abs(dp[k]) / A['p'][k]:.1f} %) "
          f"at x = {A['x'][k]:.3f}")
    print("   x      alpha_A  alpha_B    a_i A     a_i B    ratio    p A [bar]  p B [bar]")
    for xq in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00):
        j = int(np.argmin(np.abs(A["x"] - xq)))
        print(f"  {A['x'][j]:.3f}  {A['ag'][j]:7.4f}  {np.interp(A['x'][j], B['x'], B['ag']):7.4f}  "
              f"{A['ai'][j]:8.0f}  {ai_b_on_a[j]:8.0f}  {ratio[j]:6.2f}   {A['p'][j]:8.2f}   {p_b_on_a[j]:8.2f}")
    print(f"figure -> {out}")
    if cli.show:
        plt.show()


if __name__ == "__main__":
    main()
