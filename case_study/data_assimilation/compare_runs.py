"""
Compare converged two-fluid solutions (e.g. regime_blend linear vs smoothstep).
==============================================================================

The first run is the REFERENCE; every other run is compared to it.

    python compare_runs.py <run A> <run B> [<run C> ...] [--labels A B ...]
                           [--out results/compare.png] [--alpha-b 0.3 --alpha-d 0.7]

    python compare_runs.py results/cmp_linear_S2_N200_* results/cmp_smooth_S2_N200_* \
                           --labels linear smoothstep

A run is a results folder holding solution.csv (columns x, A, p, alpha_g, u_g,
u_l, T_g, T_l, h_g, h_l, rho_g, rho_l).  Runs on different grids are compared on
the reference grid (linear interpolation in x).

Figure 1 <out>          the fields of all runs overlaid
Figure 2 <out>_diff     each run minus the reference
Both shade where the reference is in the bubbly->droplet transition
(alpha_b < alpha_g < alpha_d), where the drag / regime-HTC blend acts.
A summary table (mass flow, exit state, flashing onset, largest differences)
is printed and written next to the figures as <out>.txt.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent

# Categorical slots in fixed order (reference palette): run 1 = reference.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BAND = "#dcdcd6"          # transition band shading (recessive neutral)


def load(run):
    path = pathlib.Path(run)
    if not path.is_absolute():
        path = HERE / path
    csv = path / "solution.csv" if path.is_dir() else path
    d = pd.read_csv(csv)
    d["slip"] = d.u_g - d.u_l
    m_g = d.A * d.alpha_g * d.rho_g * d.u_g
    m_l = d.A * (1.0 - d.alpha_g) * d.rho_l * d.u_l
    d["m_dot"] = m_g + m_l
    d["quality"] = m_g / d.m_dot                       # vapour mass-flow fraction
    return d


def on_grid(d, x):
    """All columns of d interpolated onto x."""
    return pd.DataFrame({c: np.interp(x, d.x, d[c]) for c in d.columns})


def shade_band(ax, ref, alpha_b, alpha_d):
    inside = ((ref.alpha_g > alpha_b) & (ref.alpha_g < alpha_d)).to_numpy()
    if inside.any():
        ax.fill_between(ref.x, 0, 1, where=inside, color=BAND, alpha=0.6, lw=0,
                        transform=ax.get_xaxis_transform(), zorder=0)


def onset(d, level=0.01):
    """First x where alpha_g exceeds `level` (flashing onset marker)."""
    i = np.flatnonzero(d.alpha_g.to_numpy() > level)
    return float(d.x.iloc[i[0]]) if i.size else np.nan


# (column, axis label, scale) for the overlay; gas/liquid pairs share a panel
PANELS = [
    (("p",), "pressure [bar]", 1e-5),
    (("alpha_g",), r"void fraction $\alpha_g$ [-]", 1.0),
    (("u_g", "u_l"), "velocity [m/s]", 1.0),
    (("slip",), r"slip $u_g - u_l$ [m/s]", 1.0),
    (("T_g", "T_l"), "temperature [K]", 1.0),
    (("quality",), "vapour mass-flow fraction [-]", 1.0),
    (("h_g", "h_l"), "enthalpy [kJ/kg]", 1e-3),
    (("m_dot",), "mass flow rate [kg/s]", 1.0),
]
DIFF = [("p", "Δp [bar]", 1e-5), ("alpha_g", r"Δ$\alpha_g$ [-]", 1.0),
        ("u_g", r"Δ$u_g$ [m/s]", 1.0), ("u_l", r"Δ$u_l$ [m/s]", 1.0),
        ("T_g", r"Δ$T_g$ [K]", 1.0), ("T_l", r"Δ$T_l$ [K]", 1.0),
        ("quality", "Δ vapour mass-flow fraction [-]", 1.0), ("m_dot", "Δ mass flow [kg/s]", 1.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="results folders (or solution.csv files); first = reference")
    ap.add_argument("--labels", nargs="+", help="one label per run (default: folder names)")
    ap.add_argument("--out", help="figure path (default: results/compare_<labels>.png)")
    ap.add_argument("--alpha-b", type=float, default=0.3, help="start of the transition band")
    ap.add_argument("--alpha-d", type=float, default=0.7, help="end of the transition band")
    ap.add_argument("--show", action="store_true", help="also open the figures")
    a = ap.parse_args()

    if len(a.runs) < 2:
        raise SystemExit("give at least two runs (the first is the reference)")
    labels = a.labels or [pathlib.Path(r).name[:40] for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit("--labels needs one label per run")
    runs = [load(r) for r in a.runs]
    ref = runs[0]
    cmp = [on_grid(d, ref.x.to_numpy()) for d in runs[1:]]

    out = pathlib.Path(a.out) if a.out else HERE / "results" / f"compare_{'_vs_'.join(labels)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "font.size": 9})

    # ── Figure 1: overlay ─────────────────────────────────────────────────────
    fig, axs = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for ax, (cols, ylabel, scale) in zip(axs.flat, PANELS):
        shade_band(ax, ref, a.alpha_b, a.alpha_d)
        for k, (d, lab) in enumerate(zip(runs, labels)):
            for col, ls in zip(cols, ("-", "--")):
                tag = "" if len(cols) == 1 else (" gas" if col.endswith("_g") else " liquid")
                ax.plot(d.x, d[col] * scale, color=COLORS[k % len(COLORS)], ls=ls, lw=1.6,
                        label=f"{lab}{tag}")
        ax.set_ylabel(ylabel)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[1, 0].legend(fontsize=8, frameon=False, ncol=1)        # velocity: gas / liquid per run
    for ax in axs[-1]:
        ax.set_xlabel("x [m]")
    fig.suptitle(f"Solutions (shaded: reference in {a.alpha_b} < α_g < {a.alpha_d})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)

    # ── Figure 2: differences to the reference ─────────────────────────────────
    fig2, axs2 = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for ax, (col, ylabel, scale) in zip(axs2.flat, DIFF):
        shade_band(ax, ref, a.alpha_b, a.alpha_d)
        ax.axhline(0.0, color="#8a8a85", lw=0.8)
        for k, (d, lab) in enumerate(zip(cmp, labels[1:]), start=1):
            ax.plot(ref.x, (d[col] - ref[col]) * scale, color=COLORS[k % len(COLORS)], lw=1.6,
                    label=f"{lab} − {labels[0]}")
        ax.set_ylabel(ylabel)
    axs2[0, 0].legend(fontsize=8, frameon=False)
    for ax in axs2[-1]:
        ax.set_xlabel("x [m]")
    fig2.suptitle(f"Difference to the reference ({labels[0]})", fontsize=11)
    fig2.tight_layout()
    out_diff = out.with_name(out.stem + "_diff" + out.suffix)
    fig2.savefig(out_diff, dpi=180)

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = []
    for d, lab in zip(runs, labels):
        rows.append({"run": lab, "N": len(d), "m_dot [kg/s]": float(np.median(d.m_dot)),
                     "m_dot spread [%]": float(100 * (d.m_dot.max() - d.m_dot.min()) / np.median(d.m_dot)),
                     "x(alpha=1%) [m]": onset(d), "p_exit [bar]": float(d.p.iloc[-1] * 1e-5),
                     "alpha_exit": float(d.alpha_g.iloc[-1]), "quality_exit": float(d.quality.iloc[-1]),
                     "max |slip| [m/s]": float(d.slip.abs().max())})
    lines = [pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5g}"), ""]
    for d, lab in zip(cmp, labels[1:]):
        lines.append(f"largest differences, {lab} - {labels[0]}:")
        for col, name, scale in DIFF:
            diff = (d[col] - ref[col]) * scale
            i = int(np.argmax(np.abs(diff.to_numpy())))
            unit = name[name.rfind("["):]             # plain text: the console may be cp1252
            lines.append(f"  d {col:<9s} {unit:<9s} {diff.iloc[i]:+.4g}  at x = {ref.x.iloc[i]:.4f} m "
                         f"(alpha_ref = {ref.alpha_g.iloc[i]:.3f})")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"figures: {out}\n         {out_diff}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
