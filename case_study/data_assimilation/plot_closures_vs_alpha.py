"""
Interphase closures as functions of the void fraction alpha_g.
==============================================================

Drag force and interfacial heat transfer coefficients of the two-fluid closure
(source_two_fluid.py) over 0 < alpha_g < 1, for a fixed thermodynamic state:
saturated CO2 at --p, slip --slip.  Not a solution of any case - a check of the
correlations themselves and of the bubbly -> droplet transition blend.

    python plot_closures_vs_alpha.py
    python plot_closures_vs_alpha.py --p 30e5 --slip 20 --area polynomial --d-min 6.5e-5

Two columns:
  left   fixed diameter d = --d          -> isolates the regime blend
  right  d(alpha) from the area law      -> the closures as the solver sees them
Rows: drag force, its slope d/d(alpha), gas HTC, liquid HTC, and the regime
weight s(alpha) (left) / interfacial area and diameter (right).

Curves: drag with regime_blend linear / smoothstep; HTC with the default NICFD
closure (regime_htc: false - no transition, independent of the blend) and the
regime closure (regime_htc: true) with each blend.  Shaded: alpha_b..alpha_d.
A consistency report (jumps at alpha_b / alpha_d, sign changes, NaN, negative
values) is printed and saved next to the figure.
"""

import argparse
import pathlib

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

from nozzle_hyperbolic.fvm_solver.source_two_fluid import (
    calculate_drag_force_one_component,
    calculate_heat_transfer_one_component,
    ranz_marshall_htc_two_fluid,
    get_interfacial_area,
    regime_weight,
)

HERE = pathlib.Path(__file__).resolve().parent
C_LIN, C_SMOOTH, C_NICFD = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
BAND = "#dcdcd6"
NU_G, NU_L = 120.0, 12.0          # internal Nusselt numbers used by two_fluid_closure


def saturated_props(fluid, p):
    out = {}
    for name, q in (("g", 1), ("l", 0)):
        out[name] = {"rho": CP.PropsSI("D", "P", p, "Q", q, fluid),
                     "mu": CP.PropsSI("V", "P", p, "Q", q, fluid),
                     "k": CP.PropsSI("L", "P", p, "Q", q, fluid),
                     "cp": CP.PropsSI("C", "P", p, "Q", q, fluid)}
    return out


def closures(alpha, d, pr, slip, blend, alpha_b, alpha_d):
    """Drag force on the gas [N/m3] and the regime HTCs (gas, liquid) [W/m2K] at one alpha."""
    g, l = pr["g"], pr["l"]
    u_g, u_l = slip, 0.0
    F = calculate_drag_force_one_component(alpha, 1.0 - alpha, g["rho"], l["rho"], g["mu"], l["mu"],
                                           u_g, u_l, d, alpha_b=alpha_b, alpha_d=alpha_d, blend=blend)
    h_g, h_l = calculate_heat_transfer_one_component(alpha, g, l, u_g, u_l, d, Nu1_int=NU_G, Nu2_int=NU_L,
                                                     alpha_b=alpha_b, alpha_d=alpha_d, blend=blend)
    return F, h_g, h_l


def nicfd_htc(d, pr, slip):
    """Default closure (regime_htc false): k_g Nu_g / d and Ranz-Marshall, liquid continuous."""
    return pr["g"]["k"] * NU_G / d, ranz_marshall_htc_two_fluid(pr["l"], slip, d)


def check(name, a, y, dy, alpha_b, alpha_d, eps):
    """Report value / slope jumps at the regime limits and suspicious features of y(alpha)."""
    lines = [f"{name}:"]
    y = np.asarray(y); dy = np.asarray(dy)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(dy)):
        lines.append("   NON-FINITE values or slopes")
    if np.any(y < 0):
        lines.append(f"   NEGATIVE values for alpha in [{a[y < 0].min():.3g}, {a[y < 0].max():.3g}]")
    for ab in (alpha_b, alpha_d):
        i0, i1 = np.searchsorted(a, ab - eps), np.searchsorted(a, ab + eps)
        v0, v1, s0, s1 = y[i0 - 1], y[i1], dy[i0 - 1], dy[i1]
        lines.append(f"   at alpha={ab}: value {v0:.4g} -> {v1:.4g} (jump {abs(v1 - v0) / max(abs(v0), 1e-300):.1e}),"
                     f"  slope {s0:+.4g} -> {s1:+.4g}")
    flips = np.flatnonzero(np.diff(np.sign(dy)) != 0)
    if flips.size:
        lines.append("   slope changes sign at alpha = " + ", ".join(f"{a[i]:.3f}" for i in flips[:8])
                     + (" ..." if flips.size > 8 else ""))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fluid", default="CO2")
    ap.add_argument("--p", type=float, default=20e5, help="pressure [Pa] (saturated properties)")
    ap.add_argument("--slip", type=float, default=10.0, help="u_g - u_l [m/s]")
    ap.add_argument("--d", type=float, default=100e-6, help="fixed diameter of the left column [m]")
    ap.add_argument("--area", default="polynomial", help="area law of the right column")
    ap.add_argument("--d-min", type=float, default=6.5e-5, help="its d_min (polynomial / parabola)")
    ap.add_argument("--Nb", type=float, default=1e13, help="bubble number (area law 'original')")
    ap.add_argument("--Nd", type=float, default=1e10, help="droplet number (area law 'original')")
    ap.add_argument("--alpha-b", type=float, default=0.3)
    ap.add_argument("--alpha-d", type=float, default=0.7)
    ap.add_argument("--out", default=str(HERE / "results" / "closures_vs_alpha.png"))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    pr = saturated_props(a.fluid, a.p)
    alpha = np.linspace(1e-4, 1.0 - 1e-4, 4001)
    params = None if a.area == "original" else {"d_min": a.d_min}
    area_law = get_interfacial_area(a.area, params)
    ai_d = jax.vmap(lambda x: jnp.stack(area_law(x, alpha_b=a.alpha_b, alpha_d=a.alpha_d,
                                                 Nb=a.Nb, Nd=a.Nd, d_min_b=5e-6)))(jnp.asarray(alpha))
    ai, d_alpha = np.asarray(ai_d[:, 0]), np.asarray(ai_d[:, 1])

    def curves(d_of):
        """Values and alpha-slopes of drag / htc for both blends, d = d_of(alpha)."""
        res = {}
        for blend in ("linear", "smoothstep"):
            f = lambda x, i, b=blend: closures(x, d_of(x), pr, a.slip, b, a.alpha_b, a.alpha_d)[i]
            res[blend] = [(np.asarray(jax.vmap(lambda x, i=i: f(x, i))(alpha)),
                           np.asarray(jax.vmap(jax.grad(lambda x, i=i: f(x, i)))(alpha))) for i in range(3)]
        hg = jax.vmap(lambda x: nicfd_htc(d_of(x), pr, a.slip)[0])(alpha)
        hl = jax.vmap(lambda x: nicfd_htc(d_of(x), pr, a.slip)[1])(alpha)
        res["nicfd"] = [np.asarray(hg), np.asarray(hl)]
        return res

    d_fixed = lambda x: a.d
    d_law = lambda x: area_law(x, alpha_b=a.alpha_b, alpha_d=a.alpha_d, Nb=a.Nb, Nd=a.Nd, d_min_b=5e-6)[1]
    cols = [("fixed d = %.0f um" % (a.d * 1e6), curves(d_fixed)),
            (f"d(alpha) from area law '{a.area}'" + ("" if params is None else f", d_min {a.d_min * 1e6:g} um"),
             curves(d_law))]

    plt.rcParams.update({"axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "font.size": 9})
    fig, axs = plt.subplots(5, 2, figsize=(11, 15), sharex=True)
    for j, (title, r) in enumerate(cols):
        for ax in axs[:, j]:
            ax.axvspan(a.alpha_b, a.alpha_d, color=BAND, alpha=0.6, lw=0, zorder=0)
        axs[0, j].set_title(title, fontsize=10)
        # drag force and its slope
        for blend, c in (("linear", C_LIN), ("smoothstep", C_SMOOTH)):
            F, dF = r[blend][0]
            axs[0, j].plot(alpha, F, color=c, lw=1.6, label=f"drag, {blend}")
            axs[1, j].plot(alpha, dF, color=c, lw=1.6, label=blend)
        axs[0, j].set_yscale("log")
        axs[0, j].set_ylabel(f"drag force on gas [N/m3]\n(slip {a.slip:g} m/s)")
        axs[1, j].set_yscale("symlog", linthresh=max(1e-3 * np.nanmax(np.abs(r["linear"][0][1])), 1e-30))
        axs[1, j].set_ylabel("d(drag)/d(alpha)")
        # heat transfer coefficients
        for row, k, phase in ((2, 1, "gas"), (3, 2, "liquid")):
            axs[row, j].plot(alpha, r["nicfd"][k - 1], color=C_NICFD, lw=1.6,
                             label="default (regime_htc false)")
            for blend, c in (("linear", C_LIN), ("smoothstep", C_SMOOTH)):
                axs[row, j].plot(alpha, r[blend][k][0], color=c, lw=1.6, label=f"regime_htc, {blend}")
            axs[row, j].set_yscale("log")
            axs[row, j].set_ylabel(f"{phase} HTC [W/m2K]")
        axs[0, j].legend(fontsize=8, frameon=False)
        axs[2, j].legend(fontsize=8, frameon=False)
    # bottom row: regime weight (left), area law (right)
    for blend, c in (("linear", C_LIN), ("smoothstep", C_SMOOTH)):
        axs[4, 0].plot(alpha, np.asarray(regime_weight(jnp.asarray(alpha), a.alpha_b, a.alpha_d, blend)),
                       color=c, lw=1.6, label=blend)
    axs[4, 0].set_ylabel("droplet-regime weight s")
    axs[4, 0].legend(fontsize=8, frameon=False)
    axs[4, 1].plot(alpha, ai, color=C_LIN, lw=1.6, label="a_i [1/m]")
    axs[4, 1].set_yscale("log")
    axs[4, 1].set_ylabel("interfacial area a_i [1/m]")
    ax2 = axs[4, 1].twinx()                      # same row, labelled separately: d in um
    ax2.plot(alpha, d_alpha * 1e6, color=C_SMOOTH, lw=1.6, ls="--", label="d [um]")
    ax2.set_ylabel("diameter d [um] (dashed)")
    for ax in axs[-1]:
        ax.set_xlabel("void fraction alpha_g [-]")
    fig.suptitle(f"Closures vs alpha_g: saturated {a.fluid} at {a.p / 1e5:g} bar, slip {a.slip:g} m/s "
                 f"(shaded: {a.alpha_b} < alpha < {a.alpha_d})", fontsize=11)
    fig.tight_layout()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)

    eps = 2.0 * (alpha[1] - alpha[0])
    lines = [f"saturated {a.fluid} at {a.p / 1e5:g} bar: rho_g {pr['g']['rho']:.4g}, rho_l {pr['l']['rho']:.4g} kg/m3; "
             f"slip {a.slip:g} m/s", ""]
    for title, r in cols:
        lines.append(f"=== {title}")
        for blend in ("linear", "smoothstep"):
            for i, name in enumerate(("drag", "htc_g (regime)", "htc_l (regime)")):
                lines += check(f"  {name}, {blend}", alpha, r[blend][i][0], r[blend][i][1], a.alpha_b, a.alpha_d, eps)
        lines.append("")
    text = "\n".join(lines)
    print(text)
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"figure: {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
