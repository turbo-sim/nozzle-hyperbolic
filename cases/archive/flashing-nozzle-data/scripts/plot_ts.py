"""Regenerate every T-s figure from the case index.

For each case in ``index/cases_index.csv`` this draws the temperature-entropy diagram of the
expansion: the saturation dome, the inlet stagnation state, the equilibrium isentrope down to
the lowest pressure the case reports, the point where that isentrope crosses saturation
(flashing inception under equilibrium), and -- where the source provides one -- the measured
pressure taps mapped onto that isentrope.

The metastable liquid extension is drawn when CoolProp will evaluate it: it is the path a
subcooled flashing flow actually follows below the saturation pressure, and the gap between it
and the equilibrium isentrope is the modelling problem these data sets exist to constrain.

Nothing here is a model result. The only measured quantity on the figure is the pressure; the
temperatures come from an equilibrium property evaluation at the measured pressure.

Usage
-----
    python scripts/plot_ts.py
    python scripts/plot_ts.py --skip-placeholder
    python scripts/plot_ts.py --case abuaf1981__run_003
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import CoolProp.CoolProp as CP
from CoolProp import AbstractState, iphase_liquid

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index" / "cases_index.csv"
FIGURES = ROOT / "figures"

N_DOME = 300
N_ISENTROPE = 200

COLOR_DOME = "0.35"
COLOR_EQ = "#1f77b4"
COLOR_META = "#d62728"
COLOR_TAPS = "#2ca02c"


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def read_csv_with_comments(path):
    """Read a CSV, skipping the ``#`` comment lines that precede the header (see SCHEMA.md)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return list(csv.DictReader(lines[i:]))
    return []


def as_float(row, key):
    value = (row.get(key) or "").strip()
    return float(value) if value else None


def saturation_dome(fluid):
    """Liquid and vapour saturation lines, from the triple point to the critical point."""
    t_min = CP.PropsSI("Ttriple", fluid) + 0.1
    t_crit = CP.PropsSI("Tcrit", fluid)
    temperatures = [
        t_min + (t_crit - 0.01 - t_min) * i / (N_DOME - 1) for i in range(N_DOME)
    ]
    s_liq, s_vap, t_ok = [], [], []
    for temperature in temperatures:
        try:
            s_liq.append(CP.PropsSI("S", "T", temperature, "Q", 0, fluid))
            s_vap.append(CP.PropsSI("S", "T", temperature, "Q", 1, fluid))
            t_ok.append(temperature)
        except ValueError:
            continue
    return s_liq, s_vap, t_ok


def bisect(func, lo, hi, iterations=80):
    """Plain bisection, so that this file stays runnable with CoolProp and matplotlib alone."""
    f_lo, f_hi = func(lo), func(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def saturation_crossing(fluid, entropy, t_lo, t_hi):
    """Temperature at which the saturated-liquid entropy equals ``entropy``.

    Returns ``(T, p_sat)``, or ``None`` when the entropy is outside the bracket.
    """
    try:
        t_sat = bisect(
            lambda temperature: CP.PropsSI("S", "T", temperature, "Q", 0, fluid) - entropy,
            t_lo,
            t_hi,
        )
        if t_sat is None:
            return None
        return t_sat, CP.PropsSI("P", "T", t_sat, "Q", 0, fluid)
    except ValueError:
        return None


def equilibrium_isentrope(fluid, entropy, p_hi, p_lo):
    """States at constant entropy between two pressures, evaluated in equilibrium."""
    pressures = [
        p_hi * (p_lo / p_hi) ** (i / (N_ISENTROPE - 1)) for i in range(N_ISENTROPE)
    ]
    s_out, t_out = [], []
    for pressure in pressures:
        try:
            t_out.append(CP.PropsSI("T", "P", pressure, "S", entropy, fluid))
            s_out.append(entropy)
        except ValueError:
            continue
    return s_out, t_out


def metastable_liquid_isentrope(fluid, entropy, p_sat, p_lo, t_ref):
    """The subcooled-liquid branch continued below the saturation pressure.

    A pressure-entropy flash is no good here: CoolProp answers it with the *equilibrium*
    two-phase solve and hands back a mixture, ignoring ``specify_phase``. So the liquid root is
    pinned with a temperature-pressure update instead -- which does honour ``specify_phase`` and
    returns a genuine superheated-liquid density -- and the temperature carrying the inlet
    entropy is found by bisection at each pressure.

    The branch stops where CoolProp can no longer converge a liquid root, which is somewhere
    short of the true liquid spinodal.
    """
    try:
        state = AbstractState("HEOS", fluid)
    except ValueError:
        return [], []
    state.specify_phase(iphase_liquid)

    def liquid_entropy(pressure, temperature):
        state.update(CP.PT_INPUTS, pressure, temperature)
        return state.smass()

    pressures = [
        p_sat * (p_lo / p_sat) ** (i / (N_ISENTROPE - 1)) for i in range(N_ISENTROPE)
    ]
    s_out, t_out, p_out = [], [], []
    for pressure in pressures:
        try:
            temperature = bisect(
                lambda t: liquid_entropy(pressure, t) - entropy, t_ref - 40.0, t_ref + 40.0
            )
        except Exception:  # CoolProp raises assorted types once the liquid root is gone
            break
        if temperature is None:
            break
        t_out.append(temperature)
        s_out.append(entropy)
        p_out.append(pressure)
    return s_out, t_out, p_out


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------


def plot_case(row, out_path, full_dome=False):
    """Draw one T-s figure. Returns True on success, False when the case cannot be evaluated."""
    fluid = (row.get("fluid") or "").strip()
    p0 = as_float(row, "p0_Pa")
    t0 = as_float(row, "T0_K")
    if not fluid or p0 is None or t0 is None:
        print(f"  skip {row['uid']}: needs fluid, p0_Pa and T0_K")
        return False

    # the lowest pressure this case reaches, from whichever of the three the source reports
    candidates = [as_float(row, k) for k in ("p_exit_Pa", "p_back_Pa")]
    taps = []
    stem = (row.get("profile_p") or "").strip()
    if stem:
        profile_path = ROOT / "sources" / row["bibkey"] / "profiles" / f"{stem}.csv"
        if profile_path.exists():
            taps = [
                (as_float(r, "x_m"), as_float(r, "p_Pa"))
                for r in read_csv_with_comments(profile_path)
                if as_float(r, "p_Pa") is not None
            ]
            candidates += [p for _, p in taps]
    pressures = [p for p in candidates if p is not None and p > 0]
    p_lo = min(pressures) if pressures else 0.2 * p0

    try:
        s0 = CP.PropsSI("S", "P", p0, "T", t0, fluid)
    except ValueError as exc:
        print(f"  skip {row['uid']}: CoolProp cannot evaluate the inlet state ({exc})")
        return False

    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    s_liq, s_vap, t_dome = saturation_dome(fluid)
    ax.plot(s_liq, t_dome, color=COLOR_DOME, lw=1.2)
    ax.plot(s_vap, t_dome, color=COLOR_DOME, lw=1.2, label="saturation dome")

    s_eq, t_eq = equilibrium_isentrope(fluid, s0, p0, p_lo)
    ax.plot(s_eq, t_eq, color=COLOR_EQ, lw=2.0, label="equilibrium isentrope")

    s_meta, t_meta = [], []
    crossing = saturation_crossing(
        fluid, s0, CP.PropsSI("Ttriple", fluid) + 0.1, CP.PropsSI("Tcrit", fluid) - 0.01
    )
    if crossing is not None:
        t_flash, p_flash = crossing
        ax.plot(
            [s0], [t_flash], "o", ms=9, mfc="none", mew=2.0, color=COLOR_EQ, zorder=6,
            label=f"saturation crossing ({p_flash / 1e5:.2f} bar)",
        )
        s_meta, t_meta, p_meta = metastable_liquid_isentrope(fluid, s0, p_flash, p_lo, t0)
        if s_meta:
            # An isentropic metastable liquid barely changes temperature, so this branch is a
            # short segment rather than a curve. The endpoint marker is what carries the
            # information: how far the liquid is superheated once it reaches the exit pressure.
            ax.plot(
                s_meta, t_meta, color=COLOR_META, lw=2.5, ls="--", zorder=4,
                label="metastable liquid extension",
            )
            superheat = t_meta[-1] - CP.PropsSI("T", "P", p_meta[-1], "Q", 0, fluid)
            ax.plot(
                [s_meta[-1]], [t_meta[-1]], "D", ms=6, color=COLOR_META, zorder=6,
                label=(
                    f"metastable liquid at {p_meta[-1] / 1e5:.2f} bar "
                    f"({superheat:.1f} K superheat)"
                ),
            )

    if taps:
        s_taps, t_taps = [], []
        for _, pressure in sorted(taps, key=lambda item: -item[1]):
            try:
                t_taps.append(CP.PropsSI("T", "P", pressure, "S", s0, fluid))
                s_taps.append(s0)
            except ValueError:
                continue
        ax.plot(
            s_taps, t_taps, "s", ms=4, color=COLOR_TAPS,
            label="measured taps on the equilibrium isentrope",
        )

    ax.plot([s0], [t0], "*", ms=14, color="k", label="inlet stagnation state", zorder=5)

    subcooling = as_float(row, "subcooling_K")
    subtitle = f"{fluid} -- {p0 / 1e5:.2f} bar, {t0:.1f} K"
    if subcooling is not None:
        subtitle += f", {subcooling:.1f} K subcooled"
    ax.set_title(f"{row['uid']}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Entropy [J/kg/K]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")

    # Keep the expansion in view. On the full diagram a flashing expansion is a hairline at
    # constant entropy against a dome several thousand J/kg/K wide, so the default view is
    # cropped to the liquid side; --full-dome puts the vapour branch back.
    t_all = [t for t in t_eq + t_meta if t is not None] + [t0]
    span = max(max(t_all) - min(t_all), 5.0)
    ax.set_ylim(min(t_all) - 0.35 * span, max(t_all) + 0.35 * span)

    if full_dome:
        s_span = max(s_vap) - min(s_liq)
        ax.set_xlim(min(s_liq) - 0.02 * s_span, max(s_vap) + 0.02 * s_span)
    else:
        t_view_lo, t_view_hi = ax.get_ylim()
        s_view = [s0] + [s for s, t in zip(s_liq, t_dome) if t_view_lo <= t <= t_view_hi]
        pad = max(0.15 * (max(s_view) - min(s_view)), 50.0)
        ax.set_xlim(min(s_view) - pad, max(s_view) + pad)

    if (row.get("status") or "").strip() == "placeholder":
        ax.text(
            0.5, 0.5, "PLACEHOLDER DATA", transform=ax.transAxes, fontsize=26,
            color="red", alpha=0.18, ha="center", va="center", rotation=25, zorder=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate every T-s figure.")
    parser.add_argument("--case", action="append", help="uid to plot; repeatable, default all")
    parser.add_argument("--skip-placeholder", action="store_true", help="ignore placeholder rows")
    parser.add_argument("--full-dome", action="store_true", help="show the vapour branch too")
    parser.add_argument("--out", type=Path, default=FIGURES, help="output directory")
    args = parser.parse_args(argv)

    if not INDEX.exists():
        print(f"{INDEX} not found; running build_index.py first")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("build_index.py"))]
        )
        if result.returncode != 0 or not INDEX.exists():
            return 1

    rows = read_csv_with_comments(INDEX)
    if args.case:
        rows = [r for r in rows if r["uid"] in set(args.case)]
    if args.skip_placeholder:
        rows = [r for r in rows if (r.get("status") or "").strip() != "placeholder"]
    if not rows:
        print("nothing to plot")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        out_path = args.out / f"ts_{row['uid']}.png"
        if plot_case(row, out_path, full_dome=args.full_dome):
            print(f"  wrote {out_path}")
            written += 1

    print(f"\n{written}/{len(rows)} figures written to {args.out}")
    return 0 if written == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
