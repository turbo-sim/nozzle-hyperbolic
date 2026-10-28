"""
Two-fluid initial condition from the HEM inverse design.
========================================================

Writes design_hem/hem_seed.csv in the nozzlex CSV format read by the driver's
`ic.type: nozzlex_csv` (columns distance, pressure, velocity_gas,
velocity_liquid, void_fraction, enthalpy_gas, enthalpy_liquid).

Why: a cold start from rest has to pass through intermediate back-pressures,
and for this case every one of them chokes the throat as soon as the liquid
flashes and parks a flashing shock right behind it (try01-try05).  The HEM
design is already the supersonic, shock-free expansion to p_out = 30 bar, so
the solver only has to relax the non-equilibrium differences.

The state on the design pressure profile p(x), on the reservoir isentrope:
  * HEM mixture h(p, s0) (CoolProp equilibrium), u = sqrt(2 (H0 - h)), no slip;
  * two-phase (0 < Q < 1): liquid and vapour at saturation, alpha from Q;
  * single-phase: the liquid is the mixture, the ghost vapour sits at the pin
    target h_sat,g(min(p, 0.98 p_c)) - the solver's clamped saturation table -
    with alpha = bc.void_fraction.

    python make_hem_seed.py                       # case 3 -> design_hem/hem_seed.csv
    python make_hem_seed.py --smooth 10           # 10 passes of a 1-2-1 filter on
                                                  # p, u, alpha, h_l -> hem_seed_s10.csv
    python make_hem_seed.py --design design_case4 --T0 295.0 --smooth 10

--design is a folder holding the nozzlex nozzle_data.csv; the seed is written
there.  --p0/--T0 must be the reservoir the design was made for.

Smoothing: the HEM front is a one-cell jump alpha 1e-3 -> 0.14, which the
two-fluid model cannot start from (try08: the last ghost cell's h_g crashed
below the vapour table).  After smoothing, h_g is reset to saturation
h_sat,g(min(p, 0.98 p_c)) everywhere, i.e. the ghost-pin target.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

HERE = pathlib.Path(__file__).resolve().parent
FLUID = "CO2"
_ap = argparse.ArgumentParser(description="Two-fluid initial condition from the HEM inverse design")
_ap.add_argument("--design", default="design_hem", help="folder with nozzle_data.csv")
_ap.add_argument("--p0", type=float, default=120.0e5, help="reservoir pressure [Pa]")
_ap.add_argument("--T0", type=float, default=310.15, help="reservoir temperature [K]")
_ap.add_argument("--length", type=float, default=1.0, help="design length [m]")
_ap.add_argument("--alpha", type=float, default=1.0e-3, help="ghost void fraction")
_ap.add_argument("--smooth", type=int, default=0, help="1-2-1 filter passes")
_args = _ap.parse_args()

P0, T0 = _args.p0, _args.T0   # reservoir (bc.p0, bc.T_l0)
LENGTH = _args.length         # m, geometry_export.length of the design
ALPHA_GHOST = _args.alpha     # bc.void_fraction
SMOOTH_PASSES = _args.smooth
DESIGN = HERE / _args.design

design = pd.read_csv(DESIGN / "nozzle_data.csv")
x = design["x_norm"].to_numpy() * LENGTH
p = design["p_pa"].to_numpy()

H0 = CP.PropsSI("H", "P", P0, "T", T0, FLUID)
s0 = CP.PropsSI("S", "P", P0, "T", T0, FLUID)
p_sat_top = 0.98 * CP.PropsSI("pcrit", FLUID)


def sat(prop, p_, q):
    return CP.PropsSI(prop, "P", min(p_, p_sat_top), "Q", q, FLUID)


rows = []
for xi, pi in zip(x, p):
    h = CP.PropsSI("H", "P", pi, "S", s0, FLUID)
    u = np.sqrt(max(2.0 * (H0 - h), 0.0))
    q = CP.PropsSI("Q", "P", pi, "S", s0, FLUID) if pi < p_sat_top else -1.0
    if 0.0 < q < 1.0:
        h_l, h_g = sat("H", pi, 0), sat("H", pi, 1)
        rho_l, rho_g = sat("D", pi, 0), sat("D", pi, 1)
        alpha = (q / rho_g) / (q / rho_g + (1.0 - q) / rho_l)
        alpha = max(alpha, ALPHA_GHOST)
    else:
        h_l, h_g, alpha = h, sat("H", pi, 1), ALPHA_GHOST
    rows.append((xi, pi, u, u, alpha, h_g, h_l))

out = pd.DataFrame(rows, columns=["distance", "pressure", "velocity_gas",
                                  "velocity_liquid", "void_fraction",
                                  "enthalpy_gas", "enthalpy_liquid"])


def smooth_121(f, n):
    """n passes of a 1-2-1 filter, end values kept."""
    f = np.asarray(f, dtype=float).copy()
    for _ in range(n):
        f[1:-1] = 0.25 * f[:-2] + 0.5 * f[1:-1] + 0.25 * f[2:]
    return f


if SMOOTH_PASSES:
    for c in ("pressure", "velocity_gas", "velocity_liquid", "void_fraction",
              "enthalpy_liquid"):
        out[c] = smooth_121(out[c], SMOOTH_PASSES)
    out["void_fraction"] = np.maximum(out["void_fraction"], ALPHA_GHOST)
    out["enthalpy_gas"] = [sat("H", pi, 1) for pi in out["pressure"]]

name = f"hem_seed_s{SMOOTH_PASSES}.csv" if SMOOTH_PASSES else "hem_seed.csv"
path = DESIGN / name
out.to_csv(path, index=False)

print(f"HEM seed -> {path}  ({len(out)} points)")
print(f"  p     {out.pressure.max()/1e5:.2f} -> {out.pressure.min()/1e5:.2f} bar")
print(f"  u     {out.velocity_liquid.min():.1f} -> {out.velocity_liquid.max():.1f} m/s")
print(f"  alpha {out.void_fraction.min():.2e} -> {out.void_fraction.max():.3f}")
print(f"  h_l   {out.enthalpy_liquid.min()/1e3:.1f} -> {out.enthalpy_liquid.max()/1e3:.1f} kJ/kg")
print(f"  h_g   {out.enthalpy_gas.min()/1e3:.1f} -> {out.enthalpy_gas.max()/1e3:.1f} kJ/kg")
