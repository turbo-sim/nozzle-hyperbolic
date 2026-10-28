"""
Fill the missing THERMAL CONDUCTIVITY of R1233zd(E) in the generated tables.
===========================================================================

Why this exists
---------------
CoolProp has no thermal-conductivity model for R1233zd(E):

    PropsSI("CONDUCTIVITY", "T", 300, "Q", 0, "R1233ZDE")
    -> ValueError: Thermal conductivity model is not available for this fluid

(viscosity IS available; only conductivity is missing).  The table generator
catches that internally and stores NaN at every node, and the two-fluid closure
reads conductivity for BOTH interfacial heat-transfer coefficients:

    sources.py:242   k_g = stg_ex["conductivity"][0]      ->  htc_g = k_g*Nu/d
    sources.py:267   "k": stl_ex["k"][0]                  ->  ranz_marshall_htc

so an unpatched table makes the interfacial heat transfer - and therefore the
flashing mass transfer that this whole case is about - NaN on the first step.

The estimate
------------
Two-parameter corresponding states from **R245fa**, which is the standard
surrogate for R1233zd(E) and here an unusually close one:

                    T_c [K]    p_c [bar]   M [g/mol]
    R1233zd(E)      439.60      36.24       130.50
    R245fa          427.01      36.51       134.05
    difference       2.9 %       0.7 %        2.7 %

Each table node is mapped to R245fa at the SAME reduced temperature and reduced
density, and the result is corrected with the classical corresponding-states
conductivity reducing group

    Gamma = T_c^(1/6) * M^(1/2) * p_c^(-2/3)        (T_c [K], M [g/mol], p_c [atm])

via  k_R1233 = k_R245fa * Gamma_R245fa / Gamma_R1233 .  The group correction is
about 1 %, i.e. the estimate is dominated by the (very good) reduced-state match
rather than by the correction.

Accuracy and scope
------------------
This is an ESTIMATE, not a measurement.  Expect roughly 5-10 % on k, which
propagates to the interfacial HTC and hence to the flashing rate.  It does NOT
affect the EOS: density, temperature, enthalpy, entropy, speed of sound and
viscosity all come from the real R1233zd(E) EOS and are untouched here.  Only
`conductivity` is written.

Idempotent: re-running on an already-patched table simply recomputes the same
values.  Run automatically by build_tables.py; can also be run directly.
"""

import sys
import pickle
import pathlib

import numpy as np
import CoolProp.CoolProp as CP

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import case_config as cfg

from jaxprop.bicubic.bicubic_interpolation import compute_bicubic_coefficients

SURROGATE = "R245fa"


def _gamma(fluid):
    """Corresponding-states conductivity reducing group
    Gamma = Tc^(1/6) * M^(1/2) / pc^(2/3),  with Tc [K], M [g/mol], pc [atm]."""
    Tc = CP.PropsSI("Tcrit", fluid)
    pc = CP.PropsSI("pcrit", fluid) / 101325.0          # atm
    M  = CP.PropsSI("molar_mass", fluid) * 1e3          # g/mol
    return Tc ** (1.0 / 6.0) * M ** 0.5 / pc ** (2.0 / 3.0)


def conductivity_field(T, rho):
    """Corresponding-states conductivity of R1233zd(E) on the (T, rho) grid."""
    Tc_t, rhoc_t = CP.PropsSI("Tcrit", cfg.FLUID_NAME), CP.PropsSI("rhocrit", cfg.FLUID_NAME)
    Tc_s, rhoc_s = CP.PropsSI("Tcrit", SURROGATE),      CP.PropsSI("rhocrit", SURROGATE)
    scale = _gamma(SURROGATE) / _gamma(cfg.FLUID_NAME)

    # Match reduced temperature and reduced density, then clamp into the
    # surrogate's valid domain (only the frozen, unphysical corners clamp).
    T_s   = np.clip(T / Tc_t * Tc_s,
                    CP.PropsSI("Ttriple", SURROGATE) * 1.001, 1200.0)
    rho_s = np.clip(rho / rhoc_t * rhoc_s, 1e-6, 2.0e3)

    k = np.empty_like(T_s, dtype=float)
    flat_T, flat_r, flat_k = T_s.ravel(), rho_s.ravel(), k.ravel()
    n_fail = 0
    for i in range(flat_T.size):
        try:
            flat_k[i] = CP.PropsSI("CONDUCTIVITY", "T", float(flat_T[i]),
                                   "Dmass", float(flat_r[i]), SURROGATE)
        except Exception:
            flat_k[i] = np.nan
            n_fail += 1
    k = flat_k.reshape(T_s.shape) * scale

    # Repair any residual holes by nearest-valid fill along the enthalpy axis,
    # then along pressure - a hole only ever occurs deep in the frozen region.
    if not np.isfinite(k).all():
        for axis in (0, 1):
            k = np.apply_along_axis(_fill_1d, axis, k)
    return k, scale, n_fail


def _fill_1d(v):
    """Forward/backward nearest-valid fill of NaNs in a 1-D slice."""
    ok = np.isfinite(v)
    if not ok.any():
        return v
    idx = np.where(ok, np.arange(v.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = v[idx]
    # backward pass for leading NaNs
    ok2 = np.isfinite(out)
    if not ok2.all():
        idx2 = np.where(ok2, np.arange(out.size), out.size - 1)
        idx2 = np.minimum.accumulate(idx2[::-1])[::-1]
        out = out[idx2]
    return out


def patch(pkl_path: pathlib.Path) -> bool:
    with open(pkl_path, "rb") as f:
        table = pickle.load(f)

    meta = table["metadata"]
    h_vals = np.linspace(meta["h_min"], meta["h_max"], meta["N_h"])
    logP   = np.linspace(np.log(meta["p_min"]), np.log(meta["p_max"]), meta["N_p"])
    p_vals = np.exp(logP)
    delta_h, delta_logP = meta["delta_h"], meta["delta_logP"]

    T   = np.asarray(table["temperature"]["value"], dtype=float)
    rho = np.asarray(table["density"]["value"], dtype=float)

    was_nan = int(np.sum(~np.isfinite(np.asarray(table["conductivity"]["value"]))))
    k, scale, n_fail = conductivity_field(T, rho)

    if not np.isfinite(k).all():
        raise RuntimeError(f"conductivity still has "
                           f"{int(np.sum(~np.isfinite(k)))} non-finite nodes")

    # Gradients consistent with the table's storage convention:
    #   grad_h = df/dh,  grad_logP = df/dlogP,  grad_p = grad_logP / p,
    #   grad_hlogP = d2f/dh dlogP,  grad_ph = grad_hlogP / p
    grad_h     = np.gradient(k, h_vals, axis=0, edge_order=2)
    grad_logP  = np.gradient(k, logP,   axis=1, edge_order=2)
    grad_hlogP = np.gradient(grad_h, logP, axis=1, edge_order=2)
    P2 = np.tile(p_vals[None, :], (meta["N_h"], 1))

    table["conductivity"]["value"]      = k
    table["conductivity"]["grad_h"]     = grad_h
    table["conductivity"]["grad_logP"]  = grad_logP
    table["conductivity"]["grad_p"]     = grad_logP / P2
    table["conductivity"]["grad_hlogP"] = grad_hlogP
    table["conductivity"]["grad_ph"]    = grad_hlogP / P2
    table["conductivity"]["coeffs"] = np.asarray(compute_bicubic_coefficients(
        k, grad_h, grad_logP, grad_hlogP, delta_h, delta_logP))

    # Saturation-line conductivity (read by the closure through the sat props).
    if "saturation_props" in table and "conductivity" in table["saturation_props"]:
        sp = table["saturation_props"]
        p_sat = np.asarray(sp["pressure"]["value"], dtype=float)
        T_sat = np.asarray(sp["temperature"]["value"], dtype=float)
        rho_sat = np.asarray(sp["density"]["value"], dtype=float)
        k_sat, _, _ = conductivity_field(T_sat[None, :], rho_sat[None, :])
        k_sat = k_sat[0]
        sp["conductivity"]["value"] = k_sat
        sp["conductivity"]["grad_p"] = np.gradient(k_sat, p_sat, edge_order=2)

    with open(pkl_path, "wb") as f:
        pickle.dump(table, f)

    print(f"  {pkl_path.name}")
    print(f"    NaN conductivity nodes before : {was_nan} / {k.size}")
    print(f"    corresponding-states scale    : {scale:.5f} "
          f"(Gamma_{SURROGATE} / Gamma_R1233zd(E))")
    print(f"    surrogate calls that failed   : {n_fail} (nearest-valid filled)")
    print(f"    k range after patch           : {k.min():.5f} .. {k.max():.5f} W/m/K")
    return True


def main():
    print(f"Patching thermal conductivity from {SURROGATE} "
          f"(corresponding states) ...")
    names = [f"{cfg.FLUID_NAME}_meta_liquid_{cfg.N_H}x{cfg.N_P}.pkl",
             f"{cfg.FLUID_NAME}_meta_vapor_{cfg.N_H}x{cfg.N_P}.pkl"]
    missing = [n for n in names if not (cfg.TABLE_DIR / n).exists()]
    if missing:
        print(f"  ERROR: table(s) not found: {', '.join(missing)}")
        print("  Run build_tables.py first.")
        return 1
    for n in names:
        patch(cfg.TABLE_DIR / n)
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
