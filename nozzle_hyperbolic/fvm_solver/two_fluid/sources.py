"""
Source-term assembly for Städtke's two-fluid model.

Two public functions:
  D_s_primitive      — primitive entropy-frame source (6,) from Table B.46
  D_VE_conservative  — conservative total-energy source (6,) = J Phi^{-1} D_s

Closure helper:
  two_fluid_closure  — wraps correlations from source_two_fluid.py into the
                       TwoFluidClosure NamedTuple consumed by D_s_primitive.

Source formulas: pipeline doc Appendix A (stadtke_twofluid_implementation_pipeline_with_sources.md)
k → k*rho correction applied throughout (k alone in Städtke's Table B.46 is a typo).
"""

from typing import NamedTuple

import jax.numpy as jnp
import jaxprop as jxp

from .matrices import PhaseProps, CommonQty, J_matrix, Phi_inv_matrix

# Closure correlations from the existing two-fluid source library
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source_two_fluid import (
    calculate_drag_force_one_component,
    calculate_mass_transfer_lv,
    interfacial_area,
    ranz_marshall_htc_two_fluid,
)


# ---------------------------------------------------------------------------
# Physical-source container
# ---------------------------------------------------------------------------

class TwoFluidClosure(NamedTuple):
    """
    Physical closure quantities consumed by D_s_primitive.

    All quantities are per-unit-volume and are scalars (per-cell).
    Sign convention: sigma_Mg > 0 = evaporation (liquid → gas).
    """
    sigma_Mg : jnp.ndarray  # gas mass source  [kg/(m^3 s)]
    sigma_Qg : jnp.ndarray  # interfacial heat to gas   [W/m^3]
    Fv_g     : jnp.ndarray  # interfacial drag on gas   [N/m^3]
    F_g      : jnp.ndarray  # external body force on gas [N/m^3]
    F_l      : jnp.ndarray  # external body force on liquid [N/m^3]
    Q_g      : jnp.ndarray  # wall/external heat to gas  [W/m^3]
    Q_l      : jnp.ndarray  # wall/external heat to liquid [W/m^3]
    u_ex     : jnp.ndarray  # velocity of transferred mass [m/s]
    u_int    : jnp.ndarray  # interface velocity [m/s]
    h_ex     : jnp.ndarray  # enthalpy of transferred mass [J/kg]
    dlnA_dx  : jnp.ndarray  # (1/A) dA/dx — log area derivative [1/m]

    # TODO: add heat fluxes among the phasees


# ---------------------------------------------------------------------------
# Closure helper
# ---------------------------------------------------------------------------

def two_fluid_closure(
    U_h,
    eos_g: PhaseProps,
    eos_l: PhaseProps,
    fluid_g,               # FluidBicubic — for viscosity/conductivity lookup
    fluid_l,               # FluidBicubic
    T_sat,                 # saturation temperature at current pressure [K]
    h_sat_g,               # saturation vapour enthalpy [J/kg]
    h_sat_l,               # saturation liquid enthalpy [J/kg]
    dlnA_dx=0.0,           # area derivative [1/m]
    F_g=0.0,               # gravity or other external force on gas [N/m^3]
    F_l=0.0,               # gravity or other external force on liquid [N/m^3]
    Q_g=0.0,               # wall heat to gas [W/m^3]
    Q_l=0.0,               # wall heat to liquid [W/m^3]
    Nb=1e10,               # bubble/droplet number density [1/m^3]
    Nd=1e10,
    alpha_b=0.3,           # TA-BD transition void fractions
    alpha_d=0.7,
) -> TwoFluidClosure:
    """
    Assemble a TwoFluidClosure from state + saturation properties.

    Uses:
      - interfacial_area    → dispersed-structure diameter d and area ai
      - ranz_marshall_htc_two_fluid  → HTC for each phase
      - calculate_drag_force_one_component  → Fv_g
      - calculate_mass_transfer_lv  → sigma_Mg

    The saturation properties (T_sat, h_sat_g, h_sat_l) must be supplied
    externally from a saturation EOS call so that this function stays
    differentiable inside jax.jit (no secondary EOS solve needed).

    Parameters
    ----------
    U_h : (6,)  primitive state  [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g, eos_l : PhaseProps pre-computed at U_h
    fluid_g, fluid_l : FluidBicubic tables (needed for mu, k, cp properties)
    T_sat, h_sat_g, h_sat_l : saturation properties at pressure U_h[0]
    dlnA_dx : logarithmic area gradient [1/m]
    F_g, F_l : external body forces [N/m^3]
    Q_g, Q_l : wall heat sources [W/m^3]
    Nb, Nd   : number densities for TA-BD interfacial area model
    alpha_b, alpha_d : TA-BD transition void fractions
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  T_g = eos_g.T
    rl = eos_l.rho;  T_l = eos_l.T

    # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
    p_arr  = jnp.expand_dims(p, 0)
    stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_g, 0), p_arr)
    stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_l, 0), p_arr)

    # ── Interfacial area and dispersed-structure diameter ────────────────────
    ai, d = interfacial_area(ag, alpha_b=alpha_b, alpha_d=alpha_d,
                             Nb=Nb, Nd=Nd)

    # ── Drag force on gas per unit volume (sign: F > 0 accelerates gas) ─────
    mu_g = stg_ex["viscosity"][0]
    mu_l = stl_ex["viscosity"][0]
    # Drag force per unit volume: F > 0 means force on gas in direction of u_l - u_g
    Fv_g = -calculate_drag_force_one_component(
        ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
        alpha_b=alpha_b, alpha_d=alpha_d,
    )

    # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
    du_abs = jnp.abs(u_g - u_l)
    # Gas-phase HTC: continuous phase depends on flow regime (use liquid as
    # continuous in bubbly, gas in droplet; transition blended).
    # For simplicity use liquid-continuous HTC for gas bubbles.
    htc_g = ranz_marshall_htc_two_fluid(
        {k: v[0] for k, v in stl_ex.items()},   # liquid as continuous
        du_abs, d,
    )
    htc_l = ranz_marshall_htc_two_fluid(
        {k: v[0] for k, v in stg_ex.items()},   # gas as continuous (droplet regime)
        du_abs, d,
    )

    # ── Interfacial heat transfer rates ──────────────────────────────────────
    Qvi = htc_g * ai * (T_sat - T_g)    # heat from interface to gas [W/m^3]
    Qli = htc_l * ai * (T_sat - T_l)    # heat from interface to liquid [W/m^3]

    # ── Mass transfer (evaporation/condensation) ─────────────────────────────
    sigma_Mg = calculate_mass_transfer_lv(
        T_l, T_g, T_sat, htc_l, htc_g, ai,
        h_sat_l, h_sat_g, h_l, h_g, Qvi, Qli,
    )

    # ── Interface kinematics ─────────────────────────────────────────────────
    u_mix2 = (ag * rg * u_g + al * rl * u_l) / (ag * rg + al * rl)  # mass-weighted
    u_ex   = u_mix2        # velocity of transferred mass
    u_int  = u_mix2        # interface velocity (viscous-work calculation)
    # Enthalpy of transferred mass: saturation enthalpy of appearing phase
    h_ex   = jnp.where(sigma_Mg >= 0.0, h_sat_g, h_sat_l)

    # TODO: check the source terms implementation and compare it with nozzlex and with thee sources in Stadtke

    return TwoFluidClosure(
        sigma_Mg=sigma_Mg,
        sigma_Qg=Qvi,
        Fv_g=Fv_g,
        F_g=F_g,
        F_l=F_l,
        Q_g=Q_g,
        Q_l=Q_l,
        u_ex=u_ex,
        u_int=u_int,
        h_ex=h_ex,
        dlnA_dx=dlnA_dx,
    )


# ---------------------------------------------------------------------------
# D_s_primitive  — Städtke Table B.46 with k → k*rho correction
# ---------------------------------------------------------------------------

def D_s_primitive(U_h, eos_g: PhaseProps, eos_l: PhaseProps,
                  cq: CommonQty, closure: TwoFluidClosure, k) -> jnp.ndarray:
    """
    Primitive entropy-frame source vector D_s (6,).

    Implements Städtke Table B.46 with the k → k*rho correction applied
    throughout rows 2 and 3 (every bare 'k' next to a density in B.46
    must be read as k*rho_mix).

    Parameters
    ----------
    U_h     : (6,) primitive state [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g, eos_l : PhaseProps at U_h
    cq      : CommonQty at U_h  (includes rho_mix, rho_s, rh2, etc.)
    closure : TwoFluidClosure containing physical source inputs
    k       : virtual-mass coefficient

    Returns
    -------
    D_s : (6,) — [D_p, D_{u_g}, D_{u_l}, D_{alpha}, D_{s_g}, D_{s_l}]
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  ag_sq = eos_g.a**2;  T_g = eos_g.T
    rl = eos_l.rho;  al_sq = eos_l.a**2;  T_l = eos_l.T
    rg_h = eos_g.rho_h;  rl_h = eos_l.rho_h

    rho_mix = cq.rho_mix;  rho_s = cq.rho_s;  rh2 = cq.rh2;  a0sq = cq.a0sq
    du      = cq.du

    # k*rho_mix (not stored in cq; derive from rg_hat = rg + k*rho_mix)
    k_rho   = cq.rg_hat - rg     # = k * rho_mix

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    Fv_g     = closure.Fv_g
    F_g      = closure.F_g
    F_l      = closure.F_l
    Q_g      = closure.Q_g
    Q_l      = closure.Q_l
    u_ex     = closure.u_ex
    u_int    = closure.u_int
    h_ex     = closure.h_ex
    dlnA_dx  = closure.dlnA_dx

    sigma_Ml = -sigma_Mg
    sigma_Ql = -sigma_Qg
    Fv_l     = -Fv_g

    # ── Entropy sources T_i * sigma_i^S  (Städtke eq. 4.50) ─────────────────
    Tg_sSg = (sigma_Mg * ((h_ex - h_g) + 0.5 * (u_ex - u_g)**2)
              + Fv_g  * (u_int - u_g)
              + sigma_Qg + Q_g)
    Tl_sSl = (sigma_Ml * ((h_ex - h_l) + 0.5 * (u_ex - u_l)**2)
              + Fv_l  * (u_int - u_l)
              + sigma_Ql + Q_l)
    sSg = Tg_sSg / T_g   # sigma_g^S
    sSl = Tl_sSl / T_l   # sigma_l^S

    # ── Recurring mass+entropy group (Appendix A) ────────────────────────────
    Mg = sigma_Mg - (T_g * rg_h / rg) * sSg
    Ml = sigma_Ml - (T_l * rl_h / rl) * sSl

    # ── Row 1: pressure ──────────────────────────────────────────────────────
    Ds_1 = (rl * a0sq / rho_s * Mg
          + rg * a0sq / rho_s * Ml
          - rl * rg / rho_s * (ag * u_g + al * u_l) * a0sq * dlnA_dx)

    # ── Row 2: gas velocity (k → k*rho correction applied) ───────────────────
    Ds_2 = (rl / (ag * rh2) * (Fv_g + sigma_Mg * (u_ex - u_g))
          - sigma_Mg * (k_rho * rg + al * (rl**2 - rg**2)) / (rg * rh2) * du)
        #   + (F_g * (rl + ag * k_rho) + F_l * ag * k_rho) / (ag * rh2))

    # ── Row 3: liquid velocity (k → k*rho correction applied) ────────────────
    Ds_3 = (-rg / (al * rh2) * (Fv_g + sigma_Mg * (u_ex - u_l))
           - sigma_Mg * (k_rho * rl + ag * (rg**2 - rl**2)) / (rl * rh2) * du)
        #    + (F_g * al * k_rho + F_l * (rg + al * k_rho)) / (al * rh2))

    # ── Row 4: void fraction ─────────────────────────────────────────────────
    Ds_4 = (al * a0sq / (rho_s * al_sq) * Mg
          - ag * a0sq / (rho_s * ag_sq) * Ml
          + al * ag * a0sq / rho_s * (rl * u_l / ag_sq - rg * u_g / al_sq) * dlnA_dx)

    # ── Rows 5, 6: entropy ───────────────────────────────────────────────────
    Ds_5 = sSg / (ag * rg)
    Ds_6 = sSl / (al * rl)

    return jnp.array([Ds_1, Ds_2, Ds_3, Ds_4, Ds_5, Ds_6])


# ---------------------------------------------------------------------------
# D_VE_conservative  — D_VE = J Phi^{-1} D_s
# ---------------------------------------------------------------------------

def D_VE_conservative(U_h, eos_g: PhaseProps, eos_l: PhaseProps,
                      cq: CommonQty, closure: TwoFluidClosure, k) -> jnp.ndarray:
    """
    Conservative total-energy source D_VE (6,) = J @ Phi^{-1} @ D_s.

    Chain (derivation §2.7 of pipeline doc):
      1. Compute D_s from D_s_primitive (Table B.46)
      2. Apply Phi^{-1}: only rows 5, 6 change
            D_h[4] = D_s[0]/rho_g + T_g * D_s[4]
            D_h[5] = D_s[0]/rho_l + T_l * D_s[5]
      3. Multiply by J = dV/dU_h to get conservative source

    Returns
    -------
    D_VE : (6,) — conservative source [D_{m_g}, D_{m_l}, D_{p_g}, D_{p_l}, D_{E_g}, D_{E_l}]
    """
    D_s = D_s_primitive(U_h, eos_g, eos_l, cq, closure, k)

    rg  = eos_g.rho;  T_g = eos_g.T
    rl  = eos_l.rho;  T_l = eos_l.T

    # TODO: where is D_h being assinged

    # Phi^{-1}: identity on rows 0-3; rows 4, 5 mix pressure row (D_s[0]) with entropy rows
    D_h = D_s.at[4].set(D_s[0] / rg + T_g * D_s[4])
    D_h = D_h.at[5].set(D_s[0] / rl + T_l * D_s[5])

    # J @ D_h
    J = J_matrix(U_h, eos_g, eos_l)
    return J @ D_h
