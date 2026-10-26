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

import jax
import jax.numpy as jnp
import jaxprop as jxp

from .matrices import PhaseProps, CommonQty, J_matrix, Phi_inv_matrix

# Closure correlations from the existing two-fluid source library
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source_two_fluid import (
    LIQUID_HTC_BUBBLY,
    LIQUID_HTC_DROPLET,
    DRAG_LAWS,
    calculate_drag_force_one_component,
    calculate_heat_transfer_one_component,
    calculate_mass_transfer_lv,
    get_interfacial_area,
    jakob_nusselt,
    liao_nusselt,
    ranz_marshall_htc_two_fluid,
    select_liquid_htc,
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
    sigma_Ql : jnp.ndarray  # interfacial heat to gas   [W/m^3]
    Fv_g     : jnp.ndarray  # interfacial drag on gas   [N/m^3]
    F_g      : jnp.ndarray  # external body force on gas [N/m^3]
    F_l      : jnp.ndarray  # external body force on liquid [N/m^3]
    Q_g      : jnp.ndarray  # wall/external heat to gas  [W/m^3]
    Q_l      : jnp.ndarray  # wall/external heat to liquid [W/m^3]
    u_ex     : jnp.ndarray  # velocity of transferred mass [m/s]
    u_int    : jnp.ndarray  # interface velocity [m/s]
    h_ex_g   : jnp.ndarray  # enthalpy of mass entering gas phase [J/kg]  (h_sat_g for evaporation)
    h_ex_l   : jnp.ndarray  # enthalpy of mass leaving liquid phase [J/kg] (h_sat_l for evaporation)
    dlnA_dx  : jnp.ndarray  # (1/A) dA/dx — log area derivative [1/m]

    # TODO: add heat fluxes among the phasees


# ---------------------------------------------------------------------------
# Closure helper
# ---------------------------------------------------------------------------

def evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag, gate_dT=0.5, gate_dh=2.0e3,
                       dh_gate=1.0, dT_gate=0.001, p=None, p_max=None):
    """Evaporation-gate weight in [0, 1] — liquid superheat OR appreciable α.

    w → 1 where the liquid is superheated (h AND T criteria), w → 0 where
    subcooled.  Single source of truth shared by the mass-transfer closure
    (σ gate) and the solver's ghost-vapor ENERGY pin (pin weight = 1 − w): the
    pin releases exactly where evaporation switches on, so the gate can never
    prevent flashing from starting.

    α-AVAILABILITY CLAUSE (`| ag > 5e-2`): w is ALSO forced to 1 once the vapor
    is physically real (α > 5%), regardless of liquid superheat.  This
    deliberately breaks the older "liquid quantities only" rule, and it is the
    fix for the shock-condensation case flagged in TODO_two_phase.md: with a
    pure superheat gate, a recompression shock re-subcools the liquid at α~0.5
    and would RE-PIN genuinely real vapor to h_sat — i.e. force equilibrium
    condensation.  Keyed on α, a real phase is never pinned.

    Safe w.r.t. the solver's NO-DEADLOCK invariant: the clause only WIDENS the
    w=1 (evaporation-on / pin-off) region, never narrows it, so nothing in the
    α-growth chain can be gated off by it.

    SIDE EFFECT to be aware of: σ_Mg is likewise ungated wherever α > 5%, so in
    a subcooled region with real vapor the closure now returns σ < 0 — the
    CONDENSATION branch is active there (intended for shocks; it was previously
    suppressed by the evaporation-only choice).

    PRESSURE CUT-OFF (p, p_max): with p_max given, w = 0 wherever p >= p_max —
    no phase change at all (neither clause) and the ghost pin stays on.  For a
    supercritical inlet: the saturation table ends at 0.98 p_crit, above it the
    lookup is clamped, and the clamped T_sat/h_sat must not drive phase change.
    Valid when the vapor above p_max is a ghost: a REAL vapor (α > 5%) there
    would be pinned to the clamped h_sat_g.  p_max = None → no cut-off.
    """
    if p_max is not None:
        w = evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag, gate_dT=gate_dT,
                               gate_dh=gate_dh, dh_gate=dh_gate, dT_gate=dT_gate)
        return jnp.where(p < p_max, w, 0.0)
    # BINARY on/off gate (2026-07-21 A/B, nozzlex parity): hard switch on the
    # same h-AND-T criterion.  Flips BOTH the σ gate and the ghost pin weight
    # (1−w) together — pre-onset: σ=0 & fully pinned; post-onset: σ on & free.
    # Watch for: (a) per-cell flicker in time at the moving front (add
    # hysteresis if seen, don't go back to the blend silently), (b) Newton
    # Jacobian sees zero derivative across the switch (pin removes the fragile
    # h_g dof, expected harmless).
    # NOTE the parentheses around (ag > ...): in Python `&`/`|` bind TIGHTER than
    # `>`, so `A & B | ag > 5e-2` parses as `((A & B) | ag) > 5e-2` → bitwise-OR
    # of a bool array with the float64 α (TypeError under JAX).
    return jnp.where((h_l > h_sat_l + dh_gate) & (T_l > T_sat + dT_gate)
                     | (ag > 5e-2),
                     1.0, 0.0)
    # SMOOTH C¹ version (WORKING baseline 2026-07-21 — full-physics steady
    # state, SSR 3.4e-6; kept for revert if the binary gate flickers):
    # _sig = lambda z: 0.5 * (1.0 + jnp.tanh(z))
    # return _sig(jnp.minimum((h_l - h_sat_l - dh_gate) / gate_dh,
    #                         (T_l - T_sat   - dT_gate) / gate_dT))


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
    mass_transfer=True,
    regime_htc=False,      # True → regime-dependent HTC (bubbly/transition/droplet);
                           # False → NICFD closure (bubbly-regime HTC at every α)
    regime_blend="linear", # transition alpha_b..alpha_d of the drag and of the
                           # regime HTC: 'linear' | 'smoothstep' (C1, regime_weight)
    gate_dT= 1e-3, # 0.5,          # smooth-gate temperature width [K].  NOTE 2026-07-21:
                           # widening to 2.0 K (onset spread ~4-8 cells) was
                           # PROBED and is NEGATIVE — died EARLIER (step 292 vs
                           # 386, dt collapse at t=5.9e-4 vs 1.55e-3 s).  The
                           # front-cell blow-up is not a σ(x) sharpness problem.
    gate_dh= 0.5, #2.0e3,         # smooth-gate enthalpy width [J/kg]  (~cp_l·gate_dT)
    p_mt_max=None,         # [Pa] no phase change at p >= p_mt_max (see
                           # evaporation_weight); None → no cut-off
    area_law="polynomial", # interfacial area law: original | polynomial | parabola
                           # (source_two_fluid.INTERFACIAL_AREA_LAWS)
    area_params=None,      # dict of keyword overrides for that law, e.g. {"d_min": 6.5e-5}
    ai_scale=1.0,          # correction of the law in this cell (field inversion, beta):
                           # ai -> ai_scale * ai, and d follows from the scaled ai
    smooth_slip=False,     # True → Ranz-Marshall Re from the smooth effective slip
                           # sqrt(du² + smooth_slip_u0²); False → |du| (cusp at du = 0).
                           # Off since 2026-10-23: htc_liquid_bubbly='max' removes the cusp
                           # without moving ṁ (THROAT_BUMP_HISTORY.md)
    smooth_slip_u0=2.0,    # [m/s] slip scale of the smooth effective slip (2: smallest
                           # tested that removes the case-4 throat bump, THROAT_BUMP_HISTORY.md)
    nu_g_int=None,         # internal Nusselt number of the vapour (bubbles, htc_g = k_g Nu / d);
                           # None → 120 with mass transfer, 12 without (the earlier constants)
    htc_liquid_bubbly="ranz_marshall",  # liquid-side HTC, bubbly regime: ranz_marshall | jakob | max | liao
    htc_liquid_droplet="internal",      # liquid-side HTC, droplet regime: internal | jakob | max
    drag_bubbly="schiller_naumann",     # drag coefficient, bubbly regime: schiller_naumann | ishii_zuber
    drag_droplet="schiller_naumann",    # drag coefficient, droplet regime: schiller_naumann | ishii_zuber
) -> TwoFluidClosure:
    """
    Assemble a TwoFluidClosure from state + saturation properties.

    Uses:
      - interfacial_area    → dispersed-structure diameter d and area ai
      - ranz_marshall_htc_two_fluid  → HTC for each phase (NICFD, regime_htc=False)
      - calculate_heat_transfer_one_component  → HTC for each phase (regime_htc=True)
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
    regime_htc : False → NICFD HTC (internal Nu on the vapor side + Ranz-Marshall
                 with liquid continuous, at every α); True → same for α ≤ alpha_b,
                 mirrored droplet closure for α ≥ alpha_d, linear blend in between
    regime_blend : 'linear' | 'smoothstep' weight across alpha_b..alpha_d for the
                 drag and (with regime_htc) the HTC; smoothstep makes both C1
    area_law, area_params : interfacial area law by name and its keyword overrides
                 (get_interfacial_area); Nb / Nd only matter for 'original'
    ai_scale   : correction factor of the area law in this cell (field inversion): ai is
                 multiplied by it BEFORE d is derived, so drag and HTCs see the corrected d
                 too. 1 = the law itself (bit-identical)
    smooth_slip, smooth_slip_u0 : True → the Ranz-Marshall HTCs use the smooth
                 effective slip sqrt(du² + u0²) instead of |du|.  With |du|, Nu has a
                 cusp (infinite slope) at du = 0 and in the bubbly regime σ_Mg follows
                 sqrt|du|: where the slip changes sign it swings ~2× cell to cell
                 (measured 2026-10-22 on CO2 case 4 at x ≈ 0.57, odd-even 29% of σ_Mg
                 with p, α and superheat smooth to ~1%).  False → original closure.
    nu_g_int   : internal Nusselt number of the vapour inside bubbles (vapour-side HTC of
                 the bubbly regime).  None keeps the constants used so far (120 in the
                 mass-transfer branch, 12 in the no-mass-transfer branch); a value sets both.
    htc_liquid_bubbly, htc_liquid_droplet : liquid-side HTC of each regime.  'jakob' is the
                 thermal-diffusion bubble growth Nu = max(2, (12/π) Ja) (Plesset & Zwick 1954,
                 source_two_fluid.jakob_nusselt), Ja = ρ_l cp_l (T_l - T_sat) / (ρ_g h_fg) with
                 the local ρ_g; independent of the slip.  'max' takes the larger of it and the
                 regime's own closure (Ranz-Marshall bubbly, internal Nu droplet).  'liao'
                 (bubbly only) = sqrt(12/π² Ja² + Pe/(3π)), thermal diffusion + convection
                 (source_two_fluid.liao_nusselt); its Pe uses the slip as Ranz-Marshall does
                 (smooth_slip).  Defaults = the closure used so far.  Jakob needs mass_transfer=True; without regime_htc
                 only the bubbly option applies.
    drag_bubbly, drag_droplet : drag coefficient of each regime.  'ishii_zuber' = viscous
                 regime of Ishii & Zuber (1979), C_d = 24/Re_m (1 + 0.1 Re_m^0.75) with the
                 mixture viscosity μ_c (1 - α_d)^(-2.5 μ*) (crowding; μ_c/(1-α) for bubbles,
                 μ_c (1-α)^-2.5 for drops); distorted / Newton regimes not included.
                 Defaults = Schiller-Naumann in both regimes (the closure used so far).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag
    slip_u0 = float(smooth_slip_u0) if smooth_slip else 0.0
    if htc_liquid_bubbly not in LIQUID_HTC_BUBBLY or htc_liquid_droplet not in LIQUID_HTC_DROPLET:
        raise ValueError(f"htc_liquid_bubbly in {LIQUID_HTC_BUBBLY}, htc_liquid_droplet in "
                         f"{LIQUID_HTC_DROPLET}; got {htc_liquid_bubbly!r}, {htc_liquid_droplet!r}")
    if not mass_transfer and (htc_liquid_bubbly, htc_liquid_droplet) != ("ranz_marshall", "internal"):
        raise ValueError("the Jakob liquid HTC needs mass_transfer=True")
    if drag_bubbly not in DRAG_LAWS or drag_droplet not in DRAG_LAWS:
        raise ValueError(f"drag_bubbly / drag_droplet in {DRAG_LAWS}; got {drag_bubbly!r}, {drag_droplet!r}")

    rg = eos_g.rho;  T_g = eos_g.T
    rl = eos_l.rho;  T_l = eos_l.T

    # --- Two-phase gate (SYMMETRIC, enthalpy AND temperature) ----------------
    # Mass transfer activates only when a phase is away from saturation in the
    # direction that drives a real phase change, requiring BOTH the enthalpy and
    # the temperature to agree:
    #   evaporation  if the LIQUID is superheated  (h_l > h_sat_l AND T_l > T_sat)
    #   condensation if the VAPOUR is subcooled     (h_g < h_sat_g AND T_g < T_sat)
    # and OFF otherwise.  Requiring both h and T is the conservative choice and
    # stays well-posed toward the CRITICAL region, where the saturation split (and
    # any single h- or T-only test) becomes degenerate: if one variable is
    # spuriously triggered by table extrapolation, the other guards it.  Small
    # deadbands (dh_gate, dT_gate) cover the ~mK / ~tens-of-J/kg inconsistency
    # between the metastable property tables and the saturation table (test_T0).
    #
    # Replaces the old `is_two_phase = True` override, which forced mass transfer
    # even at saturation and let a superheated VAPOUR spuriously drive evaporation.
    if mass_transfer == True:
        # SMOOTH C¹ gate (2026-07-21) — replaces the binary lax.cond gate.
        # The binary gate is a checkerboard PUMP at the flashing front: with
        # mK/J-scale deadbands, neighbouring cells straddle the threshold and
        # σ_Mg flips full-on/off (even evap/cond) cell-to-cell — measured 76%
        # 2Δx oscillation in σ_Mg itself at the blow-up.  Worse, lax.cond has
        # ZERO derivative across the branch switch, so the implicit Newton's
        # Jacobian cannot see the source activating and overshoots into the
        # flip every step.  Same physics and the same h-AND-T conservatism,
        # expressed as sigmoid weights over the LEAST-activated of the two
        # criteria (smooth AND); the two directions combine as a smooth OR.
        # Widths gate_dT/gate_dh set the transition scale; the old deadbands
        # (dh_gate, dT_gate) remain as the sigmoid centres, covering the
        # table/saturation inconsistency (test_T0).  As widths → 0 this
        # recovers the binary gate exactly.
        # OLD binary gate (kept for reference / quick revert):
        # can_evaporate = (h_l > h_sat_l + dh_gate) & (T_l > T_sat + dT_gate)
        # can_condense  = (h_g < h_sat_g - dh_gate) & (T_g < T_sat - dT_gate)
        # is_two_phase  = can_evaporate | can_condense
        # Shared helper — SAME weight the solver's ghost-vapor pin uses as 1−w.
        w_evap  = evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag,
                                     gate_dT=gate_dT, gate_dh=gate_dh,
                                     p=p, p_max=p_mt_max)
        # EVAPORATION-ONLY (2026-07-21, nozzlex parity) — nozzlex gates with
        # is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat): NO condensation branch.
        # Our condensation weight half-opens AT the saturation line, so the
        # subcooled pre-flashing ghost (h_g ≈ h_sat_g, T_g ≈ T_sat) was DRAINED
        # by σ<0 onto the α clip floor; the non-conservative clip then fights
        # the source every step — measured as the u_g/h_g 2Δx pump pinned on
        # the floor plateau cells at flashing onset (x≈0.08, α=1e-3 exactly,
        # u_g=-5.7 m/s between +199 m/s neighbours).  nozzlex's implied σ in
        # that region is EXACTLY 0; ours was −100..−470 kg/(m³s).
        # w_cond = _sig(jnp.minimum((h_sat_g - h_g - dh_gate) / gate_dh,
        #                           (T_sat   - T_g - dT_gate) / gate_dT))
        # w_mt   = 1.0 - (1.0 - w_evap) * (1.0 - w_cond)   # smooth OR
        w_mt    = w_evap
        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr  = jnp.expand_dims(p, 0)

        # analytic EOS (FluidPerfectGas) has no bounds → PHYSICAL floors, not ±inf
        # (p<0 → rho<0 → sqrt NaN; observed 2026-07-14 Bernoulli ghost case).
        p_arr = jnp.clip(p_arr, getattr(fluid_l, "p_min", 1e2), getattr(fluid_l, "p_max", 1e9))
        p_arr = jnp.clip(p_arr, getattr(fluid_g, "p_min", 1e2), getattr(fluid_g, "p_max", 1e9))

        h_g = jnp.clip(jnp.expand_dims(h_g, 0), getattr(fluid_g, "h_min", 1e3), getattr(fluid_g, "h_max", 5e6))
        h_l = jnp.clip(jnp.expand_dims(h_l, 0), getattr(fluid_l, "h_min", 1e3), getattr(fluid_l, "h_max", 5e6))

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # d_min_b=5e-5 matches nozzlex (nozzle_two_fluid_one_component_autonomous).
        ai, d = get_interfacial_area(area_law, area_params)(ag, alpha_b=alpha_b, alpha_d=alpha_d,
                                Nb=Nb, Nd=Nd, d_min_b=5e-6, ai_scale=ai_scale)
        # ai, d = 100, 5e-5
        # ── Drag force on gas per unit volume (sign: F > 0 accelerates gas) ─────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]   
        mu_l = stl_ex["viscosity"][0]
        # Drag force per unit volume: F > 0 means force on gas in direction of u_l - u_g
        Fv_g = -calculate_drag_force_one_component(
            ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
            alpha_b=alpha_b, alpha_d=alpha_d, blend=regime_blend,
            drag_bubbly=drag_bubbly, drag_droplet=drag_droplet,
        )

        # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
        du_abs = jnp.abs(u_g - u_l)
        Nu_g, Nu_l = (120.0 if nu_g_int is None else float(nu_g_int)), 12.0
        ja_l = rl * stl_ex["cp"][0] * (T_l - T_sat) / (rg * (h_sat_g - h_sat_l))   # liquid Jakob number

        if regime_htc:
            # Regime-dependent HTC: bubbly (α ≤ alpha_b) identical to the NICFD
            # closure below, droplet (α ≥ alpha_d) mirrored with vapor continuous,
            # linear blend in between (as done for the drag).
            htc_g, htc_l = calculate_heat_transfer_one_component(
                ag,
                {"rho": stg_ex["rho"][0], "mu": stg_ex["mu"][0],
                 "k": stg_ex["k"][0], "cp": stg_ex["cp"][0]},
                {"rho": stl_ex["rho"][0], "mu": stl_ex["mu"][0],
                 "k": stl_ex["k"][0], "cp": stl_ex["cp"][0]},
                u_g, u_l, d, Nu1_int=Nu_g, Nu2_int=Nu_l,
                alpha_b=alpha_b, alpha_d=alpha_d, blend=regime_blend, slip_u0=slip_u0,
                ja=ja_l, liquid_bubbly=htc_liquid_bubbly, liquid_droplet=htc_liquid_droplet,
            )
        else:
            # NICFD closure: bubbly-regime HTC at every α.
            # Gas-phase HTC: continuous phase depends on flow regime (use liquid as
            # continuous in bubbly, gas in droplet; transition blended).
            # For simplicity use liquid-continuous HTC for gas bubbles.
            # htc_g = ranz_marshall_htc_two_fluid(
            #     {k: v[0] for k, v in stl_ex.items()},   # liquid as continuous
            #     du_abs, d,
            # )

            htc_g = (k_g * Nu_g) / d

            htc_l = ranz_marshall_htc_two_fluid(
                {                                         # liquid as continuous (bubbly regime)
                    "rho": stl_ex["rho"][0],             # FluidState alias → density
                    "mu":  stl_ex["mu"][0],              # FluidState alias → viscosity
                    "k":   stl_ex["k"][0],               # FluidState alias → conductivity
                    "cp":  stl_ex["cp"][0],              # FluidState alias → isobaric_heat_capacity
                },
                du_abs, d, slip_u0,
            )
            u_pe = jnp.sqrt(du_abs**2 + slip_u0**2) if slip_u0 > 0.0 else du_abs
            pe_l = u_pe * d * stl_ex["rho"][0] * stl_ex["cp"][0] / stl_ex["k"][0]
            htc_l = select_liquid_htc(htc_liquid_bubbly, htc_l, stl_ex["k"][0] * jakob_nusselt(ja_l) / d,
                                      stl_ex["k"][0] * liao_nusselt(ja_l, pe_l) / d)

        # # ── Interfacial heat transfer rates ──────────────────────────────────────
        # Qvi = htc_g * ai * (T_sat - T_g)    # heat from interface to gas [W/m^3]
        # Qli = htc_l * ai * (T_sat - T_l)    # heat from interface to liquid [W/m^3]

        # # ── Mass transfer (evaporation/condensation) ─────────────────────────────
        # sigma_Mg = calculate_mass_transfer_lv(
        #     T_l, T_g, T_sat, htc_l, htc_g, ai,
        #     h_sat_l, h_sat_g, h_l, h_g, Qvi, Qli,
        # )

        # ── OLD binary dispatch (kept for reference / quick revert) ──────────
        # def _source_terms_two_phase(_args):
        #     _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv = _args
        #     _Qvi = _ht_v * _Ai * (_Tsat - _Tv)
        #     _Qli = _ht_l * _Ai * (_Tsat - _Tl)
        #     _m_dot_lv = calculate_mass_transfer_lv(
        #         _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv, _Qvi, _Qli,
        #     )
        #     return _Qvi, _Qli, _m_dot_lv
        #
        # def _source_terms_single_phase(_args):
        #     return (0.0, 0.0, 0.0)
        #
        # Qvi, Qli, sigma_Mg = jax.lax.cond(
        #     is_two_phase,
        #     _source_terms_two_phase,
        #     _source_terms_single_phase,
        #     (T_l, T_g, T_sat, htc_l, htc_g, ai, h_sat_l, h_sat_g, h_l, h_g),
        # )

        # Always evaluate, then scale ALL THREE interfacial exchanges by the
        # SAME smooth weight — this keeps the interface energy balance
        # σ_Mg = -(Qvi+Qli)/(h_sat_g - h_sat_l) exact at every gate level
        # (scaling σ alone would create/destroy interfacial energy).
        _Qvi = htc_g * ai * (T_sat - T_g)
        _Qli = htc_l * ai * (T_sat - T_l)

        _sMg = calculate_mass_transfer_lv(
            T_l, T_g, T_sat, htc_l, htc_g, ai, h_sat_l, h_sat_g, h_l, h_g, _Qvi, _Qli,
        )
        Qvi      = w_mt * _Qvi
        Qli      = w_mt * _Qli
        sigma_Mg = w_mt * _sMg
    else:     # No mass transfer, just heat and drag

        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr  = jnp.expand_dims(p, 0)

        # analytic EOS (FluidPerfectGas) has no bounds → PHYSICAL floors, not ±inf
        # (p<0 → rho<0 → sqrt NaN; observed 2026-07-14 Bernoulli ghost case).
        p_arr = jnp.clip(p_arr, getattr(fluid_l, "p_min", 1e2), getattr(fluid_l, "p_max", 1e9))
        p_arr = jnp.clip(p_arr, getattr(fluid_g, "p_min", 1e2), getattr(fluid_g, "p_max", 1e9))

        h_g = jnp.clip(jnp.expand_dims(h_g, 0), getattr(fluid_g, "h_min", 1e3), getattr(fluid_g, "h_max", 5e6))
        h_l = jnp.clip(jnp.expand_dims(h_l, 0), getattr(fluid_l, "h_min", 1e3), getattr(fluid_l, "h_max", 5e6))

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # d_min_b=5e-5 matches nozzlex (nozzle_two_fluid_one_component_autonomous).
        ai, d = get_interfacial_area(area_law, area_params)(ag, alpha_b=alpha_b, alpha_d=alpha_d,
                                Nb=Nb, Nd=Nd, d_min_b=5e-6, ai_scale=ai_scale)
        # ai, d = 100, 5e-5
        # ── Drag force on gas per unit volume (sign: F > 0 accelerates gas) ─────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]   
        mu_l = stl_ex["viscosity"][0]
        # Drag force per unit volume: F > 0 means force on gas in direction of u_l - u_g
        Fv_g = -calculate_drag_force_one_component(
            ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
            alpha_b=alpha_b, alpha_d=alpha_d, blend=regime_blend,
            drag_bubbly=drag_bubbly, drag_droplet=drag_droplet,
        )

        # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
        du_abs = jnp.abs(u_g - u_l)
        Nu_g = 12 if nu_g_int is None else float(nu_g_int)

        if regime_htc:
            # Regime-dependent HTC: bubbly (α ≤ alpha_b) identical to the NICFD
            # closure below, droplet (α ≥ alpha_d) mirrored with vapor continuous,
            # linear blend in between (as done for the drag).
            htc_g, htc_l = calculate_heat_transfer_one_component(
                ag,
                {"rho": stg_ex["rho"][0], "mu": stg_ex["mu"][0],
                 "k": stg_ex["k"][0], "cp": stg_ex["cp"][0]},
                {"rho": stl_ex["rho"][0], "mu": stl_ex["mu"][0],
                 "k": stl_ex["k"][0], "cp": stl_ex["cp"][0]},
                u_g, u_l, d,
                Nu1_int=Nu_g, alpha_b=alpha_b, alpha_d=alpha_d, blend=regime_blend,
                slip_u0=slip_u0,
            )
        else:
            # NICFD closure: bubbly-regime HTC at every α.
            # Gas-phase HTC: continuous phase depends on flow regime (use liquid as
            # continuous in bubbly, gas in droplet; transition blended).
            # For simplicity use liquid-continuous HTC for gas bubbles.
            # htc_g = ranz_marshall_htc_two_fluid(
            #     {k: v[0] for k, v in stl_ex.items()},   # liquid as continuous
            #     du_abs, d,
            # )

            htc_g = (k_g * Nu_g) / d

            htc_l = ranz_marshall_htc_two_fluid(
                {                                         # liquid as continuous (bubbly regime)
                    "rho": stl_ex["rho"][0],             # FluidState alias → density
                    "mu":  stl_ex["mu"][0],              # FluidState alias → viscosity
                    "k":   stl_ex["k"][0],               # FluidState alias → conductivity
                    "cp":  stl_ex["cp"][0],              # FluidState alias → isobaric_heat_capacity
                },
                du_abs, d, slip_u0,
            )

        htc = 1 / (1/htc_g + 1/htc_l)

        # # ── Interfacial heat transfer rates ──────────────────────────────────────
        Qvi = htc * ai * (T_l - T_g)    # heat from interface to gas [W/m^3]
        Qli = - Qvi                     # heat from interface to liquid [W/m^3]

        # # ── Mass transfer (evaporation/condensation) ─────────────────────────────
        sigma_Mg = 0.0

    # ── Interface kinematics ─────────────────────────────────────────────────
    # Donor/upwind exchanged-mass velocity: the transferred mass carries the
    # velocity of the phase it LEAVES (evaporation σ_Mg>0: from liquid → u_l;
    # condensation σ_Mg<0: from gas → u_g).  This keeps the entropy slip term
    # ½(u_ex - u_i)² ≥ 0 unambiguous (RELAP5/CATHARE/TRACE convention).
    u_mix2 = (ag * rg * u_g + al * rl * u_l) / (ag * rg + al * rl)  # mass-weighted mixture
    u_ex   = jax.lax.cond(
        sigma_Mg >= 0.0,
        lambda _: u_l,       # evaporation: mass leaves the liquid phase
        lambda _: u_g,       # condensation: mass leaves the gas phase
        operand=None,
    )
    u_int  = u_mix2          # mass-weighted mixture interface velocity (viscous-work)
    # Enthalpy of transferred mass: saturation enthalpy of appearing phase

    # TODO: check the source terms implementation and compare it with nozzlex and with thee sources in Stadtke
    # TODO: check the sign of u_ex and u_v
    # TODO: h_ex is different beetween the phases. The difference is in the fact that theee phase appears at saturatio condition but on the liquid side it appears as sat liquid, onthe gas side as saturated gas
    # TODO: there should be the velocity differencee at the denominator of the mass source term?

    return TwoFluidClosure(
        sigma_Mg=sigma_Mg,
        sigma_Qg=Qvi,
        sigma_Ql=Qli,
        Fv_g=Fv_g,
        F_g=F_g,
        F_l=F_l,
        Q_g=Q_g,
        Q_l=Q_l,
        u_ex=u_ex,
        u_int=u_int,
        h_ex_g=h_sat_g,
        h_ex_l=h_sat_l,
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
    sigma_Ql = closure.sigma_Ql
    Fv_g     = closure.Fv_g
    F_g      = closure.F_g
    F_l      = closure.F_l
    Q_g      = closure.Q_g
    Q_l      = closure.Q_l
    u_ex     = closure.u_ex
    u_int    = closure.u_int
    h_ex_g   = closure.h_ex_g
    h_ex_l   = closure.h_ex_l
    dlnA_dx  = closure.dlnA_dx

    sigma_Ml = -sigma_Mg
    Fv_l     = -Fv_g

    # ── Entropy sources T_i * sigma_i^S  (Städtke eq. 4.50) ─────────────────
    Tg_sSg = (sigma_Mg * ((h_ex_g - h_g) + 0.5 * (u_ex - u_g)**2)
              + Fv_g  * (u_int - u_g)
              + sigma_Qg + Q_g)
    Tl_sSl = (sigma_Ml * ((h_ex_l - h_l) + 0.5 * (u_ex - u_l)**2)
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

    # Correction terms. Not present in the book but rederived from the primitve equations!
    u_bar = ag * u_g + al * u_l
    Ds_2 = Ds_2 + (rg + rl) * al * rl * du * u_bar * dlnA_dx / rh2
    Ds_3 = Ds_3 - (rg + rl) * ag * rg * du * u_bar * dlnA_dx / rh2

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


def D_VE_algebric(U_h, cq, closure, eos_g, eos_l, k):

    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g     = closure.Fv_g
    u_ex     = closure.u_ex
    u_int    = closure.u_int
    h_ex_g     = closure.h_ex_g
    h_ex_l     = closure.h_ex_l

    sigma_Ml = -sigma_Mg
    Fv_l     = -Fv_g

    rg  = eos_g.rho;  T_g = eos_g.T
    rl  = eos_l.rho;  T_l = eos_l.T

    rho_mix = cq.rho_mix;  rh2 = cq.rh2;  rg_hat = cq.rg_hat;  rl_hat = cq.rl_hat
    a0sq  = cq.a0sq;   du  = cq.du;   G_pp   = cq.G_pp;    G_uu   = cq.G_uu;  G_aa = cq.G_aa


    A_common = k * rho_mix * (ag * rg * u_l + al * rl * u_g) - al * ag * (rl**2 - rg**2) * (u_g - u_l)
    density_term = rg * rl / rh2 - 1

    # ── Building D_VE ───────────────────────────────────────────────────────
    D1 = sigma_Mg
    D2 = sigma_Ml
    D3 = rg * rl / rh2 * (Fv_g + sigma_Mg * u_ex) + sigma_Mg / rh2 * A_common
    D4 = rg * rl / rh2 * (Fv_l + sigma_Ml * u_ex) + sigma_Ml / rh2 * A_common
    D5 = sigma_Mg * (h_ex_g + 0.5 * u_ex ** 2 + u_ex * u_g * density_term + u_g * A_common / rh2) + Fv_g * (u_int + u_g * density_term) + sigma_Qg
    D6 = sigma_Ml * (h_ex_l + 0.5 * u_ex ** 2 + u_ex * u_l * density_term + u_l * A_common / rh2) + Fv_l * (u_int + u_l * density_term) + sigma_Ql

    # NOTE: area-change is a SEPARATE source (D_VE_area, below) so it can be
    # toggled independently; wall friction still TODO.

    return jnp.array([D1, D2, D3, D4, D5, D6])

# TODO: delete this if old and not used!
def D_VE_algebric_two_component(U_h, cq, closure, eos_g, eos_l, k):
    """
    For cases where there is no mass transfer 
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g     = closure.Fv_g
    u_int    = closure.u_int

    Fv_l     = -Fv_g

    rg  = eos_g.rho;
    rl  = eos_l.rho;

    rh2 = cq.rh2;

    density_term = rg * rl / rh2 - 1

    # ── Building D_VE ───────────────────────────────────────────────────────
    D3 = rg * rl / rh2 * Fv_g 
    D4 = rg * rl / rh2 * Fv_l 
    D5 = Fv_g * (u_int + u_g * density_term) + sigma_Qg
    D6 = Fv_l * (u_int + u_l * density_term) + sigma_Ql

    # NOTE: area-change is a SEPARATE source (D_VE_area, below) so it can be
    # toggled independently; wall friction still TODO.

    return jnp.array([0.0, 0.0, D3, D4, D5, D6])



# ---------------------------------------------------------------------------
# D_VE_area  — quasi-1D area-change source (Form-A "leftover"), separate & toggleable
# ---------------------------------------------------------------------------

def D_VE_area(U_h, cq, closure, eos_g, eos_l):
    r"""
    Quasi-1D area-change source in conservative variables q =
    (alpha_g*rho_g, alpha_l*rho_l, mom_g, mom_l, E_g, E_l).

    The solver area-weights the flux divergence (Af * F_hat), which ALREADY
    supplies the mass and convective-momentum/energy geometric terms
    (-F * G).  The only pieces the area-weighting does NOT produce are the
    pressure-on-wall force and the interfacial p-V work, so this "leftover" is
    all that must be added:

        D_area = [ 0, 0, ag*p*G, al*p*G, -p*Dalpha, +p*Dalpha ]

    with the Städtke void-fraction area source
        Dalpha = ag*al * (a0^2/rho_s) * (rl*ul/ag^2 - rg*ug/al^2) * G
    and the logarithmic area gradient
        G = (1/A) dA/dx = closure.dlnA_dx.

    Every term is ∝ G, so this vanishes identically for constant area (pipe).
    In the single-phase limit (ag -> 1, one phase) the momentum term reduces to
    the single-phase p*G wall force and Dalpha -> 0, so the energy terms vanish —
    matching fvm.py's [0, p*G, 0] geometric source.

    Kept SEPARATE from D_VE_algebric so the area source can be enabled/disabled
    independently (e.g. for the ghost-phase vs. single-phase verification).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    G = closure.dlnA_dx

    rg = eos_g.rho;  ag_sq = eos_g.a**2
    rl = eos_l.rho;  al_sq = eos_l.a**2
    a0sq  = cq.a0sq
    rho_s = cq.rho_s

    D_alpha = ag * al * (a0sq / rho_s) * (rl * u_l / ag_sq - rg * u_g / al_sq) * G

    z  = jnp.zeros_like(p)
    D3 = ag * p * G
    D4 = al * p * G
    D5 = -p * D_alpha
    D6 =  p * D_alpha
    # D5 = z
    # D6 = z

    return jnp.array([z, z, D3, D4, D5, D6])
