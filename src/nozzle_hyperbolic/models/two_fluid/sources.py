# Port reference: origin/alberto-branch faafff704e54fd866abff95fcb88fc0a5217405c.
# Intentional safety changes are documented in docs/notes/two_fluid_migration.md.
"""Alberto's interphase exchange and separate quasi-1D area source.

``two_fluid_closure`` dispatches the model-local constitutive laws. Production
uses ``D_VE_algebric`` or ``D_VE_algebric_two_component``, plus ``D_VE_area``.
The independent ``D_s_primitive`` / ``D_VE_conservative`` transformation remains
for source-identity verification. Alberto's notation, component order and
pressure-work derivation blocks are retained. See docs/notes/two_fluid_migration.md.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxprop as jxp

from .matrices import PhaseProps, CommonQty, J_matrix, Phi_inv_matrix

# Frozen Alberto constitutive laws, packaged without sys.path mutation.
from .closures import (
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
    """Cell-local physical exchange quantities in Alberto's phase convention.

    Attributes
    ----------
    sigma_Mg : scalar
        Gas mass source in kg/(m**3 s), positive for evaporation.
    sigma_Qg, sigma_Ql : scalar
        Interfacial heat into gas and liquid, respectively, in W/m**3.
    Fv_g : scalar
        Signed axial drag force on gas in N/m**3; liquid receives its negative.
    F_g, F_l : scalar
        External phase forces in N/m**3, used by verification assembly only.
    Q_g, Q_l : scalar
        External phase heat in W/m**3, used by verification assembly only.
    u_ex : scalar
        Donor velocity of transferred mass in m/s.
    u_int : scalar
        Mass-weighted interface velocity in m/s.
    h_ex_g, h_ex_l : scalar
        Separate exchanged phase enthalpies in J/kg.
    dlnA_dx : scalar
        Log-area gradient in 1/m for density-form verification. The integrated
        area kernel instead receives an exact face-area difference in m**2.
    """

    sigma_Mg: jnp.ndarray  # gas mass source  [kg/(m^3 s)]
    sigma_Qg: jnp.ndarray  # interfacial heat to gas   [W/m^3]
    sigma_Ql: jnp.ndarray  # interfacial heat to liquid   [W/m^3]
    Fv_g: jnp.ndarray  # interfacial drag on gas   [N/m^3]
    F_g: jnp.ndarray  # external body force on gas [N/m^3]
    F_l: jnp.ndarray  # external body force on liquid [N/m^3]
    Q_g: jnp.ndarray  # wall/external heat to gas  [W/m^3]
    Q_l: jnp.ndarray  # wall/external heat to liquid [W/m^3]
    u_ex: jnp.ndarray  # velocity of transferred mass [m/s]
    u_int: jnp.ndarray  # interface velocity [m/s]
    h_ex_g: jnp.ndarray  # gas exchanged enthalpy [J/kg]
    h_ex_l: jnp.ndarray  # liquid exchanged enthalpy [J/kg]
    dlnA_dx: jnp.ndarray  # (1/A) dA/dx — log area derivative [1/m]


# ---------------------------------------------------------------------------
# Closure helper
# ---------------------------------------------------------------------------


def evaporation_weight(
    h_l,
    T_l,
    h_sat_l,
    T_sat,
    ag,
    gate_dT=0.5,
    gate_dh=2.0e3,
    dh_gate=1.0,
    dT_gate=0.001,
    p=None,
    p_max=None,
):
    """Return the binary phase-change gate shared with the saturation-energy pin.

    Parameters
    ----------
    h_l, h_sat_l : scalar
        Liquid and saturated-liquid enthalpies in J/kg.
    T_l, T_sat : scalar
        Liquid and saturation temperatures in K.
    ag : scalar
        Gas volume fraction.
    gate_dT, gate_dh : float, optional
        Historical smooth-gate widths; retained arguments, unused by the binary law.
    dh_gate, dT_gate : float, optional
        Strict superheat thresholds in J/kg and K.
    p : scalar or None, optional
        Pressure in Pa, required when ``p_max`` is supplied.
    p_max : float or None, optional
        Pressure cutoff in Pa; the gate is zero at and above it.

    Returns
    -------
    scalar
        One for liquid superheat in both variables or gas fraction above 0.05;
        zero otherwise, with the optional pressure cutoff taking precedence.

    Notes
    -----
    The shared gate releases the saturation-energy pin exactly where mass
    exchange starts. The gas-availability clause also permits condensation
    in subcooled regions containing a real vapor phase. A pressure cutoff
    is appropriate only for ghost vapor; it would pin a real vapor phase too.
    The binary gate has no derivative across its switching surface.
    """
    if p_max is not None:
        w = evaporation_weight(
            h_l,
            T_l,
            h_sat_l,
            T_sat,
            ag,
            gate_dT=gate_dT,
            gate_dh=gate_dh,
            dh_gate=dh_gate,
            dT_gate=dT_gate,
        )
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
    return jnp.where(
        (h_l > h_sat_l + dh_gate) & (T_l > T_sat + dT_gate) | (ag > 5e-2), 1.0, 0.0
    )


def two_fluid_closure(
    U_h,
    eos_g: PhaseProps,
    eos_l: PhaseProps,
    fluid_g,  # FluidBicubic — for viscosity/conductivity lookup
    fluid_l,  # FluidBicubic
    T_sat,  # saturation temperature at current pressure [K]
    h_sat_g,  # saturation vapour enthalpy [J/kg]
    h_sat_l,  # saturation liquid enthalpy [J/kg]
    dlnA_dx=0.0,  # area derivative [1/m]
    F_g=0.0,  # gravity or other external force on gas [N/m^3]
    F_l=0.0,  # gravity or other external force on liquid [N/m^3]
    Q_g=0.0,  # wall heat to gas [W/m^3]
    Q_l=0.0,  # wall heat to liquid [W/m^3]
    Nb=1e10,  # bubble/droplet number density [1/m^3]
    Nd=1e10,
    alpha_b=0.3,  # TA-BD transition void fractions
    alpha_d=0.7,
    mass_transfer=True,
    regime_htc=False,  # True → regime-dependent HTC (bubbly/transition/droplet);
    # False → NICFD closure (bubbly-regime HTC at every α)
    regime_blend="linear",  # transition alpha_b..alpha_d of the drag and of the
    # regime HTC: 'linear' | 'smoothstep' (C1, regime_weight)
    gate_dT=1e-3,  # historical smooth-gate width [K]; binary law ignores it
    gate_dh=0.5,  # historical smooth-gate width [J/kg]; binary law ignores it
    p_mt_max=None,  # [Pa] no phase change at p >= p_mt_max (see
    # evaporation_weight); None → no cut-off
    area_law="polynomial",  # interfacial area law: original | polynomial | parabola
    # (closures.INTERFACIAL_AREA_LAWS, including segments)
    area_params=None,  # dict of keyword overrides for that law, e.g. {"d_min": 6.5e-5}
    ai_scale=1.0,  # correction of the law in this cell (field inversion, beta):
    # ai -> ai_scale * ai, and d follows from the scaled ai
    smooth_slip=False,  # True → Ranz-Marshall Re from the smooth effective slip
    # sqrt(du² + smooth_slip_u0²); False → |du| (cusp at du = 0).
    smooth_slip_u0=2.0,  # [m/s] optional effective-slip scale
    nu_g_int=None,  # internal Nusselt number of the vapour (bubbles, htc_g = k_g Nu / d);
    # None → 120 with mass transfer, 12 without (the earlier constants)
    htc_liquid_bubbly="ranz_marshall",  # liquid-side HTC, bubbly regime: ranz_marshall | jakob | max | liao
    htc_liquid_droplet="internal",  # liquid-side HTC, droplet regime: internal | jakob | max
    drag_bubbly="schiller_naumann",  # drag coefficient, bubbly regime: schiller_naumann | ishii_zuber
    drag_droplet="schiller_naumann",  # drag coefficient, droplet regime: schiller_naumann | ishii_zuber
) -> TwoFluidClosure:
    """Assemble interphase exchange using Alberto's phase and sign conventions.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.
    fluid_g, fluid_l : object
        Phase fluids providing transport properties through JAXProp.
    T_sat : scalar
        Saturation temperature in K, supplied by the caller.
    h_sat_g, h_sat_l : scalar
        Saturated gas/liquid enthalpies in J/kg, supplied by the caller.
    dlnA_dx : scalar, optional
        Logarithmic area derivative in 1/m, stored for verification assembly.
    F_g, F_l : scalar, optional
        External phase forces in N/m**3, stored for verification assembly.
    Q_g, Q_l : scalar, optional
        External phase heat in W/m**3, stored for verification assembly.
    Nb, Nd : float, optional
        Bubble and droplet number densities in 1/m**3 for the original area law.
    alpha_b, alpha_d : float, optional
        Bubbly/droplet gas-fraction thresholds.
    mass_transfer : bool, optional
        Enable gated phase change; otherwise exchange heat directly between phases.
    regime_htc : bool, optional
        Mirror the HTC laws in the droplet regime; false uses bubbly HTCs everywhere.
    regime_blend : {'linear', 'smoothstep'}, optional
        Weight for regime-dependent drag and heat transfer.
    gate_dT, gate_dh : float, optional
        Retained historical smooth-gate widths; the active gate is binary.
    p_mt_max : float or None, optional
        Phase-change pressure cutoff in Pa.
    area_law : str, optional
        Registered interfacial-area correlation.
    area_params : dict or None, optional
        Correlation overrides; explicit law parameters take precedence over defaults.
    ai_scale : scalar, optional
        Area multiplier, applied before deriving the dispersed diameter.
    smooth_slip : bool, optional
        Enable the smooth effective slip in Ranz-Marshall HTCs.
    smooth_slip_u0 : float, optional
        Effective-slip scale in m/s.
    nu_g_int : float or None, optional
        Internal gas Nusselt number; None retains 120 with phase change, 12 without.
    htc_liquid_bubbly, htc_liquid_droplet : str, optional
        Liquid HTC selections; Jakob/Liao options require mass transfer.
    drag_bubbly, drag_droplet : str, optional
        Drag laws for bubbly and droplet regimes.

    Returns
    -------
    TwoFluidClosure
        Cell-local mass, heat and force exchange, interface velocity and enthalpies.

    Notes
    -----
    Positive sigma_Mg means evaporation; sigma_Ml=-sigma_Mg.
    The heat sources and mass source receive the same gate so their interface
    energy balance is preserved. Gas drag is the negative of the signed
    resistance returned by the drag correlation. Transferred mass uses the
    donor-phase velocity; exchanged gas/liquid enthalpies are their separate
    saturation enthalpies. Interface velocity is mass weighted.

    With no mass transfer, the two HTCs act in series and heat sources are
    equal and opposite. With phase change, each phase exchanges heat with
    the saturated interface. Saturation lookup remains outside this kernel.

    F_g/F_l/Q_g/Q_l are retained for the verification transformation only.
    The active algebraic source and ClosureConfig do not expose general
    external force or wall-heat support; no wall-friction law is supplied.
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag
    slip_u0 = float(smooth_slip_u0) if smooth_slip else 0.0
    if (
        htc_liquid_bubbly not in LIQUID_HTC_BUBBLY
        or htc_liquid_droplet not in LIQUID_HTC_DROPLET
    ):
        raise ValueError(
            f"htc_liquid_bubbly in {LIQUID_HTC_BUBBLY}, htc_liquid_droplet in "
            f"{LIQUID_HTC_DROPLET}; got {htc_liquid_bubbly!r}, {htc_liquid_droplet!r}"
        )
    if not mass_transfer and (htc_liquid_bubbly, htc_liquid_droplet) != (
        "ranz_marshall",
        "internal",
    ):
        raise ValueError("the Jakob liquid HTC needs mass_transfer=True")
    if drag_bubbly not in DRAG_LAWS or drag_droplet not in DRAG_LAWS:
        raise ValueError(
            f"drag_bubbly / drag_droplet in {DRAG_LAWS}; got {drag_bubbly!r}, {drag_droplet!r}"
        )

    rg = eos_g.rho
    T_g = eos_g.T
    rl = eos_l.rho
    T_l = eos_l.T

    # --- Shared phase-change gate (liquid superheat OR available vapor) -----
    # The shared binary gate requires liquid superheat in both h and T, or
    # an appreciable vapor fraction. The paired superheat criteria avoid
    # activation from only one inconsistent table/saturation quantity.
    # Gas availability also permits condensation after a recompression shock;
    # do not replace this with a separate gas-subcooling gate that can drain
    # the pre-flashing ghost phase before liquid superheat develops.
    if mass_transfer == True:
        # This is the same binary gate used by the optional energy pin.
        # Keep their activation regions aligned so the pin cannot block onset.
        # A hard threshold has no derivative across its switching surface;
        # changing to smooth weights would change the selected equations.
        w_evap = evaporation_weight(
            h_l,
            T_l,
            h_sat_l,
            T_sat,
            ag,
            gate_dT=gate_dT,
            gate_dh=gate_dh,
            p=p,
            p_max=p_mt_max,
        )
        w_mt = w_evap
        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr = jnp.expand_dims(p, 0)

        h_g = jnp.expand_dims(h_g, 0)
        h_l = jnp.expand_dims(h_l, 0)

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # Keep Alberto's 5e-6 m bubble floor in the original area-law path.
        ai, d = get_interfacial_area(area_law)(
            ag,
            **(
                dict(
                    alpha_b=alpha_b,
                    alpha_d=alpha_d,
                    Nb=Nb,
                    Nd=Nd,
                    d_min_b=5e-6,
                    ai_scale=ai_scale,
                )
                | (area_params or {})
            ),
        )
        # ── Signed axial drag force on gas per unit volume ────────────────────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]
        mu_l = stl_ex["viscosity"][0]
        # Negate the resistance: Fv_g opposes u_g-u_l in the axial direction.
        Fv_g = -calculate_drag_force_one_component(
            ag,
            al,
            rg,
            rl,
            mu_g,
            mu_l,
            u_g,
            u_l,
            d,
            alpha_b=alpha_b,
            alpha_d=alpha_d,
            blend=regime_blend,
            drag_bubbly=drag_bubbly,
            drag_droplet=drag_droplet,
        )

        # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
        du_abs = jnp.abs(u_g - u_l)
        Nu_g, Nu_l = (120.0 if nu_g_int is None else float(nu_g_int)), 12.0
        ja_l = (
            rl * stl_ex["cp"][0] * (T_l - T_sat) / (rg * (h_sat_g - h_sat_l))
        )  # liquid Jakob number

        if regime_htc:
            # Regime-dependent HTC: bubbly (α ≤ alpha_b) identical to the NICFD
            # closure below, droplet (α ≥ alpha_d) mirrored with vapor continuous,
            # linear blend in between (as done for the drag).
            htc_g, htc_l = calculate_heat_transfer_one_component(
                ag,
                {
                    "rho": stg_ex["rho"][0],
                    "mu": stg_ex["mu"][0],
                    "k": stg_ex["k"][0],
                    "cp": stg_ex["cp"][0],
                },
                {
                    "rho": stl_ex["rho"][0],
                    "mu": stl_ex["mu"][0],
                    "k": stl_ex["k"][0],
                    "cp": stl_ex["cp"][0],
                },
                u_g,
                u_l,
                d,
                Nu1_int=Nu_g,
                Nu2_int=Nu_l,
                alpha_b=alpha_b,
                alpha_d=alpha_d,
                blend=regime_blend,
                slip_u0=slip_u0,
                ja=ja_l,
                liquid_bubbly=htc_liquid_bubbly,
                liquid_droplet=htc_liquid_droplet,
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
                {  # liquid as continuous (bubbly regime)
                    "rho": stl_ex["rho"][0],  # FluidState alias → density
                    "mu": stl_ex["mu"][0],  # FluidState alias → viscosity
                    "k": stl_ex["k"][0],  # FluidState alias → conductivity
                    "cp": stl_ex["cp"][0],  # FluidState alias → isobaric_heat_capacity
                },
                du_abs,
                d,
                slip_u0,
            )
            u_pe = jnp.sqrt(du_abs**2 + slip_u0**2) if slip_u0 > 0.0 else du_abs
            pe_l = u_pe * d * stl_ex["rho"][0] * stl_ex["cp"][0] / stl_ex["k"][0]
            htc_l = select_liquid_htc(
                htc_liquid_bubbly,
                htc_l,
                stl_ex["k"][0] * jakob_nusselt(ja_l) / d,
                stl_ex["k"][0] * liao_nusselt(ja_l, pe_l) / d,
            )

        # Assemble interface heat first, then obtain mass exchange from
        # the same balance. Apply one common gate to all three sources.
        _Qvi = htc_g * ai * (T_sat - T_g)
        _Qli = htc_l * ai * (T_sat - T_l)

        _sMg = calculate_mass_transfer_lv(
            T_l,
            T_g,
            T_sat,
            htc_l,
            htc_g,
            ai,
            h_sat_l,
            h_sat_g,
            h_l,
            h_g,
            _Qvi,
            _Qli,
        )
        Qvi = w_mt * _Qvi
        Qli = w_mt * _Qli
        sigma_Mg = w_mt * _sMg
    else:  # No mass transfer, just heat and drag

        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr = jnp.expand_dims(p, 0)

        h_g = jnp.expand_dims(h_g, 0)
        h_l = jnp.expand_dims(h_l, 0)

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # Keep Alberto's 5e-6 m bubble floor in the original area-law path.
        ai, d = get_interfacial_area(area_law)(
            ag,
            **(
                dict(
                    alpha_b=alpha_b,
                    alpha_d=alpha_d,
                    Nb=Nb,
                    Nd=Nd,
                    d_min_b=5e-6,
                    ai_scale=ai_scale,
                )
                | (area_params or {})
            ),
        )
        # ── Signed axial drag force on gas per unit volume ────────────────────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]
        mu_l = stl_ex["viscosity"][0]
        # Negate the resistance: Fv_g opposes u_g-u_l in the axial direction.
        Fv_g = -calculate_drag_force_one_component(
            ag,
            al,
            rg,
            rl,
            mu_g,
            mu_l,
            u_g,
            u_l,
            d,
            alpha_b=alpha_b,
            alpha_d=alpha_d,
            blend=regime_blend,
            drag_bubbly=drag_bubbly,
            drag_droplet=drag_droplet,
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
                {
                    "rho": stg_ex["rho"][0],
                    "mu": stg_ex["mu"][0],
                    "k": stg_ex["k"][0],
                    "cp": stg_ex["cp"][0],
                },
                {
                    "rho": stl_ex["rho"][0],
                    "mu": stl_ex["mu"][0],
                    "k": stl_ex["k"][0],
                    "cp": stl_ex["cp"][0],
                },
                u_g,
                u_l,
                d,
                Nu1_int=Nu_g,
                alpha_b=alpha_b,
                alpha_d=alpha_d,
                blend=regime_blend,
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
                {  # liquid as continuous (bubbly regime)
                    "rho": stl_ex["rho"][0],  # FluidState alias → density
                    "mu": stl_ex["mu"][0],  # FluidState alias → viscosity
                    "k": stl_ex["k"][0],  # FluidState alias → conductivity
                    "cp": stl_ex["cp"][0],  # FluidState alias → isobaric_heat_capacity
                },
                du_abs,
                d,
                slip_u0,
            )

        htc = 1 / (1 / htc_g + 1 / htc_l)

        # # ── Interfacial heat transfer rates ──────────────────────────────────────
        Qvi = htc * ai * (T_l - T_g)  # heat from interface to gas [W/m^3]
        Qli = -Qvi  # heat from interface to liquid [W/m^3]

        # # ── Mass transfer (evaporation/condensation) ─────────────────────────────
        sigma_Mg = 0.0

    # ── Interface kinematics ─────────────────────────────────────────────────
    # Donor/upwind exchanged-mass velocity: the transferred mass carries the
    # velocity of the phase it LEAVES (evaporation σ_Mg>0: from liquid → u_l;
    # condensation σ_Mg<0: from gas → u_g).  This keeps the entropy slip term
    # ½(u_ex - u_i)² ≥ 0 unambiguous (RELAP5/CATHARE/TRACE convention).
    u_mix2 = (ag * rg * u_g + al * rl * u_l) / (
        ag * rg + al * rl
    )  # mass-weighted mixture
    u_ex = jax.lax.cond(
        sigma_Mg >= 0.0,
        lambda _: u_l,  # evaporation: mass leaves the liquid phase
        lambda _: u_g,  # condensation: mass leaves the gas phase
        operand=None,
    )
    u_int = u_mix2  # mass-weighted mixture interface velocity (viscous-work)
    # Enthalpy of transferred mass: saturation enthalpy of appearing phase

    # Donor velocity and separate saturated phase enthalpies follow the
    # retained closure policy; broader physical validation remains necessary.

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
# NB: used only for verification
def D_s_primitive(
    U_h,
    eos_g: PhaseProps,
    eos_l: PhaseProps,
    cq: CommonQty,
    closure: TwoFluidClosure,
    k,
) -> jnp.ndarray:
    """Evaluate the entropy-frame source retained for independent identity checks.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.
    closure : TwoFluidClosure
        Cell-local exchange quantities in SI units.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.
    k : float
        Nonnegative dimensionless virtual-mass coefficient.

    Returns
    -------
    jax.Array, shape (6,)
        Primitive entropy-frame source from Alberto's Table B.46 transformation.

    Notes
    -----
    Primitive entropy-frame source vector D_s (6,).

    Implements Städtke Table B.46 with the k → k*rho correction applied
    throughout rows 2 and 3 (every bare 'k' next to a density in B.46
    must be read as k*rho_mix).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho
    ag_sq = eos_g.a**2
    T_g = eos_g.T
    rl = eos_l.rho
    al_sq = eos_l.a**2
    T_l = eos_l.T
    rg_h = eos_g.rho_h
    rl_h = eos_l.rho_h

    rho_mix = cq.rho_mix
    rho_s = cq.rho_s
    rh2 = cq.rh2
    a0sq = cq.a0sq
    du = cq.du

    # k*rho_mix (not stored in cq; derive from rg_hat = rg + k*rho_mix)
    k_rho = cq.rg_hat - rg  # = k * rho_mix

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g = closure.Fv_g
    F_g = closure.F_g
    F_l = closure.F_l
    Q_g = closure.Q_g
    Q_l = closure.Q_l
    u_ex = closure.u_ex
    u_int = closure.u_int
    h_ex_g = closure.h_ex_g
    h_ex_l = closure.h_ex_l
    dlnA_dx = closure.dlnA_dx

    sigma_Ml = -sigma_Mg
    Fv_l = -Fv_g

    # ── Entropy sources T_i * sigma_i^S  (Städtke eq. 4.50) ─────────────────
    Tg_sSg = (
        sigma_Mg * ((h_ex_g - h_g) + 0.5 * (u_ex - u_g) ** 2)
        + Fv_g * (u_int - u_g)
        + sigma_Qg
        + Q_g
    )
    Tl_sSl = (
        sigma_Ml * ((h_ex_l - h_l) + 0.5 * (u_ex - u_l) ** 2)
        + Fv_l * (u_int - u_l)
        + sigma_Ql
        + Q_l
    )
    sSg = Tg_sSg / T_g  # sigma_g^S
    sSl = Tl_sSl / T_l  # sigma_l^S

    # ── Recurring mass+entropy group (Appendix A) ────────────────────────────
    Mg = sigma_Mg - (T_g * rg_h / rg) * sSg
    Ml = sigma_Ml - (T_l * rl_h / rl) * sSl

    # ── Row 1: pressure ──────────────────────────────────────────────────────
    Ds_1 = (
        rl * a0sq / rho_s * Mg
        + rg * a0sq / rho_s * Ml
        - rl * rg / rho_s * (ag * u_g + al * u_l) * a0sq * dlnA_dx
    )

    # ── Row 2: gas velocity (k → k*rho correction applied) ───────────────────
    Ds_2 = (
        rl / (ag * rh2) * (Fv_g + sigma_Mg * (u_ex - u_g))
        - sigma_Mg * (k_rho * rg + al * (rl**2 - rg**2)) / (rg * rh2) * du
    )
    #   + (F_g * (rl + ag * k_rho) + F_l * ag * k_rho) / (ag * rh2))

    # ── Row 3: liquid velocity (k → k*rho correction applied) ────────────────
    Ds_3 = (
        -rg / (al * rh2) * (Fv_g + sigma_Mg * (u_ex - u_l))
        - sigma_Mg * (k_rho * rl + ag * (rg**2 - rl**2)) / (rl * rh2) * du
    )
    #    + (F_g * al * k_rho + F_l * (rg + al * k_rho)) / (al * rh2))

    # # Correction terms. Not present in the book but rederived from the primitve equations!

    # ── Row 4: void fraction ─────────────────────────────────────────────────
    Ds_4 = (
        al * a0sq / (rho_s * al_sq) * Mg
        - ag * a0sq / (rho_s * ag_sq) * Ml
        + al * ag * a0sq / rho_s * (rl * u_l / ag_sq - rg * u_g / al_sq) * dlnA_dx
    )

    # ── Rows 5, 6: entropy ───────────────────────────────────────────────────
    Ds_5 = sSg / (ag * rg)
    Ds_6 = sSl / (al * rl)

    return jnp.array([Ds_1, Ds_2, Ds_3, Ds_4, Ds_5, Ds_6])


# ---------------------------------------------------------------------------
# D_VE_conservative  — D_VE = J Phi^{-1} D_s
# ---------------------------------------------------------------------------
# NB: used only for verification
def D_VE_conservative(
    U_h,
    eos_g: PhaseProps,
    eos_l: PhaseProps,
    cq: CommonQty,
    closure: TwoFluidClosure,
    k,
) -> jnp.ndarray:
    """Transform the verification source to conservative total-energy variables.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.
    closure : TwoFluidClosure
        Cell-local exchange quantities in SI units.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.
    k : float
        Nonnegative dimensionless virtual-mass coefficient.

    Returns
    -------
    jax.Array, shape (6,)
        ``J @ Phi_inv @ D_s_primitive`` for verification against algebraic assembly.

    Notes
    -----
    This independent route is exercised by matrix/source identity tests;
    production interphase exchange uses the algebraic assembly.
    """
    D_s = D_s_primitive(U_h, eos_g, eos_l, cq, closure, k)

    rg = eos_g.rho
    T_g = eos_g.T
    rl = eos_l.rho
    T_l = eos_l.T

    # D_h is the entropy-to-enthalpy transformation below.

    # Phi^{-1}: identity on rows 0-3; rows 4, 5 mix pressure row (D_s[0]) with entropy rows
    D_h = D_s.at[4].set(D_s[0] / rg + T_g * D_s[4])
    D_h = D_h.at[5].set(D_s[0] / rl + T_l * D_s[5])

    # J @ D_h
    J = J_matrix(U_h, eos_g, eos_l)
    return J @ D_h


def D_VE_algebric(U_h, cq, closure, eos_g, eos_l, k):
    """Assemble the active six-row source with phase change and pressure work.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.
    closure : TwoFluidClosure
        Cell-local exchange quantities in SI units.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.
    k : float
        Nonnegative dimensionless virtual-mass coefficient.

    Returns
    -------
    jax.Array, shape (6,)
        Conservative source density before cell-volume integration.

    Notes
    -----
    The historical spelling 'algebric' is retained for Alberto's review.
    Gas and liquid energies include opposite ``p*D_alpha`` work. Area change
    is assembled separately by ``D_VE_area``. External F/Q fields in the
    closure container are not consumed by this algebraic assembly.
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g = closure.Fv_g
    u_ex = closure.u_ex
    u_int = closure.u_int
    h_ex_g = closure.h_ex_g
    h_ex_l = closure.h_ex_l

    sigma_Ml = -sigma_Mg
    Fv_l = -Fv_g

    rg = eos_g.rho
    T_g = eos_g.T
    rl = eos_l.rho
    T_l = eos_l.T

    rho_mix = cq.rho_mix
    rh2 = cq.rh2
    rg_hat = cq.rg_hat
    rl_hat = cq.rl_hat
    a0sq = cq.a0sq
    du = cq.du
    G_pp = cq.G_pp
    G_uu = cq.G_uu
    G_aa = cq.G_aa

    A_common = k * rho_mix * (ag * rg * u_l + al * rl * u_g) - al * ag * (
        rl**2 - rg**2
    ) * (u_g - u_l)
    density_term = rg * rl / rh2 - 1

    # ── Building D_VE ───────────────────────────────────────────────────────
    D1 = sigma_Mg
    D2 = sigma_Ml
    D3 = rg * rl / rh2 * (Fv_g + sigma_Mg * u_ex) + sigma_Mg / rh2 * A_common
    D4 = rg * rl / rh2 * (Fv_l + sigma_Ml * u_ex) + sigma_Ml / rh2 * A_common
    D5 = (
        sigma_Mg
        * (h_ex_g + 0.5 * u_ex**2 + u_ex * u_g * density_term + u_g * A_common / rh2)
        + Fv_g * (u_int + u_g * density_term)
        + sigma_Qg
    )
    D6 = (
        sigma_Ml
        * (h_ex_l + 0.5 * u_ex**2 + u_ex * u_l * density_term + u_l * A_common / rh2)
        + Fv_l * (u_int + u_l * density_term)
        + sigma_Ql
    )

    # ── Interfacial p·dα/dt work of the source-driven void-fraction change ──
    # (the area part of Dα is already in D_VE_area; this is the closure part,
    #  = row 4 of D_s_primitive at dlnA_dx = 0)
    ag_sq = eos_g.a**2
    al_sq = eos_l.a**2
    rho_s = cq.rho_s
    Tg_sSg = (
        sigma_Mg * ((h_ex_g - h_g) + 0.5 * (u_ex - u_g) ** 2)
        + Fv_g * (u_int - u_g)
        + sigma_Qg
    )
    Tl_sSl = (
        sigma_Ml * ((h_ex_l - h_l) + 0.5 * (u_ex - u_l) ** 2)
        + Fv_l * (u_int - u_l)
        + sigma_Ql
    )
    Mg = sigma_Mg - eos_g.rho_h / rg * Tg_sSg
    Ml = sigma_Ml - eos_l.rho_h / rl * Tl_sSl
    D_alpha = al * a0sq / (rho_s * al_sq) * Mg - ag * a0sq / (rho_s * ag_sq) * Ml
    D5 = D5 - p * D_alpha
    D6 = D6 + p * D_alpha
    # NOTE: area-change is a SEPARATE source (D_VE_area, below) so it can be
    # toggled independently; no wall-friction law is supplied.

    return jnp.array([D1, D2, D3, D4, D5, D6])


# Active no-mass-transfer branch selected by ClosureConfig.mass_transfer=False.
def D_VE_algebric_two_component(U_h, cq, closure, eos_g, eos_l, k):
    """Assemble active drag/heat sources when mass transfer is disabled.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.
    closure : TwoFluidClosure
        Cell-local exchange quantities in SI units.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.
    k : float
        Nonnegative dimensionless virtual-mass coefficient.

    Returns
    -------
    jax.Array, shape (6,)
        Conservative source density with zero gas/liquid mass sources.

    Notes
    -----
    This is the active ``mass_transfer=False`` path, not unused legacy code.
    The phase-energy rows retain opposite pressure work from the source-driven
    void change. Geometry is added separately; external F/Q fields are unused.
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g = closure.Fv_g
    u_int = closure.u_int

    Fv_l = -Fv_g

    rg = eos_g.rho
    rl = eos_l.rho

    rh2 = cq.rh2

    density_term = rg * rl / rh2 - 1

    # ── Building D_VE ───────────────────────────────────────────────────────
    D3 = rg * rl / rh2 * Fv_g
    D4 = rg * rl / rh2 * Fv_l
    D5 = Fv_g * (u_int + u_g * density_term) + sigma_Qg
    D6 = Fv_l * (u_int + u_l * density_term) + sigma_Ql

    p = U_h[0]
    ag_sq = eos_g.a**2
    al_sq = eos_l.a**2
    a0sq = cq.a0sq
    rho_s = cq.rho_s
    Mg = -eos_g.rho_h / rg * (Fv_g * (u_int - u_g) + sigma_Qg)
    Ml = -eos_l.rho_h / rl * (Fv_l * (u_int - u_l) + sigma_Ql)
    D_alpha = al * a0sq / (rho_s * al_sq) * Mg - ag * a0sq / (rho_s * ag_sq) * Ml
    D5 = D5 - p * D_alpha
    D6 = D6 + p * D_alpha

    # NOTE: area-change is a SEPARATE source (D_VE_area, below) so it can be
    # toggled independently; no wall-friction law is supplied.

    return jnp.array([0.0, 0.0, D3, D4, D5, D6])


# ---------------------------------------------------------------------------
# D_VE_area  — quasi-1D area-change source (Form-A "leftover"), separate & toggleable
# ---------------------------------------------------------------------------


def D_VE_area(U_h, cq, closure, eos_g, eos_l):
    """Evaluate Alberto's separate quasi-one-dimensional area source.

    Parameters
    ----------
    U_h : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.
    closure : object
        Provides ``dlnA_dx``: log-area gradient for density form, or exact
        face-area difference for the model's integrated form.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h``.

    Returns
    -------
    jax.Array, shape (6,)
        Area source in conservative ordering with opposite phase pressure work.

    Notes
    -----
    For the density kernel ``dlnA_dx=(1/A)*dA/dx``. The active model instead
    passes ``A_right-A_left`` to obtain the integrated source directly; it
    must not multiply that result by cell volume again. This pairs geometry
    with the exact face measures in the flux divergence.

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
    matching the single-phase nozzle's [0, p*G, 0] density-form source.

    Kept SEPARATE from D_VE_algebric so the area source can be enabled/disabled
    independently (e.g. for the ghost-phase vs. single-phase verification).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    G = closure.dlnA_dx

    rg = eos_g.rho
    ag_sq = eos_g.a**2
    rl = eos_l.rho
    al_sq = eos_l.a**2
    a0sq = cq.a0sq
    rho_s = cq.rho_s

    D_alpha = ag * al * (a0sq / rho_s) * (rl * u_l / ag_sq - rg * u_g / al_sq) * G

    z = jnp.zeros_like(p)
    D3 = ag * p * G
    D4 = al * p * G
    D5 = -p * D_alpha
    D6 = p * D_alpha
    return jnp.array([z, z, D3, D4, D5, D6])
