"""
Implicit Newton time-stepping for Städtke's two-fluid FVS solver.

Newton variable:  W = (p, α_g, m_g, u_g, m_l, u_l, m_g·e_g, m_l·e_l)   shape (N, 8)

  m_i = α_i · ρ_i   (partial densities)
  e_i = h_i - p/ρ_i (specific internal energies)

Two key polynomial maps (no EOS, no inner Newton):

  V(W) = (m_g, m_l, m_g·u_g, m_l·u_l,
          m_g·e_g + m_g·u_g²/2,  m_l·e_l + m_l·u_l²/2)    ← conservative

  U_h(W) = (p, u_g, u_l, α_g,
             (m_g·e_g + p·α_g)/m_g,                         ← h_g = e_g + p/ρ_g
             (m_l·e_l + p·(1−α_g))/m_l)                    ← h_l

EOS is needed only for fluxes, nc correction, and sources (same as before).

Residual for interior cell i (6-component, same physics as before):

  R_i = Vc_i·(V(W_new)_i − V(W_n)_i)
      + dt·(A_{i+1/2}·F̂_{i+1/2} − A_{i-1/2}·F̂_{i-1/2})
      + dt·A_i · X_i @ solve(K_i, F̂_{i+1/2} − F̂_{i-1/2})
      − dt·Vc_i · D_VE_i

  R ∈ R^{6·(N-2)},  W_int ∈ R^{8·(N-2)}
  Jacobian ∂R/∂W is (6·(N-2)) × (8·(N-2)) — rectangular.
  Newton update: dW = lstsq(J, −R)  [minimum-norm solution].

API
---
  W_n   = W_from_U_h_batch(U_h_0, fluid_g, fluid_l)   ← initial condition
  step, compute_dt = make_two_fluid_step(...)
  W_new, ssr = step(W_n, dt)
  U_h   = U_h_from_W_batch(W_new)                      ← back to primitives
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jaxprop as jxp

from .matrices import (
    phase_props, common_quantities,
    K_matrix, X_matrix, eigenstructure,
)
from .flux import fvs_interface_flux, nonconservative_correction
from .sources import TwoFluidClosure, D_VE_conservative, D_VE_algebric_two_component

# Void-fraction floor: α_g is hard-clipped to (ε, 1-ε) in the projection and after
# each Newton update to guard the EOS lookups and keep both phases realizable.
# Set to 1e-6 so a "ghost" phase (single-phase-limit tests) can sit at α≈1e-6
# without the clamp fighting it — matching the space-marching solver's floor.
_ALPHA_EPS = 1e-6


# ---------------------------------------------------------------------------
# W ↔ V  and  W ↔ U_h  (polynomial / algebraic — no EOS)
# ---------------------------------------------------------------------------

def V_from_W(W):
    """W (8,) → V (6,) — polynomial, no EOS."""
    p, ag, m_g, u_g, m_l, u_l, mg_eg, ml_el = W
    return jnp.array([
        m_g,
        m_l,
        m_g * u_g,
        m_l * u_l,
        mg_eg + 0.5 * m_g * u_g**2,
        ml_el + 0.5 * m_l * u_l**2,
    ])


def U_h_from_W(W):
    """W (8,) → U_h (6,) — algebraic, no EOS.

    h_g = e_g + p/ρ_g = (m_g·e_g + p·α_g) / m_g   (since p/ρ_g = p·α_g/m_g)
    h_l = e_l + p/ρ_l = (m_l·e_l + p·α_l) / m_l
    """
    p, ag, m_g, u_g, m_l, u_l, mg_eg, ml_el = W
    al  = 1.0 - ag
    h_g = (mg_eg + p * ag) / m_g
    h_l = (ml_el + p * al) / m_l
    return jnp.array([p, u_g, u_l, ag, h_g, h_l])


def W_from_U_h_batch(U_h, fluid_g, fluid_l):
    """U_h (N,6) → W (N,8).  Needs one EOS batch call to get ρ_g, ρ_l."""
    p   = U_h[:, 0];  u_g = U_h[:, 1];  u_l = U_h[:, 2]
    ag  = U_h[:, 3];  h_g = U_h[:, 4];  h_l = U_h[:, 5]
    al  = 1.0 - ag
    stg = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
    stl = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p)
    rg  = stg["rho"];  rl = stl["rho"]
    m_g = ag * rg;  m_l = al * rl
    e_g = h_g - p / rg;  e_l = h_l - p / rl
    return jnp.stack([p, ag, m_g, u_g, m_l, u_l, m_g * e_g, m_l * e_l], axis=1)


def U_h_from_W_batch(W):
    """W (N,8) → U_h (N,6) — purely algebraic, no EOS."""
    return jax.vmap(U_h_from_W)(W)


def _apply_ramped_inlet(U_h, U_h_bc, n_cells=3):
    """Blend a prescribed inlet state into the first n_cells of the domain."""
    weights = jnp.linspace(1.0, 0.0, n_cells)

    def _body(i, state):
        w = weights[i]
        return state.at[i].set((1.0 - w) * state[i] + w * U_h_bc)

    return jax.lax.fori_loop(0, n_cells, _body, U_h)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx,
    dx,
    CFL=0.5,
    k=0.5,
    delta_factor=1e-6,
    newton_iter=5,
    recon='muscl_van_leer',
    sources_fn=None,
    lowmach_diss=0.0,
    nc_scale=1.0,
    eps_void=0.0,
    eps_void_ucut=None,
    eps_void_uwidth=None,
    eps_void_scale='wood',
    du_blend_lo=1e-2,
    du_blend_hi=1e-1,
    debug_newton=False,
    jacobian_mode='dense',
    linear_solve='lstsq',
    linear_reg=0.0,
    bc_inlet='transmissive',
    bc_outlet='transmissive',
    H_g0=None,
    H_l0=None,
    alpha_g0=None,
    s_g0=None,
    quality=None,
    slip=None,
    U_h_inlet=None,
    p_out=None,
    p_out_relax=1.0,
    eos_project=True,
    use_saturation=True,
    return_debug=False,
):
    """
    Build @jax.jit step and compute_dt functions for Städtke's two-fluid system.

    Parameters
    ----------
    fluid_g, fluid_l : FluidBicubic tables (metastable vapor / liquid)
    A       : (N,)   cell cross-section areas [m²]
    A_face  : (N-1,) face cross-section areas [m²]
    dAdx    : (N,)   dA/dx at cell centres [m]
    dx      : scalar uniform cell width [m]
    CFL     : Courant number
    k       : virtual-mass coefficient
    delta_factor : Harten entropy-fix fraction
    newton_iter  : Newton iterations (unrolled inside @jax.jit)
    recon   : 'muscl_van_leer' or 'constant'
    sources_fn : optional callable with signature
    lowmach_diss : small blend factor for a Rusanov/LF stabilization term,
                   used to damp the low-Mach checkerboard mode at the interface flux
                   (u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l) → D_VE (6,)
                 where sat_g = fluid_g.get_state_saturation(p) — sat. vapor props dict,
                       sat_l = fluid_l.get_state_saturation(p) — sat. liquid props dict.
                 Use sat_g["enthalpy"], sat_g["temperature"], etc. (canonical names).
                 If None, only the geometric area-variation source is included.
    bc_inlet   : 'transmissive' (default), 'total_state', 'stagnation',
                 'stagnation_ramped', 'static_state', or 'subsonic_inflow'.
                 'total_state' mirrors _bc_inlet_real from the single-phase solver:
                   extrapolate p, u_g, u_l from interior; impose H_g0, H_l0, alpha_g0.
    H_g0       : total gas enthalpy at inlet [J/kg]  (required when bc_inlet='total_state')
    H_l0       : total liquid enthalpy at inlet [J/kg]
    alpha_g0   : void fraction at inlet [-]

    Returns
    -------
    step       : jit-compiled  step(W_n, dt) → (W_new, ssr)
    compute_dt : jit-compiled  compute_dt(W_n) → dt

    Initial condition: W_n = W_from_U_h_batch(U_h_0, fluid_g, fluid_l)
    Post-processing:   U_h = U_h_from_W_batch(W_new)
    """
    N       = len(A)
    A_j     = jnp.asarray(A,      dtype=jnp.float64)
    Af_j    = jnp.asarray(A_face, dtype=jnp.float64)
    V_cell  = A_j * dx
    dlnA_dx = jnp.asarray(dAdx, dtype=jnp.float64) / A_j

    # EOS table bounds for MUSCL fallback check (on U_h values)
    p_lo  = float(getattr(fluid_g, 'p_min', 1e4))   * 1.01
    p_hi  = float(getattr(fluid_g, 'p_max', 300e5)) * 0.99
    hg_lo = float(getattr(fluid_g, 'h_min', 50e3))  + 1.0
    hg_hi = float(getattr(fluid_g, 'h_max', 850e3)) - 1.0
    hl_lo = float(getattr(fluid_l, 'h_min', 50e3))  + 1.0
    hl_hi = float(getattr(fluid_l, 'h_max', 850e3)) - 1.0

    # ── Precomputed reservoir GAS isentrope (s = s_g0) for the stagnation inlets ──
    # The inlet needs p ON the reservoir isentrope at the current static h_g.  A
    # per-step REVERSE (h,s)→p solve (HmassSmass) is fragile — it hard-crashes via
    # lineax when h_g falls below the h where the isentrope leaves the table
    # (s=s_g0 wants p<p_min) — and slow.  Instead sample the isentrope ONCE here
    # (setup): fixed p grid → h via the reliable forward-ish PSmass call, giving a
    # monotone h_iso→p_iso lookup.  The inlet then does p_s = interp(h_g, h_iso,
    # p_iso): exact on-isentrope pressure, always finite, no runtime reverse solve.
    # jnp.interp CLAMPS below the range (h_g < h_iso[0] → p_min) so the runaway
    # regime is bounded instead of crashing.  Only built when s_g0 is given (the
    # stagnation inlets); other BCs never reference _h_iso/_p_iso.
    if s_g0 is not None:
        _N_ISO = 400
        _p_iso = jnp.linspace(p_lo, p_hi, _N_ISO)
        _h_iso = fluid_g.get_state(jxp.PSmass_INPUTS, _p_iso,
                                   jnp.full(_N_ISO, jnp.float64(s_g0)))["h"]

    # ── BC in U_h space ───────────────────────────────────────────────────────
    # BCs are applied to U_h = (p, u_g, u_l, α_g, h_g, h_l) — purely algebraic,
    # no EOS calls.  W is only used for the Newton iterate (dV term); ghost cells
    # in W are placeholders and do not carry physical BC information.
    #
    # Outlet: always transmissive (zero-gradient).
    # Inlet: transmissive (default) or total-state imposition.
    #
    # Total-state inlet:
    #   1. Extrapolate p, u_g, u_l from interior cells [1] and [2].
    #   2. Static enthalpies: h_k = H_k0 - u_k²/2  (no EOS needed).
    #   3. Impose α_g = alpha_g0.
    #
    # Key difference vs single-phase: p is extrapolated (not derived from an
    # isentrope) because two-phase flow has no single per-mixture isentrope.
    # U_h indices: 0=p, 1=u_g, 2=u_l, 3=α_g, 4=h_g, 5=h_l

    # ── Inlet BC (writes ghost cell U_h[0]) ───────────────────────────────────
    if bc_inlet == 'total_state':
        _H_g0     = jnp.float64(H_g0)
        _H_l0     = jnp.float64(H_l0)
        _alpha_g0 = jnp.float64(alpha_g0)

        def _bc_inlet(U_h):
            p_in   = 2.0 * U_h[1, 0] - U_h[2, 0]
            u_g_in = 2.0 * U_h[1, 1] - U_h[2, 1]
            u_l_in = 2.0 * U_h[1, 2] - U_h[2, 2]
            p_in   = jnp.clip(p_in,   p_lo,  p_hi)
            h_g_in = jnp.clip(_H_g0 - 0.5 * u_g_in**2, hg_lo, hg_hi)
            h_l_in = jnp.clip(_H_l0 - 0.5 * u_l_in**2, hl_lo, hl_hi)
            U_h0   = jnp.array([p_in, u_g_in, u_l_in, _alpha_g0, h_g_in, h_l_in])
            return U_h.at[0].set(U_h0)

    elif bc_inlet == 'stagnation':
        # Stagnation (total-state) inlet — characteristic-consistent reservoir BC.
        #
        # At a subsonic inlet only ONE characteristic is outgoing (u_mix − a < 0),
        # so we may extrapolate exactly ONE quantity: the mass-weighted MIXTURE
        # velocity u_mix = (α_g ρ_g u_g + α_l ρ_l u_l)/ρ_mix (linear extrapolation
        # from interior cells 1,2).  The two phase velocities are NOT independent —
        # they are tied by the imposed SLIP ratio S = u_g/u_l and, together with the
        # imposed QUALITY x (gas mass fraction), fix the void fraction via
        #     α_g = 1 / (1 + ((1−x)/x)·(ρ_g/ρ_l)·S).
        # Imposed (5): quality x0, slip S0, stagnation enthalpies H_g0/H_l0, and gas
        # stagnation entropy s_g0.  Derived: α_g (from x0,S0,ρ), the velocity SPLIT
        # (from u_mix and S0), h_k = H_k0 − u_k²/2, and the shared static p from the
        # GAS isentrope s_g(h_g,p)=s_g0.  Because ρ_g,ρ_l depend on (p,h) which depend
        # on the velocities, this is a small fixed point → a few unrolled passes.
        # The velocity MAGNITUDE (mass flux) still emerges from the interior u_mix,
        # so the nozzle can choke; slip/quality only set the composition + split.
        _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        _s_g0 = jnp.float64(s_g0)
        _x0   = jnp.float64(quality)          # gas quality (mass fraction)
        _S0   = jnp.float64(slip)             # slip ratio u_g/u_l
        _rx   = (1.0 - _x0) / _x0             # (1−x)/x

        def _umix_of(uh_cell):
            p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
            al_ = 1.0 - ag_
            rg_ = phase_props(fluid_g, p_, hg_).rho
            rl_ = phase_props(fluid_l, p_, hl_).rho
            return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)

        def _split(u_mix, ag, rg, rl):
            al = 1.0 - ag
            rmix = ag * rg + al * rl
            u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
            return _S0 * u_l, u_l          # (u_g, u_l)

        def _bc_inlet(U_h):
            # ZERO-ORDER extrapolation of the single outgoing characteristic (u_mix
            # = first interior cell), NOT the linear 2·u1−u2.  Linear extrapolation
            # is a HIGH-PASS filter: it amplifies inlet 2Δx noise and, if a cell
            # momentarily reverses, injects that reverse flow into the ghost → the
            # velocity split flips sign → garbage h_g → the isentrope Newton below
            # explodes p to the table edge.  Zero-order is 1st-order accurate, stable,
            # and matches Städtke's use of the actual first-cell mixture velocity.
            u_mix = _umix_of(U_h[1])
            p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)
            h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
            h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
            for _ in range(4):                                  # fixed point (ρ ⇄ u ⇄ p,h)
                rg = phase_props(fluid_g, p_s, h_g).rho
                rl = phase_props(fluid_l, p_s, h_l).rho
                ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)        # void from quality + slip
                u_g, u_l = _split(u_mix, ag, rg, rl)
                h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
                h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
                # p ON the reservoir isentrope at h_g — PRECOMPUTED isentrope lookup
                # (interp), NOT a per-step reverse (h,s)→p solve.  Exact on-isentrope
                # pressure, always finite, clamps below-range instead of crashing.
                p_s = jnp.clip(jnp.interp(h_g, _h_iso, _p_iso), p_lo, p_hi)
            rg = phase_props(fluid_g, p_s, h_g).rho
            rl = phase_props(fluid_l, p_s, h_l).rho
            ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
            u_g, u_l = _split(u_mix, ag, rg, rl)
            U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])
            return U_h.at[0].set(U_h0)

    elif bc_inlet == 'stagnation_ramped':
        # Same characteristic-consistent reservoir BC as 'stagnation', but blend
        # the imposed state into the first three cells to soften the inlet forcing.
        _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        _s_g0 = jnp.float64(s_g0)
        _x0   = jnp.float64(quality)          # gas quality (mass fraction)
        _S0   = jnp.float64(slip)             # slip ratio u_g/u_l
        _rx   = (1.0 - _x0) / _x0             # (1−x)/x

        def _umix_of(uh_cell):
            p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
            al_ = 1.0 - ag_
            rg_ = phase_props(fluid_g, p_, hg_).rho
            rl_ = phase_props(fluid_l, p_, hl_).rho
            return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)

        def _split(u_mix, ag, rg, rl):
            al = 1.0 - ag
            rmix = ag * rg + al * rl
            u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
            return _S0 * u_l, u_l          # (u_g, u_l)

        def _bc_inlet(U_h):
            u_mix = 2.0 * _umix_of(U_h[1]) - _umix_of(U_h[2])   # extrapolate the 1 outgoing char
            p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)
            h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
            h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
            for _ in range(4):                                  # fixed point (ρ ⇄ u ⇄ p,h)
                rg = phase_props(fluid_g, p_s, h_g).rho
                rl = phase_props(fluid_l, p_s, h_l).rho
                ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)        # void from quality + slip
                u_g, u_l = _split(u_mix, ag, rg, rl)
                h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
                h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
                st  = fluid_g.get_state(jxp.HmassP_INPUTS,
                                        jnp.expand_dims(h_g, 0), jnp.expand_dims(p_s, 0))
                p_s = jnp.clip(p_s + st["rho"][0] * st["T"][0] * (st["s"][0] - _s_g0),
                               p_lo, p_hi)                       # gas isentrope Newton step
            rg = phase_props(fluid_g, p_s, h_g).rho
            rl = phase_props(fluid_l, p_s, h_l).rho
            ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
            u_g, u_l = _split(u_mix, ag, rg, rl)
            U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])
            return _apply_ramped_inlet(U_h, U_h0, n_cells=3)

    elif bc_inlet == "stagnation_ramped_soft":
        # Characteristic-consistent reservoir BC, blended into the first three
        # cells via UNDER-RELAXATION (not a hard overwrite) to avoid injecting
        # grid-scale noise at the interior/BC interface. The extrapolation of
        # the single outgoing characteristic is also damped (limited) instead
        # of a raw linear extrapolation, since raw linear extrapolation is a
        # high-pass filter on 2dx noise and will amplify checkerboard modes.
        _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        _s_g0 = jnp.float64(s_g0)
        _x0   = jnp.float64(quality)
        _S0   = jnp.float64(slip)
        _rx   = (1.0 - _x0) / _x0

        _OMEGA   = jnp.float64(0.3)     # relaxation gain, e.g. 0.2-0.4
        _EXTRAP_LIMIT = jnp.float64(0.5)        # 0 = no extrap (1st order),
                                                # 1 = full linear extrap

        def _umix_of(uh_cell):
            p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
            al_ = 1.0 - ag_
            rg_ = phase_props(fluid_g, p_, hg_).rho
            rl_ = phase_props(fluid_l, p_, hl_).rho
            return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)

        def _split(u_mix, ag, rg, rl):
            al = 1.0 - ag
            rmix = ag * rg + al * rl
            u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
            return _S0 * u_l, u_l

        def _damped_extrap(u1, u2, u3):
            # Minmod-limited extrapolation of the outgoing characteristic instead
            # of raw 2*u1 - u2. Falls back toward u1 (1st order / zero-gradient)
            # whenever the interior gradient changes sign or is noisy, which is
            # exactly the checkerboard signature -> kills it at the source.
            d1 = u1 - u2
            d2 = u2 - u3
            same_sign = jnp.sign(d1) == jnp.sign(d2)
            slope = jnp.where(same_sign,
                            jnp.sign(d1) * jnp.minimum(jnp.abs(d1), jnp.abs(d2)),
                            0.0)
            return u1 + _EXTRAP_LIMIT * slope

        def _bc_inlet(U_h):
            u_mix = _damped_extrap(_umix_of(U_h[1]), _umix_of(U_h[2]), _umix_of(U_h[3]))

            # p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)   # (was) seed from the SPIKING cell 1
            p_s = jnp.clip(U_h[0, 0], p_lo, p_hi)     # warm-start from previous inlet (ghost)
            h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
            h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
            for _ in range(4):
                rg = phase_props(fluid_g, p_s, h_g).rho
                rl = phase_props(fluid_l, p_s, h_l).rho
                ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
                u_g, u_l = _split(u_mix, ag, rg, rl)
                h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
                h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
                # OLD gas-isentrope NEWTON step (commented — ρT~6000 → overshoots p to
                # the table edge → the inlet spike; under-relax/cap BROKE convergence):
                # st  = fluid_g.get_state(jxp.HmassP_INPUTS,
                #                         jnp.expand_dims(h_g, 0), jnp.expand_dims(p_s, 0))
                # p_s = jnp.clip(p_s + st["rho"][0] * st["T"][0] * (st["s"][0] - _s_g0),
                #             p_lo, p_hi)
                # p ON the reservoir isentrope at h_g — PRECOMPUTED isentrope lookup
                # (interp), NOT a per-step reverse (h,s)→p solve.  Exact on-isentrope
                # pressure, always finite, clamps below-range instead of crashing.
                p_s = jnp.clip(jnp.interp(h_g, _h_iso, _p_iso), p_lo, p_hi)
            rg = phase_props(fluid_g, p_s, h_g).rho
            rl = phase_props(fluid_l, p_s, h_l).rho
            ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
            u_g, u_l = _split(u_mix, ag, rg, rl)
            U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])

            return _apply_ramped_inlet_relaxed(U_h, U_h0, n_cells=3, omega=_OMEGA)

        def _apply_ramped_inlet_relaxed(U_h, U_h0, n_cells, omega):
            # Instead of overwriting cells 1..n_cells with a spatial ramp toward
            # U_h0 (hard reset every step -> impedance mismatch at cell n_cells+1
            # -> reflected 2dx noise), nudge each cell a fraction `omega` of the
            # way toward a *per-cell* target that itself ramps in weight from the
            # boundary inward. This is a soft spring, not a Dirichlet reset, so
            # residual-based (implicit) convergence is unaffected at steady state
            # -- omega only controls transient path, not the converged solution,
            # since at convergence U_h == U_h0 makes the correction vanish anyway.
            # GHOST cell 0 MUST carry the imposed reservoir state — it is what the
            # inlet flux (face between ghost 0 and cell 1) reads.  The soft ramp only
            # touches cells 1..n_cells, so WITHOUT this the ghost floats to garbage
            # (observed: p→0, u→−55, T<0 spike at x≈0).  Hard-set it like every other
            # inlet variant; the under-relaxation below is only for the interior cells.
            U_h = U_h.at[0].set(U_h0)
            idx = jnp.arange(1, n_cells + 1)
            ramp_w = jnp.linspace(1.0, 1.0 / n_cells, n_cells)   # 1 at inlet -> weaker inward
            target = U_h0[None, :]                                # same target state, weighted blend
            blended_target = ramp_w[:, None] * target + (1.0 - ramp_w[:, None]) * U_h[idx]
            U_h = U_h.at[idx].set(U_h[idx] + omega * (blended_target - U_h[idx]))
            return U_h

    elif bc_inlet == 'static_state':
        # Fully-fixed (Dirichlet) STATIC inlet — mirrors the space-marching demo:
        # impose the whole inlet state U_h_inlet = (p, u_g, u_l, α_g, h_g, h_l),
        # with p at the static inlet pressure, h_g = h_sat_g(p) (saturated vapour →
        # T_g = T_sat), h_l = h_sat_l(p) + Δh_l (superheated liquid).  Pinning p also
        # anchors the pressure level (transmissive outlet alone leaves it free).
        # Over-specifies the subsonic inflow by one characteristic (as the demo's
        # IVP does), but is the faithful reproduction of the demo inlet.
        _U_h_in = jnp.asarray(U_h_inlet, dtype=jnp.float64)

        def _bc_inlet(U_h):
            return U_h.at[0].set(_U_h_in)

    elif bc_inlet == 'subsonic_inflow':
        # Subsonic inflow: impose the 5 INCOMING characteristics — velocities,
        # void fraction, and phase enthalpies (u_g, u_l, α_g, h_g, h_l) from
        # U_h_inlet — and EXTRAPOLATE p (the one outgoing/backward-acoustic
        # characteristic, u_mix − a_mix < 0) from the interior.  Pressure is NOT
        # pinned here — it must be anchored downstream by a back-pressure outlet
        # (bc_outlet='pressure'); otherwise the pressure level is a free mode.
        _U_h_in = jnp.asarray(U_h_inlet, dtype=jnp.float64)

        def _bc_inlet(U_h):
            p_in = jnp.clip(U_h[1, 0], p_lo, p_hi)   # zero-order extrapolation from cell 1
            U_h0 = jnp.array([p_in, _U_h_in[1], _U_h_in[2],
                              _U_h_in[3], _U_h_in[4], _U_h_in[5]])
            return U_h.at[0].set(U_h0)

    else:  # transmissive: zero-gradient ghost
        def _bc_inlet(U_h):
            return U_h.at[0].set(U_h[1])

    # ── Outlet BC (writes ghost cell U_h[-1]) ─────────────────────────────────
    if bc_outlet == 'transmissive':
        def _bc_outlet(U_h):
            return U_h.at[-1].set(U_h[-2])
    elif bc_outlet == 'pressure':
        # Non-reflecting subsonic-outflow back-pressure (LODI / characteristic).
        # A naive "impose p_out, copy u" ghost changes ONLY pressure at the outlet
        # face → the FVS flux reads that pure pressure jump as an incoming acoustic
        # → REFLECTS it → the last cell's V is driven unrealizable and its EOS
        # projection fails (relEOS blows up) → dt collapses.  Instead we move p AND
        # u TOGETHER along the OUTGOING acoustic characteristic, preserving the
        # Riemann invariant J⁺ = u + ∫dp/(ρa), so imposing the back-pressure emits
        # NO reflected wave:
        #     u_ghost = u_interior + (p_interior − p_ghost) / (ρ_mix a_mix)
        # ρ_mix a_mix is the mixture acoustic impedance (Wood's frozen speed).  The
        # same velocity correction is applied to both phase velocities.  p_out_relax
        # blends the imposed pressure (1 = full back-pressure; <1 = softer anchor).
        _p_out = jnp.float64(p_out)
        _relax = jnp.float64(p_out_relax)

        def _bc_outlet(U_h):
            uh   = U_h[-2]
            p98  = uh[0];  ag = uh[3];  al = 1.0 - ag
            eg   = phase_props(fluid_g, p98, uh[4])
            el   = phase_props(fluid_l, p98, uh[5])
            rmix = ag * eg.rho + al * el.rho
            comp = ag / (eg.rho * eg.a**2) + al / (el.rho * el.a**2)
            imped = jnp.sqrt(rmix / comp)             # ρ_mix · a_Wood
            p_gh = p98 + _relax * (_p_out - p98)       # relax=1 → full back-pressure
            du   = (p98 - p_gh) / imped                # co-move u along outgoing char
            U_hN = jnp.array([p_gh, uh[1] + du, uh[2] + du, ag, uh[4], uh[5]])
            return U_h.at[-1].set(U_hN)
    elif bc_outlet == 'characteristic':
        # Regime-aware outlet that AUTO-SWITCHES subsonic ↔ supersonic, mirroring
        # the single-phase Städtke outlet (fvm.py make_jax_stadtke_step._bc_outlet).
        # The two-fluid eigenstructure is analogous: exactly ONE acoustic wave
        # (u_mix − a_mix) can enter at a subsonic outlet; all material waves
        # (u_g, u_l) are outgoing.  So:
        #   subsonic  (M_mix<1): impose p_out and co-move u along the outgoing
        #                        acoustic (the LODI 'pressure' ghost above),
        #   supersonic(M_mix>1): extrapolate all 6 (the 'transmissive' ghost).
        # Blend the two by a SMOOTH tanh weight on the MIXTURE Mach so the implicit
        # Newton keeps a non-zero BC Jacobian near M=1 (lets p_out propagate
        # upstream).  Because both ghosts share the interior (α, h_g, h_l), the
        # blend collapses to scaling the LODI pressure correction by w:
        #     p_ghost = p98 + w·relax·(p_out − p98),   u += (p98 − p_ghost)/impedance
        # w→1 (subsonic) = full back-pressure;  w→0 (supersonic) = pure extrapolation.
        #
        # CRITICAL: the sub/supersonic Mach MUST be measured with the SCHEME's OWN
        # characteristic speeds — the Städtke eigenstructure `u_mix ± a` (which the
        # flux and CFL use, incl. virtual mass k via rho_hat_s and slip via
        # delta_a_sq) — NOT the Wood speed a_W = √(1/(ρ_mix·Σαᵢ/(ρᵢaᵢ²))).  Using the
        # Wood speed (which differs from a_Städtke) mis-classifies the outlet: it
        # reads M_W ≥ 1 → w ≈ 0 → the back-pressure is silently dropped and the
        # outlet behaves TRANSMISSIVE even when the true outgoing acoustic u_mix−a is
        # still < 0 (subsonic).  We therefore take M_out from eigenstructure().  The
        # Wood impedance is kept ONLY for the LODI velocity co-move (shared with the
        # working 'pressure' outlet; it sets the reflection quality, not the regime).
        _p_out = jnp.float64(p_out)
        _relax = jnp.float64(p_out_relax)
        _msw   = jnp.float64(0.05)          # tanh half-width in Mach (switch sharpness)

        def _bc_outlet(U_h):
            uh   = U_h[-2]
            p98  = uh[0];  ag = uh[3];  al = 1.0 - ag
            eg   = phase_props(fluid_g, p98, uh[4])
            el   = phase_props(fluid_l, p98, uh[5])
            rmix = ag * eg.rho + al * el.rho
            comp = ag / (eg.rho * eg.a**2) + al / (el.rho * el.a**2)
            imped = jnp.sqrt(rmix / comp)                  # ρ_mix · a_Wood (LODI co-move only)
            # Regime switch from the scheme's OWN eigenstructure (Städtke wave speeds):
            # eigvals = [u_g, u_l, u_mix+a, u_mix−a, u_g, u_l]  ⇒  a=½(ev₂−ev₃),
            # u_mix=½(ev₂+ev₃).  Subsonic outlet ⇔ outgoing acoustic ev₃=u_mix−a < 0.
            cq   = common_quantities(uh, eg, el, k)
            ev   = eigenstructure(uh, eg, el, cq)[0]
            a_s  = 0.5 * (ev[2] - ev[3])                   # Städtke mixture sound speed (>0)
            u_s  = 0.5 * (ev[2] + ev[3])                   # hat-weighted mixture velocity
            M_out = jnp.abs(u_s) / (a_s + 1e-300)
            w     = 0.5 * (1.0 + jnp.tanh((1.0 - M_out) / _msw))  # 1 sub, 0 super
            p_gh  = p98 + w * _relax * (_p_out - p98)
            du    = (p98 - p_gh) / imped
            U_hN  = jnp.array([p_gh, uh[1] + du, uh[2] + du, ag, uh[4], uh[5]])
            return U_h.at[-1].set(U_hN)
    else:
        raise ValueError(f"unknown bc_outlet={bc_outlet!r} "
                         f"(only 'transmissive', 'pressure', 'characteristic')")

    def _bc_U_h(U_h):
        return _bc_outlet(_bc_inlet(U_h))

    # ── MUSCL van Leer in U_h space ───────────────────────────────────────────

    def _muscl(U_h):
        dU = U_h[1:] - U_h[:-1]
        if recon == 'muscl_van_leer':
            lim = lambda a, b: jnp.where(
                a * b > 0, 2.0 * a * b / (a + b + 1e-300), 0.0)
        elif recon == 'muscl_minmod':
            lim = lambda a, b: jnp.where(
                a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
        else:   # mc
            lim = lambda a, b: jnp.where(
                a * b > 0,
                jnp.sign(a) * jnp.minimum(
                    jnp.minimum(2*jnp.abs(a), 2*jnp.abs(b)),
                    0.5*jnp.abs(a + b)), 0.0)
        sigma  = jnp.zeros_like(U_h).at[1:-1].set(lim(dU[:-1], dU[1:]))
        U_L_m  = U_h[:-1] + 0.5 * sigma[:-1]
        U_R_m  = U_h[1:]  - 0.5 * sigma[1:]
        bad_L  = ((U_L_m[:, 0] < p_lo)  | (U_L_m[:, 0] > p_hi)  |
                  (U_L_m[:, 4] < hg_lo) | (U_L_m[:, 4] > hg_hi) |
                  (U_L_m[:, 5] < hl_lo) | (U_L_m[:, 5] > hl_hi))
        bad_R  = ((U_R_m[:, 0] < p_lo)  | (U_R_m[:, 0] > p_hi)  |
                  (U_R_m[:, 4] < hg_lo) | (U_R_m[:, 4] > hg_hi) |
                  (U_R_m[:, 5] < hl_lo) | (U_R_m[:, 5] > hl_hi))
        bad    = (bad_L | bad_R)[:, None]
        return jnp.where(bad, U_h[:-1], U_L_m), jnp.where(bad, U_h[1:], U_R_m)

    # ── FVS fluxes at all N-1 faces ───────────────────────────────────────────

    def _fvs_all_faces(U_h):
        if recon != 'constant':
            U_L, U_R = _muscl(U_h)
        else:
            U_L, U_R = U_h[:-1], U_h[1:]
        return jax.vmap(
            lambda ul, ur: fvs_interface_flux(
                ul, ur, fluid_g, fluid_l, k, delta_factor, lowmach_diss=lowmach_diss,
                du_blend_lo=du_blend_lo, du_blend_hi=du_blend_hi,
            )
        )(U_L, U_R)

    # ── Non-conservative correction at interior cells ─────────────────────────

    def _nc_all_interior(U_h, F_hat):
        dF = F_hat[1:] - F_hat[:-1]

        def _nc_one(u_h, df):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            return nonconservative_correction(u_h, df, eos_g_c, eos_l_c, cq_c)

        return jax.vmap(_nc_one)(U_h[1:-1], dF)

    # ── Source terms at interior cells ────────────────────────────────────────

    def _src_all_interior(U_h):
        def _src_one(u_h, dlnA):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            if sources_fn is not None:
                # Saturation properties at current cell pressure (scalar p = u_h[0]).
                # fluid_g is the vapor table → sat. vapor props.
                # fluid_l is the liquid table → sat. liquid props.
                # sources_fn signature:
                #   (u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l) → D_VE (6,)
                # use_saturation=False: tables built WITHOUT the metastable/saturation
                # extension (e.g. Air, or a purely subcooled liquid) have no saturation
                # props, so skip the lookup and pass zero placeholders.  Only safe when
                # the sources_fn does not read sat_g/sat_l (e.g. area source only, or
                # mass transfer disabled).
                if use_saturation:
                    sat_g_c = fluid_g.get_state_saturation(u_h[0])
                    sat_l_c = fluid_l.get_state_saturation(u_h[0])
                else:
                    _z = jnp.zeros(())
                    sat_g_c = {"temperature": _z, "enthalpy": _z}
                    sat_l_c = {"temperature": _z, "enthalpy": _z}
                return sources_fn(u_h, eos_g_c, eos_l_c, cq_c, dlnA, sat_g_c, sat_l_c)
            # FALLBACK ONLY (sources_fn is None): primitive-frame area source with a
            # zero physical closure.  NOT used when a sources_fn is supplied (the
            # driver always supplies one → the ALGEBRAIC D_VE_algebric/D_VE_area path
            # above runs instead).  Kept for the no-sources_fn convenience case.
            z  = jnp.zeros(())
            cl = TwoFluidClosure(
                sigma_Mg=z, sigma_Qg=z, sigma_Ql=z, Fv_g=z,
                F_g=z, F_l=z, Q_g=z, Q_l=z,
                u_ex=z, u_int=z, h_ex_g=z, h_ex_l=z, dlnA_dx=dlnA,
            )
            return D_VE_conservative(u_h, eos_g_c, eos_l_c, cq_c, cl, k)

        return jax.vmap(_src_one)(U_h[1:-1], dlnA_dx[1:-1])

    # ── Residual:  R ∈ R^{6·(N-2)},  W_int ∈ R^{8·(N-2)} ───────────────────

    def _residual(W_int_flat, W_n_flat, dt):
        W_int = W_int_flat.reshape(N - 2, 8)
        W_n   = W_n_flat.reshape(N, 8)

        W = jnp.concatenate([W_n[:1], W_int, W_n[-1:]], axis=0)

        # Conservative variables — POLYNOMIAL in W, no EOS chain in dV/dW
        # Only interior cells [1:-1] enter dV, so ghost-cell W values don't matter.
        V_new = jax.vmap(V_from_W)(W)
        V_old = jax.vmap(V_from_W)(W_n)

        # Primitives U_h — ALGEBRAIC from W, then BCs applied in U_h space
        U_h = jax.vmap(U_h_from_W)(W)
        U_h = _bc_U_h(U_h)

        # FVS fluxes (EOS called internally by fvs_interface_flux)
        F_hat = _fvs_all_faces(U_h)

        # Non-conservative correction and sources (from the UNfiltered flux)
        nc  = _nc_all_interior(U_h, F_hat)
        src = _src_all_interior(U_h)

        # ── 4th-order (JST-type) background filter ────────────────────────────
        # Damps the 2Δx void/alpha checkerboard that the FVS leaves undamped at
        # low phase velocity (λ=u_g,u_l→0 ⇒ ~zero material-wave dissipation).
        # Added as a conservative FLUX (telescopes → mass/momentum/energy exactly
        # conserved), scaled by the local spectral radius so it does NOT vanish as
        # u_g,u_l→0.  4th-order ⇒ perturbs the SMOOTH solution only at O(eps_void·Δx³)
        # (negligible / ε-insensitive) while strongly damping the checkerboard.
        # eps_void=0 → identically off.  Needs the ±2 stencil ⇒ _sw widened below.
        #
        # LOW-VELOCITY GATING (eps_void_ucut): the α/void checkerboard is a MATERIAL-
        # wave mode — it rides the phase-velocity eigenvalues λ=u_g,u_l, whose FVS
        # dissipation ~|λ|·Δx vanishes as u_g,u_l→0.  So the physically-correct
        # "smallness" sensor is the PHASE VELOCITY itself, NOT a Mach number: in a
        # two-phase mixture a_mix is the Wood speed, a strongly non-monotone function
        # of α, so dividing by it would move the gate with α rather than with the
        # velocity.  Sensor: u_mat = min(|u_g|,|u_l|) [m/s] — the slowest material
        # wave (min ⇒ the filter stays ON if EITHER phase slows, robust under slip).
        # Gate: multiply the filter by a smooth low-pass weight
        #     w = ½(1 + tanh((u0 − u_mat)/Δu))   → 1 for u_mat≪u0, → 0 for u_mat≫u0,
        # so the filter is active ONLY in the low-velocity (inlet) region and is OFF
        # at the throat/supersonic region (full high-order accuracy preserved there).
        # u0=eps_void_ucut [m/s], Δu=eps_void_uwidth [m/s] (defaults to u0/3 if None).
        # eps_void_ucut=None ⇒ ungated (uniform, the original behaviour).  Reads the
        # velocities from the SAME ±2-stencil cells ⇒ does NOT widen the band.
        if eps_void > 0.0:
            def _lam_umat_cell(u_h):
                eg_ = phase_props(fluid_g, u_h[0], u_h[4])
                el_ = phase_props(fluid_l, u_h[0], u_h[5])
                ev, _, _, _ = eigenstructure(
                    u_h, eg_, el_, common_quantities(u_h, eg_, el_, k))
                # eigvals = [u_g, u_l, u_mix+a, u_mix−a, u_g, u_l]  (matrices.py)
                u_mat = jnp.minimum(jnp.abs(ev[0]), jnp.abs(ev[1]))  # min(|u_g|,|u_l|)
                # FILTER-STRENGTH scale (eps_void_scale):
                #   'wood'   → the model's true mixture eigenvalue |u_mix|+a_s.  a_s is
                #             the Städtke/Wood mixture speed, which DIPS to a deep
                #             minimum at α≈0.5 → the filter self-weakens exactly where
                #             the slip/void checkerboard peaks (see nc/eps_void notes).
                #   'frozen' → a NON-dipping surrogate |u_mix| + max(a_g,a_l) built from
                #             the single-phase (frozen) sound speeds, which have no
                #             α-dependence.  Keeps the damping magnitude ~constant in α.
                #             Used ONLY here for the 4th-order filter scale — the fluxes
                #             and CFL still use the true (dipping) a_s.  Retune eps_void
                #             down to compensate the larger magnitude.
                if eps_void_scale == 'frozen':
                    u_mix = 0.5 * (ev[2] + ev[3])                    # u_mix = ½((u+a)+(u−a))
                    lam   = jnp.abs(u_mix) + jnp.maximum(eg_.a, el_.a)
                else:                                                # 'wood' (default)
                    lam = jnp.max(jnp.abs(ev))
                return lam, u_mat
            lam_c, umat_c = jax.vmap(_lam_umat_cell)(U_h)        # (N,) each
            r_face = 0.5 * (lam_c[:-1] + lam_c[1:])              # (N-1,) face scale
            if eps_void_ucut is not None:
                u0 = jnp.float64(eps_void_ucut)
                du = jnp.float64(eps_void_uwidth if eps_void_uwidth is not None
                                 else eps_void_ucut / 3.0)
                w_c    = 0.5 * (1.0 + jnp.tanh((u0 - umat_c) / du))  # 1 slow → 0 fast
                w_face = 0.5 * (w_c[:-1] + w_c[1:])              # (N-1,)
                r_face = r_face * w_face                         # gate the filter scale
            third  = (V_new[3:] - 3.0 * V_new[2:-1]
                      + 3.0 * V_new[1:-2] - V_new[:-3])          # (N-3,6): faces 1..N-3
            d_face = jnp.zeros_like(F_hat).at[1:-1].set(
                eps_void * r_face[1:-1, None] * third)           # +sign DAMPS (verified); 0 at bdry
            F_hat  = F_hat + d_face

        Vc      = V_cell[1:-1, None]
        dV      = V_new[1:-1] - V_old[1:-1]
        fdiv    = Af_j[1:, None] * F_hat[1:] - Af_j[:-1, None] * F_hat[:-1]
        nc_term = nc_scale * A_j[1:-1, None] * nc   # nc_scale=0 → NC coupling OFF (diagnostic)

        R = Vc * dV + dt * fdiv + dt * nc_term - dt * Vc * src
        return R.flatten()   # (6*(N-2),)

    # ── Jacobian: dense (exact AD) or colored (banded compression) ────────────
    # R ∈ R^{6(N-2)}, W ∈ R^{8(N-2)}.  The residual is block-banded: residual
    # row-block for cell i touches only column-blocks i-sw..i+sw
    #   sw = 1 (constant recon)  |  sw = 2 (MUSCL, ±2 reconstruction stencil).
    # Both BCs fold inward (total_state inlet reads cells 1,2; transmissive outlet
    # reads cell -2) so they never widen the band — verified via jac_diff.
    #
    # Column coloring (forward mode, matching jacfwd — avoids transpose solves
    # through the internal K^{-1}/lstsq in flux & nc): columns ≥ col_stride apart
    # never share a residual row, so they are probed together in ONE JVP and
    # de-interleaved exactly.  n_colors is fixed by the stencil (24 constant,
    # 40 MUSCL), INDEPENDENT of N → O(N) instead of O(N²) Jacobian cost.
    # eps_void>0 adds a ±2-stencil 4th-order filter → widen the band to sw=2 even
    # for constant recon so the colored Jacobian stays exact (verify via jac_diff).
    _sw          = 2 if (recon != 'constant' or eps_void > 0.0) else 1
    _col_stride  = 8 * (2 * _sw + 1)             # DOF stride (8 vars/cell): 24 | 40
    _m_cols      = 8 * (N - 2)                     # total W DOFs
    _n_rows      = 6 * (N - 2)                     # total residual rows
    _n_colors    = min(_col_stride, _m_cols)       # graceful if N tiny

    _jac_dense = jax.jacfwd(_residual, argnums=0)

    def _jac_colored(W_int_flat, W_n_flat, dt):
        # Linearize once (expensive nonlinear residual + EOS chain runs a single
        # time); each colored seed is then a cheap linear JVP.
        _, jvp_fn = jax.linearize(lambda w: _residual(w, W_n_flat, dt), W_int_flat)
        cols_idx  = jnp.arange(_m_cols)
        color_ids = jnp.arange(_n_colors)
        S  = (cols_idx[None, :] % _col_stride == color_ids[:, None]).astype(W_int_flat.dtype)
        Jc = jax.vmap(jvp_fn)(S)                   # (n_colors, 6(N-2)): summed columns
        # De-interleave: residual row r → cell lr = r//6.  In-band columns are
        # cells lr-sw..lr+sw × 8 vars; each has a unique colour within the band,
        # so J[r, c] = Jc[c % col_stride, r].
        rows = jnp.arange(_n_rows)
        lr   = rows // 6
        J    = jnp.zeros((_n_rows, _m_cols), dtype=W_int_flat.dtype)
        for dc in range(-_sw, _sw + 1):
            for vc in range(8):
                lc     = lr + dc
                c      = lc * 8 + vc
                valid  = (lc >= 0) & (lc <= (N - 3))
                c_safe = jnp.clip(c, 0, _m_cols - 1)
                val    = jnp.where(valid, Jc[c_safe % _col_stride, rows], 0.0)
                J      = J.at[rows, c_safe].add(val)   # invalid → adds 0 at clipped col
        return J

    _jac_fn = _jac_colored if jacobian_mode == 'colored' else _jac_dense

    # ── Newton inner linear solve ─────────────────────────────────────────────
    # The Newton system is underdetermined: J is 6(N-2) × 8(N-2) (fewer equations
    # than unknowns), so we take the minimum-norm solution dW = Jᵀ(J Jᵀ)⁻¹(−R).
    #
    # 'lstsq'     : dense SVD min-norm least-squares (robust, rank-tolerant).
    #               Gold-standard accuracy; slowest.
    # 'normal_eq' : min-norm via the normal equations, Cholesky on J Jᵀ (SPD).
    #               ~15× faster than SVD, but FORMS J Jᵀ → squares the condition
    #               number (κ(JJᵀ)=κ(J)²), losing ~half the digits on the stiff
    #               source-coupled Jacobian.  Tiny Tikhonov for rank-deficiency.
    # 'qr'        : min-norm via the LQ factorization (QR of Jᵀ).  With Jᵀ = Q R
    #               (economic), J Jᵀ = RᵀR, so the min-norm solution is
    #                   dW = Jᵀ(JJᵀ)⁻¹(−R) = Q R⁻ᵀ(−R)
    #               i.e. solve Rᵀ y = (−R) (triangular), dW = Q y.  QR does NOT
    #               square κ (κ(R)=κ(J)) → ~SVD accuracy at ~3-5× the SVD speed.
    #               Assumes J full row rank; near exact zero slip add a small
    #               Tikhonov or fall back to 'lstsq'.
    _I_rows = jnp.eye(6 * (N - 2), dtype=jnp.float64)

    if linear_solve == 'normal_eq':
        def _lin_solve(J_sc, rhs):                     # rhs = -R
            JJt = J_sc @ J_sc.T
            # With row equilibration cond(J)~1e3 → JJt is well-conditioned, so
            # linear_reg defaults to 0 (no Tikhonov).  A small value only floors
            # exact singularity; large values BIAS the min-norm step (they swamp
            # the small singular value that dominates the true solution).
            reg = linear_reg * jnp.mean(jnp.diagonal(JJt))
            y   = jsl.solve(JJt + reg * _I_rows, rhs, assume_a='pos')
            return J_sc.T @ y
    elif linear_solve == 'qr':
        def _lin_solve(J_sc, rhs):                     # rhs = -R
            # Jᵀ (n×m, tall) = Q (n×m) · R (m×m upper-tri);  dW = Q · (Rᵀ \ rhs)
            Q, R = jnp.linalg.qr(J_sc.T, mode='reduced')
            y    = jsl.solve_triangular(R, rhs, lower=False, trans='T')
            return Q @ y
    else:
        def _lin_solve(J_sc, rhs):
            return jnp.linalg.lstsq(J_sc, rhs, rcond=None)[0]

    def jac_diff(W_n, dt):
        """max|J_colored - J_dense| at state W_n — sparsity validation guard.
        Must be ~0 (same AD graph, reordered sums).  VALIDATE AT A GRADIENT STATE:
        at a uniform field the MUSCL ±2 coupling vanishes and a wrong sw=1 coloring
        would falsely pass."""
        W_int = W_n[1:-1].reshape(-1)
        W_nf  = W_n.reshape(-1)
        return jnp.max(jnp.abs(_jac_colored(W_int, W_nf, dt)
                               - _jac_dense(W_int, W_nf, dt)))

    # ── CFL time step ─────────────────────────────────────────────────────────

    # Per-cell spectral radius (max |eigenvalue|) of the coupled two-fluid system,
    # i.e. the Städtke mixture eigenvalues (u_mix ± a_mix, u_g, u_l), NOT the
    # per-phase EOS sound speeds a_g, a_l.  For bubbly flow a_mix ≪ a_l, so using
    # a_l would over-estimate the wave speed ~10×.  Same eigenstructure the flux
    # uses, so the CFL bound and the SSR time-scale are consistent with the scheme.
    def _max_wave_speed_cells(U_h_all):
        def _lam_one(u_h):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            eigvals, _, _, _ = eigenstructure(u_h, eos_g_c, eos_l_c, cq_c)
            return jnp.max(jnp.abs(eigvals))
        return jax.vmap(_lam_one)(U_h_all)

    @jax.jit
    def compute_dt(W_n):
        U_h     = jax.vmap(U_h_from_W)(W_n)   # W → U_h algebraic, no EOS
        lam_max = _max_wave_speed_cells(U_h)
        return CFL * dx / jnp.max(lam_max[1:-1])

    # ── Implicit Newton step ──────────────────────────────────────────────────

    # Characteristic scales for W = (p, α_g, m_g, u_g, m_l, u_l, mg_eg, ml_el).
    # lstsq minimises ||dW||² in absolute space, which causes huge steps in
    # dimensionless variables like α_g (whose Jacobian column is O(dt) ≈ 1e-5×
    # smaller than the m_g column from the Vc*dV term).  Scaling the columns of
    # J by w_ref before solving — and unscaling dW afterward — makes lstsq
    # minimise ||dW / w_ref||² instead, i.e. relative changes, which keeps α_g
    # safely inside (0, 1).
    _w_char = jnp.tile(
        jnp.array([1e5, 0.05, 0.1, 0.5, 10.0, 0.5, 1e4, 1e6]),
        N - 2,
    )   # (8*(N-2),) — floor values; actual w_ref = max(|W_k|, _w_char)

    # ── EOS-manifold projection ────────────────────────────────────────────────
    # The Newton system is underdetermined (6 conservation eqs, 8 unknowns/cell),
    # so the min-norm step leaves (p, α_g) UNPINNED and W drifts off the EOS
    # manifold  m_g = α_g ρ_g(p,h_g),  m_l = (1-α_g) ρ_l(p,h_l)  (diagnosed in
    # test_T3b: rel_eos → 0.55 over 400 steps → the α_g checkerboard/void mode).
    # V_from_W does NOT depend on p or α_g, so we re-pin them to the conserved V
    # by Newton-solving the two closures while holding the other 6 W-components
    # (hence V) fixed.  Provably a no-op on already-consistent states
    # (test_eos_projection), so it never disturbs a converged / steady state.
    _PROJ_ITER = 8

    def _project_cell(W):
        p0, ag0, m_g, u_g, m_l, u_l, mg_eg, ml_el = W

        def _res(pa):
            p, ag = pa
            al  = 1.0 - ag
            h_g = (mg_eg + p * ag) / m_g
            h_l = (ml_el + p * al) / m_l
            # Clip EOS inputs to the table domain so the projection Newton NEVER
            # extrapolates into the NaN-producing region (same guard the closure
            # and flux paths use).  α_g enters the residual directly (α_g·ρ_g),
            # so the 2×2 Jacobian stays non-singular even where h is clamped.
            p_c  = jnp.clip(p,   p_lo,  p_hi)
            h_gc = jnp.clip(h_g, hg_lo, hg_hi)
            h_lc = jnp.clip(h_l, hl_lo, hl_hi)
            rg  = fluid_g.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_gc), jnp.atleast_1d(p_c))["density"][0]
            rl  = fluid_l.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_lc), jnp.atleast_1d(p_c))["density"][0]
            return jnp.array([ag * rg - m_g, al * rl - m_l])

        _jac = jax.jacfwd(_res)

        def _body(_, pa):
            dpa = jnp.linalg.solve(_jac(pa), -_res(pa))
            dpa = jnp.where(jnp.all(jnp.isfinite(dpa)), dpa, 0.0)   # singular → no move
            # Damp: cap the per-iteration change so an overshoot can't leave the
            # valid region (p by 50%, α_g by 0.2), then clip α_g to (ε, 1-ε).
            dpa = dpa.at[0].set(jnp.clip(dpa[0], -0.5 * jnp.abs(pa[0]), 0.5 * jnp.abs(pa[0])))
            dpa = dpa.at[1].set(jnp.clip(dpa[1], -0.2, 0.2))
            pa  = pa + dpa
            return pa.at[1].set(jnp.clip(pa[1], _ALPHA_EPS, 1.0 - _ALPHA_EPS))

        pa = jax.lax.fori_loop(0, _PROJ_ITER, _body, jnp.array([p0, ag0]))
        # Safety net: a failed projection must never be worse than not projecting —
        # if anything went non-finite, keep the pre-projection (p, α_g).
        pa = jnp.where(jnp.all(jnp.isfinite(pa)), pa, jnp.array([p0, ag0]))
        return W.at[0].set(pa[0]).at[1].set(pa[1])

    def _project_interior(W_int_flat):
        Wp = jax.vmap(_project_cell)(W_int_flat.reshape(N - 2, 8))
        return Wp.reshape(-1)

    def step(W_n, dt):
        """
        Advance one implicit time step.

        Parameters
        ----------
        W_n : (N, 8) — W state at time n
        dt  : scalar

        Returns
        -------
        W_new : (N, 8)
        ssr   : scalar — normalised steady-state residual
        """
        W_n_flat = W_n.reshape(-1)          # (8*N,)
        W_k      = W_n[1:-1].reshape(-1)    # (8*(N-2),) initial Newton guess

        if debug_newton:
            import numpy as _np
            _eq  = ["m_g", "m_l", "mom_g", "mom_l", "E_g", "E_l"]
            _var = ["p", "ag", "m_g", "u_g", "m_l", "u_l", "mg_eg", "ml_el"]
            print(f"\n  [Newton]  dt={float(dt):.3e}  N_int={N-2}")
            for _it in range(newton_iter):
                R_k = _residual(W_k, W_n_flat, dt)
                _nan_R = bool(jnp.any(jnp.isnan(R_k)))
                _inf_R = bool(jnp.any(jnp.isinf(R_k)))
                print(f"    it {_it:2d}  max|R|={float(jnp.nanmax(jnp.abs(R_k))):.3e}"
                      f"  NaN={_nan_R}  Inf={_inf_R}")
                if _nan_R or _inf_R:
                    _R_mat = _np.array(R_k).reshape(N - 2, 6)
                    _W_mat = _np.array(W_k).reshape(N - 2, 8)
                    _bad = _np.where(_np.isnan(_R_mat).any(1) | _np.isinf(_R_mat).any(1))[0]
                    print(f"           bad cells: {_bad[:8]}")
                    for _c in _bad[:3]:
                        print(f"             cell {_c}  R = {dict(zip(_eq, _R_mat[_c]))}")
                        print(f"                      W = {dict(zip(_var, _W_mat[_c]))}")
                    break

                J_k = _jac_fn(W_k, W_n_flat, dt)
                _nan_J = bool(jnp.any(jnp.isnan(J_k)))
                _inf_J = bool(jnp.any(jnp.isinf(J_k)))
                print(f"           max|J|={float(jnp.nanmax(jnp.abs(J_k))):.3e}"
                      f"  NaN={_nan_J}  Inf={_inf_J}")
                if _nan_J or _inf_J:
                    _J_np  = _np.array(J_k)
                    _bad_r = _np.where(_np.isnan(_J_np).any(1) | _np.isinf(_J_np).any(1))[0]
                    _bad_c = _np.where(_np.isnan(_J_np).any(0) | _np.isinf(_J_np).any(0))[0]
                    print(f"           bad J rows: {_bad_r[:8]}  → cells {_bad_r[:8]//6}  eqs {[_eq[r%6] for r in _bad_r[:8]]}")
                    print(f"           bad J cols: {_bad_c[:8]}  → cells {_bad_c[:8]//8}  vars {[_var[c%8] for c in _bad_c[:8]]}")
                    break

                _w_ref = jnp.maximum(jnp.abs(W_k), _w_char)
                _J_sc  = J_k * _w_ref[None, :]
                _r_sc  = 1.0 / (jnp.max(jnp.abs(_J_sc), axis=1) + 1e-300)  # row equilibration
                _dW_sc = _lin_solve(_J_sc * _r_sc[:, None], -R_k * _r_sc)
                dW     = _dW_sc * _w_ref
                _nan_dW = bool(jnp.any(jnp.isnan(dW)))
                print(f"           max|dW_rel|={float(jnp.nanmax(jnp.abs(_dW_sc))):.3e}"
                      f"  max|dW|={float(jnp.nanmax(jnp.abs(dW))):.3e}"
                      f"  NaN={_nan_dW}")
                W_k = W_k + dW
                # Hard clip α_g to (ε, 1-ε): physically required, guards EOS
                W_k = W_k.at[1::8].set(jnp.clip(W_k[1::8], _ALPHA_EPS, 1.0 - _ALPHA_EPS))
                _Wm = _np.array(W_k).reshape(N - 2, 8)
                print(f"           p=[{_Wm[:,0].min():.3e},{_Wm[:,0].max():.3e}]"
                      f"  ag=[{_Wm[:,1].min():.3f},{_Wm[:,1].max():.3f}]"
                      f"  u_g=[{_Wm[:,3].min():.2f},{_Wm[:,3].max():.2f}]"
                      f"  u_l=[{_Wm[:,5].min():.2f},{_Wm[:,5].max():.2f}]")
                if _nan_dW:
                    break
        else:
            def _newton_body(_, W_k):
                R_k   = _residual(W_k, W_n_flat, dt)        # (6*(N-2),)
                J_k   = _jac_fn(W_k, W_n_flat, dt)          # (6*(N-2), 8*(N-2))
                w_ref = jnp.maximum(jnp.abs(W_k), _w_char)
                J_sc  = J_k * w_ref[None, :]                # column scaling (relative dW)
                # Row equilibration: scale each residual row to O(1).  For the
                # underdetermined min-norm this is EXACTLY invariant (same solution
                # set) — it only fixes conditioning (cond ~1e8 → ~1e3 from the
                # mass/energy row-magnitude spread), not the answer.
                r_sc  = 1.0 / (jnp.max(jnp.abs(J_sc), axis=1) + 1e-300)
                dW_sc = _lin_solve(J_sc * r_sc[:, None], -R_k * r_sc)
                W_new = W_k + dW_sc * w_ref
                # Hard clip α_g to (ε, 1-ε): physically required, guards EOS
                W_new = W_new.at[1::8].set(jnp.clip(W_new[1::8], _ALPHA_EPS, 1.0 - _ALPHA_EPS))
                # Project each iterate onto the EOS manifold, INSIDE the loop, so
                # the Newton evaluates its next residual on an EOS-consistent state
                # and co-converges to a jointly conservation+EOS-consistent W.
                # (Projecting only ONCE after the loop changes the fluxes post-hoc
                # and fights the min-norm → dp≈1 at the inlet → adaptive-dt collapse.)
                if eos_project:
                    W_new = _project_interior(W_new)
                return W_new

            W_k = jax.lax.fori_loop(0, newton_iter, _newton_body, W_k)

        # Re-pin (p, α_g) to the conserved V — removes the min-norm EOS drift
        # (applies to both the debug and production Newton paths).
        if eos_project:
            W_k = _project_interior(W_k)

        W_new = jnp.concatenate([W_n[:1], W_k.reshape(N - 2, 8), W_n[-1:]], axis=0)

        # Steady-state residual (dV = 0 since same W passed for new and old)
        R_ss = _residual(
            W_new[1:-1].reshape(-1), W_new.reshape(-1), jnp.float64(1.0)
        ).reshape(N - 2, 6)

        U_h_new = _bc_U_h(jax.vmap(U_h_from_W)(W_new))
        lam_max = _max_wave_speed_cells(U_h_new)   # Städtke mixture wave speed
        t_char  = dx / (jnp.max(lam_max) + 1e-300)
        V_new   = jax.vmap(V_from_W)(W_new)
        ssr     = jnp.sqrt(jnp.mean(
            (t_char * R_ss
             / (V_cell[1:-1, None] * jnp.abs(V_new[1:-1]) + 1e-300)) ** 2
        ))

        # Write the BC-consistent ghost cells back into W so post-processing sees
        # the actual boundary state (transmissive outlet copy / fixed inlet), not
        # the stale input ghosts.  Re-applied every residual eval, so this does
        # not affect the iteration — only the returned/plotted state.  (2 EOS calls.)
        W_gh  = W_from_U_h_batch(jnp.stack([U_h_new[0], U_h_new[-1]]), fluid_g, fluid_l)
        W_new = W_new.at[0].set(W_gh[0]).at[-1].set(W_gh[-1])

        return W_new, ssr

    # JIT the production path (fori_loop is XLA-compatible).
    # debug_newton uses Python for-loops and print() — not JIT-able.
    if not debug_newton:
        step = jax.jit(step)

    # Diagnostics hook (Jacobian coloring / benchmarking): expose the raw
    # residual and its dense AD Jacobian.  Backward compatible — default off.
    if return_debug:
        return step, compute_dt, _residual, _jac_fn, jac_diff

    return step, compute_dt

# TODO: check if the time stepping works correctly. Are we actually computing the correct pipeeline with W. Also we need Hnc and D at the timestep n+1!
