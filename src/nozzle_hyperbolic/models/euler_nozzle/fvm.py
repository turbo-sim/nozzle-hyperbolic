"""
1D finite-volume solver for quasi-1D compressible Euler equations.

Governing equations (area-weighted conservative form):
    d(QA)/dt + d(FA)/dx = S_A

    Q = [rho, rho*u, rho*E]^T
    F = [rho*u,  rho*u^2+p,  (rho*E+p)*u]^T
    S_A = [0,  p*dA/dx - tau_w*4A/D_h,  0]^T

    The energy source is zero: at a no-slip adiabatic wall u_wall=0 so wall
    friction does no work on the fluid. Total enthalpy H = h + u^2/2 is
    conserved (consistent with Fanno-flow theory).

Semi-discrete FVM (dividing by A_i*dx):
    dQ_i/dt = -(FA)_{i+1/2} - (FA)_{i-1/2}) / (A_i*dx) + S_A,i / A_i
"""

import time as _time
import numpy as np
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from .flux import F_numeric
from scipy.optimize import brentq as _brentq

_FluidBicubic = getattr(jxp, 'FluidBicubic', None)

# Relocated definitions remain available here for existing callers.
from nozzle_hyperbolic.thermodynamics.recovery import (
    _get_state_ps,
    _get_state_hs,
    state_from_rho_u_batch,
    _state_rho_u,
)
from nozzle_hyperbolic.models.euler_nozzle.initialization import (
    isentropic_initialization,
    _shock_tube_velocities,
    shock_tube_Q_initialization,
    shock_tube_puh_initialization,
    stadtke_puh_initialization,
    stadtke_puh_initialization_subsonic,
    stadtke_puh_initialization_shocked,
    stadtke_puh_initialization_simple,
)
from nozzle_hyperbolic.models.euler_nozzle.equations import (
    euler_eigenvalues_perfect,
    euler_eigenvalues,
    euler_flux,
    euler_flux_perfect,
    euler_roe_matrix_perfect,
    euler_roe_matrix,
    _H_fix,
)
from nozzle_hyperbolic.models.euler_nozzle.boundaries import (
    _bc_inlet_perfect,
    _bc_outlet_perfect,
    _bc_inlet_real,
    _bc_outlet_real,
    _bc_inlet_real_jax,
    _bc_outlet_real_jax,
)
from nozzle_hyperbolic.models.euler_nozzle.sources import (
    _real_gas_state_and_friction,
    _source_term,
)
from nozzle_hyperbolic.models.euler_nozzle.closures.wall import (
    _haaland_jnp,
    Haaland,
    viscosity_Beattie,
    viscosity_MacAdams,
    LM_gronnerud,
    LM_condensation,
    f_HEM_purdue,
    richardson,
)
from nozzle_hyperbolic.numerics.reconstruction import (
    _minmod,
    _van_leer,
    _superbee,
    _van_albada,
    _mc,
    _LIMITERS,
    muscl_van_leer_ph,
    _muscl_reconstruct,
)
from nozzle_hyperbolic.models.euler_nozzle.reconstruction import (
    _muscl_reconstruct_prim,
)
from nozzle_hyperbolic.models.euler_nozzle.flux_jax import (
    _jax_euler_flux,
    _jax_F_rusanov,
    _jax_F_hll,
    _jax_F_lf,
    _jax_F_hllc,
    _jax_F_fvs_stadtke,
)
from nozzle_hyperbolic.models.euler_nozzle.state import (
    _conservative_to_puh,
    _puh_to_conservative,
)
from nozzle_hyperbolic.solvers.time_marching.cfl import (
    _compute_dt,
)



# ---------------------------------------------------------------------------
# Spatial operator:  R(Q) = -d(FA)/dx / A
# ---------------------------------------------------------------------------

def R(Q, parameters):
    """
    Flux divergence for all cells.

        RHS[:, i] = -(FA_{i+1/2} - FA_{i-1/2}) / (A_i * dx)

    Supports piecewise-constant (1st order) and MUSCL (2nd order) reconstruction
    controlled by parameters['reconstruction']:
        'constant'        — 1st-order Godunov
        'muscl_minmod'    — 2nd-order MUSCL with minmod limiter
        'muscl_van_leer'  — 2nd-order MUSCL with van Leer limiter
        'muscl_superbee'  — 2nd-order MUSCL with superbee limiter
    """
    _, Nj = Q.shape
    A     = parameters['A_nodes']
    dx    = parameters['dx']
    recon = parameters.get('reconstruction', 'constant')
    real  = not parameters['perfect_gas']

    if recon == 'constant':
        Q_L, Q_R = Q[:, :-1], Q[:, 1:]
        if real and 'p' in parameters and 'c' in parameters:
            parameters['p_L'] = parameters['p'][:-1]
            parameters['p_R'] = parameters['p'][1:]
            parameters['c_L'] = parameters['c'][:-1]
            parameters['c_R'] = parameters['c'][1:]
    elif recon in _LIMITERS:
        if real:
            Q_L, Q_R = _muscl_reconstruct_prim(Q, parameters, recon)
        else:
            Q_L, Q_R = _muscl_reconstruct(Q, recon)
    else:
        raise ValueError(f"Unknown reconstruction '{recon}'. "
                         f"Choose from: constant, {', '.join(_LIMITERS)}")

    FA  = F_numeric(Q_L, Q_R, parameters)   # shape (3, Nj-1)
    RHS = np.zeros((3, Nj))
    RHS[:, 1:-1] = -(FA[:, 1:] - FA[:, :-1]) / (A[1:-1] * dx)
    return RHS


def make_jax_step(fluid, A, dAdx, D_h, roughness, dx, CFL, p_out, h0, s0,
                  recon, flux_method, time_method, wall_friction=True,
                  transmissive_bc=False):
    """
    Returns a @jax.jit compiled step function for the real-gas Euler solver.
    BCs are applied inside step() — no Python calls needed between iterations.
    Returns (Q_next, p, c, dt) — all JAX arrays.
    """
    A_j   = jnp.asarray(A,    dtype=jnp.float64)
    D_h_j = jnp.asarray(D_h,  dtype=jnp.float64)
    A_L   = A_j[:-1];  A_R = A_j[1:]

    # dA/dx / A  — use analytical derivative from Geometry object
    _dA_dx_j = jnp.asarray(dAdx, dtype=jnp.float64) / A_j

    h_lo    = float(getattr(fluid, 'h_min', 50e3))  + 1.0
    h_hi    = float(getattr(fluid, 'h_max', 850e3)) - 1.0
    p_lo_r  = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi_r  = float(getattr(fluid, 'p_max', 300e5)) * 0.99
    h0_j    = jnp.float64(h0)
    s0_j    = jnp.float64(s0)

    _LIMS = {
        'muscl_minmod':   lambda a, b: jnp.where(jnp.sign(a) == jnp.sign(b),
                              jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0),
        'muscl_van_leer': lambda a, b: jnp.where(a * b > 0, 2 * a * b / (a + b + 1e-300), 0.0),
        'muscl_superbee': lambda a, b: jnp.maximum(
            jnp.where(jnp.sign(a) == jnp.sign(b), jnp.sign(a) * jnp.minimum(2*jnp.abs(a), jnp.abs(b)), 0.0),
            jnp.where(jnp.sign(a) == jnp.sign(b), jnp.sign(a) * jnp.minimum(jnp.abs(a), 2*jnp.abs(b)), 0.0)),
    }
    _limiter = _LIMS.get(recon)  # None → constant reconstruction

    _FLUX = {
        'hllc':    lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_hllc(QL, QR, pL, cL, pR, cR, A_L, A_R, fluid),
        'hll':     lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_hll(QL, QR, pL, cL, pR, cR, A_L, A_R),
        'rusanov': lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_rusanov(QL, QR, pL, cL, pR, cR, A_L, A_R),
        'lf':      lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_lf(QL, QR, pL, cL, pR, cR, A_L, A_R, dx, dt),
    }
    if flux_method not in _FLUX:
        raise ValueError(f"JAX step not supported for '{flux_method}' (Roe uses scalar loops). "
                         f"Choose from: {list(_FLUX)}")
    _flux_fn = _FLUX[flux_method]

    def _eos_batch(rho, e):
        p_from_hmin = (h_lo - e) * rho
        p_from_hmax = (h_hi - e) * rho
        p_lo = jnp.where((p_from_hmin > p_lo_r) & (p_from_hmin < p_hi_r),
                         jnp.maximum(p_lo_r, p_from_hmin), p_lo_r)
        p_hi = jnp.where((p_from_hmax > p_lo_r) & (p_from_hmax < p_hi_r),
                         jnp.minimum(p_hi_r, p_from_hmax), p_hi_r)
        bad  = p_lo >= p_hi
        p_lo = jnp.where(bad, p_lo_r, p_lo)
        p_hi = jnp.where(bad, p_hi_r, p_hi)
        p    = 0.5 * (p_lo + p_hi)
        def _bisect_eos(_, carry):
            p_lo, p_hi, p = carry
            h        = jnp.clip(e + p / rho, h_lo + 1.0, h_hi - 1.0)
            rho_pred = fluid.get_state(jxp.HmassP_INPUTS, h, p).density
            residual = rho_pred - rho
            p_lo     = jnp.where(residual < 0, p, p_lo)
            p_hi     = jnp.where(residual > 0, p, p_hi)
            return p_lo, p_hi, 0.5 * (p_lo + p_hi)
        p_lo, p_hi, p = jax.lax.fori_loop(0, 15, _bisect_eos, (p_lo, p_hi, p))
        h_f = jnp.clip(e + p / rho, h_lo + 1.0, h_hi - 1.0)
        _st = fluid.get_state(jxp.HmassP_INPUTS, h_f, p)
        return _st.pressure, _st.speed_sound, _st.viscosity

    def _recon(Q_n, p_c, c_c):
        if _limiter is None:
            return Q_n[:, :-1], Q_n[:, 1:], p_c[:-1], p_c[1:], c_c[:-1], c_c[1:]
        rho   = Q_n[0];  u = Q_n[1] / rho
        W     = jnp.stack([rho, u, p_c])
        dW    = W[:, 1:] - W[:, :-1]
        sigma = jnp.zeros_like(W).at[:, 1:-1].set(_limiter(dW[:, :-1], dW[:, 1:]))
        W_L_raw = W[:, :-1] + 0.5 * sigma[:, :-1]
        W_R_raw = W[:, 1:]  - 0.5 * sigma[:, 1:]
        W_L = W_L_raw.at[0].set(jnp.maximum(W_L_raw[0], 1e-6)).at[2].set(jnp.maximum(W_L_raw[2], 1.0))
        W_R = W_R_raw.at[0].set(jnp.maximum(W_R_raw[0], 1e-6)).at[2].set(jnp.maximum(W_R_raw[2], 1.0))
        rL, uL, pL = W_L[0], W_L[1], W_L[2]
        rR, uR, pR = W_R[0], W_R[1], W_R[2]
        stL  = fluid.get_state(jxp.DmassP_INPUTS, rL, pL)
        stR  = fluid.get_state(jxp.DmassP_INPUTS, rR, pR)
        Q_L  = jnp.stack([rL, rL * uL, rL * (stL.umass + 0.5 * uL**2)])
        Q_R  = jnp.stack([rR, rR * uR, rR * (stR.umass + 0.5 * uR**2)])
        return Q_L, Q_R, pL, pR, stL.speed_sound, stR.speed_sound

    def _R(Q_n, p_c, c_c, dt):
        Q_L, Q_R, p_L, p_R, c_L, c_R = _recon(Q_n, p_c, c_c)
        FA   = _flux_fn(Q_L, Q_R, p_L, c_L, p_R, c_R, dt)
        RHS  = jnp.zeros_like(Q_n).at[:, 1:-1].set(
                   -(FA[:, 1:] - FA[:, :-1]) / (A_j[1:-1] * dx))
        return RHS

    def _bc_inlet(Q_n):
        if transmissive_bc:
            return Q_n.at[:, 0].set(Q_n[:, 1])
        # Extrapolate velocity; find isentropic static state via 12-iter bisection.
        # Uses 1-element arrays so FluidBicubic's vmap path is exercised, not scalar.
        u_in  = 2.0 * Q_n[1, 1] / Q_n[0, 1] - Q_n[1, 2] / Q_n[0, 2]
        h_s   = jnp.expand_dims(h0_j - 0.5 * u_in**2, 0)   # shape (1,)
        plo   = jnp.array([p_lo_r], dtype=jnp.float64)
        phi   = jnp.array([p_hi_r], dtype=jnp.float64)
        def _bisect_inlet(_, carry):
            plo, phi = carry
            pm    = 0.5 * (plo + phi)
            s_mid = fluid.get_state(jxp.HmassP_INPUTS, h_s, pm).entropy
            plo   = jnp.where(s_mid > s0_j, pm, plo)
            phi   = jnp.where(s_mid < s0_j, pm, phi)
            return plo, phi
        plo, phi = jax.lax.fori_loop(0, 12, _bisect_inlet, (plo, phi))
        p_s   = 0.5 * (plo + phi)
        st    = fluid.get_state(jxp.HmassP_INPUTS, h_s, p_s)
        rho_s = st.rhomass[0]
        e_s   = st.umass[0]
        Q_n   = Q_n.at[0, 0].set(rho_s)
        Q_n   = Q_n.at[1, 0].set(rho_s * u_in)
        Q_n   = Q_n.at[2, 0].set(rho_s * (e_s + 0.5 * u_in**2))
        return Q_n

    def _bc_outlet(Q_n):
        if transmissive_bc:
            return Q_n.at[:, -1].set(Q_n[:, -2])
        rho_out  = 2.0 * Q_n[0, -2] * A_j[-2] / A_j[-1] - Q_n[0, -3] * A_j[-3] / A_j[-1]
        rhou_out = 2.0 * Q_n[1, -2] * A_j[-2] / A_j[-1] - Q_n[1, -3] * A_j[-3] / A_j[-1]
        # Clamp rho to physical range before EOS call (high-CFL extrapolation can go negative)
        rho_out  = jnp.clip(rho_out, 1.0, 1e5)
        u_out    = rhou_out / rho_out
        # Find h_out s.t. rho(h_out, p_out) = rho_out via bisection on HmassP_INPUTS
        # (avoids DmassP_INPUTS which uses an internal Newton solver that can fail).
        # rho decreases with h at constant p, so the bracket [h_lo, h_hi] always works.
        rho_t  = jnp.expand_dims(rho_out, 0)
        p_arr  = jnp.array([p_out], dtype=jnp.float64)
        hlo    = jnp.array([h_lo],  dtype=jnp.float64)
        hhi    = jnp.array([h_hi],  dtype=jnp.float64)
        def _bisect_outlet(_, carry):
            hlo, hhi = carry
            hm  = 0.5 * (hlo + hhi)
            rhm = fluid.get_state(jxp.HmassP_INPUTS, hm, p_arr).rhomass
            hlo = jnp.where(rhm > rho_t, hm, hlo)
            hhi = jnp.where(rhm < rho_t, hm, hhi)
            return hlo, hhi
        hlo, hhi = jax.lax.fori_loop(0, 15, _bisect_outlet, (hlo, hhi))
        h_out  = 0.5 * (hlo + hhi)
        st_out = fluid.get_state(jxp.HmassP_INPUTS, h_out, p_arr)
        e_out  = st_out.umass[0]
        Q_n    = Q_n.at[0, -1].set(rho_out)
        Q_n    = Q_n.at[1, -1].set(rhou_out)
        Q_n    = Q_n.at[2, -1].set(rho_out * (e_out + 0.5 * u_out**2))
        return Q_n

    @jax.jit
    def step(Q_n):
        Q_n = _bc_inlet(Q_n)
        Q_n = _bc_outlet(Q_n)

        rho = Q_n[0];  u = Q_n[1] / rho
        e   = Q_n[2] / rho - 0.5 * u**2

        p, c, mu = _eos_batch(rho, e)
        p = p.at[-1].set(p_out)

        Re    = rho * jnp.abs(u) * D_h_j / jnp.maximum(mu, 1e-12)
        tau_w = _haaland_jnp(Re, roughness, D_h_j) * rho * u**2 / 8.0
        tau_w = tau_w if wall_friction else jnp.zeros_like(tau_w)

        S = jnp.stack([jnp.zeros_like(p), p * _dA_dx_j - tau_w * 4.0 / D_h_j, jnp.zeros_like(p)])

        # ── Local Time Stepping (LTS) ─────────────────────────────────────────
        # Each cell advances at dt_i = CFL * dx / (|u_i| + c_i) instead of the
        # global minimum.  Subsonic cells away from the throat use a much larger
        # dt_i, removing the throat bottleneck and accelerating convergence to
        # steady state without affecting the final solution.
        lam    = jnp.abs(u) + c
        dt_vec = CFL * dx / jnp.maximum(lam, 1e-10)   # (Nj,) per-cell time step
        dt_min = jnp.min(dt_vec)                        # scalar: throat-limited dt
        dt_b   = dt_vec[jnp.newaxis, :]                 # (1, Nj) for broadcasting

        # Evaluate the steady-state residual R(Q)+S once at Q_n.
        # This is the true convergence metric: zero at steady state regardless of CFL.
        # Reusing it in the time update also saves one flux call for RK methods.
        # dt_min passed to _R only for the Lax-Friedrichs flux (ignored by others).
        RHS_n = _R(Q_n, p, c, dt_min) + S

        # Normalised steady-state residual (SSR): dimensionless, CFL-independent.
        # t_min = dt_min/CFL = dx/max(lam): physical advection time at the throat.
        t_min = dt_min / CFL
        ssr = jnp.sqrt(jnp.mean(
            (t_min * RHS_n[:, 1:-1] / (jnp.abs(Q_n[:, 1:-1]) + 1e-300))**2))

        if time_method == 'euler':
            Q_new = Q_n + dt_b * RHS_n
        elif time_method == 'RK3':
            Q1    = Q_n + dt_b * RHS_n                     # stage 1: reuse RHS_n
            R2    = _R(Q1, p, c, dt_min)
            Q2    = 0.75 * Q_n + 0.25 * (Q1 + dt_b * (R2 + S))
            R3    = _R(Q2, p, c, dt_min)
            Q_new = (Q_n + 2.0 * (Q2 + dt_b * (R3 + S))) / 3.0
        elif time_method == 'RK4':
            k1    = RHS_n                                   # reuse RHS_n
            k2    = _R(Q_n + 0.5 * dt_b * k1, p, c, dt_min) + S
            k3    = _R(Q_n + 0.5 * dt_b * k2, p, c, dt_min) + S
            k4    = _R(Q_n +       dt_b * k3,  p, c, dt_min) + S
            Q_new = Q_n + (dt_b / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        elif time_method == 'implicit':
            # Per-cell denom: each cell uses its own dt_vec[i], giving uniformly
            # strong damping across the domain — not just at the throat.
            lam_face = jnp.maximum(lam[:-1], lam[1:])
            lam_L    = jnp.concatenate([lam_face[:1],  lam_face])
            lam_R    = jnp.concatenate([lam_face, lam_face[-1:]])
            denom    = 1.0 + dt_vec * (lam_L + lam_R) / dx   # (Nj,) per-cell
            Q_new    = Q_n + dt_b * RHS_n / denom[jnp.newaxis, :]
        else:
            Q_new = Q_n + dt_b * RHS_n  # fallback to Euler

        return Q_new, p, c, dt_min, ssr

    return step

def make_jax_stadtke_step(fluid, A, A_face, dAdx, D_h, roughness, dx, CFL, p_out, h0, s0,
                           delta_factor=0.05, newton_iter=20, wall_friction=True,
                           transmissive_bc=False, recon='muscl_van_leer',
                           jacobian_mode='dense',
                           eps_filter=0.0, eps_filter_ucut=None, eps_filter_uwidth=None,
                           linear_solver='dense', newton_tol=0.0,
                           flux_local_jac=False, cell_local_jac=False):
    """
    Build a @jax.jit step function implementing the Städtke Newton implicit scheme.

    The solver works in primitive (p, u, h) variables:
      - recon='muscl_van_leer' (default) or 'constant' (1st-order) reconstruction in (p,u,h).
      - Städtke FVS interface flux  (_jax_F_fvs_stadtke).
      - Fully implicit residual assembled in conservative form.
      - Newton-Raphson: Jacobian via jax.jacobian, solve via jnp.linalg.solve.
      - `newton_iter` iterations are unrolled inside @jax.jit.

    Speed-up switches (dev_scripts/SPEEDUP_REPORT_FVS_SINGLE_PHASE.md).  The
    defaults reproduce the original code bit-for-bit.
      linear_solver  : 'dense' (default) — dense J + jnp.linalg.solve, O(N³);
                       'banded' — 3×3 blocks read straight from the colored
                       JVP columns (no dense matrix) and a block-tridiagonal LU
                       on super-blocks of `stencil width` cells, O(N).  Same
                       Newton step to ~1e-12.  Uses the colored compression
                       whatever `jacobian_mode` says.
      newton_tol     : 0.0 (default) = always `newton_iter` iterations.  > 0 stops
                       once max|ΔU/U_ref| < newton_tol (lax.while_loop;
                       newton_iter stays the cap).  Not reverse-differentiable.
      flux_local_jac : custom JVP on the face flux: each face is linearized once
                       (exact 3×6 block), each colored seed becomes a mat-vec.
      cell_local_jac : same for the per-cell EOS outputs (ρ, a, μ; 3×3 block).

    step(U_n) → (U_new, dt, ssr)
      U_n   : (N, 3) — primitive state at time n
      U_new : (N, 3) — primitive state at time n+1
      dt    : scalar — time step used
      ssr   : scalar — max relative change (convergence metric)
    """
    if linear_solver not in ('dense', 'banded'):
        raise ValueError(f"linear_solver must be 'dense' or 'banded', got {linear_solver!r}")
    N      = len(A)
    A_j    = jnp.asarray(A,   dtype=jnp.float64)
    D_h_j  = jnp.asarray(D_h, dtype=jnp.float64)
    # A_face = 0.5 * (A_j[:-1] + A_j[1:])   # (N-1,) — face areas
    V_cell = A_j * dx                       # (N,)   — cell volumes

    # dA/dx / A  — analytical derivative from Geometry object
    dAdx_A = jnp.asarray(dAdx, dtype=jnp.float64) / A_j  # (N,)

    h_lo  = float(getattr(fluid, 'h_min', 50e3))  + 1.0
    h_hi  = float(getattr(fluid, 'h_max', 850e3)) - 1.0
    p_lo  = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi  = float(getattr(fluid, 'p_max', 300e5)) * 0.99
    h0_j  = jnp.float64(h0)
    s0_j  = jnp.float64(s0)
    pout_j = jnp.float64(p_out)

    # ── Inner functions (all JAX-traceable) ──────────────────────────────

    def _V_from_U(U):
        """Primitive (N,3) → conservative V=(ρ, ρu, ρH−p) (N,3)."""
        p_, u_, h_ = U[:, 0], U[:, 1], U[:, 2]
        st   = fluid.get_state(jxp.HmassP_INPUTS, h_, p_)
        rho_ = st["rho"]
        H_   = h_ + 0.5 * u_**2
        return jnp.stack([rho_, rho_ * u_, rho_ * H_ - p_], axis=1)

    def _muscl_ph(U):
        """MUSCL reconstruction in (p,u,h) with the selected limiter.

        Per-face fallback to 1st-order when the reconstructed p or h would
        land outside the EOS table bounds (spec §6.4).
        """
        dU    = U[1:] - U[:-1]                           # (N-1, 3)
        if recon == 'muscl_van_albada':
            # Van Albada is smooth (no r=1 kink) for same-sign inputs, better Jacobian
            # than minmod/van_leer there. But ab(a+b)/(a²+b²) is non-zero for opposite
            # signs when a+b≠0 → anti-diffusion → NaN. Must zero out for a·b ≤ 0.
            lim = lambda a, b: jnp.where(
                a * b > 0,
                (a**2 * b + a * b**2) / (a**2 + b**2 + 1e-300),
                0.0)
        elif recon == 'muscl_minmod':
            lim = lambda a, b: jnp.where(
                a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
        elif recon == 'muscl_mc':
            lim = lambda a, b: jnp.where(
                a * b > 0,
                jnp.sign(a) * jnp.minimum(jnp.minimum(2*jnp.abs(a), 2*jnp.abs(b)),
                                           0.5*jnp.abs(a + b)), 0.0)
        else:  # muscl_van_leer (default)
            lim = lambda a, b: jnp.where(a * b > 0, 2.0 * a * b / (a + b + 1e-300), 0.0)
        sigma = jnp.zeros_like(U).at[1:-1].set(lim(dU[:-1], dU[1:]))
        U_L_m = U[:-1] + 0.5 * sigma[:-1]   # MUSCL candidate
        U_R_m = U[1:]  - 0.5 * sigma[1:]    # MUSCL candidate

        # First-order fallback states (cell-centred)
        U_L_0 = U[:-1]
        U_R_0 = U[1:]

        # Fall back if p or h of EITHER side is outside EOS table bounds
        bad_L = ((U_L_m[:, 0] < p_lo) | (U_L_m[:, 0] > p_hi) |
                 (U_L_m[:, 2] < h_lo) | (U_L_m[:, 2] > h_hi))
        bad_R = ((U_R_m[:, 0] < p_lo) | (U_R_m[:, 0] > p_hi) |
                 (U_R_m[:, 2] < h_lo) | (U_R_m[:, 2] > h_hi))
        bad   = (bad_L | bad_R)[:, None]    # broadcast over 3 primitives

        U_L = jnp.where(bad, U_L_0, U_L_m)
        U_R = jnp.where(bad, U_R_0, U_R_m)
        return U_L, U_R

    def _bc_inlet(U):
        """Inlet BC: zero-gradient (shock tube) or isentropic reservoir (nozzle).

        Solves p on the reservoir isentrope, s(h_s, p) = s0, with 3 unrolled
        Newton steps using the exact thermodynamic derivative
            (∂s/∂p)|_h = -1/(ρT)   →   Δp = ρT (s - s0).
        Initial guess = pressure of the first interior cell (near-exact at
        steady state).  Replaces a 12-iteration fori_loop bisection whose
        repeated retracing under jax.jacobian dominated the step cost:
        3 inline EOS calls instead of a traced loop with 12.
        """
        if transmissive_bc:
            return U.at[0].set(U[1])
        u_in = 2.0 * U[1, 1] - U[2, 1]
        h_s  = jnp.expand_dims(
                   jnp.clip(h0_j - 0.5 * u_in**2,
                             jnp.float64(h_lo + 1.0),
                             jnp.float64(h_hi - 1.0)), 0)
        p_s  = jnp.expand_dims(jnp.clip(U[1, 0], p_lo, p_hi), 0)
        for _ in range(3):   # unrolled — no fori_loop, no retrace overhead
            st  = fluid.get_state(jxp.HmassP_INPUTS, h_s, p_s)
            p_s = jnp.clip(p_s + st["rho"] * st["T"] * (st["s"] - s0_j),
                            p_lo, p_hi)
        return U.at[0].set(jnp.array([p_s[0], u_in, h_s[0]]))

        # ── BISECTION VERSION (reference) — commented out ─────────────────────
        # 12-iter fori_loop bisection on the isentrope. Robust but the loop is
        # retraced inside every jax.jacobian context → dominated step cost.
        #
        # plo  = jnp.array([p_lo], dtype=jnp.float64)
        # phi  = jnp.array([p_hi], dtype=jnp.float64)
        # def _bisect_inlet(_, carry):
        #     plo, phi = carry
        #     pm    = 0.5 * (plo + phi)
        #     s_mid = fluid.get_state(jxp.HmassP_INPUTS, h_s, pm)["s"]
        #     plo   = jnp.where(s_mid > s0_j, pm, plo)
        #     phi   = jnp.where(s_mid < s0_j, pm, phi)
        #     return plo, phi
        # plo, phi = jax.lax.fori_loop(0, 12, _bisect_inlet, (plo, phi))
        # p_s = 0.5 * (plo + phi)
        # return U.at[0].set(jnp.array([p_s[0], u_in, h_s[0]]))

    def _bc_outlet(U):
        """Outlet BC in primitive (p, u, h) space.

        Subsonic (M < 1):
          u_ghost  = area-weighted linear extrapolation of ρuA / ρ_ghost
          h_ghost  = linear extrapolation  2·h[-2] - h[-3]
          p_ghost  = p_out  (directly imposed — no bisection needed in primitive space)

        Supersonic (M ≥ 1):
          all variables extrapolated linearly (no pressure fixing)

        The M-switch is a smooth tanh so jax.jacobian sees a non-zero gradient
        near M=1 and Newton can propagate p_out upstream.

        NOTE: a bisection version that also enforces ρ(h_ghost, p_out) = ρ_ghost
        (area-weighted extrapolation of ρA, §5.2 reference) is kept below as
        reference. It is correct but adds 17 EOS calls per residual evaluation,
        making the Jacobian ~17× more expensive in the BC rows.
        """
        if transmissive_bc:
            return U.at[-1].set(U[-2])

        h_m2 = U[-2, 2];  p_m2 = U[-2, 0];  u_m2 = U[-2, 1]
        h_m3 = U[-3, 2];  p_m3 = U[-3, 0];  u_m3 = U[-3, 1]

        # EOS at last two interior cells — needed for ρ (momentum extrapolation) and a (Mach)
        st_m2  = fluid.get_state(jxp.HmassP_INPUTS,
                                  jnp.expand_dims(h_m2, 0), jnp.expand_dims(p_m2, 0))
        st_m3  = fluid.get_state(jxp.HmassP_INPUTS,
                                  jnp.expand_dims(h_m3, 0), jnp.expand_dims(p_m3, 0))
        rho_m2 = st_m2["rho"][0];  a_m2 = st_m2["a"][0]
        rho_m3 = st_m3["rho"][0]

        # ── area-weighted extrapolation of ρuA → u_ghost ─────────────────────
        rho_g  = jnp.clip(
            (2.0*rho_m2*A_j[-2] - rho_m3*A_j[-3]) / A_j[-1], 1e-3, 1e4)
        rhou_g = (2.0*rho_m2*u_m2*A_j[-2] - rho_m3*u_m3*A_j[-3]) / A_j[-1]
        u_g    = rhou_g / rho_g

        # ── smooth Mach blend weight ─────────────────────────────────────────
        M_out = jnp.abs(u_m2) / (a_m2 + 1e-300)
        w     = 0.5 * (1.0 + jnp.tanh((1.0 - M_out) / 0.004))  # 1 subsonic, 0 supersonic

        # ── subsonic: p_out imposed, h extrapolated ───────────────────────────
        h_sub = 2.0*h_m2 - h_m3
        p_sub = pout_j

        # ── supersonic: full linear extrapolation ─────────────────────────────
        h_sup = 2.0*h_m2 - h_m3   # same extrapolation — blend removes p_out
        p_sup = 2.0*p_m2 - p_m3

        # ── blend ─────────────────────────────────────────────────────────────
        h_g = jnp.clip(w * h_sub + (1.0 - w) * h_sup, h_lo, h_hi)
        p_g = w * p_sub + (1.0 - w) * jnp.clip(p_sup, p_lo, p_hi)
        return U.at[-1].set(jnp.array([p_g, u_g, h_g]))

        # ── BISECTION VERSION (reference, §5.2) — commented out ───────────────
        # Enforces ρ(h_ghost, p_out) = ρ_ghost (area-weighted ρA extrapolation).
        # Correct but adds 15 EOS calls per residual eval → ~17× BC cost.
        #
        # rho_t = jnp.expand_dims(rho_g, 0)
        # p_arr = jnp.full((1,), pout_j, dtype=jnp.float64)
        # hlo   = jnp.full((1,), h_lo,   dtype=jnp.float64)
        # hhi   = jnp.full((1,), h_hi,   dtype=jnp.float64)
        # def _bisect_out(_, carry):
        #     hlo, hhi = carry
        #     hm  = 0.5 * (hlo + hhi)
        #     rhm = fluid.get_state(jxp.HmassP_INPUTS, hm, p_arr)["rho"]
        #     hlo = jnp.where(rhm > rho_t, hm, hlo)
        #     hhi = jnp.where(rhm < rho_t, hm, hhi)
        #     return hlo, hhi
        # hlo, hhi = jax.lax.fori_loop(0, 15, _bisect_out, (hlo, hhi))
        # h_sub = 0.5 * (hlo[0] + hhi[0])
        # p_sub = pout_j

    # ── SPEED-UP (flux_local_jac / cell_local_jac): local custom JVPs ────────
    # The colored Jacobian pushes 15 seeds (9 for constant recon) through the
    # residual.  Each face flux depends on 6 inputs and each cell's EOS outputs
    # on 3, so linearizing them once (exact local block) and serving every seed
    # as a small mat-vec replaces 15 tangent passes by 6 (faces) / 3 (cells).
    # Chain rule regrouped: values unchanged, derivatives to ~1e-15.
    def _flux_one(ul, ur):
        return _jax_F_fvs_stadtke(ul[None], ur[None], fluid, delta_factor)[0]

    def _flux_and_jac_one(ul, ur):
        xf = jnp.concatenate([ul, ur])
        F, lin = jax.linearize(lambda x_: _flux_one(x_[:3], x_[3:]), xf)
        return F, jax.vmap(lin, out_axes=1)(jnp.eye(6, dtype=xf.dtype))   # (3,6)

    @jax.custom_jvp
    def _flux_faces_local(U_L, U_R):
        return _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)

    @_flux_faces_local.defjvp
    def _flux_faces_local_jvp(primals, tangents):
        (U_L, U_R), (dL, dR) = primals, tangents
        F, Jx = jax.vmap(_flux_and_jac_one)(U_L, U_R)
        return F, jnp.einsum('fij,fj->fi', Jx, jnp.concatenate([dL, dR], axis=1))

    def _eos_cells(U):
        st = fluid.get_state(jxp.HmassP_INPUTS, U[:, 2], U[:, 0])
        return jnp.stack([st["rho"], st["a"], st["mu"]], axis=1)

    @jax.custom_jvp
    def _eos_cells_local(U):
        return _eos_cells(U)

    @_eos_cells_local.defjvp
    def _eos_cells_local_jvp(primals, tangents):
        (U,), (dU,) = primals, tangents

        def one(u):
            y, lin = jax.linearize(lambda v: _eos_cells(v[None])[0], u)
            return y, jax.vmap(lin, out_axes=-1)(jnp.eye(3, dtype=u.dtype))
        Y, Jc = jax.vmap(one)(U)
        return Y, jnp.einsum('bij,bj->bi', Jc, dU)

    def _residual(U_int_flat, U_n_flat, dt):
        """
        Conservative residual for interior cells 1..N-2.

        U_int_flat : (3*(N-2),) — Newton unknowns (interior cells only)
        U_n_flat   : (3*N,)     — full state at time n (fixed)
        Returns    : (3*(N-2),) — residual R (zero at steady state)
        """
        U_int = U_int_flat.reshape(N - 2, 3)
        U_n   = U_n_flat.reshape(N, 3)

        # Assemble full domain; BCs overwrite ghost cells 0 and N-1
        U   = jnp.concatenate([U_n[:1], U_int, U_n[-1:]], axis=0)
        U   = _bc_inlet(U)
        U   = _bc_outlet(U)

        if cell_local_jac:
            E_U   = _eos_cells_local(U)                      # (N,3): ρ, a, μ
            H_U   = U[:, 2] + 0.5 * U[:, 1]**2
            V_new = jnp.stack([E_U[:, 0], E_U[:, 0] * U[:, 1],
                               E_U[:, 0] * H_U - U[:, 0]], axis=1)
        else:
            V_new = _V_from_U(U)
        V_old = _V_from_U(U_n)

        U_L, U_R = _muscl_ph(U) if recon != 'constant' else (U[:-1], U[1:])
        if flux_local_jac:
            F_hat = _flux_faces_local(U_L, U_R)                         # (N-1, 3)
        else:
            F_hat = _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)  # (N-1, 3)

        # Source: area variation + wall friction — momentum only (theory §7.3)
        if cell_local_jac:
            st_U = {"rho": E_U[:, 0], "a": E_U[:, 1], "mu": E_U[:, 2]}
        else:
            st_U = fluid.get_state(jxp.HmassP_INPUTS, U[:, 2], U[:, 0])

        # ── 4th-order (JST-type) background filter — port of the two-fluid
        #    eps_void filter (two_fluid/solver.py), 2026-07-14 ─────────────────
        # Cures the low-Mach PRESSURE odd-even (2Δx) decoupling: at Ma ≪ 1 the
        # FVS leaves a p-checkerboard mode undamped (proven by the two-fluid
        # eigen-analysis: raw operator has unstable p-2Δx modes, the 4th-order
        # filter takes them to 0 unstable).  Conservative flux (telescopes →
        # mass/momentum/energy exactly conserved), scaled by the local spectral
        # radius |u|+a; 4th-order ⇒ perturbs the smooth solution only at
        # O(eps·Δx³) — verify ε-independence.  eps_filter=0 → identically OFF
        # (default; behaviour unchanged).  Optional velocity gate as in the
        # two-fluid filter: w = ½(1+tanh((u0−|u|)/Δu)) turns it off above
        # eps_filter_ucut [m/s] (None → ungated/uniform).
        if eps_filter > 0.0:
            lam_c  = jnp.abs(U[:, 1]) + st_U["a"]              # (N,) |u| + a
            r_face = 0.5 * (lam_c[:-1] + lam_c[1:])            # (N-1,)
            if eps_filter_ucut is not None:
                _u0 = jnp.float64(eps_filter_ucut)
                _du = jnp.float64(eps_filter_uwidth if eps_filter_uwidth is not None
                                  else eps_filter_ucut / 3.0)
                w_c    = 0.5 * (1.0 + jnp.tanh((_u0 - jnp.abs(U[:, 1])) / _du))
                r_face = r_face * 0.5 * (w_c[:-1] + w_c[1:])
            third  = (V_new[3:] - 3.0 * V_new[2:-1]
                      + 3.0 * V_new[1:-2] - V_new[:-3])        # (N-3,3): faces 1..N-3
            F_hat  = F_hat.at[1:-1].add(eps_filter * r_face[1:-1, None] * third)
        rho_U = st_U["rho"]
        mu_U  = st_U["mu"]
        Re    = rho_U * jnp.abs(U[:, 1]) * D_h_j / jnp.maximum(mu_U, 1e-12)
        tau_w = _haaland_jnp(Re, roughness, D_h_j) * rho_U * U[:, 1]**2 / 8.0
        tau_w = tau_w if wall_friction else jnp.zeros_like(tau_w)
        src   = jnp.zeros((N, 3)).at[:, 1].set(
                    U[:, 0] * dAdx_A - tau_w * 4.0 / D_h_j)

        # Residual for interior cells 1..N-2  (indices in arrays: [1:-1])
        # Right face of cell i is face i; left face is face i-1.
        # For cells 1..N-2: right faces = F_hat[1:], left faces = F_hat[:-1]
        Vc    = V_cell[1:-1, None]                                         # (N-2, 1)
        dV    = V_new[1:-1] - V_old[1:-1]                                  # (N-2, 3)
        fdiv  = A_face[1:, None] * F_hat[1:] - A_face[:-1, None] * F_hat[:-1]  # (N-2, 3)
        R     = Vc * dV + dt * fdiv - dt * Vc * src[1:-1]
        return R.flatten()

    # ── Jacobian: dense (exact AD) or colored (banded compression) ───────────
    # The residual is block-pentadiagonal: row for cell i depends only on cells
    # i-2..i+2 (MUSCL ±2 stencil; constant ±1).  Both BCs are 2-cell-local, so
    # ghost substitution folds dependencies INWARD — it never widens the band.
    # Coloring exploits this: columns ≥ (2*sw+1) cells apart never share a
    # residual row, so they can be probed together in one JVP and de-interleaved
    # exactly.  The number of colors is fixed by the stencil — INDEPENDENT of N
    # (15 for MUSCL, 9 for constant) — so cost scales O(N) instead of O(N²).
    #
    # CAVEAT: exact ONLY while every residual row stays within ±stencil_width
    # cells (local BCs, local stencil).  Any non-local term (global constraint,
    # wider stencil, BC reading far cells) breaks it → wrong Jacobian.  The
    # startup `jacobian_validate` check (max|J_colored - J_dense|) is the guard.
    # eps_filter>0 adds a ±2-cell 4th-order stencil → widen the band even for
    # constant recon so the colored Jacobian stays exact (guard: jacobian_validate).
    _stencil_width = 1 if (recon == 'constant' and eps_filter == 0.0) else 2
    _cell_stride   = 2 * _stencil_width + 1          # cells between same-color columns
    _stride        = 3 * _cell_stride                # DOF stride (3 vars/cell)
    _m_dof         = 3 * (N - 2)                      # total interior DOFs
    _n_colors      = min(_stride, _m_dof)             # graceful if N tiny

    _jac_dense = jax.jacobian(_residual, argnums=0)

    def _colored_columns(U_int_flat, U_n_flat, dt):
        m_loc = U_int_flat.shape[0]
        # Linearize once: the expensive nonlinear residual (EOS chain) runs a
        # single time; each colored seed is then a cheap linear JVP.
        _, jvp_fn = jax.linearize(lambda u: _residual(u, U_n_flat, dt), U_int_flat)
        cols_idx  = jnp.arange(m_loc)
        color_ids = jnp.arange(_n_colors)
        S  = (cols_idx[None, :] % _stride == color_ids[:, None]).astype(U_int_flat.dtype)
        return jax.vmap(jvp_fn)(S)                     # (n_colors, m): summed columns

    def _jac_colored(U_int_flat, U_n_flat, dt):
        m_loc = U_int_flat.shape[0]
        Jc    = _colored_columns(U_int_flat, U_n_flat, dt)
        # De-interleave: for each row r (cell lr=r//3), the in-band columns are
        # cells lr-sw..lr+sw × 3 vars; each has a unique color within the band,
        # so J[r, c] = Jc[color(c), r].
        rows = jnp.arange(m_loc)
        lr   = rows // 3
        J    = jnp.zeros((m_loc, m_loc), dtype=U_int_flat.dtype)
        for dc in range(-_stencil_width, _stencil_width + 1):
            for vc in range(3):
                lc     = lr + dc
                c      = lc * 3 + vc
                valid  = (lc >= 0) & (lc <= (N - 3))
                c_safe = jnp.clip(c, 0, m_loc - 1)
                val    = jnp.where(valid, Jc[c_safe % _stride, rows], 0.0)
                J      = J.at[rows, c_safe].add(val)   # invalid → adds 0 at clipped col
        return J

    _jac_fn = _jac_colored if jacobian_mode == 'colored' else _jac_dense

    # ── SPEED-UP (linear_solver='banded'): O(N) Newton solve ─────────────────
    # The dense path builds a (3(N-2))² matrix and LU-factors it: O(N³), ~0.5 s
    # per solve at N=500 (99 % of the step).  The Jacobian is block-banded with
    # 3×3 blocks at cell offsets -sw..+sw, so:
    #   1. read the blocks straight from the colored columns (no dense matrix);
    #   2. group sw cells into one super-block (3sw×3sw) → block-TRIDIAGONAL;
    #   3. block Thomas: one LU per super-block, [G_i | g_i] = D'_i⁻¹[U_i | b'_i],
    #      back substitution x_i = g_i − G_i x_{i+1} (mat-vecs only).
    # Pivoting happens inside each super-block only (no pivoting across
    # blocks); matched the dense LU to ~1e-12 on every tested case, including
    # SER at CFL 50.  The sparsity guard is the same as for the colored
    # Jacobian (jacobian_validate).
    _Nc   = N - 2                                 # interior cells
    _sw   = _stencil_width
    _D    = 2 * _sw + 1                           # block offsets per row
    _s    = _sw                                   # cells per super-block
    _M    = -(-_Nc // _s)                         # number of super-blocks
    _Ncp  = _M * _s                               # padded cell count
    _S3   = 3 * _s                                # super-block size

    def _jac_blocks(U_int_flat, U_n_flat, dt):
        """J_blk[i, d] = ∂R_i/∂U_{i+d-sw}  (Nc, D, 3, 3), from the colored columns."""
        Jc  = _colored_columns(U_int_flat, U_n_flat, dt)
        i   = jnp.arange(_Nc)
        blk = []
        for d in range(_D):
            lc    = i + d - _sw
            valid = (lc >= 0) & (lc <= _Nc - 1)
            cols  = []
            for vc in range(3):
                col  = (lc * 3 + vc) % _stride
                vals = jnp.stack([Jc[col, 3 * i + vr] for vr in range(3)], axis=1)
                cols.append(jnp.where(valid[:, None], vals, 0.0))
            blk.append(jnp.stack(cols, axis=2))         # (Nc, 3, 3)  [row var, col var]
        return jnp.stack(blk, axis=1)                   # (Nc, D, 3, 3)

    def _banded_solve(J_blk, R):
        """Solve J dU = -R with J given as 3×3 blocks (block-tridiagonal LU)."""
        Jp = jnp.pad(J_blk, ((0, _Ncp - _Nc), (0, 0), (0, 0), (0, 0)))
        if _Ncp > _Nc:                                  # padded cells: identity rows
            Jp = Jp.at[_Nc:, _sw].set(jnp.eye(3, dtype=Jp.dtype))
        Dm = jnp.zeros((_M, _S3, _S3), Jp.dtype)        # diagonal super-blocks
        Um = jnp.zeros((_M, _S3, _S3), Jp.dtype)        # upper  (super-block k+1)
        Lm = jnp.zeros((_M, _S3, _S3), Jp.dtype)        # lower  (super-block k-1)
        for a in range(_s):                             # local row cell
            for d in range(_D):
                off = a + d - _sw                       # local column cell
                blk = Jp[a::_s, d]                      # (M, 3, 3)
                if 0 <= off < _s:
                    Dm = Dm.at[:, 3*a:3*a+3, 3*off:3*off+3].set(blk)
                elif off >= _s:
                    o = off - _s
                    Um = Um.at[:, 3*a:3*a+3, 3*o:3*o+3].set(blk)
                else:
                    o = off + _s
                    Lm = Lm.at[:, 3*a:3*a+3, 3*o:3*o+3].set(blk)
        b = jnp.pad(-R, (0, 3 * (_Ncp - _Nc))).reshape(_M, _S3)

        def fwd(carry, inp):
            G_prev, g_prev = carry
            D_i, U_i, L_i, b_i = inp
            Dp = D_i - L_i @ G_prev
            bp = b_i - L_i @ g_prev
            Z  = jnp.linalg.solve(Dp, jnp.concatenate([U_i, bp[:, None]], axis=1))
            return (Z[:, :-1], Z[:, -1]), (Z[:, :-1], Z[:, -1])

        z0 = (jnp.zeros((_S3, _S3), Dm.dtype), jnp.zeros((_S3,), Dm.dtype))
        _, (G, g) = jax.lax.scan(fwd, z0, (Dm, Um, Lm, b))

        def bwd(x_next, inp):
            G_i, g_i = inp
            x_i = g_i - G_i @ x_next
            return x_i, x_i
        _, X = jax.lax.scan(bwd, jnp.zeros((_S3,), Dm.dtype), (G, g), reverse=True)
        return X.reshape(-1)[:3 * _Nc]

    def _newton_dU(U_k, U_n_flat, dt):
        """One Newton update dU = -J⁻¹R."""
        R_k = _residual(U_k, U_n_flat, dt)
        if linear_solver == 'banded':
            return _banded_solve(_jac_blocks(U_k, U_n_flat, dt), R_k)
        J_k = _jac_fn(U_k, U_n_flat, dt)
        return jnp.linalg.solve(J_k, -R_k)

    def _newton_scale(U_k):
        """Reference magnitudes for the newton_tol test: |p| per cell, global
        max|u| + 1 m/s, global max|h| (h can be ~0 for some reference states)."""
        U  = U_k.reshape(-1, 3)
        return jnp.stack([jnp.abs(U[:, 0]),
                          jnp.full_like(U[:, 1], jnp.max(jnp.abs(U[:, 1])) + 1.0),
                          jnp.full_like(U[:, 2], jnp.max(jnp.abs(U[:, 2])))],
                         axis=1).reshape(-1)

    @jax.jit
    def jac_diff(U_n, dt):
        """max|J_colored - J_dense| at the given state — validation guard.
        Must be ~0 (same AD graph, reordered sums).  Nonzero ⇒ sparsity broke."""
        U_int = U_n[1:-1].reshape(-1)
        U_nf  = U_n.reshape(-1)
        return jnp.max(jnp.abs(_jac_colored(U_int, U_nf, dt)
                               - _jac_dense(U_int, U_nf, dt)))

    @jax.jit
    def compute_dt(U_n):
        """CFL-limited time step at state U_n. Returns (dt, a_c).
        Ghost cells (indices 0 and N-1) are excluded from the CFL limit —
        they can hold unphysical intermediate states that would collapse dt."""
        st_c = fluid.get_state(jxp.HmassP_INPUTS, U_n[:, 2], U_n[:, 0])
        a_c  = st_c["a"]
        dt   = CFL * dx / jnp.max(jnp.abs(U_n[1:-1, 1]) + a_c[1:-1])
        return dt, a_c

    @jax.jit
    def step(U_n, dt):
        """Advance one implicit Newton step with the given dt. Returns (U_new, ssr)."""
        U_n_flat  = U_n.reshape(-1)                    # (3*N,)
        U_k       = U_n[1:-1].reshape(-1)              # (3*(N-2),) — initial Newton iterate

        # ── Newton iterations ─────────────────────────────────────────────
        if newton_tol > 0.0:
            # SPEED-UP (newton_tol): stop once the relative update drops below
            # newton_tol (remaining error ~newton_tol² by quadratic convergence).
            # At CFL 0.5 Newton needs 5–6 iterations in transients and 1–3 near
            # steady state; newton_iter stays the hard cap.
            def _nt_cond(c):
                return (c[0] < newton_iter) & (c[2] > newton_tol)

            def _nt_body(c):
                dU = _newton_dU(c[1], U_n_flat, dt)
                return (c[0] + 1, c[1] + dU,
                        jnp.max(jnp.abs(dU) / (_newton_scale(c[1]) + 1e-300)))

            _, U_k, _ = jax.lax.while_loop(
                _nt_cond, _nt_body, (0, U_k, jnp.float64(jnp.inf)))
        else:
            def body(_, U_k):
                return U_k + _newton_dU(U_k, U_n_flat, dt)

            U_k = jax.lax.fori_loop(0, newton_iter, body, U_k)

        # ── Reconstruct full solution with updated BCs ────────────────────
        U_new = jnp.concatenate([U_n[:1], U_k.reshape(N - 2, 3), U_n[-1:]], axis=0)
        U_new = _bc_inlet(U_new)
        U_new = _bc_outlet(U_new)

        # ── Steady-state residual (dt-independent convergence metric) ────
        # Calling _residual(U_new, U_new, 1.0) makes the Vc*dV term vanish
        # (U_int == U_n), leaving only R_ss = fdiv - Vc*src — zero at steady
        # state regardless of how small dt is.
        R_ss   = _residual(U_new[1:-1].reshape(-1), U_new.reshape(-1),
                           jnp.float64(1.0)).reshape(N - 2, 3)
        st_new = fluid.get_state(jxp.HmassP_INPUTS, U_new[:, 2], U_new[:, 0])
        t_char = dx / (jnp.max(jnp.abs(U_new[:, 1]) + st_new["a"]) + 1e-300)
        V_new  = _V_from_U(U_new)
        # R_ss units: [kg/s, N, W]  (flux × area)
        # V_new units: [kg/m³, kg/m²/s, J/m³]  (conservative per volume)
        # V_cell × V_new gives [kg, kg·m/s, J]  → ratio with t_char×R_ss is dimensionless
        ssr    = jnp.sqrt(jnp.mean(
            (t_char * R_ss
             / (V_cell[1:-1, None] * jnp.abs(V_new[1:-1]) + 1e-300)) ** 2))

        return U_new, ssr

    @jax.jit
    def face_mass_flux(U):
        """Quantity B — conserved face mass flux ṁ_j = A_face[j]·F̂_mass[j].
        Uses the SAME MUSCL + BC + FVS path as the residual, so it is exactly
        the mass flux the scheme conserves. Flat to ~machine precision at a
        properly converged steady state. Input/returns: U (N,3) → (N-1,)."""
        U = _bc_inlet(U)
        U = _bc_outlet(U)
        U_L, U_R = _muscl_ph(U) if recon != 'constant' else (U[:-1], U[1:])
        F_hat    = _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)   # (N-1, 3)
        return A_face * F_hat[:, 0]

    return step, compute_dt, jac_diff, face_mass_flux


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def fvm_euler_1d(x, t, parameters):
    dx          = parameters['dx']
    geo         = parameters.get('geometry')
    if geo is not None:
        A_nodes     = np.asarray(geo.A)
        dAdx_nodes  = np.asarray(geo.dAdx)
        D_hydraulic = np.asarray(geo.D_h)
        roughness   = float(geo.roughness)
    else:
        A_nodes     = parameters['A_nodes']
        D_hydraulic = parameters['D_hydraulic']
        roughness   = parameters['roughness']
        dx_         = parameters['dx']
        dAdx_nodes  = np.concatenate([
            (A_nodes[1:2]  - A_nodes[:1])    / dx_,
            (A_nodes[2:]   - A_nodes[:-2])   / (2.0 * dx_),
            (A_nodes[-1:]  - A_nodes[-2:-1]) / dx_,
        ])
    Nj          = len(x)
    N           = parameters['N_eq']
    CFL         = parameters['CFL']
    p_stag      = parameters['pT_stagnation'][0]
    T_stag      = parameters['pT_stagnation'][1]
    p_out       = parameters['p_out']
    perfect_gas = parameters['perfect_gas']
    t_end       = t[-1]
    warmup      = 50

    # Gas-specific setup
    flux_method  = parameters['space_method']
    method       = parameters['time_method']
    recon        = parameters.get('reconstruction', 'constant')
    warmup_cfg   = parameters.get('warmup_cfg', None)
    wall_friction    = parameters.get('wall_friction', True)
    _nozzle_type     = parameters.get('nozzle_type', 'nozzle')
    transmissive_bc  = (_nozzle_type == 'shock_tube')

    if perfect_gas:
        cp, cv  = hyp.fvm_solver.specific_heat2()
        R_const = cp - cv
        gamma   = cp / cv
    else:
        fluid      = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
        stag_state = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
        s0         = float(stag_state.smass)
        h0         = float(stag_state.hmass)
        parameters['s0'] = s0

    # Allocate storage
    chunk = 1000
    Q      = np.zeros((N, Nj, chunk))
    t_vec  = np.zeros(chunk)
    Q[:, :, 0] = parameters['Q_o'](x)

    steady_thr = parameters['residuals']
    max_iter   = parameters['max_iterations']
    n = 0

    print("Starting time integration "
          f"(max_iter={max_iter}, tol={steady_thr:.1e}) ...", flush=True)

    # ── Städtke Newton implicit path (real gas, (p,u,h) primitives) ─────
    use_stadtke = (not perfect_gas) and (method == 'fvs_newton')
    if use_stadtke:
        Q_init  = Q[:, :, 0].copy()
        init_strategy = parameters.get('initialization', 'simple')
        if parameters.get('nozzle_type', 'nozzle') == 'shock_tube':
            U0 = shock_tube_puh_initialization(x, A_nodes, parameters)
        elif init_strategy == 'isentropic':
            U0 = stadtke_puh_initialization(x, A_nodes, parameters)
        elif init_strategy == 'subsonic':
            U0 = stadtke_puh_initialization_subsonic(x, A_nodes, parameters)
        elif init_strategy == 'shocked':
            U0 = stadtke_puh_initialization_shocked(x, A_nodes, parameters)
        else:  # 'simple' (default)
            U0 = stadtke_puh_initialization_simple(x, A_nodes, parameters)
        print(f"  FVS Newton init: {init_strategy}", flush=True)

        A_face = 0.5 * (A_nodes[:-1] + A_nodes[1:])

        # ── WELL-BALANCED area gradient (2026-07-13, same fix as two-fluid) ────
        # The flux divergence uses FACE areas Δ(A_face·F̂); the wall-pressure
        # source p·dAdx/A used the ANALYTIC geo.dAdx.  Discretely dAdx_i·dx ≠
        # (A_{i+1/2}−A_{i-1/2}) wherever A(x) is curved, so a SMOOTH state cannot
        # zero the steady residual → the solver absorbs the mismatch as a 2Δx
        # wiggle near the steepest area change (the low-Ma near-throat sawtooth /
        # SSR plateau).  Face-consistent gradient → the source cancels the
        # smooth-flow flux divergence EXACTLY (still 2nd-order accurate).
        # Invariant to preserve: dAdx·dx ≡ ΔA_face with the SAME A_face the
        # flux uses.  (Root cause + probes: memory nozzle_hyperbolic_slip_
        # crossing_instability.md; two-fluid fix in main_two_fluid_single_phase.py.)
        dAdx_nodes = np.asarray(dAdx_nodes, dtype=float).copy()
        dAdx_nodes[1:-1] = (A_face[1:] - A_face[:-1]) / dx

        jacobian_mode = parameters.get('jacobian', 'dense')
        # 4th-order JST filter for the low-Mach p-checkerboard (0.0 = OFF,
        # identical to previous behaviour).  eps_filter ~0.05 typical; optional
        # velocity gate eps_filter_ucut/uwidth [m/s] as in the two-fluid solver.
        _eps_filter = float(parameters.get('eps_filter', 0.0))
        _ef_ucut    = parameters.get('eps_filter_ucut', None)
        _ef_uwidth  = parameters.get('eps_filter_uwidth', None)
        # Speed-up switches (SPEEDUP_REPORT_FVS_SINGLE_PHASE.md); the defaults
        # reproduce the original code path bit-for-bit.
        _linear_solver  = str(parameters.get('linear_solver', 'dense'))
        _newton_tol     = float(parameters.get('newton_tol', 0.0))
        _flux_local_jac = bool(parameters.get('flux_local_jac', False))
        _cell_local_jac = bool(parameters.get('cell_local_jac', False))
        _fast_incr      = bool(parameters.get('fast_increments', False))
        _ser_beta       = float(parameters.get('ser_beta', 0.0))
        _ser_cfl_max    = float(parameters.get('ser_cfl_max', 50.0))
        _stadtke_step, _compute_dt, _jac_diff, _face_mass_flux = make_jax_stadtke_step(
            fluid, A_nodes, A_face, dAdx_nodes, D_hydraulic, roughness,
            dx, CFL, p_out, h0, s0, wall_friction=wall_friction,
            transmissive_bc=transmissive_bc, recon=recon,
            delta_factor=parameters.get('delta_factor', 0.1),
            newton_iter=parameters.get('newton_iter', 20),
            jacobian_mode=jacobian_mode,
            eps_filter=_eps_filter,
            eps_filter_ucut=None if _ef_ucut in (None, "", "none") else float(_ef_ucut),
            eps_filter_uwidth=None if _ef_uwidth in (None, "", "none") else float(_ef_uwidth),
            linear_solver=_linear_solver, newton_tol=_newton_tol,
            flux_local_jac=_flux_local_jac, cell_local_jac=_cell_local_jac)
        print(f"  Jacobian mode: {jacobian_mode}   eps_filter={_eps_filter}"
              + (f" (ucut={_ef_ucut})" if _ef_ucut not in (None, "", "none") else ""),
              flush=True)
        print(f"  [Speed-up] linear_solver={_linear_solver}  newton_tol={_newton_tol:g}"
              f"  flux_local_jac={_flux_local_jac}  cell_local_jac={_cell_local_jac}"
              f"  fast_increments={_fast_incr}  ser_beta={_ser_beta:g}"
              + (f" (CFL<={_ser_cfl_max:g})" if _ser_beta > 0.0 else ""), flush=True)

        @jax.jit
        def _increments_jit(U_old, U_new, a_c):
            """Jitted twin of the eager dp/dh/du check below (fast_increments):
            same formulas, one dispatch instead of ~20 eager ops."""
            return jnp.stack([
                jnp.max(jnp.abs(U_new[:, 0] - U_old[:, 0]) / (U_old[:, 0] + 1e-10)),
                jnp.max(jnp.abs(U_new[:, 2] - U_old[:, 2]) / (jnp.abs(U_old[:, 2]) + 1e-10)),
                jnp.max(jnp.abs(U_new[:, 1] - U_old[:, 1]) / (jnp.abs(U_old[:, 1]) + a_c + 1e-10))])

        U_jax = jnp.asarray(U0, dtype=jnp.float64)

        print("Compiling Städtke Newton step ...", flush=True)
        t0_compile = _time.perf_counter()

        # Trigger JIT compilation (discard result — compilation warmup only)
        _dt_init_j, _ = _compute_dt(U_jax)
        _U_warm, _    = _stadtke_step(U_jax, _dt_init_j)
        jax.block_until_ready(_U_warm)
        compile_time = _time.perf_counter() - t0_compile
        print(f"Compiled in {compile_time:.1f} s. Running ...", flush=True)

        # ── Jacobian sparsity guard ──────────────────────────────────────────
        # Compares colored vs dense AD Jacobian at the initial state. Must be ~0;
        # nonzero means a non-local term broke the banded sparsity assumption.
        if parameters.get('jacobian_validate', False):
            _d = float(_jac_diff(U_jax, _dt_init_j))
            print(f"  Jacobian validation: max|J_colored - J_dense| = {_d:.3e}",
                  flush=True)
            if _d > 1e-6:
                print("  WARNING: colored Jacobian differs from dense AD — "
                      "banded sparsity assumption violated (non-local term?). "
                      "Use jacobian: dense.", flush=True)

        # ── Time-step controller parameters ──────────────────────────────
        _MAX_HALVINGS = 8     # max halvings per step before accepting anyway
        _DP_MAX = 0.20        # max |Δp|/p   accepted per step
        _DH_MAX = 0.10        # max |Δh|/h   accepted per step
        _DU_MAX = 0.20        # max |Δu|/(|u|+a) accepted per step
        _DT_GROW = 1.20       # dt growth factor after accepted step
        _DT_MIN  = dx / 1e12  # absolute floor: if dt collapses below this, stop
        # adaptive_dt=False: take the full CFL-capped dt every step and accept
        # unconditionally — no increment check, no halving (faster, but no safety
        # net against divergence). True (default): adaptive halving as below.
        adaptive_dt = parameters.get('adaptive_dt', True)
        print(f"  Adaptive dt limiter: {adaptive_dt}", flush=True)

        n        = 0
        t_phys   = 0.0
        res      = float('inf')
        res_hist = [float('inf')] * 5
        dt       = float(_dt_init_j)   # current target dt; grows/shrinks adaptively
        t0       = _time.perf_counter()
        # phase timers (s): where does the per-iteration time go?
        _tm = {'dt': 0.0, 'step': 0.0, 'check': 0.0}
        # SER CFL ramp (ser_beta > 0, steady cases only — it gives up time
        # accuracy): CFL_eff = CFL·clip((SSR_1/SSR)^β, 1, ser_cfl_max/CFL), the
        # same pseudo-transient continuation as the two-fluid driver.  The
        # increment checks / halvings below remain the safety net.
        _ser_ref  = None    # SSR after the first step
        _cfl_mult = 1.0     # current CFL multiplier (1 = plain CFL)

        def _ser_cap(dt_cap):
            nonlocal _ser_ref, _cfl_mult
            if _ser_beta > 0.0 and math.isfinite(res):
                if _ser_ref is None:
                    _ser_ref = res
                _cfl_mult = max(1.0, min(_ser_cfl_max / CFL,
                                         (_ser_ref / max(res, 1e-300)) ** _ser_beta))
                dt_cap *= _cfl_mult
            return dt_cap

        try:
            while n < max_iter and t_phys < t_end:
                # ── Convergence / plateau check ───────────────────────────
                below_thr = res < steady_thr
                hist_max  = max(res_hist)
                hist_min  = min(res_hist) + 1e-300
                plateau   = (n >= warmup) and (hist_max / hist_min < 1.5) and (res < 3.0 * steady_thr)
                if below_thr or plateau:
                    break

                if adaptive_dt:
                    # ── CFL cap for current state; grow dt toward it ──────────
                    _t_a = _time.perf_counter()
                    dt_max_j, a_j = _compute_dt(U_jax)
                    dt = min(dt * _DT_GROW, _ser_cap(float(dt_max_j)))
                    _tm['dt'] += _time.perf_counter() - _t_a

                    if dt < _DT_MIN:
                        print(f"  dt collapsed to {dt:.2e} (< min {_DT_MIN:.2e}) — "
                              f"stopping at SSR={res:.3e}")
                        break

                    # ── Adaptive retry: halve dt on large increments ──────────
                    halvings = 0
                    for halvings in range(_MAX_HALVINGS + 1):
                        _t_b = _time.perf_counter()
                        U_new, res_j = _stadtke_step(U_jax, jnp.float64(dt))
                        jax.block_until_ready(U_new)
                        _t_c = _time.perf_counter()
                        _tm['step'] += _t_c - _t_b

                        if _fast_incr:
                            dp, dh, du = (float(v) for v in jax.device_get(
                                _increments_jit(U_jax, U_new, a_j)))
                        else:
                            dp = float(jnp.max(jnp.abs(U_new[:, 0] - U_jax[:, 0])
                                               / (U_jax[:, 0] + 1e-10)))
                            dh = float(jnp.max(jnp.abs(U_new[:, 2] - U_jax[:, 2])
                                               / (jnp.abs(U_jax[:, 2]) + 1e-10)))
                            du = float(jnp.max(jnp.abs(U_new[:, 1] - U_jax[:, 1])
                                               / (jnp.abs(U_jax[:, 1]) + a_j + 1e-10)))
                        _tm['check'] += _time.perf_counter() - _t_c

                        if dp < _DP_MAX and dh < _DH_MAX and du < _DU_MAX:
                            break           # accepted
                        if halvings < _MAX_HALVINGS:
                            dt *= 0.5       # halve and retry

                    # ── Advance ───────────────────────────────────────────────
                    U_jax  = U_new
                    if halvings >= _MAX_HALVINGS:
                        # Newton stalled: dt is too small to trust the residual.
                        # Inflate so the plateau/threshold checks can't fire.
                        res = max(float(res_j), float(steady_thr) * 10)
                    else:
                        res = float(res_j)
                else:
                    # ── Limiter off: full CFL-capped dt, accept unconditionally ──
                    _t_a = _time.perf_counter()
                    dt_max_j, a_j = _compute_dt(U_jax)
                    dt = _ser_cap(float(dt_max_j))
                    _tm['dt'] += _time.perf_counter() - _t_a

                    _t_b = _time.perf_counter()
                    U_new, res_j = _stadtke_step(U_jax, jnp.float64(dt))
                    jax.block_until_ready(U_new)
                    _tm['step'] += _time.perf_counter() - _t_b

                    U_jax = U_new
                    res   = float(res_j)
                t_phys += dt
                n      += 1
                res_hist = res_hist[1:] + [res]

                _cfl_txt = f"  CFL_eff={CFL * _cfl_mult:.1f}" if _ser_beta > 0.0 else ""
                if adaptive_dt and halvings > 0:
                    print(f"  Iter {n:5d}  t={t_phys:.3e}  dt={dt:.3e}  "
                          f"SSR={res:.3e}  [{halvings} halving(s), "
                          f"dp={dp:.2f} dh={dh:.2f} du={du:.2f}]{_cfl_txt}", flush=True)
                elif n < 10 or (n < 100 and n % 10 == 0) or n % 50 == 0:
                    ms_iter = (_time.perf_counter() - t0) / n * 1e3
                    print(f"  Iter {n:5d}  t={t_phys:.3e}  dt={dt:.3e}  SSR={res:.3e}  "
                          f"({ms_iter:.1f} ms/iter){_cfl_txt}", flush=True)

        except KeyboardInterrupt:
            print("Interrupted — returning current solution.")

        elapsed  = _time.perf_counter() - t0
        n_done   = n
        t_done   = t_phys
        res_done = res
        if res_done < float(steady_thr):
            reason = "below threshold"
        elif t_done >= float(t_end):
            reason = f"t_end reached ({t_done:.3e} s)"
        else:
            reason = "plateau / max_iter"
        print(f"Done: {n_done} iterations ({reason}). "
              f"SSR = {res_done:.3e}.  Computational time = {elapsed:.1f} s "
              f"({elapsed / max(n_done, 1) * 1e3:.1f} ms/iter)", flush=True)
        _nit = max(n_done, 1)
        _other = elapsed - sum(_tm.values())
        print(f"  per-iter breakdown:  compute_dt={_tm['dt']/_nit*1e3:7.1f} ms | "
              f"newton_step={_tm['step']/_nit*1e3:7.1f} ms | "
              f"accept_check={_tm['check']/_nit*1e3:7.1f} ms | "
              f"other={_other/_nit*1e3:7.1f} ms", flush=True)

        # ── Mass-flux conservation diagnostic (Quantity A vs B) ──────────────
        # A = cell-centre ρuA (NOT conserved pointwise; smears at shocks)
        # B = face FVS mass flux A_face·F̂_mass (the truly conserved quantity)
        U_np   = np.asarray(U_jax)
        st_c   = fluid.get_state(jxp.HmassP_INPUTS, U_jax[:, 2], U_jax[:, 0])
        rho_c  = np.asarray(st_c["rho"])
        mdotA  = rho_c * U_np[:, 1] * A_nodes                  # cell-centre (N,)
        mA_int = mdotA[1:-1]
        varA   = np.max(np.abs(mA_int - mA_int.mean())) / (abs(mA_int.mean()) + 1e-300)

        mdotB  = np.asarray(_face_mass_flux(U_jax))            # face flux (N-1,)
        mB_int = mdotB[1:-1]                                   # drop boundary faces
        meanB  = mB_int.mean()
        varB   = np.max(np.abs(mB_int - meanB)) / (abs(meanB) + 1e-300)
        iB     = int(np.argmax(np.abs(mB_int - meanB))) + 1    # face index of worst dev
        glob   = abs(mdotB[0] - mdotB[-1]) / (abs(mdotB[0]) + 1e-300)
        print(f"  Mass flux  A (cell-centre, interior): max dev = {varA*100:.3f} %  "
              f"[smears at shocks — NOT a conservation metric]", flush=True)
        print(f"  Mass flux  B (face FVS flux, interior): max dev = {varB*100:.3e} %  "
              f"at face {iB} (x≈{x[iB]:.4f})  |  inlet→outlet global = {glob*100:.3e} %",
              flush=True)
        if varB < 1e-6:
            print("    → B is flat: scheme IS conservative; A's scatter is shock "
                  "smearing (truncation), not a bug.", flush=True)
        else:
            print("    → B is NOT flat: real conservation issue — see "
                  "mass_flow_diagnostic.md §4.", flush=True)

        Q_final = _puh_to_conservative(U_np, fluid)
        # Primitive (p, u, h) result and run statistics for callers that want
        # them without the conservative → primitive round trip.
        parameters['U_puh_final'] = U_np
        parameters['run_stats'] = dict(steps=n_done, wall_s=elapsed, ssr=res_done,
                                       reason=reason, t_phys=t_done,
                                       compile_s=compile_time)

        Q_out = np.zeros((N, Nj, 2))
        Q_out[:, :, 0] = Q_init
        Q_out[:, :, 1] = Q_final
        return Q_out, np.array([0.0, t_done])

    # ── JAX path (real gas, non-Roe) ─────────────────────────────────────
    use_jax = (not perfect_gas) and (flux_method != 'roe') and (method != 'fvs_newton')
    if use_jax:
        _jax_step = make_jax_step(fluid, A_nodes, dAdx_nodes, D_hydraulic, roughness,
                                  dx, CFL, p_out, h0, s0, recon, flux_method, method,
                                  wall_friction=wall_friction,
                                  transmissive_bc=transmissive_bc)
        Q_jax  = jnp.asarray(Q[:, :, 0])
        Q_init = Q[:, :, 0].copy()

        # ── Optional warmup phase ─────────────────────────────────────────────
        # Run N_wu iterations with cheap settings (e.g. Rusanov, constant recon)
        # to establish a decent initial field before the main solver takes over.
        # Compiled as a single fori_loop — no Python overhead between warmup iters.
        if warmup_cfg is not None:
            wu_n     = int(warmup_cfg.get('n_iterations',  200))
            wu_flux  = str(warmup_cfg.get('flux_method',   'rusanov'))
            wu_recon = str(warmup_cfg.get('reconstruction','constant'))
            wu_cfl   = float(warmup_cfg.get('CFL',          CFL))
            # Only build warmup step when settings differ from main settings
            wu_differs = (wu_flux != flux_method or wu_recon != recon or wu_cfl != CFL)
            if wu_differs:
                _wu_step = make_jax_step(fluid, A_nodes, dAdx_nodes, D_hydraulic, roughness,
                                         dx, wu_cfl, p_out, h0, s0,
                                         wu_recon, wu_flux, method,
                                         wall_friction=wall_friction,
                                         transmissive_bc=transmissive_bc)
            else:
                _wu_step = _jax_step   # identical settings: reuse compiled kernel

            @jax.jit
            def _wu_run(Q0):
                def body(i, Q):
                    Q_new, _, _, _, _ = _wu_step(Q)
                    return Q_new
                return jax.lax.fori_loop(0, wu_n, body, Q0)

            print(f"Warmup: {wu_n} iters with flux={wu_flux}, recon={wu_recon}, "
                  f"CFL={wu_cfl} ...", flush=True)
            t_wu = _time.perf_counter()
            Q_jax = _wu_run(Q_jax)
            jax.block_until_ready(Q_jax)
            print(f"Warmup done in {_time.perf_counter()-t_wu:.1f} s", flush=True)

        # ── Fully compiled solver: jax.lax.while_loop stays on-device ────────
        # No Python↔XLA sync between iterations → maximum throughput.
        # Convergence checks (threshold + plateau) compiled as JAX ops inside
        # the loop — zero Python overhead per iteration.
        _max_iter = jnp.int32(max_iter)
        _tol      = jnp.float64(steady_thr)
        _warmup_j = jnp.int32(warmup)
        _t_end_j  = jnp.float64(t_end)   # physical end time — used for transient cases

        # @jax.jit  # disabled: skip XLA compilation, lax.while_loop runs eagerly
        def _jax_run(Q0):

            def cond_fn(carry):
                _, n, t_phys, res, res_hist = carry
                below_thr = res < _tol
                hist_max  = jnp.max(res_hist)
                hist_min  = jnp.min(res_hist) + 1e-300
                plateau   = (n >= _warmup_j) & (hist_max / hist_min < 1.5) & (res < 3.0 * _tol)
                time_done = t_phys >= _t_end_j   # transient stop: physical time reached
                return (n < _max_iter) & ~(below_thr | plateau | time_done)

            def body_fn(carry):
                Q, n, t_phys, _, res_hist = carry
                # ssr (steady-state residual) is returned free from step() —
                # it is computed there anyway as part of RHS_n, at no extra cost.
                Q_new, _, _, dt, res = _jax_step(Q)
                res_hist_new = jnp.roll(res_hist, -1).at[-1].set(res)

                # Adaptive print frequency: every 1 / 10 / 50 iters.
                should_print = jnp.where(
                    n < 10,  jnp.bool_(True),
                    jnp.where(n < 100, n % 10 == 0, n % 50 == 0))
                jax.lax.cond(
                    should_print,
                    lambda args: jax.debug.print(
                        "Iter {n:5d}  SSR = {res:.3e}", n=args[0], res=args[1]),
                    lambda args: None,
                    (n, res),
                )

                return Q_new, n + 1, t_phys + dt, res, res_hist_new

            res_hist0 = jnp.ones(5, dtype=jnp.float64)
            init      = (Q0, jnp.int32(0), jnp.float64(0.0), jnp.float64(1.0), res_hist0)
            Q_f, n_f, t_f, res_f, _ = jax.lax.while_loop(cond_fn, body_fn, init)
            return Q_f, n_f, t_f, res_f

        print("Running solver (eager mode, no XLA compilation) ...", flush=True)
        t0 = _time.perf_counter()
        try:
            Q_final, n_final, t_final, res_final = _jax_run(Q_jax)
            jax.block_until_ready(Q_final)
        except KeyboardInterrupt:
            print("Interrupted — returning initial solution.")
            Q_out = np.zeros((N, Nj, 2))
            Q_out[:, :, 0] = Q_init
            Q_out[:, :, 1] = Q_init
            return Q_out, np.array([0.0, 0.0])

        elapsed  = _time.perf_counter() - t0
        n_done   = int(n_final)
        t_done   = float(t_final)
        res_done = float(res_final)
        if res_done < float(steady_thr):
            reason = "below threshold"
        elif t_done >= float(t_end):
            reason = "physical time reached"
        elif n_done >= max_iter:
            reason = "max iterations"
        else:
            reason = "plateau"
        print(f"Done: {n_done} iterations ({reason}). "
              f"t_phys = {t_done:.3e} s,  NRMS = {res_done:.3e}.  "
              f"Computational time = {elapsed:.1f} s", flush=True)

        Q_out = np.zeros((N, Nj, 2))
        Q_out[:, :, 0] = Q_init
        Q_out[:, :, 1] = np.asarray(Q_final)
        return Q_out, np.array([0.0, t_done])

    # ── NumPy path (perfect gas or Roe) ──────────────────────────────────
    try:
        while t_vec[n] <= t_end and n < max_iter:
            if n + 1 >= Q.shape[2]:
                Q     = np.concatenate([Q,     np.zeros((N, Nj, 100))], axis=2)
                t_vec = np.concatenate([t_vec, np.zeros(100)])

            Q_n = Q[:, :, n]
            _t0 = _time.perf_counter() if n == 0 else None

            if perfect_gas:
                _bc_inlet_perfect(Q_n, p_stag, T_stag, gamma, R_const)
                _bc_outlet_perfect(Q_n, p_out, A_nodes, gamma)
            else:
                _bc_inlet_real(Q_n, h0, s0, fluid)
                _bc_outlet_real(Q_n, p_out, A_nodes, fluid)

            if n == 0:
                print(f"  BCs:     {_time.perf_counter()-_t0:.2f}s", flush=True)
                _t0 = _time.perf_counter()

            rho = Q_n[0];  u = Q_n[1] / rho
            if perfect_gas:
                p     = (gamma - 1.0) * (Q_n[2] - 0.5 * Q_n[1]**2 / rho)
                c     = np.sqrt(gamma * p / rho)
                tau_w = np.zeros(Nj)
            else:
                p, c, tau_w = _real_gas_state_and_friction(Q_n, D_hydraulic, roughness, fluid,
                                                            wall_friction=wall_friction)
                p[-1] = p_out

            if n == 0:
                print(f"  EOS loop:{_time.perf_counter()-_t0:.2f}s", flush=True)
                _t0 = _time.perf_counter()

            parameters['p'] = p;  parameters['c'] = c
            S  = _source_term(p, tau_w, A_nodes, D_hydraulic, dx)
            dt = _compute_dt(u, c, dx, CFL)
            parameters['dt'] = dt

            if method == 'euler':
                RHS = R(Q_n, parameters)
                if n == 0:
                    print(f"  Flux:    {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q[:, :, n + 1] = Q_n + dt * (RHS + S)
            elif method == 'RK3':
                R1 = R(Q_n, parameters)
                if n == 0:
                    print(f"  Flux R1: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q1 = Q_n + dt * (R1 + S)
                R2 = R(Q1, parameters)
                if n == 0:
                    print(f"  Flux R2: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q2 = 0.75 * Q_n + 0.25 * (Q1 + dt * (R2 + S))
                R3 = R(Q2, parameters)
                if n == 0:
                    print(f"  Flux R3: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q[:, :, n + 1] = (Q_n + 2.0 * (Q2 + dt * (R3 + S))) / 3.0
            elif method == 'RK4':
                k1 = R(Q_n, parameters) + S
                k2 = R(Q_n + 0.5 * dt * k1, parameters) + S
                k3 = R(Q_n + 0.5 * dt * k2, parameters) + S
                k4 = R(Q_n +       dt * k3,  parameters) + S
                Q[:, :, n + 1] = Q_n + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            elif method == 'implicit':
                lam      = np.abs(u) + c
                lam_face = np.maximum(lam[:-1], lam[1:])
                lam_L    = np.concatenate([lam_face[:1],  lam_face])
                lam_R    = np.concatenate([lam_face, lam_face[-1:]])
                denom    = 1.0 + dt * (lam_L + lam_R) / dx
                RHS = R(Q_n, parameters)
                Q[:, :, n + 1] = Q_n + dt * (RHS + S) / denom[np.newaxis, :]

            if n > 0:
                dQ       = (np.abs(Q[:, 1:-1, n+1] - Q[:, 1:-1, n])
                            / (np.abs(Q[:, 1:-1, n]) + 1e-300))
                L2       = np.sqrt(np.mean(dQ**2, axis=1))
                residual = float(np.max(L2))
                freq = 1 if n < 10 else (10 if n < 100 else 50)
                if n % freq == 0:
                    print(f"Iter {n:5d}  residual = {residual:.4e}"
                          f"  dt = {dt:.3e}  p_min = {p.min():.4e}", flush=True)
                if n > warmup and residual < steady_thr:
                    print(f"Converged after {n} iterations, t = {t_vec[n]:.4e} s", flush=True)
                    n += 1
                    break

            t_vec[n + 1] = t_vec[n] + dt
            n += 1

    except KeyboardInterrupt:
        print("Interrupted — returning current solution.")

    return Q[:, :, :n], t_vec[:n]
