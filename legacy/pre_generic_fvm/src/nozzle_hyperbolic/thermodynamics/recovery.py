"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq as _brentq
_FluidBicubic = getattr(jxp, "FluidBicubic", None)


def _get_state_ps(fluid, p, s):
    """
    Get thermodynamic state from (p, s).
    FluidBicubic's PSmass_INPUTS handler uses a 2D Newton solver that fails when
    the table has NaN entries.  Instead, do a 1D brentq on h:
        find h s.t.  entropy(h, p) = s
    using HmassP_INPUTS (direct bicubic lookup, no internal solver).
    Entropy is monotone in h at constant p, so the bracket always works.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        h_min = float(getattr(fluid, 'h_min', 50e3))
        h_max = float(getattr(fluid, 'h_max', 850e3))
        p_f   = float(p)
        s_f   = float(s)
        def _f(h):
            return float(fluid.get_state(jxp.HmassP_INPUTS, h, p_f).entropy) - s_f
        try:
            h_sol = _brentq(_f, h_min, h_max, xtol=1.0, maxiter=60)
        except Exception:
            h_sol = (h_min + h_max) * 0.5
        return fluid.get_state(jxp.HmassP_INPUTS, h_sol, p_f)
    return fluid.get_state(jxp.PSmass_INPUTS, p, s)


def _get_state_hs(fluid, h, s):
    """
    Get thermodynamic state from (h, s).
    FluidBicubic's HmassSmass_INPUTS handler uses a 1D Newton on p that can fail
    when the table has NaN entries near the saturation boundary.  Instead, do a
    1D brentq on p:
        find p s.t.  entropy(h, p) = s
    using HmassP_INPUTS (direct bicubic lookup).
    Entropy decreases with pressure at constant h, so the bracket always works.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        p_min = float(getattr(fluid, 'p_min', 1e4))
        p_max = float(getattr(fluid, 'p_max', 300e5))
        h_f   = float(h)
        s_f   = float(s)
        def _f(p):
            return float(fluid.get_state(jxp.HmassP_INPUTS, h_f, p).entropy) - s_f
        try:
            p_sol = _brentq(_f, p_min, p_max, xtol=1.0, maxiter=60)
        except Exception:
            p_sol = (p_min + p_max) * 0.5
        return fluid.get_state(jxp.HmassP_INPUTS, h_f, p_sol)
    return fluid.get_state(jxp.HmassSmass_INPUTS, h, s)


def state_from_rho_u_batch(fluid, rho_arr, u_int_arr):
    """
    Vectorized (rho, u_int) → state for an entire array of cells.

    Replaces N individual brentq calls with ~32 batch EOS calls via bisection.
    Each batch call goes through FluidBicubic.get_state which uses jax.vmap
    internally, so all N values are processed in a single JIT-compiled pass.
    Falls back to the scalar loop for non-FluidBicubic fluids.
    """
    if not (_FluidBicubic and isinstance(fluid, _FluidBicubic)):
        states = [fluid.get_state(jxp.DmassUmass_INPUTS, float(r), float(u))
                  for r, u in zip(rho_arr, u_int_arr)]
        return states  # list of scalar states

    p_lo_raw = float(getattr(fluid, 'p_min', 1e4))
    p_hi_raw = float(getattr(fluid, 'p_max', 300e5))
    h_lo_raw = float(getattr(fluid, 'h_min', 50e3))
    h_hi_raw = float(getattr(fluid, 'h_max', 850e3))

    rho   = np.asarray(rho_arr,   dtype=float)
    u_int = np.asarray(u_int_arr, dtype=float)

    # Per-cell h-range constraint on the pressure bracket
    p_lo = np.full(len(rho), p_lo_raw)
    p_hi = np.full(len(rho), p_hi_raw)
    p_from_hmin = (h_lo_raw - u_int) * rho
    p_from_hmax = (h_hi_raw - u_int) * rho
    valid_lo = (p_from_hmin > p_lo_raw) & (p_from_hmin < p_hi_raw)
    valid_hi = (p_from_hmax > p_lo_raw) & (p_from_hmax < p_hi_raw)
    p_lo = np.where(valid_lo, np.maximum(p_lo, p_from_hmin), p_lo)
    p_hi = np.where(valid_hi, np.minimum(p_hi, p_from_hmax), p_hi)
    bad  = p_lo >= p_hi
    p_lo = np.where(bad, p_lo_raw, p_lo)
    p_hi = np.where(bad, p_hi_raw, p_hi)
    p = 0.5 * (p_lo + p_hi)

    # Bisection: 35 batch calls instead of N×30 scalar calls
    for _ in range(35):
        h = np.clip(u_int + p / rho, h_lo_raw + 1.0, h_hi_raw - 1.0)
        try:
            rho_pred = np.asarray(
                fluid.get_state(jxp.HmassP_INPUTS, h, p).density)
        except Exception:
            break
        residual = rho_pred - rho
        p_lo = np.where(residual < 0, p, p_lo)
        p_hi = np.where(residual > 0, p, p_hi)
        p = 0.5 * (p_lo + p_hi)

    h_final = np.clip(u_int + p / rho, h_lo_raw + 1.0, h_hi_raw - 1.0)
    return fluid.get_state(jxp.HmassP_INPUTS, h_final, p)


def _state_rho_u(fluid, rho, u_int):
    """
    Get thermodynamic state from (density, internal_energy).

    FluidBicubic table is indexed by (h, logP).  HmassP_INPUTS is a direct
    bicubic lookup (no internal solver), so we use it inside a brentq over p:
        residual(p) = density(h = u_int + p/rho,  p) - rho = 0
    The p bracket is constrained so that h = u_int + p/rho stays inside the
    table's h range, avoiding extrapolation at the bracket endpoints.
    For jxp.Fluid the direct DmassUmass_INPUTS EOS call is used.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        p_lo_raw = float(getattr(fluid, 'p_min', 1e4))
        p_hi_raw = float(getattr(fluid, 'p_max', 300e5))
        h_min    = float(getattr(fluid, 'h_min', 50e3))
        h_max    = float(getattr(fluid, 'h_max', 850e3))
        rho_f    = float(rho)
        u_f      = float(u_int)
        # Clamp bracket so h = u+p/rho stays within the table h range
        p_lo = max(p_lo_raw, (h_min - u_f) * rho_f)
        p_hi = min(p_hi_raw, (h_max - u_f) * rho_f)
        if p_lo >= p_hi or p_lo <= 0.0:   # fallback: use full table p range
            p_lo, p_hi = p_lo_raw, p_hi_raw
        def _f(p):
            h = u_f + p / rho_f
            return float(fluid.get_state(jxp.HmassP_INPUTS, h, p).density) - rho_f
        try:
            p_sol = _brentq(_f, p_lo, p_hi, xtol=1.0, maxiter=60)
        except Exception:
            p_sol = (p_lo + p_hi) * 0.5
        h_sol = u_f + p_sol / rho_f
        return fluid.get_state(jxp.HmassP_INPUTS, h_sol, p_sol)
    return fluid.get_state(jxp.DmassUmass_INPUTS, rho, u_int)
