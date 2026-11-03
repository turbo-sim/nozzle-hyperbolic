"""Alberto's eight-state representation and its six transported contents.

Reference: faafff704e54fd866abff95fcb88fc0a5217405c, solver.py:64-105.
No local projection, phase clipping, or hidden storage-recovery solver.
"""

import jax.numpy as jnp
import jaxprop as jxp


def conserved(W):
    """Map eight iteration variables to six transported densities.

    Parameters
    ----------
    W : array_like, shape (..., 8)
        States ordered ``(p, alpha_g, m_g, u_g, m_l, u_l, E_g, E_l)``, where
        phase masses and internal energies are per mixture volume.

    Returns
    -------
    jax.Array, shape (..., 6)
        Gas/liquid masses, momenta and total energies, in that order.
        Geometry weighting is applied separately by the numerical operator.
    """
    p, alpha, mg, ug, ml, ul, eg, el = jnp.moveaxis(jnp.asarray(W), -1, 0)
    return jnp.stack((mg, ml, mg * ug, ml * ul,
                      eg + 0.5 * mg * ug**2, el + 0.5 * ml * ul**2), axis=-1)


def primitives(W):
    """Recover enthalpy primitives from the eight-state representation.

    Parameters
    ----------
    W : array_like, shape (..., 8)
        Iteration states with strictly positive phase masses and open phase
        fractions. Pure phases are outside this representation's domain.

    Returns
    -------
    jax.Array, shape (..., 6)
        Pressure, gas/liquid velocities, gas void fraction and gas/liquid
        specific enthalpies: ``(p, u_g, u_l, alpha_g, h_g, h_l)``.
    """
    p, alpha, mg, ug, ml, ul, eg, el = jnp.moveaxis(jnp.asarray(W), -1, 0)
    return jnp.stack((p, ug, ul, alpha, (eg + p * alpha) / mg,
                      (el + p * (1 - alpha)) / ml), axis=-1)


def from_primitives(U, fluid_g, fluid_l):
    """Construct EOS-consistent iteration states from enthalpy primitives.

    Parameters
    ----------
    U : array_like, shape (..., 6)
        ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in physical SI units, with
        ``0 < alpha_g < 1`` and admissible phase thermodynamic states.
    fluid_g, fluid_l : object
        JAX-callable phase fluids exposing ``get_state(HmassP_INPUTS, h, p)``.

    Returns
    -------
    jax.Array, shape (..., 8)
        ``(p, alpha_g, m_g, u_g, m_l, u_l, E_g, E_l)`` with phase masses and
        internal energies per mixture volume. No phase clipping is applied.
    """
    p, ug, ul, alpha, hg, hl = jnp.moveaxis(jnp.asarray(U), -1, 0)
    rg = fluid_g.get_state(jxp.HmassP_INPUTS, hg, p)["rho"]
    rl = fluid_l.get_state(jxp.HmassP_INPUTS, hl, p)["rho"]
    mg, ml = alpha * rg, (1 - alpha) * rl
    return jnp.stack((p, alpha, mg, ug, ml, ul,
                      mg * hg - p * alpha, ml * hl - p * (1 - alpha)), axis=-1)


def eos_constraints(W, fluid_g, fluid_l):
    """Evaluate unscaled intrinsic-density EOS defects for both phases.

    Parameters
    ----------
    W : jax.Array, shape (..., 8)
        Iteration states with positive phase masses and open void fractions.
    fluid_g, fluid_l : object
        JAX-callable phase fluids exposing ``get_state(HmassP_INPUTS, h, p)``.

    Returns
    -------
    jax.Array, shape (..., 2)
        Gas and liquid intrinsic densities minus their EOS values, in kg/m^3.
        The operator applies fixed constraint scales, independent of timestep,
        cell volume and local phase fraction.
    """
    U = primitives(W)
    rg = fluid_g.get_state(jxp.HmassP_INPUTS, U[..., 4], U[..., 0])["rho"]
    rl = fluid_l.get_state(jxp.HmassP_INPUTS, U[..., 5], U[..., 0])["rho"]
    # Intrinsic density constraints retain sensitivity to a disappearing phase.
    # Partial-density residuals would vanish in proportion to alpha despite a
    # finite relative EOS error. Pure phases are outside this representation.
    return jnp.stack((W[..., 2] / W[..., 1] - rg,
                      W[..., 4] / (1 - W[..., 1]) - rl), axis=-1)
