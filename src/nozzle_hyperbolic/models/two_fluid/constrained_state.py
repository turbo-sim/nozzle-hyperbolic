"""Alberto's eight-state representation and its six transported contents.

Reference: faafff704e54fd866abff95fcb88fc0a5217405c, solver.py:64-105.
No local projection, phase clipping, or hidden storage-recovery solver.
"""

import jax.numpy as jnp
import jaxprop as jxp


def conserved(W):
    p, alpha, mg, ug, ml, ul, eg, el = jnp.moveaxis(jnp.asarray(W), -1, 0)
    return jnp.stack((mg, ml, mg * ug, ml * ul,
                      eg + 0.5 * mg * ug**2, el + 0.5 * ml * ul**2), axis=-1)


def primitives(W):
    p, alpha, mg, ug, ml, ul, eg, el = jnp.moveaxis(jnp.asarray(W), -1, 0)
    return jnp.stack((p, ug, ul, alpha, (eg + p * alpha) / mg,
                      (el + p * (1 - alpha)) / ml), axis=-1)


def from_primitives(U, fluid_g, fluid_l):
    p, ug, ul, alpha, hg, hl = jnp.moveaxis(jnp.asarray(U), -1, 0)
    rg = fluid_g.get_state(jxp.HmassP_INPUTS, hg, p)["rho"]
    rl = fluid_l.get_state(jxp.HmassP_INPUTS, hl, p)["rho"]
    mg, ml = alpha * rg, (1 - alpha) * rl
    return jnp.stack((p, alpha, mg, ug, ml, ul,
                      mg * hg - p * alpha, ml * hl - p * (1 - alpha)), axis=-1)


def eos_constraints(W, fluid_g, fluid_l):
    U = primitives(W)
    rg = fluid_g.get_state(jxp.HmassP_INPUTS, U[..., 4], U[..., 0])["rho"]
    rl = fluid_l.get_state(jxp.HmassP_INPUTS, U[..., 5], U[..., 0])["rho"]
    # Intrinsic density constraints retain sensitivity to a disappearing phase.
    # Partial-density residuals would vanish in proportion to alpha despite a
    # finite relative EOS error. Pure phases are outside this representation.
    return jnp.stack((W[..., 2] / W[..., 1] - rg,
                      W[..., 4] / (1 - W[..., 1]) - rl), axis=-1)
