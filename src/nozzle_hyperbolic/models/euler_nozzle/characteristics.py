"""General-EOS conservative Euler eigenspace projectors."""

import jax.numpy as jnp


def conservative_projectors(U, rho, a, gruneisen):
    """Return conservative Euler projectors without inverting the flux Jacobian.

    For primitive W=(p,u,h), C=dQ/dW has determinant rho**2/a**2.
    The thermodynamic identities rho_p=(1+G)/a**2 and
    rho_h=-rho*G/a**2 give the explicit inverse below. With A the
    primitive Euler matrix and P_i=v_i*l_i, K=dF/dW=C*A, so
    K*P_i*K^-1=C*P_i*C^-1 wherever K is invertible. The C representation
    also remains regular at u=0 and u=+/-a, where K is singular.

    Returns shape (..., 3 waves, 3 balances, 3 conservative variables).
    Admissible EOS states require finite rho>0 and a>0.
    """
    u, h = U[..., 1], U[..., 2]
    rho_p = (1.0 + gruneisen) / a**2
    rho_h = -rho * gruneisen / a**2
    total_h = h + 0.5 * u**2
    zero = jnp.zeros_like(u)
    one = jnp.ones_like(u)
    storage = jnp.stack(
        [
            jnp.stack([rho_p, zero, rho_h], axis=-1),
            jnp.stack([u * rho_p, rho, u * rho_h], axis=-1),
            jnp.stack(
                [total_h * rho_p - one, rho * u, total_h * rho_h + rho],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    chi = 0.5 * u**2 - h
    inverse_storage = jnp.stack(
        [
            jnp.stack(
                [a**2 + gruneisen * chi, -gruneisen * u, gruneisen],
                axis=-1,
            ),
            jnp.stack([-u / rho, one / rho, zero], axis=-1),
            jnp.stack(
                [
                    (a**2 + (gruneisen + one) * chi) / rho,
                    -(gruneisen + one) * u / rho,
                    (gruneisen + one) / rho,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    right = jnp.stack(
        [
            jnp.stack([rho * a, one, a], axis=-1),
            jnp.stack([-rho * a, one, -a], axis=-1),
            jnp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    left = jnp.stack(
        [
            jnp.stack([0.5 / (rho * a), 0.5 * one, zero], axis=-1),
            jnp.stack([-0.5 / (rho * a), 0.5 * one, zero], axis=-1),
            jnp.stack([-one / rho, zero, one], axis=-1),
        ],
        axis=-2,
    )
    conservative_right = jnp.einsum("...ij,...kj->...ki", storage, right)
    conservative_left = jnp.einsum("...ki,...ij->...kj", left, inverse_storage)
    return conservative_right[..., :, None] * conservative_left[..., None, :]
