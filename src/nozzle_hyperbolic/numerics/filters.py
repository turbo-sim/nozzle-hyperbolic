"""Conservative face insertion for optional fourth-difference damping."""

import jax.numpy as jnp


def fourth_difference_flux(flux, conserved, face_speed, strength):
    """Add internal-face third differences of transported conserved densities.

    Inputs are unweighted transported face fluxes (N+1,V), point densities
    (N+2,V), and model-provided face speed/sensor weights (N+1,). The assembler
    applies face measures after this operation. Endpoint fluxes are untouched,
    so the same actual-flux diagnostics retain exact telescoping conservation.
    """
    third = (
        conserved[3:] - 3.0 * conserved[2:-1] + 3.0 * conserved[1:-2] - conserved[:-3]
    )
    return flux.at[1:-1].add(strength * jnp.asarray(face_speed)[1:-1, None] * third)
