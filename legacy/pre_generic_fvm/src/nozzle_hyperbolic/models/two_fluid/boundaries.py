"""Unchanged definitions relocated from the TwoFluid solver modules."""
import jax
import jax.numpy as jnp


def _apply_ramped_inlet(U_h, U_h_bc, n_cells=3):
    """Blend a prescribed inlet state into the first n_cells of the domain."""
    weights = jnp.linspace(1.0, 0.0, n_cells)

    def _body(i, state):
        w = weights[i]
        return state.at[i].set((1.0 - w) * state[i] + w * U_h_bc)

    return jax.lax.fori_loop(0, n_cells, _body, U_h)
