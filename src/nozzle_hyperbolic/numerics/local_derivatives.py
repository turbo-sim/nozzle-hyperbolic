"""Exact local Jacobians reused across differentiation seed directions.

The callback acts independently along its batch axis, as required by the FVM
face and cell-storage contracts. Geometry and constitutive data stay opaque
closure context. Derivatives are recomputed at every linearization point;
nothing is cached across Newton iterations. Neighbor-coupled sources and
boundary equations must not use this wrapper.
"""

import jax
import jax.numpy as jnp


def local_jvp(function):
    """Wrap a batch-local map of ``(items, variables)`` array arguments.

    The returned map preserves primal evaluation. Its custom JVP constructs
    small per-item Jacobians once and applies them to subsequent seed vectors,
    rather than propagating every global color through the full local graph.
    Inputs may have different variable counts; outputs are ``(items, outputs)``.
    """

    @jax.custom_jvp
    def evaluate(*states):
        return function(*states)

    @evaluate.defjvp
    def derivative(primals, tangents):
        if not primals or any(state.ndim != 2 for state in primals):
            raise ValueError("Local maps require (items, variables) arrays")
        count = primals[0].shape[0]
        if any(state.shape[0] != count for state in primals):
            raise ValueError("Local map arguments must have matching item counts")
        value, linear = jax.linearize(function, *primals)
        if value.ndim != 2 or value.shape[0] != count:
            raise ValueError("Local map output must be (items, outputs)")
        total = sum(state.shape[1] for state in primals)
        identity = jnp.eye(total, dtype=primals[0].dtype)
        seeds, offset = [], 0
        for state in primals:
            width = state.shape[1]
            seeds.append(
                jnp.broadcast_to(
                    identity[:, None, offset : offset + width], (total, count, width)
                )
            )
            offset += width
        columns = jax.vmap(linear)(*seeds)
        blocks = jnp.moveaxis(columns, 0, -1)
        tangent = jnp.concatenate(tangents, axis=1)
        return value, jnp.einsum("foi,fi->fo", blocks, tangent)

    return evaluate
