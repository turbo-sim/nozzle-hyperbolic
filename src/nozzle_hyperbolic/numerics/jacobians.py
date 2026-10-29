"""Local colored-JVP Jacobians for square cell-major balance systems."""

import operator

import jax
import jax.numpy as jnp

from .linear import solve_block_banded


def make_block_operators(
    residual, nblocks, nvar, radius, *, jacobian_mode="dense", linear_solver="dense"
):
    """Build dimension-parameterized Jacobian and Newton-direction operators.

    ``residual(new_flat, *args)`` must return a flat square system in the same
    cell-major ordering. Every row block i may depend only on blocks within
    i +/- radius. Algebraic boundary blocks are ordinary rows for coloring;
    include them in nblocks and in the actual dependency-radius calculation.

    The declared band is a mathematical assumption, not inferred from sampled
    derivatives. ``jacobian_difference`` compares colored and dense AD and is
    the required guard when model stencils change. Operators are traceable
    functions; callers choose their JIT boundary.
    """
    nblocks, nvar, radius = map(operator.index, (nblocks, nvar, radius))
    if nblocks < 1 or nvar < 1 or radius < 0:
        raise ValueError("nblocks/nvar must be positive and radius nonnegative")
    if jacobian_mode not in ("dense", "colored"):
        raise ValueError("jacobian_mode must be 'dense' or 'colored'")
    if linear_solver not in ("dense", "banded"):
        raise ValueError("linear_solver must be 'dense' or 'banded'")
    ndof = nblocks * nvar
    stride = nvar * (2 * radius + 1)
    ncolors = min(stride, ndof)
    dense_jacobian = jax.jacobian(residual, argnums=0)

    def colored_columns(new_flat, *args):
        _, jvp = jax.linearize(lambda point: residual(point, *args), new_flat)
        columns = jnp.arange(ndof)
        colors = jnp.arange(ncolors)
        seeds = (columns[None, :] % stride == colors[:, None]).astype(new_flat.dtype)
        return jax.vmap(jvp)(seeds)

    def colored_jacobian(new_flat, *args):
        compressed = colored_columns(new_flat, *args)
        rows = jnp.arange(ndof)
        row_blocks = rows // nvar
        jacobian = jnp.zeros((ndof, ndof), dtype=new_flat.dtype)
        for offset in range(-radius, radius + 1):
            for variable in range(nvar):
                column_blocks = row_blocks + offset
                columns = column_blocks * nvar + variable
                valid = (column_blocks >= 0) & (column_blocks < nblocks)
                safe = jnp.clip(columns, 0, ndof - 1)
                values = jnp.where(valid, compressed[safe % stride, rows], 0.0)
                jacobian = jacobian.at[rows, safe].add(values)
        return jacobian

    def jacobian_blocks(new_flat, *args):
        compressed = colored_columns(new_flat, *args)
        rows = jnp.arange(nblocks)
        diagonals = []
        for diagonal in range(2 * radius + 1):
            column_blocks = rows + diagonal - radius
            valid = (column_blocks >= 0) & (column_blocks < nblocks)
            columns = []
            for variable in range(nvar):
                color = (column_blocks * nvar + variable) % stride
                values = jnp.stack(
                    [compressed[color, nvar * rows + row_var] for row_var in range(nvar)],
                    axis=1,
                )
                columns.append(jnp.where(valid[:, None], values, 0.0))
            diagonals.append(jnp.stack(columns, axis=2))
        return jnp.stack(diagonals, axis=1)

    jacobian = colored_jacobian if jacobian_mode == "colored" else dense_jacobian

    def solve_blocks(blocks, rhs):
        return solve_block_banded(blocks, rhs, radius)

    def direction(new_flat, *args):
        value = residual(new_flat, *args)
        if linear_solver == "banded":
            return solve_blocks(jacobian_blocks(new_flat, *args), -value)
        return jnp.linalg.solve(jacobian(new_flat, *args), -value)

    def jacobian_difference(new_flat, *args):
        return jnp.max(jnp.abs(
            colored_jacobian(new_flat, *args) - dense_jacobian(new_flat, *args)
        ))

    return {
        "dense_jacobian": dense_jacobian,
        "colored_jacobian": colored_jacobian,
        "colored_columns": colored_columns,
        "jacobian_blocks": jacobian_blocks,
        "jacobian": jacobian,
        "solve_blocks": solve_blocks,
        "direction": direction,
        "jacobian_difference": jacobian_difference,
    }
