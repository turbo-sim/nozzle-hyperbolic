"""Development-only conservative aggregation of spatial Jacobian blocks.

The coarse nonlinear residual is always the original rediscretized residual.
Only its Newton matrix is approximated.  For the transported, integrated rows,
the approximation is ``R J_spatial,f P``: R sums child balances and P gives both
children the same conservative-density increment.  P is a local derivative of
the conservative recovery map, rather than primitive-variable interpolation.
Endpoint traces have zero storage and use identity injection.  The coarse
storage, cell constraints and boundary equations retain their own derivatives.

Assembly stores local band blocks throughout.  Dense matrices below occur only
in the explicitly requested, small-mesh ``validate_aggregation`` audit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.numerics.jacobians import make_block_operators


def _local_jacobian(function, cells):
    """Differentiate a batch-local physical callback without a global matrix."""
    _, push = jax.linearize(function, cells)
    seeds = jnp.broadcast_to(
        jnp.eye(cells.shape[1], dtype=cells.dtype)[:, None, :],
        (cells.shape[1], *cells.shape),
    )
    return jnp.moveaxis(jax.vmap(push)(seeds), 0, -1)


class AggregatedJacobian:
    """Bind adjacent 2:1 levels for conservative spatial block aggregation.

    ``spatial_blocks`` computes a fresh fine spatial AD Jacobian, whose cost
    must be included in benchmark assembly counters/timers.  Deeper levels can
    instead call ``aggregate_blocks`` with the already aggregated parent matrix
    and avoid rediscretized coarse spatial AD entirely.  During the ensuing
    coarse visit this spatial matrix is frozen; ``implicit_blocks``
    recomposes the exact coarse storage/algebraic part at each pseudo step.
    A caller must still check the original nonlinear residual and refresh or
    fall back when this approximate direction is inadequate.

    The physical contract needs an invertible local map from primitive state
    to (conservative densities, cell constraints).  Its inverse derivative is
    solved locally.  This is not a claim that the exact rediscretized coarse
    spatial Jacobian equals the aggregated fine Jacobian, especially at shocks
    and limiter switches.  Only local boundary coupling is supported here.
    """

    def __init__(self, fine_level, coarse_level):
        self.fine_level, self.coarse_level = fine_level, coarse_level
        fine, coarse = fine_level.operator, coarse_level.operator
        self.fine_operator, self.coarse_operator = fine, coarse
        nf, nc = fine.model.mesh.n_cells, coarse.model.mesh.n_cells
        nv, nt = fine.model.nvar, fine.ntransport
        rf, rc = fine.radius, coarse.radius
        if nf != 2 * nc:
            raise ValueError("Jacobian aggregation requires adjacent 2:1 levels")
        if (
            coarse.model.nvar != nv
            or coarse.ntransport != nt
            or coarse.nconstraints != fine.nconstraints
        ):
            raise ValueError("Both levels must have the same physical row contract")
        if any(
            getattr(op.model, "boundary_coupling", None) != "local"
            for op in (fine, coarse)
        ):
            raise ValueError("This banded aggregation supports local boundaries only")
        if fine.equation_policy_active or coarse.equation_policy_active:
            raise ValueError("Equation-changing policies cannot be aggregated")
        np.testing.assert_allclose(
            np.asarray(fine.model.mesh.volumes).reshape(nc, 2).sum(axis=1),
            np.asarray(coarse.model.mesh.volumes),
            rtol=2e-12,
            atol=1e-20,
        )
        self.nfine, self.ncoarse, self.nvar = nf, nc, nv
        self.ntransport, self.fine_radius, self.coarse_radius = nt, rf, rc

        # Parent indices include independent, zero-storage endpoint traces.
        parent = np.r_[0, 1 + np.arange(nf) // 2, nc + 1]
        self.parent = jnp.asarray(parent)
        row, diagonal, column, coarse_row, coarse_diagonal = [], [], [], [], []
        for i in range(1, nf + 1):
            for d in range(2 * rf + 1):
                j = i + d - rf
                if 0 <= j < nf + 2:
                    offset = int(parent[j] - parent[i])
                    if abs(offset) > rc:
                        raise ValueError("Aggregated stencil exceeds coarse block band")
                    row.append(i)
                    diagonal.append(d)
                    column.append(j)
                    coarse_row.append(parent[i])
                    coarse_diagonal.append(offset + rc)
        indices = tuple(
            jnp.asarray(index, dtype=jnp.int32)
            for index in (row, diagonal, column, coarse_row, coarse_diagonal)
        )

        def spatial_rows(flat):
            full = flat.reshape(fine.shape)
            rows = jnp.zeros(fine.shape, dtype=full.dtype)
            return rows.at[1:-1, :nt].set(fine.spatial_residual(full)).ravel()

        self.spatial_rows = jax.jit(spatial_rows)
        # Force local colored assembly even when the parent chooses dense AD;
        # no global fine ndof-by-ndof array is needed in this experiment.
        spatial_operators = make_block_operators(
            spatial_rows, nf + 2, nv, rf,
            jacobian_mode="colored", linear_solver="banded",
        )
        self.fine_spatial_blocks = jax.jit(spatial_operators["jacobian_blocks"])

        def prolongation_blocks(fine_state, coarse_state):
            qf = _local_jacobian(fine.model.cell_conserved, fine_state[1:-1])
            qh = _local_jacobian(coarse.model.cell_conserved, coarse_state[1:-1])
            # Scaling only conditions this small local solve; it cancels from
            # both sides, leaving physical density increments unchanged.
            qscale = jnp.asarray(fine.model.conservative_scales)[None, :, None]
            lhs, rhs = qf / qscale, jnp.repeat(qh, 2, axis=0) / qscale
            if fine.nconstraints:
                gf = _local_jacobian(fine.model.cell_constraints, fine_state[1:-1])
                gscale = jnp.asarray(fine.model.constraint_scales)[None, :, None]
                lhs = jnp.concatenate((lhs, gf / gscale), axis=1)
                rhs = jnp.concatenate((rhs, jnp.zeros_like(gf)), axis=1)
            interior = jnp.linalg.solve(lhs, rhs)
            identity = jnp.eye(nv, dtype=fine_state.dtype)[None, :, :]
            return jnp.concatenate((identity, interior, identity), axis=0)

        self.prolongation_blocks = jax.jit(prolongation_blocks)

        def aggregate_blocks(fine_state, coarse_state, fine_blocks):
            expected = (nf + 2, 2 * rf + 1, nv, nv)
            if fine_blocks.shape != expected:
                raise ValueError("Supplied fine spatial blocks have the wrong band shape")
            prolongation = prolongation_blocks(fine_state, coarse_state)
            i, d, j, parent_i, parent_d = indices
            contributions = jnp.matmul(fine_blocks[i, d], prolongation[j])
            result = jnp.zeros(
                (nc + 2, 2 * rc + 1, nv, nv), dtype=fine_state.dtype
            )
            return result.at[parent_i, parent_d].add(contributions)

        self.aggregate_blocks = jax.jit(aggregate_blocks)

        def spatial_blocks(fine_state, coarse_state):
            fine_blocks = self.fine_spatial_blocks(fine_state.ravel())
            return aggregate_blocks(fine_state, coarse_state, fine_blocks)

        self.spatial_blocks = jax.jit(spatial_blocks)

        def storage_algebraic_rows(flat, dt, time):
            full = flat.reshape(coarse.shape)
            conditioning = jnp.maximum(1.0, dt / coarse_level.reference_time)
            transport = coarse.storage(full[1:-1]) / (
                conditioning * coarse_level.content_scale
            )
            rows = jnp.zeros(coarse.shape, dtype=full.dtype)
            rows = rows.at[1:-1, :nt].set(transport)
            if coarse.nconstraints:
                rows = rows.at[1:-1, nt:].set(coarse.cell_constraints(full[1:-1]))
            boundary = coarse.boundary_residual(full, time)
            rows = rows.at[0].set(boundary[0])
            return rows.at[-1].set(boundary[1]).ravel()

        self.storage_algebraic_rows = jax.jit(storage_algebraic_rows)
        algebraic_operators = make_block_operators(
            storage_algebraic_rows, nc + 2, nv, rc,
            jacobian_mode="colored", linear_solver="banded",
        )
        self.storage_algebraic_blocks = jax.jit(
            algebraic_operators["jacobian_blocks"]
        )

        def implicit_blocks(coarse_state, dt, time, aggregated_spatial):
            exact = self.storage_algebraic_blocks(coarse_state.ravel(), dt, time)
            conditioning = jnp.maximum(1.0, dt / coarse_level.reference_time)
            weight = jnp.zeros(coarse.shape, dtype=coarse_state.dtype)
            weight = weight.at[1:-1, :nt].set(
                (dt / conditioning) / coarse_level.content_scale
            )
            return exact + weight[:, None, :, None] * aggregated_spatial

        self.implicit_blocks = jax.jit(implicit_blocks)


def _dense_from_blocks(blocks, radius):
    """Small validation helper; never used by production aggregation paths."""
    blocks = np.asarray(blocks)
    nblocks, _, nv, _ = blocks.shape
    matrix = np.zeros((nblocks, nv, nblocks, nv), dtype=blocks.dtype)
    for i in range(nblocks):
        for d in range(2 * radius + 1):
            j = i + d - radius
            if 0 <= j < nblocks:
                matrix[i, :, j, :] += blocks[i, d]
    return matrix.reshape(nblocks * nv, nblocks * nv)


def validate_aggregation(binding, fine_state, coarse_state, dt=1.0, time=0.0):
    """Audit true RJP and exact nonspatial rows on a deliberately small mesh.

    This validation uses dense AD solely as an independent reference.  It also
    reports (rather than assumes away) the difference from the native coarse
    spatial Jacobian.  No nonlinear solve or altered residual is introduced.
    """
    nf, nc, nv, nt = (
        binding.nfine, binding.ncoarse, binding.nvar, binding.ntransport
    )
    if nf > 32:
        raise ValueError("Dense aggregation audit is limited to 32 fine cells")
    fine_state, coarse_state = jnp.asarray(fine_state), jnp.asarray(coarse_state)
    jf = np.asarray(jax.jacfwd(binding.spatial_rows)(fine_state.ravel()))
    pblocks = np.asarray(binding.prolongation_blocks(fine_state, coarse_state))
    parents = np.asarray(binding.parent)
    prolongation = np.zeros((nf + 2, nv, nc + 2, nv))
    restriction = np.zeros((nc + 2, nv, nf + 2, nv))
    for j, parent in enumerate(parents):
        prolongation[j, :, parent, :] = pblocks[j]
    for i in range(1, nf + 1):
        restriction[parents[i], :nt, i, :nt] = np.eye(nt)
    p = prolongation.reshape((nf + 2) * nv, (nc + 2) * nv)
    r = restriction.reshape((nc + 2) * nv, (nf + 2) * nv)
    expected = r @ jf @ p
    spatial = binding.spatial_blocks(fine_state, coarse_state)
    actual = _dense_from_blocks(spatial, binding.coarse_radius)
    exact_nonspatial = np.asarray(
        jax.jacfwd(binding.storage_algebraic_rows)(coarse_state.ravel(), dt, time)
    )
    nonspatial = _dense_from_blocks(
        binding.storage_algebraic_blocks(coarse_state.ravel(), dt, time),
        binding.coarse_radius,
    )
    full = _dense_from_blocks(
        binding.implicit_blocks(coarse_state, dt, time, spatial),
        binding.coarse_radius,
    )
    weights = np.zeros((nc + 2, nv))
    weights[1:-1, :nt] = (
        dt / max(1.0, dt / binding.coarse_level.reference_time)
        / np.asarray(binding.coarse_level.content_scale)
    )
    expected_full = exact_nonspatial + weights.ravel()[:, None] * expected
    coarse = binding.coarse_operator

    def native_spatial(flat):
        state = flat.reshape(coarse.shape)
        rows = jnp.zeros(coarse.shape, dtype=state.dtype)
        return rows.at[1:-1, :nt].set(coarse.spatial_residual(state)).ravel()

    native = np.asarray(jax.jacfwd(native_spatial)(coarse_state.ravel()))
    qfine = np.asarray(_local_jacobian(binding.fine_operator.model.cell_conserved, fine_state[1:-1]))
    qcoarse = np.asarray(_local_jacobian(coarse.model.cell_conserved, coarse_state[1:-1]))
    density_error = np.max(np.abs(
        qfine @ pblocks[1:-1] - np.repeat(qcoarse, 2, axis=0)
    ))
    magnitude = max(float(np.max(np.abs(expected))), 1.0)
    return {
        "fine_cells": nf,
        "coarse_cells": nc,
        "rjp_max_abs_error": float(np.max(np.abs(actual - expected))),
        "rjp_relative_error": float(np.max(np.abs(actual - expected))) / magnitude,
        "supplied_blocks_wrapper_max_abs_error": float(np.max(np.abs(
            _dense_from_blocks(binding.aggregate_blocks(
                fine_state, coarse_state, binding.fine_spatial_blocks(fine_state.ravel())
            ), binding.coarse_radius) - actual
        ))),
        "exact_nonspatial_max_abs_error": float(np.max(np.abs(nonspatial - exact_nonspatial))),
        "implicit_composition_max_abs_error": float(np.max(np.abs(full - expected_full))),
        "conservative_density_prolongation_max_abs_error": float(density_error),
        "endpoint_identity_error": float(np.max(np.abs(pblocks[[0, -1]] - np.eye(nv)))),
        "native_coarse_spatial_difference_relative": float(np.max(np.abs(actual - native)))
        / max(float(np.max(np.abs(native))), 1.0),
        "coarse_residual": "original rediscretized nonlinear residual, unchanged",
        "approximation": "R(sum integrated transport rows) J_spatial,f P(conservative density increments)",
    }
