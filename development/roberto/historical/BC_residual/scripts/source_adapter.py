"""Explicit production operators for the retained coupled-boundary study.

The factory exposes its physical residual separately from its boundary maps
and publishes numerical construction metadata. Both study formulations use
that same physical code and the shared colored/banded numerical engine. No
private closure inspection, cloning or monkeypatching is required.
"""

import jax
import jax.numpy as jnp

from nozzle_hyperbolic.numerics.jacobians import make_block_operators


def value(function, name):
    """Compatibility lookup for the study's three historical setting names.

    CoupledStep's call sites remain unchanged, but values now come from the
    returned kernel's explicit public metadata rather than Python closures.
    Unsupported names fail loudly instead of inspecting implementation details.
    """
    aliases = {
        "_newton_scale": "newton_scale",
        "newton_iter": "newton_iter",
        "newton_tol": "newton_tol",
    }
    metadata = getattr(function, "solver_metadata", None)
    if metadata is None or name not in aliases or aliases[name] not in metadata:
        raise RuntimeError(f"Source kernel does not publish study setting {name!r}")
    return metadata[aliases[name]]


def _metadata(problem):
    metadata = problem.operators.get("solver_metadata")
    if metadata is None:
        raise RuntimeError("Source factory must publish solver_metadata with return_operators=True")
    if metadata["nvar"] != 3:
        raise ValueError("This retained nozzle study expects three primitive variables")
    return metadata


def banded_newton_for_full_state(problem, residual):
    """Build augmented study rows through the production shared block engine."""
    metadata = _metadata(problem)
    if metadata["linear_solver"] != "banded":
        raise ValueError("The matched comparison requires source banded Newton")
    # Boundary equations read their boundary and nearest two physical cells.
    # The full augmented radius is therefore at least two, including constant
    # reconstruction; do not inherit a narrower interior-only band silently.
    radius = max(2, metadata["stencil_radius"])
    operators = make_block_operators(
        residual, len(problem.x), metadata["nvar"], radius,
        jacobian_mode=metadata["jacobian_mode"], linear_solver="banded",
    )

    def solve(blocks, residual_value):
        # Preserve CoupledStep's existing residual-to-Newton-direction contract.
        return operators["solve_blocks"](blocks, -residual_value)

    return operators["direction"], operators["jacobian_blocks"], solve


def independent_ghost_operators(problem):
    """Expose identical physical rows with independent current ghost values.

    The transient compatibility wrapper keeps the study's former signature.
    Its old-state argument contains current boundary slots and old physical
    slots, as explicitly assembled by CoupledStep. No boundary map is invoked
    by this wrapper or the spatial monitor.
    """
    _metadata(problem)
    required = ("physical_residual", "inlet_boundary", "outlet_boundary")
    missing = [name for name in required if name not in problem.operators]
    if missing:
        raise RuntimeError(f"Source factory does not publish physical study operators: {missing}")
    physical = problem.operators["physical_residual"]
    shape = (len(problem.x), 3)

    def transient(interior_flat, mixed_old_flat, dt):
        mixed = mixed_old_flat.reshape(shape)
        current = jnp.concatenate(
            [mixed[:1], interior_flat.reshape(shape[0] - 2, 3), mixed[-1:]], axis=0
        )
        return physical(current, mixed, dt)

    @jax.jit
    def spatial(full):
        return physical(full, full, jnp.float64(1.0))

    return dict(
        spatial=spatial,
        inlet=problem.operators["inlet_boundary"],
        outlet=problem.operators["outlet_boundary"],
        transient=transient,
    )
