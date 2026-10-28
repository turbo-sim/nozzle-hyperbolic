"""Read-only experimental access to the existing factory's private closures.

The production file is neither edited nor monkeypatched. A separate Python
function shares its residual code and numerical kernels, replacing only its
two ghost-elimination closures with identity maps. This deliberately private
adapter avoids copying the FVM formulas; parity checks guard the experiment.
It must fail loudly if the production factory's closure layout changes.
"""

from types import FunctionType

import jax
import jax.numpy as jnp


def closure(function):
    function = getattr(function, "__wrapped__", function)
    return dict(zip(function.__code__.co_freevars, function.__closure__ or ()))


def value(function, name):
    try:
        return closure(function)[name].cell_contents
    except KeyError as error:
        raise RuntimeError(f"Production closure changed: missing {name}") from error


def cell(content):
    return (lambda: content).__closure__[0]


def bind(function, replacements):
    """Clone a source closure with explicit substitutions; never mutate it."""
    slots = closure(function)
    if not replacements.keys() <= slots.keys():
        raise RuntimeError(
            f"Source closure changed: {replacements.keys() - slots.keys()}"
        )
    return FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        tuple(
            cell(replacements[name]) if name in replacements else slots[name]
            for name in function.__code__.co_freevars
        ),
    )


def banded_newton_for_full_state(problem, residual):
    """Reuse source colored assembly and block Thomas code for N ghost-inclusive rows.

    Only residual and cell-count bindings change. Ghost rows use the same
    three-variable cell layout and the existing width-two local stencil.
    """
    newton = value(problem.step, "_newton_dU")
    if value(newton, "linear_solver") != "banded":
        raise ValueError("The matched comparison requires source banded Newton")
    original_blocks = value(newton, "_jac_blocks")
    original_columns = value(original_blocks, "_colored_columns")
    original_solve = value(newton, "_banded_solve")
    width = value(original_blocks, "_sw")
    if width != 2:
        raise ValueError("Coupled BCs require the baseline width-two MUSCL stencil")
    count = len(problem.x)
    columns = bind(
        original_columns,
        {
            "_residual": residual,
            "_n_colors": min(value(original_columns, "_stride"), 3 * count),
        },
    )
    blocks = bind(original_blocks, {"_Nc": count, "_colored_columns": columns})
    groups = -(-count // width)
    solve = bind(original_solve, {"_Nc": count, "_M": groups, "_Ncp": groups * width})
    direction = bind(
        newton,
        {
            "_residual": residual,
            "_jac_blocks": blocks,
            "_banded_solve": solve,
        },
    )
    return direction, blocks, solve


def independent_ghost_operators(problem):
    original = value(value(problem.step, "_newton_dU"), "_residual")
    slots = closure(original)
    if not {"_bc_inlet", "_bc_outlet"}.issubset(slots):
        raise RuntimeError("Production residual no longer exposes both BC maps")
    original_inlet = slots["_bc_inlet"].cell_contents
    original_outlet = slots["_bc_outlet"].cell_contents
    identity = lambda state: state
    replaced = tuple(
        cell(identity) if name in ("_bc_inlet", "_bc_outlet") else slots[name]
        for name in original.__code__.co_freevars
    )
    raw = FunctionType(
        original.__code__,
        original.__globals__,
        "experimental_residual_with_independent_ghosts",
        original.__defaults__,
        replaced,
    )

    @jax.jit
    def spatial(full):
        return raw(full[1:-1].ravel(), full.ravel(), jnp.float64(1.0))

    return dict(
        spatial=spatial,
        inlet=jax.jit(original_inlet),
        outlet=jax.jit(original_outlet),
        transient=raw,
    )
