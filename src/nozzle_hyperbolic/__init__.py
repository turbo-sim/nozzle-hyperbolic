"""Model-independent finite volumes with explicitly selected physical models.

Importing the package has no plotting, exception-handler, EOS or model-loading
side effects. Public convenience exports are loaded only when requested.
"""

from importlib import import_module

__version__ = "0.0.1"

_EXPORTS = {
    name: (".solvers.fvm", name)
    for name in (
        "Mesh",
        "Scheme",
        "SteadySettings",
        "Solution",
        "ResidualPolicy",
        "Operator",
        "solve_single_grid",
        "solve_grid_sequence",
        "solve_transient",
        "solve_parameter_sequence",
        "prepare_continuation_state",
        "save_restart",
        "load_restart",
    )
}
_EXPORTS.update(
    {
        "NozzleModel": (".models.euler_nozzle.model", "NozzleModel"),
        "AnnularModel": (".models.euler_annular.model", "AnnularModel"),
        "AdvectionModel": (".models.scalar.advection", "AdvectionModel"),
        "BurgersModel": (".models.scalar.burgers", "BurgersModel"),
        "TwoFluidModel": (".models.two_fluid.model", "TwoFluidModel"),
    }
)
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
