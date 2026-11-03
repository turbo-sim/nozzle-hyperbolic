"""General-EOS annular Euler physics for the shared FVM machinery."""

__all__ = ["AnnularModel", "AnnularQuantities", "heat_flux_wall", "shear_stress_wall"]


def __getattr__(name):
    if name == "AnnularModel":
        from .model import AnnularModel

        return AnnularModel
    if name == "AnnularQuantities":
        from .diagnostics import AnnularQuantities

        return AnnularQuantities
    if name in ("heat_flux_wall", "shear_stress_wall"):
        from ..euler_nozzle import closures

        return getattr(closures, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
