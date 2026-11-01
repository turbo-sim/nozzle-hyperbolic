"""Independent analytic/semi-analytic nozzle references, never FVM dependencies.

Perfect-gas references deliberately use ideal-gas relations. General-EOS
references accept EOS properties independently of the numerical model. The
density-EOS reference supports equilibrium branches through a supplied bounded
EOS, including bicubic properties. Imports are lazy, so importing the numerical
model does not load these tools.
"""

from importlib import import_module

_EXPORTS = {
    **{
        name: ".perfect_gas"
        for name in (
            "PerfectGasReference",
            "area_mach",
            "mach_from_area_ratio",
            "pressure_ratio",
            "normal_shock",
            "subsonic_reference",
        )
    },
    "sod_exact": ".shock_tube",
    "sod_exact_with_sound_speeds": ".shock_tube",
    "nozzle_exact_real_gas": ".general_eos",
    "SmoothPath": ".density_eos",
    "solve_shock_location": ".density_eos",
    "sample_profile": ".density_eos",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
