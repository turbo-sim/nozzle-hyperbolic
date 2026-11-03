"""Active constitutive laws, grouped by physical mechanism.

Phase 1 is gas/vapor; phase 2 is liquid. All dimensional inputs use SI units.
These model-local kernels retain Alberto's names; configure a simulation with
``ClosureConfig`` rather than copying the dispatch logic from ``sources``.
"""

from .regimes import (
    REGIME_BLENDS,
    regime_weight,
)

from .mass_transfer import (
    calculate_mass_transfer_lv,
)

from .drag import (
    DRAG_LAWS,
    ishii_zuber_mixture_viscosity,
    calculate_drag_force_one_component,
)

from .heat_transfer import (
    LIQUID_HTC_BUBBLY,
    LIQUID_HTC_DROPLET,
    jakob_nusselt,
    liao_nusselt,
    select_liquid_htc,
    ranz_marshall_htc_two_fluid,
    calculate_heat_transfer_one_component,
)

from .interfacial_area import (
    interfacial_area_original,
    dispersed_fraction,
    interfacial_area_poly,
    interfacial_area_para,
    interfacial_area_segments,
    INTERFACIAL_AREA_LAWS,
    get_interfacial_area,
)

__all__ = [
    "REGIME_BLENDS",
    "regime_weight",
    "calculate_mass_transfer_lv",
    "DRAG_LAWS",
    "ishii_zuber_mixture_viscosity",
    "calculate_drag_force_one_component",
    "LIQUID_HTC_BUBBLY",
    "LIQUID_HTC_DROPLET",
    "jakob_nusselt",
    "liao_nusselt",
    "select_liquid_htc",
    "ranz_marshall_htc_two_fluid",
    "calculate_heat_transfer_one_component",
    "interfacial_area_original",
    "dispersed_fraction",
    "interfacial_area_poly",
    "interfacial_area_para",
    "interfacial_area_segments",
    "INTERFACIAL_AREA_LAWS",
    "get_interfacial_area",
]
