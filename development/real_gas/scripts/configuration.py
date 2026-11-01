"""Physical and numerical settings associated with one benchmark case."""
from copy import deepcopy
import json

from .common import STUDY, identity


def case_settings(profile, fluid_case, regime, definitions=None):
    if definitions is None:
        definitions = json.loads((STUDY / "cases.json").read_text(encoding="utf-8"))
    physical = deepcopy(next(c for c in definitions["cases"] if c["id"] == fluid_case))
    physical["table"].update(profile.get("table_overrides", {}).get(fluid_case, {}))
    geometry = definitions["geometry"] | physical.pop("geometry_override", {})
    return dict(physical=physical, geometry=geometry, regime=regime,
                scheme=profile["scheme"], steady=profile["steady"], residual=profile["residual"],
                cold_start=profile.get("cold_start"), grid_sequence=profile["grids"])


def case_settings_id(profile, fluid_case, regime):
    return identity(case_settings(profile, fluid_case, regime))


def scheme_options(profile, method):
    """Resolve a report method to its explicit production Scheme options."""
    if method not in profile['fluxes']:
        raise ValueError(f'Unknown benchmark flux: {method}')
    options = dict(profile['scheme'], flux=method)
    options.update(profile.get('flux_options', {}).get(method, {}))
    return options


def scheme_matches(profile, method, saved):
    """Check all declared controls, allowing extra serialized default fields."""
    return all(saved.get(key) == value
               for key, value in scheme_options(profile, method).items())
