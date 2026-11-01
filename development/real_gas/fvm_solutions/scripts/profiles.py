"""Case-specific compatibility of honest, immutable campaign profile versions.

Changing another fluid's property table does not change this case's equations
or controls. Original global profile hashes remain on every saved result.
"""
import json
from copy import deepcopy

from .common import BASELINE, OUTPUT, identity, save


def register(profile):
    path = OUTPUT / "profile_registry.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = identity(profile)
    physical = json.loads((BASELINE / "cases.json").read_text(encoding="utf-8"))
    entry = dict(profile=profile, physical_definitions=physical)
    if key in records and records[key] != entry:
        raise RuntimeError("Physical definitions changed beneath an existing numerical profile")
    if key in records:
        return key
    records[key] = entry
    save(path, records)
    return key


def effective_profile(entry, fluid_case, regime):
    profile = entry["profile"]
    definitions = entry["physical_definitions"]
    physical = deepcopy(next(c for c in definitions["cases"] if c["id"] == fluid_case))
    physical["table"].update(profile.get("table_overrides", {}).get(fluid_case, {}))
    geometry = definitions["geometry"] | physical.pop("geometry_override", {})
    return dict(physical=physical, geometry=geometry, regime=regime,
                scheme=profile["scheme"], steady=profile["steady"], residual=profile["residual"],
                cold_start=profile.get("cold_start"), grid_sequence=profile["grids"])


def compatible(old_key, new_key, fluid_case, regime):
    path = OUTPUT / "profile_registry.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if old_key == new_key:
        return True
    if old_key not in records or new_key not in records:
        return False
    return effective_profile(records[old_key], fluid_case, regime) == effective_profile(records[new_key], fluid_case, regime)


def validate_existing(profile):
    new_key = register(profile)
    for path in (OUTPUT / "runs").glob("*/N*.json"):
        row = json.loads(path.read_text())
        regime = "adapted" if row["case_id"].endswith("_adapted") else "shock"
        fluid_case = row["case_id"][:-(len(regime) + 1)]
        if not compatible(row["profile_sha256"], new_key, fluid_case, regime):
            raise RuntimeError(f"Existing stage {row['id']} has different effective case settings")
    return new_key
