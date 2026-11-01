"""Strict physical-case schema and package-resource loading."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
import re

import numpy as np
import yaml


MODELS = ("euler_nozzle", "euler_annular")
_TOP_KEYS = {
    "schema_version", "model", "id", "label", "description", "fluid",
    "geometry", "inlet", "boundaries", "reference", "walls", "accuracy",
    "known_limit",
}


def _keys(value, allowed, required, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{context}: unknown keys {sorted(unknown)}, missing {sorted(missing)}")


def _positive(value, context, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{context} must be finite and {'nonnegative' if zero else 'positive'}")


def _validate(data):
    _keys(data, _TOP_KEYS, _TOP_KEYS - {"description", "known_limit"}, "case")
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise ValueError("Only physical case schema_version 1 is supported")
    model = data["model"]
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if not isinstance(data["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", data["id"]):
        raise ValueError("id must be a lowercase identifier")
    if not isinstance(data["label"], str) or not data["label"].strip():
        raise ValueError("label must be nonempty text")
    if data.get("known_limit") is not None and not isinstance(data["known_limit"], str):
        raise ValueError("known_limit must be text or null")
    _keys(data["fluid"], ("backend", "name"), ("backend", "name"), "fluid")
    if data["fluid"]["backend"] != "perfect_gas" or not isinstance(data["fluid"]["name"], str):
        raise ValueError("These analytical cases require a named perfect_gas backend")
    geometry = data["geometry"]
    kind = geometry.get("kind") if isinstance(geometry, dict) else None
    geometry_fields = {
        "symmetric_nozzle": ("kind", "length", "inlet_area", "throat_area"),
        "radial": ("kind", "r_in", "r_out", "height"),
        "conical": ("kind", "length", "r_in", "height", "inclination_degrees", "semi_divergence_degrees"),
        "circular_arc": ("kind", "center", "radius", "theta_start_degrees", "theta_end_degrees", "height"),
    }
    if kind not in geometry_fields or (model == "euler_nozzle") != (kind == "symmetric_nozzle"):
        raise ValueError("Geometry kind is incompatible with physical model")
    _keys(geometry, geometry_fields[kind], geometry_fields[kind], "geometry")
    for key in ("length", "inlet_area", "throat_area", "r_in", "r_out", "height", "radius"):
        if key in geometry:
            _positive(geometry[key], "geometry." + key)
    if kind == "symmetric_nozzle" and geometry["throat_area"] >= geometry["inlet_area"]:
        raise ValueError("The nozzle verification family requires throat_area < inlet_area")
    inlet = data["inlet"]
    permitted = ("p0", "T0") if model == "euler_nozzle" else ("p0", "T0", "M_m", "alpha_degrees", "alpha_tangent")
    _keys(inlet, permitted, ("p0", "T0") if model == "euler_nozzle" else ("p0", "T0", "M_m"), "inlet")
    for key in ("p0", "T0"):
        _positive(inlet[key], "inlet." + key)
    if model == "euler_annular":
        if inlet["M_m"] != "sonic_exit":
            _positive(inlet["M_m"], "inlet.M_m")
        if "alpha_degrees" in inlet and "alpha_tangent" in inlet:
            raise ValueError("Specify alpha_degrees or alpha_tangent, not both")
        for key in ("alpha_degrees", "alpha_tangent"):
            if key in inlet and (isinstance(inlet[key], bool) or not np.isfinite(inlet[key])):
                raise ValueError(f"inlet.{key} must be finite")
        if abs(inlet.get("alpha_degrees", 0)) >= 90:
            raise ValueError("Inlet flow angle must be strictly within (-90,90) degrees")
    boundaries = data["boundaries"]
    _keys(boundaries, ("inlet", "outlet", "outlet_formulation"), ("inlet", "outlet", "outlet_formulation"), "boundaries")
    if boundaries["inlet"] not in (("subsonic",) if model == "euler_nozzle" else ("subsonic", "supersonic")):
        raise ValueError("Unsupported inlet regime")
    if boundaries["outlet"] not in ("subsonic", "supersonic", "auto"):
        raise ValueError("Unsupported outlet regime")
    if boundaries["outlet_formulation"] not in ("acoustic", "finite_wave"):
        raise ValueError("Unsupported outlet formulation")
    reference = data["reference"]
    if model == "euler_nozzle":
        _keys(reference, ("condition", "value"), ("condition",), "reference")
        condition = reference["condition"]
        if condition not in ("throat_mach", "critical", "shock_fraction", "shock_exit", "external_midpoint", "adapted", "adapted_factor"):
            raise ValueError("Unsupported nozzle reference condition")
        if condition in ("throat_mach", "shock_fraction", "adapted_factor"):
            _positive(reference.get("value"), "reference.value")
            if reference["value"] > 1 or (condition == "shock_fraction" and reference["value"] >= 1):
                raise ValueError("Reference Mach/fraction/factor is outside its valid range")
        elif "value" in reference:
            raise ValueError("This reference condition has no value parameter")
    else:
        _keys(reference, ("shock_fraction",), ("shock_fraction",), "reference")
        fraction = reference["shock_fraction"]
        if fraction is not None:
            _positive(fraction, "reference.shock_fraction")
            if fraction >= 1 or boundaries["inlet"] != "supersonic":
                raise ValueError("Internal annular shocks need 0 < fraction < 1 and supersonic inlet")
    _keys(data["walls"], ("shear_stress", "heat_flux"), ("shear_stress", "heat_flux"), "walls")
    if data["walls"] != {"shear_stress": 0.0, "heat_flux": 0.0}:
        raise ValueError("Steady inviscid analytical cases require zero wall shear and heat flux")
    _keys(data["accuracy"], ("recommended_finest_cells", "normalized_field_l1", "conservative_normalized_l1", "Cp_length_mean_absolute", "shock_position_cells"), ("recommended_finest_cells",), "accuracy")
    finest = data["accuracy"]["recommended_finest_cells"]
    if isinstance(finest, bool) or not isinstance(finest, int) or finest < 2:
        raise ValueError("accuracy.recommended_finest_cells must be an integer >= 2")
    for key, value in data["accuracy"].items():
        if key != "recommended_finest_cells":
            _positive(value, "accuracy." + key)
    return data


@dataclass(frozen=True)
class CaseDefinition:
    """Validated physical data and the exact YAML/source identity.

    ``data``/``config``/``to_dict`` return independent copies so consumers cannot
    silently modify a definition while retaining its original fingerprint.
    """

    _data: dict
    source_name: str
    source_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "_data", _validate(deepcopy(self._data)))
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("Case source_name must be nonempty text")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("Case source_sha256 must be a lowercase SHA-256 digest")

    @property
    def case_id(self):
        return f"{self._data['model']}/{self._data['id']}"

    @property
    def model_name(self):
        return self._data["model"]

    @property
    def name(self):
        return self._data["id"]

    @property
    def label(self):
        return self._data["label"]

    @property
    def data(self):
        return deepcopy(self._data)

    config = data

    def to_dict(self):
        return self.data


def case_from_dict(data, *, source_name="<mapping>"):
    """Validate application-provided physical data without numerical settings."""
    data = _validate(deepcopy(data))
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return CaseDefinition(data, source_name, sha256(payload).hexdigest())


def available_cases(model=None):
    """Return stable qualified IDs from installed package resources."""
    if model is not None and model not in MODELS:
        raise ValueError(f"Unknown physical model {model!r}")
    result = []
    for name in MODELS if model is None else (model,):
        directory = files("nozzle_hyperbolic").joinpath("models", name, "cases")
        result.extend(f"{name}/{item.name[:-5]}" for item in directory.iterdir() if item.name.endswith(".yaml"))
    return tuple(sorted(result))


def load_case(case_id):
    """Load one versioned YAML definition from the installed physical model."""
    if not isinstance(case_id, str) or case_id.count("/") != 1:
        raise ValueError("Use a qualified case ID, for example euler_nozzle/subsonic")
    model, name = case_id.split("/")
    if model not in MODELS or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Invalid physical case ID")
    resource = files("nozzle_hyperbolic").joinpath("models", model, "cases", name + ".yaml")
    if not resource.is_file():
        raise ValueError(f"Unknown case {case_id!r}; choose from available_cases()")
    raw = resource.read_bytes()
    data = _validate(yaml.safe_load(raw))
    if data["model"] != model or data["id"] != name:
        raise ValueError("Case resource name must agree with its model/id fields")
    return CaseDefinition(data, f"models/{model}/cases/{name}.yaml", sha256(raw).hexdigest())
