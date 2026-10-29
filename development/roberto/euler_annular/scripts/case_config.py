"""Validated YAML inputs for the annular verification study.

Geometry, physical data, numerical settings and monitor choices are data, not
case-name branches. Perfect-gas reference construction remains verification
code; the production annular solver still accepts general-EOS fluid objects.
"""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import yaml

from nozzle_hyperbolic.geometry.annular import (
    circular_arc_channel, conical_channel, radial_channel)
from nozzle_hyperbolic.models.euler_annular.analytical import PerfectGasReference


STUDY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STUDY / "cases" / "verification.yaml"


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys rather than silently overwriting inputs."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if (isinstance(value, dict) and isinstance(result.get(key), dict)
                and not (key == "geometry" and "kind" in value
                         and value["kind"] != result[key].get("kind"))):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _keys(value, allowed, where, required=()):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where} must be a mapping with string keys")
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        raise ValueError(f"Invalid {where}: unknown={sorted(extra)}, missing={sorted(missing)}")


def _positive(value, where, *, zero=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{where} must be finite and {'nonnegative' if zero else 'positive'}")
    if integer and (not isinstance(value, int) or value < (0 if zero else 1)):
        raise ValueError(f"{where} must be an {'nonnegative' if zero else 'positive'} integer")


def _read(path, stack, sources):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Cyclic YAML inheritance: {path}")
    raw = path.read_bytes()
    document = yaml.load(raw, Loader=_UniqueLoader)
    _keys(document, ("extends", "schema_version", "defaults", "default_case", "cases"), str(path))
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version in {path}; expected 1")
    sources[str(path)] = sha256(raw).hexdigest()
    inherited = {}
    if "extends" in document:
        if not isinstance(document["extends"], str):
            raise ValueError("extends must be one relative or absolute YAML filename")
        inherited = _read(path.parent / document["extends"], (*stack, path), sources)
    return _merge(inherited, {key: value for key, value in document.items() if key != "extends"})


def available_cases(path=DEFAULT_CONFIG):
    document = _read(path, (), {})
    cases = document.get("cases", {})
    _keys(cases, cases, "cases")
    return tuple(cases)


def validate_config(config):
    """Reject unsupported or misspelled settings before geometry/JIT work."""
    allowed = ("model", "fluid", "geometry", "inlet", "boundaries", "reference",
               "mesh", "numerical", "walls", "residuals", "monitors", "run")
    _keys(config, allowed, "case", allowed)
    if config["model"] != "euler_annular":
        raise ValueError("This runner currently supports model: euler_annular only")
    _keys(config["fluid"], ("model", "name"), "fluid", ("model", "name"))
    if config["fluid"]["model"] != "perfect_gas" or not isinstance(config["fluid"]["name"], str):
        raise ValueError("The verification runner requires fluid.model: perfect_gas and a fluid name")
    geometry = config["geometry"]
    kind = geometry.get("kind") if isinstance(geometry, dict) else None
    geometry_keys = {
        "radial": ("kind", "r_in", "r_out", "height"),
        "conical": ("kind", "length", "r_in", "height", "inclination_degrees", "semi_divergence_degrees"),
        "circular_arc": ("kind", "center", "radius", "theta_start_degrees", "theta_end_degrees", "height"),
    }
    if kind not in geometry_keys:
        raise ValueError(f"Unsupported geometry kind: {kind!r}")
    _keys(geometry, geometry_keys[kind], "geometry", geometry_keys[kind])
    for key in ("r_in", "r_out", "height", "length", "radius"):
        if key in geometry:
            _positive(geometry[key], f"geometry.{key}")
    for key, value in geometry.items():
        if key.endswith("_degrees") and (isinstance(value, bool) or not np.isscalar(value)
                                       or not np.isfinite(value)):
            raise ValueError(f"geometry.{key} must be finite")
    if kind == "radial" and geometry["r_in"] == geometry["r_out"]:
        raise ValueError("Radial inlet and outlet radii must differ")
    if kind == "circular_arc":
        center = np.asarray(geometry["center"], dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("geometry.center must contain two finite coordinates")
        if geometry["theta_start_degrees"] == geometry["theta_end_degrees"]:
            raise ValueError("Circular-arc angles must differ")
    inlet = config["inlet"]
    _keys(inlet, ("p0", "T0", "M_m", "alpha_degrees", "alpha_tangent"), "inlet", ("p0", "T0", "M_m"))
    for key in ("p0", "T0"):
        _positive(inlet[key], f"inlet.{key}")
    if inlet["M_m"] != "sonic_exit":
        _positive(inlet["M_m"], "inlet.M_m")
    if "alpha_degrees" in inlet and "alpha_tangent" in inlet:
        raise ValueError("Choose either inlet.alpha_degrees or inlet.alpha_tangent")
    for key in ("alpha_degrees", "alpha_tangent"):
        if key in inlet and (isinstance(inlet[key], bool) or not np.isscalar(inlet[key])
                             or not np.isfinite(inlet[key])):
            raise ValueError(f"inlet.{key} must be finite")
    boundaries = config["boundaries"]
    _keys(boundaries, ("inlet", "outlet", "back_pressure"), "boundaries", ("inlet", "outlet", "back_pressure"))
    if any(boundaries[key] not in ("subsonic", "supersonic") for key in ("inlet", "outlet")):
        raise ValueError("Boundary regimes must be subsonic or supersonic")
    if boundaries["back_pressure"] != "reference":
        _positive(boundaries["back_pressure"], "boundaries.back_pressure")
    reference = config["reference"]
    _keys(reference, ("shock_fraction",), "reference", ("shock_fraction",))
    shock = reference["shock_fraction"]
    if shock is not None and (isinstance(shock, bool) or not isinstance(shock, (int, float))
                              or not np.isfinite(shock) or not 0 < shock < 1):
        raise ValueError("reference.shock_fraction must be null or strictly between 0 and 1")
    mesh = config["mesh"]
    _keys(mesh, ("cells", "refinement"), "mesh", ("cells", "refinement"))
    _positive(mesh["cells"], "mesh.cells", integer=True)
    if mesh["cells"] < 2 or not isinstance(mesh["refinement"], list) or not mesh["refinement"]:
        raise ValueError("Use at least two physical cells and a nonempty refinement list")
    for count in mesh["refinement"]:
        _positive(count, "mesh.refinement", integer=True)
    if mesh["refinement"] != sorted(set(mesh["refinement"])) or min(mesh["refinement"]) < 2:
        raise ValueError("mesh.refinement must contain increasing distinct cell counts >= 2")
    _keys(config["numerical"], ("reconstruction", "flux_method", "delta_factor", "newton_iterations",
          "newton_tolerance", "newton_residual_tolerance", "max_backtracks", "linear_solver", "jacobian_mode"), "numerical")
    from nozzle_hyperbolic.models.euler_annular.fvm import AnnularOptions
    AnnularOptions(**config["numerical"])
    _keys(config["walls"], ("shear_stress", "heat_flux"), "walls", ("shear_stress", "heat_flux"))
    _positive(config["walls"]["shear_stress"], "walls.shear_stress", zero=True)
    if (isinstance(config["walls"]["heat_flux"], bool)
            or not isinstance(config["walls"]["heat_flux"], (int, float))
            or not np.isfinite(config["walls"]["heat_flux"])):
        raise ValueError("walls.heat_flux must be finite")
    _keys(config["residuals"], ("norm", "weighting", "relative_tolerance", "absolute_tolerance",
          "boundary_tolerance", "global_tolerance", "initial_floor"), "residuals")
    if config["residuals"].get("norm", "l2") not in ("l1", "l2"):
        raise ValueError("residuals.norm must be l1 or l2")
    if config["residuals"].get("weighting", "volume") not in ("volume", "cell"):
        raise ValueError("residuals.weighting must be volume or cell")
    for key, value in config["residuals"].items():
        if key not in ("norm", "weighting"):
            _positive(value, f"residuals.{key}", zero=key == "initial_floor")
    monitors = config["monitors"]
    if not isinstance(monitors, list):
        raise ValueError("monitors must be a list; an empty list disables monitors")
    names = set()
    for monitor in monitors:
        _keys(monitor, ("name", "quantity", "reduction", "parameters", "convergence", "window",
              "absolute_tolerance", "relative_tolerance", "reference"), "monitor", ("name", "quantity", "reduction"))
        name = monitor["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Monitor names must be nonempty and unique")
        names.add(name)
        choices = {"pressure_coefficient": ("outlet", "length_mean"), "mass_flow": ("inlet", "outlet", "mean")}
        if monitor["quantity"] not in choices or monitor["reduction"] not in choices[monitor["quantity"]]:
            raise ValueError(f"Unsupported quantity/reduction for monitor {name!r}")
        _keys(monitor.get("parameters", {}), ("pressure_total",), f"monitor {name}.parameters")
        if monitor["quantity"] == "mass_flow" and monitor.get("parameters"):
            raise ValueError("mass_flow monitors do not accept parameters")
        if "pressure_total" in monitor.get("parameters", {}):
            _positive(monitor["parameters"]["pressure_total"], "monitor.parameters.pressure_total")
        if not isinstance(monitor.get("convergence", False), bool):
            raise ValueError("monitor.convergence must be true or false")
        _positive(monitor.get("window", 20), "monitor.window", integer=True)
        if monitor.get("window", 20) < 2:
            raise ValueError("Monitor stability needs a window of at least two samples")
        for key in ("absolute_tolerance", "relative_tolerance"):
            _positive(monitor.get(key, 0.0), f"monitor.{key}", zero=True)
        if (monitor.get("absolute_tolerance", 1e-8) == 0
                and monitor.get("relative_tolerance", 0.0) == 0):
            raise ValueError("A monitor needs at least one positive stability tolerance")
        _positive(monitor.get("reference", 1.0), "monitor.reference")
    run = config["run"]
    _keys(run, ("max_steps", "cfl", "cfl_growth", "cfl_shrink", "cfl_max", "max_retries", "stagnation_steps", "print_every"), "run")
    for key, value in run.items():
        _positive(value, f"run.{key}", zero=key in ("max_retries", "max_steps", "stagnation_steps"),
                  integer=key in ("max_steps", "print_every", "max_retries", "stagnation_steps"))
    if not isinstance(run.get("max_retries", 12), int):
        raise ValueError("run.max_retries must be an integer")
    if run.get("cfl_shrink", .5) >= 1 or not 1 < run.get("cfl_growth", 1.2) <= 2:
        raise ValueError("Use cfl_shrink < 1 and 1 < cfl_growth <= 2")
    if run.get("cfl_max", 1e6) < run.get("cfl", .2):
        raise ValueError("run.cfl_max must be >= run.cfl")
    return config


def load_case_config(path=DEFAULT_CONFIG, name=None):
    sources = {}
    document = _read(path, (), sources)
    cases = document.get("cases", {})
    _keys(cases, cases, "cases")
    name = document.get("default_case") if name is None else name
    if name not in cases:
        raise ValueError(f"Unknown case {name!r}; choose from {', '.join(cases)}")
    if not isinstance(cases[name], dict):
        raise ValueError(f"Case override {name!r} must be a mapping")
    config = validate_config(_merge(document.get("defaults", {}), cases[name]))
    return dict(name=name, config=config, provenance=dict(schema_version=1,
                root_file=str(Path(path).resolve()), source_sha256=sources))


@dataclass
class Case:
    name: str
    geometry: object
    reference: PerfectGasReference
    alpha: float
    shock_m: float | None = None
    inlet_regime: str = "subsonic"
    outlet_regime: str = "subsonic"
    config: dict | None = None
    config_provenance: dict | None = None
    fluid: object = None


def build_case(name, fluid, *, config_path=DEFAULT_CONFIG):
    resolved = load_case_config(config_path, name)
    config = resolved["config"]
    geometry_data = config["geometry"]
    kind = geometry_data["kind"]
    if kind == "radial":
        geometry = radial_channel(geometry_data["r_in"], geometry_data["r_out"], geometry_data["height"])
    elif kind == "conical":
        geometry = conical_channel(geometry_data["length"], geometry_data["r_in"], geometry_data["height"],
            np.deg2rad(geometry_data["inclination_degrees"]), np.deg2rad(geometry_data["semi_divergence_degrees"]))
    else:
        geometry = circular_arc_channel(tuple(geometry_data["center"]), geometry_data["radius"],
            np.deg2rad(geometry_data["theta_start_degrees"]), np.deg2rad(geometry_data["theta_end_degrees"]),
            geometry_data["height"])
    inlet = config["inlet"]
    alpha = (np.deg2rad(inlet["alpha_degrees"]) if "alpha_degrees" in inlet else
             np.arctan(inlet.get("alpha_tangent", 0.0)))
    mach = inlet["M_m"]
    if mach == "sonic_exit":
        if alpha != 0 or config["boundaries"]["inlet"] != "subsonic":
            raise ValueError("sonic_exit reference selection requires zero swirl and a subsonic inlet")
        ratio = float(geometry.evaluate(0.0).A / geometry.evaluate(geometry.length).A)
        gamma = float(fluid.constants.gamma)
        def area_mach(value):
            return (2 / (gamma + 1) * (1 + (gamma - 1) * value**2 / 2))**((gamma + 1) / (2 * (gamma - 1))) / value
        mach = brentq(lambda value: area_mach(value) - ratio, 1e-9, 1.0)
    shock_fraction = config["reference"]["shock_fraction"]
    shock_m = None if shock_fraction is None else shock_fraction * geometry.length
    reference = PerfectGasReference.from_inlet(fluid, geometry, p0=inlet["p0"], T0=inlet["T0"], M_m=mach, alpha=alpha)
    return Case(name, geometry, reference, alpha, shock_m, config["boundaries"]["inlet"],
                config["boundaries"]["outlet"], config, resolved["provenance"], fluid)
