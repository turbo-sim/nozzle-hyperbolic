"""Canonical annular meshes and independent perfect-gas diffuser references.

The cache stores the production ``AnnularMesh`` arrays without resampling or
approximating them.  The analytical profile is a verification reference; the
deliberately approximate ``cold_initial`` is the numerical initial guess.
"""

from dataclasses import fields
from hashlib import sha256
from importlib import metadata
import inspect
import json
from pathlib import Path
from uuid import uuid4

import jaxprop as jp
import numpy as np
import nurbspy.numpy as nrb
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq

from nozzle_hyperbolic.geometry.annular import (
    AnnularMesh, constant_height, make_annular_geometry,
)


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
CACHE = STUDY / "outputs" / "diffuser" / "geometry_cache"
_ARRAY_NAMES = tuple(item.name for item in fields(AnnularMesh)
                     if item.name != "geometry")


def build_geometry(c):
    """Fixed-tip expansion with smooth entrance/exit slopes and an axial tail.

    Both slope blends lie inside the requested expansion length. The inlet and
    final tail are axial, so their exact normal-area ratio is the requested
    annular area ratio. The normal-height NURBS approximates the fixed-tip
    constraint to roundoff-scale error, checked separately on dense samples.
    """
    g = c["geometry"]
    R = float(g["tip_radius_m"])
    L = float(g["constant_length_m"])
    D = float(g["expansion_length_m"])
    E = float(g.get("extension_length_m", 0.1))
    w = float(g["junction_blend_m"])
    v = float(g.get("exit_blend_m", w))
    hub_in = float(g["inlet_hub_tip"])
    ratio = float(g["outlet_area_ratio"])
    square = 1 - ratio * (1 - hub_in**2)
    if not (R > 0 and L > 0 and D > 0 and E >= 0 and w > 0 and v > 0
            and w + v < D and 0 < hub_in < 1 and ratio > 1 and square > 0):
        raise ValueError("Require positive dimensions, two interior slope blends, "
                         "and a realizable annular outlet area ratio")
    hub_out = np.sqrt(square)
    rin, rout = R * (1 + hub_in) / 2, R * (1 + hub_out) / 2
    slope = (rout - rin) / (D - (w + v) / 2)
    xs = [0.0, L, L + w, L + D - v, L + D]
    rs = [rin, rin, rin + slope * w / 2, rout - slope * v / 2, rout]
    ds = [0.0, 0.0, slope, slope, 0.0]
    if E > 0:
        xs.append(L + D + E)
        rs.append(rout)
        ds.append(0.0)
    xs, rs, ds = map(np.asarray, (xs, rs, ds))
    controls = [np.array([xs[0], rs[0]])]
    for index, dx in enumerate(np.diff(xs)):
        controls.extend([
            np.array([xs[index] + dx / 3, rs[index] + ds[index] * dx / 3]),
            np.array([xs[index + 1] - dx / 3,
                      rs[index + 1] - ds[index + 1] * dx / 3]),
            np.array([xs[index + 1], rs[index + 1]]),
        ])
    knots = np.r_[np.repeat(xs[0], 4), np.repeat(xs[1:-1], 3),
                  np.repeat(xs[-1], 4)] / xs[-1]
    mid = nrb.NurbsCurve(np.array(controls).T, np.ones(len(controls)), 3, knots)
    provisional = make_annular_geometry(mid, constant_height(2 * (R - rin)))
    length = provisional.length

    def samples(lower, upper):
        xi = np.linspace(lower, upper, 17) / xs[-1]
        distance = np.array([provisional.xi_of_m.arclength_at(float(t)) for t in xi])
        position = mid.get_value(xi)
        derivative = mid.get_derivative(xi, order=1)
        x_prime = derivative[0] / np.linalg.norm(derivative, axis=0)
        height = 2 * (R - position[1]) / x_prime
        return distance / length, height

    def linear_segment(lower, upper, first, last):
        return np.r_[np.repeat(lower, 6), np.repeat(upper, 6)], np.linspace(first, last, 6)

    def height_second(gap, local_slope, slope_prime):
        square = 1 + local_slope**2
        return length**2 * (
            -2 * slope_prime - 2 * slope_prime * local_slope**2 / square
            + 2 * gap * slope_prime**2 * (1 - local_slope**2) / square**2
        ) / np.sqrt(square)

    first_sigma, first_height = samples(L, L + w)
    last_sigma, last_height = samples(L + D - v, L + D)
    # db/dm = -2*s + 2*(R-r)*s*(ds/dx)/(1+s*s).
    # Retain derivative jumps where the midline curvature changes; smoothing
    # those joins would move the physical tip away from its fixed radius.
    first_end = length * (-2 * slope + 2 * (R - rs[2]) * slope**2
                          / (w * (1 + slope**2)))
    last_start = length * (-2 * slope - 2 * (R - rs[3]) * slope**2
                           / (v * (1 + slope**2)))
    first_blend = make_interp_spline(
        first_sigma, first_height, k=5,
        bc_type=([(1, 0.0), (2, height_second(R - rin, 0.0, slope / w))],
                 [(1, first_end), (2, height_second(R - rs[2], slope, slope / w))]))
    last_blend = make_interp_spline(
        last_sigma, last_height, k=5,
        bc_type=([(1, last_start), (2, height_second(R - rs[3], slope, -slope / v))],
                 [(1, 0.0), (2, height_second(R - rout, 0.0, -slope / v))]))
    segments = [
        linear_segment(0.0, first_sigma[0], first_height[0], first_height[0]),
        (first_blend.t, first_blend.c),
        linear_segment(first_sigma[-1], last_sigma[0], first_height[-1], last_height[0]),
        (last_blend.t, last_blend.c),
    ]
    if E > 0:
        segments.append(linear_segment(last_sigma[-1], 1.0,
                                       last_height[-1], last_height[-1]))
    height_knots, height_controls = map(np.copy, segments[0])
    for segment_knots, segment_controls in segments[1:]:
        height_knots = np.r_[height_knots[:-1], segment_knots[6:]]
        height_controls = np.r_[height_controls, segment_controls[1:]]
    height = nrb.NurbsCurve(height_controls[None, :], np.ones(len(height_controls)),
                            5, height_knots)
    return make_annular_geometry(mid, height)


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _cache_identity(config):
    """Invalidate on geometry definition, production evaluator or dependencies."""
    source = ROOT / "src/nozzle_hyperbolic/geometry/annular.py"
    identity = dict(
        format_version=1,
        geometry=config["geometry"],
        annular_source_sha256=sha256(source.read_bytes()).hexdigest(),
        study_builder_sha256=sha256(
            inspect.getsource(build_geometry).encode("utf-8")).hexdigest(),
        nurbspy_version=_version("nurbspy"),
        numpy_version=_version("numpy"),
        scipy_version=_version("scipy"),
        array_fields=list(_ARRAY_NAMES),
    )
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest(), identity


def _write_npz(path, arrays):
    """Use a unique temporary file so interrupted writes cannot poison caches."""
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_npz(path, names):
    with np.load(path, allow_pickle=False) as retained:
        if set(retained.files) != set(names):
            raise ValueError(f"Unexpected array fields in geometry cache {path}")
        return {name: np.array(retained[name], copy=True) for name in names}


def native_meshes(config, counts=(50, 100, 200)):
    """Return canonical geometry, meshes by cell count, and 601 wall samples.

    Every cache miss calls the existing production geometry's ``mesh`` method.
    Hits reconstruct its public immutable ``AnnularMesh`` from the exact saved
    arrays.  The geometry is still built from the study's existing midline and
    normal-height definition on every call.  All generated files are confined
    to ``outputs/diffuser/geometry_cache`` within this study.
    """
    counts = tuple(counts)
    if not counts or any(isinstance(n, (bool, np.bool_))
                         or not isinstance(n, (int, np.integer)) or n < 2
                         for n in counts):
        raise ValueError("counts must contain positive integer cell counts >= 2")
    counts = tuple(dict.fromkeys(map(int, counts)))
    key, identity = _cache_identity(config)
    cache = CACHE / key
    cache.mkdir(parents=True, exist_ok=True)
    geom = build_geometry(config)
    meshes = {}
    for n in counts:
        path = cache / f"mesh_{n}.npz"
        if path.exists():
            meshes[n] = AnnularMesh(geom, **_read_npz(path, _ARRAY_NAMES))
            if meshes[n].n_cells != n:
                raise ValueError(f"Wrong cell count in geometry cache {path}")
            print(f"Native annular mesh {n}: exact cache hit", flush=True)
        else:
            print(f"Native annular mesh {n}: production preprocessing", flush=True)
            meshes[n] = geom.mesh(n)
            _write_npz(path, {name: getattr(meshes[n], name)
                              for name in _ARRAY_NAMES})
    dense_names = ("m", "x", "r", "b", "A", "tip_x", "hub_x", "tip_r", "hub_r")
    dense_path = cache / "dense_601.npz"
    if dense_path.exists():
        dense = _read_npz(dense_path, dense_names)
    else:
        distances = np.linspace(0.0, geom.length, 601)
        metric = geom.evaluate(distances)
        dense = dict(
            m=distances,
            x=metric.x,
            r=metric.r,
            b=metric.b,
            A=metric.A,
            tip_x=metric.x - metric.b * metric.r_prime / 2,
            hub_x=metric.x + metric.b * metric.r_prime / 2,
            tip_r=metric.r + metric.b * metric.x_prime / 2,
            hub_r=metric.r - metric.b * metric.x_prime / 2,
        )
        _write_npz(dense_path, dense)
    manifest = dict(identity, cache_key=key,
                    production_class="nozzle_hyperbolic.geometry.annular.AnnularMesh",
                    dense_samples=601,
                    available_cell_counts=sorted(int(p.stem[5:])
                                                for p in cache.glob("mesh_*.npz")))
    (cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return geom, meshes, dense


def _nominal(config, native, mach=None):
    """Perfect-gas nominal inlet and invariant data in SI units."""
    fluid = jp.FluidPerfectGas(config["fluid"])
    gamma, gas_constant = float(fluid.constants.gamma), float(fluid.constants.R)
    cp = gamma * gas_constant / (gamma - 1)
    inlet = config["inlet"]
    mach = float(inlet["mach"] if mach is None else mach)
    angle = np.deg2rad(float(inlet["angle_deg"]))
    t0 = float(inlet["stagnation_temperature_K"])
    temperature = t0 / (1 + 0.5 * (gamma - 1) * mach**2)
    pressure = float(inlet["stagnation_pressure_Pa"]) * (
        temperature / t0) ** (gamma / (gamma - 1))
    sound = np.sqrt(gamma * gas_constant * temperature)
    velocity = mach * sound * np.cos(angle)
    ell = float(native.r_face[0]) * mach * sound * np.sin(angle)
    if not (0 < velocity / sound < 1):
        raise ValueError("The diffuser reference requires a positive subsonic "
                         "nominal inlet meridional Mach number")
    rho = pressure / (gas_constant * temperature)
    return dict(gamma=gamma, R=gas_constant, cp=cp, T=temperature, p=pressure,
                v_m=velocity, ell=ell, h0=cp * t0,
                mach=mach,
                mass_flow_kg_s=rho * velocity * float(native.A_face[0]))


def _subsonic_velocity(reference, radius, area):
    """Solve the smooth, isentropic *meridionally subsonic* steady branch."""
    gamma, cp = reference["gamma"], reference["cp"]
    available = reference["h0"] - 0.5 * (reference["ell"] / radius)**2
    if available <= 0 or area <= 0:
        raise ValueError("Nonpositive available enthalpy or area in diffuser reference")
    sonic = np.sqrt(2 * (gamma - 1) * available / (gamma + 1))

    def mass_residual(velocity):
        temperature = (available - 0.5 * velocity**2) / cp
        density = (reference["p"] / (reference["R"] * reference["T"])) * (
            temperature / reference["T"]) ** (1 / (gamma - 1))
        return density * velocity * area - reference["mass_flow_kg_s"]

    if mass_residual(sonic) <= 0:
        raise ValueError("Nominal mass flow reaches or exceeds the local "
                         "meridional choking limit in the analytical reference")
    return brentq(mass_residual, 0.0, sonic, xtol=1e-10,
                  rtol=8 * np.finfo(float).eps)


def _points(native):
    return (np.r_[native.m_faces[0], native.m_centers, native.m_faces[-1]],
            np.r_[native.r_face[0], native.r_center, native.r_face[-1]],
            np.r_[native.A_face[0], native.A_center, native.A_face[-1]])


def _exit_state(reference, native):
    radius, area = float(native.r_face[-1]), float(native.A_face[-1])
    velocity = _subsonic_velocity(reference, radius, area)
    enthalpy = reference["h0"] - 0.5 * (velocity**2 + (reference["ell"] / radius)**2)
    temperature = enthalpy / reference["cp"]
    pressure = reference["p"] * (temperature / reference["T"]) ** (
        reference["gamma"] / (reference["gamma"] - 1))
    return np.array([pressure, velocity, reference["ell"], enthalpy])


def design_backpressure(config, native, nominalMach):
    """Return steady outlet pressure [Pa] for an explicitly selected inlet Mach.

    This design calculation does not impose that Mach in the numerical boundary
    equations. ``analytical_profile`` instead solves its actual reference Mach
    from the configured fixed backpressure.
    """
    return float(_exit_state(_nominal(config, native, nominalMach), native)[0])


def _steady_reference(config, native):
    """Select the positive meridionally subsonic branch at fixed backpressure."""
    outlet = config.get("outlet", {})
    pressure = outlet.get("pressure_Pa")
    if pressure is None:
        return _nominal(config, native)
    pressure = float(pressure)
    reservoir_pressure = float(config["inlet"]["stagnation_pressure_Pa"])
    cosine = np.cos(np.deg2rad(float(config["inlet"]["angle_deg"])))
    if not (0 < pressure < reservoir_pressure and cosine > 0):
        raise ValueError("Fixed backpressure must be between zero and reservoir "
                         "pressure, with a positive inlet meridional direction")
    lower, upper = 1e-6, (1 - 1e-7) / cosine

    def residual(mach):
        return design_backpressure(config, native, mach) - pressure

    f_lower, f_upper = residual(lower), residual(upper)
    if f_lower * f_upper > 0:
        raise ValueError("Fixed backpressure does not bracket a positive "
                         "meridionally subsonic steady reference: "
                         f"p_out range [{f_upper + pressure}, {f_lower + pressure}] Pa")
    mach = brentq(residual, lower, upper, xtol=2e-12,
                  rtol=8 * np.finfo(float).eps)
    return _nominal(config, native, mach)


def cold_initial(config, native):
    """Return a physical, deliberately inexact ``(N+2, 4)`` primitive guess.

    The reference inlet Mach is recovered from fixed outlet pressure; its value
    is not imposed by a boundary condition. Only endpoint steady states enter
    the initial guess, rather than the independently verified exact profile.
    Pressure and meridional velocity are linearly interpolated between these
    endpoints.  Angular momentum and stagnation enthalpy are uniform; entropy
    and mass flux are deliberately not forced to their exact steady profiles.
    """
    ref = _steady_reference(config, native)
    distance, radius, area = _points(native)
    exit_pressure, exit_velocity = _exit_state(ref, native)[:2]
    fraction = (distance - distance[0]) / (distance[-1] - distance[0])
    velocity = ref["v_m"] + fraction * (exit_velocity - ref["v_m"])
    pressure = ref["p"] + fraction * (exit_pressure - ref["p"])
    enthalpy = ref["h0"] - 0.5 * (velocity**2 + (ref["ell"] / radius)**2)
    states = np.column_stack((pressure, velocity,
                              np.full_like(velocity, ref["ell"]), enthalpy))
    if not np.isfinite(states).all() or np.any(states[:, (0, 1, 3)] <= 0):
        raise ValueError("The cold diffuser initialization is not physically admissible")
    return states


def analytical_profile(config, native, *, inlet_mach=None):
    """Independent isentropic steady verification, never used as a cold seed.

    ``state`` includes both physical boundary traces (shape ``N+2, 4``).
    ``fields``, ``x`` and ``m`` contain the ``N`` physical cell centers and use
    the production model's field naming.  Scalar invariants and outlet pressure
    are also returned for design checks and pressure-outlet calibration. With
    a configured backpressure, the inlet Mach is a scalar root of that pressure
    condition, independent of the nominal design Mach stored in the YAML.
    ``inlet_mach`` explicitly overrides this only for a labeled design profile.
    """
    ref = (_steady_reference(config, native) if inlet_mach is None
           else _nominal(config, native, inlet_mach))
    distance, radius, area = _points(native)
    velocity = np.array([_subsonic_velocity(ref, r, a)
                         for r, a in zip(radius, area)])
    swirl = ref["ell"] / radius
    enthalpy = ref["h0"] - 0.5 * (velocity**2 + swirl**2)
    temperature = enthalpy / ref["cp"]
    pressure = ref["p"] * (temperature / ref["T"]) ** (
        ref["gamma"] / (ref["gamma"] - 1))
    density = pressure / (ref["R"] * temperature)
    sound = np.sqrt(ref["gamma"] * ref["R"] * temperature)
    ell = np.full_like(velocity, ref["ell"])
    state = np.column_stack((pressure, velocity, ell, enthalpy))
    all_fields = dict(p=pressure, v_m=velocity, v_theta=swirl, ell=ell,
                      h=enthalpy, T=temperature, rho=density, a=sound,
                      M=np.hypot(velocity, swirl) / sound,
                      M_m=velocity / sound, M_theta=swirl / sound,
                      alpha_deg=np.rad2deg(np.arctan2(swirl, velocity)),
                      mdot=density * velocity * area,
                      h0=np.full_like(velocity, ref["h0"]))
    return dict(state=state,
                fields={name: value[1:-1] for name, value in all_fields.items()},
                boundary_fields={name: value[[0, -1]]
                                 for name, value in all_fields.items()},
                x=np.array(native.x_center, copy=True), m=distance[1:-1],
                mass_flow_kg_s=ref["mass_flow_kg_s"],
                inlet_mach=ref["mach"],
                reference_mode=("fixed_backpressure" if inlet_mach is None
                                and config.get("outlet", {}).get("pressure_Pa") is not None
                                else "explicit_design_mach"),
                inlet_total_enthalpy_J_kg=ref["h0"],
                outlet_pressure_Pa=float(pressure[-1]))
