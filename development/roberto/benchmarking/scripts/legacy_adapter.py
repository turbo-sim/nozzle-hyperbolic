"""Isolated adapters for the frozen, pre-integration production kernels.

Import this module before importing ``nozzle_hyperbolic`` in a legacy worker.
Physical case payloads must be materialized by the current case catalog in a
different process. No source file in ``legacy/pre_generic_fvm`` is modified.

The nozzle adapter retains the native ghost maps, mesh convention, local-JVP
reuse and state-dependent update norm. The annular adapter retains its native
coupled equations and scaling; requested Newton controls are supplied at the
factory boundary. A caller owns the common pseudo-time controller and audits.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "legacy/pre_generic_fvm"
ARCHIVE = SNAPSHOT / "src/nozzle_hyperbolic"
_verified_manifest: str | None = None


class UnsupportedLegacyCase(ValueError):
    """The frozen implementation cannot represent the requested case/choice."""


def verify_archive() -> dict[str, Any]:
    """Validate the frozen snapshot without rewriting measurement provenance."""
    global _verified_manifest
    manifest_path = SNAPSHOT / "source_manifest.json"
    raw = manifest_path.read_bytes()
    digest = sha256(raw).hexdigest()
    entries = json.loads(raw)
    if digest != _verified_manifest:
        bad = [
            name
            for name, expected in entries.items()
            if not (SNAPSHOT / name).is_file()
            or sha256((SNAPSHOT / name).read_bytes()).hexdigest() != expected
        ]
        if bad:
            raise RuntimeError(f"Frozen production manifest mismatch: {bad}")
        _verified_manifest = digest
    return {
        "snapshot": SNAPSHOT.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest,
        "manifest_files": len(entries),
        "verified": True,
    }


def load_legacy_package():
    """Select the frozen package in a fresh, dedicated worker process.

    Refusing a mixed namespace prevents silent current/legacy module reuse.
    Calling this again after successful selection is harmless.
    """
    verify_archive()
    package = sys.modules.get("nozzle_hyperbolic")
    loaded = [
        name
        for name in sys.modules
        if name == "nozzle_hyperbolic" or name.startswith("nozzle_hyperbolic.")
    ]
    if loaded:
        origin = getattr(package, "__file__", None)
        if origin is not None and Path(origin).resolve().is_relative_to(ARCHIVE):
            mismatches = [
                name
                for name in loaded
                if (path := getattr(sys.modules[name], "__file__", None))
                and not Path(path).resolve().is_relative_to(ARCHIVE)
            ]
            if not mismatches:
                return package
        raise RuntimeError(
            "A legacy worker must not import the current nozzle_hyperbolic "
            "package before selecting the snapshot. Send a plain case payload "
            f"from the parent process instead. Loaded modules: {loaded[:8]}"
        )
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "nozzle_hyperbolic",
        ARCHIVE / "__init__.py",
        submodule_search_locations=[str(ARCHIVE)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the preserved package at {ARCHIVE}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["nozzle_hyperbolic"] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        for name in list(sys.modules):
            if name == "nozzle_hyperbolic" or name.startswith("nozzle_hyperbolic."):
                del sys.modules[name]
        raise
    return package


@dataclass
class LegacyProblem:
    """Common numerical surface of one bound native production problem."""

    model_name: str
    initial_state: np.ndarray
    step: Callable
    spatial_residual: Callable
    boundary_residual: Callable
    admissible: Callable
    wave_dt: Callable
    fields: Callable
    coordinates: np.ndarray
    points: np.ndarray
    faces: np.ndarray
    volumes: np.ndarray
    conservative_scales: np.ndarray
    global_flux_scales: np.ndarray
    variable_scales: np.ndarray
    reference_time: float
    metadata: dict[str, Any]
    residual: Callable | None = None
    direction: Callable | None = None
    native_solver: Any = None

    @property
    def nvar(self):
        return self.initial_state.shape[1]

    @property
    def shape(self):
        return self.initial_state.shape

    @property
    def centers(self):
        return self.coordinates

    @property
    def initial(self):
        return self.initial_state

    @property
    def mesh(self):
        return SimpleNamespace(
            centers=self.coordinates,
            points=self.points,
            faces=self.faces,
            volumes=self.volumes,
        )


@contextmanager
def _capture_newton(module, **overrides):
    """Capture public factory inputs, with explicit benchmark-only controls."""
    original = module.make_newton_solver
    captured = {}

    def capture(*args, **kwargs):
        effective = dict(kwargs, **overrides)
        result = original(*args, **effective)
        captured.update(
            solve=result,
            residual=args[0],
            direction=args[1],
            scale=args[2],
            options=effective,
        )
        return result

    module.make_newton_solver = capture
    try:
        yield captured
    finally:
        module.make_newton_solver = original


def _number(data, *names):
    for name in names:
        if name in data:
            return float(data[name])
    raise ValueError(f"Missing physical datum; expected one of {names}")


def _fixed_scales(payload, defaults):
    result = {}
    for name, default in defaults.items():
        value = payload.get(name, default)
        result[name] = (
            float(value) if name == "reference_time" else np.asarray(value, float)
        )
        if not np.all(np.isfinite(result[name])) or np.any(
            np.asarray(result[name]) <= 0
        ):
            raise ValueError(f"Invalid fixed benchmark scale: {name}")
    return result


def _make_fluid(payload):
    import jaxprop as jxp

    data = payload.get("fluid", {})
    if data.get("backend", data.get("model", "perfect_gas")) != "perfect_gas":
        raise UnsupportedLegacyCase(
            "The maintained native-comparison adapter currently materializes "
            "the perfect-gas reference cases only."
        )
    return jxp.FluidPerfectGas(data.get("name", "Air"))


def bind_legacy(
    payload: dict[str, Any],
    *,
    cells: int | None = None,
    reconstruction: str = "constant",
    newton_cap: int = 10,
    newton_backtracking: bool = False,
    update_tolerance: float = 1e-10,
    residual_tolerance: float = 1e-12,
    delta_factor: float = 0.1,
    jacobian_mode: str = "colored",
    linear_solver: str = "banded",
    flux: str = "characteristic",
    max_backtracks: int = 20,
) -> LegacyProblem:
    """Bind a resolved catalog case without advancing any numerical state."""
    load_legacy_package()
    import jax

    if not jax.config.jax_enable_x64:
        raise ValueError("Enable jax_enable_x64 before binding benchmark operators")
    supplied = np.asarray(payload["initial_state"], dtype=float)
    if supplied.ndim != 2:
        raise ValueError(
            "Case payload initial_state must be a full two-dimensional array"
        )
    inferred = len(supplied) - 2
    if cells is not None and int(cells) != inferred:
        raise ValueError("Requested cell count differs from the resolved case payload")
    if inferred < 2 or isinstance(newton_cap, bool) or int(newton_cap) < 1:
        raise ValueError("Use at least two physical cells and a positive Newton cap")
    if reconstruction not in ("constant", "muscl_van_leer", "muscl_van_leer_smooth"):
        raise UnsupportedLegacyCase(
            f"Unsupported native reconstruction: {reconstruction}"
        )
    model_name = payload.get("model", payload.get("model_name"))
    options = dict(
        reconstruction=reconstruction,
        newton_cap=int(newton_cap),
        newton_backtracking=bool(newton_backtracking),
        update_tolerance=float(update_tolerance),
        residual_tolerance=float(residual_tolerance),
        delta_factor=float(delta_factor),
        jacobian_mode=jacobian_mode,
        linear_solver=linear_solver,
        flux=flux,
        max_backtracks=int(max_backtracks),
    )
    if model_name == "euler_nozzle":
        return _bind_nozzle(payload, supplied, options)
    if model_name == "euler_annular":
        return _bind_annular(payload, supplied, options)
    raise UnsupportedLegacyCase(f"No native adapter for physical model {model_name!r}")


def _bind_nozzle(payload, initial, options):
    import jax
    import jax.numpy as jnp
    import jaxprop as jxp
    from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
    from nozzle_hyperbolic.models.euler_nozzle import fvm as native

    if initial.shape[1] != 3:
        raise ValueError("A native nozzle case needs three primitive variables")
    if options["flux"] != "characteristic":
        raise UnsupportedLegacyCase(
            "The native nozzle factory exposes characteristic FVS only"
        )
    if options["newton_backtracking"]:
        raise UnsupportedLegacyCase(
            "The native nozzle comparison retains its original undamped Newton specialization"
        )
    walls = payload.get("walls", {})
    if any(
        np.any(np.asarray(walls.get(key, 0.0)) != 0)
        for key in ("shear_stress", "heat_flux")
    ):
        raise UnsupportedLegacyCase(
            "Native nozzle prescribed wall profiles cannot be equated to its original Haaland switch"
        )
    data = payload["geometry"]
    kind = data.get("kind", "symmetric_nozzle")
    if kind not in ("symmetric_nozzle", "symmetric", "parabolic", "nozzle"):
        raise UnsupportedLegacyCase(
            f"Native nozzle geometry is not implemented: {kind}"
        )
    length = _number(data, "length", "L")
    inlet_area = _number(data, "inlet_area", "A_inlet")
    throat_area = _number(data, "throat_area", "A_throat")
    cells = len(initial) - 2
    points = np.linspace(0.0, length, cells + 2)
    dx = float(points[1] - points[0])
    geometry = get_symmetric_nozzle(
        points,
        L=length,
        A_inlet=inlet_area,
        A_throat=throat_area,
        roughness=float(data.get("roughness", 0.0)),
    )
    face_area = 0.5 * (geometry.A[:-1] + geometry.A[1:])
    gradient = np.asarray(geometry.dAdx).copy()
    # This is the original native public driver's face-consistent area gradient.
    gradient[1:-1] = np.diff(face_area) / dx
    reservoir = payload.get("reservoir", {})
    p0, T0 = _number(reservoir, "p0"), _number(reservoir, "T0")
    fluid = _make_fluid(payload)
    stagnation = fluid.get_state(jxp.PT_INPUTS, p0, T0)
    h0, s0, rho0, a0 = (float(stagnation[key]) for key in ("h", "s", "rho", "a"))
    back_pressure = float(payload["back_pressure"])
    with _capture_newton(native) as captured:
        factory = native.make_jax_stadtke_step(
            fluid,
            geometry.A,
            face_area,
            gradient,
            geometry.D_h,
            geometry.roughness,
            dx,
            1.0,
            back_pressure,
            h0,
            s0,
            delta_factor=options["delta_factor"],
            newton_iter=options["newton_cap"],
            wall_friction=False,
            recon=options["reconstruction"],
            jacobian_mode=options["jacobian_mode"],
            linear_solver=options["linear_solver"],
            newton_tol=options["update_tolerance"],
            flux_local_jac=True,
            cell_local_jac=True,
            eps_filter=0.0,
            return_operators=True,
        )
    _, compute_dt, _, _, operators = factory
    solve = captured["solve"]
    complete = operators["state_from_interior"]
    initial = np.asarray(complete(jnp.asarray(initial[1:-1]).ravel()))
    defaults = dict(
        variable_scales=np.array([p0, a0, a0**2]),
        conservative_scales=rho0 * np.array([1.0, a0, a0**2]),
        global_flux_scales=inlet_area * a0 * rho0 * np.array([1.0, a0, a0**2]),
        reference_time=length / a0,
    )
    scales = _fixed_scales(payload, defaults)
    variable_scale = jnp.asarray(scales["variable_scales"])

    @jax.jit
    def step(full, dt):
        physical, info = solve(full[1:-1].ravel(), full.ravel(), dt)
        return complete(physical), info

    @jax.jit
    def spatial(full):
        return operators["physical_residual"](full, full, jnp.float64(1.0)).reshape(
            cells, 3
        )

    @jax.jit
    def boundary(full):
        mapped = complete(full[1:-1].ravel())
        return (full[jnp.array([0, -1])] - mapped[jnp.array([0, -1])]) / variable_scale

    @jax.jit
    def admissible(full):
        values = fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(full[:, 0] > 0)
            & jnp.all(jnp.isfinite(values["rho"]))
            & jnp.all(jnp.isfinite(values["a"]))
            & jnp.all(jnp.isfinite(values["T"]))
            & jnp.all(values["rho"] > 0)
            & jnp.all(values["a"] > 0)
            & jnp.all(values["T"] > 0)
        )

    @jax.jit
    def fields(full):
        physical = full[1:-1]
        p, u, h = (physical[:, i] for i in range(3))
        state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        return dict(
            p=p,
            u=u,
            v_m=u,
            h=h,
            rho=state["rho"],
            a=state["a"],
            M=u / state["a"],
            T=state["T"],
            s=state["s"],
            h0=h + 0.5 * u**2,
            mdot=state["rho"] * u * jnp.asarray(geometry.A[1:-1]),
        )

    metadata = dict(
        backend="legacy",
        model="euler_nozzle",
        package_file=str(ARCHIVE / "__init__.py"),
        implementation_file=str(ARCHIVE / "models/euler_nozzle/fvm.py"),
        **verify_archive(),
        options=options,
        unknowns=cells * 3,
        stencil_radius=operators["solver_metadata"]["stencil_radius"],
        local_jacobians=True,
        boundary_treatment="original approximate ghost maps eliminated on each residual",
        boundary_physics="original inlet extrapolation/reservoir map and Mach-blended outlet",
        mesh_convention="Uniform native nodes including ghost slots; physical volumes A_i*dx; averaged face areas",
        initial_state_policy="Identical supplied physical-cell values by index; native ghost maps completed",
        newton_policy="Original full-step specialization with state-dependent update scales; no inner residual criterion",
        native_residual_tolerance=0.0,
        native_backtracking=False,
        source_gradient="Original public driver's face-consistent area gradient",
        comparison_limits=[
            "Native physical-cell coordinates and measures differ from current cell-centered geometry",
            "Native boundary maps differ from current acoustic/finite-wave algebraic laws",
            "Native inner update scaling and residual stopping differ from current Newton binding",
            "Native candidate admissibility is less extensive than current reconstructed-state checks",
        ],
    )
    return LegacyProblem(
        model_name="euler_nozzle",
        initial_state=initial,
        step=step,
        spatial_residual=spatial,
        boundary_residual=boundary,
        admissible=admissible,
        wave_dt=jax.jit(lambda full: compute_dt(full)[0]),
        fields=fields,
        coordinates=points[1:-1],
        points=points,
        faces=0.5 * (points[:-1] + points[1:]),
        volumes=np.asarray(operators["cell_volumes"])[1:-1],
        metadata=metadata,
        residual=jax.jit(captured["residual"]),
        direction=jax.jit(captured["direction"]),
        native_solver=operators,
        **scales,
    )


def _annular_geometry(data):
    from nozzle_hyperbolic.geometry.annular import (
        circular_arc_channel,
        conical_channel,
        radial_channel,
    )

    kind = data["kind"]
    if kind == "radial":
        return radial_channel(data["r_in"], data["r_out"], data["height"])
    if kind == "conical":
        return conical_channel(
            data["length"],
            data["r_in"],
            data["height"],
            np.deg2rad(data["inclination_degrees"]),
            np.deg2rad(data["semi_divergence_degrees"]),
        )
    if kind == "circular_arc":
        return circular_arc_channel(
            tuple(data["center"]),
            data["radius"],
            np.deg2rad(data["theta_start_degrees"]),
            np.deg2rad(data["theta_end_degrees"]),
            data["height"],
        )
    raise UnsupportedLegacyCase(f"Native annular geometry is not implemented: {kind}")


def _bind_annular(payload, initial, options):
    import jax
    import jax.numpy as jnp
    from nozzle_hyperbolic.models.euler_annular import fvm as native

    if initial.shape[1] != 4:
        raise ValueError("A native annular case needs four primitive variables")
    if options["reconstruction"] == "muscl_van_leer_smooth":
        raise UnsupportedLegacyCase(
            "The native annular factory has no smooth MUSCL selector"
        )
    if options["flux"] not in ("characteristic", "rusanov"):
        raise UnsupportedLegacyCase(
            f"Native annular flux is not implemented: {options['flux']}"
        )
    for name in ("inlet_regime", "outlet_regime"):
        if payload.get(name, "subsonic") not in ("subsonic", "supersonic"):
            raise UnsupportedLegacyCase(
                "Native annular boundaries require static inlet/outlet regimes"
            )
    geometry = _annular_geometry(payload["geometry"])
    cells = len(initial) - 2
    faces = payload.get("faces", payload.get("native_faces"))
    mesh = (
        geometry.mesh(cells)
        if faces is None
        else geometry.mesh(m_faces=np.asarray(faces, float))
    )
    if mesh.n_cells != cells:
        raise ValueError("Resolved annular face count differs from the initial state")
    fluid = _make_fluid(payload)
    walls = payload.get("walls", {})
    with _capture_newton(
        native, backtracking=options["newton_backtracking"]
    ) as captured:
        solver = native.make_annular_solver(
            fluid,
            mesh,
            inlet_data=payload["inlet_data"],
            outlet_pressure=payload.get(
                "back_pressure", payload.get("outlet_pressure")
            ),
            inlet_regime=payload.get("inlet_regime", "subsonic"),
            outlet_regime=payload.get("outlet_regime", "subsonic"),
            outlet_formulation=payload.get("outlet_formulation", "acoustic"),
            tau_w=walls.get("shear_stress", 0.0),
            q_w=walls.get("heat_flux", 0.0),
            reconstruction=options["reconstruction"],
            flux_method=options["flux"],
            delta_factor=options["delta_factor"],
            jacobian_mode=options["jacobian_mode"],
            linear_solver=options["linear_solver"],
            newton_iterations=options["newton_cap"],
            newton_tolerance=options["update_tolerance"],
            newton_residual_tolerance=options["residual_tolerance"],
            max_backtracks=options["max_backtracks"],
        )
    sound = solver.reference_sound_speed
    defaults = dict(
        variable_scales=np.array([solver.scales[k] for k in ("p", "a", "ell", "h")]),
        conservative_scales=solver.conservative_scales,
        global_flux_scales=solver.global_flux_scales,
        reference_time=geometry.length / sound,
    )
    scales = _fixed_scales(payload, defaults)

    @jax.jit
    def fields(full):
        result = solver.fields(full)
        return dict(
            **result,
            u=result["v_m"],
            mdot=result["rho"] * result["v_m"] * jnp.asarray(mesh.A_center),
            alpha_deg=jnp.rad2deg(jnp.arctan2(result["v_theta"], result["v_m"])),
        )

    metadata = dict(
        backend="legacy",
        model="euler_annular",
        package_file=str(ARCHIVE / "__init__.py"),
        implementation_file=str(ARCHIVE / "models/euler_annular/fvm.py"),
        **verify_archive(),
        options=options,
        unknowns=(cells + 2) * 4,
        stencil_radius=2,
        local_jacobians=False,
        boundary_treatment="native simultaneous zero-storage trace equations",
        boundary_physics=payload.get("outlet_formulation", "acoustic"),
        mesh_convention="Native annular physical cells and physical-face traces; resolved face coordinates",
        initial_state_policy="Identical supplied full cold state; no analytical initialization",
        newton_policy="Native residual/direction/admissibility; matched constructor tolerances and requested damping",
        original_native_backtracking=True,
        effective_backtracking=options["newton_backtracking"],
        constructor_override="make_newton_solver backtracking keyword only; restored immediately after binding",
        comparison_limits=[
            "Native physical backward-Euler rows are divided by content_scale only",
            "Current operator also normalizes physical rows by max(1,dt/reference_time)",
            "Same numeric residual tolerances can therefore select different inner convergence trajectories",
            "Native outer controller is not used; the benchmark supplies its common strict controller",
        ],
    )
    return LegacyProblem(
        model_name="euler_annular",
        initial_state=np.array(initial, copy=True),
        step=solver.step,
        spatial_residual=solver.spatial_residual,
        boundary_residual=solver.boundary_residual,
        admissible=solver.is_admissible,
        wave_dt=jax.jit(lambda full: solver.compute_dt(full, 1.0)),
        fields=fields,
        coordinates=np.asarray(mesh.m_centers),
        points=np.r_[mesh.m_faces[0], mesh.m_centers, mesh.m_faces[-1]],
        faces=np.asarray(mesh.m_faces),
        volumes=np.asarray(mesh.volumes),
        metadata=metadata,
        residual=solver.residual,
        direction=jax.jit(captured["direction"]),
        native_solver=solver,
        **scales,
    )
