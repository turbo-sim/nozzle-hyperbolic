"""General-EOS annular finite volumes with simultaneous boundary equations.

Primitive unknowns are (p, v_m, ell, h), including two independent physical-face
boundary traces. Geometry is preprocessed once; state, flux, boundary and full
Newton residual differentiation remain JAX operations.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from ...numerics.jacobians import make_block_operators
from ...numerics.reconstruction import _van_leer_jax_safe
from ...solvers.finite_volume import flux_divergence, make_coupled_residual
from ...solvers.time_marching.implicit import make_newton_solver
from ...thermodynamics.adapters import as_jax_fluid
from .boundaries import boundary_mach, make_boundary_residual
from .flux_jax import numerical_flux
from .initialization import uniform_initial_state
from .sources import integrated_source
from .state import (admissible, primitive_fields, primitive_to_conservative,
                    state_property, thermodynamic_state, wave_speed)


@dataclass(frozen=True)
class AnnularOptions:
    reconstruction: str = "constant"
    flux_method: str = "characteristic"
    delta_factor: float = 0.1
    linear_solver: str = "banded"
    jacobian_mode: str = "colored"
    newton_iterations: int = 12
    newton_tolerance: float = 1e-10
    newton_residual_tolerance: float = 1e-10
    max_backtracks: int = 16

    def __post_init__(self):
        if self.reconstruction not in ("constant", "muscl_van_leer"):
            raise ValueError("reconstruction must be 'constant' or 'muscl_van_leer'")
        if self.flux_method not in ("characteristic", "rusanov"):
            raise ValueError("Unknown annular flux method")
        if not np.isfinite(self.delta_factor) or self.delta_factor <= 0:
            raise ValueError("delta_factor must be finite and positive")
        if self.newton_iterations < 1 or self.max_backtracks < 0:
            raise ValueError("Invalid Newton iteration/trial limits")
        if self.newton_tolerance < 0 or self.newton_residual_tolerance <= 0:
            raise ValueError("Newton residual tolerance must be positive")


@dataclass
class AnnularSolution:
    state: np.ndarray
    success: bool
    message: str
    history: list[dict]
    final_metrics: dict
    physical_time: float = 0.0
    monitoring: dict | None = None


class AnnularSolver:
    """A compiled model factory plus explicit steady/physical-time drivers.

    Boundary regimes are explicit factory choices. Their signed meridional
    Mach numbers are audited at the converged state; no boundary root solve,
    state projection, or hidden automatic change of equations is performed.
    """

    def __init__(self, fluid, mesh, inlet_data, outlet_pressure, options,
                 inlet_regime, outlet_regime, tau_w=0.0, q_w=0.0):
        self.input_fluid = fluid
        fluid = as_jax_fluid(fluid)
        self.fluid, self.mesh, self.options = fluid, mesh, options
        self.inlet_regime, self.outlet_regime = inlet_regime, outlet_regime
        self.outlet_pressure = outlet_pressure
        self.inlet_data = dict(inlet_data)
        if inlet_regime == "subsonic":
            if "h0" not in self.inlet_data or "s0" not in self.inlet_data:
                reservoir = fluid.get_state(jxp.PT_INPUTS,
                                           self.inlet_data["p0"], self.inlet_data["T0"])
                self.inlet_data["h0"] = float(state_property(reservoir, "h"))
                self.inlet_data["s0"] = float(state_property(reservoir, "s"))
            reference_p = float(self.inlet_data.get("p0", outlet_pressure))
            reference_h = float(self.inlet_data["h0"])
        else:
            prescribed = np.asarray(self.inlet_data["state"], dtype=float)
            reference_p, reference_h = float(prescribed[0]), float(prescribed[3])
        reference = fluid.get_state(jxp.HmassP_INPUTS, reference_h, reference_p)
        for name in ("rho", "a", "G", "T", "s"):
            value = np.asarray(state_property(reference, name))
            if value.size != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"The supplied fluid HP reference state lacks finite {name!r}")
        rho_ref, a_ref, temperature_ref = (
            float(state_property(reference, key)) for key in ("rho", "a", "T"))
        if not all(np.isfinite(value) and value > 0 for value in
                   (reference_p, rho_ref, a_ref, temperature_ref)):
            raise ValueError("The reference EOS state must be finite and admissible")
        self.reference_pressure, self.reference_enthalpy = reference_p, reference_h
        self.reference_density, self.reference_sound_speed = rho_ref, a_ref
        radius_ref = float(mesh.r_face[0])
        self.scales = dict(p=reference_p, a=a_ref, ell=radius_ref*a_ref,
                           h=a_ref*a_ref, s=a_ref*a_ref/temperature_ref)
        nphysical = mesh.n_cells
        self.shape = (nphysical + 2, 4)
        self.r_full = jnp.asarray(np.r_[mesh.r_face[0], mesh.r_center, mesh.r_face[-1]])
        radius = jnp.asarray(mesh.r_center)
        r_face, area_face = map(jnp.asarray, (mesh.r_face, mesh.A_face))
        volumes, radius_slope = map(jnp.asarray, (mesh.volumes, mesh.r_prime))
        points = jnp.asarray(np.r_[mesh.m_faces[0], mesh.m_centers, mesh.m_faces[-1]])
        faces, widths = map(jnp.asarray, (mesh.m_faces, mesh.widths))
        q_ref = jnp.asarray([rho_ref, rho_ref*a_ref, rho_ref*radius_ref*a_ref,
                             rho_ref*a_ref*a_ref])
        content_scale = volumes[:, None]*q_ref
        flux_scale = content_scale*a_ref/mesh.geometry.length
        global_flux_scale = float(mesh.A_face[0])*a_ref*q_ref
        # Fixed physical scales are also inputs to the opt-in equation-wise
        # monitoring adapter. They do not alter the signed Newton equations.
        self.conservative_scales = np.asarray(q_ref)
        self.spatial_flux_scales = np.asarray(flux_scale)
        self.global_flux_scales = np.asarray(global_flux_scale)
        variable_scale = jnp.asarray([self.scales[key] for key in ("p", "a", "ell", "h")])

        def reconstruct(full):
            if options.reconstruction == "constant":
                return full[:-1], full[1:]
            gradients = (full[1:]-full[:-1])/(points[1:]-points[:-1])[:, None]
            slopes = jnp.zeros_like(full).at[1:-1].set(
                _van_leer_jax_safe(gradients[:-1], gradients[1:]))
            left = full[:-1]+slopes[:-1]*(faces-points[:-1])[:, None]
            right = full[1:]+slopes[1:]*(faces-points[1:])[:, None]
            return left, right

        def face_flux(full):
            left, right = reconstruct(full)
            return numerical_flux(left, right, r_face, area_face, fluid,
                                  options.delta_factor, options.flux_method)

        self.tau_w = _wall_values(tau_w, nphysical, "tau_w", nonnegative=True)
        self.q_w = _wall_values(q_w, nphysical, "q_w")
        tau_wall, heat_wall = jnp.asarray(self.tau_w), jnp.asarray(self.q_w)
        height = jnp.asarray(mesh.b)

        def source(full):
            return integrated_source(full[1:-1], radius, radius_slope,
                                     area_face[:-1], area_face[1:], volumes, fluid,
                                     height=height, tau_w=tau_wall, q_w=heat_wall)

        def storage(physical):
            return volumes[:, None]*primitive_to_conservative(physical, radius, fluid)

        def spatial(full):
            return flux_divergence(face_flux(full))-source(full)

        boundary = make_boundary_residual(
            fluid, mesh.m_centers, mesh.m_faces, mesh.r_face, self.inlet_data,
            outlet_pressure, self.scales, inlet_regime, outlet_regime)
        assembly = make_coupled_residual(
            storage, spatial, boundary, nphysical=nphysical, nvar=4,
            content_scale=content_scale, flux_scale=flux_scale)
        block = make_block_operators(
            assembly["residual"], nphysical+2, 4, 2,
            jacobian_mode=options.jacobian_mode, linear_solver=options.linear_solver)

        def valid(flat):
            full = flat.reshape(self.shape)
            left, right = reconstruct(full)
            basic = (jnp.all(jnp.isfinite(full)) & jnp.all(full[:, 0] > 0)
                     & jnp.all(full[:, 1] >= 0)
                     & jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(right))
                     & jnp.all(left[:, 0] > 0) & jnp.all(right[:, 0] > 0)
                     & jnp.all(left[:, 1] >= 0) & jnp.all(right[:, 1] >= 0))
            return jax.lax.cond(
                basic, lambda: (jnp.all(admissible(full, self.r_full, fluid))
                                & jnp.all(admissible(left, r_face, fluid))
                                & jnp.all(admissible(right, r_face, fluid))),
                lambda: jnp.asarray(False))

        newton = make_newton_solver(
            assembly["residual"], block["direction"],
            lambda point: jnp.broadcast_to(variable_scale, self.shape).ravel(),
            max_iterations=options.newton_iterations,
            tolerance=options.newton_tolerance, residual_tolerance=options.newton_residual_tolerance,
            admissible=valid, backtracking=True, max_backtracks=options.max_backtracks)

        def step(full, dt):
            result, info = newton(full.ravel(), full.ravel(), dt)
            return result.reshape(self.shape), info

        def metrics(full):
            physical = spatial(full)
            scaled = physical/flux_scale
            bc = boundary(full)
            balance = face_flux(full)[-1]-face_flux(full)[0]-jnp.sum(source(full), axis=0)
            return dict(
                spatial_rms=jnp.sqrt(jnp.mean(scaled**2)),
                spatial_max=jnp.max(jnp.abs(scaled)),
                boundary_rms=jnp.sqrt(jnp.mean(bc**2)),
                boundary_max=jnp.max(jnp.abs(bc)),
                conservation_max=jnp.max(jnp.abs(balance/global_flux_scale)),
                boundary_mach=boundary_mach(full, fluid),
                admissible=valid(full.ravel()),
            )

        self.reconstruct = jax.jit(reconstruct)
        self.face_flux, self.source = jax.jit(face_flux), jax.jit(source)
        self.storage, self.spatial_residual = jax.jit(storage), jax.jit(spatial)
        self.boundary_residual = jax.jit(boundary)
        self.residual = jax.jit(assembly["residual"])
        self.steady_residual = jax.jit(assembly["steady"])
        self.jacobian = jax.jit(block["jacobian"])
        self.dense_jacobian = jax.jit(block["dense_jacobian"])
        self.jacobian_difference = jax.jit(block["jacobian_difference"])
        self.step, self.metrics = jax.jit(step), jax.jit(metrics)
        self.is_admissible = jax.jit(lambda full: valid(full.ravel()))
        self.compute_dt = jax.jit(lambda full, cfl:
            cfl*jnp.min(widths/wave_speed(full[1:-1], radius, fluid)))
        self.fields = jax.jit(lambda full: primitive_fields(full[1:-1], radius, fluid))

    def initial_state(self, *, pressure=None, enthalpy=None, v_m=None):
        """Uniform physical cold guess, with independent boundary trace guesses."""
        if self.inlet_regime == "supersonic":
            datum = np.asarray(self.inlet_data["state"])
            return uniform_initial_state(self.r_full, datum[0], datum[3], datum[1],
                                         v_theta=datum[2]/float(self.r_full[0]))
        velocity = 0.05*self.reference_sound_speed if v_m is None else float(v_m)
        angle = self.inlet_data.get("alpha_in", None)
        if angle is None:
            v_theta = float(self.inlet_data["ell_in"])/float(self.r_full[0])
            static_h = float(self.inlet_data["h0"])-0.5*(velocity**2+v_theta**2)
        else:
            v_theta = velocity*np.tan(angle)
            static_h = float(self.inlet_data["h0"])-0.5*(velocity**2+v_theta**2)
        static_p = (float(self.outlet_pressure) if self.outlet_pressure is not None
                    else self.reference_pressure)
        return uniform_initial_state(
            self.r_full, static_p if pressure is None else pressure,
            static_h if enthalpy is None else enthalpy, velocity, v_theta=v_theta)

    def _audit(self, state):
        result = jax.device_get(self.metrics(state))
        return {key: np.asarray(value).tolist() for key, value in result.items()}

    @staticmethod
    def _invalid_metrics():
        return dict(spatial_rms=np.inf, spatial_max=np.inf, boundary_rms=np.inf,
                    boundary_max=np.inf, conservation_max=np.inf,
                    boundary_mach=[np.nan, np.nan], admissible=False)

    def _regimes_match(self, metrics, sonic_tolerance=2e-3):
        mach_in, mach_out = metrics["boundary_mach"]
        def matches(value, regime):
            return value >= 0 and (value <= 1+sonic_tolerance if regime == "subsonic"
                                   else value >= 1-sonic_tolerance)
        return matches(mach_in, self.inlet_regime) and matches(mach_out, self.outlet_regime)

    def solve_steady(self, initial_state=None, *, tolerance=1e-8, max_steps=2000,
                     cfl=20.0, cfl_max=1e6, growth=1.4, max_retries=12, callback=None):
        """Implicit pseudo-time march; success independently audits original rows."""
        state = jnp.asarray(self.initial_state() if initial_state is None else initial_state,
                            dtype=jnp.float64)
        if state.shape != self.shape:
            raise ValueError(f"Initial state must have shape {self.shape}")
        if (not np.all(np.isfinite([tolerance, cfl, cfl_max, growth]))
                or tolerance <= 0 or cfl <= 0 or cfl_max < cfl or not 1 < growth <= 2
                or max_steps < 0 or max_retries < 0):
            raise ValueError("Invalid steady tolerance/CFL/controller settings")
        history = []
        if not bool(self.is_admissible(state)):
            metrics = self._invalid_metrics()
            return AnnularSolution(np.asarray(state), False, "Invalid initial state", history, metrics)
        metrics = self._audit(state)
        def measure(audit):
            return max(audit["spatial_rms"], audit["boundary_max"], audit["conservation_max"])
        previous = measure(metrics)
        for accepted in range(max_steps+1):
            if np.isfinite(previous) and previous < tolerance:
                consistent = self._regimes_match(metrics)
                message = ("Original spatial, boundary and conservation targets met" if consistent
                           else "Converged algebraic root violates selected boundary regime")
                return AnnularSolution(np.asarray(state), consistent, message, history, metrics)
            if accepted == max_steps:
                break
            accepted_trial = False
            for retry in range(max_retries+1):
                dt = float(self.compute_dt(state, cfl))
                candidate, info = self.step(state, dt)
                info = jax.device_get(info)
                record = dict(step=accepted+1, retry=retry, dt=dt, cfl=cfl,
                              inner_iterations=int(info.iterations),
                              inner_residual=float(info.residual_norm),
                              inner_status=int(info.status),
                              rejected_newton_trials=int(info.rejected_trials),
                              accepted=bool(info.converged and info.admissible))
                if record["accepted"]:
                    next_metrics = self._audit(candidate)
                    current = measure(next_metrics)
                    record.update(next_metrics)
                    if not next_metrics["admissible"] or not np.isfinite(current):
                        record["accepted"] = False
                history.append(record)
                if record["accepted"]:
                    state, metrics = candidate, next_metrics
                    # A physical shock transient need not decrease the steady
                    # spatial norm monotonically. Grow from successful inner
                    # work, rather than freezing pseudo-time after that rise.
                    work = int(info.iterations)
                    if work <= max(2, self.options.newton_iterations//3):
                        factor = growth
                    elif work <= 2*self.options.newton_iterations//3:
                        factor = np.sqrt(growth)
                    else:
                        factor = 1.0
                    record["next_cfl_factor"] = factor
                    cfl = min(cfl_max, cfl*factor)
                    previous = current
                    accepted_trial = True
                    if callback is not None:
                        callback(record)
                    break
                cfl *= 0.5
            if not accepted_trial:
                return AnnularSolution(np.asarray(state), False, "Implicit timestep retries exhausted",
                                       history, metrics)
        return AnnularSolution(np.asarray(state), False, "Steady update limit reached", history, metrics)

    def solve_steady_monitored(self, initial_state=None, *, residual_policy=None,
                               monitors=(), pressure_total=None, max_steps=16000,
                               cfl=0.2, cfl_growth=1.2, cfl_shrink=0.5,
                               cfl_max=1e6, max_retries=12, stagnation_steps=0, callback=None):
        """Opt-in equation-wise steady stopping and accepted-state monitors.

        Every active equation must meet the configured initial-relative
        reduction and absolute safeguard. Boundary/global constraints and
        required monitor windows are checked separately. Legacy steady and
        physical-time methods retain their existing definitions.
        """
        from .monitoring import solve_steady_monitored
        return solve_steady_monitored(
            self, initial_state, residual_policy=residual_policy, monitors=monitors,
            pressure_total=pressure_total, max_steps=max_steps, cfl=cfl,
            cfl_growth=cfl_growth, cfl_shrink=cfl_shrink, cfl_max=cfl_max,
            max_retries=max_retries, stagnation_steps=stagnation_steps, callback=callback)

    def advance(self, initial_state, final_time, *, cfl=0.5, max_steps=100000,
                max_retries=12, initial_bc_tolerance=1e-8, callback=None):
        """Backward-Euler physical time; no steady or plateau termination."""
        state = jnp.asarray(initial_state, dtype=jnp.float64)
        if (state.shape != self.shape or not np.all(np.isfinite([final_time, cfl]))
                or final_time < 0 or cfl <= 0 or max_steps < 0 or max_retries < 0):
            raise ValueError("Invalid transient state/time/CFL")
        if not bool(self.is_admissible(state)):
            raise ValueError("Physical transient initialization must be EOS-admissible")
        metrics = self._audit(state)
        if metrics["boundary_max"] > initial_bc_tolerance:
            raise ValueError("Physical transient initialization must satisfy the algebraic BCs")
        if not self._regimes_match(metrics):
            raise ValueError("Initial physical transient state violates the selected boundary regime")
        time, history = 0.0, []
        for count in range(max_steps):
            if time >= final_time:
                return AnnularSolution(np.asarray(state), True, "Requested physical time reached",
                                       history, metrics, time)
            dt = min(float(self.compute_dt(state, cfl)), final_time-time)
            for retry in range(max_retries+1):
                candidate, info = self.step(state, dt)
                info = jax.device_get(info)
                accepted = bool(info.converged and info.admissible)
                history.append(dict(step=count+1, retry=retry, time=time, dt=dt,
                                    accepted=accepted, inner_status=int(info.status),
                                    inner_iterations=int(info.iterations)))
                if accepted:
                    next_metrics = self._audit(candidate)
                    if not self._regimes_match(next_metrics):
                        history[-1]["accepted"] = False
                        history[-1]["boundary_mach"] = next_metrics["boundary_mach"]
                        return AnnularSolution(
                            np.asarray(state), False, "Physical boundary regime change requires a new block",
                            history, metrics, time)
                    state, time, metrics = candidate, time+dt, next_metrics
                    history[-1]["time_new"] = time
                    if callback is not None:
                        callback(history[-1], state)
                    if time >= final_time:
                        return AnnularSolution(np.asarray(state), True, "Requested physical time reached",
                                               history, metrics, time)
                    break
                dt *= 0.5
            else:
                return AnnularSolution(np.asarray(state), False, "Physical timestep retries exhausted",
                                       history, metrics, time)
        if time >= final_time:
            return AnnularSolution(np.asarray(state), True, "Requested physical time reached",
                                   history, metrics, time)
        return AnnularSolution(np.asarray(state), False, "Physical timestep limit reached",
                               history, metrics, time)


def make_annular_solver(fluid, mesh, *, inlet_data, outlet_pressure=None,
                        inlet_regime="subsonic", outlet_regime="subsonic",
                        tau_w=0.0, q_w=0.0, **options):
    """Build one general-EOS model using shared implicit and banded infrastructure.

    Subsonic inlet_data supplies p0,T0,alpha_in, or h0,s0,p0,alpha_in.
    Supersonic inlet_data supplies state=(p,v_m,ell,h). Angle inputs are radians.
    Outlet regime is explicit; subsonic requires a finite positive backpressure.
    tau_w is prescribed nonnegative wall shear magnitude in Pa; q_w is signed
    heat flux in W/m^2, positive into the fluid. Each is scalar or length N and
    defaults to zero. Profiles are fixed at construction; rebuild to change
    them. The named wall closures are placeholders, not empirical correlations.
    """
    if inlet_regime not in ("subsonic", "supersonic") or outlet_regime not in ("subsonic", "supersonic"):
        raise ValueError("Boundary regimes must be 'subsonic' or 'supersonic'")
    if outlet_regime == "subsonic" and (
            outlet_pressure is None or not np.isfinite(outlet_pressure) or outlet_pressure <= 0):
        raise ValueError("A subsonic outlet requires a finite positive backpressure")
    return AnnularSolver(fluid, mesh, inlet_data, outlet_pressure, AnnularOptions(**options),
                         inlet_regime, outlet_regime, tau_w=tau_w, q_w=q_w)


def _wall_values(values, count, name, *, nonnegative=False):
    """Validate and capture fixed scalar/per-cell wall inputs before tracing."""
    try:
        result = np.broadcast_to(np.asarray(values, dtype=float), (count,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be a scalar or length-N array") from exc
    if not np.all(np.isfinite(result)) or (nonnegative and np.any(result < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    result.setflags(write=False)
    return result
