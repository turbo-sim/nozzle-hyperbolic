"""General-EOS nozzle physics implementing the conservative FVM interface.

The model owns storage, physical fluxes, characteristic data, wall/source laws
and algebraic boundary traces. Numerical reconstruction and time integration
are selected independently through the shared solver.
"""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from .characteristics import conservative_projectors
from nozzle_hyperbolic.models.euler_roe import euler_roe_data, roe_metadata
from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid
from nozzle_hyperbolic.models.euler_outlet import EulerOutlet, transmissive_residual
from nozzle_hyperbolic.models.filtering import VelocityGate, face_filter_speed


class NozzleModel:
    """Primitive variables $(p,u,h)$ with user supplied JAXProp thermodynamics.

    ``initial_state`` is either an ``(N+2,3)`` array or a callable receiving
    ``mesh.points``. The first and last rows are independent boundary traces.
    ``wall_closure(W,r,fluid)`` returns shear magnitude and heat flux into the
    fluid. Its optional ``dependency_radius`` declares physical stencil reach;
    optional ``explicit_timestep`` supplies an additional source stability cap.
    ``outlet_regime='auto'`` permits a transient to move from pressure outflow
    through sonic choking to fully outgoing extrapolation. Static regimes
    preserve their previously selected boundary equations.
    """

    nvar = 3
    ntransport = 3
    nconstraints = 0
    nboundary = 3
    nonconservative_mode = "none"
    equation_names = ("Mass", "Momentum", "Energy")
    name = "euler_nozzle"
    source_dependency_radius = 0
    boundary_dependency_radius = 2
    boundary_coupling = "local"

    def __init__(
        self,
        mesh,
        fluid,
        *,
        p0=2.0e5,
        T0=300.0,
        back_pressure=None,
        inlet_regime="subsonic",
        outlet_regime="subsonic",
        transmissive=False,
        outlet_formulation="acoustic",
        initial_state=None,
        wall_closure=None,
        wall_perimeter=None,
        filter_velocity_cutoff=None,
        filter_velocity_width=None,
    ):
        self.mesh = mesh
        self.fluid = as_jax_fluid(fluid)
        self.p0, self.T0 = float(p0), float(T0)
        if not np.all(np.isfinite([self.p0, self.T0])) or min(self.p0, self.T0) <= 0:
            raise ValueError(
                "Reservoir pressure and temperature must be positive and finite"
            )
        if mesh.area_derivative_integrals is None:
            raise ValueError(
                "Nozzle geometry must provide integrated smooth area derivatives"
            )
        if mesh.area_centers is None:
            raise ValueError("Nozzle geometry must provide physical cell-center areas")
        self.back_pressure = None if back_pressure is None else float(back_pressure)
        self.inlet_regime, self.outlet_regime = inlet_regime, outlet_regime
        self.transmissive = bool(transmissive)
        self.outlet_formulation = outlet_formulation
        self._initial_state = initial_state
        self.wall_closure = wall_closure
        self._filter_gate = VelocityGate(filter_velocity_cutoff, filter_velocity_width)
        self._configure_walls(wall_perimeter)
        if not self.transmissive and inlet_regime != "subsonic":
            raise ValueError("The reservoir nozzle has a subsonic inlet")
        if outlet_regime not in ("subsonic", "supersonic", "auto"):
            raise ValueError("Choose the physical outlet regime explicitly")
        if (
            not self.transmissive
            and outlet_regime in ("subsonic", "auto")
            and self.back_pressure is None
        ):
            raise ValueError("Subsonic outflow requires prescribed backpressure")
        reservoir = self.fluid.get_state(jxp.PT_INPUTS, self.p0, self.T0)
        self.h0, self.s0, self.rho0, self.a0 = (
            float(reservoir[key]) for key in ("h", "s", "rho", "a")
        )
        self._roe_reference = (self.rho0, self.p0, self.a0, self.h0)
        self.roe_metadata = roe_metadata(self._roe_reference)
        self.variable_scales = np.array([self.p0, self.a0, self.a0 * self.a0])
        self.conservative_scales = self.rho0 * np.array(
            [1.0, self.a0, self.a0 * self.a0]
        )
        self.reference_time = float(mesh.faces[-1] - mesh.faces[0]) / self.a0
        self.global_flux_scales = (
            float(mesh.area_faces[0]) * self.a0 * self.conservative_scales
        )
        self.entropy_scale = self.a0 * self.a0 / self.T0
        self._left_factor = float(
            (mesh.faces[0] - mesh.centers[0]) / (mesh.centers[0] - mesh.centers[1])
        )
        self._right_factor = float(
            (mesh.faces[-1] - mesh.centers[-1]) / (mesh.centers[-1] - mesh.centers[-2])
        )
        kind = (
            "transmissive"
            if self.transmissive
            else (
                "pressure"
                if outlet_formulation == "finite_wave" or outlet_regime == "subsonic"
                else (
                    "automatic_outflow"
                    if outlet_regime == "auto"
                    else "supersonic_outflow"
                )
            )
        )
        self.outlet = EulerOutlet(
            self.fluid,
            self.back_pressure,
            self.variable_scales,
            self.entropy_scale,
            self._right_factor,
            outlet_formulation,
            kind,
        )

    def _configure_walls(self, perimeter):
        """Set physical wall measures; circular equivalent is the default duct."""
        areas = np.asarray(self.mesh.area_centers)
        self._radius = jnp.asarray(np.sqrt(areas / np.pi))
        if perimeter is None:
            perimeter = 2 * np.sqrt(np.pi * areas)
        values = np.broadcast_to(
            np.asarray(perimeter, dtype=float), (self.mesh.n_cells,)
        ).copy()
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("wall_perimeter must contain positive finite values")
        self._wall_measure = jnp.asarray(values * np.asarray(self.mesh.widths))
        self.source_dependency_radius = getattr(
            self.wall_closure, "dependency_radius", 0
        )

    def thermodynamic_state(self, W):
        W = jnp.asarray(W)
        return self.fluid.get_state(jxp.HmassP_INPUTS, W[..., 2], W[..., 0])

    def face_data(self, W):
        """One EOS call supplies all face physics; auxiliary sound is opaque.

        The numerical engine can reuse this bundle for a numerical flux. The
        ordinary model callbacks below retain the same canonical definitions.
        """
        W = jnp.asarray(W)
        state = self.thermodynamic_state(W)
        rho, sound = state["rho"], state["a"]
        p, u, h = (W[..., i] for i in range(3))
        q = jnp.stack((rho, rho * u, rho * (h + 0.5 * u * u) - p), axis=-1)
        flux = jnp.stack(
            (rho * u, rho * u * u + p, rho * u * (h + 0.5 * u * u)), axis=-1
        )
        return dict(
            conserved=q,
            physical_flux=flux,
            speed_bounds=(u - sound, u + sound),
            auxiliary={"rho": rho, "a": sound, "G": state["G"]},
        )

    def face_conserved(self, W):
        return self.face_data(W)["conserved"]

    cell_conserved = face_conserved

    filter_conserved = face_conserved

    def filter_signal_speed(self, full):
        return face_filter_speed(
            full[:, 1], self.thermodynamic_state(full)["a"], self._filter_gate
        )

    def face_physical_flux(self, W):
        return self.face_data(W)["physical_flux"]

    def face_speed_bounds(self, W):
        return self.face_data(W)["speed_bounds"]

    def face_characteristics(self, left, right, delta_factor):
        """Production conservative projectors and model-specific entropy width."""
        sl, sr = (
            self.thermodynamic_state(left)["a"],
            self.thermodynamic_state(right)["a"],
        )
        return self.face_characteristics_from_data(left, right, delta_factor, sl, sr)

    def face_characteristics_from_data(
        self, left, right, delta_factor, sound_left, sound_right
    ):
        """Reuse left/right EOS data; evaluate only the averaging state here."""
        if isinstance(sound_left, dict):
            sound_left, sound_right = sound_left["a"], sound_right["a"]
        average = 0.5 * (left + right)
        center = self.thermodynamic_state(average)
        projectors = conservative_projectors(
            average, center["rho"], center["a"], center["G"]
        )

        def eigenvalues(W, a):
            u = W[..., 1]
            return jnp.stack((u + a, u - a, u), axis=-1)

        eigen = eigenvalues(average, center["a"])
        spread = eigenvalues(right, sound_right) - eigenvalues(left, sound_left)
        base = delta_factor * center["a"][..., None]
        width = base + 0.25 * (spread + jnp.sqrt(spread * spread + base * base))
        return projectors, eigen, width

    def face_roe_data(self, left, right, delta_factor):
        """Model-owned finite-jump general-EOS homogeneous Euler data."""
        left_state, right_state = self.thermodynamic_state(
            left
        ), self.thermodynamic_state(right)
        return self.face_roe_data_from_data(
            left,
            right,
            delta_factor,
            {key: left_state[key] for key in ("rho", "a", "G")},
            {key: right_state[key] for key in ("rho", "a", "G")},
        )

    def face_roe_data_from_data(self, left, right, delta_factor, aux_left, aux_right):
        """Reuse both PH EOS evaluations from fused physical face data."""
        return euler_roe_data(
            left, right, aux_left, aux_right, self._roe_reference, delta_factor
        )

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        """Integrate p A' using exact smooth-model derivative integrals.

        For the quadratic nozzle, midpoint integration of A' is exact on every
        cell. This evaluates A' analytically; no neighboring-cell area difference
        is interpreted as a derivative. At rest this also balances pressure flux
        through the exact face areas to roundoff.
        """
        full = jnp.asarray(full)
        source = jnp.zeros((self.mesh.n_cells, self.nvar), dtype=full.dtype)
        source = source.at[:, 1].set(
            full[1:-1, 0] * jnp.asarray(self.mesh.area_derivative_integrals)
        )
        if self.wall_closure is not None:
            cells = full[1:-1]
            shear, heat = self.wall_closure(cells, self._radius, self.fluid)
            coefficient = getattr(self.wall_closure, "drag_coefficient", None)
            if callable(coefficient):
                drag = coefficient(cells, self._radius, self.fluid) * cells[:, 1]
            else:
                speed = jnp.sqrt(jnp.where(cells[:, 1] ** 2 > 0, cells[:, 1] ** 2, 1.0))
                drag = shear * cells[:, 1] / speed
            source = source.at[:, 1].add(-self._wall_measure * drag)
            source = source.at[:, 2].set(self._wall_measure * heat)
        return source

    def explicit_timestep(self, full):
        """Optional physical source restriction, independent of integrator choice."""
        bound = getattr(self.wall_closure, "explicit_timestep", None)
        if not callable(bound):
            return jnp.asarray(jnp.inf)
        wall_per_volume = self._wall_measure / jnp.asarray(self.mesh.volumes)
        return jnp.min(bound(full[1:-1], self._radius, self.fluid, wall_per_volume))

    def boundary_residual(self, full, time=0.0):
        """Independent boundary traces; no nested solve or state projection."""
        full = jnp.asarray(full)
        if self.transmissive:
            return transmissive_residual(full, self.variable_scales)
        cells = full[1:-1]
        extrap_left = cells[0] + self._left_factor * (cells[0] - cells[1])
        st = self.thermodynamic_state(cells)
        impedance = st["rho"] * st["a"]
        inlet = full[0]
        inlet_state = self.thermodynamic_state(inlet)
        rows_in = jnp.stack(
            (
                (inlet[2] + 0.5 * inlet[1] ** 2 - self.h0) / (self.a0 * self.a0),
                (inlet_state["s"] - self.s0) / self.entropy_scale,
                (inlet[0] - extrap_left[0] - impedance[0] * (inlet[1] - extrap_left[1]))
                / self.p0,
            )
        )
        rows_out = self.outlet.residual(full[-1], cells)
        return jnp.stack((rows_in, rows_out))

    def boundary_face_flux(self, full, flux):
        """Unweighted transported flux; the engine applies area exactly once."""
        if self.outlet_formulation != "finite_wave" and self.outlet_regime != "auto":
            return flux
        return flux.at[-1].set(
            jax.lax.cond(
                self.outlet.direct_flux(full[1:-1]),
                lambda: self.face_physical_flux(full[-1]),
                lambda: flux[-1],
            )
        )

    def boundary_wave_speed(self, full):
        traces = full[jnp.array([0, -1])]
        speeds = jnp.abs(traces[:, 1]) + self.thermodynamic_state(traces)["a"]
        return speeds.at[-1].set(
            jnp.maximum(speeds[-1], self.outlet.signal_speed(full[1:-1]))
        )

    def boundary_diagnostics(self, full):
        if self.outlet_formulation == "finite_wave":
            wave = self.outlet.wave(full[1:-1])
            return dict(
                mode=wave.mode,
                direct_flux=wave.direct_flux,
                head_speed=wave.head_speed,
                tail_speed=wave.tail_speed,
            )
        if self.outlet_regime == "auto":
            direct = self.outlet.direct_flux(full[1:-1])
            sound = self.thermodynamic_state(full[-1])["a"]
            sonic = jnp.abs(full[-1, 1] - sound) <= 1e-8 * self.variable_scales[1]
            return dict(
                mode=jnp.where(direct, jnp.where(sonic, 6, 2), 1),
                direct_flux=direct,
                sonic_limited=direct & sonic,
            )
        return dict(mode=jnp.int32(0), direct_flux=jnp.asarray(False))

    def _admissible_states(self, W):
        W = jnp.asarray(W)
        domain = jnp.all(jnp.isfinite(W)) & jnp.all(W[..., 0] > 0)
        for field, index in (("p", 0), ("h", 2)):
            for suffix, comparison in (
                ("min", jnp.greater_equal),
                ("max", jnp.less_equal),
            ):
                bound = getattr(self.fluid, f"{field}_{suffix}", None)
                if bound is not None:
                    domain = domain & jnp.all(comparison(W[..., index], bound))

        def eos_valid():
            state = self.thermodynamic_state(W)
            valid = jnp.array(True)
            for key in ("rho", "a", "T"):
                value = state[key]
                valid = valid & jnp.all(jnp.isfinite(value) & (value > 0))
            valid = valid & jnp.all(jnp.isfinite(state["G"]))
            # Reject EOS calls that silently clamp or replace HP coordinates.
            # Sound-speed scales remain invariant under enthalpy-reference shifts.
            coordinate_rtol = getattr(self.fluid, "state_consistency_rtol", 1e-8)
            for key, index, physical_scale in (
                ("p", 0, state["rho"] * state["a"] ** 2),
                ("h", 2, state["a"] ** 2),
            ):
                scale = jnp.maximum(
                    jnp.maximum(jnp.abs(W[..., index]), physical_scale), 1.0
                )
                valid = valid & jnp.all(
                    jnp.isfinite(state[key])
                    & (jnp.abs(state[key] - W[..., index]) <= coordinate_rtol * scale)
                )
            # Catch an EOS that silently floors temperature at invalid enthalpy.
            _, entropy_h = jax.jvp(
                lambda h: self.fluid.get_state(jxp.HmassP_INPUTS, h, W[..., 0])["s"],
                (W[..., 2],),
                (jnp.ones_like(W[..., 2]),),
            )
            tolerance = getattr(self.fluid, "thermodynamic_consistency_rtol", 1e-3)
            valid = valid & jnp.all(
                jnp.isfinite(entropy_h)
                & (jnp.abs(state["T"] * entropy_h - 1) < tolerance)
            )
            return valid

        return jax.lax.cond(domain, eos_valid, lambda: jnp.array(False))

    def admissible(self, full, left, right):
        return (
            self._admissible_states(full)
            & self._admissible_states(left)
            & self._admissible_states(right)
        )

    cell_admissible = _admissible_states
    boundary_admissible = _admissible_states

    def wave_speed(self, Wphysical):
        return (
            jnp.abs(jnp.asarray(Wphysical)[..., 1])
            + self.thermodynamic_state(Wphysical)["a"]
        )

    def initial_state(self):
        """Simple cold start, independent of analytical profiles/shock location."""
        if self._initial_state is not None:
            values = (
                self._initial_state(self.mesh.points)
                if callable(self._initial_state)
                else self._initial_state
            )
            values = jnp.asarray(values)
            if values.shape != (self.mesh.n_cells + 2, self.nvar):
                raise ValueError(
                    "initial_state must provide the full (N+2,3) primitive state"
                )
            return values
        if self.transmissive:
            return jnp.broadcast_to(
                jnp.asarray([self.p0, 0.0, self.h0]), (self.mesh.n_cells + 2, self.nvar)
            )
        if self.outlet_regime == "supersonic":
            # An arbitrary smooth acceleration establishes the selected outgoing
            # regime; no area-Mach root or analytical field is used here.
            fraction = (self.mesh.points - self.mesh.faces[0]) / (
                self.mesh.faces[-1] - self.mesh.faces[0]
            )
            velocity = self.a0 * (0.2 + 1.3 * fraction)
            # Supersonic outflow has no prescribed pressure law. An optional
            # backpressure is only a cold-start datum in this branch.
            pressure_end = (
                self.back_pressure if self.back_pressure is not None else 0.1 * self.p0
            )
            pressure = self.p0 + (pressure_end - self.p0) * fraction
            return jnp.asarray(
                np.stack(
                    (pressure, velocity, self.h0 - 0.5 * velocity * velocity), axis=-1
                )
            )
        area = np.r_[
            self.mesh.area_faces[0], self.mesh.area_centers, self.mesh.area_faces[-1]
        ]
        # Bernoulli/continuity cold guess. A hard-coded enthalpy floor would
        # depend on the EOS energy reference, so use only a declared bound or
        # a reference-invariant drop from the reservoir enthalpy.
        pressure = np.linspace(self.p0, self.back_pressure, len(self.mesh.points))
        h_lower = getattr(self.fluid, "h_min", None)
        floor = (
            self.h0 - self.a0**2
            if h_lower is None
            else float(h_lower) + 1e-8 * self.a0**2
        )
        velocity_max = 0.9 * np.sqrt(max(2 * (self.h0 - floor), 0.0))
        at_rest = self.fluid.get_state(
            jxp.HmassP_INPUTS, jnp.full(len(pressure), self.h0), jnp.asarray(pressure)
        )
        density, sound = np.asarray(at_rest["rho"]), np.asarray(at_rest["a"])
        velocity_out = min(
            np.sqrt(max(2 * (self.p0 - self.back_pressure) / density[-1], 0.0)),
            velocity_max,
        )
        mass_flow = min(
            density[-1] * velocity_out * area[-1],
            float(np.min(0.5 * density * sound * area)),
        )
        velocity = np.minimum(mass_flow / (density * area), velocity_max)
        enthalpy = np.maximum(self.h0 - 0.5 * velocity**2, floor)
        density = np.asarray(
            self.fluid.get_state(
                jxp.HmassP_INPUTS, jnp.asarray(enthalpy), jnp.asarray(pressure)
            )["rho"]
        )
        velocity = np.minimum(mass_flow / (density * area), velocity_max)
        enthalpy = np.maximum(self.h0 - 0.5 * velocity**2, floor)
        return jnp.asarray(np.stack((pressure, velocity, enthalpy), axis=-1))

    def primitive_from_conserved(self, q):
        """Direct EOS recovery when the backend supports density/energy.

        The shared explicit step instead solves storage(W)=target through the
        HP EOS and supports backends without the density/energy input pair.
        """
        q = jnp.asarray(q)
        rho = q[..., 0]
        velocity = q[..., 1] / rho
        internal = q[..., 2] / rho - 0.5 * velocity * velocity
        state = self.fluid.get_state(jxp.DmassUmass_INPUTS, rho, internal)
        return jnp.stack((state["p"], velocity, state["h"]), axis=-1)

    def fields(self, full):
        """Physical-cell fields; boundary traces are not cell averages."""
        cells = jnp.asarray(full)[1:-1]
        state = self.thermodynamic_state(cells)
        rho, a = state["rho"], state["a"]
        return dict(
            p=cells[:, 0],
            rho=rho,
            u=cells[:, 1],
            M=cells[:, 1] / a,
            T=state["T"],
            h=cells[:, 2],
            h0=cells[:, 2] + 0.5 * cells[:, 1] ** 2,
            mdot=rho * cells[:, 1] * jnp.asarray(self.mesh.area_centers),
        )

    def quantities(self, full, face_flux):
        return dict(mdot=jnp.mean(jnp.asarray(face_flux)[:, 0]))

    def regime_audit(self, full, sonic_tolerance=2e-3):
        if self.transmissive:
            return True
        boundary = jnp.asarray(full)[jnp.array([0, -1])]
        mach = boundary[:, 1] / self.thermodynamic_state(boundary)["a"]
        inlet_ok = (mach[0] >= 0) & (mach[0] <= 1 + sonic_tolerance)
        outlet_ok = (mach[1] >= 0) & (
            True
            if self.outlet_formulation == "finite_wave" or self.outlet_regime == "auto"
            else (
                (mach[1] <= 1 + sonic_tolerance)
                if self.outlet_regime == "subsonic"
                else (mach[1] >= 1 - sonic_tolerance)
            )
        )
        forward = jnp.all(jnp.asarray(full)[1:-1, 1] >= -1e-12 * self.a0)
        return bool(inlet_ok & outlet_ok & forward)

    solution_audit = regime_audit

    def transient_solution_audit(self, full, sonic_tolerance=2e-3):
        """Audit transient characteristic bounds separately from steady direction.

        Automatic outflow allows signed interior velocities and a signed
        subsonic reservoir trace near startup. A negative pressure-outlet
        velocity still requires exterior thermodynamic data absent from this
        model. The shared driver independently checks algebraic equations and
        reconstructed-state admissibility. Steady forward-flow tests remain
        unchanged and are not used to label intermediate physical snapshots.
        """
        if self.transmissive or self.outlet_regime != "auto":
            return self.regime_audit(full, sonic_tolerance)
        if not bool(self.boundary_admissible(full)):
            return False
        boundary = jnp.asarray(full)[jnp.array([0, -1])]
        mach = boundary[:, 1] / self.thermodynamic_state(boundary)["a"]
        return bool((jnp.abs(mach[0]) <= 1 + sonic_tolerance) & (mach[1] >= 0))
