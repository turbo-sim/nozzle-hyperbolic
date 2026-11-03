"""Constrained eight-state Alberto model for the generic finite-volume engine.

The compatibility discretization is explicitly identified: frozen FVS followed
by cell X K^-1 flux-difference correction, then independently selected filters.
This module contains no time integrator, Newton algorithm, or Jacobian cache.
"""

from dataclasses import asdict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid
from .constrained_state import conserved, primitives, from_primitives, eos_constraints
from .configuration import BoundaryConfig, ClosureConfig, CompatibilityConfig, StabilizationConfig, PhasePinConfig
from ._alberto_matrices import phase_props, common_quantities, eigenstructure, flux_vector
from ._alberto_flux import fvs_interface_flux, nonconservative_correction
from ._alberto_sources import two_fluid_closure, D_VE_algebric, D_VE_algebric_two_component, D_VE_area, evaporation_weight


def _table_stencil_mask(fluid, transport):
    """Audit populated bicubic cells without replacing or modifying properties."""
    table = getattr(fluid, "table", None)
    if table is None or not hasattr(fluid, "N_h"):
        return None
    shape = (fluid.N_h, fluid.N_p)
    nodes = np.ones(shape, dtype=bool)
    cells = np.ones((shape[0]-1, shape[1]-1), dtype=bool)
    positive = {"density", "temperature", "speed_of_sound"}
    required = positive | {"gruneisen", "entropy"}
    if transport:
        positive |= {"viscosity", "conductivity", "isobaric_heat_capacity"}
        required |= positive
    for key in required:
        values = np.asarray(table[key]["value"])
        nodes &= np.isfinite(values)
        if key in positive:
            nodes &= values > 0
        cells &= np.all(np.isfinite(np.asarray(table[key]["coeffs"])), axis=-1)
    # Metastable builders freeze failed spinodal extensions by copying the
    # previous density row. Such copied states are not accepted physical data.
    phase = getattr(fluid, "metastable_phase", None)
    density = np.asarray(table["density"]["value"])
    if phase == "liquid":
        frozen = np.zeros_like(nodes)
        frozen[1:] = np.maximum.accumulate(density[1:] == density[:-1], axis=0)
        nodes &= ~frozen
    elif phase == "vapor":
        frozen = np.zeros_like(nodes)
        frozen[:-1] = np.maximum.accumulate((density[:-1] == density[1:])[::-1], axis=0)[::-1]
        nodes &= ~frozen
    return jnp.asarray(cells & nodes[:-1, :-1] & nodes[1:, :-1] & nodes[:-1, 1:] & nodes[1:, 1:])


class TwoFluidModel:
    """Six phase balances and two local EOS equations in eight unknowns.

    Supply two JAXProp phase fluids and initial enthalpy primitives
    ``(p,u_g,u_l,alpha_g,h_g,h_l)`` or an eight-state ``initial_state``.
    ``closures=None`` explicitly disables interphase exchange for verification.
    Use ``ClosureConfig()`` for upstream one-component mass/heat/drag physics.
    Fixed phase scales are never multiplied by a local void fraction.
    """

    name = "two_fluid_alberto"
    reference_revision = "faafff704e54fd866abff95fcb88fc0a5217405c"
    nvar, ntransport, nconstraints, nboundary = 8, 6, 2, 8
    nonconservative_mode = "flux_difference"
    supported_fluxes = ("model_compatibility",)
    equation_names = ("Gas mass", "Liquid mass", "Gas momentum", "Liquid momentum", "Gas energy", "Liquid energy")
    constraint_names = ("Gas EOS", "Liquid EOS")
    source_dependency_radius = 0
    nonconservative_dependency_radius = 1
    boundary_dependency_radius = 2
    boundary_coupling = "local"
    equation_dependency_radius = 0
    invariant_names = ("Total mass",)
    invariant_projections = np.array(((1., 1., 0., 0., 0., 0.),))

    def __init__(self, mesh, fluid_g, fluid_l, *, initial_primitives=None,
                 initial_state=None, boundaries=None, closures=None,
                 compatibility=None, stabilization=None, pins=None, virtual_mass=0.5,
                 source_scale=1.0, area_source=True):
        self.mesh = mesh
        self.fluid_g, self.fluid_l = as_jax_fluid(fluid_g), as_jax_fluid(fluid_l)
        self.boundaries = boundaries or BoundaryConfig()
        self.closures = closures
        self.compatibility = compatibility or CompatibilityConfig()
        self.stabilization = stabilization or StabilizationConfig()
        self.pins = pins or PhasePinConfig()
        self.has_equation_pins = self.pins.saturation_energy or self.pins.momentum
        self.steady_only = self.has_equation_pins
        self.virtual_mass = float(virtual_mass)
        self.source_scale = float(source_scale)
        self.area_source = bool(area_source)
        if not np.isfinite(self.virtual_mass) or self.virtual_mass < 0:
            raise ValueError("virtual_mass must be finite and nonnegative")
        if not np.isfinite(self.source_scale) or self.source_scale < 0:
            raise ValueError("source_scale must be finite and nonnegative")
        if mesh.area_centers is None:
            raise ValueError("Two-fluid geometry requires area_centers")
        if (initial_state is None) == (initial_primitives is None):
            raise ValueError("Supply exactly one of initial_primitives or initial_state")
        if initial_state is None:
            U = initial_primitives(mesh.points) if callable(initial_primitives) else initial_primitives
            U = np.broadcast_to(np.asarray(U, dtype=float), (mesh.n_cells + 2, 6)).copy()
            if np.any((U[:, 3] <= self.stabilization.alpha_min) | (U[:, 3] >= 1-self.stabilization.alpha_min)):
                raise ValueError("Initial state violates open phase-fraction bounds; pure phases require a reduced model")
            self._initial = from_primitives(U, self.fluid_g, self.fluid_l)
        else:
            W = initial_state(mesh.points) if callable(initial_state) else initial_state
            self._initial = jnp.asarray(np.broadcast_to(np.asarray(W, dtype=float), (mesh.n_cells + 2, 8)).copy())
            alpha = np.asarray(self._initial[:, 1])
            if np.any((alpha <= self.stabilization.alpha_min) | (alpha >= 1-self.stabilization.alpha_min)):
                raise ValueError("Initial state violates open phase-fraction bounds; pure phases require a reduced model")
        U0 = np.asarray(primitives(self._initial))[0]
        pg, pl = self._properties(jnp.asarray(U0))
        rg, rl = float(pg.rho), float(pl.rho)
        sound = max(float(pg.a), float(pl.a))
        pressure = float(U0[0])
        if min(pressure, rg, rl, sound) <= 0 or not np.all(np.isfinite([pressure, rg, rl, sound])):
            raise ValueError("Initial thermodynamic reference must be finite and physical")
        self.variable_scales = np.array((pressure, 1., rg, sound, rl, sound, rg * sound**2, rl * sound**2))
        self.conservative_scales = np.array((rg, rl, rg * sound, rl * sound, rg * sound**2, rl * sound**2))
        self.constraint_scales = np.array((rg, rl))
        self.primitive_scales = jnp.array((pressure, sound, sound, 1., sound**2, sound**2))
        self.reconstruction_scales = np.asarray(self.primitive_scales)
        self.entropy_scale = sound**2 / max(float(pg.T), float(pl.T))
        self.reference_time = float(mesh.faces[-1] - mesh.faces[0]) / sound
        self.global_flux_scales = mesh.area_faces[0] * sound * self.conservative_scales
        self.invariant_flux_scales = np.array((self.global_flux_scales[0]+self.global_flux_scales[1],))
        self.stabilization_dependency_radius = 2 if (self.stabilization.void_strength or self.stabilization.thin_phase_strength) else 0
        self._delta_area = jnp.asarray(np.diff(mesh.area_faces))
        self._volumes = jnp.asarray(mesh.volumes)
        self._areas = jnp.asarray(mesh.area_centers)
        self._face_areas = jnp.asarray(mesh.area_faces)
        self._phase_stencil_masks = tuple(_table_stencil_mask(fluid, closures is not None and self.source_scale != 0)
                                         for fluid in (self.fluid_g, self.fluid_l))
        self._left_factor = float((mesh.faces[0] - mesh.centers[0]) / (mesh.centers[0] - mesh.centers[1]))
        if self._needs_saturation:
            for fluid in (self.fluid_g, self.fluid_l):
                if not hasattr(fluid, "get_state_saturation"):
                    raise ValueError("Selected closures require explicit phase saturation tables")
                if hasattr(fluid, "table") and not fluid.table.get("saturation_props"):
                    raise ValueError("Selected closures require populated phase saturation tables")
        if not bool(self.cell_admissible(self._initial)):
            raise ValueError("Initial state violates phase, PH-domain or hyperbolicity checks")

    @property
    def _needs_saturation(self):
        c = self.closures
        return self.pins.saturation_energy or (c is not None and (c.mass_transfer or c.htc_liquid_bubbly != "ranz_marshall" or c.htc_liquid_droplet != "internal"))

    def validate_scheme(self, scheme):
        if scheme.flux not in self.supported_fluxes:
            raise ValueError("Two-fluid compatibility discretization requires flux='model_compatibility'")
        if self.has_equation_pins and scheme.time_integrator != "euler-implicit":
            raise ValueError("Equation-changing two-fluid pins require implicit steady solution")

    def transform_cell_equations(self, full, integrated_rows, dt=None):
        """Apply selected pin laws once to spatial or backward-Euler balances.

        The generic engine retains the original physical residual separately.
        Energy's weight is binary, so choosing the fixed reference time for
        its steady algebraic row does not alter the steady root. Momentum's
        finite tau makes its selected root independent of pseudo-time step.
        """
        if not self.has_equation_pins:
            return integrated_rows
        U = primitives(full[1:-1])
        mg = full[1:-1, 2]
        result = integrated_rows
        if self.pins.saturation_energy:
            def one(u):
                sat_g = self.fluid_g.get_state_saturation(u[0])
                sat_l = self.fluid_l.get_state_saturation(u[0])
                liquid = phase_props(self.fluid_l, u[0], u[5])
                free = evaporation_weight(u[5], liquid.T, sat_l["enthalpy"],
                                          sat_l["temperature"], u[3],
                                          p=u[0], p_max=self.pins.pressure_cutoff)
                return free, u[4]-sat_g["enthalpy"]
            free, enthalpy_delta = jax.lax.map(one, U)
            rate = 1/self.reference_time if dt is None else 1.
            pin = self._volumes*mg*enthalpy_delta*rate
            result = result.at[:, 4].set(free*result[:, 4]+(1-free)*pin)
        if self.pins.momentum:
            free = jax.nn.sigmoid(4*jnp.log(U[:, 3]/self.pins.momentum_alpha))
            rate = (1. if dt is None else dt)/self.pins.momentum_time
            pin = self._volumes*mg*(U[:, 1]-U[:, 2])*rate
            result = result.at[:, 2].set(free*result[:, 2]+(1-free)*pin)
        return result

    def _properties(self, U):
        return phase_props(self.fluid_g, U[0], U[4]), phase_props(self.fluid_l, U[0], U[5])

    def _wave_data(self, U):
        eg, el = self._properties(U)
        cq = common_quantities(U, eg, el, self.virtual_mass)
        return eigenstructure(U, eg, el, cq)

    def initial_state(self):
        return jnp.array(self._initial)

    cell_conserved = staticmethod(conserved)
    face_conserved = staticmethod(conserved)
    filter_conserved = staticmethod(conserved)

    def cell_constraints(self, W):
        return eos_constraints(W, self.fluid_g, self.fluid_l)

    reconstruction_state = staticmethod(primitives)

    def from_reconstruction_state(self, U):
        return from_primitives(U, self.fluid_g, self.fluid_l)

    def face_physical_flux(self, W):
        def one(U):
            eg, el = self._properties(U)
            return flux_vector(U, eg, el)
        return jax.vmap(one)(primitives(W))

    def face_speed_bounds(self, W):
        ev = jax.vmap(lambda U: self._wave_data(U)[0])(primitives(W))
        return jnp.min(ev, axis=-1), jnp.max(ev, axis=-1)

    def wave_speed(self, W):
        lower, upper = self.face_speed_bounds(W)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def boundary_wave_speed(self, full):
        return self.wave_speed(full[jnp.array((0, full.shape[0] - 1))])

    @staticmethod
    def flux_to_conserved_frame(flux):
        return flux

    def numerical_face_flux(self, left, right, scheme):
        if scheme.flux != "model_compatibility":
            raise ValueError("TwoFluidModel supports only the explicit model_compatibility flux")
        cfg = self.compatibility
        def one(pair):
            ul, ur, dissipation = pair
            value = fvs_interface_flux(ul, ur, self.fluid_g, self.fluid_l, self.virtual_mass,
                                      delta_factor=cfg.delta_factor, lowmach_diss=dissipation,
                                      du_blend_lo=cfg.du_blend_lo, du_blend_hi=cfg.du_blend_hi,
                                      single_branch=False)
            if self.stabilization.mass_upwind_cutoff is not None:
                egL, elL = self._properties(ul)
                egR, elR = self._properties(ur)
                for row, ui, aL, aR, rhoL, rhoR in (
                    (0, 1, ul[3], ur[3], egL.rho, egR.rho),
                    (1, 2, 1-ul[3], 1-ur[3], elL.rho, elR.rho)):
                    cutoff = jnp.log10(self.stabilization.mass_upwind_cutoff)
                    wL = 0.5 * (1-jnp.tanh((jnp.log10(aL)-cutoff)/0.25))
                    wR = 0.5 * (1-jnp.tanh((jnp.log10(aR)-cutoff)/0.25))
                    weight = 1-(1-wL)*(1-wR)
                    sign = jnp.tanh(0.5*(ul[ui]+ur[ui])/1e-3)
                    up = 0.5*(1+sign)*aL*rhoL*ul[ui]+0.5*(1-sign)*aR*rhoR*ur[ui]
                    value = value.at[row].set(weight*up+(1-weight)*value[row])
            return value
        # lax.map preserves scalar branch execution at equal slip; vmapped
        # conditionals evaluate the singular inactive full-slip decomposition.
        dissipation = jnp.full((left.shape[0],), cfg.lowmach_diss)
        if cfg.lowmach_diss_boundary is not None:
            dissipation = dissipation.at[0].set(cfg.lowmach_diss_boundary).at[-1].set(cfg.lowmach_diss_boundary)
        return jax.lax.map(one, (primitives(left), primitives(right), dissipation))

    def integrated_nonconservative(self, full, unfiltered_flux):
        U = primitives(full[1:-1])
        cfg = self.compatibility
        if cfg.nc_form == "cell":
            delta = self._areas[:, None] * (unfiltered_flux[1:] - unfiltered_flux[:-1])
        else:
            delta = (self._face_areas[1:, None]*unfiltered_flux[1:]
                     - self._face_areas[:-1, None]*unfiltered_flux[:-1])
            if cfg.nc_form == "wb":
                delta = delta-self._delta_area[:, None]*self.face_physical_flux(full[1:-1])
        def one(args):
            u, df = args
            eg, el = self._properties(u)
            cq = common_quantities(u, eg, el, self.virtual_mass)
            return nonconservative_correction(u, df, eg, el, cq)
        return jax.lax.map(one, (U, delta))

    def integrated_source(self, full):
        def one(args):
            u, volume, delta_area = args
            eg, el = self._properties(u)
            cq = common_quantities(u, eg, el, self.virtual_mass)
            value = jnp.zeros(6, dtype=u.dtype)
            if self.closures is not None and self.source_scale != 0:
                if self._needs_saturation:
                    sg = self.fluid_g.get_state_saturation(u[0])
                    sl = self.fluid_l.get_state_saturation(u[0])
                    Ts, hgs, hls = sg["temperature"], sg["enthalpy"], sl["enthalpy"]
                else:
                    Ts, hgs, hls = 0., 0., 0.
                closure = two_fluid_closure(u, eg, el, self.fluid_g, self.fluid_l,
                                           Ts, hgs, hls, **asdict(self.closures))
                assembly = D_VE_algebric if self.closures.mass_transfer else D_VE_algebric_two_component
                value = value + self.source_scale * volume * assembly(u, cq, closure, eg, el, self.virtual_mass)
            if self.area_source:
                # Exact integrated area change pairs the wall pressure with the
                # same face measures used by the generic flux divergence.
                value = value + D_VE_area(u, cq, SimpleNamespace(dlnA_dx=delta_area), eg, el)
            return value
        return jax.lax.map(one, (primitives(full[1:-1]), self._volumes, self._delta_area))

    def boundary_residual(self, full, time):
        U = primitives(full)
        bc = self.boundaries
        left, neighbor = U[0], U[1]
        if bc.inlet == "transmissive":
            inlet = (left-neighbor)/self.primitive_scales
        elif bc.inlet == "static_state":
            inlet = (left-jnp.asarray(bc.inlet_state))/self.primitive_scales
        elif bc.inlet == "subsonic_inflow":
            target = jnp.asarray(bc.inlet_state).at[0].set(neighbor[0])
            inlet = (left-target)/self.primitive_scales
        elif bc.inlet == "total_state":
            extrap = neighbor+self._left_factor*(neighbor-U[2])
            target = extrap.at[3].set(bc.alpha0).at[4].set(bc.H_g0-0.5*left[1]**2).at[5].set(bc.H_l0-0.5*left[2]**2)
            inlet = (left-target)/self.primitive_scales
        else:
            eg, el = self._properties(left)
            ng, nl = self._properties(neighbor)
            mass_g, mass_l = left[3]*eg.rho, (1-left[3])*el.rho
            mix_u = (mass_g*left[1]+mass_l*left[2])/(mass_g+mass_l)
            mix_neighbor = (neighbor[3]*ng.rho*neighbor[1]+(1-neighbor[3])*nl.rho*neighbor[2])/(neighbor[3]*ng.rho+(1-neighbor[3])*nl.rho)
            fluid, hi = (self.fluid_g, 4) if bc.dominant_phase == "gas" else (self.fluid_l, 5)
            entropy = fluid.get_state(jxp.HmassP_INPUTS, left[hi], left[0])["s"]
            # Cross-multiplied quality avoids division by a possibly zero
            # inlet velocity and determines alpha through EOS and fixed slip.
            composition = ((1-bc.quality)*mass_g*bc.slip-bc.quality*mass_l)/(self.constraint_scales[0]+self.constraint_scales[1])
            inlet = jnp.stack(((entropy-bc.s0)/self.entropy_scale,
                               (left[1]-bc.slip*left[2])/self.primitive_scales[1],
                               (mix_u-mix_neighbor)/self.primitive_scales[1], composition,
                               (left[4]+0.5*left[1]**2-bc.H_g0)/self.primitive_scales[4],
                               (left[5]+0.5*left[2]**2-bc.H_l0)/self.primitive_scales[5]))
        inside = U[-2]
        target = inside
        if bc.outlet != "transmissive":
            eg, el = self._properties(inside)
            ev, right = self._wave_data(inside)[:2]
            pressure = bc.back_pressure(time) if callable(bc.back_pressure) else bc.back_pressure
            weight = 1.
            if bc.outlet in ("characteristic", "characteristic_old"):
                mach = jnp.abs(ev[2]+ev[3])/(ev[2]-ev[3])
                weight = 0.5*(1+jnp.tanh((1-mach)/bc.mach_width))
            dp = weight*bc.pressure_relaxation*(pressure-inside[0])
            if bc.outlet == "characteristic":
                target = inside + dp*right[:, 3]/right[0, 3]
            else:
                rho = inside[3]*eg.rho+(1-inside[3])*el.rho
                compliance = inside[3]/(eg.rho*eg.a**2)+(1-inside[3])/(el.rho*el.a**2)
                du = -dp/jnp.sqrt(rho/compliance)
                target = inside.at[0].add(dp).at[1].add(du).at[2].add(du)
        outlet = (U[-1]-target)/self.primitive_scales
        traces = full[jnp.array((0, full.shape[0]-1))]
        constraints = self.cell_constraints(traces)/jnp.asarray(self.constraint_scales)
        return jnp.concatenate((jnp.stack((inlet, outlet)), constraints), axis=1)

    def cell_admissible(self, W):
        U = primitives(W)
        valid = (jnp.all(jnp.isfinite(W)) & jnp.all(W[..., 0] > 0)
                 & jnp.all(W[..., 2] > 0) & jnp.all(W[..., 4] > 0)
                 & jnp.all(W[..., 1] > self.stabilization.alpha_min)
                 & jnp.all(W[..., 1] < 1-self.stabilization.alpha_min))
        for fluid, hi, stencil_mask in zip((self.fluid_g, self.fluid_l), (4, 5), self._phase_stencil_masks):
            p, h = U[..., 0], U[..., hi]
            valid = valid & jnp.all((p >= getattr(fluid, "p_min", 0.)) & (p <= getattr(fluid, "p_max", jnp.inf)))
            valid = valid & jnp.all((h >= getattr(fluid, "h_min", -jnp.inf)) & (h <= getattr(fluid, "h_max", jnp.inf)))
            if stencil_mask is not None:
                i = jnp.clip(jnp.floor((h-fluid.h_min)/fluid.delta_h).astype(int), 0, fluid.N_h-2)
                index_pressure = jnp.where(p > 0, p, 1.)
                j = jnp.clip(jnp.floor((jnp.log(index_pressure)-jnp.log(fluid.p_min))/fluid.delta_logP).astype(int), 0, fluid.N_p-2)
                valid = valid & jnp.all(stencil_mask[i, j])
            state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
            valid = valid & jnp.all(jnp.isfinite(state["rho"]) & (state["rho"] > 0)
                                    & jnp.isfinite(state["a"]) & (state["a"] > 0)
                                    & jnp.isfinite(state["T"]) & (state["T"] > 0))
            if self.closures is not None and self.source_scale:
                for key in ("viscosity", "conductivity", "cp"):
                    valid = valid & jnp.all(jnp.isfinite(state[key]) & (state[key] > 0))
            if self._needs_saturation:
                sat_grid = getattr(fluid, "logP_sat_vals", None)
                if sat_grid is not None:
                    valid = valid & jnp.all((p >= jnp.exp(sat_grid[0])) & (p <= jnp.exp(sat_grid[-1])))
                sat = fluid.get_state_saturation(p)
                valid = valid & jnp.all(jnp.isfinite(sat["enthalpy"]) & jnp.isfinite(sat["temperature"]) & (sat["temperature"] > 0))
        if self._needs_saturation:
            sg = self.fluid_g.get_state_saturation(U[..., 0])
            sl = self.fluid_l.get_state_saturation(U[..., 0])
            valid = valid & jnp.all(sg["enthalpy"] > sl["enthalpy"])
        speed = self.wave_speed(W)
        return valid & jnp.all(jnp.isfinite(speed))

    def boundary_admissible(self, full):
        return self.cell_admissible(full[jnp.array((0, full.shape[0]-1))])

    def admissible(self, full, left, right):
        return self.cell_admissible(full) & self.cell_admissible(left) & self.cell_admissible(right)

    def solution_audit(self, full):
        """Check the physical characteristic branch independently of residuals.

        Signed interior velocities remain admissible. Reservoir/pressure laws
        prescribe forward-flow exterior data; sustained reverse inflow needs
        different exterior specifications and is not certified by these laws.
        """
        U = primitives(full)
        valid = self.cell_admissible(full)
        if self.boundaries.inlet == "stagnation":
            ev = self._wave_data(U[0])[0]
            valid = valid & (U[0, 1] > 0) & (U[0, 2] > 0) & (ev[3] < 0) & (ev[2] > 0)
        if self.boundaries.outlet != "transmissive":
            valid = valid & (U[-1, 1] >= 0) & (U[-1, 2] >= 0)
        return valid

    def fields(self, full):
        U = primitives(full)
        return {name: U[:, i] for i, name in enumerate(("pressure", "velocity_gas", "velocity_liquid", "void_fraction", "enthalpy_gas", "enthalpy_liquid"))}

    def stabilization_flux(self, full, raw_flux):
        cfg = self.stabilization
        result = jnp.zeros_like(raw_flux)
        if cfg.void_strength == 0 and cfg.thin_phase_strength == 0:
            return result
        U, q = primitives(full), conserved(full)
        speed = self.filter_signal_speed(full)
        third = q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3]
        if cfg.void_strength:
            face_speed = speed
            if cfg.void_velocity_cutoff is not None:
                slow = jnp.minimum(jnp.abs(U[:, 1]), jnp.abs(U[:, 2]))
                gate = 0.5*(1+jnp.tanh((cfg.void_velocity_cutoff-slow)/cfg.void_velocity_width))
                face_speed = speed*0.5*(gate[:-1]+gate[1:])
            weights = jnp.ones_like(third)
            if cfg.void_mass_cutoff is not None:
                for row, alpha in ((0, U[:, 3]), (1, 1-U[:, 3])):
                    gate = 0.5*(1+jnp.tanh((jnp.log10(alpha)-jnp.log10(cfg.void_mass_cutoff))/0.25))
                    weights = weights.at[:, row].set(gate[:-3]*gate[1:-2]*gate[2:-1]*gate[3:])
            result = result.at[1:-1].add(cfg.void_strength*face_speed[1:-1, None]*third*weights)
        if cfg.thin_phase_strength:
            weights = []
            for alpha in (U[:, 3], 1-U[:, 3]):
                loga = jnp.log10(alpha)
                gate = 0.5*(1-jnp.tanh((loga-jnp.log10(cfg.thin_phase_cutoff))/cfg.thin_phase_log_width))
                gate *= 0.5*(1+jnp.tanh((loga-jnp.log10(cfg.thin_phase_lower))/cfg.thin_phase_log_width))
                weights.append(0.5*(gate[:-1]+gate[1:]))
            wg, wl = weights
            zero = jnp.zeros_like(wg)
            rows = jnp.stack((zero, zero, wg, wl, wg, wl), axis=1)
            p = U[:, 0]
            curvature = jnp.abs(p[2:]-2*p[1:-1]+p[:-2])/(p[2:]+2*p[1:-1]+p[:-2])
            sensor = jnp.zeros_like(p).at[1:-1].set(jnp.minimum(1., 10*curvature))
            sf = jnp.maximum(sensor[:-1], sensor[1:])[1:-1, None]
            diff = q[2:-1]-q[1:-2]
            result = result.at[1:-1].add(cfg.thin_phase_strength*speed[1:-1, None]*rows[1:-1]*((1-sf)*third-cfg.thin_phase_second_order*sf*diff))
        return result

    def filter_signal_speed(self, full):
        if self.stabilization.speed_scale == "wood":
            speed = self.wave_speed(full)
        else:
            def one(U):
                eg, el = self._properties(U)
                ev = self._wave_data(U)[0]
                return jnp.abs(0.5*(ev[2]+ev[3]))+jnp.maximum(eg.a, el.a)
            speed = jax.vmap(one)(primitives(full))
        return 0.5*(speed[:-1]+speed[1:])

    def explicit_timestep(self, full):
        """Conservative filter bound and local source Jacobian rate estimate.

        The source restriction uses the infinity norm of the exact constrained
        local source linearization in fixed scaled variables. It is a local
        stiffness estimate, not a positivity theorem across closure switches.
        """
        cfg = self.stabilization
        filter_rate = jnp.asarray(0., dtype=full.dtype)
        if cfg.void_strength or cfg.thin_phase_strength:
            speed = jnp.max(self.filter_signal_speed(full))
            # Row sum bounds of first/third difference flux divergences are 4/16.
            strength = 16*cfg.void_strength + (16+4*cfg.thin_phase_second_order)*cfg.thin_phase_strength
            filter_rate = strength*speed*jnp.max(jnp.maximum(self._face_areas[:-1], self._face_areas[1:])/self._volumes)
        source_rate = jnp.asarray(0., dtype=full.dtype)
        if self.closures is not None and self.source_scale:
            # Source is cell local. Obtain its complete block Jacobian using
            # one common seed for all cells (no cross-cell dependencies).
            cells = full[1:-1]
            def density_source(C):
                state = full.at[1:-1].set(C)
                return self.integrated_source(state)/self._volumes[:, None]
            basis = jnp.eye(self.nvar, dtype=full.dtype)
            source_blocks = jax.vmap(lambda seed: jax.jvp(density_source, (cells,), (jnp.broadcast_to(seed, cells.shape),))[1])(basis)
            source_blocks = jnp.moveaxis(source_blocks, 0, -1)
            storage = jax.vmap(jax.jacfwd(lambda w: jnp.concatenate((conserved(w), self.cell_constraints(w)))))(cells)
            padded = jnp.concatenate((source_blocks, jnp.zeros((cells.shape[0], 2, 8), dtype=full.dtype)), axis=1)
            evolution = jnp.linalg.solve(storage, padded)
            scales = jnp.asarray(self.variable_scales)
            scaled = evolution*scales[None, None, :]/scales[None, :, None]
            source_rate = jnp.max(jnp.sum(jnp.abs(scaled), axis=-1))
        rate = filter_rate+source_rate
        return jnp.where(rate > 0, 1/jnp.where(rate > 0, rate, 1.), jnp.inf)
