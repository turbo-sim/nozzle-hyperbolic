"""Constrained eight-state Alberto model for the generic finite-volume engine.

The compatibility discretization is explicitly identified: frozen FVS followed
by cell X K^-1 flux-difference correction, then independently selected filters.
CompatibilityConfig(k_free=True) selects instead the K-free face flux and the
path-integrated nonconservative term (operator mode 'path').
This module contains no time integrator, Newton algorithm, or Jacobian cache.
"""

from dataclasses import asdict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid
from . import boundaries, stabilization, thermodynamics
from .constrained_state import conserved, primitives, from_primitives, eos_constraints
from .configuration import (
    BoundaryConfig,
    ClosureConfig,
    CompatibilityConfig,
    StabilizationConfig,
    PhasePinConfig,
)
from .matrices import phase_props, common_quantities, eigenstructure, flux_vector
from .flux import (
    fvs_interface_flux,
    nonconservative_correction,
    k_free_interface_flux,
    k_free_path_nonconservative,
)
from .sources import (
    two_fluid_closure,
    D_VE_algebric,
    D_VE_algebric_two_component,
    D_VE_area,
)


class TwoFluidModel:
    """Alberto's two-fluid physics adapted to the shared finite-volume engine.

    Parameters
    ----------
    mesh : Mesh
        Area/volume mesh with N physical cells and two endpoint traces.
    fluid_g, fluid_l : object
        JAXProp-compatible gas and liquid fluids.
    initial_primitives : array_like or callable, optional
        ``(p,u_g,u_l,alpha_g,h_g,h_l)`` in SI units, broadcastable to (N+2, 6),
        or a callable of mesh points. Supply exactly one initial representation.
    initial_state : array_like or callable, optional
        Constrained W states, broadcastable to (N+2, 8), or a callable of points.
    boundaries : BoundaryConfig or None, optional
        Algebraic trace laws; None selects transmissive endpoints.
    closures : ClosureConfig or None, optional
        Interphase exchange; None disables it explicitly for verification.
    compatibility : CompatibilityConfig or None, optional
        Alberto's FVS and nonconservative discretization policies.
    stabilization : StabilizationConfig or None, optional
        Optional filters and admissibility threshold; filters default to off.
    pins : PhasePinConfig or None, optional
        Equation-changing phase pins, restricted to implicit steady solution.
    virtual_mass : float, optional
        Nonnegative dimensionless added-mass coefficient; default 0.5.
    source_scale : float, optional
        Nonnegative interphase-source multiplier; does not scale area geometry.
    area_source : bool, optional
        Include the separate area source; default True.
    runtime_inputs : bool, optional
        Declare the run-time correction fields and back pressure (see
        :meth:`parameter_defaults`); default False keeps the configured
        constants in the compiled kernels unchanged.

    Notes
    -----
    The iteration state is W=(p,alpha_g,m_g,u_g,m_l,u_l,m_g*e_g,m_l*e_l).
    Six transport balances and two intrinsic-density EOS constraints determine
    eight unknowns. Fixed phase scales never include the local void fraction.
    ``constrained_state`` owns storage maps, ``matrices`` / ``flux`` the frozen
    compatibility kernels, and ``sources`` / ``closures`` interphase physics.
    Boundary, admissibility and stabilization policies have separate modules;
    Newton algorithms and time integrators remain in the generic solver.
    """

    name = "two_fluid_alberto"
    reference_revision = "faafff704e54fd866abff95fcb88fc0a5217405c"
    nvar, ntransport, nconstraints, nboundary = 8, 6, 2, 8
    nonconservative_mode = "flux_difference"
    supported_fluxes = ("model_compatibility",)
    equation_names = (
        "Gas mass",
        "Liquid mass",
        "Gas momentum",
        "Liquid momentum",
        "Gas energy",
        "Liquid energy",
    )
    constraint_names = ("Gas EOS", "Liquid EOS")
    source_dependency_radius = 0
    nonconservative_dependency_radius = 1
    boundary_dependency_radius = 2
    boundary_coupling = "local"
    equation_dependency_radius = 0
    invariant_names = ("Total mass",)
    invariant_projections = np.array(((1.0, 1.0, 0.0, 0.0, 0.0, 0.0),))

    def __init__(
        self,
        mesh,
        fluid_g,
        fluid_l,
        *,
        initial_primitives=None,
        initial_state=None,
        boundaries=None,
        closures=None,
        compatibility=None,
        stabilization=None,
        pins=None,
        virtual_mass=0.5,
        source_scale=1.0,
        area_source=True,
        runtime_inputs=False
    ):
        self.mesh = mesh
        self.fluid_g, self.fluid_l = as_jax_fluid(fluid_g), as_jax_fluid(fluid_l)
        self.boundaries = boundaries or BoundaryConfig()
        self.closures = closures
        self.compatibility = compatibility or CompatibilityConfig()
        self.stabilization = stabilization or StabilizationConfig()
        # k_free replaces the flux-difference correction by path integrals of
        # the reconstructed face states, supplied by the operator's 'path' mode.
        self.nonconservative_mode = (
            "path" if self.compatibility.k_free else "flux_difference"
        )
        if self.compatibility.k_free and self.stabilization.mass_upwind_cutoff is not None:
            raise ValueError("k_free is not defined with mass_upwind_cutoff")
        self.pins = pins or PhasePinConfig()
        self.has_equation_pins = self.pins.saturation_energy or self.pins.momentum
        self.steady_only = self.has_equation_pins
        self.virtual_mass = float(virtual_mass)
        self.source_scale = float(source_scale)
        self.area_source = bool(area_source)
        self.runtime_inputs = bool(runtime_inputs)
        if not np.isfinite(self.virtual_mass) or self.virtual_mass < 0:
            raise ValueError("virtual_mass must be finite and nonnegative")
        if not np.isfinite(self.source_scale) or self.source_scale < 0:
            raise ValueError("source_scale must be finite and nonnegative")
        if mesh.area_centers is None:
            raise ValueError("Two-fluid geometry requires area_centers")
        if (initial_state is None) == (initial_primitives is None):
            raise ValueError(
                "Supply exactly one of initial_primitives or initial_state"
            )
        if initial_state is None:
            U = (
                initial_primitives(mesh.points)
                if callable(initial_primitives)
                else initial_primitives
            )
            U = np.broadcast_to(
                np.asarray(U, dtype=float), (mesh.n_cells + 2, 6)
            ).copy()
            if np.any(
                (U[:, 3] <= self.stabilization.alpha_min)
                | (U[:, 3] >= 1 - self.stabilization.alpha_min)
            ):
                raise ValueError(
                    "Initial state violates open phase-fraction bounds; pure phases require a reduced model"
                )
            self._initial = from_primitives(U, self.fluid_g, self.fluid_l)
        else:
            W = initial_state(mesh.points) if callable(initial_state) else initial_state
            self._initial = jnp.asarray(
                np.broadcast_to(
                    np.asarray(W, dtype=float), (mesh.n_cells + 2, 8)
                ).copy()
            )
            alpha = np.asarray(self._initial[:, 1])
            if np.any(
                (alpha <= self.stabilization.alpha_min)
                | (alpha >= 1 - self.stabilization.alpha_min)
            ):
                raise ValueError(
                    "Initial state violates open phase-fraction bounds; pure phases require a reduced model"
                )
        U0 = np.asarray(primitives(self._initial))[0]
        pg, pl = self._properties(jnp.asarray(U0))
        rg, rl = float(pg.rho), float(pl.rho)
        sound = max(float(pg.a), float(pl.a))
        pressure = float(U0[0])
        if min(pressure, rg, rl, sound) <= 0 or not np.all(
            np.isfinite([pressure, rg, rl, sound])
        ):
            raise ValueError(
                "Initial thermodynamic reference must be finite and physical"
            )
        self.variable_scales = np.array(
            (pressure, 1.0, rg, sound, rl, sound, rg * sound**2, rl * sound**2)
        )
        self.conservative_scales = np.array(
            (rg, rl, rg * sound, rl * sound, rg * sound**2, rl * sound**2)
        )
        self.constraint_scales = np.array((rg, rl))
        self.primitive_scales = jnp.array(
            (pressure, sound, sound, 1.0, sound**2, sound**2)
        )
        self.reconstruction_scales = np.asarray(self.primitive_scales)
        self.entropy_scale = sound**2 / max(float(pg.T), float(pl.T))
        self.reference_time = float(mesh.faces[-1] - mesh.faces[0]) / sound
        self.global_flux_scales = mesh.area_faces[0] * sound * self.conservative_scales
        self.invariant_flux_scales = np.array(
            (self.global_flux_scales[0] + self.global_flux_scales[1],)
        )
        self.stabilization_dependency_radius = (
            2
            if (
                self.stabilization.void_strength
                or self.stabilization.thin_phase_strength
            )
            else 0
        )
        self._delta_area = jnp.asarray(np.diff(mesh.area_faces))
        self._volumes = jnp.asarray(mesh.volumes)
        self._areas = jnp.asarray(mesh.area_centers)
        self._face_areas = jnp.asarray(mesh.area_faces)
        self._phase_stencil_masks = tuple(
            thermodynamics.table_stencil_mask(
                fluid, closures is not None and self.source_scale != 0
            )
            for fluid in (self.fluid_g, self.fluid_l)
        )
        self._left_factor = float(
            (mesh.faces[0] - mesh.centers[0]) / (mesh.centers[0] - mesh.centers[1])
        )
        if self._needs_saturation:
            for fluid in (self.fluid_g, self.fluid_l):
                if not hasattr(fluid, "get_state_saturation"):
                    raise ValueError(
                        "Selected closures require explicit phase saturation tables"
                    )
                if hasattr(fluid, "table") and not fluid.table.get("saturation_props"):
                    raise ValueError(
                        "Selected closures require populated phase saturation tables"
                    )
        if not bool(self.cell_admissible(self._initial)):
            raise ValueError(
                "Initial state violates phase, PH-domain or hyperbolicity checks"
            )

    @property
    def _needs_saturation(self):
        """Return whether selected mass/heat exchange or phase pins read saturation data."""
        c = self.closures
        return self.pins.saturation_energy or (
            c is not None
            and (
                c.mass_transfer
                or c.htc_liquid_bubbly != "ranz_marshall"
                or c.htc_liquid_droplet != "internal"
            )
        )

    def validate_scheme(self, scheme):
        """Validate the flux selection and implicit-only equation-pin contract."""
        if scheme.flux not in self.supported_fluxes:
            raise ValueError(
                "Two-fluid compatibility discretization requires flux='model_compatibility'"
            )
        if self.has_equation_pins and scheme.time_integrator != "euler-implicit":
            raise ValueError(
                "Equation-changing two-fluid pins require implicit steady solution"
            )

    def transform_cell_equations(self, full, integrated_rows, dt=None):
        """Delegate to :func:`stabilization.transform_cell_equations`."""
        return stabilization.transform_cell_equations(self, full, integrated_rows, dt)

    def _properties(self, U):
        """Evaluate gas and liquid properties for one enthalpy-primitive state."""
        return phase_props(self.fluid_g, U[0], U[4]), phase_props(
            self.fluid_l, U[0], U[5]
        )

    def _wave_data(self, U):
        """Evaluate the six-wave eigenstructure for one enthalpy-primitive state."""
        eg, el = self._properties(U)
        cq = common_quantities(U, eg, el, self.virtual_mass)
        return eigenstructure(U, eg, el, cq)

    def initial_state(self):
        """Return a copy of the configured initial cells and endpoint traces.

        Returns
        -------
        jax.Array, shape (N+2, 8)
            Constrained iteration states, including both algebraic traces.
        """
        return jnp.array(self._initial)

    cell_conserved = staticmethod(conserved)
    face_conserved = staticmethod(conserved)
    filter_conserved = staticmethod(conserved)

    def cell_constraints(self, W):
        """Evaluate the two intrinsic-density EOS defects.

        Parameters
        ----------
        W : jax.Array, shape (..., 8)
            Constrained states in SI units.

        Returns
        -------
        jax.Array, shape (..., 2)
            Gas/liquid density defects in kg/m**3, before fixed scaling.
        """
        return eos_constraints(W, self.fluid_g, self.fluid_l)

    reconstruction_state = staticmethod(primitives)

    def from_reconstruction_state(self, U):
        """Recover EOS-consistent constrained states from enthalpy primitives.

        Parameters
        ----------
        U : jax.Array, shape (..., 6)
            Enthalpy primitives in SI units.

        Returns
        -------
        jax.Array, shape (..., 8)
            Constrained states populated using both configured phase EOSs.
        """
        return from_primitives(U, self.fluid_g, self.fluid_l)

    def face_physical_flux(self, W):
        """Evaluate phase flux density before applying geometry measures.

        Parameters
        ----------
        W : jax.Array, shape (P, 8)
            Constrained states at P cells or faces.

        Returns
        -------
        jax.Array, shape (P, 6)
            Phase mass, momentum and total-energy fluxes.
        """

        def one(U):
            eg, el = self._properties(U)
            return flux_vector(U, eg, el)

        return jax.vmap(one)(primitives(W))

    def face_speed_bounds(self, W):
        """Evaluate the minimum and maximum speeds of all six wave families.

        Parameters
        ----------
        W : jax.Array, shape (P, 8)
            Constrained states.

        Returns
        -------
        lower, upper : jax.Array, shape (P,)
            Signed wave-speed bounds in m/s.
        """
        ev = jax.vmap(lambda U: self._wave_data(U)[0])(primitives(W))
        return jnp.min(ev, axis=-1), jnp.max(ev, axis=-1)

    def wave_speed(self, W):
        """Return the maximum absolute characteristic speed.

        Parameters
        ----------
        W : jax.Array, shape (P, 8)
            Constrained states.

        Returns
        -------
        jax.Array, shape (P,)
            Speed magnitude in m/s.
        """
        lower, upper = self.face_speed_bounds(W)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def boundary_wave_speed(self, full):
        """Return maximum wave speeds at the two endpoint traces, in m/s."""
        return self.wave_speed(full[jnp.array((0, full.shape[0] - 1))])

    @staticmethod
    def flux_to_conserved_frame(flux):
        """Return the flux unchanged: compatibility kernels already use conserved ordering."""
        return flux

    def numerical_face_flux(self, left, right, scheme):
        """Assemble one scalar-branch compatibility flux per reconstructed face.

        Parameters
        ----------
        left, right : jax.Array, shape (N+1, 8)
            Reconstructed constrained face states.
        scheme : Scheme
            Must select ``flux='model_compatibility'``.

        Returns
        -------
        jax.Array, shape (N+1, 6)
            Numerical flux density, including optional phase-mass upwind blending.

        Notes
        -----
        Scalar lax.map preserves the safe equal-slip branch under AD. Each
        configured coefficient is passed explicitly to the low-level FVS kernel.
        ``k_free`` selects :func:`k_free_interface_flux` instead of
        :func:`fvs_interface_flux`; ``smooth`` is passed to either kernel.
        """
        if scheme.flux != "model_compatibility":
            raise ValueError(
                "TwoFluidModel supports only the explicit model_compatibility flux"
            )
        cfg = self.compatibility

        def one(pair):
            ul, ur, dissipation = pair
            if cfg.k_free:
                value = k_free_interface_flux(
                    ul,
                    ur,
                    self.fluid_g,
                    self.fluid_l,
                    self.virtual_mass,
                    delta_factor=cfg.delta_factor,
                    lowmach_diss=dissipation,
                    du_blend_lo=cfg.du_blend_lo,
                    du_blend_hi=cfg.du_blend_hi,
                    smooth=cfg.smooth,
                    smooth_width=cfg.smooth_width,
                )
            else:
                value = fvs_interface_flux(
                    ul,
                    ur,
                    self.fluid_g,
                    self.fluid_l,
                    self.virtual_mass,
                    delta_factor=cfg.delta_factor,
                    lowmach_diss=dissipation,
                    du_blend_lo=cfg.du_blend_lo,
                    du_blend_hi=cfg.du_blend_hi,
                    single_branch=False,
                    smooth=cfg.smooth,
                    smooth_width=cfg.smooth_width,
                )
            if self.stabilization.mass_upwind_cutoff is not None:
                egL, elL = self._properties(ul)
                egR, elR = self._properties(ur)
                for row, ui, aL, aR, rhoL, rhoR in (
                    (0, 1, ul[3], ur[3], egL.rho, egR.rho),
                    (1, 2, 1 - ul[3], 1 - ur[3], elL.rho, elR.rho),
                ):
                    cutoff = jnp.log10(self.stabilization.mass_upwind_cutoff)
                    wL = 0.5 * (1 - jnp.tanh((jnp.log10(aL) - cutoff) / 0.25))
                    wR = 0.5 * (1 - jnp.tanh((jnp.log10(aR) - cutoff) / 0.25))
                    weight = 1 - (1 - wL) * (1 - wR)
                    sign = jnp.tanh(0.5 * (ul[ui] + ur[ui]) / 1e-3)
                    up = (
                        0.5 * (1 + sign) * aL * rhoL * ul[ui]
                        + 0.5 * (1 - sign) * aR * rhoR * ur[ui]
                    )
                    value = value.at[row].set(weight * up + (1 - weight) * value[row])
            return value

        # lax.map preserves scalar branch execution at equal slip; vmapped
        # conditionals evaluate the singular inactive full-slip decomposition.
        dissipation = jnp.full((left.shape[0],), cfg.lowmach_diss)
        if cfg.lowmach_diss_boundary is not None:
            dissipation = (
                dissipation.at[0]
                .set(cfg.lowmach_diss_boundary)
                .at[-1]
                .set(cfg.lowmach_diss_boundary)
            )
        return jax.lax.map(one, (primitives(left), primitives(right), dissipation))

    def integrated_nonconservative(self, full, unfiltered_flux, faces=None):
        """Assemble cell-local X K-inverse actions from selected flux differences.

        Parameters
        ----------
        full : jax.Array, shape (N+2, 8)
            Physical cells and endpoint traces.
        unfiltered_flux : jax.Array, shape (N+1, 6)
            Compatibility flux density, before optional filter flux.
        faces : tuple of jax.Array, shape (N+1, 8) each, optional
            Reconstructed left/right face states; required by ``k_free``.

        Returns
        -------
        jax.Array, shape (N, 6)
            Integrated nonconservative correction in conservative ordering.

        Notes
        -----
        The selected nc_form controls geometry measures. The well-balanced
        'wb' choice subtracts the cell physical flux times exact area change.

        With ``k_free`` (operator mode 'path') nc_form is not used: each face
        jump path integral goes half to each neighbour, weighted by its face
        area, plus the cell area times the within-cell path from the cell's
        left-face to right-face state (zero length under constant
        reconstruction). Uniform states give exactly zero; no K solve.
        """
        cfg = self.compatibility
        if cfg.k_free:
            if faces is None:
                raise ValueError(
                    "k_free requires the reconstructed faces of nonconservative_mode='path'"
                )
            left, right = primitives(faces[0]), primitives(faces[1])

            def path(pair):
                ua, ub = pair
                return k_free_path_nonconservative(
                    ua, ub, self.fluid_g, self.fluid_l, self.virtual_mass
                )

            jumps = jax.lax.map(path, (left, right))
            inside = jax.lax.map(path, (right[:-1], left[1:]))
            return (
                0.5 * self._face_areas[1:, None] * jumps[1:]
                + 0.5 * self._face_areas[:-1, None] * jumps[:-1]
                + self._areas[:, None] * inside
            )

        U = primitives(full[1:-1])
        if cfg.nc_form == "cell":
            delta = self._areas[:, None] * (unfiltered_flux[1:] - unfiltered_flux[:-1])
        else:
            delta = (
                self._face_areas[1:, None] * unfiltered_flux[1:]
                - self._face_areas[:-1, None] * unfiltered_flux[:-1]
            )
            if cfg.nc_form == "wb":
                delta = delta - self._delta_area[:, None] * self.face_physical_flux(
                    full[1:-1]
                )

        def one(args):
            u, df = args
            eg, el = self._properties(u)
            cq = common_quantities(u, eg, el, self.virtual_mass)
            return nonconservative_correction(u, df, eg, el, cq)

        return jax.lax.map(one, (U, delta))

    def parameter_defaults(self):
        """Declare the run-time correction fields and outlet pressure.

        Returns
        -------
        dict of str to jax.Array
            Empty unless ``runtime_inputs`` is set. Then, with closures:
            per-cell ``beta_ai``, ``beta_cd``, ``beta_htc_g`` and ``beta_htc_l``
            fields of shape (N,), all ones; with a pressure outlet and numeric
            ``back_pressure``: the scalar ``back_pressure`` in Pa.

        Notes
        -----
        Each field multiplies the corresponding ``ClosureConfig`` factor in its
        cell; the run-time ``back_pressure`` replaces the configured value. The
        defaults reproduce the configuration to roundoff (the factors become
        kernel inputs instead of constants). New values reuse the compiled
        kernels; see the operator's run-time parameter contract.
        """
        values = {}
        if not self.runtime_inputs:
            return values
        if self.closures is not None:
            ones = jnp.ones(self.mesh.n_cells)
            values.update(beta_ai=ones, beta_cd=ones, beta_htc_g=ones, beta_htc_l=ones)
        bc = self.boundaries
        if bc.outlet != "transmissive" and not callable(bc.back_pressure):
            values["back_pressure"] = jnp.asarray(float(bc.back_pressure))
        return values

    def validate_parameters(self, values):
        """Require positive correction fields and a positive back pressure.

        Parameters
        ----------
        values : dict of str to numpy.ndarray
            Complete run-time parameter mapping with defaults filled in.

        Raises
        ------
        ValueError
            If any correction factor or the back pressure is not positive.
        """
        for name, value in values.items():
            if np.any(np.asarray(value) <= 0):
                raise ValueError(f"Run-time parameter {name} must be positive")

    def integrated_source(self, full, params=None):
        """Integrate interphase exchange and exact face-area source measures.

        Parameters
        ----------
        full : jax.Array, shape (N+2, 8)
            Physical cells and endpoint traces.
        params : dict or None, optional
            Run-time parameters; the ``beta_*`` fields scale the closure factors
            in each cell. None uses the configured closure unchanged.

        Returns
        -------
        jax.Array, shape (N, 6)
            Integrated phase sources.

        Notes
        -----
        Exchange density receives source_scale and cell volume. The separate
        area kernel receives A_right-A_left directly and needs no extra volume factor.
        """
        closure_options = {} if self.closures is None else asdict(self.closures)
        fields = ("beta_ai", "beta_cd", "beta_htc_g", "beta_htc_l")
        corrected = params is not None and self.closures is not None
        factors = (
            jnp.stack([params[name] for name in fields], axis=1)
            if corrected
            else jnp.ones((self.mesh.n_cells, len(fields)))
        )

        def one(args):
            u, volume, delta_area, factor = args
            eg, el = self._properties(u)
            cq = common_quantities(u, eg, el, self.virtual_mass)
            value = jnp.zeros(6, dtype=u.dtype)
            if self.closures is not None and self.source_scale != 0:
                if self._needs_saturation:
                    sg = self.fluid_g.get_state_saturation(u[0])
                    sl = self.fluid_l.get_state_saturation(u[0])
                    Ts, hgs, hls = sg["temperature"], sg["enthalpy"], sl["enthalpy"]
                else:
                    Ts, hgs, hls = 0.0, 0.0, 0.0
                options = closure_options
                if corrected:
                    # Per-cell fields multiply the configured correction factors.
                    options = closure_options | {
                        name: closure_options[name] * factor[i]
                        for i, name in enumerate(fields)
                    }
                closure = two_fluid_closure(
                    u,
                    eg,
                    el,
                    self.fluid_g,
                    self.fluid_l,
                    Ts,
                    hgs,
                    hls,
                    **options
                )
                assembly = (
                    D_VE_algebric
                    if self.closures.mass_transfer
                    else D_VE_algebric_two_component
                )
                value = value + self.source_scale * volume * assembly(
                    u, cq, closure, eg, el, self.virtual_mass
                )
            if self.area_source:
                # Exact integrated area change pairs the wall pressure with the
                # same face measures used by the generic flux divergence.
                value = value + D_VE_area(
                    u, cq, SimpleNamespace(dlnA_dx=delta_area), eg, el
                )
            return value

        return jax.lax.map(
            one, (primitives(full[1:-1]), self._volumes, self._delta_area, factors)
        )

    def boundary_residual(self, full, time, params=None):
        """Delegate to :func:`boundaries.boundary_residual`."""
        return boundaries.boundary_residual(self, full, time, params)

    def cell_admissible(self, W):
        """Delegate to :func:`thermodynamics.cell_admissible`."""
        return thermodynamics.cell_admissible(self, W)

    def boundary_admissible(self, full):
        """Require physical admissibility of both algebraic endpoint traces."""
        return self.cell_admissible(full[jnp.array((0, full.shape[0] - 1))])

    def admissible(self, full, left, right):
        """Require physical cells, traces and both reconstructed face states to be admissible."""
        return (
            self.cell_admissible(full)
            & self.cell_admissible(left)
            & self.cell_admissible(right)
        )

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

    def acoustic_diagnostics(self, states):
        """Evaluate mixture kinematics from the model's two acoustic families.

        Parameters
        ----------
        states : jax.Array, shape (P, 8)
            Admissible iteration states at any P cells or endpoint traces.

        Returns
        -------
        dict
            ``velocity`` is the acoustic-pair center in m/s; ``sound_speed``
            is its positive half-width in m/s; ``mach`` is the absolute ratio.
            These are the compatibility model's mixture definitions, evaluated
            through the configured phase EOSs, rather than a perfect-gas law.
        """
        eigenvalues = jax.vmap(lambda u: self._wave_data(u)[0])(primitives(states))
        velocity = 0.5 * (eigenvalues[:, 2] + eigenvalues[:, 3])
        sound = 0.5 * (eigenvalues[:, 2] - eigenvalues[:, 3])
        return dict(
            velocity=velocity, sound_speed=sound, mach=jnp.abs(velocity) / sound
        )

    def fields(self, full):
        """Expose enthalpy primitives for plotting and reporting.

        Parameters
        ----------
        full : jax.Array, shape (N+2, 8)
            Physical cells and endpoint traces.

        Returns
        -------
        dict of str to jax.Array
            Pressure [Pa], phase velocities [m/s], gas fraction and phase enthalpies
            [J/kg], each including both endpoint traces.
        """
        U = primitives(full)
        return {
            name: U[:, i]
            for i, name in enumerate(
                (
                    "pressure",
                    "velocity_gas",
                    "velocity_liquid",
                    "void_fraction",
                    "enthalpy_gas",
                    "enthalpy_liquid",
                )
            )
        }

    def stabilization_flux(self, full, raw_flux):
        """Delegate to :func:`stabilization.stabilization_flux`."""
        return stabilization.stabilization_flux(self, full, raw_flux)

    def filter_signal_speed(self, full):
        """Delegate to :func:`stabilization.filter_signal_speed`."""
        return stabilization.filter_signal_speed(self, full)

    def explicit_timestep(self, full, params=None):
        """Delegate to :func:`stabilization.explicit_timestep`."""
        return stabilization.explicit_timestep(self, full, params)
