"""One shared finite-volume steady problem for all three solver strategies."""

from time import perf_counter

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pysolver_view as psv

from development.roberto.euler_nozzle import verify_euler_nozzle_steady as baseline
from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
from nozzle_hyperbolic.models.euler_nozzle.analytical.nozzle_exact_real_gas import (
    nozzle_exact_real_gas,
)
from nozzle_hyperbolic.models.euler_nozzle.fvm import make_jax_stadtke_step
from nozzle_hyperbolic.models.euler_nozzle.initialization import (
    stadtke_puh_initialization_simple,
)


class SteadyNozzleProblem:
    """Scaled interior (p,u,h) unknowns with the production spatial residual.

    Variables are scaled by reservoir pressure, sound speed, and enthalpy.
    Each integrated balance is scaled by V_cell*q_reference/(dx/a0).
    These scales are FIXED for a run, so all methods share the same root and
    RMS residual. The existing state-dependent SSR is reported separately.
    """

    def __init__(self, case, n_points=baseline.N_POINTS, *, solver_settings=None):
        """Build the reference, initial field, and opt-in production operators."""
        self.case = dict(case)
        self.settings = dict(baseline.SOLVER)
        if solver_settings is not None:
            self.settings.update(solver_settings)
        self.fluid = jxp.FluidPerfectGas(name="Air")
        self.x = np.linspace(0.0, baseline.NOZZLE["L"], n_points)
        self.dx = float(self.x[1] - self.x[0])
        self.geometry = get_symmetric_nozzle(self.x, **baseline.NOZZLE)
        self.area = np.asarray(self.geometry.A)
        self.face_area = 0.5 * (self.area[:-1] + self.area[1:])
        # Match the production driver's face-consistent pressure-area source.
        dAdx = np.asarray(self.geometry.dAdx).copy()
        dAdx[1:-1] = np.diff(self.face_area) / self.dx
        self.p0 = baseline.P_STAGNATION
        self.T0 = baseline.T_STAGNATION
        stagnation = self.fluid.get_state(jxp.PT_INPUTS, self.p0, self.T0)
        self.h0 = float(stagnation["h"])
        self.s0 = float(stagnation["s"])
        self.rho0 = float(stagnation["rho"])
        self.a0 = float(stagnation["a"])

        x_fine = np.linspace(self.x[0], self.x[-1], baseline.N_EXACT)
        area_fine = np.asarray(get_symmetric_nozzle(x_fine, **baseline.NOZZLE).A)
        if case["branch"] == "subsonic":
            fine = baseline.subsonic_reference(
                area_fine, self.fluid, case["throat_mach"]
            )
            self.reference = baseline.subsonic_reference(
                self.area, self.fluid, case["throat_mach"]
            )
            self.p_back = float(fine["p"][-1])
        else:
            fraction = case.get("shock_fraction")
            throat = 0.5 * baseline.NOZZLE["L"]
            shock = (
                None if fraction is None else throat + fraction * (self.x[-1] - throat)
            )
            args = dict(
                fluid=self.fluid,
                p0=self.p0,
                T0=self.T0,
                x_shock=shock,
                A_throat=baseline.NOZZLE["A_throat"],
            )
            fine = nozzle_exact_real_gas(x_fine, area_fine, **args)[case["branch"]]
            self.reference = nozzle_exact_real_gas(self.x, self.area, **args)[
                case["branch"]
            ]
            self.p_back = float(fine["p"][-1])
        self.x_fine = x_fine
        self.reference_fine = fine

        s = self.settings
        self.step, self.compute_dt, _, self.face_mass_flux, self.operators = (
            make_jax_stadtke_step(
                self.fluid,
                self.area,
                self.face_area,
                dAdx,
                self.geometry.D_h,
                self.geometry.roughness,
                self.dx,
                s["CFL"],
                self.p_back,
                self.h0,
                self.s0,
                delta_factor=s["delta_factor"],
                newton_iter=s["newton_iter"],
                wall_friction=False,
                recon=s["reconstruction"],
                jacobian_mode=s["jacobian"],
                eps_filter=s["eps_filter"],
                linear_solver=s["linear_solver"],
                newton_tol=s["newton_tol"],
                flux_local_jac=s["flux_local_jac"],
                cell_local_jac=s["cell_local_jac"],
                return_operators=True,
                flux_variant=s.get("flux_variant", "regularized"),
                smooth_limiter_coefficient=s.get("smooth_limiter_coefficient", 1.0),
                limiter_scales=[self.p0, self.a0, self.h0],
            )
        )
        params = dict(
            fluid=self.fluid,
            fluid_name="Air",
            p_out=self.p_back,
            pT_stagnation=np.array([self.p0, self.T0]),
        )
        # Same simple initialization rule for every method, with no exact seed.
        self.initial_state = jnp.asarray(
            stadtke_puh_initialization_simple(self.x, self.area, params)
        )
        self.variable_scale = np.tile([self.p0, self.a0, self.h0], n_points - 2)
        storage_scale = self.rho0 * np.array([1.0, self.a0, self.h0])
        volumes = np.asarray(self.operators["cell_volumes"])[1:-1]
        self.residual_scale = (
            volumes[:, None] * storage_scale[None, :] * self.a0 / self.dx
        ).ravel()
        variable_scale = jnp.asarray(self.variable_scale)
        residual_scale = jnp.asarray(self.residual_scale)

        @jax.jit
        def scaled_residual(z):
            return (
                self.operators["steady_residual"](z * variable_scale) / residual_scale
            )

        @jax.jit
        def scaled_jacobian(z):
            jac = self.operators["steady_jacobian"](z * variable_scale)
            return jac * variable_scale[None, :] / residual_scale[:, None]

        self._residual = scaled_residual
        self._jacobian = scaled_jacobian
        self.x0 = self.pack(self.initial_state)
        self.reset_counters()

    def pack(self, state):
        """Convert full physical primitive state to scaled interior unknowns."""
        return np.asarray(state)[1:-1].ravel() / self.variable_scale

    def full_state(self, z):
        """Restore physical primitives and apply the common model boundaries."""
        return self.operators["state_from_interior"](
            jnp.asarray(z * self.variable_scale)
        )

    def _validate(self, z):
        """Reject undefined EOS/forward-flow trials through PSV's domain API."""
        z = np.asarray(z, dtype=float)
        if z.shape != self.x0.shape:
            raise ValueError("Incorrect interior-state shape")
        physical = (z * self.variable_scale).reshape(-1, 3)
        if not np.isfinite(physical).all() or np.any(physical <= 0.0):
            # This experiment targets forward nozzle flow. The legacy flux is
            # singular at rest; the regularized flux also supports stationary
            # states, but this problem's chosen forward branch remains positive.
            # No artificial residual is supplied for invalid trial states.
            raise psv.DomainError("Trial p, u, or h is non-positive/non-finite")
        return z

    def residual(self, z):
        """Return the common dimensionless spatial residual, shape (n,)."""
        self.residual_calls += 1
        start = perf_counter()
        try:
            result = np.asarray(self._residual(jnp.asarray(self._validate(z))))
            if not np.isfinite(result).all():
                raise psv.DomainError("Non-finite spatial residual")
            return result
        finally:
            self.residual_s += perf_counter() - start

    def gradient(self, z):
        """Return the exact AD steady Jacobian, shape (n,n), for PSV."""
        self.jacobian_calls += 1
        start = perf_counter()
        try:
            result = np.asarray(self._jacobian(jnp.asarray(self._validate(z))))
            if not np.isfinite(result).all():
                raise psv.DomainError("Non-finite spatial Jacobian")
            return result
        finally:
            self.jacobian_s += perf_counter() - start

    def reset_counters(self):
        """Start independent model-work counters for a measured solve."""
        self.residual_calls = self.jacobian_calls = 0
        self.residual_s = self.jacobian_s = 0.0

    def model_work(self):
        """Return explicitly timed model calls, excluding final diagnostics."""
        return dict(
            residual_calls=self.residual_calls,
            jacobian_calls=self.jacobian_calls,
            residual_s=self.residual_s,
            jacobian_s=self.jacobian_s,
        )

    def warm_up(self):
        """Compile once without using the warm-up update as an initial guess."""
        costs = {}
        for name, operation in (
            ("steady_residual_first_call_s", lambda: self.residual(self.x0)),
            ("steady_jacobian_first_call_s", lambda: self.gradient(self.x0)),
            ("boundary_state_first_call_s", lambda: self.full_state(self.x0)),
            (
                "implicit_step_first_call_s",
                lambda: self.step(
                    self.initial_state, self.compute_dt(self.initial_state)[0]
                ),
            ),
            (
                "existing_ssr_first_call_s",
                lambda: self.operators["steady_residual_norm"](
                    jnp.asarray(self.x0 * self.variable_scale)
                ),
            ),
            ("face_flux_first_call_s", lambda: self.face_mass_flux(self.initial_state)),
        ):
            started = perf_counter()
            jax.block_until_ready(operation())
            costs[name] = perf_counter() - started
        self.reset_counters()
        return costs

    def diagnostics(self, z):
        """Independently audit root residual, original SSR, flow, and reference."""
        residual = self.residual(z)
        state = self.full_state(np.asarray(z))
        props = self.fluid.get_state(jxp.HmassP_INPUTS, state[:, 2], state[:, 0])
        fields = dict(
            p=np.asarray(state[:, 0]),
            rho=np.asarray(props["rho"]),
            M=np.asarray(state[:, 1] / props["a"]),
        )
        scales = dict(p=self.p0, rho=self.rho0, M=1.0)
        error = {
            key: float(
                np.mean(np.abs(fields[key][1:-1] - self.reference[key][1:-1]))
                / scales[key]
            )
            for key in fields
        }
        flux = np.asarray(self.face_mass_flux(state))
        return dict(
            residual_rms=float(np.linalg.norm(residual) / np.sqrt(residual.size)),
            residual_inf=float(np.max(np.abs(residual))),
            existing_ssr=float(
                self.operators["steady_residual_norm"](
                    jnp.asarray(z * self.variable_scale)
                )
            ),
            normalized_L1=error,
            analytical_pass=max(error.values()) < baseline.MAX_NORMALIZED_L1,
            mass_imbalance=float(abs(flux[0] - flux[-1]) / abs(flux[0])),
            throat_M_numerical=float(fields["M"][np.argmin(self.area)]),
            fields=fields,
        )
