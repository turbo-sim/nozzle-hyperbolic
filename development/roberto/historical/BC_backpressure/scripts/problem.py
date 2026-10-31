"""Reuse the production FVM physics with study-local outlet substitutions.

Only boundary completion and numerical wiring are defined here. Conservative
storage, area sources, reconstructed FVS residuals, inlet physics and numerical
linear/Newton engines are imported without changing their implementations.
"""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pysolver_view as psv

from development.roberto.historical.newton_solve.scripts.problem import SteadyNozzleProblem
from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver
from nozzle_hyperbolic.numerics.reconstruction import muscl_van_leer_ph
from nozzle_hyperbolic.models.euler_nozzle.flux_jax import _jax_F_fvs_stadtke
from .outlet_laws import make_outlet_law


class BoundaryProblem(SteadyNozzleProblem):
    """One common physical problem and implicit machinery for all outlet laws."""

    def __init__(self, case, *, variant="source", n_points=201, solver_settings=None):
        super().__init__(case, n_points=n_points, solver_settings=solver_settings)
        if self.settings["reconstruction"] not in ("constant", "muscl_van_leer"):
            raise ValueError(
                "This matched prototype supports constant or ordinary Van Leer"
            )
        if (
            self.settings["eps_filter"] != 0
            or self.settings.get("flux_variant", "regularized") != "regularized"
        ):
            raise ValueError("This study retains the regularized unfiltered baseline")
        self.variant = variant
        self.record_boundary_trace = False
        self.boundary_trace = []
        self.source_step = self.step
        self.source_operators = dict(self.operators)
        required = (
            "physical_residual",
            "inlet_boundary",
            "outlet_boundary",
            "solver_metadata",
        )
        missing = [name for name in required if name not in self.source_operators]
        if missing:
            raise RuntimeError(
                "This study requires the current source factory public operators: "
                + ", ".join(missing)
            )
        self.source_face_mass_flux = self.face_mass_flux
        metadata = self.source_operators["solver_metadata"]
        physical = self.source_operators["physical_residual"]
        inlet = self.source_operators["inlet_boundary"]
        outlet, self.boundary_diagnostics = make_outlet_law(
            self.source_operators["outlet_boundary"],
            self.fluid,
            self.p_back,
            variant,
        )
        n = len(self.x)

        @jax.jit
        def complete(interior):
            cells = interior.reshape(n - 2, 3)
            return outlet(
                inlet(jnp.concatenate([cells[:1], cells, cells[-1:]], axis=0))
            )

        p_lo = float(getattr(self.fluid, "p_min", 1e4)) * 1.01
        p_hi = float(getattr(self.fluid, "p_max", 300e5)) * 0.99
        h_lo = float(getattr(self.fluid, "h_min", 50e3)) + 1
        h_hi = float(getattr(self.fluid, "h_max", 850e3)) - 1

        def reconstructed_flux(full):
            """Same imported limiter/flux and EOS-domain fallback as source."""
            if self.settings["reconstruction"] == "constant":
                left, right = full[:-1], full[1:]
            else:
                left, right = muscl_van_leer_ph(full)

                def outside(cells):
                    return (
                        (cells[:, 0] < p_lo)
                        | (cells[:, 0] > p_hi)
                        | (cells[:, 2] < h_lo)
                        | (cells[:, 2] > h_hi)
                    )

                bad = (outside(left) | outside(right))[:, None]
                left = jnp.where(bad, full[:-1], left)
                right = jnp.where(bad, full[1:], right)
            return _jax_F_fvs_stadtke(
                left, right, self.fluid, self.settings["delta_factor"]
            )

        def outlet_flux_correction(full):
            """Use the sampled incoming trace directly, avoiding a second FVS split.

            Feeding only a new ghost through an averaged-characteristic FVS
            can discard a physically incoming finite compression. Only this
            final face is replaced; every interior physical row/flux is reused.
            """
            if variant != "pressure_wave":
                return jnp.zeros(3, dtype=full.dtype)
            info = self.boundary_diagnostics(full)
            mode = info["mode"]
            incoming = (mode == 3) | (mode == 4) | (mode == 5) | (mode == 7)

            def correction(_):
                pressure, velocity, enthalpy = full[-1]
                density = self.fluid.get_state(jxp.HmassP_INPUTS, enthalpy, pressure)[
                    "rho"
                ]
                flux = jnp.array(
                    [
                        density * velocity,
                        density * velocity**2 + pressure,
                        density * velocity * (enthalpy + 0.5 * velocity**2),
                    ]
                )
                return flux - reconstructed_flux(full)[-1]

            return jax.lax.cond(
                incoming,
                correction,
                lambda _: jnp.zeros(3, dtype=full.dtype),
                operand=None,
            )

        def balances(full, old, dt):
            rows = physical(full, old, dt).reshape(n - 2, 3)
            return (
                rows.at[-1]
                .add(dt * self.face_area[-1] * outlet_flux_correction(full))
                .reshape(-1)
            )

        def transient(interior, old_flat, dt):
            return balances(complete(interior), old_flat.reshape(n, 3), dt)

        @jax.jit
        def spatial(interior):
            full = complete(interior)
            return balances(full, full, jnp.float64(1.0))

        transient_blocks = make_block_operators(
            transient,
            n - 2,
            3,
            metadata["stencil_radius"],
            jacobian_mode=metadata["jacobian_mode"],
            linear_solver=metadata["linear_solver"],
        )
        spatial_blocks = make_block_operators(
            spatial,
            n - 2,
            3,
            metadata["stencil_radius"],
            jacobian_mode=metadata["jacobian_mode"],
            linear_solver=metadata["linear_solver"],
        )
        inner = make_newton_solver(
            transient,
            transient_blocks["direction"],
            metadata["newton_scale"],
            max_iterations=metadata["newton_iter"],
            tolerance=metadata["newton_tol"],
        )
        volumes = self.source_operators["cell_volumes"]
        conservative = self.source_operators["conservative_state"]

        @jax.jit
        def ssr(interior):
            full = complete(interior)
            r = spatial(interior).reshape(n - 2, 3)
            props = self.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
            tchar = self.dx / (jnp.max(jnp.abs(full[:, 1]) + props["a"]) + 1e-300)
            v = conservative(full)
            return jnp.sqrt(
                jnp.mean(
                    (tchar * r / (volumes[1:-1, None] * jnp.abs(v[1:-1]) + 1e-300)) ** 2
                )
            )

        @jax.jit
        def step(old, dt):
            interior, _ = inner(old[1:-1].reshape(-1), old.reshape(-1), dt)
            return complete(interior), ssr(interior)

        @jax.jit
        def mass_flux(full):
            full = complete(full[1:-1].reshape(-1))
            flux = reconstructed_flux(full)
            flux = flux.at[-1].add(outlet_flux_correction(full))
            return self.face_area * flux[:, 0]

        storage_scale = jnp.asarray(self.residual_scale * self.dx / self.a0)

        @jax.jit
        def checked_step(old, dt):
            interior, info = inner(old[1:-1].reshape(-1), old.reshape(-1), dt)
            defect = transient(interior, old.reshape(-1), dt) / storage_scale
            return (
                complete(interior),
                ssr(interior),
                info,
                jnp.sqrt(jnp.mean(defect**2)),
            )

        self.checked_step = checked_step
        self.step = step
        self.face_mass_flux = mass_flux
        self.operators = dict(
            self.source_operators,
            state_from_interior=complete,
            steady_residual=spatial,
            steady_jacobian=jax.jit(spatial_blocks["jacobian"]),
            steady_residual_norm=ssr,
            outlet_boundary=outlet,
            steady_newton_step=jax.jit(spatial_blocks["direction"]),
        )
        self.spatial_blocks = spatial_blocks
        self.transient_blocks = transient_blocks
        variable_scale = jnp.asarray(self.variable_scale)
        residual_scale = jnp.asarray(self.residual_scale)
        self._residual = jax.jit(lambda z: spatial(z * variable_scale) / residual_scale)
        self._jacobian = jax.jit(
            lambda z: spatial_blocks["jacobian"](z * variable_scale)
            * variable_scale[None, :]
            / residual_scale[:, None]
        )

    def _validate(self, z):
        point = np.asarray(z, dtype=float)
        if point.shape != self.x0.shape or not np.isfinite(point).all():
            raise psv.DomainError("Invalid interior coordinates")
        physical = (point * self.variable_scale).reshape(-1, 3)
        if np.any(physical[:, [0, 2]] <= 0):
            raise psv.DomainError("Pressure and enthalpy must remain positive")
        info = self.boundary_diagnostics(self.full_state(point))
        if bool(info["backflow"]):
            raise psv.DomainError(
                "This forward-outflow prototype has no exterior backflow model"
            )
        return point

    def state_is_admissible(self, full):
        full = np.asarray(full)
        if not np.isfinite(full).all() or not np.all(full[1:-1, [0, 2]] > 0):
            return False
        # Audit exactly the pack/unpack coordinates used by residual(). Near
        # zero boundary velocity, a different roundtrip can reverse its sign.
        completed = np.asarray(self.full_state(self.pack(full)))
        return bool(
            np.isfinite(completed).all()
            and np.all(completed[:, [0, 2]] > 0)
            and not self.boundary_diagnostics(jnp.asarray(completed))["backflow"]
        )

    def residual(self, point):
        result = super().residual(point)
        if self.record_boundary_trace:
            info = self.boundary_diagnostics(self.full_state(np.asarray(point)))
            self.boundary_trace.append(
                {key: np.asarray(value).item() for key, value in info.items()}
            )
        return result

    def diagnostics(self, point):
        result = super().diagnostics(np.asarray(point))
        full = self.full_state(np.asarray(point))
        info = {
            key: np.asarray(value).item()
            for key, value in self.boundary_diagnostics(full).items()
        }
        result.update(
            boundary=info,
            exit_M=float(result["fields"]["M"][-2]),
            ghost_pressure_Pa=float(result["fields"]["p"][-1]),
            boundary_pressure_relative_error=float(
                abs(result["fields"]["p"][-1] - self.p_back) / self.p_back
            ),
        )
        mach = result["fields"]["M"][1:-1]
        indices = np.flatnonzero((mach[:-1] > 1) & (mach[1:] < 1))
        result["shock_location_m"] = (
            None
            if len(indices) == 0
            else float(0.5 * (self.x[1:-1][indices[0]] + self.x[1:-1][indices[0] + 1]))
        )
        return result

    def assembly_audit(self, point):
        """Check exact source reuse, stencil coloring, and mass monitor consistency."""
        physical = jnp.asarray(np.asarray(point) * self.variable_scale)
        full = self.full_state(np.asarray(point))
        dense = np.asarray(self.spatial_blocks["dense_jacobian"](physical))
        colored = np.asarray(self.spatial_blocks["colored_jacobian"](physical))
        r = np.asarray(self.operators["steady_residual"](physical)).reshape(-1, 3)
        mdot = np.asarray(self.face_mass_flux(full))
        output = dict(
            colored_dense_relative_max=float(
                np.max(np.abs(dense - colored)) / (1 + np.max(np.abs(dense)))
            ),
            mass_flux_divergence_max=float(np.max(np.abs(np.diff(mdot) - r[:, 0]))),
            finite_jacobian=bool(np.isfinite(dense).all()),
        )
        if self.variant == "source":
            original = np.asarray(self.source_operators["steady_residual"](physical))
            source_jac = np.asarray(self.source_operators["steady_jacobian"](physical))
            output.update(
                source_residual_scaled_max=float(
                    np.max(np.abs(original - r.ravel()) / self.residual_scale)
                ),
                source_jacobian_relative_max=float(
                    np.max(np.abs(source_jac - colored))
                    / (1 + np.max(np.abs(source_jac)))
                ),
            )
            dt = self.compute_dt(full)[0] / 8
            old_step = np.asarray(self.source_step(full, dt)[0])
            new_step = np.asarray(self.step(full, dt)[0])
            output["source_step_scaled_max"] = float(
                np.max(
                    np.abs(new_step - old_step) / np.array([self.p0, self.a0, self.h0])
                )
            )
        return output
