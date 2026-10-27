"""Exact local conservative storage and wave timescales for steady PTC.

The model's spatial residual is unchanged. This adapter supplies its physical
transient Jacobian in the same scaled primitive coordinates as the direct
solvers, without including ghost-cell storage or a shock-specific model.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pysolver_view as psv


class PhysicalStorage:
    """Construct d[V*Q]/dz with the problem's fixed equation/variable scales.

    Q = (rho, rho*u, rho*(h+u**2/2)-p), w=(p,u,h), and z=w/variable_scale.
    The returned matrix is

        diag(1/residual_scale) * blockdiag(V*dQ/dw) * diag(variable_scale).

    It has units of time: multiplying by a physical shift 1/dt produces a
    matrix with the same units as the scaled spatial Jacobian. In primitive
    coordinates it is generally nonsymmetric; it is not an identity shift.
    Only the EOS density function is needed, so no perfect-gas derivative or
    one-dimensional analytic nozzle relation is assumed.
    """

    def __init__(self, problem):
        self.problem = problem
        self.variable_scale = np.array(problem.variable_scale, dtype=float, copy=True)
        self.residual_scale = np.array(problem.residual_scale, dtype=float, copy=True)
        volumes = np.asarray(problem.operators["cell_volumes"], dtype=float)[1:-1]
        if (
            self.variable_scale.shape != self.residual_scale.shape
            or self.variable_scale.ndim != 1
            or self.variable_scale.size != 3 * volumes.size
            or volumes.size == 0
        ):
            raise ValueError(
                "Storage requires three scaled primitive unknowns per interior cell"
            )
        if not all(
            np.isfinite(value).all() and np.all(value > 0.0)
            for value in (self.variable_scale, self.residual_scale, volumes)
        ):
            raise ValueError(
                "Storage volumes and fixed scales must be positive and finite"
            )
        self.baseline_cfl = float(problem.settings["CFL"])
        if not np.isfinite(self.baseline_cfl) or self.baseline_cfl <= 0:
            raise ValueError("The model CFL must be positive and finite")
        self.interior_cells = volumes.size
        fluid = problem.fluid
        variable_scale = jnp.asarray(self.variable_scale.reshape(-1, 3))
        residual_scale = jnp.asarray(self.residual_scale.reshape(-1, 3))
        volume = jnp.asarray(volumes)
        indices = 3 * jnp.arange(volumes.size)[:, None] + jnp.arange(3)[None, :]

        def conservative_cell(primitive):
            """Evaluate one cell with a batch-compatible EOS input shape."""
            pressure, velocity, enthalpy = primitive
            properties = fluid.get_state(
                jxp.HmassP_INPUTS, enthalpy[None], pressure[None]
            )
            density = properties["rho"][0]
            return jnp.stack(
                [
                    density,
                    density * velocity,
                    density * (enthalpy + 0.5 * velocity**2) - pressure,
                ]
            )

        cell_jacobian = jax.jacfwd(conservative_cell)

        @jax.jit
        def storage(point):
            primitive = point.reshape(-1, 3) * variable_scale
            conservative = jax.vmap(conservative_cell)(primitive)
            return (volume[:, None] * conservative / residual_scale).ravel()

        @jax.jit
        def matrix(point):
            primitive = point.reshape(-1, 3) * variable_scale
            physical = jax.vmap(cell_jacobian)(primitive)
            blocks = (
                volume[:, None, None]
                * physical
                * variable_scale[:, None, :]
                / residual_scale[:, :, None]
            )
            dense = jnp.zeros((point.size, point.size), dtype=point.dtype)
            return dense.at[indices[:, :, None], indices[:, None, :]].set(blocks)

        self._storage = storage
        self._matrix = matrix
        self.reset_counters()

    def matrix(self, point):
        """Return the exact dense block-diagonal storage Jacobian at point."""
        self.matrix_calls += 1
        started = perf_counter()
        try:
            result = np.asarray(
                self._matrix(jnp.asarray(self.problem._validate(point)))
            )
            if not np.isfinite(result).all():
                raise psv.DomainError("Non-finite conservative storage Jacobian")
            return result
        finally:
            self.matrix_s += perf_counter() - started

    def conservative_storage(self, point):
        """Return scaled interior V*Q, useful for independent storage audits."""
        self.storage_calls += 1
        started = perf_counter()
        try:
            result = np.asarray(
                self._storage(jnp.asarray(self.problem._validate(point)))
            )
            if not np.isfinite(result).all():
                raise psv.DomainError("Non-finite conservative storage")
            return result
        finally:
            self.storage_s += perf_counter() - started

    def wave_timescales(self, point, cfl=1.0):
        """Return local physical-cell and global earliest wave timesteps.

        The global value uses the existing model compute_dt kernel and rescales
        its configured CFL. Ghost cells are excluded by that kernel and by
        the local values here. No physical timestep is advanced.
        """
        cfl = float(cfl)
        if not np.isfinite(cfl) or cfl <= 0.0:
            raise ValueError("Requested CFL must be positive and finite")
        self.timescale_calls += 1
        started = perf_counter()
        try:
            state = self.problem.full_state(self.problem._validate(point))
            baseline_dt, sound_speed = self.problem.compute_dt(state)
            state = np.asarray(state)
            sound_speed = np.asarray(sound_speed)
            global_dt = float(baseline_dt) * cfl / self.baseline_cfl
            local_dt = (
                cfl * self.problem.dx / (np.abs(state[1:-1, 1]) + sound_speed[1:-1])
            )
            if (
                not np.isfinite(global_dt)
                or global_dt <= 0.0
                or not (np.isfinite(local_dt).all() and np.all(local_dt > 0.0))
            ):
                raise psv.DomainError(
                    "Nonpositive or non-finite characteristic timestep"
                )
            return dict(global_dt=global_dt, local_dt=local_dt, cfl=cfl)
        finally:
            self.timescale_s += perf_counter() - started

    def characteristic_dt(self, point, cfl=1.0):
        """Return the global earliest wave timestep at the requested CFL."""
        return self.wave_timescales(point, cfl)["global_dt"]

    def timestep(self, point, cfl=1.0):
        """Alias for characteristic_dt; no time integration is performed."""
        return self.characteristic_dt(point, cfl)

    def warm_up(self, point=None, cfl=1.0):
        """Compile once outside solve timing, then reset independent counters."""
        if point is None:
            point = self.problem.x0
        costs = {}
        for name, operation in (
            (
                "conservative_storage_first_call_s",
                lambda: self.conservative_storage(point),
            ),
            ("storage_matrix_first_call_s", lambda: self.matrix(point)),
            ("wave_timescale_first_call_s", lambda: self.characteristic_dt(point, cfl)),
        ):
            started = perf_counter()
            operation()
            costs[name] = perf_counter() - started
        self.reset_counters()
        return costs

    def reset_counters(self):
        """Reset storage work independently of the spatial problem's counters."""
        self.matrix_calls = self.storage_calls = self.timescale_calls = 0
        self.matrix_s = self.storage_s = self.timescale_s = 0.0

    def model_work(self):
        """Report counted and timed storage/matrix/timescale evaluations."""
        return dict(
            matrix_calls=self.matrix_calls,
            matrix_s=self.matrix_s,
            storage_calls=self.storage_calls,
            storage_s=self.storage_s,
            timescale_calls=self.timescale_calls,
            timescale_s=self.timescale_s,
        )
