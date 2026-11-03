"""Current shared-FVM bindings for the nozzle sensitivity clients.

Physical data come from the installed verification catalogue. Numerical choices
belong to these studies; no retired example or solver factory is imported.
"""

from dataclasses import asdict
from collections import OrderedDict
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import (
    PerfectGasReference,
)
from nozzle_hyperbolic.solvers.fvm import (
    ResidualPolicy,
    Scheme,
    SteadySettings,
    SteadyWorkspace,
    solve_single_grid,
)
from nozzle_hyperbolic.verification import build_case

N_POINTS = 201
RMS_TOL = 1e-8
SCHEMA = "shared_fvm_sensitivity_v1"
CASE_IDS = dict(
    subsonic="subsonic",
    sonic="sonic_limit",
    shock35="shock_35",
    shock70="shock_70",
    supersonic="adapted",
)


def residual_policy():
    """Require every spatial equation, trace and balance to pass independently."""
    return ResidualPolicy(
        criterion="absolute",
        absolute_tolerance=RMS_TOL,
        boundary_tolerance=RMS_TOL,
        global_tolerance=RMS_TOL,
    )


def scheme(newton_iterations=10):
    """Use the current corrected Stadtke flux and strict coupled backward Euler."""
    return Scheme(
        reconstruction="muscl_van_leer",
        delta_factor=1e-6,
        newton_iterations=newton_iterations,
        newton_update_tolerance=1e-8,
        newton_residual_tolerance=1e-10,
    )


class ImplicitNozzle:
    """Bind current operators by pressure; keep compilation outside warm solves.

    ``n_points`` includes two zero-storage endpoint traces. Interior centers
    and integrated geometry follow the current finite-volume mesh, rather than
    the former uniform nodal discretization. Every pressure has its own immutable
    model/workspace; changing a captured boundary value in place is forbidden.
    """

    def __init__(self, n_points=N_POINTS, *, numerical_scheme=None):
        if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 4:
            raise ValueError("Use at least four points including two endpoint traces")
        self.problem = build_case("euler_nozzle/subsonic", cells=n_points - 2)
        model = self.problem.model
        self.fluid, self.mesh = model.fluid, model.mesh
        self.x = np.asarray(self.mesh.points)
        self.area = self.mesh.geometry.sample(self.x)["area"]
        self.h0, self.s0, self.a0, self.rho0 = model.h0, model.s0, model.a0, model.rho0
        self.scheme = scheme() if numerical_scheme is None else numerical_scheme
        self.reference = PerfectGasReference(
            self.fluid,
            p0=model.p0,
            T0=model.T0,
            length=self.mesh.geometry.length,
            inlet_area=self.mesh.geometry.inlet_area,
            throat_area=self.mesh.geometry.throat_area,
        )
        self.settings = dict(
            schema=SCHEMA,
            scheme=asdict(self.scheme),
            outlet="automatic acoustic outflow",
            cells=n_points - 2,
            physical_case=self.problem.definition.to_dict(),
        )
        self._workspaces = OrderedDict()

    def workspace(self, pressure):
        """Return kernels bound to one explicit backpressure and fixed scheme."""
        pressure = float(pressure)
        if pressure not in self._workspaces:
            model = NozzleModel(
                self.mesh,
                self.fluid,
                p0=self.reference.p0,
                T0=self.reference.T0,
                back_pressure=pressure,
                outlet_regime="auto",
            )
            self._workspaces[pressure] = SteadyWorkspace(
                model, self.scheme, residual_policy()
            )
            # A complete sweep has many pressures. Keep only recent bindings;
            # retaining every compiled operator can exhaust host memory.
            if len(self._workspaces) > 2:
                self._workspaces.popitem(last=False)
        self._workspaces.move_to_end(pressure)
        return self._workspaces[pressure]

    def cold_initial(self, pressure):
        return np.asarray(self.workspace(pressure).model.initial_state())

    def metrics(self, full, pressure):
        metrics = self.workspace(pressure).sample(jnp.asarray(full))
        # The second value is a global balance defect, never the retired SSR.
        return float(np.max(metrics["equation_norm"])), float(
            np.max(metrics["global_defect"])
        )

    def face_flux(self, full, pressure):
        return self.workspace(pressure).operator.face_flux(jnp.asarray(full))[:, 0]

    def warm_up(self, initial, pressure):
        workspace = self.workspace(pressure)
        operator = workspace.operator
        start = perf_counter()
        state = jnp.asarray(initial)
        workspace.sample(state)
        dt = operator.timestep(state, 0.2).dt
        candidate, _ = operator.step(state, dt, 0.0)
        candidate.block_until_ready()
        return perf_counter() - start

    def solve(
        self,
        initial,
        pressure,
        max_updates,
        *,
        cfl=0.2,
        growth=1.2,
        maximum=1e6,
        callback=None
    ):
        workspace = self.workspace(pressure)
        settings = SteadySettings(
            residual=residual_policy(),
            cfl=cfl,
            cfl_growth=growth,
            cfl_max=maximum,
            max_steps=max_updates,
        )
        return solve_single_grid(
            workspace.model,
            self.scheme,
            settings,
            initial_state=initial,
            workspace=workspace,
            callback=callback,
        )

    def integrate(self, initial, pressure, max_updates):
        """Return study records using only accepted states from the shared driver."""
        warm_up_s = self.warm_up(initial, pressure)
        solution = self.solve(initial, pressure, max_updates)
        history, trials = [], []
        for row in solution.history:
            if row["phase"] != "initial":
                trials.append(
                    dict(
                        update=row["step"],
                        retry=row["retry"],
                        dt_s=row["dt"],
                        accepted=row["accepted"],
                        inner_iterations=row["inner_iterations"],
                    )
                )
            if row["accepted"]:
                history.append(
                    dict(
                        update=row["step"],
                        spatial_rms=float(np.max(row["equation_norm"])),
                        global_defect=float(np.max(row["global_defect"])),
                        boundary_max=float(row["boundary_max"]),
                        dt_s=row["dt"],
                        cfl=row["cfl"],
                    )
                )
        return solution.state, dict(
            history=history,
            trials=trials,
            reason=solution.reason,
            solve_s=solution.elapsed_s,
            converged=solution.success,
            warm_up_s=warm_up_s,
            metrics=solution.metrics,
        )
