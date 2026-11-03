"""Complete solver checks for exactly preserved states and wall-source balances."""

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.analytical.transient import (
    exact_homogeneous,
)
from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

from tests._support.regression import TRANSIENT_SCHEME, assert_transient_budget
from tests._support.two_fluid import model as two_fluid_model

pytestmark = pytest.mark.slow


def test_annular_wall_sources_follow_the_homogeneous_exact_trajectory():
    fluid = jxp.FluidPerfectGas("Air")
    settings = dict(
        density=1.2,
        temperature=300.0,
        v_m=30.0,
        v_theta=3.0,
        radius=0.2,
        height=0.005,
        tau_w=1.0,
        q_w=20.0,
    )
    initial_q, initial_w, _ = exact_homogeneous(0.0, fluid, **settings)
    time = 4e-4
    expected_q, expected_w, _ = exact_homogeneous(time, fluid, **settings)
    mesh, native = build_annular_mesh(conical_channel(0.12, 0.2, 0.005, 0.0), 8)
    speed0 = np.hypot(settings["v_m"], settings["v_theta"])
    direction = np.array([settings["v_m"], settings["v_theta"]]) / speed0
    gas_constant = float(fluid.constants.R)
    gamma = float(fluid.constants.gamma)
    cv, cp = gas_constant / (gamma - 1), gamma * gas_constant / (gamma - 1)
    h_shift = initial_w[-1] - cp * settings["temperature"]

    class ForcedTraceAnnulus(AnnularModel):
        """Supply the exact time-dependent exterior for this homogeneous problem.

        Only boundary data are specialized. Storage, source physics, fluxes,
        reconstruction, recovery and stepping use the production implementation.
        """

        def boundary_residual(self, full, time=0.0):
            speed = speed0 - 2 * settings["tau_w"] * time / (
                settings["density"] * settings["height"]
            )
            energy = initial_q[-1] / settings["density"] + 2 * settings[
                "q_w"
            ] * time / (settings["density"] * settings["height"])
            temperature = (energy - 0.5 * speed**2 - h_shift) / cv
            target = jnp.array(
                [
                    settings["density"] * gas_constant * temperature,
                    direction[0] * speed,
                    settings["radius"] * direction[1] * speed,
                    cp * temperature + h_shift,
                ]
            )
            return (full[jnp.array([0, -1])] - target) / self.variable_scales

    initial = jnp.tile(jnp.asarray(initial_w), (mesh.n_cells + 2, 1))
    model = ForcedTraceAnnulus(
        mesh,
        native,
        fluid,
        inlet_data=dict(p0=initial_w[0], T0=300.0, alpha_in=np.arctan2(3.0, 30.0)),
        outlet_pressure=initial_w[0],
        initial_state=initial,
        tau_w=settings["tau_w"],
        q_w=settings["q_w"],
    )
    solution = solve_transient(model, (time,), TRANSIENT_SCHEME, cfl=0.4)[-1]
    actual = np.asarray(model.cell_conserved(jnp.asarray(solution.state[1:-1])))
    np.testing.assert_allclose(
        actual / model.conservative_scales,
        np.tile(expected_q / model.conservative_scales, (mesh.n_cells, 1)),
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        solution.state / model.variable_scales,
        np.tile(expected_w / model.variable_scales, (mesh.n_cells + 2, 1)),
        rtol=2e-10,
        atol=2e-12,
    )
    assert_transient_budget(model, initial, solution)


@pytest.mark.parametrize("velocity", [0.0, 25.0], ids=["rest", "co-moving-phases"])
def test_two_fluid_uniform_equilibrium_survives_complete_rk3_advance(velocity):
    model = two_fluid_model(velocity=velocity, closures=None)
    initial = model.initial_state()
    scheme = Scheme(
        flux="model_compatibility",
        reconstruction="muscl_van_leer",
        time_integrator="runge-kutta-3",
    )
    solution = solve_transient(model, (1e-6,), scheme, initial_state=initial)[-1]
    np.testing.assert_allclose(
        solution.state / model.variable_scales,
        initial / model.variable_scales,
        rtol=2e-11,
        atol=2e-12,
    )
    assert_transient_budget(model, initial, solution)
