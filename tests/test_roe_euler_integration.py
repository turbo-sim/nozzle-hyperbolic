"""Real Euler callbacks composed with conservative Roe time integration."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from nozzle_hyperbolic.verification import build_case, load_case


@pytest.mark.parametrize(
    "case", ["euler_nozzle/subsonic", "euler_annular/conical_subsonic"]
)
@pytest.mark.parametrize(
    "integrator,treatment",
    [
        ("euler-explicit", "segregated"),
        ("runge-kutta-3", "segregated"),
        ("euler-implicit", "coupled"),
        ("euler-implicit", "segregated"),
    ],
)
def test_roe_euler_step_completes_boundaries_and_conserves_integrated_transfer(
    case, integrator, treatment
):
    problem = build_case(load_case(case), cells=5)
    operator = Operator(
        problem.model,
        Scheme(
            flux="roe",
            time_integrator=integrator,
            boundary_treatment=treatment,
            newton_iterations=30,
        ),
    )
    old, completed = operator.complete_boundaries(
        jnp.asarray(problem.initial_state), 0.0
    )
    assert bool(completed.converged) and bool(completed.admissible)
    assert bool(operator.is_admissible(old))
    dt = float(operator.timestep(old, 0.03).dt)
    new, info, transfer = operator.step_with_balance(old, dt, dt)
    assert bool(info.converged) and bool(info.admissible), info
    change = jnp.sum(operator.storage(new[1:-1]) - operator.storage(old[1:-1]), axis=0)
    scale = np.sum(problem.mesh.volumes) * np.asarray(problem.model.conservative_scales)
    np.testing.assert_allclose((change - transfer) / scale, 0.0, atol=4e-12)
    assert float(jnp.max(jnp.abs(operator.boundary_residual(new, dt)))) < 2e-11
    assert np.isfinite(np.asarray(new)).all()


def test_roe_annular_transverse_shear_limits_explicit_step_by_interface_speed():
    from nozzle_hyperbolic.geometry.annular import conical_channel
    from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
    from nozzle_hyperbolic.models.euler_annular import AnnularModel
    import jaxprop as jxp

    fluid = jxp.FluidPerfectGas("Air")
    reservoir = fluid.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    geometry = conical_channel(0.1, 0.2, 0.005, 0.0)
    mesh, native = build_annular_mesh(geometry, 6)
    model = AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=dict(
            p0=2e5, h0=float(reservoir["h"]), s0=float(reservoir["s"]), alpha_in=0.0
        ),
        outlet_pressure=1.9e5,
    )
    # A large stationary transverse shear raises the interface Roe sound speed,
    # even though each side has the same normal velocity and thermodynamic state.
    state = jnp.tile(jnp.array([2e5, 0.0, 0.0, reservoir["h"]]), (8, 1))
    # Cell radii are constant here; all reconstructed states share the face radius.
    # Keep reservoir-adjacent cells unswirled so their inlet is physically reachable.
    state = state.at[:, 2].set(
        jnp.array([0.0, 0.0, -2000.0, -2000.0, 2000.0, 2000.0, 0.0, 0.0]) * 0.2
    )
    operator = Operator(model, Scheme(flux="roe", time_integrator="euler-explicit"))
    data = operator.roe_data(state)
    physical = float(jnp.max(model.wave_speed(state[1:-1])))
    assert float(jnp.max(jnp.abs(data["eigenvalues"]))) > physical
    # Boundary completion may alter endpoint traces; the interior shear remains.
    limits = operator.timestep(state, 0.2)
    assert bool(limits.valid)
    assert float(limits.wave_dt) < float(jnp.min(jnp.asarray(mesh.widths))) / physical
