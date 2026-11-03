"""Wall-source contracts for angular-momentum annular Euler."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.models.euler_annular import (
    AnnularModel,
    heat_flux_wall,
    shear_stress_wall,
)
from nozzle_hyperbolic.models.euler_annular.sources import integrated_source
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


def _operator(
    fluid, native, *, inlet_data, outlet_pressure, reconstruction="constant", **physical
):
    mesh, native = build_annular_mesh(
        native.geometry, native.n_cells, faces=native.m_faces
    )
    model = AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=inlet_data,
        outlet_pressure=outlet_pressure,
        **physical
    )
    return Operator(model, Scheme(reconstruction=reconstruction))


@pytest.fixture(scope="module")
def physical_setup():
    fluid = jxp.FluidPerfectGas(name="Air")
    mesh = conical_channel(0.05, 0.1, 0.005, 0.0).mesh(4)
    return fluid, mesh


def test_named_placeholders_are_zero_default_prescribed_broadcast_jax_kernels(
    physical_setup,
):
    fluid, _ = physical_setup
    W = jnp.tile(jnp.asarray([1e5, 3.0, -0.4, 3e5]), (2, 1))
    for closure in (shear_stress_wall, heat_flux_wall):
        np.testing.assert_array_equal(
            jax.jit(lambda state: closure(state, 0.1, fluid))(W), [0.0, 0.0]
        )
        actual = jax.jit(
            lambda state: closure(state, 0.1, fluid, prescribed=jnp.array([2.0, 3.0]))
        )(W)
        np.testing.assert_array_equal(actual, [2.0, 3.0])
        np.testing.assert_array_equal(
            jax.jacfwd(lambda state: closure(state, 0.1, fluid))(W), 0.0
        )


def test_wall_shear_is_angular_torque_and_heat_has_the_requested_sign(physical_setup):
    fluid, _ = physical_setup
    W = jnp.asarray([[1e5, 3.0, -0.8, 3e5], [1e5, 3.0, 1.6, 3e5]])
    # Speeds are (3,-4) and (3,4) m/s. Both wall measures are .04 m^2.
    # Shear magnitudes 2/3 Pa imply drag magnitudes .08/.12 N. Third loads
    # must include the respective .2/.4 m moment arms, in N m rather than N.
    source = integrated_source(
        W,
        jnp.array([0.2, 0.4]),
        0.0,
        0.0,
        jnp.array([1e-4, 2e-4]),
        fluid,
        height=jnp.array([0.005, 0.01]),
        tau_w=jnp.array([2.0, 3.0]),
        q_w=jnp.array([1000.0, -500.0]),
    )
    np.testing.assert_allclose(
        source,
        [[0.0, -0.048, 0.0128, 40.0], [0.0, -0.072, -0.0384, -20.0]],
        rtol=2e-14,
        atol=2e-14,
    )
    shear_only = integrated_source(
        W,
        jnp.array([0.2, 0.4]),
        0.0,
        0.0,
        jnp.array([1e-4, 2e-4]),
        fluid,
        height=jnp.array([0.005, 0.01]),
        tau_w=2.0,
    )
    np.testing.assert_array_equal(shear_only[:, 3], 0.0)


def test_shear_direction_and_its_ad_remain_finite_at_rest(physical_setup):
    fluid, _ = physical_setup
    rest = jnp.array([1e5, 0.0, 0.0, 3e5])
    source = lambda state: integrated_source(
        state, 0.2, 0.0, 0.0, 1e-4, fluid, height=0.005, tau_w=2.0, q_w=1000.0
    )
    np.testing.assert_array_equal(jax.jit(source)(rest), [0.0, 0.0, 0.0, 40.0])
    assert np.isfinite(jax.jacfwd(source)(rest)).all()
    assert np.isfinite(jax.jacrev(source)(rest)).all()


@pytest.mark.parametrize(
    "wall",
    [
        dict(tau_w=-1.0),
        dict(tau_w=float("inf")),
        dict(q_w=float("nan")),
        dict(tau_w=[1.0, 2.0]),
        dict(q_w=[1.0, 2.0]),
    ],
)
def test_factory_validates_fixed_wall_profiles_before_jit(physical_setup, wall):
    fluid, mesh = physical_setup
    with pytest.raises(ValueError):
        _operator(
            fluid,
            mesh,
            inlet_data=dict(p0=2e5, T0=300.0, alpha_in=0.3),
            outlet_pressure=199000.0,
            **wall
        )


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_wall_profiles_reach_factory_source_and_full_colored_jacobian(
    physical_setup, reconstruction
):
    fluid, mesh = physical_setup
    solver = _operator(
        fluid,
        mesh,
        inlet_data=dict(p0=2e5, T0=300.0, alpha_in=0.3),
        outlet_pressure=199000.0,
        reconstruction=reconstruction,
        tau_w=jnp.array([1.0, 2.0, 3.0, 4.0]),
        q_w=1000.0,
    )
    base = solver.model.initial_state().at[:, 1].set(30.0)
    source = np.asarray(solver.source(base))
    np.testing.assert_allclose(
        source[:, 3], 2 * mesh.volumes / mesh.b * 1000.0, rtol=2e-14
    )
    assert np.all(source[:, 1] < 0.0)
    assert np.all(source[:, 2] < 0.0)
    assert (
        not np.asarray(solver.model.tau_wall).flags.writeable
        and not np.asarray(solver.model.heat_wall).flags.writeable
    )
    fraction = jnp.linspace(0.0, 1.0, 6)
    point = (
        base.at[:, 0]
        .add(500 * fraction)
        .at[:, 1]
        .add(fraction)
        .at[:, 2]
        .add(0.2 * fraction)
        .at[:, 3]
        .add(500 * fraction)
    )
    colored = np.asarray(solver.jacobian(point.ravel(), base.ravel(), 1e-5, 0.0))
    dense = np.asarray(solver.dense_jacobian(point.ravel(), base.ravel(), 1e-5, 0.0))
    scale = np.tile([2e5, 400.0, 40.0, 1.6e5], 6)
    np.testing.assert_allclose(colored * scale, dense * scale, rtol=2e-10, atol=2e-10)
