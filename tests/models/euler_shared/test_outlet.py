"""Physical wave identities and shared nonlinear boundary completion."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_outlet import EulerOutlet, transmissive_residual
from nozzle_hyperbolic.solvers.boundaries import make_boundary_solver


def fixture(mach, ratio, nvar=3):
    fluid = jxp.FluidPerfectGas(name="Air")
    state = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    W = jnp.array([state["p"], mach * state["a"], state["h"]])
    if nvar == 4:
        W = jnp.array([W[0], W[1], 8.0, W[2]])
    scales = (
        jnp.array([1e5, 350.0, 1e5])
        if nvar == 3
        else jnp.array([1e5, 350.0, 70.0, 1e5])
    )
    outlet = EulerOutlet(fluid, ratio * 1e5, scales, 400.0, 0.5, "finite_wave")
    return fluid, W, outlet


@pytest.mark.parametrize("nvar", [3, 4])
def test_finite_compression_enters_supersonic_interior_and_satisfies_jump_laws(nvar):
    fluid, W, outlet = fixture(2.2, 8.0, nvar)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    assert int(wave.mode) == 3 and float(wave.head_speed) < 0 and bool(wave.direct_flux)
    assert float(W[1] / outlet.thermodynamics(W)["a"]) > 1.0
    left = outlet.thermodynamics(W)
    right = outlet.thermodynamics(wave.state)
    speed = wave.head_speed
    u0, u1 = W[1] - speed, wave.state[1] - speed
    np.testing.assert_allclose(left["rho"] * u0, right["rho"] * u1, rtol=2e-14)
    np.testing.assert_allclose(
        W[0] + left["rho"] * u0**2, wave.state[0] + right["rho"] * u1**2, rtol=2e-14
    )
    np.testing.assert_allclose(
        W[-1] + 0.5 * u0**2, wave.state[-1] + 0.5 * u1**2, rtol=2e-14
    )
    assert float(right["s"]) > float(left["s"])
    if nvar == 4:
        assert wave.state[2] == W[2]


@pytest.mark.parametrize(
    "mach,ratio,mode", [(2.2, 1.2, 1), (0.3, 0.8, 4), (0.3, 0.01, 5), (1.2, 0.5, 1)]
)
def test_wave_branch_entropy_and_sonic_sampling(mach, ratio, mode):
    fluid, W, outlet = fixture(mach, ratio)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    assert int(wave.mode) == mode
    state = outlet.thermodynamics(wave.state)
    if mode in (4, 5):
        np.testing.assert_allclose(
            state["s"], outlet.thermodynamics(W)["s"], atol=2e-12
        )
    if mode == 5:
        np.testing.assert_allclose(wave.state[1] / state["a"], 1.0, atol=3e-15)
    if mode == 1:
        np.testing.assert_array_equal(wave.state, W)


def test_negative_velocity_is_retained_and_wave_ad_matches_finite_differences():
    _, W, outlet = fixture(2.2, 14.0)
    cells = jnp.stack((W, W))
    wave = outlet.wave(cells)
    assert float(wave.state[1]) < 0.0
    np.testing.assert_allclose(outlet.residual(wave.state, cells), 0.0, atol=1e-14)
    evaluate = lambda u: outlet.wave(cells.at[-1, 1].set(u)).state
    derivative = jax.jit(jax.jacfwd(evaluate))(W[1])
    finite = (evaluate(W[1] + 1e-3) - evaluate(W[1] - 1e-3)) / 2e-3
    np.testing.assert_allclose(derivative, finite, rtol=1e-7, atol=2e-8)


def test_both_outlet_formulations_use_an_hp_only_eos():
    base, W, _ = fixture(0.3, 0.8)

    class HPOnly:
        def get_state(self, pair, h, p):
            assert pair == jxp.HmassP_INPUTS
            return base.get_state(pair, h, p)

    fluid = HPOnly()
    outlet = EulerOutlet(fluid, 8e4, [1e5, 350.0, 1e5], 400.0, 0.5)
    cells = jnp.stack((W, W))
    st = outlet.thermodynamics(W)
    velocity = W[1] + (W[0] - 8e4) / (st["rho"] * st["a"])
    # Isentropic h is obtained through the configured EOS, not an ideal formula.
    trace_state = base.get_state(jxp.PSmass_INPUTS, 8e4, st["s"])
    trace = jnp.array([8e4, velocity, trace_state["h"]])
    np.testing.assert_allclose(jax.jit(outlet.residual)(trace, cells), 0.0, atol=3e-14)
    finite = EulerOutlet(fluid, 8e4, [1e5, 350.0, 1e5], 400.0, 0.5, "finite_wave")
    exact = EulerOutlet(base, 8e4, [1e5, 350.0, 1e5], 400.0, 0.5, "finite_wave")
    np.testing.assert_allclose(
        jax.jit(finite.wave)(cells).state,
        exact.wave(cells).state,
        rtol=3e-11,
    )


def test_nonlinear_boundary_solve_leaves_cells_fixed_and_permits_negative_traces():
    full = jnp.ones((6, 1)).at[0, 0].set(-0.8)

    def residual(z, time):
        return jnp.stack((z[0] ** 3 + 1.0 + time, z[-1] - z[-2]))

    solve = make_boundary_solver(residual, full.shape, jnp.ones(1))
    complete, info = solve(full, 0.1)
    assert bool(info.converged & info.admissible)
    np.testing.assert_array_equal(complete[1:-1], full[1:-1])
    np.testing.assert_allclose(residual(complete, 0.1), 0.0, atol=1e-12)
    assert float(complete[0, 0]) < 0.0


def test_sod_constraints_are_transmissive_and_not_walls():
    full = jnp.array(
        [[1e5, 25.0, 3e5], [1e5, 25.0, 3e5], [1e4, 15.0, 2.5e5], [1e4, 15.0, 2.5e5]]
    )
    residual = transmissive_residual(full, [1e5, 350.0, 1e5])
    np.testing.assert_array_equal(residual, 0.0)
    assert full[0, 1] == full[1, 1] and full[-1, 1] == full[-2, 1]


def test_production_annular_selects_finite_wave_flux_and_accepts_signed_trials():
    from nozzle_hyperbolic.geometry.annular import conical_channel
    from nozzle_hyperbolic.models.euler_annular import AnnularModel
    from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
    from nozzle_hyperbolic.models.euler_annular.equations import physical_flux

    fluid, W, _ = fixture(0.2, 1.2, nvar=4)
    static = fluid.get_state(jxp.HmassP_INPUTS, W[-1], W[0])
    h0 = float(W[-1] + 0.5 * W[1] ** 2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, h0, static["s"])
    mesh, native = build_annular_mesh(conical_channel(0.1, 0.2, 0.005, 0.0), 4)
    model = AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=dict(
            h0=h0, s0=float(static["s"]), p0=float(reservoir["p"]), alpha_in=0.0
        ),
        outlet_pressure=1.2e5,
        outlet_formulation="finite_wave",
    )
    solver = Operator(model, Scheme())
    old = jnp.tile(jnp.asarray([W[0], W[1], 0.0, W[-1]]), (mesh.n_cells + 2, 1))
    assert bool(solver.is_admissible(old.at[:, 1].set(-20.0)))
    full, info = solver.complete_boundaries(old, 0.0)
    assert bool(info.converged & info.admissible)
    assert bool(model.outlet.direct_flux(full[1:-1]))
    np.testing.assert_allclose(solver.boundary_residual(full), 0.0, atol=1e-9)
    expected = mesh.area_faces[-1] * physical_flux(full[-1], native.r_face[-1], fluid)
    np.testing.assert_allclose(solver.face_flux(full)[-1], expected, rtol=1e-13)
