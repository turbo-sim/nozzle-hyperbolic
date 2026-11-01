"""Roe capability and integration tests using an EOS-free user model."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.roe import roe, roe_absolute
from nozzle_hyperbolic.solvers.fvm import Mesh, Operator, Scheme


class UserRoeModel:
    """Two nonlinear stored quantities sharing one repeated advection speed.

    There are no Euler identities, EOS or face_speed_bounds callbacks. One
    rank-two projector exercises a wave count independent of equation count.
    """

    nvar = 2
    equation_names = ("first", "second")
    conservative_scales = variable_scales = global_flux_scales = np.array([1.0, 2.0])
    reference_time = 1.0
    source_dependency_radius = boundary_dependency_radius = 1
    boundary_coupling = "local"
    velocity = 1.7

    def __init__(self, *, valid=True, width=0.0):
        faces = np.linspace(0.0, 1.0, 6)
        self.mesh = Mesh(
            faces, 0.5 * (faces[1:] + faces[:-1]), np.diff(faces), np.ones(6)
        )
        self.valid = valid
        self.width = width

    def cell_conserved(self, values):
        return values + 0.03 * values**3

    face_conserved = cell_conserved

    def face_physical_flux(self, values):
        return self.velocity * self.cell_conserved(values)

    def face_roe_data(self, left, right, delta_factor):
        leading = left.shape[:-1]
        return dict(
            projectors=jnp.broadcast_to(jnp.eye(2), (*leading, 1, 2, 2)),
            eigenvalues=jnp.full((*leading, 1), self.velocity),
            width=jnp.full((*leading, 1), self.width),
            valid=jnp.full(leading, self.valid),
        )

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return jnp.zeros_like(full[1:-1])

    def boundary_residual(self, full, time):
        return jnp.stack((full[0] - jnp.array([1.0, 2.0]), full[-1] - full[-2]))

    def wave_speed(self, cells):
        return jnp.full(cells.shape[:-1], self.velocity)

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full))

    def initial_state(self):
        x = jnp.asarray(self.mesh.points)
        full = jnp.stack(
            (1.0 + 0.1 * jnp.sin(3 * x), 2.0 + 0.2 * jnp.sin(2 * x)), axis=-1
        )
        return full.at[0].set(jnp.array([1.0, 2.0])).at[-1].set(full[-2])


def test_safe_zero_width_roe_absolute_and_repeated_wave_contraction():
    eigen = jnp.array([0.0, -2.0, 3.0])
    widths = jnp.array([0.0, 0.0, 4.0])
    np.testing.assert_allclose(roe_absolute(eigen, widths), [0.0, 2.0, 25.0 / 8.0])
    assert np.isfinite(jax.jacfwd(lambda x: roe_absolute(x, widths))(eigen)).all()
    assert np.isfinite(jax.jacrev(lambda x: roe_absolute(x, widths))(eigen)).all()
    left, right = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    flux = roe(
        2 * left,
        2 * right,
        right - left,
        jnp.eye(2)[None],
        jnp.array([2.0]),
        jnp.zeros(1),
    )
    np.testing.assert_allclose(flux, 2 * left)


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_user_roe_capability_has_no_eos_or_required_physical_eigensystem(
    reconstruction,
):
    model = UserRoeModel()
    operator = Operator(model, Scheme(flux="roe", reconstruction=reconstruction))
    state = model.initial_state()
    left, _ = operator.reconstruct(state)
    np.testing.assert_allclose(
        operator.face_flux(state), model.face_physical_flux(left)
    )
    assert bool(operator.is_admissible(state))
    matrix = operator.jacobian(state.ravel(), state.ravel(), 0.01, 0.0)
    assert all(np.isfinite(np.asarray(part)).all() for part in jax.tree.leaves(matrix))
    np.testing.assert_allclose(
        operator.jacobian_difference(state.ravel(), state.ravel(), 0.01, 0.0),
        0.0,
        atol=2e-13,
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
def test_user_roe_integrators_preserve_integrated_budget(integrator, treatment):
    model = UserRoeModel()
    operator = Operator(
        model,
        Scheme(
            flux="roe",
            time_integrator=integrator,
            boundary_treatment=treatment,
            newton_iterations=20,
        ),
    )
    old = model.initial_state()
    dt = 0.1 * float(operator.timestep(old, 1.0).wave_dt)
    new, info, transfer = operator.step_with_balance(old, dt, dt)
    assert bool(info.converged) and bool(info.admissible), info
    change = jnp.sum(operator.storage(new[1:-1]) - operator.storage(old[1:-1]), axis=0)
    np.testing.assert_allclose(change, transfer, atol=3e-12, rtol=2e-10)
    assert not np.array_equal(np.asarray(old), np.asarray(new))


@pytest.mark.parametrize("valid,width", [(False, 0.0), (True, -1.0)])
def test_roe_invalid_data_is_rejected_without_hidden_hll_fallback(valid, width):
    model = UserRoeModel(valid=valid, width=width)
    operator = Operator(model, Scheme(flux="roe"))
    state = model.initial_state()
    assert not bool(operator.is_admissible(state))
    _, info = operator.step(state, 0.01, 0.01)
    assert not bool(info.converged) and not bool(info.admissible)


def test_unsupported_roe_and_invalid_boolean_contract_fail_before_solve():
    model = UserRoeModel()
    model.face_roe_data = None
    with pytest.raises(ValueError, match="does not supply Roe"):
        Operator(model, Scheme(flux="roe"))
    model = UserRoeModel(valid=1.0)
    with pytest.raises(ValueError, match="boolean face vector"):
        Operator(model, Scheme(flux="roe"))


def test_explicit_roe_timestep_accounts_for_interface_dissipation():
    class LargerInterfaceBound(UserRoeModel):
        def face_roe_data(self, left, right, delta_factor):
            data = super().face_roe_data(left, right, delta_factor)
            return dict(data, width=jnp.full_like(data["width"], 20.0))

    model = LargerInterfaceBound()
    explicit = Operator(model, Scheme(flux="roe", time_integrator="euler-explicit"))
    implicit = Operator(model, Scheme(flux="roe"))
    state = model.initial_state()
    viscosity = float(roe_absolute(jnp.asarray(model.velocity), jnp.asarray(20.0)))
    np.testing.assert_allclose(
        explicit.timestep(state, 0.5).dt, 0.5 * np.min(model.mesh.widths) / viscosity
    )
    np.testing.assert_allclose(
        implicit.timestep(state, 0.5).dt,
        0.5 * np.min(model.mesh.widths) / model.velocity,
    )
