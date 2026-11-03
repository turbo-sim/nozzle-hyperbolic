"""Independent scalar physics embedded in a redundant constrained state."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.numerics.osher import OsherFlux
from nozzle_hyperbolic.numerics.reconstruction import reconstruct_faces
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


class _MappedBurgers:
    nvar, ntransport, nconstraints, nboundary = 2, 1, 1, 2
    nonconservative_mode = "none"
    equation_names = ("scalar content",)
    constraint_names = ("redundant constitutive variable",)
    variable_scales = np.ones(2)
    conservative_scales = global_flux_scales = constraint_scales = np.ones(1)
    reconstruction_scales = np.ones(1)
    reference_time = 1.
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self):
        faces = np.array([0., .1, .25, .55, .8, 1.])
        self.mesh = Mesh(faces, .5 * (faces[:-1] + faces[1:]), np.diff(faces), np.ones(6))

    def reconstruction_state(self, full):
        return full[..., :1]

    def from_reconstruction_state(self, values):
        return jnp.concatenate((values, values ** 2), axis=-1)

    def cell_conserved(self, state):
        return state[..., :1]

    face_conserved = cell_conserved

    def cell_constraints(self, state):
        return state[..., 1:] - state[..., :1] ** 2

    def face_physical_flux(self, state):
        return .5 * state[..., :1] ** 2

    def face_speed_bounds(self, state):
        return state[..., 0], state[..., 0]

    def face_characteristics(self, left, right, delta_factor):
        speeds = .5 * (left[..., :1] + right[..., :1])
        return jnp.ones(speeds.shape + (1, 1)), speeds, jnp.zeros_like(speeds)

    def face_state_path(self, left, right, parameter):
        value = left[..., :1] + parameter * (right[..., :1] - left[..., :1])
        tangent = right[..., :1] - left[..., :1]
        return self.from_reconstruction_state(value), jnp.concatenate((tangent, 2 * value * tangent), axis=-1)

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return jnp.zeros((self.mesh.n_cells, 1))

    def boundary_residual(self, full, time=0.):
        return jnp.stack((full[0] - full[1], full[-1] - full[-2]))

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full)) & jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(right))

    def wave_speed(self, cells):
        return jnp.abs(cells[..., 0])

    def initial_state(self):
        x = jnp.asarray(self.mesh.points)
        # Unequal adjacent slopes keep finite-difference checks away from
        # limiter and Rusanov speed-selection kinks.
        return self.from_reconstruction_state((1 + .3 * x + .17 * x**2)[:, None])


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_minmod", "muscl_van_albada"])
def test_mapped_reconstruction_projects_face_constraints_and_has_exact_stencil(reconstruction):
    model = _MappedBurgers()
    operator = Operator(model, Scheme(flux="rusanov", reconstruction=reconstruction))
    # Newton may explore off-manifold cell densities. Face reconstruction uses
    # only the declared physical representation and recovers the EOS at faces.
    full = model.initial_state().at[:, 1].add(jnp.linspace(.1, .2, 7))
    left, right = operator.reconstruct(full)
    expected = reconstruct_faces(full[:, :1], jnp.asarray(model.mesh.points), jnp.asarray(model.mesh.faces), reconstruction)
    np.testing.assert_allclose(left[:, :1], expected[0], atol=1e-15)
    np.testing.assert_allclose(right[:, :1], expected[1], atol=1e-15)
    np.testing.assert_allclose(model.cell_constraints(left), 0, atol=1e-15)
    np.testing.assert_allclose(model.cell_constraints(right), 0, atol=1e-15)
    direct = jax.jacfwd(lambda flat: operator.spatial_residual(flat.reshape(full.shape)))(full.ravel())
    np.testing.assert_array_equal(np.asarray(direct)[..., 1::2], 0)
    args = (full.ravel(), full.ravel(), .03, .1)
    np.testing.assert_allclose(operator.jacobian(*args), operator.dense_jacobian(*args), atol=2e-14)
    direction = np.random.default_rng(4).normal(size=full.size)
    epsilon = 1e-6
    difference = (operator.residual(args[0] + epsilon * direction, *args[1:]) -
                  operator.residual(args[0] - epsilon * direction, *args[1:])) / (2 * epsilon)
    np.testing.assert_allclose(operator.jacobian(*args) @ direction, difference, atol=3e-9)


def test_sonic_partitioned_osher_uses_transported_count_for_constrained_state():
    model = _MappedBurgers()
    left = model.from_reconstruction_state(jnp.array([[-1.], [2.], [1.]]))
    right = model.from_reconstruction_state(jnp.array([[2.], [-1.], [3.]]))
    method = OsherFlux(integration="sonic_partitioned")

    def flux(l, r):
        return method(model, l, r, .1, model.face_physical_flux(l), model.face_physical_flux(r))

    result = jax.jit(flux)(left, right)
    expected = .5 * (jnp.maximum(left[:, :1], 0) ** 2 + jnp.minimum(right[:, :1], 0) ** 2)
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result, expected, atol=1e-13)
    derivative = jax.jacfwd(flux, argnums=0)(left, right)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(np.asarray(derivative)[..., 1], 0)


def test_neighbor_flux_nonconservative_callback_includes_muscl_stencil_extension():
    class WideNonconservative(_MappedBurgers):
        nonconservative_mode = "flux_difference"
        nonconservative_dependency_radius = 2
        supported_fluxes = ("rusanov",)

        def integrated_nonconservative(self, full, raw):
            cells = jnp.arange(self.mesh.n_cells)
            return .08 * (raw[jnp.minimum(cells + 2, self.mesh.n_cells)] -
                          raw[jnp.maximum(cells - 1, 0)])

    model = WideNonconservative()
    operator = Operator(model, Scheme(flux="rusanov", reconstruction="muscl_van_albada"))
    assert operator.radius == 3
    full = model.initial_state()
    args = (full.ravel(), full.ravel(), .02, 0.)
    dense = np.asarray(operator.dense_jacobian(*args))
    np.testing.assert_allclose(operator.jacobian(*args), dense, rtol=2e-14, atol=2e-14)
    # Physical row one genuinely reaches full-state block four after MUSCL.
    assert dense[2, 8] != 0
