"""Osher path quadrature, sonic roots and binding contracts."""

import jax
import jax.numpy as jnp
import pytest

from nozzle_hyperbolic.numerics.osher import OsherFlux
from nozzle_hyperbolic.models.scalar.advection import AdvectionModel


INTEGRATIONS = ("gauss", "sonic_partitioned")


CONFIGURATIONS = [(points, policy) for points in (3, 5, 9) for policy in INTEGRATIONS]


class Advection:
    def __init__(self, speed):
        self.speed = speed

    def face_conserved(self, state):
        return state

    def face_physical_flux(self, state):
        return self.speed * state

    def face_characteristics(self, left, right, delta_factor):
        shape = left.shape[:-1] + (1,)
        return (
            jnp.ones(shape + (1, 1)),
            jnp.full(shape, self.speed),
            jnp.full(shape, delta_factor),
        )

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )


class Burgers(Advection):
    def __init__(self):
        super().__init__(0.0)

    def face_physical_flux(self, state):
        return 0.5 * state**2

    def face_characteristics(self, left, right, delta_factor):
        eigenvalues = 0.5 * (left + right)
        width = delta_factor * jnp.maximum(jnp.maximum(abs(left), abs(right)), 1.0)
        return jnp.ones(left.shape[:-1] + (1, 1, 1)), eigenvalues, width


class MixedStorage:
    """Fused-only model: nonlinear storage, mixed balances and both wave signs."""

    faces = 3
    transform = jnp.asarray([[1.0, 0.4], [-0.3, 1.2]])
    inverse = jnp.linalg.inv(transform)
    speeds = jnp.asarray([-0.7, 2.1])

    def base_storage(self, state):
        return state + jnp.asarray([0.2, 0.1]) * state**3

    def face_data(self, state):
        # A path quadrature axis must never be flattened into model geometry.
        assert state.shape == (self.faces, 2)
        base = self.base_storage(state)
        return dict(
            conserved=base @ self.transform.T,
            physical_flux=(base * self.speeds) @ self.transform.T,
            auxiliary=state,
        )

    def face_characteristics_from_data(
        self, left, right, delta_factor, left_data, right_data
    ):
        projectors = jnp.einsum("ik,kj->kij", self.transform, self.inverse)
        return (
            jnp.broadcast_to(projectors, (self.faces, 2, 2, 2)),
            jnp.broadcast_to(self.speeds, left.shape),
            jnp.full(left.shape, delta_factor),
        )

    def face_state_path(self, left, right, parameter):
        assert parameter.shape == (self.faces, 1)
        curvature = jnp.asarray([0.2, -0.1])
        return (
            left + parameter * (right - left) + parameter * (1 - parameter) * curvature,
            right - left + (1 - 2 * parameter) * curvature,
        )

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )


def flux_function(model, flux, delta_factor=0.1):
    def evaluate(pair):
        left, right = pair[0:1], pair[1:2]
        return flux(
            model,
            left,
            right,
            delta_factor,
            model.face_physical_flux(left),
            model.face_physical_flux(right),
        )[0, 0]

    return evaluate


class WideHartenAdvection(Advection):
    def __init__(self, width):
        super().__init__(1.0)
        self.width = width

    def face_characteristics(self, left, right, delta_factor):
        projectors, eigenvalues, _ = super().face_characteristics(
            left, right, delta_factor
        )
        return projectors, eigenvalues, jnp.full_like(eigenvalues, self.width)


@pytest.fixture(scope="module", params=(3, 5, 9))
def sonic_burgers(request):
    evaluate = flux_function(Burgers(), OsherFlux(request.param, "sonic_partitioned"))
    return jax.jit(evaluate), jax.jit(jax.jacfwd(evaluate))


class OscillatoryScalar(Advection):
    def __init__(self):
        super().__init__(0.0)

    def face_physical_flux(self, state):
        return (1 - jnp.cos(7 * jnp.pi * state)) / (7 * jnp.pi) + 0.012 * state

    def face_characteristics(self, left, right, delta_factor):
        eigenvalues = jnp.sin(7 * jnp.pi * 0.5 * (left + right)) + 0.012
        return (
            jnp.ones(left.shape[:-1] + (1, 1, 1)),
            eigenvalues,
            jnp.full_like(eigenvalues, delta_factor),
        )


class InvalidInteriorPath(Advection):
    def __init__(self):
        super().__init__(1.0)

    def face_state_path(self, left, right, parameter):
        return left + parameter * (right - left) - 8 * parameter * (
            1 - parameter
        ), right - left - 8 * (1 - 2 * parameter)

    def admissible(self, full, left, right):
        return (
            super().admissible(full, left, right)
            & jnp.all(left > 0)
            & jnp.all(right > 0)
        )


class LinearBalance(AdvectionModel):
    """Complete EOS-free FVM model with a declared characteristic width."""

    def __init__(self, width=0.1, physical_face_bound=1.0):
        super().__init__(4, speed=1.0)
        self.width, self.physical_face_bound = width, physical_face_bound

    def face_characteristics(self, left, right, delta_factor):
        return (
            jnp.ones((left.shape[0], 1, 1, 1)),
            jnp.ones_like(left),
            jnp.full_like(left, self.width),
        )

    def face_speed_bounds(self, state):
        bound = jnp.full(state.shape[0], self.physical_face_bound)
        return -bound, bound
