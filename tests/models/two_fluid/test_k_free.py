"""K-free compatibility option and smooth entropy fix of the two-fluid model."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.nozzle import (
    ConstantAreaGeometry,
    SymmetricNozzleGeometry,
)
from nozzle_hyperbolic.models.two_fluid import (
    CompatibilityConfig,
    StabilizationConfig,
    primitives,
    from_primitives,
)
from nozzle_hyperbolic.models.two_fluid.flux import (
    fvs_interface_flux,
    k_free_interface_flux,
)
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme

from tests._support.two_fluid import model, raw_flux

SCHEME = SimpleNamespace(flux="model_compatibility")


def constant_faces(W):
    return W[:-1], W[1:]


def nozzle_model(**kwargs):
    faces = 0.2 * (np.linspace(0.0, 0.2, 7) / 0.2) ** 1.4
    return model(mesh=SymmetricNozzleGeometry().build_mesh(faces=faces), **kwargs)


def test_default_selects_native_kernels_and_ignores_smooth_width():
    m = nozzle_model()
    assert m.nonconservative_mode == "flux_difference"
    explicit = nozzle_model(
        compatibility=CompatibilityConfig(k_free=False, smooth=False, smooth_width=0.5)
    )
    U = primitives(m.initial_state())
    U = U.at[:, 3].set(jnp.linspace(0.3, 0.5, U.shape[0])).at[:, 1].add(30.0)
    W = from_primitives(U, m.fluid_g, m.fluid_l)
    F = raw_flux(m, W)
    np.testing.assert_array_equal(F, raw_flux(explicit, W))
    np.testing.assert_array_equal(
        m.integrated_nonconservative(W, F), explicit.integrated_nonconservative(W, F)
    )


@pytest.mark.parametrize("smooth", (False, True))
@pytest.mark.parametrize("velocity", (0.0, 25.0, -25.0))
def test_k_free_uniform_state_has_physical_flux_and_zero_path_term(velocity, smooth):
    m = nozzle_model(
        velocity=velocity, compatibility=CompatibilityConfig(k_free=True, smooth=smooth)
    )
    assert m.nonconservative_mode == "path"
    W = m.initial_state()
    F = raw_flux(m, W)
    np.testing.assert_allclose(F, m.face_physical_flux(W[:-1]), rtol=2e-13, atol=1e-9)
    np.testing.assert_array_equal(
        m.integrated_nonconservative(W, F, faces=constant_faces(W)), 0.0
    )
    derivative = jax.jacfwd(lambda left: m.numerical_face_flux(left, W[1:], SCHEME))(
        W[:-1]
    )
    assert np.isfinite(np.asarray(derivative)).all()


@pytest.mark.parametrize("slip", (1.0, 10.0, 40.0))  # zero-slip, blend and full branches
def test_k_free_path_term_rows_and_finite_derivatives(slip):
    m = nozzle_model(compatibility=CompatibilityConfig(k_free=True))
    U = primitives(m.initial_state())
    n = U.shape[0]
    U = (
        U.at[:, 0].multiply(jnp.linspace(1.0, 0.8, n))
        .at[:, 1].add(slip)
        .at[:, 3].set(jnp.linspace(0.3, 0.5, n))
        .at[:, 4].add(jnp.linspace(0.0, -2e4, n))
    )
    W = from_primitives(U, m.fluid_g, m.fluid_l)
    nc = m.integrated_nonconservative(W, raw_flux(m, W), faces=constant_faces(W))
    np.testing.assert_array_equal(nc[:, :2], 0.0)
    np.testing.assert_allclose(
        nc[:, 2] + nc[:, 3], 0.0, atol=1e-12 * float(jnp.max(jnp.abs(nc[:, 2])))
    )
    assert float(jnp.max(jnp.abs(nc[:, 2]))) > 0
    for function in (
        lambda w: m.numerical_face_flux(w[:-1], w[1:], SCHEME),
        lambda w: m.integrated_nonconservative(w, None, faces=constant_faces(w)),
    ):
        assert np.isfinite(np.asarray(jax.jacfwd(function)(W))).all()


@pytest.mark.parametrize("kernel", ("k_inverse", "k_free"))
def test_smooth_fix_changes_only_waves_slower_than_its_width(kernel):
    fluid = model().fluid_g
    flux = fvs_interface_flux if kernel == "k_inverse" else k_free_interface_flux

    def face(speed, smooth):
        left = jnp.array((2e5, speed + 3.0, speed, 0.40, 3.0e5, 2.8e5))
        right = jnp.array((1.98e5, speed + 3.5, speed + 0.4, 0.41, 2.99e5, 2.8e5))
        return flux(left, right, fluid, fluid, 0.5, 1e-6, 1.0, 5.0, 20.0, smooth=smooth)

    # Every wave faster than 0.02 a_mix and no sonic crossing: identical weights.
    np.testing.assert_allclose(face(60.0, True), face(60.0, False), rtol=1e-14, atol=0.0)
    slow_on, slow_off = face(0.5, True), face(0.5, False)
    assert not np.allclose(slow_on, slow_off, rtol=1e-12, atol=0.0)
    derivative = jax.jacfwd(lambda s: face(s, True))(0.5)
    assert np.isfinite(np.asarray(derivative)).all()


def test_k_free_reconstructed_colored_jacobian_matches_dense():
    mesh = ConstantAreaGeometry().build_mesh(faces=np.linspace(0, 1, 5) ** 1.3)
    m = model(
        mesh=mesh,
        stabilization=StabilizationConfig(void_strength=0.01),
        compatibility=CompatibilityConfig(k_free=True, smooth=True),
    )
    U = primitives(m.initial_state())
    # Strictly monotone slopes avoid the limiter's flat-state corner.
    U = (
        U
        + jnp.asarray(mesh.points)[:, None]
        * jnp.array((100.0, 2.0, 3.0, 0.01, 100.0, 200.0))[None, :]
    )
    W = from_primitives(U, m.fluid_g, m.fluid_l)
    op = Operator(
        m, Scheme(flux="model_compatibility", reconstruction="muscl_van_albada")
    )
    args = (W.ravel(), W.ravel(), 1e-6, 0.0)
    dense = np.asarray(op.dense_jacobian(*args))
    colored = np.asarray(op.jacobian(*args))
    scale = np.tile(m.variable_scales, W.shape[0])
    np.testing.assert_allclose(colored * scale, dense * scale, rtol=2e-11, atol=2e-11)
    assert np.isfinite(dense).all()


@pytest.mark.parametrize(
    "options",
    (
        dict(k_free=1),
        dict(smooth="yes"),
        dict(smooth_width=0.0),
        dict(smooth_width=float("nan")),
    ),
)
def test_invalid_k_free_and_smooth_settings_rejected(options):
    with pytest.raises(ValueError):
        CompatibilityConfig(**options)


def test_k_free_requires_faces_and_excludes_mass_upwind():
    with pytest.raises(ValueError, match="mass_upwind_cutoff"):
        model(
            compatibility=CompatibilityConfig(k_free=True),
            stabilization=StabilizationConfig(mass_upwind_cutoff=1e-3),
        )
    m = model(compatibility=CompatibilityConfig(k_free=True))
    W = m.initial_state()
    with pytest.raises(ValueError, match="reconstructed faces"):
        m.integrated_nonconservative(W, raw_flux(m, W))
