"""Independent correction switches for the shared, model-agnostic Stadtke flux."""

from dataclasses import asdict
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.characteristic import (
    stadtke_flux_with_entropy_viscosity,
    harten_absolute,
    stadtke_flux_with_corrections,
)
from nozzle_hyperbolic.solvers.fvm import Mesh, Operator, Scheme
from nozzle_hyperbolic.models.scalar.advection import AdvectionModel


def test_named_default_and_serializable_explicit_flags():
    current = Scheme()
    assert current.flux == "stadtke"
    assert current.custom_jump_correction is True
    assert current.custom_endpoint_envelope is True
    assert Scheme(custom_jump_correction=False).custom_endpoint_envelope is True
    assert Scheme(custom_endpoint_envelope=False).custom_jump_correction is True
    for jump, endpoint in product((False, True), repeat=2):
        scheme = Scheme(custom_jump_correction=jump, custom_endpoint_envelope=endpoint)
        assert asdict(scheme)["custom_jump_correction"] is jump
        assert asdict(scheme)["custom_endpoint_envelope"] is endpoint
        assert Scheme(**asdict(scheme)) == scheme
    assert Scheme("constant", "stadtke", 0.2).delta_factor == 0.2


def test_removed_public_flux_name_is_rejected():
    with pytest.raises(ValueError, match="Unsupported numerical flux"):
        Scheme(flux="characteristic")


@pytest.mark.parametrize(
    "field", ["custom_jump_correction", "custom_endpoint_envelope"]
)
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_correction_flags_require_booleans(field, invalid):
    with pytest.raises(ValueError, match=field):
        Scheme(**{field: invalid})


def _algebra_inputs():
    # Dense complementary projectors in a conservative frame unrelated to Euler.
    basis = np.array([[1.0, 0.4], [-0.3, 1.2]])
    inverse = np.linalg.inv(basis)
    projectors = np.stack([basis[:, k, None] * inverse[None, k, :] for k in range(2)])[
        None
    ]
    return tuple(
        map(
            jnp.asarray,
            (
                [[0.7, -1.3]],
                [[-0.2, 2.1]],
                [[0.8, -0.5]],
                projectors,
                [[-0.3, 0.08]],
                [[0.2, 0.15]],
            ),
        )
    )


@pytest.mark.parametrize("jump,endpoint", product((False, True), repeat=2))
def test_four_corrections_match_independent_additive_identity(jump, endpoint):
    fl, fr, du, p, lam, width = _algebra_inputs()
    side_speeds = (jnp.asarray([[-1.2, 0.6]]), jnp.asarray([[-0.5, 0.9]]))
    old = stadtke_flux_with_entropy_viscosity(fl, fr, du, p, lam, width)
    phi = harten_absolute(lam, width)
    projected_u = jnp.einsum("...kij,...j->...ki", p, du)
    projected_f = jnp.einsum("...kij,...j->...ki", p, fr - fl)
    correction = 0.5 * jnp.sum(
        (lam / phi)[..., None] * (projected_f - lam[..., None] * projected_u), axis=-2
    )
    envelope = jnp.maximum(phi, jnp.maximum(abs(side_speeds[0]), abs(side_speeds[1])))
    extra = 0.5 * jnp.sum((envelope - phi)[..., None] * projected_u, axis=-2)
    expected = old + (correction if jump else 0.0) - (extra if endpoint else 0.0)
    actual = stadtke_flux_with_corrections(
        fl,
        fr,
        du,
        p,
        lam,
        width,
        custom_jump_correction=jump,
        custom_endpoint_envelope=endpoint,
        endpoint_eigenvalues=side_speeds,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)
    if not jump and not endpoint:
        np.testing.assert_array_equal(actual, old)
        compiled_old = jax.jit(stadtke_flux_with_entropy_viscosity)(
            fl, fr, du, p, lam, width
        )
        compiled_new = jax.jit(stadtke_flux_with_corrections)(fl, fr, du, p, lam, width)
        np.testing.assert_array_equal(compiled_new, compiled_old)
    if endpoint and not jump:
        combined = stadtke_flux_with_corrections(
            fl,
            fr,
            du,
            p,
            lam,
            width,
            custom_jump_correction=True,
            custom_endpoint_envelope=True,
            endpoint_eigenvalues=side_speeds,
        )
        assert not np.allclose(actual, combined)


def test_jump_correction_vanishes_for_a_finite_jump_linearization():
    fl, _, du, p, lam, width = _algebra_inputs()
    df = jnp.einsum("...k,...kij,...j->...i", lam, p, du)
    expected = stadtke_flux_with_entropy_viscosity(fl, fl + df, du, p, lam, width)
    corrected = stadtke_flux_with_corrections(
        fl, fl + df, du, p, lam, width, custom_jump_correction=True
    )
    np.testing.assert_allclose(corrected, expected, rtol=3e-15, atol=3e-15)
    with pytest.raises(ValueError, match="Endpoint"):
        stadtke_flux_with_corrections(
            fl, fl + df, du, p, lam, width, custom_endpoint_envelope=True
        )


@pytest.mark.parametrize("jump,endpoint", product((False, True), repeat=2))
def test_equal_states_are_consistent(jump, endpoint):
    fl, _, _, p, lam, width = _algebra_inputs()
    result = stadtke_flux_with_corrections(
        fl,
        fl,
        jnp.zeros_like(fl),
        p,
        lam,
        width,
        custom_jump_correction=jump,
        custom_endpoint_envelope=endpoint,
        endpoint_eigenvalues=(lam, lam),
    )
    np.testing.assert_array_equal(result, fl)


class _CubicModel:
    """Scalar nonconvex conservation law with a non-Roe mean eigenspeed."""

    nvar = 1
    equation_names = ("content",)
    variable_scales = conservative_scales = global_flux_scales = np.ones(1)
    reference_time = 1.0
    source_dependency_radius = boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self):
        faces = np.linspace(0.0, 1.0, 5)
        self.mesh = Mesh(
            faces,
            0.5 * (faces[:-1] + faces[1:]),
            np.diff(faces),
            np.linspace(1.0, 1.4, 5),
        )

    def cell_conserved(self, state):
        return 2.5 * state

    def face_conserved(self, state):
        return state

    def face_physical_flux(self, state):
        return state**3 / 3.0

    def face_speed_bounds(self, state):
        return jnp.zeros(state.shape[0]), state[:, 0] ** 2

    def face_characteristics(self, left, right, delta):
        return (
            jnp.ones((left.shape[0], 1, 1, 1)),
            (0.5 * (left + right)) ** 2,
            jnp.full_like(left, delta),
        )

    def face_roe_data(self, left, right, delta):
        return dict(
            projectors=jnp.ones((left.shape[0], 1, 1, 1)),
            eigenvalues=(left**2 + left * right + right**2) / 3.0,
            width=jnp.full_like(left, delta),
            valid=jnp.ones(left.shape[0], dtype=bool),
        )

    def flux_to_conserved_frame(self, flux):
        return 2.5 * flux

    def integrated_source(self, full):
        return jnp.zeros((self.mesh.n_cells, 1))

    def boundary_residual(self, full, time=0.0):
        return jnp.stack((full[0] + 0.9, full[-1] - 1.3))

    def wave_speed(self, state):
        return state[:, 0] ** 2

    def admissible(self, full, left, right):
        return all_finite(full) & all_finite(left) & all_finite(right)

    def initial_state(self):
        return jnp.asarray([[-0.9], [-0.6], [0.2], [0.8], [1.1], [1.3]])


def all_finite(value):
    return jnp.all(jnp.isfinite(value))


class _FusedCubicModel(_CubicModel):
    def face_data(self, state):
        return dict(
            conserved=state,
            physical_flux=state**3 / 3.0,
            speed_bounds=self.face_speed_bounds(state),
            auxiliary=state,
        )

    def face_characteristics_from_data(self, left, right, delta, aux_left, aux_right):
        return _CubicModel.face_characteristics(self, aux_left, aux_right, delta)

    def face_characteristics(self, *args):
        raise AssertionError("fused characteristic callback must be used")


@pytest.mark.parametrize("jump,endpoint", product((False, True), repeat=2))
@pytest.mark.parametrize("model_type", [_CubicModel, _FusedCubicModel])
def test_operator_switches_frame_area_fused_data_and_outer_ad(
    jump, endpoint, model_type
):
    model = model_type()
    scheme = Scheme(custom_jump_correction=jump, custom_endpoint_envelope=endpoint)
    operator = Operator(model, scheme)
    state = model.initial_state()
    left, right = operator.reconstruct(state)
    p, lam, width = _CubicModel.face_characteristics(
        model, left, right, scheme.delta_factor
    )
    expected = (
        stadtke_flux_with_corrections(
            left**3 / 3.0,
            right**3 / 3.0,
            right - left,
            p,
            lam,
            width,
            custom_jump_correction=jump,
            custom_endpoint_envelope=endpoint,
            endpoint_eigenvalues=(left**2, right**2),
        )
        * 2.5
        * jnp.asarray(model.mesh.area_faces)[:, None]
    )
    np.testing.assert_allclose(
        operator.face_flux(state), expected, rtol=3e-15, atol=3e-15
    )
    jacobian = jax.jit(jax.jacfwd(operator.face_flux))(state)
    assert np.isfinite(np.asarray(jacobian)).all()
    direction = jnp.asarray([[0.1], [-0.2], [0.3], [-0.1], [0.15], [-0.05]])
    h = 2e-6
    fd = (
        operator.face_flux(state + h * direction)
        - operator.face_flux(state - h * direction)
    ) / (2 * h)
    ad = jnp.einsum("aibj,bj->ai", jacobian, direction)
    np.testing.assert_allclose(ad, fd, rtol=2e-8, atol=2e-10)
    if not jump and not endpoint:

        def original_assembly(full):
            l, r = full[:-1], full[1:]
            if callable(getattr(model, "face_data", None)):
                dl, dr = model.face_data(l), model.face_data(r)
                ql, qr = dl["conserved"], dr["conserved"]
                fl, fr = dl["physical_flux"], dr["physical_flux"]
                eig = model.face_characteristics_from_data(
                    l, r, scheme.delta_factor, dl["auxiliary"], dr["auxiliary"]
                )
            else:
                ql, qr = model.face_conserved(l), model.face_conserved(r)
                fl, fr = model.face_physical_flux(l), model.face_physical_flux(r)
                eig = model.face_characteristics(l, r, scheme.delta_factor)
            flux = stadtke_flux_with_entropy_viscosity(fl, fr, qr - ql, *eig)
            return jnp.asarray(model.mesh.area_faces)[
                :, None
            ] * model.flux_to_conserved_frame(flux)

        np.testing.assert_array_equal(
            operator.face_flux(state), jax.jit(original_assembly)(state)
        )


@pytest.mark.parametrize("flux", ["rusanov", "hll", "roe"])
def test_other_fluxes_ignore_custom_flags(flux):
    model = _CubicModel()
    state = model.initial_state()
    off = Operator(
        model,
        Scheme(flux=flux, custom_jump_correction=False, custom_endpoint_envelope=False),
    )
    on = Operator(
        model,
        Scheme(flux=flux, custom_jump_correction=True, custom_endpoint_envelope=True),
    )
    np.testing.assert_array_equal(off.face_flux(state), on.face_flux(state))


def test_stadtke_requires_model_capability():
    with pytest.raises(ValueError, match="characteristic"):
        Operator(AdvectionModel(4), Scheme(flux="stadtke"))


class _WideLinearModel(_CubicModel):
    """Physical speed one, but selected Harten viscosity seventeen eighths."""

    def __init__(self, physical_face_bound=1.0):
        super().__init__()
        self.physical_face_bound = physical_face_bound

    def face_physical_flux(self, state):
        return state

    def face_speed_bounds(self, state):
        speed = jnp.full(state.shape[0], self.physical_face_bound)
        return -speed, speed

    def face_characteristics(self, left, right, delta):
        return (
            jnp.ones((left.shape[0], 1, 1, 1)),
            jnp.ones_like(left),
            jnp.full_like(left, 4.0),
        )

    def wave_speed(self, state):
        return jnp.ones(state.shape[0])


@pytest.mark.parametrize("jump,endpoint", product((False, True), repeat=2))
def test_explicit_stadtke_bounds_selected_viscosity_but_preserves_off_dt(
    jump, endpoint
):
    model = _WideLinearModel()
    operator = Operator(
        model,
        Scheme(
            time_integrator="euler-explicit",
            custom_jump_correction=jump,
            custom_endpoint_envelope=endpoint,
        ),
    )
    info = operator.timestep(model.initial_state(), 0.4)
    speed = 17.0 / 8.0 if jump or endpoint else 1.0
    old_cell_bound = jnp.min(jnp.asarray(model.mesh.widths) / speed)
    assert bool(info.valid)
    np.testing.assert_array_equal(info.wave_dt, old_cell_bound)
    np.testing.assert_array_equal(info.dt, 0.4 * old_cell_bound)


def test_stadtke_explicit_viscosity_guard_keeps_endpoint_physical_bounds():
    model = _WideLinearModel(physical_face_bound=3.0)
    operator = Operator(model, Scheme(flux="stadtke", time_integrator="euler-explicit"))
    info = operator.timestep(model.initial_state(), 0.4)
    expected = np.min(model.mesh.widths) / 3.0
    assert bool(info.valid)
    np.testing.assert_allclose(info.wave_dt, expected, rtol=2e-15)
    np.testing.assert_allclose(info.dt, 0.4 * expected, rtol=2e-15)
