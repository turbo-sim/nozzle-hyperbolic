"""Independent finite-jump, general-EOS and differentiation audits for Roe."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import radial_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.models.euler_roe import euler_roe_data
from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid


class NobleAbelFluid:
    """Thermodynamically consistent nonlinear covolume EOS, PH/PT interface.

    Constants are intentionally private to this independent fixture. Density
    dependent G and nonideal sound speed expose ideal-gas averaging shortcuts.
    """

    def __init__(self, shift=0.0):
        self.shift = shift
        self.calls = []

    def get_state(self, pair, first, second):
        self.calls.append(pair)
        gamma, cp, b = 1.4, 1004.5, 0.08
        gas_constant = cp * (gamma - 1) / gamma
        if pair == jxp.HmassP_INPUTS:
            h, p = first, second
            temperature = (h - self.shift - b * p) / cp
        elif pair == jxp.PT_INPUTS:
            p, temperature = first, second
            h = cp * temperature + b * p + self.shift
        else:
            raise AssertionError("Only setup PT and numerical PH calls are permitted")
        rho = p / (gas_constant * temperature + b * p)
        free_volume = 1 - b * rho
        return dict(
            p=p,
            h=h,
            T=temperature,
            rho=rho,
            a=jnp.sqrt(gamma * p / (rho * free_volume)),
            G=(gamma - 1) / free_volume,
            s=cp * jnp.log(temperature / 300) - gas_constant * jnp.log(p / 1e5),
            mu=jnp.ones_like(p) * 1.8e-5,
        )


def _reference(fluid):
    state = fluid.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    return tuple(state[key] for key in ("rho", "p", "a", "h"))


def _data(fluid, left, right, radius=None, delta=0.05):
    thermo = lambda w: fluid.get_state(jxp.HmassP_INPUTS, w[..., -1], w[..., 0])
    return euler_roe_data(
        left,
        right,
        thermo(left),
        thermo(right),
        _reference(fluid),
        delta,
        radius=radius,
    )


def _qf(fluid, state, radius=None):
    p, u, h = state[..., 0], state[..., 1], state[..., -1]
    rho = fluid.get_state(jxp.HmassP_INPUTS, h, p)["rho"]
    v = 0.0 if radius is None else state[..., 2] / radius
    total_h = h + 0.5 * (u**2 + v**2)
    q = [rho, rho * u]
    flux = [rho * u, rho * u * u + p]
    if radius is not None:
        q.append(rho * v)
        flux.append(rho * u * v)
    return jnp.stack((*q, rho * total_h - p), -1), jnp.stack(
        (*flux, rho * u * total_h), -1
    )


@pytest.mark.parametrize("annular", [False, True])
def test_nonlinear_eos_finite_jump_and_grouped_projector_identities(annular):
    fluid = NobleAbelFluid()
    rng = np.random.default_rng(718)
    left = np.column_stack(
        (
            rng.uniform(0.6e5, 2.4e5, 32),
            rng.uniform(-250, 650, 32),
            rng.uniform(2.3e5, 4e5, 32),
        )
    )
    right = np.column_stack(
        (
            rng.uniform(0.6e5, 2.4e5, 32),
            rng.uniform(-250, 650, 32),
            rng.uniform(2.3e5, 4e5, 32),
        )
    )
    radius = None
    if annular:
        radius = jnp.linspace(0.1, 0.3, 32)
        left = np.insert(left, 2, rng.uniform(-10, 10, 32), axis=-1)
        right = np.insert(right, 2, rng.uniform(-10, 10, 32), axis=-1)
    left, right = jnp.asarray(left), jnp.asarray(right)
    data = _data(fluid, left, right, radius)
    assert np.all(data["valid"])
    q_l, f_l = _qf(fluid, left, radius)
    q_r, f_r = _qf(fluid, right, radius)
    scale = np.array([1.0, 400.0] + ([400.0] if annular else []) + [1.6e5])
    np.testing.assert_allclose(
        (jnp.einsum("...ij,...j->...i", data["matrix"], q_r - q_l) - (f_r - f_l))
        / scale,
        0,
        atol=3e-12,
    )
    projectors, speeds = data["projectors"], data["eigenvalues"]
    normalized = scale[None, :] / scale[:, None]
    np.testing.assert_allclose(
        (projectors.sum(-3) - np.eye(len(scale))) * normalized, 0, atol=3e-15
    )
    for i in range(3):
        p = projectors[:, i]
        np.testing.assert_allclose((p @ p - p) * normalized, 0, atol=2e-14)
        np.testing.assert_allclose(
            (data["matrix"] @ p - speeds[:, i, None, None] * p) * normalized,
            0,
            atol=3e-11,
        )
    np.testing.assert_allclose(
        jnp.trace(projectors[..., 2, :, :], axis1=-2, axis2=-1),
        2 if annular else 1,
        atol=3e-14,
    )
    np.testing.assert_array_equal(data["width"][..., 2], 0.0)


@pytest.mark.parametrize(
    "radius,state", [(None, [1.3e5, 0.0, 3.2e5]), (0.17, [1.3e5, 350.0, -7.0, 3.2e5])]
)
def test_coincident_matrix_matches_independent_eos_jacobian_and_has_finite_ad(
    radius, state
):
    fluid, state = NobleAbelFluid(), jnp.asarray(state)
    q_jac = jax.jacfwd(lambda w: _qf(fluid, w, radius)[0])(state)
    f_jac = jax.jacfwd(lambda w: _qf(fluid, w, radius)[1])(state)
    expected = f_jac @ jnp.linalg.inv(q_jac)
    data = _data(fluid, state, state, radius)
    scale = jnp.asarray([1.0, 400.0] + ([400.0] if radius else []) + [1.6e5])
    np.testing.assert_allclose(
        (data["matrix"] - expected) * scale[None, :] / scale[:, None], 0, atol=3e-11
    )
    function = lambda pair: _data(fluid, pair[0], pair[1], radius)["matrix"]
    pair = jnp.stack((state, state))
    forward, reverse = jax.jacfwd(function)(pair), jax.jacrev(function)(pair)
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, rtol=2e-11, atol=2e-10)
    # Symmetry requires each endpoint derivative to contribute half the
    # derivative of the coincident physical Jacobian.
    np.testing.assert_allclose(
        forward[..., 0, :], forward[..., 1, :], rtol=3e-12, atol=3e-10
    )


@pytest.mark.parametrize("radius", [None, 0.19])
def test_arbitrary_enthalpy_reference_shift_is_a_similarity(radius):
    fluid, shifted = NobleAbelFluid(), NobleAbelFluid(-7.3e5)
    left = jnp.asarray(
        [1e5, -15.0, 2.8e5] if radius is None else [1e5, -15.0, 3.0, 2.8e5]
    )
    right = jnp.asarray(
        [1.8e5, 480.0, 3.4e5] if radius is None else [1.8e5, 480.0, -9.0, 3.4e5]
    )
    a = _data(fluid, left, right, radius)
    b = _data(
        shifted, left.at[-1].add(shifted.shift), right.at[-1].add(shifted.shift), radius
    )
    transform = np.eye(len(left))
    transform[-1, 0] = shifted.shift
    inverse = np.eye(len(left))
    inverse[-1, 0] = -shifted.shift
    np.testing.assert_allclose(a["eigenvalues"], b["eigenvalues"], rtol=4e-15)
    scale = np.array([1.0, 400.0] + ([400.0] if radius else []) + [1.6e5])
    np.testing.assert_allclose(
        (b["matrix"] - transform @ a["matrix"] @ inverse)
        * scale[None, :]
        / scale[:, None],
        0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        (b["projectors"] - transform @ a["projectors"] @ inverse)
        * scale[None, :]
        / scale[:, None],
        0,
        atol=5e-14,
    )


def test_tiny_jump_and_c2_transition_jvp_are_resolved_and_secant_audited():
    fluid = NobleAbelFluid()
    state = jnp.asarray([1.2e5, 100.0, 3e5])
    direction = jnp.asarray([1e5, 80.0, 2e5])
    # Include coincidence, unresolvable differences, transition and finite jumps.
    values = jnp.asarray([0.0, 1e-12, 1e-9, 2e-6, 4e-6, 8e-6, 1e-4])
    evaluate = lambda step: _data(fluid, state, state + step * direction)
    result = jax.jit(jax.vmap(evaluate))(values)
    assert np.all(result["valid"])
    assert np.max(np.abs(result["pressure_secant_defect"])) < 2e-14
    reference = _reference(fluid)
    rho_ref, p_ref, a_ref, h_ref = reference
    energy_ref = h_ref - p_ref / rho_ref
    coordinates = lambda t: jnp.stack(
        (
            fluid.get_state(
                jxp.HmassP_INPUTS,
                (state + t * direction)[-1],
                (state + t * direction)[0],
            )["rho"]
            / rho_ref,
            (
                _qf(fluid, state + t * direction)[0][-1]
                - 0.5
                * _qf(fluid, state + t * direction)[0][0]
                * (state + t * direction)[1] ** 2
                - energy_ref * _qf(fluid, state + t * direction)[0][0]
            )
            / (rho_ref * a_ref**2),
        )
    )
    coordinate_speed = jnp.linalg.norm(jax.jacfwd(coordinates)(jnp.asarray(0.0)))
    transition_step = float(jnp.finfo(jnp.float64).eps ** (1 / 3) / coordinate_speed)
    for step in (
        0.0,
        1.05 * transition_step,
        1.4 * transition_step,
        1.95 * transition_step,
    ):
        function = lambda t: evaluate(t)["matrix"]
        tangent = jax.jvp(function, (jnp.asarray(step),), (jnp.asarray(1.0),))[1]
        fd_step = 0.1 * transition_step
        fd = (function(step + fd_step) - function(step - fd_step)) / (2 * fd_step)
        scale = np.array([1.0, 400.0, 1.6e5])
        normalized = scale[None, :] / scale[:, None]
        # Differencing a projected machine-size secant amplifies value noise.
        # A spacing of 0.1*tau resolves the transition while remaining above
        # its observed FP noise floor (smaller spacings were also audited).
        assert np.max(np.abs((tangent - fd) * normalized)) < 1e-5 * np.max(
            np.abs(tangent * normalized)
        )


@pytest.mark.parametrize("radius", [None, 0.17])
def test_stationary_contact_and_shear_have_no_acoustic_dissipation(radius):
    fluid = NobleAbelFluid()
    left = jnp.asarray([1e5, 0.0, 2.8e5] if radius is None else [1e5, 0.0, 5.0, 2.8e5])
    right = jnp.asarray(
        [1e5, 0.0, 3.4e5] if radius is None else [1e5, 0.0, -3.0, 3.4e5]
    )
    data = _data(fluid, left, right, radius)
    q_l, _ = _qf(fluid, left, radius)
    q_r, _ = _qf(fluid, right, radius)
    scale = np.array([1.0, 400.0] + ([400.0] if radius else []) + [1.6e5])
    acoustic_jump = jnp.einsum(
        "...kij,...j->...ki", data["projectors"][..., :2, :, :], q_r - q_l
    )
    np.testing.assert_allclose(acoustic_jump / scale, 0, atol=2e-15)
    assert data["eigenvalues"][-1] == 0 and data["width"][-1] == 0


def test_pure_velocity_jump_classical_perfect_gas_limit_and_zero_entropy_width():
    fluid = jxp.FluidPerfectGas(name="Air")
    state = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    left = jnp.asarray([state["p"], -200.0, state["h"]])
    right = left.at[1].set(600.0)
    data = _data(fluid, left, right, delta=0.0)
    total_h = state["h"] + 0.25 * (left[1] ** 2 + right[1] ** 2)
    expected_a2 = state["G"] * (total_h - 0.5 * (0.5 * (left[1] + right[1])) ** 2)
    np.testing.assert_allclose(data["sound_speed_squared"], expected_a2, rtol=5e-15)
    np.testing.assert_allclose(data["chi"], 0, atol=2e-10)
    coincident = lambda pair: _data(fluid, pair[0], pair[1], delta=0.0)["width"]
    pair = jnp.stack((left, left))
    np.testing.assert_array_equal(coincident(pair), 0.0)
    assert np.all(np.isfinite(jax.jacfwd(coincident)(pair)))
    assert np.all(np.isfinite(jax.jacrev(coincident)(pair)))


def test_inactive_subnormal_secant_denominator_keeps_forward_and_reverse_ad_finite():
    fluid = NobleAbelFluid()
    left = jnp.asarray([1e-100, 100.0, 3e5])
    right = jnp.asarray([1e-100 * (1 + 1e-5), 110.0, 3e5])
    function = lambda pair: _data(fluid, pair[0], pair[1])["matrix"]
    pair = jnp.stack((left, right))
    assert np.all(np.isfinite(function(pair)))
    assert np.all(np.isfinite(jax.jacfwd(function)(pair)))
    assert np.all(np.isfinite(jax.jacrev(function)(pair)))


def test_invalid_roe_sound_speed_is_reported_without_hidden_clipping():
    fluid = NobleAbelFluid()
    left, right = jnp.asarray([1e5, -1000.0, 3e5]), jnp.asarray([1e5, 1000.0, 3e5])
    thermo = fluid.get_state(jxp.HmassP_INPUTS, left[-1], left[0])
    thermo = dict(thermo, G=jnp.asarray(-1.0))
    data = euler_roe_data(left, right, thermo, thermo, _reference(fluid), 0.1)
    assert data["sound_speed_squared"] < 0
    assert not bool(data["valid"])


@pytest.mark.parametrize("kind", ["nozzle", "annular"])
def test_model_fused_callback_reuses_ph_data_and_preserves_characteristics(kind):
    fluid = NobleAbelFluid()
    if kind == "nozzle":
        model = NozzleModel(
            ConstantAreaGeometry().build_mesh(4), fluid, back_pressure=1.7e5
        )
        left = jnp.tile(jnp.asarray([1e5, 30.0, 2.8e5]), (5, 1))
        right = left.at[:, 0].set(1.4e5).at[:, 1].set(300.0).at[:, -1].set(3.3e5)
        scale = np.array([1.0, 400.0, 1.6e5])
    else:
        mesh, native = build_annular_mesh(radial_channel(0.1, 0.2, 0.005), 4)
        model = AnnularModel(
            mesh,
            native,
            fluid,
            inlet_data=dict(p0=2e5, T0=300.0, alpha_in=0.2),
            outlet_pressure=1.7e5,
        )
        left = jnp.tile(jnp.asarray([1e5, 30.0, 4.0, 2.8e5]), (5, 1))
        right = (
            left.at[:, 0]
            .set(1.4e5)
            .at[:, 1]
            .set(300.0)
            .at[:, 2]
            .set(-6.0)
            .at[:, -1]
            .set(3.3e5)
        )
        scale = np.array([1.0, 400.0, 400.0, 1.6e5])
    fluid.calls.clear()
    l_data, r_data = model.face_data(left), model.face_data(right)
    assert fluid.calls == [jxp.HmassP_INPUTS, jxp.HmassP_INPUTS]
    fused = model.face_roe_data_from_data(
        left, right, 0.05, l_data["auxiliary"], r_data["auxiliary"]
    )
    assert len(fluid.calls) == 2
    direct = model.face_roe_data(left, right, 0.05)
    for key in ("projectors", "eigenvalues", "width", "valid"):
        np.testing.assert_array_equal(fused[key], direct[key])
    old = model.face_characteristics(left, right, 0.05)
    reused = model.face_characteristics_from_data(
        left, right, 0.05, l_data["auxiliary"], r_data["auxiliary"]
    )
    for actual, expected in zip(reused, old):
        np.testing.assert_array_equal(actual, expected)
    jump = jnp.einsum(
        "...ij,...j->...i", fused["matrix"], r_data["conserved"] - l_data["conserved"]
    )
    np.testing.assert_allclose(
        (jump - (r_data["physical_flux"] - l_data["physical_flux"])) / scale,
        0,
        atol=2e-12,
    )
    if kind == "annular":
        transported = model.flux_to_conserved_frame(jump)
        np.testing.assert_allclose(
            transported[:, 2], jump[:, 2] * model._r_face, rtol=1e-15
        )


@pytest.mark.parametrize("radius", [None, 0.17])
def test_actual_co2_ph_backend_finite_jump_and_coincident_jacobian(radius):
    fluid = as_jax_fluid(jxp.Fluid("CO2", backend="HEOS"))
    left_thermo = fluid.get_state(jxp.PT_INPUTS, 3e6, 320.0)
    right_thermo = fluid.get_state(jxp.PT_INPUTS, 8e6, 350.0)
    left = jnp.asarray(
        [3e6, 45.0, left_thermo["h"]]
        if radius is None
        else [3e6, 45.0, 3.0, left_thermo["h"]]
    )
    right = jnp.asarray(
        [8e6, 230.0, right_thermo["h"]]
        if radius is None
        else [8e6, 230.0, -5.0, right_thermo["h"]]
    )
    reference = tuple(right_thermo[key] for key in ("rho", "p", "a", "h"))
    thermo = lambda w: fluid.get_state(jxp.HmassP_INPUTS, w[-1], w[0])
    data = euler_roe_data(
        left, right, thermo(left), thermo(right), reference, 0.05, radius=radius
    )
    assert bool(data["valid"])
    assert abs(float(thermo(left)["G"]) - float(thermo(right)["G"])) > 0.01
    q_l, f_l = _qf(fluid, left, radius)
    q_r, f_r = _qf(fluid, right, radius)
    np.testing.assert_allclose(
        data["matrix"] @ (q_r - q_l), f_r - f_l, rtol=5e-13, atol=1e-6
    )
    q_jac = jax.jacfwd(lambda w: _qf(fluid, w, radius)[0])(right)
    f_jac = jax.jacfwd(lambda w: _qf(fluid, w, radius)[1])(right)
    expected = f_jac @ jnp.linalg.inv(q_jac)
    coincident = euler_roe_data(
        right, right, thermo(right), thermo(right), reference, 0.05, radius=radius
    )
    scale = np.array([1.0, 400.0] + ([400.0] if radius else []) + [1.6e5])
    # The configured JAXProp host bridge differentiates properties by FD.
    np.testing.assert_allclose(
        (coincident["matrix"] - expected) * scale[None, :] / scale[:, None],
        0,
        atol=0.008,
    )
