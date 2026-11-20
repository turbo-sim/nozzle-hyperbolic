"""Closure scales, correction factors and run-time two-fluid inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.two_fluid import (
    BoundaryConfig,
    ClosureConfig,
    primitives,
    from_primitives,
)
from nozzle_hyperbolic.models.two_fluid.sources import two_fluid_closure
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme

from tests._support.two_fluid import model, physical_data

SCHEME = Scheme(flux="model_compatibility")


def closure(alpha, slip=12.0, **options):
    U = jnp.array((2e5, 20.0 + slip, 20.0, alpha, 3.1e5, 2.8e5))
    fluid, eg, el, _ = physical_data(U)
    return two_fluid_closure(U, eg, el, fluid, fluid, 260.0, 5e5, 2.5e5, **options)


@pytest.mark.parametrize("alpha", (0.2, 0.85))
def test_drag_scales_act_per_regime_and_beta_cd_on_both(alpha):
    options = dict(mass_transfer=False, area_law="original")
    base = closure(alpha, **options).Fv_g
    bubbly = closure(alpha, drag_scale_bubbly=0.4, **options).Fv_g
    droplet = closure(alpha, drag_scale_droplet=2.5, **options).Fv_g
    np.testing.assert_allclose(bubbly, base * (0.4 if alpha < 0.3 else 1.0), rtol=1e-13)
    np.testing.assert_allclose(droplet, base * (2.5 if alpha > 0.7 else 1.0), rtol=1e-13)
    np.testing.assert_allclose(closure(alpha, beta_cd=1.7, **options).Fv_g, 1.7 * base, rtol=1e-13)


def test_htc_factors_scale_heat_and_nu_l_int_acts_on_droplets():
    options = dict(mass_transfer=False, area_law="original", regime_htc=True)
    base = closure(0.2, **options).sigma_Qg
    scaled = closure(0.2, beta_htc_g=1.3, beta_htc_l=1.3, **options).sigma_Qg
    np.testing.assert_allclose(scaled, 1.3 * base, rtol=1e-13)
    assert closure(0.2, beta_htc_l=0.5, **options).sigma_Qg != base
    np.testing.assert_array_equal(closure(0.2, nu_l_int=30.0, **options).sigma_Qg, base)
    droplet = closure(0.85, **options).sigma_Qg
    assert closure(0.85, nu_l_int=30.0, **options).sigma_Qg != droplet


@pytest.mark.parametrize("option", ("htc_liquid_bubbly", "htc_liquid_droplet"))
def test_sum_adds_the_bubble_growth_term(option):
    alpha = 0.2 if option == "htc_liquid_bubbly" else 0.85
    options = dict(mass_transfer=True, area_law="original", regime_htc=True)
    base = closure(alpha, **options)
    summed = closure(alpha, **{option: "sum"}, **options)
    # Superheated liquid: the growth term increases the liquid-side heat flux.
    assert abs(summed.sigma_Ql) > abs(base.sigma_Ql)


def test_beta_ai_scales_area_before_the_diameter():
    options = dict(mass_transfer=False, area_law="parabola")
    assert closure(0.5, beta_ai=2.0, **options).Fv_g != closure(0.5, **options).Fv_g


@pytest.mark.parametrize(
    "options",
    (
        dict(drag_scale_bubbly=0.0),
        dict(drag_scale_droplet=-1.0),
        dict(beta_cd=0.0),
        dict(beta_htc_g=-0.5),
        dict(beta_htc_l=0.0),
        dict(nu_l_int=0.0),
        dict(mass_transfer=False, htc_liquid_bubbly="sum"),
    ),
)
def test_invalid_scales_and_factors_rejected(options):
    with pytest.raises(ValueError):
        ClosureConfig(**options)


def corrected_model(runtime_inputs=True, **closure_options):
    return model(
        boundaries=BoundaryConfig(outlet="characteristic", back_pressure=1.9e5),
        closures=ClosureConfig(mass_transfer=False, area_law="original", **closure_options),
        runtime_inputs=runtime_inputs,
    )


def slipping_state(m):
    U = primitives(m.initial_state())
    U = U.at[:, 1].add(jnp.linspace(8.0, 14.0, U.shape[0]))
    return from_primitives(U, m.fluid_g, m.fluid_l)


def test_runtime_inputs_are_opt_in_and_follow_the_configuration():
    assert corrected_model(runtime_inputs=False).parameter_defaults() == {}
    assert not Operator(corrected_model(runtime_inputs=False), SCHEME).has_parameters
    assert model(runtime_inputs=True).parameter_defaults() == {}
    values = corrected_model().parameter_defaults()
    assert set(values) == {"beta_ai", "beta_cd", "beta_htc_g", "beta_htc_l", "back_pressure"}
    assert values["beta_cd"].shape == (3,) and float(values["back_pressure"]) == 1.9e5


def test_default_runtime_inputs_reproduce_the_configuration():
    m = corrected_model(beta_cd=1.4)
    W = slipping_state(m)
    defaults = m.parameter_defaults()
    # Inputs instead of compiled constants: equal to roundoff.
    np.testing.assert_allclose(m.integrated_source(W, defaults), m.integrated_source(W), rtol=1e-15)
    np.testing.assert_array_equal(m.boundary_residual(W, 0.0, defaults), m.boundary_residual(W, 0.0))


def test_uniform_fields_equal_configured_factors_and_act_per_cell():
    configured = corrected_model(beta_ai=0.8, beta_cd=1.7, beta_htc_g=0.6, beta_htc_l=1.3)
    base = corrected_model()
    W = slipping_state(base)
    ones = jnp.ones(3)
    fields = dict(beta_ai=0.8 * ones, beta_cd=1.7 * ones, beta_htc_g=0.6 * ones, beta_htc_l=1.3 * ones)
    np.testing.assert_allclose(base.integrated_source(W, fields), configured.integrated_source(W), rtol=1e-14)
    defaults = base.parameter_defaults()
    local = dict(defaults, beta_cd=jnp.array((1.0, 2.0, 1.0)))
    change = np.abs(np.asarray(base.integrated_source(W, local) - base.integrated_source(W, defaults)))
    assert change[1].max() > 0 and change[[0, 2]].max() == 0


def test_runtime_back_pressure_replaces_the_configured_value():
    m = corrected_model()
    W = slipping_state(m)
    reference = model(
        boundaries=BoundaryConfig(outlet="characteristic", back_pressure=1.7e5),
        closures=ClosureConfig(mass_transfer=False, area_law="original"),
    )
    values = dict(m.parameter_defaults(), back_pressure=jnp.asarray(1.7e5))
    np.testing.assert_allclose(m.boundary_residual(W, 0.0, values), reference.boundary_residual(W, 0.0),
                               rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("values", (dict(beta_cd=-jnp.ones(3)), dict(back_pressure=0.0)))
def test_nonpositive_runtime_inputs_rejected(values):
    op = Operator(corrected_model(), SCHEME)
    with pytest.raises(ValueError, match="positive"):
        op.parameters(values)


def test_runtime_input_derivatives_are_finite_and_nonzero():
    m = corrected_model()
    op = Operator(m, SCHEME)
    W = slipping_state(m).ravel()

    def residual(beta_cd, back_pressure):
        return op.steady_residual(W, params=dict(beta_cd=beta_cd, back_pressure=back_pressure))

    d_field, d_pressure = jax.jacfwd(residual, argnums=(0, 1))(jnp.ones(3), jnp.asarray(1.9e5))
    for derivative in (d_field, d_pressure):
        assert np.isfinite(np.asarray(derivative)).all()
        assert np.abs(np.asarray(derivative)).max() > 0
