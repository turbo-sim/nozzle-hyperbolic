"""Constitutive limits, independent units/conventions, and traced behavior."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_nozzle.closures import (
    Haaland,
    HaalandWall,
    LM_condensation,
    LM_gronnerud,
    PrescribedWall,
    darcy_friction_factor,
    f_HEM_purdue,
    haaland_wall_shear,
    richardson,
    viscosity_Beattie,
    viscosity_MacAdams,
)


def test_annular_wall_exports_reuse_nozzle_implementations():
    from nozzle_hyperbolic.models import euler_annular
    from nozzle_hyperbolic.models.euler_annular import closures as annular
    from nozzle_hyperbolic.models.euler_nozzle import closures as nozzle

    for name in (
        "Haaland",
        "HaalandWall",
        "PrescribedWall",
        "darcy_friction_factor",
        "haaland_wall_shear",
        "heat_flux_wall",
        "shear_stress_wall",
    ):
        assert getattr(annular, name) is getattr(nozzle, name)
    assert euler_annular.heat_flux_wall is nozzle.heat_flux_wall
    assert euler_annular.shear_stress_wall is nozzle.shear_stress_wall


class _TransportEOS:
    """Deliberately variable density/viscosity supplied only by get_state."""

    def get_state(self, pair, h, p):
        assert pair == jxp.HmassP_INPUTS
        return {"density": 3.0 + 1e-6 * p + 1e-7 * h, "viscosity": 0.02 + 1e-9 * h}


def test_laminar_shear_has_exact_stokes_limit_and_signed_derivative():
    rho, mu, diameter = 1.3, 1.8e-5, 0.03
    stress = jax.jit(lambda u: haaland_wall_shear(u, rho, mu, diameter, 1e-5))
    velocities = jnp.asarray([-1e-8, -1e-20, 0.0, 1e-20, 1e-8])
    np.testing.assert_allclose(
        stress(velocities), 8.0 * mu * velocities / diameter, rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(stress))(0.0), 8.0 * mu / diameter, rtol=2e-15
    )
    np.testing.assert_allclose(stress(-0.1), -stress(0.1), rtol=2e-15)


def test_darcy_fanning_conventions_and_roughness_response():
    reynolds = jnp.asarray([1e-12, 1.0, 100.0, 1000.0])
    np.testing.assert_allclose(
        darcy_friction_factor(reynolds, 0.0, 0.1),
        4.0 * f_HEM_purdue(reynolds),
        rtol=2e-15,
    )
    assert np.isinf(darcy_friction_factor(0.0, 0.0, 0.1))
    assert np.isinf(f_HEM_purdue(0.0))
    assert np.isnan(darcy_friction_factor(-1.0, 0.0, 0.1))
    assert float(Haaland(1e5, 1e-3, 0.1)) > float(Haaland(1e5, 0.0, 0.1))
    assert float(Haaland(0.0, 0.0, 0.1)) == 0.0  # retained historical API
    np.testing.assert_allclose(
        darcy_friction_factor(1e5, 1e-4, 0.1), Haaland(1e5, 1e-4, 0.1)
    )


@pytest.mark.parametrize("annular", [False, True])
def test_wall_policy_uses_eos_transport_and_has_finite_zero_speed_jacobian(annular):
    fluid, radius = _TransportEOS(), 0.4
    physical = jnp.asarray([2e5, 0.0, 0.0, 4e5] if annular else [2e5, 0.0, 4e5])
    wall = HaalandWall(0.05, heat_flux=13.0)
    density, viscosity = 3.0 + 0.2 + 0.04, 0.02 + 0.0004
    tau, heat = jax.jit(lambda w: wall(w, radius, fluid))(physical)
    assert float(tau) == 0.0
    assert float(heat) == 13.0
    np.testing.assert_allclose(
        wall.drag_coefficient(physical, radius, fluid), 8.0 * viscosity / 0.05
    )
    jac = jax.jit(jax.jacfwd(lambda w: wall.drag_coefficient(w, radius, fluid)))(
        physical
    )
    assert np.all(np.isfinite(jac))
    np.testing.assert_allclose(jac[-1], 8e-9 / 0.05)
    bound = jax.jit(lambda w: wall.explicit_timestep(w, radius, fluid, 80.0))(physical)
    np.testing.assert_allclose(bound, density / (80.0 * 8.0 * viscosity / 0.05))
    assert np.isinf(wall.explicit_timestep(physical, radius, fluid, 0.0))


def test_annular_drag_is_isotropic_and_drag_bound_controls_decay():
    fluid = _TransportEOS()
    wall = HaalandWall(0.1, roughness=1e-4)
    # Same speed in meridional and circumferential directions, arbitrary EOS.
    meridional = jnp.asarray([2e5, 200.0, 0.0, 4e5])
    swirl = jnp.asarray([2e5, 0.0, 80.0, 4e5])
    tau_m, _ = wall(meridional, 0.4, fluid)
    tau_t, _ = wall(swirl, 0.4, fluid)
    np.testing.assert_allclose(tau_m, tau_t)
    timestep = wall.explicit_timestep(meridional, 0.4, fluid, 40.0)
    rho = fluid.get_state(jxp.HmassP_INPUTS, 4e5, 2e5)["density"]
    next_speed = 200.0 - timestep * 40.0 * tau_m / rho
    assert 0.0 <= float(next_speed) < 200.0
    assert np.all(
        np.isfinite(
            jax.jacfwd(lambda w: wall.explicit_timestep(w, 0.4, fluid, 40.0))(
                meridional
            )
        )
    )


def test_prescribed_profiles_and_validation():
    physical = jnp.asarray([[1e5, 1.0, 3e5], [2e5, 2.0, 4e5]])
    wall = PrescribedWall([2.0, 3.0], -7.0)
    tau, heat = jax.jit(lambda w: wall(w, None, None))(physical)
    np.testing.assert_array_equal(tau, [2.0, 3.0])
    np.testing.assert_array_equal(heat, [-7.0, -7.0])
    for kwargs in ({"tau_w": -1}, {"q_w": np.nan}):
        with pytest.raises(ValueError):
            PrescribedWall(**kwargs)
    with pytest.raises(ValueError):
        HaalandWall(0)


def test_mixture_pure_phase_limits_and_condensation_reference_identity():
    mu_l, mu_v, rho_l, rho_v = 1e-3, 1e-5, 1000.0, 2.0
    for law, args in (
        (viscosity_Beattie, (mu_l, mu_v, rho_l, rho_v)),
        (viscosity_MacAdams, (mu_l, mu_v)),
    ):
        values = jax.jit(lambda x: law(*args, x))(jnp.asarray([0.0, 1.0]))
        np.testing.assert_allclose(values, [mu_l, mu_v], rtol=2e-15)
    np.testing.assert_allclose(
        LM_gronnerud(rho_l, rho_v, mu_l, mu_v, 0.0, 100.0, 0.02), 1.0
    )
    assert float(richardson(0.0, rho_l, rho_v)) == 1.0
    assert np.isinf(richardson(1.0, rho_l, rho_v))
    # Equal fluid properties and x=1/2 imply equal liquid/vapor reference laws.
    phi_l, phi_v, tau_l, tau_v = jax.jit(LM_condensation)(
        1000.0, 2.0, 0.001, 2.0, 0.001, 0.1, 1.0, 0.5, 2.0
    )
    np.testing.assert_allclose([phi_l, phi_v], [22.0, 22.0])
    np.testing.assert_allclose(tau_l, tau_v)
    assert np.isnan(LM_gronnerud(rho_l, rho_v, mu_l, mu_v, 0.3, 0.0, 0.02))
    assert np.all(
        np.isnan(LM_condensation(0.0, rho_l, mu_l, rho_v, mu_v, 0.02, 0.0, 0.5, 0.0))
    )


@pytest.mark.parametrize(
    "law,args",
    [
        (viscosity_Beattie, (1e-3, 1e-5, 1000.0, 2.0)),
        (viscosity_MacAdams, (1e-3, 1e-5)),
        (lambda x: LM_gronnerud(1000.0, 2.0, 1e-3, 1e-5, x, 100.0, 0.02), ()),
        (
            lambda x: LM_condensation(
                1e4, 1000.0, 1e-3, 2.0, 1e-5, 0.02, 2.0, x, 100.0
            )[0],
            (),
        ),
        (lambda x: richardson(x, 1000.0, 2.0), ()),
    ],
)
def test_mixture_correlations_support_jit_vmap_and_ad(law, args):
    function = lambda x: law(*args, x)
    values = jax.jit(jax.vmap(function))(jnp.asarray([0.1, 0.3, 0.7]))
    derivatives = jax.jit(jax.vmap(jax.grad(function)))(jnp.asarray([0.1, 0.3, 0.7]))
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(derivatives))


def test_valid_domain_correlations_preserve_the_retired_library_snapshot():
    # Frozen values independently compared against the previous nozzle library
    # before retirement. No retired or development implementation is imported here.
    actual = [
        Haaland(1e5, 1e-5, 0.02),
        viscosity_Beattie(1e-3, 1e-5, 1000.0, 2.0, 0.3),
        viscosity_MacAdams(1e-3, 1e-5, 0.3),
        LM_gronnerud(1000.0, 2.0, 1e-3, 1e-5, 0.3, 100.0, 0.02),
        richardson(0.3, 1000.0, 2.0),
    ]
    np.testing.assert_allclose(
        actual,
        [
            0.019845502978819606,
            2.6157075430780776e-5,
            3.2573289902280134e-5,
            55.56425146601492,
            12099.75624994891,
        ],
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        LM_condensation(1e4, 1000.0, 1e-3, 2.0, 1e-5, 0.02, 2.0, 0.3, 100.0),
        [
            18214.360270787907,
            1.159646371158932,
            0.013686051749865624,
            1.7152299539734428,
        ],
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        f_HEM_purdue(jnp.asarray([1e3, 1e4, 1e5])), [0.016, 0.0079, 0.0046], rtol=3e-15
    )
