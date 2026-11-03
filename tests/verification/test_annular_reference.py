"""Independent invariant, normal-shock and volume-projection reference checks."""

import jaxprop as jxp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import brentq

from nozzle_hyperbolic.geometry.annular import conical_channel, radial_channel
from nozzle_hyperbolic.models.euler_annular.analytical import (
    PerfectGasReference,
    ReferenceBranchError,
    cell_reference,
    density_pulse,
    exact_homogeneous,
)


def make_reference(geometry, M_m=0.1, M_theta=0.0):
    fluid = jxp.FluidPerfectGas(name="Air")
    reference = PerfectGasReference.from_inlet(
        fluid, geometry, p0=2e5, T0=300.0, M_m=M_m, M_theta=M_theta
    )
    return fluid, reference


def test_radial_recovery_against_classical_area_mach_relation():
    geometry = radial_channel(0.1, 0.2, 0.005)
    fluid, reference = make_reference(geometry)
    m = np.linspace(0.0, geometry.length, 17)
    fields = reference.profile(m)
    gamma = float(fluid.constants.gamma)

    def area_mach(M):
        return (2 / (gamma + 1) * (1 + (gamma - 1) * M * M / 2)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        ) / M

    mach = np.array(
        [
            brentq(
                lambda M: area_mach(M) - area_mach(0.1) * r / 0.1,
                1e-10,
                1.0,
                xtol=5e-15,
            )
            for r in fields["r"]
        ]
    )
    temperature = 300.0 / (1 + (gamma - 1) * mach**2 / 2)
    pressure = 2e5 * (temperature / 300.0) ** (gamma / (gamma - 1))
    assert_allclose(fields["M_m"], mach, atol=2e-12)
    assert_allclose(fields["p"], pressure, rtol=2e-12)
    assert_allclose(fields["mdot"], reference.mdot, rtol=2e-14)
    assert_allclose(fields["p0"], 2e5, rtol=2e-12)
    assert_allclose(
        fields["h"] + 0.5 * (fields["v_m"] ** 2 + fields["v_theta"] ** 2),
        reference.h0,
        atol=2e-8,
    )
    assert np.ptp(fields["s"]) < 2e-10
    backend = fluid.get_state(jxp.PT_INPUTS, fields["p"], fields["T"])
    assert_allclose(fields["h"], backend["h"], rtol=2e-12, atol=2e-8)


def test_swirl_diffusion_conserves_invariants_and_crosses_only_total_sonic():
    geometry = radial_channel(0.1, 0.2, 0.005)
    _, reference = make_reference(geometry, M_m=0.3, M_theta=1.5)
    fields = reference.profile(np.linspace(0.0, geometry.length, 31))
    assert fields["M"][0] > 1 and fields["M"][-1] < 1
    assert np.max(fields["M_m"]) < 1
    assert_allclose(fields["r"] * fields["v_theta"], reference.ell, rtol=2e-14)
    assert_allclose(
        fields["p"] / fields["rho"] ** reference.gamma, reference.K, rtol=2e-14
    )
    assert_allclose(fields["M_m"][-1], 0.06035228656542846, rtol=2e-10)
    assert_allclose(fields["p"][-1], 150914.69877478175, rtol=2e-10)


def test_normal_shock_jump_mass_momentum_energy_and_swirl():
    geometry = radial_channel(0.1, 0.2, 0.005)
    _, reference = make_reference(geometry, M_m=1.5, M_theta=0.4)
    shock = reference.normal_shock(0.04)
    before, after = shock["upstream"], shock["downstream"]
    assert before["M_m"] > 1 > after["M_m"]
    assert shock["entropy_rise"] > 0
    assert_allclose(
        before["rho"] * before["v_m"], after["rho"] * after["v_m"], rtol=2e-12
    )
    assert_allclose(
        before["p"] + before["rho"] * before["v_m"] ** 2,
        after["p"] + after["rho"] * after["v_m"] ** 2,
        rtol=2e-12,
    )
    assert_allclose(
        before["h"] + 0.5 * (before["v_m"] ** 2 + before["v_theta"] ** 2),
        after["h"] + 0.5 * (after["v_m"] ** 2 + after["v_theta"] ** 2),
        rtol=2e-12,
    )
    assert_allclose(before["v_theta"], after["v_theta"], rtol=2e-14)
    profile = reference.profile(np.array([0.0, 0.04, geometry.length]), shock_m=0.04)
    assert profile["M_m"][1] > 1 > profile["M_m"][-1]
    assert profile["p0"][-1] < profile["p0"][0]


def test_cell_average_projection_has_physical_volume_measure():
    # Uniform axial annulus has an exactly constant solution and projection.
    geometry = conical_channel(0.1, 0.2, 0.005, phi=0.0)
    _, reference = make_reference(geometry, M_m=0.3, M_theta=0.5)
    mesh = geometry.mesh(5)
    projected = reference.cell_average(mesh, return_conserved=True)
    point = reference.profile(mesh.m_centers)
    assert_allclose(projected["W"], point["W"], rtol=2e-12, atol=2e-8)
    assert_allclose(
        projected["integrated"],
        point["q"] * mesh.volumes[:, None],
        rtol=2e-12,
        atol=2e-9,
    )


@pytest.mark.parametrize("sonic_end", ["inlet", "outlet"])
def test_sonic_endpoint_projection_against_independent_mach_quadrature(sonic_end):
    # The primitive branch has a square-root endpoint cusp in physical m.
    # Independently integrate in Mach, where the area-Mach Jacobian vanishes
    # regularly at M=1, rather than reproducing the production m->t^2 rule.
    fluid = jxp.FluidPerfectGas(name="Air")
    gamma, gas_constant = float(fluid.constants.gamma), float(fluid.constants.R)
    beta = (gamma - 1) / 2

    def area_mach(M):
        return (2 / (gamma + 1) * (1 + beta * M * M)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        ) / M

    if sonic_end == "outlet":
        geometry = radial_channel(0.2, 0.1, 0.005)
        mach_in = brentq(lambda M: area_mach(M) - 2.0, 1e-8, 1.0, xtol=5e-15)
        branch, cell = "sub", -1
    else:
        geometry = radial_channel(0.1, 0.2, 0.005)
        mach_in, branch, cell = 1.0, "sup", 0
    reference = PerfectGasReference.from_inlet(
        fluid, geometry, p0=2e5, T0=300.0, M_m=mach_in
    )
    mesh = geometry.mesh(50)
    projected = reference.cell_average(mesh, branch=branch, return_conserved=True)
    assert np.isfinite(projected["W"]).all()
    assert_allclose(
        reference.profile(geometry.length if sonic_end == "outlet" else 0.0)["M_m"],
        1.0,
        atol=2e-7,
    )
    radius_other = float(mesh.r_face[-2] if sonic_end == "outlet" else mesh.r_face[1])
    bounds = (mach_in, 1.0) if sonic_end == "outlet" else (1.0, 10.0)
    other_mach = brentq(
        lambda M: area_mach(M) - radius_other / 0.1, *bounds, xtol=5e-15
    )
    lower, upper = (other_mach, 1.0) if sonic_end == "outlet" else (1.0, other_mach)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    M = lower + (upper - lower) * (nodes + 1) / 2
    T = 300.0 / (1 + beta * M * M)
    p = 2e5 * (T / 300.0) ** (gamma / (gamma - 1))
    rho = p / (gas_constant * T)
    v_m = M * np.sqrt(gamma * gas_constant * T)
    energy = (reference.cp - gas_constant) * T + reference.h_shift + 0.5 * v_m * v_m
    q = np.column_stack((rho, rho * v_m, np.zeros_like(M), rho * energy))
    area_ratio = area_mach(M)
    area = 2 * np.pi * 0.005 * 0.1 * area_ratio
    derivative = area_ratio * (M * M - 1) / (M * (1 + beta * M * M))
    dm_dmach = 0.1 * np.abs(derivative)
    expected = (upper - lower) / 2 * (weights @ (q * (area * dm_dmach)[:, None]))
    assert_allclose(projected["integrated"][cell], expected, rtol=2e-10, atol=2e-12)


def test_angle_inlet_is_equivalent_to_mach_components():
    geometry = radial_channel(0.1, 0.2, 0.005)
    fluid, components = make_reference(geometry, M_m=0.3, M_theta=0.6)
    angle = PerfectGasReference.from_inlet(
        fluid, geometry, p0=2e5, T0=300.0, M_m=0.3, alpha=np.arctan(2.0)
    )
    assert_allclose(
        (angle.ell, angle.mdot, angle.h0, angle.K),
        (components.ell, components.mdot, components.h0, components.K),
        rtol=2e-14,
    )


def test_impossible_branch_and_noncompressive_shock_are_rejected():
    geometry = radial_channel(0.2, 0.1, 0.005)
    _, reference = make_reference(geometry, M_m=0.8)
    with pytest.raises(ReferenceBranchError, match="sonic capacity"):
        reference.profile(geometry.length)
    geometry = radial_channel(0.1, 0.2, 0.005)
    _, reference = make_reference(geometry, M_m=0.2)
    with pytest.raises(ReferenceBranchError, match="meridional Mach"):
        reference.normal_shock(0.02, upstream_branch="sub")


class _ShiftedPerfectGas:
    """Independent constant-cp backend with a deliberately nonzero datum."""

    class constants:
        gamma = 1.4
        R = 287.0

    def get_state(self, pair, first, second):
        gas_constant, gamma = self.constants.R, self.constants.gamma
        cp = gamma * gas_constant / (gamma - 1)
        if pair == jxp.PT_INPUTS:
            pressure, temperature = np.asarray(first), np.asarray(second)
            density = pressure / (gas_constant * temperature)
        elif pair == jxp.DmassP_INPUTS:
            density, pressure = np.asarray(first), np.asarray(second)
            temperature = pressure / (density * gas_constant)
        else:
            raise ValueError("Unsupported test input pair")
        return dict(
            rho=density,
            p=pressure,
            T=temperature,
            h=cp * temperature + 43000.0,
            s=cp * np.log(temperature) - gas_constant * np.log(pressure) - 200.0,
        )


def test_shifted_backend_keeps_steady_enthalpy_and_entropy_datums():
    fluid = _ShiftedPerfectGas()
    geometry = radial_channel(0.1, 0.2, 0.005)
    reference = PerfectGasReference.from_inlet(
        fluid, geometry, p0=2e5, T0=300.0, M_m=0.3, M_theta=0.6
    )
    fields = reference.profile(np.linspace(0.0, geometry.length, 11))
    backend = fluid.get_state(jxp.PT_INPUTS, fields["p"], fields["T"])
    assert_allclose(reference.h_shift, 43000.0, atol=1e-10)
    assert_allclose(fields["h"], backend["h"], rtol=2e-14)
    assert_allclose(fields["s"], backend["s"], atol=1e-12)


def test_projected_state_diagnostics_agree_with_pointwise_fields_for_point_states():
    geometry = radial_channel(0.1, 0.2, 0.005)
    _, reference = make_reference(geometry, M_m=0.3, M_theta=0.6)
    point = reference.profile(np.linspace(0.0, geometry.length, 9))
    recovered = reference.fields_from_state(point["W"], point["r"])
    for key in ("p", "v_m", "v_theta", "rho", "T", "M_m", "M_theta", "M", "h0", "ell"):
        assert_allclose(recovered[key], point[key], rtol=2e-14)
    scalar = reference.fields_from_state(point["W"][0], point["r"][0])
    assert np.ndim(scalar["p"]) == 0
    with pytest.raises(ValueError, match="radius"):
        reference.fields_from_state(point["W"], 0.0)


def test_backend_with_gamma_attribute_but_nonconstant_cp_is_rejected():
    class NonconstantCp(_ShiftedPerfectGas):
        def get_state(self, pair, first, second):
            state = super().get_state(pair, first, second)
            state["h"] = state["h"] + 0.02 * state["T"] ** 2
            return state

    with pytest.raises(ValueError, match="calorically perfect"):
        PerfectGasReference.from_inlet(
            NonconstantCp(),
            radial_channel(0.1, 0.2, 0.005),
            p0=2e5,
            T0=300.0,
            M_m=0.3,
        )


def test_homogeneous_sources_retain_datum_and_satisfy_integrated_odes():
    fluid = _ShiftedPerfectGas()
    initial = dict(
        density=1.2,
        temperature=300.0,
        v_m=30.0,
        v_theta=-40.0,
        height=0.005,
        radius=0.1,
        tau_w=2.0,
        q_w=1000.0,
    )
    q0, W0, _ = exact_homogeneous(0.0, fluid, **initial)
    time = 0.01
    q, W, temperature = exact_homogeneous(time, fluid, **initial)
    speed0 = np.hypot(initial["v_m"], initial["v_theta"])
    direction = np.array([initial["v_m"], initial["v_theta"]]) / speed0
    momentum_source = -2 * initial["tau_w"] / initial["height"] * direction
    source = np.array(
        [
            0.0,
            momentum_source[0],
            initial["radius"] * momentum_source[1],
            2 * initial["q_w"] / initial["height"],
        ]
    )
    assert_allclose(q - q0, time * source, atol=2e-10)
    cp = fluid.constants.gamma * fluid.constants.R / (fluid.constants.gamma - 1)
    assert_allclose(W0[-1], cp * initial["temperature"] + 43000.0, atol=1e-9)
    assert_allclose(W[-1], cp * temperature + 43000.0, atol=1e-9)
    assert temperature > initial["temperature"]


def test_homogeneous_shear_reference_rejects_after_stopping():
    initial = dict(
        density=1.0,
        temperature=300.0,
        v_m=1.0,
        v_theta=0.0,
        height=0.01,
        radius=0.1,
        tau_w=1.0,
        q_w=0.0,
    )
    with pytest.raises(ValueError, match="before stopping"):
        exact_homogeneous(0.01, _ShiftedPerfectGas(), **initial)


def _contact_settings():
    return dict(
        center=0.04,
        half_width=0.015,
        v_m=2.0,
        v_theta=3.0,
        density_amplitude=0.25,
        radius=0.2,
        width=0.005,
        pressure=1e5,
    )


def test_compact_contact_translates_without_changing_shape_or_baseline():
    settings = _contact_settings()
    positions = np.linspace(-0.1, 0.2, 71)
    time = 0.003
    assert_allclose(
        density_pulse(positions + settings["v_m"] * time, time, 1.2, settings),
        density_pulse(positions, 0.0, 1.2, settings),
        atol=2e-15,
    )
    assert density_pulse(np.array(settings["center"]), 0.0, 1.2, settings) == 1.5
    assert density_pulse(np.array(-1.0), 0.0, 1.2, settings) == 1.2


def test_contact_projection_keeps_energy_and_affine_constant_pressure_datum():
    settings = _contact_settings()
    geometry = conical_channel(0.1, settings["radius"], settings["width"], phi=0.0)
    mesh = geometry.mesh(13)
    fluid = _ShiftedPerfectGas()
    result = cell_reference(fluid, mesh, 0.001, 1.2, settings)
    W, q = result["W"], result["q"]
    assert_allclose(W[:, 0], settings["pressure"], rtol=2e-14)
    assert_allclose(q[:, 1], q[:, 0] * settings["v_m"], rtol=2e-14)
    kinetic = 0.5 * (settings["v_m"] ** 2 + settings["v_theta"] ** 2)
    assert_allclose(q[:, 3], q[:, 0] * (W[:, 3] + kinetic) - W[:, 0], rtol=2e-14)
    assert_allclose(result["integrated"], q * mesh.volumes[:, None], rtol=2e-14)


def test_contact_projection_rejects_wrong_geometry_and_nonlinear_eos_energy():
    settings = _contact_settings()
    fluid = _ShiftedPerfectGas()
    with pytest.raises(ValueError, match="constant-area"):
        cell_reference(
            fluid, radial_channel(0.2, 0.3, 0.005).mesh(5), 0.0, 1.2, settings
        )

    class NonAffineEnergy(_ShiftedPerfectGas):
        def get_state(self, pair, first, second):
            state = super().get_state(pair, first, second)
            state["h"] = state["h"] + 1000.0 * state["rho"] ** 2
            return state

    mesh = conical_channel(0.1, 0.2, 0.005, phi=0.0).mesh(5)
    with pytest.raises(ValueError, match="incompatible with this EOS"):
        cell_reference(NonAffineEnergy(), mesh, 0.0, 1.2, settings)
