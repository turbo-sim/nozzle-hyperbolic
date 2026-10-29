"""Independent invariant, normal-shock and volume-projection reference checks."""

import jaxprop as jxp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import brentq

from nozzle_hyperbolic.geometry.annular import conical_channel, radial_channel
from nozzle_hyperbolic.models.euler_annular.analytical import (
    PerfectGasReference, ReferenceBranchError,
)


def make_reference(geometry, M_m=.1, M_theta=0.):
    fluid = jxp.FluidPerfectGas(name="Air")
    reference = PerfectGasReference.from_inlet(
        fluid, geometry, p0=2e5, T0=300., M_m=M_m, M_theta=M_theta)
    return fluid, reference


def test_radial_recovery_against_classical_area_mach_relation():
    geometry = radial_channel(.1, .2, .005)
    fluid, reference = make_reference(geometry)
    m = np.linspace(0., geometry.length, 17)
    fields = reference.profile(m)
    gamma = float(fluid.constants.gamma)
    def area_mach(M):
        return (2/(gamma+1)*(1+(gamma-1)*M*M/2))**((gamma+1)/(2*(gamma-1)))/M
    mach = np.array([brentq(lambda M: area_mach(M) - area_mach(.1)*r/.1,
                            1e-10, 1., xtol=5e-15) for r in fields["r"]])
    temperature = 300./(1+(gamma-1)*mach**2/2)
    pressure = 2e5*(temperature/300.)**(gamma/(gamma-1))
    assert_allclose(fields["M_m"], mach, atol=2e-12)
    assert_allclose(fields["p"], pressure, rtol=2e-12)
    assert_allclose(fields["mdot"], reference.mdot, rtol=2e-14)
    assert_allclose(fields["p0"], 2e5, rtol=2e-12)
    assert_allclose(fields["h"]+.5*(fields["v_m"]**2+fields["v_theta"]**2), reference.h0, atol=2e-8)
    assert np.ptp(fields["s"]) < 2e-10
    backend = fluid.get_state(jxp.PT_INPUTS, fields["p"], fields["T"])
    assert_allclose(fields["h"], backend["h"], rtol=2e-12, atol=2e-8)


def test_swirl_diffusion_conserves_invariants_and_crosses_only_total_sonic():
    geometry = radial_channel(.1, .2, .005)
    _, reference = make_reference(geometry, M_m=.3, M_theta=1.5)
    fields = reference.profile(np.linspace(0., geometry.length, 31))
    assert fields["M"][0] > 1 and fields["M"][-1] < 1
    assert np.max(fields["M_m"]) < 1
    assert_allclose(fields["r"]*fields["v_theta"], reference.ell, rtol=2e-14)
    assert_allclose(fields["p"]/fields["rho"]**reference.gamma, reference.K, rtol=2e-14)
    assert_allclose(fields["M_m"][-1], .06035228656542846, rtol=2e-10)
    assert_allclose(fields["p"][-1], 150914.69877478175, rtol=2e-10)


def test_normal_shock_jump_mass_momentum_energy_and_swirl():
    geometry = radial_channel(.1, .2, .005)
    _, reference = make_reference(geometry, M_m=1.5, M_theta=.4)
    shock = reference.normal_shock(.04)
    before, after = shock["upstream"], shock["downstream"]
    assert before["M_m"] > 1 > after["M_m"]
    assert shock["entropy_rise"] > 0
    assert_allclose(before["rho"]*before["v_m"], after["rho"]*after["v_m"], rtol=2e-12)
    assert_allclose(before["p"]+before["rho"]*before["v_m"]**2,
                    after["p"]+after["rho"]*after["v_m"]**2, rtol=2e-12)
    assert_allclose(before["h"]+.5*(before["v_m"]**2+before["v_theta"]**2),
                    after["h"]+.5*(after["v_m"]**2+after["v_theta"]**2), rtol=2e-12)
    assert_allclose(before["v_theta"], after["v_theta"], rtol=2e-14)
    profile = reference.profile(np.array([0., .04, geometry.length]), shock_m=.04)
    assert profile["M_m"][1] > 1 > profile["M_m"][-1]
    assert profile["p0"][-1] < profile["p0"][0]


def test_cell_average_projection_has_physical_volume_measure():
    # Uniform axial annulus has an exactly constant solution and projection.
    geometry = conical_channel(.1, .2, .005, phi=0.)
    _, reference = make_reference(geometry, M_m=.3, M_theta=.5)
    mesh = geometry.mesh(5)
    projected = reference.cell_average(mesh, return_conserved=True)
    point = reference.profile(mesh.m_centers)
    assert_allclose(projected["W"], point["W"], rtol=2e-12, atol=2e-8)
    assert_allclose(projected["integrated"], point["q"]*mesh.volumes[:,None], rtol=2e-12, atol=2e-9)


@pytest.mark.parametrize("sonic_end", ["inlet", "outlet"])
def test_sonic_endpoint_projection_against_independent_mach_quadrature(sonic_end):
    # The primitive branch has a square-root endpoint cusp in physical m.
    # Independently integrate in Mach, where the area-Mach Jacobian vanishes
    # regularly at M=1, rather than reproducing the production m->t^2 rule.
    fluid = jxp.FluidPerfectGas(name="Air")
    gamma, gas_constant = float(fluid.constants.gamma), float(fluid.constants.R)
    beta = (gamma-1)/2
    def area_mach(M):
        return (2/(gamma+1)*(1+beta*M*M))**((gamma+1)/(2*(gamma-1)))/M
    if sonic_end == "outlet":
        geometry = radial_channel(.2, .1, .005)
        mach_in = brentq(lambda M: area_mach(M)-2., 1e-8, 1., xtol=5e-15)
        branch, cell = "sub", -1
    else:
        geometry = radial_channel(.1, .2, .005)
        mach_in, branch, cell = 1., "sup", 0
    reference = PerfectGasReference.from_inlet(fluid, geometry, p0=2e5, T0=300., M_m=mach_in)
    mesh = geometry.mesh(50)
    projected = reference.cell_average(mesh, branch=branch, return_conserved=True)
    assert np.isfinite(projected["W"]).all()
    assert_allclose(reference.profile(geometry.length if sonic_end == "outlet" else 0.)["M_m"],
                    1., atol=2e-7)
    radius_other = float(mesh.r_face[-2] if sonic_end == "outlet" else mesh.r_face[1])
    bounds = (mach_in, 1.) if sonic_end == "outlet" else (1., 10.)
    other_mach = brentq(lambda M: area_mach(M)-radius_other/.1, *bounds, xtol=5e-15)
    lower, upper = (other_mach, 1.) if sonic_end == "outlet" else (1., other_mach)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    M = lower + (upper-lower)*(nodes+1)/2
    T = 300./(1+beta*M*M)
    p = 2e5*(T/300.)**(gamma/(gamma-1))
    rho = p/(gas_constant*T)
    v_m = M*np.sqrt(gamma*gas_constant*T)
    energy = (reference.cp-gas_constant)*T+reference.h_shift+.5*v_m*v_m
    q = np.column_stack((rho, rho*v_m, np.zeros_like(M), rho*energy))
    area_ratio = area_mach(M)
    area = 2*np.pi*.005*.1*area_ratio
    derivative = area_ratio*(M*M-1)/(M*(1+beta*M*M))
    dm_dmach = .1*np.abs(derivative)
    expected = (upper-lower)/2 * (weights @ (q*(area*dm_dmach)[:,None]))
    assert_allclose(projected["integrated"][cell], expected, rtol=2e-10, atol=2e-12)


def test_angle_inlet_is_equivalent_to_mach_components():
    geometry = radial_channel(.1, .2, .005)
    fluid, components = make_reference(geometry, M_m=.3, M_theta=.6)
    angle = PerfectGasReference.from_inlet(fluid, geometry, p0=2e5, T0=300.,
                                          M_m=.3, alpha=np.arctan(2.))
    assert_allclose((angle.ell, angle.mdot, angle.h0, angle.K),
                    (components.ell, components.mdot, components.h0, components.K), rtol=2e-14)


def test_impossible_branch_and_noncompressive_shock_are_rejected():
    geometry = radial_channel(.2, .1, .005)
    _, reference = make_reference(geometry, M_m=.8)
    with pytest.raises(ReferenceBranchError, match="sonic capacity"):
        reference.profile(geometry.length)
    geometry = radial_channel(.1, .2, .005)
    _, reference = make_reference(geometry, M_m=.2)
    with pytest.raises(ReferenceBranchError, match="meridional Mach"):
        reference.normal_shock(.02, upstream_branch="sub")
