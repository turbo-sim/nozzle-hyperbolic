"""Density-EOS nozzle reference checked against independent perfect-gas laws."""

from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import brentq

from nozzle_hyperbolic.models.euler_nozzle.analytical import SmoothPath
from nozzle_hyperbolic.models.euler_nozzle.analytical.density_eos import (
    normal_shock,
    sample_profile,
    solve_shock_location,
)


class IdealDensityEOS:
    """Algebraic oracle implementing the supplied bounded-EOS interface."""

    gamma, gas_constant = 1.4, 287.0
    cp = gamma * gas_constant / (gamma - 1)
    settings = dict(h_min=1e4, h_max=6e5, p_min=2e3, p_max=1e6)

    def check(self, h, p):
        if not (
            np.all(np.isfinite(h))
            and np.all(np.isfinite(p))
            and np.all(np.asarray(h) >= self.settings["h_min"])
            and np.all(np.asarray(h) <= self.settings["h_max"])
            and np.all(np.asarray(p) >= self.settings["p_min"])
            and np.all(np.asarray(p) <= self.settings["p_max"])
        ):
            raise ValueError("Outside EOS bounds")

    def rho(self, h, p):
        self.check(h, p)
        return np.asarray(p) * self.cp / (self.gas_constant * np.asarray(h))

    def state(self, h, p):
        h, p = np.broadcast_arrays(np.asarray(h), np.asarray(p))
        self.check(h, p)
        sound2 = (self.gamma - 1) * h
        return dict(
            h=h,
            p=p,
            rho=self.rho(h, p),
            T=h / self.cp,
            s=self.cp * np.log(h / (self.cp * 300))
            - self.gas_constant * np.log(p / 2e5),
            a_euler=np.sqrt(sound2),
            a_table=np.sqrt(sound2),
            inverse_a2=1 / sound2,
        )


@dataclass
class TestGeometry:
    """Injected area law; the reference owns no case geometry."""

    __test__ = False
    x_in: float = -0.04
    x_out: float = 0.08
    At: float = 1e-3

    def area(self, x):
        x = np.asarray(x)
        return self.At * np.where(
            x < 0, 1 + 2 * (x / self.x_in) ** 2, 1 + 1.5 * (x / self.x_out) ** 2
        )


def area_ratio(mach, gamma=1.4):
    return ((2 + (gamma - 1) * mach**2) / (gamma + 1)) ** (
        (gamma + 1) / (2 * (gamma - 1))
    ) / mach


@pytest.fixture
def choked_path():
    eos = IdealDensityEOS()
    h0 = eos.cp * 300
    path = SmoothPath(eos, h0, 2e5, h0, eos.settings["p_min"] * 1.001)
    path.choke()
    return eos, path


def test_choking_and_adapted_profile_match_closed_form(choked_path):
    eos, path = choked_path
    geometry = TestGeometry()
    gamma, gas_constant = eos.gamma, eos.gas_constant
    pressure_star = 2e5 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    mass_flux_star = (
        2e5
        * np.sqrt(gamma / (gas_constant * 300))
        * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    assert_allclose(path.p_star, pressure_star, rtol=3e-10)
    assert_allclose(path.G_star, mass_flux_star, rtol=3e-10)
    fields = sample_profile(path, geometry, geometry.At * path.G_star, stations=41)
    mach_exit = brentq(lambda m: area_ratio(m) - 2.5, 1.000001, 10)
    pressure_exit = 2e5 * (1 + (gamma - 1) * mach_exit**2 / 2) ** (-gamma / (gamma - 1))
    assert_allclose(fields["p"][-1], pressure_exit, rtol=3e-9)
    assert_allclose(
        fields["rho"] * fields["u"] * fields["A"],
        geometry.At * mass_flux_star,
        rtol=3e-9,
    )
    assert_allclose(fields["h"] + fields["u"] ** 2 / 2, path.h0, rtol=3e-14)
    with pytest.raises(ValueError, match="outside integrated smooth path"):
        path.enthalpy(1000)


def test_normal_shock_matches_classical_ratios(choked_path):
    eos, path = choked_path
    gamma, mach1 = eos.gamma, 2.0
    factor = 1 + (gamma - 1) * mach1**2 / 2
    pre = eos.state(path.h0 / factor, 2e5 * factor ** (-gamma / (gamma - 1)))
    pre.update(u=mach1 * pre["a_euler"], M_euler=mach1)
    post = normal_shock(eos, pre, path.h0)
    compression = (gamma + 1) * mach1**2 / ((gamma - 1) * mach1**2 + 2)
    pressure_ratio = 1 + 2 * gamma / (gamma + 1) * (mach1**2 - 1)
    mach2 = np.sqrt(
        (1 + (gamma - 1) * mach1**2 / 2) / (gamma * mach1**2 - (gamma - 1) / 2)
    )
    assert_allclose(post["compression"], compression, rtol=2e-12)
    assert_allclose(post["p"] / pre["p"], pressure_ratio, rtol=2e-12)
    assert_allclose(post["M_euler"], mach2, rtol=2e-12)
    assert post["s"] > pre["s"]
    with pytest.raises(ValueError, match="not supersonic"):
        normal_shock(eos, dict(pre, M_euler=0.5), path.h0)


def test_backpressure_finds_shock_and_preserves_both_traces(choked_path):
    eos, path = choked_path
    geometry, gamma = TestGeometry(), eos.gamma
    expected_x = 0.55 * geometry.x_out
    area_shock = geometry.area(expected_x)
    mach1 = brentq(lambda m: area_ratio(m) - area_shock / geometry.At, 1.000001, 10)
    pressure1 = 2e5 * (1 + (gamma - 1) * mach1**2 / 2) ** (-gamma / (gamma - 1))
    pressure2 = pressure1 * (1 + 2 * gamma / (gamma + 1) * (mach1**2 - 1))
    mach2 = np.sqrt(
        (1 + (gamma - 1) * mach1**2 / 2) / (gamma * mach1**2 - (gamma - 1) / 2)
    )
    total_pressure2 = pressure2 * (1 + (gamma - 1) * mach2**2 / 2) ** (
        gamma / (gamma - 1)
    )
    downstream_throat_area = area_shock / area_ratio(mach2)
    exit_mach = brentq(
        lambda m: area_ratio(m)
        - geometry.area(geometry.x_out) / downstream_throat_area,
        1e-5,
        0.999999,
    )
    backpressure = total_pressure2 * (1 + (gamma - 1) * exit_mach**2 / 2) ** (
        -gamma / (gamma - 1)
    )
    mdot = geometry.At * path.G_star
    shock = solve_shock_location(eos, path, geometry, mdot, backpressure)
    assert_allclose(shock["x"], expected_x, atol=2e-10)
    assert_allclose(shock["exit"]["p"], backpressure, rtol=3e-10)
    fields = sample_profile(path, geometry, mdot, shock=shock, stations=41)
    traces = np.flatnonzero(fields["x"] == shock["x"])
    assert list(fields["side"][traces]) == ["shock_pre", "shock_post"]
    assert fields["p"][traces[1]] > fields["p"][traces[0]]
    assert_allclose(fields["rho"] * fields["u"] * fields["A"], mdot, rtol=3e-9)
    assert_allclose(fields["h"] + fields["u"] ** 2 / 2, path.h0, rtol=3e-14)
