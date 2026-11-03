"""Specified transient initial conditions and independent exact projections."""

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.analytical.transient import cell_reference
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.models.two_fluid import TwoFluidModel

from tests._support.regression import cell_averages


def sod_case(cells=128):
    """Dimensional Sod problem, stopped before its waves reach the boundaries."""
    fluid = jxp.FluidPerfectGas("Air")
    mesh = ConstantAreaGeometry(length=1).build_mesh(cells)
    left_rho, right_rho, left_p, right_p = 1.0, 0.125, 1e5, 1e4
    diaphragm, time = 0.5, 4e-4
    p = np.where(mesh.points < diaphragm, left_p, right_p)
    rho = np.where(mesh.points < diaphragm, left_rho, right_rho)
    properties = fluid.get_state(jxp.DmassP_INPUTS, jnp.asarray(rho), jnp.asarray(p))
    initial = jnp.column_stack((p, np.zeros_like(p), properties["h"]))
    left = fluid.get_state(jxp.DmassP_INPUTS, left_rho, left_p)
    model = NozzleModel(
        mesh,
        fluid,
        p0=left_p,
        T0=float(left["T"]),
        transmissive=True,
        initial_state=initial,
    )
    gamma = float(fluid.constants.gamma)
    cp = gamma * float(fluid.constants.R) / (gamma - 1)
    h_shift = float(left["h"]) - cp * float(left["T"])
    arguments = (time, diaphragm, left_rho, 0.0, left_p, right_rho, 0.0, right_p, gamma)
    star_rho, star_velocity, star_p = [
        float(x[0]) for x in sod_exact(np.array([diaphragm]), *arguments)
    ]
    beta = (gamma - 1) / (gamma + 1)
    density_right_star = (
        right_rho * (star_p / right_p + beta) / (beta * star_p / right_p + 1)
    )
    shock_speed = density_right_star * star_velocity / (density_right_star - right_rho)
    wave_fronts = (
        diaphragm - np.sqrt(gamma * left_p / left_rho) * time,
        diaphragm + (star_velocity - np.sqrt(gamma * star_p / star_rho)) * time,
        diaphragm + star_velocity * time,
        diaphragm + shock_speed * time,
    )
    assert 0 < min(wave_fronts) < max(wave_fronts) < 1

    def exact_density(coordinate):
        density, velocity, pressure = sod_exact(coordinate, *arguments)
        energy = pressure / (gamma - 1) + density * (h_shift + velocity**2 / 2)
        return np.column_stack((density, density * velocity, energy))

    expected = cell_averages(mesh.faces, exact_density, wave_fronts)
    return model, initial, time, expected, wave_fronts


def annular_contact(cells=80):
    """An exact compact contact as the physical initial condition, not a warm start."""
    fluid = jxp.FluidPerfectGas("Air")
    settings = dict(
        center=0.04,
        half_width=0.015,
        v_m=40.0,
        v_theta=3.0,
        density_amplitude=0.25,
        radius=0.2,
        width=0.005,
        pressure=1e5,
    )
    geometry = conical_channel(0.12, settings["radius"], settings["width"], 0.0)
    mesh, native = build_annular_mesh(geometry, cells)
    density_base, time = 1.2, 2e-4
    first = cell_reference(fluid, native, 0.0, density_base, settings)
    last = cell_reference(fluid, native, time, density_base, settings)
    ambient = fluid.get_state(jxp.DmassP_INPUTS, density_base, settings["pressure"])
    total_h = (
        float(ambient["h"]) + (settings["v_m"] ** 2 + settings["v_theta"] ** 2) / 2
    )
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, total_h, ambient["s"])
    trace = np.array(
        [
            settings["pressure"],
            settings["v_m"],
            settings["radius"] * settings["v_theta"],
            float(ambient["h"]),
        ]
    )
    initial = jnp.asarray(np.vstack((trace, first["W"], trace)))
    model = AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=dict(
            p0=float(reservoir["p"]),
            h0=total_h,
            s0=float(ambient["s"]),
            alpha_in=np.arctan2(settings["v_theta"], settings["v_m"]),
        ),
        outlet_pressure=settings["pressure"],
        initial_state=initial,
    )
    return model, initial, time, last["q"]


def perturbed_two_fluid(cells=5):
    """Closure-free small pressure disturbance for pinned numerical compatibility.

    Both phase EOS fixtures are perfect gases. This tests the eight-variable
    coupled implementation, not the physical accuracy of a phase-change model.
    """
    fluid = jxp.FluidPerfectGas("Air")
    mesh = ConstantAreaGeometry(length=0.2).build_mesh(cells)
    primitives = np.tile([2e5, 25.0, 25.0, 0.4, 3e5, 2.8e5], (cells + 2, 1))
    primitives[:, 0] *= 1 + 1e-4 * np.sin(np.pi * mesh.points / 0.2)
    model = TwoFluidModel(
        mesh, fluid, fluid, initial_primitives=primitives, closures=None
    )
    return model, model.initial_state(), 1e-6
