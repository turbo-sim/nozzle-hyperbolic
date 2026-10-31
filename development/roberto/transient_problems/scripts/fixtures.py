"""Physical data and references for production-model clients."""

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.nozzle import (
    ConstantAreaGeometry,
    SymmetricNozzleGeometry,
)
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact


METHODS = (
    dict(
        id="euler_explicit",
        label="Explicit Euler · segregated BC",
        integrator="euler-explicit",
        boundary_treatment="segregated",
    ),
    dict(
        id="rk3",
        label="SSPRK3 · segregated BC",
        integrator="runge-kutta-3",
        boundary_treatment="segregated",
    ),
    dict(
        id="implicit_coupled",
        label="Implicit Euler · coupled BC",
        integrator="euler-implicit",
        boundary_treatment="coupled",
    ),
    dict(
        id="implicit_segregated",
        label="Implicit Euler · segregated BC",
        integrator="euler-implicit",
        boundary_treatment="segregated",
    ),
)


def fixture(case_id, settings):
    fluid = jxp.FluidPerfectGas("Air")
    config = settings[case_id]
    count = settings["cells"]
    if case_id == "sod":
        geometry = ConstantAreaGeometry(length=config["length"], area=1.0)
        mesh = geometry.build_mesh(count)
        left, right = config["left"], config["right"]
        mask = mesh.points < config["diaphragm"]
        pressure = np.where(mask, left["p"], right["p"])
        density = np.where(mask, left["rho"], right["rho"])
        velocity = np.where(mask, left["u"], right["u"])
        state = fluid.get_state(
            jxp.DmassP_INPUTS, jnp.asarray(density), jnp.asarray(pressure)
        )
        initial = jnp.stack(
            (jnp.asarray(pressure), jnp.asarray(velocity), state["h"]), axis=-1
        )
        left_state = fluid.get_state(jxp.DmassP_INPUTS, left["rho"], left["p"])
        model = NozzleModel(
            mesh,
            fluid,
            p0=left["p"],
            T0=float(left_state["T"]),
            transmissive=True,
            initial_state=initial,
        )
        label = "Sod shock tube"
        description = (
            "A diaphragm at x=0.5 m separates two stationary states. "
            "Both ends use transmissive traces. The comparison ends before waves reach the ends."
        )
        time_scale = config["length"] / float(left_state["a"])
    else:
        geometry = SymmetricNozzleGeometry(
            config["length"], config["inlet_area"], config["throat_area"]
        )
        mesh = geometry.build_mesh(count)
        p0, T0 = config["reservoir_pressure"], config["reservoir_temperature"]
        reservoir = fluid.get_state(jxp.PT_INPUTS, p0, T0)
        initial = jnp.broadcast_to(jnp.array([p0, 0.0, reservoir["h"]]), (count + 2, 3))
        model = NozzleModel(
            mesh,
            fluid,
            p0=p0,
            T0=T0,
            back_pressure=p0 * config["back_pressure_ratio"],
            initial_state=initial,
            outlet_regime=config["outlet_regime"],
            outlet_formulation=config["outlet_formulation"],
        )
        label = "Nozzle pressure release from rest"
        description = (
            "The nozzle is initially stagnant at reservoir pressure and temperature. "
            "At t=0 the external exit pressure drops to 1% of reservoir pressure; "
            "the inlet remains connected to the same reservoir. The finite-wave outlet "
            "samples incoming, sonic and fully outgoing waves without forcing external "
            "pressure onto a supersonic exit."
        )
        time_scale = config["length"] / float(reservoir["a"])
    times = np.linspace(
        0.0, config["end_acoustic_times"] * time_scale, settings["frames"]
    )
    return model, dict(
        id=case_id,
        label=label,
        description=description,
        x=mesh.centers,
        faces=mesh.faces,
        area_centers=mesh.area_centers,
        reference_time=time_scale,
        times=times,
        physical_inputs=config,
        field_scales=dict(p=model.p0, rho=model.rho0, u=model.a0, M=1.0, T=model.T0),
    )


def reference_fields(model, case_id, settings, time):
    if case_id != "sod":
        return None
    config = settings["sod"]
    left, right = config["left"], config["right"]
    x = model.mesh.centers
    if time == 0.0:
        mask = x < config["diaphragm"]
        rho, u, p = (np.where(mask, left[key], right[key]) for key in ("rho", "u", "p"))
    else:
        rho, u, p = sod_exact(
            x,
            time,
            config["diaphragm"],
            left["rho"],
            left["u"],
            left["p"],
            right["rho"],
            right["u"],
            right["p"],
            float(model.fluid.constants.gamma),
        )
    state = model.fluid.get_state(jxp.DmassP_INPUTS, jnp.asarray(rho), jnp.asarray(p))
    return dict(
        rho=rho, u=u, p=p, M=np.asarray(u / state["a"]), T=np.asarray(state["T"])
    )


def fields(model, full):
    values = model.fields(jnp.asarray(full))
    return {name: np.asarray(values[name]) for name in ("p", "rho", "u", "M", "T")}


def trace_diagnostics(model, full):
    state = jnp.asarray(full)[-1]
    thermo = model.fluid.get_state(jxp.HmassP_INPUTS, state[-1], state[0])
    mach = float(state[1] / thermo["a"])
    if model.transmissive:
        mode = "transmissive"
    elif model.outlet_formulation == "finite_wave":
        wave_mode = int(model.outlet.wave(jnp.asarray(full)[1:-1]).mode)
        mode = {
            1: "fully outgoing",
            3: "incoming compression",
            4: "imposed-pressure expansion",
            5: "sonic expansion fan",
        }[wave_mode]
    else:
        mode = (
            "fully outgoing"
            if mach > 1.001
            else "sonic" if abs(mach - 1.0) < 1.0e-3 else "pressure controlled"
        )
    return dict(mode=mode, pressure=float(state[0]), mach=mach)
