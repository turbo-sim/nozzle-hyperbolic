"""Study-local geometry and time-dependent incoming-characteristic boundary."""

import jax
import jax.numpy as jnp
import jaxprop as jp
import numpy as np
import nurbspy.numpy as nrb
from scipy.interpolate import make_interp_spline
from nozzle_hyperbolic.geometry.annular import make_annular_geometry, constant_height
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.state import thermodynamic_state
from nozzle_hyperbolic.solvers.fvm import Mesh


def geometry(c):
    g = c["geometry"]
    R, L, D, w = (
        g[k]
        for k in (
            "tip_radius_m",
            "constant_length_m",
            "expansion_length_m",
            "junction_blend_m",
        )
    )
    rin, rout = R * (1 + g["inlet_hub_tip"]) / 2, R * (1 + g["outlet_hub_tip"]) / 2
    slope = (rout - rin) / (D - w / 2)
    xs = np.array([0, L, L + w, L + D])
    rs = np.array([rin, rin, rin + slope * w / 2, rout])
    ds = np.array([0, 0, slope, slope])
    controls = [np.array([xs[0], rs[0]])]
    for i, dx in enumerate(np.diff(xs)):
        controls.extend(
            [
                np.array([xs[i] + dx / 3, rs[i] + ds[i] * dx / 3]),
                np.array([xs[i + 1] - dx / 3, rs[i + 1] - ds[i + 1] * dx / 3]),
                np.array([xs[i + 1], rs[i + 1]]),
            ]
        )
    knots = (
        np.r_[np.repeat(xs[0], 4), np.repeat(xs[1:-1], 3), np.repeat(xs[-1], 4)]
        / xs[-1]
    )
    mid = nrb.NurbsCurve(np.array(controls).T, np.ones(len(controls)), 3, knots)
    provisional = make_annular_geometry(mid, constant_height(2 * (R - rin)))
    x = np.linspace(L, L + w, 33)
    xi = x / xs[-1]
    m = np.array([provisional.xi_of_m.arclength_at(float(t)) for t in xi])
    pos, deriv = mid.get_value(xi), mid.get_derivative(xi, order=1)
    xp = deriv[0] / np.linalg.norm(deriv, axis=0)
    b = 2 * (R - pos[1]) / xp
    sigma = m / provisional.length
    # Height is continuous, with derivative jumps where curvature changes.
    # Keep these knots instead of smoothing the exact fixed-tip constraint.
    dend = (
        -2 * slope + 2 * (R - pos[1, -1]) * slope * (slope / w) / (1 + slope * slope)
    ) * provisional.length
    h = make_interp_spline(sigma, b, k=3, bc_type=([(1, 0)], [(1, dend)]))
    first = np.full(4, b[0])
    last = np.linspace(b[-1], 2 * (R - rout) * np.sqrt(1 + slope * slope), 4)
    hc = np.r_[first, h.c[1:], last[1:]]
    ht = np.r_[
        np.zeros(4),
        np.repeat(sigma[0], 3),
        h.t[4:-4],
        np.repeat(sigma[-1], 3),
        np.ones(4),
    ]
    height = nrb.NurbsCurve(hc[None, :], np.ones(len(hc)), 3, ht)
    return make_annular_geometry(mid, height)


class DrivenAnnular(AnnularModel):
    """Time-dependent exterior Riemann state coupled by production HLL.

    The left endpoint holds exterior p0,T0,M,alpha targets; HLL supplies the
    boundary flux between exterior and reconstructed duct-side states.
    """

    def __init__(self, *args, config, driven=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.config, self.driven = config, driven

    def target(self, time):
        c, f = self.config["inlet"], self.config["forcing"]
        wave = jnp.sin(2 * jnp.pi * f["frequency_Hz"] * time) if self.driven else 0.0
        M = c["mach"] + f["mach_amplitude"] * wave
        alpha = jnp.deg2rad(c["angle_deg"] + f["angle_amplitude_deg"] * wave)
        gamma, R = self.fluid.constants.gamma, self.fluid.constants.R
        T = c["stagnation_temperature_K"] / (1 + (gamma - 1) * M * M / 2)
        p = c["stagnation_pressure_Pa"] * (T / c["stagnation_temperature_K"]) ** (
            gamma / (gamma - 1)
        )
        state = self.fluid.get_state(jp.PT_INPUTS, p, T)
        speed = M * state["a"]
        return jnp.array(
            [
                p,
                speed * jnp.cos(alpha),
                self.native_mesh.r_face[0] * speed * jnp.sin(alpha),
                state["h"],
            ]
        )

    def boundary_residual(self, full, time=0.0):
        # The left endpoint is the exterior Riemann state. Production HLL
        # couples it to the MUSCL-reconstructed duct state, including the
        # outgoing acoustic wave during subsonic operation. It is not a
        # prescribed physical interface trace in that regime.
        return jnp.stack(
            [
                (full[0] - self.target(time)) / self.variable_scales,
                self._boundary(full)[1],
            ]
        )

    def transient_solution_audit(self, full, sonic_tolerance=2e-3):
        mach = np.asarray(self.regime_metrics(full)["boundary_mach"])
        return bool(
            self.boundary_admissible(full)
            and np.min(np.asarray(full)[:, 1]) > 0
            and mach[1] >= 0
        )


def model(c, geom, n, driven=False, initial=None):
    native = geom.mesh(n)
    mesh = Mesh(
        native.m_faces,
        native.m_centers,
        native.volumes,
        native.A_face,
        native.A_center,
        native.dA_dm,
        np.diff(native.A_face),
        geom,
    )
    fluid = jp.FluidPerfectGas(c["fluid"])
    i = c["inlet"]
    gamma, R = float(fluid.constants.gamma), float(fluid.constants.R)
    T = i["stagnation_temperature_K"] / (1 + (gamma - 1) * i["mach"] ** 2 / 2)
    p = i["stagnation_pressure_Pa"] * (T / i["stagnation_temperature_K"]) ** (
        gamma / (gamma - 1)
    )
    state = fluid.get_state(jp.PT_INPUTS, p, T)
    speed = i["mach"] * float(state["a"])
    alpha = np.deg2rad(i["angle_deg"])
    inlet = np.array(
        [
            p,
            speed * np.cos(alpha),
            native.r_face[0] * speed * np.sin(alpha),
            float(state["h"]),
        ]
    )
    return DrivenAnnular(
        mesh,
        native,
        fluid,
        inlet_data={"state": inlet, "p0": i["stagnation_pressure_Pa"]},
        inlet_regime="supersonic",
        outlet_regime="auto",
        outlet_pressure=c["outlet"]["pressure_Pa"],
        initial_state=initial,
        config=c,
        driven=driven,
    )
