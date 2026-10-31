"""Simultaneous annular boundary constraints with direct HP thermodynamics.

Boundary states are independent face traces in z=(W_in,W_1,...,W_N,W_out).
This module never solves for, clips or overwrites a boundary trial state.
"""

import numpy as np
import jax.numpy as jnp

from .state import state_property, thermodynamic_state
from ..euler_outlet import EulerOutlet


def make_boundary_residual(fluid, m_centers, m_faces, r_faces, inlet_data,
                           outlet_pressure, scales, inlet_regime="subsonic",
                           outlet_regime="subsonic", *, outlet_formulation="acoustic"):
    """Create scaled four-row inlet/outlet constraints, returning shape (2,4).

    ``inlet_data`` contains h0,s0,alpha_in (radians) for a subsonic inlet,
    or a primitive ``state`` for a supersonic inlet. A separately selected
    ``ell_in`` may replace alpha_in, but both must never be imposed together.
    Positive fixed scale keys are p,a,ell,h,s. Regimes are factory-static;
    changing a regime requires restarting the overall solve with a new block.
    """
    centers, faces, radii = map(np.asarray, (m_centers, m_faces, r_faces))
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("The two-cell boundary closure requires at least two physical cells")
    if faces.shape != (len(centers)+1,) or radii.shape != faces.shape:
        raise ValueError("Boundary geometry needs N+1 face coordinates and radii")
    if (not np.all(np.isfinite(centers)) or not np.all(np.isfinite(faces))
            or not np.all(np.diff(centers) > 0) or not np.all(np.diff(faces) > 0)
            or not np.all(np.isfinite(radii)) or not np.all(radii > 0)):
        raise ValueError("Boundary geometry must have finite increasing coordinates and positive radii")
    if not np.all((centers > faces[:-1]) & (centers < faces[1:])):
        raise ValueError("Physical cell representatives must lie inside their faces")
    if inlet_regime not in ("subsonic", "supersonic") or outlet_regime not in ("subsonic", "supersonic"):
        raise ValueError("Boundary regimes must be 'subsonic' or 'supersonic'")
    fixed_scales = {name: float(scales[name]) for name in ("p", "a", "ell", "h", "s")}
    if not all(np.isfinite(value) and value > 0 for value in fixed_scales.values()):
        raise ValueError("Boundary residual scales must be finite and strictly positive")
    primitive_scale = jnp.asarray([fixed_scales[k] for k in ("p", "a", "ell", "h")])
    inlet_scale = jnp.asarray([fixed_scales[k] for k in ("h", "s", "a", "p")])
    outlet_scale = jnp.asarray([fixed_scales[k] for k in ("p", "p", "s", "ell")])
    if inlet_regime == "subsonic":
        if ("alpha_in" in inlet_data) == ("ell_in" in inlet_data):
            raise ValueError("Specify exactly one inlet swirl law: alpha_in or ell_in")
        h0, s0 = jnp.asarray(inlet_data["h0"]), jnp.asarray(inlet_data["s0"])
        fixed_angle = "alpha_in" in inlet_data
        if fixed_angle:
            angle = float(inlet_data["alpha_in"])
            if not np.isfinite(angle) or abs(angle) >= 0.5*np.pi:
                raise ValueError("Forward-flow inlet angle must lie strictly between -pi/2 and pi/2")
            cos_alpha, sin_alpha = np.cos(angle), np.sin(angle)
        else:
            prescribed_ell = jnp.asarray(inlet_data["ell_in"])
            inlet_scale = inlet_scale.at[2].set(fixed_scales["ell"])
    else:
        prescribed_state = jnp.asarray(inlet_data["state"])
        if prescribed_state.shape != (4,):
            raise ValueError("Supersonic inlet state must have four primitives")
    pressure_back = jnp.asarray(outlet_pressure) if outlet_regime == "subsonic" else None
    left_factor = float((faces[0]-centers[0])/(centers[0]-centers[1]))
    right_factor = float((faces[-1]-centers[-1])/(centers[-1]-centers[-2]))
    outlet = EulerOutlet(fluid,outlet_pressure,np.array([fixed_scales[k] for k in ("p","a","ell","h")]),
        fixed_scales["s"],right_factor,outlet_formulation,
        'pressure' if outlet_formulation=='finite_wave' or outlet_regime=='subsonic' else 'supersonic_outflow')
    r_in = float(radii[0])
    expected_shape = (len(centers)+2, 4)

    def residual(z):
        z = jnp.asarray(z)
        if z.shape != expected_shape:
            raise ValueError(f"Full state must have shape {expected_shape}")
        interior = z[1:-1]
        if inlet_regime == "subsonic" or outlet_regime == "subsonic":
            state_cells = thermodynamic_state(interior, fluid)
            impedance = state_property(state_cells, "rho")*state_property(state_cells, "a")
        extrap_left = interior[0]+left_factor*(interior[0]-interior[1])
        extrap_right = interior[-1]+right_factor*(interior[-1]-interior[-2])
        if inlet_regime == "subsonic":
            inlet = z[0]
            state_in = thermodynamic_state(inlet, fluid)
            swirl = (inlet[2]/r_in*cos_alpha-inlet[1]*sin_alpha
                     if fixed_angle else inlet[2]-prescribed_ell)
            acoustic = inlet[0]-extrap_left[0]-impedance[0]*(inlet[1]-extrap_left[1])
            rows_in = jnp.stack((
                inlet[3]+0.5*(inlet[1]**2+(inlet[2]/r_in)**2)-h0,
                state_property(state_in, "s")-s0, swirl, acoustic,
            ))/inlet_scale
        else:
            rows_in = (z[0]-prescribed_state)/primitive_scale
        rows_out=outlet.residual(z[-1],interior)
        # Preserve the established annular row ordering for acoustic pressure BCs.
        if outlet_regime=='subsonic' or outlet_formulation=='finite_wave':
            rows_out=rows_out[jnp.array([0,1,3,2])]
        return jnp.stack((rows_in, rows_out), axis=0)

    residual.inlet_regime = inlet_regime
    residual.outlet_regime = outlet_regime
    residual.scales = fixed_scales
    residual.outlet = outlet
    return residual


def boundary_mach(z, fluid):
    """Return signed meridional inlet/outlet Mach numbers for branch audits."""
    states = jnp.asarray(z)[jnp.asarray([0, -1])]
    return states[..., 1]/state_property(thermodynamic_state(states, fluid), "a")
