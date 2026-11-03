"""Phase-table audit and physical admissibility for the constrained state."""

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from .constrained_state import primitives


def table_stencil_mask(fluid, transport):
    """Audit the populated bicubic phase-table stencils without modifying EOS data.

    Parameters
    ----------
    fluid : object
        JAXProp phase fluid; analytic fluids need no stencil mask.
    transport : bool
        Require viscosity, conductivity and heat capacity as well as EOS fields.

    Returns
    -------
    jax.Array or None
        Boolean mask of shape (N_h-1, N_p-1), or None for analytic fluids.

    Notes
    -----
    Reject nonfinite coefficients, nonpositive required properties and
    metastable rows copied after failed spinodal extensions. This audit runs
    once at model construction on the host, outside compiled residuals.
    """
    table = getattr(fluid, "table", None)
    if table is None or not hasattr(fluid, "N_h"):
        return None
    shape = (fluid.N_h, fluid.N_p)
    nodes = np.ones(shape, dtype=bool)
    cells = np.ones((shape[0] - 1, shape[1] - 1), dtype=bool)
    positive = {"density", "temperature", "speed_of_sound"}
    required = positive | {"gruneisen", "entropy"}
    if transport:
        positive |= {"viscosity", "conductivity", "isobaric_heat_capacity"}
        required |= positive
    for key in required:
        values = np.asarray(table[key]["value"])
        nodes &= np.isfinite(values)
        if key in positive:
            nodes &= values > 0
        cells &= np.all(np.isfinite(np.asarray(table[key]["coeffs"])), axis=-1)
    # Metastable builders freeze failed spinodal extensions by copying the
    # previous density row. Such copied states are not accepted physical data.
    phase = getattr(fluid, "metastable_phase", None)
    density = np.asarray(table["density"]["value"])
    if phase == "liquid":
        frozen = np.zeros_like(nodes)
        frozen[1:] = np.maximum.accumulate(density[1:] == density[:-1], axis=0)
        nodes &= ~frozen
    elif phase == "vapor":
        frozen = np.zeros_like(nodes)
        frozen[:-1] = np.maximum.accumulate(
            (density[:-1] == density[1:])[::-1], axis=0
        )[::-1]
        nodes &= ~frozen
    return jnp.asarray(
        cells & nodes[:-1, :-1] & nodes[1:, :-1] & nodes[:-1, 1:] & nodes[1:, 1:]
    )


def cell_admissible(model, W):
    """Check phase fractions, thermodynamic domains, populated tables and wave speeds.

    Parameters
    ----------
    model : TwoFluidModel
        Configured phase fluids, masks and physical policies.
    W : jax.Array, shape (..., 8)
        Constrained states to audit.

    Returns
    -------
    jax.Array
        Scalar boolean requiring every supplied state to be admissible.

    Notes
    -----
    The phase-fraction threshold rejects states; it never clips them.
    Transport and saturation checks apply only when the selected physics needs them.
    """
    U = primitives(W)
    valid = (
        jnp.all(jnp.isfinite(W))
        & jnp.all(W[..., 0] > 0)
        & jnp.all(W[..., 2] > 0)
        & jnp.all(W[..., 4] > 0)
        & jnp.all(W[..., 1] > model.stabilization.alpha_min)
        & jnp.all(W[..., 1] < 1 - model.stabilization.alpha_min)
    )
    for fluid, hi, stencil_mask in zip(
        (model.fluid_g, model.fluid_l), (4, 5), model._phase_stencil_masks
    ):
        p, h = U[..., 0], U[..., hi]
        valid = valid & jnp.all(
            (p >= getattr(fluid, "p_min", 0.0))
            & (p <= getattr(fluid, "p_max", jnp.inf))
        )
        valid = valid & jnp.all(
            (h >= getattr(fluid, "h_min", -jnp.inf))
            & (h <= getattr(fluid, "h_max", jnp.inf))
        )
        if stencil_mask is not None:
            i = jnp.clip(
                jnp.floor((h - fluid.h_min) / fluid.delta_h).astype(int),
                0,
                fluid.N_h - 2,
            )
            index_pressure = jnp.where(p > 0, p, 1.0)
            j = jnp.clip(
                jnp.floor(
                    (jnp.log(index_pressure) - jnp.log(fluid.p_min)) / fluid.delta_logP
                ).astype(int),
                0,
                fluid.N_p - 2,
            )
            valid = valid & jnp.all(stencil_mask[i, j])
        state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        valid = valid & jnp.all(
            jnp.isfinite(state["rho"])
            & (state["rho"] > 0)
            & jnp.isfinite(state["a"])
            & (state["a"] > 0)
            & jnp.isfinite(state["T"])
            & (state["T"] > 0)
        )
        if model.closures is not None and model.source_scale:
            for key in ("viscosity", "conductivity", "cp"):
                valid = valid & jnp.all(jnp.isfinite(state[key]) & (state[key] > 0))
        if model._needs_saturation:
            sat_grid = getattr(fluid, "logP_sat_vals", None)
            if sat_grid is not None:
                valid = valid & jnp.all(
                    (p >= jnp.exp(sat_grid[0])) & (p <= jnp.exp(sat_grid[-1]))
                )
            sat = fluid.get_state_saturation(p)
            valid = valid & jnp.all(
                jnp.isfinite(sat["enthalpy"])
                & jnp.isfinite(sat["temperature"])
                & (sat["temperature"] > 0)
            )
    if model._needs_saturation:
        sg = model.fluid_g.get_state_saturation(U[..., 0])
        sl = model.fluid_l.get_state_saturation(U[..., 0])
        valid = valid & jnp.all(sg["enthalpy"] > sl["enthalpy"])
    speed = model.wave_speed(W)
    return valid & jnp.all(jnp.isfinite(speed))
