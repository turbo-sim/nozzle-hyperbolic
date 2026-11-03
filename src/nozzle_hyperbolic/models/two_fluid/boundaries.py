"""Algebraic inlet/outlet equations for the two-fluid endpoint traces."""

import jax.numpy as jnp
import jaxprop as jxp

from .constrained_state import primitives


def boundary_residual(model, full, time):
    """Assemble six primitive and two EOS equations at each endpoint trace.

    Parameters
    ----------
    model : TwoFluidModel
        Model with validated BoundaryConfig and fixed physical scales.
    full : jax.Array, shape (N+2, 8)
        N physical cells and two algebraic endpoint traces.
    time : scalar
        Time passed to a callable back pressure, in s.

    Returns
    -------
    jax.Array, shape (2, 8)
        Dimensionless inlet/outlet equations, including both local EOS constraints.

    Notes
    -----
    Stagnation data prescribe dominant-phase entropy, phase total enthalpies,
    quality and slip; cross-multiplied quality remains defined at rest.
    The characteristic outlet uses the incoming acoustic eigenvector.
    ``characteristic_old`` retains the shared-velocity pressure target with Mach
    blending; it is a supported policy name, not a second solver implementation.
    The generic engine chooses coupled or segregated numerical treatment.
    """
    U = primitives(full)
    bc = model.boundaries
    left, neighbor = U[0], U[1]
    if bc.inlet == "transmissive":
        inlet = (left - neighbor) / model.primitive_scales
    elif bc.inlet == "static_state":
        inlet = (left - jnp.asarray(bc.inlet_state)) / model.primitive_scales
    elif bc.inlet == "subsonic_inflow":
        target = jnp.asarray(bc.inlet_state).at[0].set(neighbor[0])
        inlet = (left - target) / model.primitive_scales
    elif bc.inlet == "total_state":
        extrap = neighbor + model._left_factor * (neighbor - U[2])
        target = (
            extrap.at[3]
            .set(bc.alpha0)
            .at[4]
            .set(bc.H_g0 - 0.5 * left[1] ** 2)
            .at[5]
            .set(bc.H_l0 - 0.5 * left[2] ** 2)
        )
        inlet = (left - target) / model.primitive_scales
    else:
        eg, el = model._properties(left)
        ng, nl = model._properties(neighbor)
        mass_g, mass_l = left[3] * eg.rho, (1 - left[3]) * el.rho
        mix_u = (mass_g * left[1] + mass_l * left[2]) / (mass_g + mass_l)
        mix_neighbor = (
            neighbor[3] * ng.rho * neighbor[1]
            + (1 - neighbor[3]) * nl.rho * neighbor[2]
        ) / (neighbor[3] * ng.rho + (1 - neighbor[3]) * nl.rho)
        fluid, hi = (
            (model.fluid_g, 4) if bc.dominant_phase == "gas" else (model.fluid_l, 5)
        )
        entropy = fluid.get_state(jxp.HmassP_INPUTS, left[hi], left[0])["s"]
        # Cross-multiplied quality avoids division by a possibly zero
        # inlet velocity and determines alpha through EOS and fixed slip.
        composition = ((1 - bc.quality) * mass_g * bc.slip - bc.quality * mass_l) / (
            model.constraint_scales[0] + model.constraint_scales[1]
        )
        inlet = jnp.stack(
            (
                (entropy - bc.s0) / model.entropy_scale,
                (left[1] - bc.slip * left[2]) / model.primitive_scales[1],
                (mix_u - mix_neighbor) / model.primitive_scales[1],
                composition,
                (left[4] + 0.5 * left[1] ** 2 - bc.H_g0) / model.primitive_scales[4],
                (left[5] + 0.5 * left[2] ** 2 - bc.H_l0) / model.primitive_scales[5],
            )
        )
    inside = U[-2]
    target = inside
    if bc.outlet != "transmissive":
        eg, el = model._properties(inside)
        ev, right = model._wave_data(inside)[:2]
        pressure = (
            bc.back_pressure(time) if callable(bc.back_pressure) else bc.back_pressure
        )
        weight = 1.0
        if bc.outlet in ("characteristic", "characteristic_old"):
            mach = jnp.abs(ev[2] + ev[3]) / (ev[2] - ev[3])
            weight = 0.5 * (1 + jnp.tanh((1 - mach) / bc.mach_width))
        dp = weight * bc.pressure_relaxation * (pressure - inside[0])
        if bc.outlet == "characteristic":
            target = inside + dp * right[:, 3] / right[0, 3]
        else:
            rho = inside[3] * eg.rho + (1 - inside[3]) * el.rho
            compliance = inside[3] / (eg.rho * eg.a**2) + (1 - inside[3]) / (
                el.rho * el.a**2
            )
            du = -dp / jnp.sqrt(rho / compliance)
            target = inside.at[0].add(dp).at[1].add(du).at[2].add(du)
    outlet = (U[-1] - target) / model.primitive_scales
    traces = full[jnp.array((0, full.shape[0] - 1))]
    constraints = model.cell_constraints(traces) / jnp.asarray(model.constraint_scales)
    return jnp.concatenate((jnp.stack((inlet, outlet)), constraints), axis=1)
