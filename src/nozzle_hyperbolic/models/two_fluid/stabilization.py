"""Optional two-fluid filters, phase pins and their explicit rate bound.

All filters and equation pins are disabled by default. Pins change the physical
equations and are restricted to implicit steady solutions. The generic engine
retains the unmodified residual for independent convergence reporting.
"""

import jax
import jax.numpy as jnp

from .constrained_state import conserved, primitives
from .matrices import phase_props
from .sources import evaporation_weight


def transform_cell_equations(model, full, integrated_rows, dt=None):
    """Apply explicit phase-pin policies to integrated cell equations.

    Parameters
    ----------
    model : TwoFluidModel
        Model-owned filter and phase-pin policies.
    full : jax.Array, shape (N+2, 8)
        Physical cells and algebraic endpoint traces.
    integrated_rows : jax.Array, shape (N, 6)
        Integrated spatial or backward-Euler balances.
    dt : scalar or None, optional
        Pseudo-time step in s; None selects the steady algebraic equations.

    Returns
    -------
    jax.Array, shape (N, 6)
        Selected equations; unpinned rows are unchanged.

    Notes
    -----
    The shared gate releases the saturation-energy pin where mass exchange
    starts. Momentum relaxation uses a finite time so its steady root is
    independent of pseudo-time step. Pins change the equations; the engine
    retains the original residual for separate reporting.
    """
    if not model.has_equation_pins:
        return integrated_rows
    U = primitives(full[1:-1])
    mg = full[1:-1, 2]
    result = integrated_rows
    if model.pins.saturation_energy:

        def one(u):
            sat_g = model.fluid_g.get_state_saturation(u[0])
            sat_l = model.fluid_l.get_state_saturation(u[0])
            liquid = phase_props(model.fluid_l, u[0], u[5])
            free = evaporation_weight(
                u[5],
                liquid.T,
                sat_l["enthalpy"],
                sat_l["temperature"],
                u[3],
                p=u[0],
                p_max=model.pins.pressure_cutoff,
            )
            return free, u[4] - sat_g["enthalpy"]

        free, enthalpy_delta = jax.lax.map(one, U)
        rate = 1 / model.reference_time if dt is None else 1.0
        pin = model._volumes * mg * enthalpy_delta * rate
        result = result.at[:, 4].set(free * result[:, 4] + (1 - free) * pin)
    if model.pins.momentum:
        free = jax.nn.sigmoid(4 * jnp.log(U[:, 3] / model.pins.momentum_alpha))
        rate = (1.0 if dt is None else dt) / model.pins.momentum_time
        pin = model._volumes * mg * (U[:, 1] - U[:, 2]) * rate
        result = result.at[:, 2].set(free * result[:, 2] + (1 - free) * pin)
    return result


def stabilization_flux(model, full, raw_flux):
    """Evaluate optional conservative void and thin-phase filter fluxes.

    Parameters
    ----------
    model : TwoFluidModel
        Model-owned filter and phase-pin policies.
    full : jax.Array, shape (N+2, 8)
        Physical cells and algebraic endpoint traces.
    raw_flux : jax.Array, shape (N+1, 6)
        Physical face flux, supplying shape and dtype.

    Returns
    -------
    jax.Array, shape (N+1, 6)
        Filter flux density with zero endpoint contributions.

    Notes
    -----
    A third difference in conservative contents produces fourth-difference
    damping after flux divergence. Thin-phase momentum/energy rows may also
    use pressure-sensor second differences. The engine applies face area
    exactly once; no phase projection or clipping is introduced.
    """
    cfg = model.stabilization
    result = jnp.zeros_like(raw_flux)
    if cfg.void_strength == 0 and cfg.thin_phase_strength == 0:
        return result
    U, q = primitives(full), conserved(full)
    speed = model.filter_signal_speed(full)
    third = q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]
    if cfg.void_strength:
        face_speed = speed
        if cfg.void_velocity_cutoff is not None:
            slow = jnp.minimum(jnp.abs(U[:, 1]), jnp.abs(U[:, 2]))
            gate = 0.5 * (
                1
                + jnp.tanh((cfg.void_velocity_cutoff - slow) / cfg.void_velocity_width)
            )
            face_speed = speed * 0.5 * (gate[:-1] + gate[1:])
        weights = jnp.ones_like(third)
        if cfg.void_mass_cutoff is not None:
            for row, alpha in ((0, U[:, 3]), (1, 1 - U[:, 3])):
                gate = 0.5 * (
                    1
                    + jnp.tanh(
                        (jnp.log10(alpha) - jnp.log10(cfg.void_mass_cutoff)) / 0.25
                    )
                )
                weights = weights.at[:, row].set(
                    gate[:-3] * gate[1:-2] * gate[2:-1] * gate[3:]
                )
        result = result.at[1:-1].add(
            cfg.void_strength * face_speed[1:-1, None] * third * weights
        )
    if cfg.thin_phase_strength:
        weights = []
        for alpha in (U[:, 3], 1 - U[:, 3]):
            loga = jnp.log10(alpha)
            gate = 0.5 * (
                1
                - jnp.tanh(
                    (loga - jnp.log10(cfg.thin_phase_cutoff)) / cfg.thin_phase_log_width
                )
            )
            gate *= 0.5 * (
                1
                + jnp.tanh(
                    (loga - jnp.log10(cfg.thin_phase_lower)) / cfg.thin_phase_log_width
                )
            )
            weights.append(0.5 * (gate[:-1] + gate[1:]))
        wg, wl = weights
        zero = jnp.zeros_like(wg)
        rows = jnp.stack((zero, zero, wg, wl, wg, wl), axis=1)
        p = U[:, 0]
        curvature = jnp.abs(p[2:] - 2 * p[1:-1] + p[:-2]) / (
            p[2:] + 2 * p[1:-1] + p[:-2]
        )
        sensor = jnp.zeros_like(p).at[1:-1].set(jnp.minimum(1.0, 10 * curvature))
        sf = jnp.maximum(sensor[:-1], sensor[1:])[1:-1, None]
        diff = q[2:-1] - q[1:-2]
        result = result.at[1:-1].add(
            cfg.thin_phase_strength
            * speed[1:-1, None]
            * rows[1:-1]
            * ((1 - sf) * third - cfg.thin_phase_second_order * sf * diff)
        )
    return result


def filter_signal_speed(model, full):
    """Evaluate face-averaged signal speeds for the optional filters.

    Parameters
    ----------
    model : TwoFluidModel
        Model-owned filter and phase-pin policies.
    full : jax.Array, shape (N+2, 8)
        Physical cells and algebraic endpoint traces.

    Returns
    -------
    jax.Array, shape (N+1,)
        Nonnegative face speed in m/s.

    Notes
    -----
    The 'wood' selector uses the model's maximum wave speed. The 'frozen'
    selector adds the acoustic-pair center magnitude and maximum phase sound speed.
    """
    if model.stabilization.speed_scale == "wood":
        speed = model.wave_speed(full)
    else:

        def one(U):
            eg, el = model._properties(U)
            ev = model._wave_data(U)[0]
            return jnp.abs(0.5 * (ev[2] + ev[3])) + jnp.maximum(eg.a, el.a)

        speed = jax.vmap(one)(primitives(full))
    return 0.5 * (speed[:-1] + speed[1:])


def explicit_timestep(model, full):
    """Estimate a conservative filter and local constrained-source time bound.

    Parameters
    ----------
    model : TwoFluidModel
        Model-owned filter and phase-pin policies.
    full : jax.Array, shape (N+2, 8)
        Physical cells and algebraic endpoint traces.

    Returns
    -------
    jax.Array
        Scalar step bound in s, or infinity when no active rate is present.

    Notes
    -----
    The source restriction uses the infinity norm of the exact constrained
    local source linearization in fixed scaled variables. It is a local
    stiffness estimate, not a positivity theorem across closure switches.
    The cell-local source Jacobian uses common seeds without cross-cell coupling.
    """
    cfg = model.stabilization
    filter_rate = jnp.asarray(0.0, dtype=full.dtype)
    if cfg.void_strength or cfg.thin_phase_strength:
        speed = jnp.max(model.filter_signal_speed(full))
        # Row sum bounds of first/third difference flux divergences are 4/16.
        strength = (
            16 * cfg.void_strength
            + (16 + 4 * cfg.thin_phase_second_order) * cfg.thin_phase_strength
        )
        filter_rate = (
            strength
            * speed
            * jnp.max(
                jnp.maximum(model._face_areas[:-1], model._face_areas[1:])
                / model._volumes
            )
        )
    source_rate = jnp.asarray(0.0, dtype=full.dtype)
    if model.closures is not None and model.source_scale:
        # Source is cell local. Obtain its complete block Jacobian using
        # one common seed for all cells (no cross-cell dependencies).
        cells = full[1:-1]

        def density_source(C):
            state = full.at[1:-1].set(C)
            return model.integrated_source(state) / model._volumes[:, None]

        basis = jnp.eye(model.nvar, dtype=full.dtype)
        source_blocks = jax.vmap(
            lambda seed: jax.jvp(
                density_source, (cells,), (jnp.broadcast_to(seed, cells.shape),)
            )[1]
        )(basis)
        source_blocks = jnp.moveaxis(source_blocks, 0, -1)
        storage = jax.vmap(
            jax.jacfwd(
                lambda w: jnp.concatenate((conserved(w), model.cell_constraints(w)))
            )
        )(cells)
        padded = jnp.concatenate(
            (source_blocks, jnp.zeros((cells.shape[0], 2, 8), dtype=full.dtype)), axis=1
        )
        evolution = jnp.linalg.solve(storage, padded)
        scales = jnp.asarray(model.variable_scales)
        scaled = evolution * scales[None, None, :] / scales[None, :, None]
        source_rate = jnp.max(jnp.sum(jnp.abs(scaled), axis=-1))
    rate = filter_rate + source_rate
    return jnp.where(rate > 0, 1 / jnp.where(rate > 0, rate, 1.0), jnp.inf)
