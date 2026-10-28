"""Experimental perfect-gas outlet closures with finite-compression detection.

These functions provide the unchanged source outlet or the pressure-wave
boundary trace. The study's FVM adapter uses the pressure-wave trace to set the
physical outlet flux; interior faces, the inlet and physical storage are reused.

All formulas assume forward outflow and a calorically perfect gas with
``h = cp*T``. The pressure-wave law includes a reflecting-wall safeguard when
the prescribed pressure would induce backflow from a nonnegative interior
velocity. Negative interior velocity remains explicitly unsupported; no exterior
entropy, velocity or temperature is invented to complete an inflow model.
"""

import math

import jax
import jax.numpy as jnp


MODE_NAMES = {
    0: "source_outlet",
    1: "outgoing_extrapolation",
    3: "incoming_compression",
    4: "incoming_expansion",
    5: "sonic_fan",
    6: "unsupported_backflow",
    7: "outflow_wall_guard",
}


def make_outlet_law(original_outlet, fluid, p_back, variant):
    """Return ``outlet(full_state), diagnostics(full_state)`` JIT functions.

    Parameters
    ----------
    original_outlet : callable
        Unmodified source map taking an ``(N,3)`` physical ``(p,u,h)`` array.
    fluid : object
        The existing JAXProp perfect-gas backend, with scalar ``constants.gamma``.
        Its enthalpy zero must be the calorically perfect-gas ``cp*T`` convention.
    p_back : float
        Positive prescribed pressure [Pa].
    variant : {"source", "pressure_wave"}
        Unchanged source control or the prescribed-pressure incoming-family
        wave curve sampled at zero normal wave speed. The pressure-wave law
        replaces a would-be negative boundary velocity with the exact
        reflecting-wall compression trace when interior velocity is nonnegative.
        This is a unilateral outflow safeguard during pseudo-time convergence,
        rather than an exterior-reservoir model for physical backflow.

    Notes
    -----
    The pressure-wave map is a partial boundary Riemann construction. For an
    incoming compression it uses the shock Hugoniot curve; for an incoming
    expansion it uses the outgoing invariant and interior entropy. If a fan
    straddles the boundary, its sonic state is sampled instead of imposing an
    excessively low pressure. Fully outgoing waves retain the source-style
    extrapolated ghost, including its existing p/h bounds and area-weighted
    velocity, but remove the source Mach blend's residual near-sonic pressure
    forcing. The source variant retains that blend unchanged.

    Regime decisions are piecewise, using ``lax.cond``. Derivatives are those
    of the selected physical branch; there is no claim of differentiability
    through shock-entry or sonic active-set changes. No positivity clipping is
    introduced beyond the source's existing extrapolated-ghost bounds. Callers
    must reject invalid thermodynamic trial states
    and any ``diagnostics(...)["backflow"]`` before accepting an update.

    Diagnostics ``active`` is a compatibility alias for
    ``new_wave_override_active``: it does not mean that every other branch is
    pressure-insensitive. ``pressure_sensitive`` separately identifies local
    dependence on the prescribed pressure inside the selected branch. The wall
    trace and sonic-fan trace are locally independent of prescribed pressure.
    """
    if variant not in ("source", "pressure_wave"):
        raise ValueError("Unknown outlet variant")
    if not math.isfinite(float(p_back)) or float(p_back) <= 0:
        raise ValueError("p_back must be finite and positive")
    gamma = float(fluid.constants.gamma)
    if not math.isfinite(gamma) or gamma <= 1:
        raise ValueError("A perfect-gas gamma greater than one is required")
    prescribed_pressure = jnp.float64(p_back)
    beta = (gamma - 1.0) / (gamma + 1.0)
    # Match the source ghost extrapolation bounds, not new interior clipping.
    pressure_lower = float(getattr(fluid, "p_min", 1e4)) * 1.01
    pressure_upper = float(getattr(fluid, "p_max", 300e5)) * 0.99

    def evaluate(full):
        pressure, velocity, enthalpy = full[-2]
        density = gamma * pressure / ((gamma - 1.0) * enthalpy)
        sound = jnp.sqrt((gamma - 1.0) * enthalpy)
        mach = velocity / sound
        source_ghost = original_outlet(full)[-1]
        outgoing_ghost = source_ghost.at[0].set(
            jnp.clip(2.0 * pressure - full[-3, 0], pressure_lower, pressure_upper)
        )
        acoustic_speed = velocity - sound

        def outgoing(_):
            return outgoing_ghost, jnp.int32(1), acoustic_speed, acoustic_speed

        if variant == "source":
            ghost, mode, head, tail = (
                source_ghost,
                jnp.int32(0),
                acoustic_speed,
                acoustic_speed,
            )
        else:
            ratio = prescribed_pressure / pressure

            def compression(_):
                # The full compression speed depends on pressure amplitude,
                # unlike a decision based only on the interior u-a.
                shock_speed = velocity - sound * jnp.sqrt(
                    (gamma + 1.0) * ratio / (2.0 * gamma)
                    + (gamma - 1.0) / (2.0 * gamma)
                )

                def incoming(_):
                    downstream_density = density * (ratio + beta) / (beta * ratio + 1.0)
                    coefficient = 2.0 / ((gamma + 1.0) * density)
                    downstream_velocity = velocity - (
                        prescribed_pressure - pressure
                    ) * jnp.sqrt(coefficient / (prescribed_pressure + beta * pressure))
                    downstream_enthalpy = (
                        gamma
                        * prescribed_pressure
                        / ((gamma - 1.0) * downstream_density)
                    )
                    state = jnp.array(
                        [prescribed_pressure, downstream_velocity, downstream_enthalpy]
                    )
                    return state, jnp.int32(3), shock_speed, shock_speed

                def exiting(_):
                    return outgoing_ghost, jnp.int32(1), shock_speed, shock_speed

                return jax.lax.cond(shock_speed < 0.0, incoming, exiting, operand=None)

            def expansion(_):
                sound_star = sound * ratio ** ((gamma - 1.0) / (2.0 * gamma))
                velocity_star = velocity + 2.0 * (sound - sound_star) / (gamma - 1.0)
                tail_speed = velocity_star - sound_star

                def enters(_):
                    def prescribed_state(_):
                        density_star = density * ratio ** (1.0 / gamma)
                        enthalpy_star = (
                            gamma * prescribed_pressure / ((gamma - 1.0) * density_star)
                        )
                        state = jnp.array(
                            [prescribed_pressure, velocity_star, enthalpy_star]
                        )
                        return state, jnp.int32(4), acoustic_speed, tail_speed

                    def sonic_state(_):
                        sound_sonic = ((gamma - 1.0) * velocity + 2.0 * sound) / (
                            gamma + 1.0
                        )
                        sonic_pressure = pressure * (sound_sonic / sound) ** (
                            2.0 * gamma / (gamma - 1.0)
                        )
                        sonic_enthalpy = sound_sonic**2 / (gamma - 1.0)
                        state = jnp.array([sonic_pressure, sound_sonic, sonic_enthalpy])
                        return state, jnp.int32(5), acoustic_speed, tail_speed

                    return jax.lax.cond(
                        tail_speed <= 0.0, prescribed_state, sonic_state, operand=None
                    )

                return jax.lax.cond(
                    acoustic_speed < 0.0, enters, outgoing, operand=None
                )

            # At zero wave strength both curves have matching first derivatives.
            ghost, mode, head, tail = jax.lax.cond(
                ratio > 1.0, compression, expansion, operand=None
            )

        attempted_backflow = (velocity < 0.0) | (ghost[1] < 0.0)
        if variant == "pressure_wave":

            def wall_trace(_):
                # Solve f(p_wall)=u_i on the compression Hugoniot, setting
                # the trace velocity to zero. This positive-M form avoids
                # sqrt(M**2) and its AD singularity at rest: dq/dM=gamma at0.
                coefficient = gamma * (gamma + 1.0) / 2.0
                wall_ratio = (
                    1.0
                    + 0.5 * coefficient * mach**2
                    + 0.5
                    * mach
                    * jnp.sqrt(
                        coefficient**2 * mach**2 + 4.0 * coefficient * (1.0 + beta)
                    )
                )
                wall_pressure = pressure * wall_ratio
                wall_density = density * (wall_ratio + beta) / (beta * wall_ratio + 1.0)
                wall_enthalpy = gamma * wall_pressure / ((gamma - 1.0) * wall_density)
                wall_speed = velocity - sound * jnp.sqrt(
                    (gamma + 1.0) * wall_ratio / (2.0 * gamma)
                    + (gamma - 1.0) / (2.0 * gamma)
                )
                state = jnp.array(
                    [wall_pressure, jnp.asarray(0.0, full.dtype), wall_enthalpy]
                )
                return state, jnp.int32(7), wall_speed, wall_speed

            eligible = (velocity >= 0.0) & (ghost[1] < 0.0)
            ghost, mode, head, tail = jax.lax.cond(
                eligible, wall_trace, lambda _: (ghost, mode, head, tail), operand=None
            )
        backflow = (velocity < 0.0) | (ghost[1] < 0.0)
        mode = jnp.where(backflow, jnp.int32(6), mode)
        override = ((mode >= 3) & (mode <= 5)) | (mode == 7)
        source_pressure_weight = 0.5 * (1.0 + jnp.tanh((1.0 - jnp.abs(mach)) / 0.004))
        pressure_sensitive = jnp.where(
            override, (mode == 3) | (mode == 4), source_pressure_weight > 0.0
        )
        if variant == "pressure_wave":
            pressure_sensitive = (mode == 3) | (mode == 4)
        completed = full.at[-1].set(ghost)
        information = {
            "mode": mode,
            "interior_M": mach,
            "ghost_M": ghost[1] / jnp.sqrt((gamma - 1.0) * ghost[2]),
            "ghost_pressure": ghost[0],
            "wave_speed": head,
            "wave_head_speed": head,
            "wave_tail_speed": tail,
            "backflow": backflow,
            "unsupported_backflow": backflow,
            "attempted_backflow": attempted_backflow,
            "wall_guard_active": mode == 7,
            "active": override,
            "new_wave_override_active": override,
            "pressure_sensitive": pressure_sensitive & ~backflow,
        }
        return completed, information

    @jax.jit
    def outlet(full):
        return evaluate(full)[0]

    @jax.jit
    def diagnostics(full):
        return evaluate(full)[1]

    return outlet, diagnostics
