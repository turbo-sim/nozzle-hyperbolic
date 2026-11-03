"""PH-EOS wave curves for a pressure outlet.

The shock branch enforces the conservative Hugoniot. The expansion branch
integrates dh=dp/rho and du=-dp/(rho*a), using the supplied sound speed.
For independently interpolated properties these curves need not preserve the
separately interpolated entropy, or have identical weak-wave acoustic limits.
Classical compressive shocks and monotone expansion fans are assumed. Invalid
roots, out-of-domain paths and nonmonotone fans return an invalid trace; there
is no property clipping, ideal-gas substitution or flux fallback.
"""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np


def _properties(fluid, h, p):
    """Two required wave properties, with table-domain rejection before lookup."""
    domain = jnp.isfinite(h) & jnp.isfinite(p) & (p > 0)
    for name, value in (("h", h), ("p", p)):
        lower, upper = getattr(fluid, name + "_min", None), getattr(fluid, name + "_max", None)
        if lower is not None:
            domain = domain & (value >= lower)
        if upper is not None:
            domain = domain & (value <= upper)

    def evaluate(_):
        state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        rho, sound = state["rho"], state["a"]
        valid = jnp.isfinite(rho) & jnp.isfinite(sound) & (rho > 0) & (sound > 0)
        return jnp.stack((rho, sound)), valid

    return jax.lax.cond(
        domain,
        evaluate,
        lambda _: (jnp.full((2,), jnp.nan, dtype=jnp.result_type(h, p)), jnp.array(False)),
        operand=None,
    )


def _root(function, lower, upper):
    """Bracketed scalar root with implicit, forward- and reverse-mode AD.

    Forty-eight bisections resolve the bracket below 4e-15 of its width.
    Differentiation uses the root equation rather than the bisection decisions.
    """
    def solve(f, initial):
        del initial

        def step(_, bracket):
            lo, hi = bracket
            middle = 0.5 * (lo + hi)
            value = f(middle)
            return jax.lax.cond(value <= 0, lambda: (middle, hi), lambda: (lo, middle))

        lo, hi = jax.lax.fori_loop(0, 48, step, (lower, upper))
        return 0.5 * (lo + hi)

    return jax.lax.custom_root(
        function, 0.5 * (lower + upper), solve,
        lambda linear, rhs: rhs / linear(jnp.ones_like(rhs)),
    )


def compression_curve(fluid, interior, back):
    """Return compressed state, left-family shock speed and validity.

    The dimensionless enthalpy increment avoids subtracting nearly equal
    enthalpies in the weak-shock root. A midpoint differential volume secant
    resolves cancellation only below sqrt(machine epsilon) in relative
    pressure; it is not a replacement acoustic-property backend.
    """
    pi, ui, hi = interior[0], interior[1], interior[-1]
    props, initial_ok = _properties(fluid, hi, pi)
    vi = 1.0 / props[0]
    dp = back - pi
    increment = dp * vi
    lower, upper = jnp.asarray(0.5, dtype=hi.dtype), jnp.asarray(1.0, dtype=hi.dtype)
    maximum_h = getattr(fluid, "h_max", None)
    if maximum_h is not None:
        upper = jnp.minimum(upper, (maximum_h - hi) / increment)

    def residual(x):
        values, _ = _properties(fluid, hi + increment * x, back)
        return x - 0.5 * (1.0 + 1.0 / (values[0] * vi))

    bracket_ok = (upper >= lower) & (residual(lower) <= 0) & (residual(upper) >= 0)
    x = _root(residual, lower, upper)
    h = hi + increment * x
    target_values, target_ok = _properties(fluid, h, back)
    volume = 1.0 / target_values[0]

    def weak_secant(_):
        def volume_on_chord(t):
            values, _ = _properties(fluid, hi + t * increment * x, pi + t * dp)
            return 1.0 / values[0]

        # Differentiate the actual density interpolant along the Hugoniot
        # chord. Dividing its JVP by dp is safe: the numerator is O(dp),
        # without a subtraction of almost equal endpoint volumes.
        return -jax.jvp(volume_on_chord, (jnp.asarray(0.5),), (jnp.asarray(1.0),))[1] / dp

    tiny = dp / pi < np.sqrt(np.finfo(np.dtype(hi.dtype)).eps)
    inverse_mass_flux_squared = jax.lax.cond(
        tiny, weak_secant, lambda _: (vi - volume) / dp, operand=None,
    )
    root_inverse = jnp.sqrt(inverse_mass_flux_squared)
    speed = ui - vi / root_inverse
    velocity = ui - dp * root_inverse
    target = interior.at[0].set(back).at[1].set(velocity).at[-1].set(h)
    valid = (
        initial_ok & target_ok & bracket_ok
        & (jnp.abs(residual(x)) <= 2e-12)
        & (inverse_mass_flux_squared > 0)
        & jnp.all(jnp.isfinite(target)) & jnp.isfinite(speed)
    )
    return target, speed, valid


def expansion_curve(fluid, interior, back, *, panels=128, log_step=0.05, tolerance=2e-7):
    """Sample a left-family expansion at a stationary right boundary.

    RK4 step doubling estimates accumulated error in units (a_i**2, a_i).
    Panels adapt in log(p), with maximum width ``log_step`` and at most
    ``panels`` attempted panels, including rejected refinement attempts. The
    local error budget is ``tolerance*abs(panel_width/total_log_interval)``;
    accepted estimates therefore sum to at most the requested tolerance.
    Each accepted panel uses the two-half-step state. A fan crossing u-a=0
    is sampled with an implicit scalar root inside that panel, without
    traversing the unreachable part of the wave below the sonic pressure.

    Returns state, sampled-mode (4 pressure or 5 sonic), fan tail and validity.
    """
    pi, ui, hi = interior[0], interior[1], interior[-1]
    initial, initial_ok = _properties(fluid, hi, pi)
    sound = initial[1]
    scale = jnp.stack((sound * sound, sound))
    distance = jnp.log1p((back - pi) / pi)
    log_pi = jnp.log(pi)
    log_back = log_pi + distance
    initial_width = jnp.minimum(jnp.abs(distance), log_step)
    interval = jnp.where(distance != 0, jnp.abs(distance), 1.0)

    def pressure_at(log_p):
        # Preserve known endpoints exactly. exp(log(p)) can round one ulp
        # outside a table whose bound equals the prescribed pressure. This
        # uses the supplied anchors, without clipping any intermediate state.
        return jnp.where(log_p == log_pi, pi, jnp.where(log_p == log_back, back, jnp.exp(log_p)))

    def derivative(log_p, y):
        pressure = pressure_at(log_p)
        values, valid = _properties(fluid, y[0], pressure)
        return jnp.stack((pressure / values[0], -pressure / (values[0] * values[1]))), valid

    def rk4(log_p, y, width):
        k1, ok1 = derivative(log_p, y)
        k2, ok2 = derivative(log_p + 0.5 * width, y + 0.5 * width * k1)
        k3, ok3 = derivative(log_p + 0.5 * width, y + 0.5 * width * k2)
        k4, ok4 = derivative(log_p + width, y + width * k3)
        return y + width * (k1 + 2 * k2 + 2 * k3 + k4) / 6, ok1 & ok2 & ok3 & ok4

    def fine_step(log_p, y, width):
        half, ok1 = rk4(log_p, y, 0.5 * width)
        fine, ok2 = rk4(log_p + 0.5 * width, half, 0.5 * width)
        return fine, ok1 & ok2

    # A statically bounded loop supports reverse AD through the accepted RK
    # path. Refinement and branch decisions are piecewise differentiable;
    # scalar sonic roots retain their implicit derivative.
    # state: h/u, log pressure, previous acoustic speed, error sum, valid,
    # sampled sonic branch, sampled fan-tail speed, next positive panel width.
    start = (
        jnp.stack((hi, ui)), log_pi, ui - sound, jnp.asarray(0.0),
        initial_ok, jnp.array(False), ui - sound, initial_width,
    )

    def panel(index, carry):
        del index
        y, log_p, acoustic, error, valid, sonic, tail, width = carry

        def advance(_):
            remaining = log_p - log_back
            positive_width = jnp.minimum(width, remaining)
            step = -positive_width
            end_logp = jnp.where(positive_width == remaining, log_back, log_p + step)
            coarse, coarse_ok = rk4(log_p, y, step)
            fine, fine_ok = fine_step(log_p, y, step)
            values, end_ok = _properties(fluid, fine[0], pressure_at(end_logp))
            new_acoustic = fine[1] - values[1]
            defect = jnp.max(jnp.abs((fine - coarse) / scale)) / 15
            budget = tolerance * positive_width / interval
            monotone = new_acoustic >= acoustic - 1e-10 * sound
            path_ok = coarse_ok & fine_ok & end_ok & monotone
            accept = path_ok & jnp.isfinite(defect) & (defect <= budget)
            next_width = jnp.minimum(
                log_step, jnp.where(defect < budget / 32, 2 * positive_width, positive_width),
            )

            def sample_sonic(_):
                def equation(fraction):
                    point, _ = fine_step(log_p, y, step * fraction)
                    state, _ = _properties(fluid, point[0], pressure_at(log_p + step * fraction))
                    return (point[1] - state[1]) / sound

                fraction = _root(equation, jnp.asarray(0.0), jnp.asarray(1.0))
                point, point_ok = fine_step(log_p, y, step * fraction)
                sampled_logp = log_p + step * fraction
                state, state_ok = _properties(fluid, point[0], pressure_at(sampled_logp))
                point = point.at[1].set(state[1])
                return (
                    point, sampled_logp, jnp.asarray(0.0), error + defect,
                    path_ok & point_ok & state_ok & (jnp.abs(equation(fraction)) <= 2e-12),
                    jnp.array(True), new_acoustic, next_width,
                )

            def continue_fan(_):
                return fine, end_logp, new_acoustic, error + defect, path_ok, sonic, new_acoustic, next_width

            def accepted(_):
                return jax.lax.cond(new_acoustic > 0, sample_sonic, continue_fan, operand=None)

            def refine(_):
                # An invalid coarse trial is never accepted or clipped. A
                # smaller trial can avoid a numerical overshoot; genuinely
                # inadmissible/nonmonotone paths exhaust the explicit budget.
                smaller = 0.5 * positive_width
                can_advance = (log_p - smaller < log_p) & (smaller > 0)
                return y, log_p, acoustic, error, valid & can_advance, sonic, tail, smaller

            return jax.lax.cond(accept, accepted, refine, operand=None)

        return jax.lax.cond((log_p > log_back) & valid & ~sonic, advance, lambda _: carry, operand=None)

    y, log_p, _, error, valid, sonic, tail, _ = jax.lax.fori_loop(0, panels, panel, start)
    pressure = jnp.where(sonic, pressure_at(log_p), back)
    target = interior.at[0].set(pressure).at[1].set(y[1]).at[-1].set(y[0])

    def zero_width(_):
        # A skipped zero-length integration has the correct value but loses
        # its derivative with respect to the initial pressure at fixed back
        # pressure. Retain the exact differential wave curve at that point.
        dp = back - pi
        point = (
            interior.at[0].set(back)
            .at[1].set(ui - dp / (initial[0] * sound))
            .at[-1].set(hi + dp / initial[0])
        )
        return point

    target = jax.lax.cond(back == pi, zero_width, lambda _: target, operand=None)
    valid = valid & (sonic | (log_p <= log_back)) & (error <= tolerance) & jnp.all(jnp.isfinite(target))
    return target, jnp.where(sonic, jnp.int32(5), jnp.int32(4)), tail, valid
