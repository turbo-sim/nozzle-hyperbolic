"""Selectable Euler endpoint laws for nozzle and annular face traces.

Primitive order is (p, normal_velocity, [advected components], h). The acoustic
law is EOS independent. The finite-wave law is currently restricted to JAXProp's
calorically perfect-gas backend; this restriction is explicit at construction.
Neither law clips velocities or substitutes a wall during nonlinear iteration.
Sustained physical inflow requires exterior data beyond a backpressure value.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np


class OutletTrace(NamedTuple):
    state: jax.Array
    mode: jax.Array  # 1 outgoing, 3 compression, 4 expansion, 5 sonic fan
    head_speed: jax.Array
    tail_speed: jax.Array

    @property
    def direct_flux(self):
        return (self.mode == 3) | (self.mode == 4) | (self.mode == 5)


@dataclass(frozen=True, eq=False)
class EulerOutlet:
    """One physical boundary definition, shared by local and coupled solves.

    ``pressure`` is prescribed external pressure. With ``finite_wave`` its
    influence is selected by wave speeds, including finite compression speeds.
    ``factor`` extrapolates the last two physical states to the endpoint.
    Scales order matches primitives; entropy_scale is separate.
    Fully outgoing extrapolation and Sod transmissive traces are separate kinds.
    """

    fluid: object
    pressure: float | None
    scales: object
    entropy_scale: float
    factor: float
    formulation: str = "acoustic"
    kind: str = "pressure"

    def __post_init__(self):
        if self.formulation not in ("acoustic", "finite_wave"):
            raise ValueError("outlet formulation must be acoustic or finite_wave")
        if self.kind not in ("pressure", "supersonic_outflow", "transmissive"):
            raise ValueError("Unknown outlet kind")
        scale = np.asarray(self.scales, dtype=float)
        if (
            scale.ndim != 1
            or scale.size < 3
            or not np.all(np.isfinite(scale) & (scale > 0))
        ):
            raise ValueError("Positive primitive scales are required")
        if (
            not np.isfinite(self.factor)
            or not np.isfinite(self.entropy_scale)
            or self.entropy_scale <= 0
        ):
            raise ValueError(
                "Finite extrapolation factor and positive entropy scale required"
            )
        if self.kind == "pressure" and (
            self.pressure is None
            or not np.isfinite(self.pressure)
            or self.pressure <= 0
        ):
            raise ValueError("A pressure outlet needs positive finite backpressure")
        # TODO: generalize finite compression Hugoniot and isentropic fan curves
        # to arbitrary EOS through get_state(). Perfect-gas algebra is confined
        # to _perfect_gas_wave below; all other physical evaluations use the EOS.
        if self.formulation == "finite_wave":
            if self.kind != "pressure":
                raise ValueError(
                    "finite_wave selects its outgoing branch; use kind='pressure'"
                )
            if not isinstance(self.fluid, jxp.FluidPerfectGas):
                raise ValueError(
                    "finite_wave currently requires jaxprop.FluidPerfectGas; use acoustic for general EOS (TODO)"
                )
            gamma = float(self.fluid.constants.gamma)
            if not np.isfinite(gamma) or gamma <= 1:
                raise ValueError("Perfect-gas gamma must exceed one")

    def thermodynamics(self, W):
        return self.fluid.get_state(jxp.HmassP_INPUTS, W[..., -1], W[..., 0])

    def extrapolate(self, cells):
        return cells[-1] + self.factor * (cells[-1] - cells[-2])

    def wave(self, cells):
        """Finite-wave target and branch diagnostics, with signed velocities."""
        if self.formulation != "finite_wave":
            raise ValueError("Wave diagnostics require finite_wave")
        interior = cells[-1]
        state = self.thermodynamics(interior)
        return self._perfect_gas_wave(
            interior, self.extrapolate(cells), state["rho"], state["a"]
        )

    def _perfect_gas_wave(self, interior, outgoing, density, sound):
        """The sole perfect-gas wave approximation; no h=cp*T zero assumption.

        Enthalpy increments preserve the supplied EOS energy reference. The
        signed continuation is useful for Newton trial states; it does not
        provide missing upstream entropy/swirl data for a physical inflow.
        """
        gamma = self.fluid.constants.gamma
        pressure, velocity, enthalpy = interior[0], interior[1], interior[-1]
        back = jnp.asarray(self.pressure)
        ratio = back / pressure
        beta = (gamma - 1.0) / (gamma + 1.0)
        acoustic = velocity - sound

        def result(target, mode, head, tail):
            return OutletTrace(target, jnp.int32(mode), head, tail)

        def outgoing_result(_):
            return result(outgoing, 1, acoustic, acoustic)

        def compression(_):
            speed = velocity - sound * jnp.sqrt(
                (gamma + 1.0) * ratio / (2.0 * gamma) + (gamma - 1.0) / (2.0 * gamma)
            )

            def incoming(_):
                rho = density * (ratio + beta) / (beta * ratio + 1.0)
                u = velocity - (back - pressure) * jnp.sqrt(
                    2.0 / ((gamma + 1.0) * density * (back + beta * pressure))
                )
                h = enthalpy + gamma / (gamma - 1.0) * (back / rho - pressure / density)
                target = interior.at[0].set(back).at[1].set(u).at[-1].set(h)
                return result(target, 3, speed, speed)

            return jax.lax.cond(
                speed < 0.0,
                incoming,
                lambda _: result(outgoing, 1, speed, speed),
                operand=None,
            )

        def expansion(_):
            a_star = sound * ratio ** ((gamma - 1.0) / (2.0 * gamma))
            u_star = velocity + 2.0 * (sound - a_star) / (gamma - 1.0)
            tail = u_star - a_star

            def incoming(_):
                def imposed(_):
                    h = enthalpy + (a_star**2 - sound**2) / (gamma - 1.0)
                    target = interior.at[0].set(back).at[1].set(u_star).at[-1].set(h)
                    return result(target, 4, acoustic, tail)

                def sonic(_):
                    a_sonic = ((gamma - 1.0) * velocity + 2.0 * sound) / (gamma + 1.0)
                    p_sonic = pressure * (a_sonic / sound) ** (
                        2.0 * gamma / (gamma - 1.0)
                    )
                    h = enthalpy + (a_sonic**2 - sound**2) / (gamma - 1.0)
                    target = (
                        interior.at[0].set(p_sonic).at[1].set(a_sonic).at[-1].set(h)
                    )
                    return result(target, 5, acoustic, tail)

                return jax.lax.cond(tail <= 0.0, imposed, sonic, operand=None)

            return jax.lax.cond(acoustic < 0.0, incoming, outgoing_result, operand=None)

        return jax.lax.cond(ratio > 1.0, compression, expansion, operand=None)

    def residual(self, trace, cells):
        scale = jnp.asarray(self.scales)
        if self.kind == "transmissive":
            return (trace - cells[-1]) / scale
        extrapolated = self.extrapolate(cells)
        if self.kind == "supersonic_outflow":
            return (trace - extrapolated) / scale
        if self.formulation == "finite_wave":
            return (trace - self.wave(cells).state) / scale
        state = self.thermodynamics(cells[-2:])
        impedance = state["rho"][-1] * state["a"][-1]
        entropy = state["s"][-1] + self.factor * (state["s"][-1] - state["s"][-2])
        trace_entropy = self.thermodynamics(trace)["s"]
        rows = (trace - extrapolated) / scale
        return (
            rows.at[0]
            .set((trace[0] - self.pressure) / scale[0])
            .at[1]
            .set(
                (trace[0] - extrapolated[0] + impedance * (trace[1] - extrapolated[1]))
                / scale[0]
            )
            .at[-1]
            .set((trace_entropy - entropy) / self.entropy_scale)
        )

    def direct_flux(self, cells):
        return (
            self.wave(cells).direct_flux
            if self.formulation == "finite_wave"
            else jnp.asarray(False)
        )

    def signal_speed(self, cells):
        if self.formulation == "finite_wave":
            wave = self.wave(cells)
            return jnp.maximum(jnp.abs(wave.head_speed), jnp.abs(wave.tail_speed))
        return jnp.asarray(0.0)


def transmissive_residual(full, scales):
    """Zero-gradient endpoint constraints used by a finite-domain Sod problem."""
    return jnp.stack((full[0] - full[1], full[-1] - full[-2])) / jnp.asarray(scales)
