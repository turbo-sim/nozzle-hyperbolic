"""Optional physical velocity sensor for conservative numerical filtering."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class VelocityGate:
    """Smoothly reduce filtering above a specified physical speed (m/s)."""

    cutoff: float | None = None
    width: float | None = None

    def __post_init__(self):
        if self.cutoff is None:
            if self.width is not None:
                raise ValueError("A filter velocity width requires a cutoff")
            return
        width = self.cutoff / 3 if self.width is None else self.width
        if (
            not np.all(np.isfinite([self.cutoff, width]))
            or min(self.cutoff, width) <= 0
        ):
            raise ValueError(
                "Filter velocity cutoff and width must be positive and finite"
            )
        object.__setattr__(self, "width", float(width))

    def __call__(self, velocity):
        if self.cutoff is None:
            return jnp.ones_like(velocity)
        return 0.5 * (1 + jnp.tanh((self.cutoff - jnp.abs(velocity)) / self.width))


def face_filter_speed(velocity, sound_speed, gate):
    """Average adjacent physical spectral radii with an optional velocity gate."""
    signal = jnp.abs(velocity) + sound_speed
    sensor = gate(velocity)
    return 0.5 * (signal[:-1] + signal[1:]) * 0.5 * (sensor[:-1] + sensor[1:])
