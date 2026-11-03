"""Development-only cellwise pseudo-time weights over production implicit kernels."""

from copy import copy

import jax
import jax.numpy as jnp


def local_operator(base):
    """Shallow-copy bindings; change only the outer step's cell time weights."""
    scheme, model = base.scheme, base.model
    if (scheme.time_integrator != "euler-implicit"
            or base.effective_boundary_treatment != "coupled"
            or base.nconstraints or base.ntransport != model.nvar
            or base.equation_policy_active or model.nonconservative_mode != "none"
            or scheme.filter_strength != 0):
        raise ValueError("Local experiment requires coupled square conservative unfiltered backward Euler")
    op = copy(base)
    widths = jnp.asarray(base.mesh.widths)
    ends = jnp.array([0, -1])
    boundary_speed = getattr(model, "boundary_wave_speed", None)

    def wave_times(full):
        speed = model.wave_speed(full[1:-1])
        times = widths / jnp.maximum(speed, 1e-300)
        if boundary_speed is not None:
            endpoint_times = widths[ends] / jnp.maximum(boundary_speed(full), 1e-300)
            times = times.at[ends].set(jnp.minimum(times[ends], endpoint_times))
        return times

    def cell_steps(full, minimum_step):
        times = wave_times(full)
        return (minimum_step * (times / jnp.min(times)))[:, None]

    def step(full, minimum_step, time=0.):
        # Compute weights from the old iterate, before entering Newton. The
        # unchanged production step differentiates only its trial state.
        return base.step(full, cell_steps(full, minimum_step), time)

    op.local_wave_times = jax.jit(wave_times)
    op.local_cell_steps = jax.jit(cell_steps)
    op.step = jax.jit(step)
    return op
