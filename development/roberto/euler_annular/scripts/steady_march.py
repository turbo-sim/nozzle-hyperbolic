"""Shared study-only compiled outer CFL loop, for the measured fixed-growth controller.

The production timestep, full Newton system and spatial operators are reused.
Default behavior preserves the original controller study. Optional diagnostics
record all four global balance components for the verification report.
"""

import jax
import jax.numpy as jnp

COLUMNS = ("step", "retry", "dt", "cfl", "inner_iterations", "inner_residual",
           "inner_status", "rejected_newton_trials", "accepted", "spatial_rms",
           "boundary_max", "conservation_max", "pseudo_time", "shock_proxy_m",
           "min_velocity_gradient", "inlet_M_m", "outlet_M_m", "residual_only_accept", "inner_relative_update")

def make_marcher(solver, max_steps, max_retries=12, cfl_max=1e6, tolerance=1e-8,
                 inner_acceptance="strict", record_balance=False):
    """One compiled trial; controller parameters are dynamic, shared kernels JIT once."""
    max_attempts = 2*max_steps+2*(max_retries+1)
    shape = (max_attempts, len(COLUMNS)+(4 if record_balance else 0))
    density, sound = solver.reference_density, solver.reference_sound_speed
    balance_scales = float(solver.mesh.A_face[0])*sound*jnp.asarray([
        density, density*sound, density*float(solver.mesh.r_face[0])*sound, density*sound**2])
    centers = jnp.asarray(solver.mesh.m_centers)

    def magnitude(metrics):
        return jnp.maximum(metrics["spatial_rms"],
                           jnp.maximum(metrics["boundary_max"], metrics["conservation_max"]))

    @jax.jit
    def march(initial, initial_cfl, growth, shrink, work_policy):
        metrics = solver.metrics(initial)
        # state, CFL, accepted count, rejected streak, attempts, time, monitor, log
        initial_carry = (initial, initial_cfl, jnp.int32(0), jnp.int32(0), jnp.int32(0),
                         jnp.float64(0), metrics, jnp.full(shape, jnp.nan))

        def condition(carry):
            _, _, accepted, retries, attempts, _, current, _ = carry
            return ((accepted < max_steps) & (retries <= max_retries)
                    & (attempts < max_attempts) & (magnitude(current) >= tolerance))

        def body(carry):
            state, cfl, accepted, retry, attempts, time, current, history = carry
            dt = solver.compute_dt(state, cfl)
            candidate, info = solver.step(state, dt)
            residual_only = ((info.status==1) & info.admissible
                             & (info.residual_norm<=solver.options.newton_residual_tolerance))
            inner_ok = info.admissible & (info.converged | (
                residual_only if inner_acceptance=="residual" else jnp.asarray(False)))
            trial_metrics = jax.lax.cond(inner_ok, lambda: solver.metrics(candidate), lambda: current)
            okay = (inner_ok & trial_metrics["admissible"]
                    & jnp.isfinite(magnitude(trial_metrics)))
            new_state = jnp.where(okay, candidate, state)
            new_metrics = jax.tree_util.tree_map(lambda a,b:jnp.where(okay,a,b), trial_metrics, current)
            new_time = time+jnp.where(okay, dt, 0.)
            gradient = jnp.diff(new_state[1:-1, 1])/jnp.diff(centers)
            index = jnp.argmin(gradient)
            shock = jnp.where(jnp.min(gradient) < -1e-8,
                              .5*(centers[index]+centers[index+1]), jnp.nan)
            # Same native work thresholds, including their change with the cap.
            cap = solver.options.newton_iterations
            adaptive_growth = jnp.where(info.iterations <= max(2, cap//3), growth,
                                       jnp.where(info.iterations <= 2*cap//3, jnp.sqrt(growth), 1.))
            factor = jnp.where(work_policy, adaptive_growth, growth)
            new_cfl = jnp.where(okay, jnp.minimum(cfl_max, cfl*factor), cfl*shrink)
            row = jnp.asarray([
                accepted+1, retry, dt, cfl, info.iterations, info.residual_norm,
                info.status, info.rejected_trials, okay, new_metrics["spatial_rms"],
                new_metrics["boundary_max"], new_metrics["conservation_max"],new_time,
                shock,jnp.min(gradient),new_metrics["boundary_mach"][0],new_metrics["boundary_mach"][1],
                okay & ~info.converged,
                info.relative_update,
            ], dtype=jnp.float64)
            if record_balance:
                flux = solver.face_flux(new_state)
                balance = (flux[-1]-flux[0]-jnp.sum(solver.source(new_state), axis=0))/balance_scales
                row = jnp.concatenate((row, balance))
            return (new_state, new_cfl, accepted+okay.astype(jnp.int32),
                    jnp.where(okay, 0, retry+1), attempts+1, new_time,
                    new_metrics, history.at[attempts].set(row))

        return jax.lax.while_loop(condition, body, initial_carry)

    return march
