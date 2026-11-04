"""Development-only nonlinear FAS on the shared physical-model contract.

No concrete equation identities, EOS inverses or boundary laws live here.
Coarse right-hand sides force transported balances only; algebraic cell and
boundary equations remain homogeneous. All levels are bound once and reused.
"""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, Solution
from nozzle_hyperbolic.solvers.fvm.drivers import sampler
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from nozzle_hyperbolic.solvers.time_marching.explicit import make_explicit_step
from nozzle_hyperbolic.solvers.time_marching.implicit import (
    NewtonInfo,
    make_newton_solver,
)


@jax.jit
def restrict_integrated(values):
    """Sum signed integrated rows over pairs of nested child cells."""
    return values.reshape((values.shape[0] // 2, 2, values.shape[1])).sum(axis=1)


restrict_contents = restrict_integrated


@jax.jit
def prolong_density(values):
    """Piecewise-constant additive density correction, preserving parent means."""
    return jnp.repeat(values, 2, axis=0)


def _combine(first, second):
    return NewtonInfo(
        first.iterations + second.iterations,
        jnp.maximum(first.relative_update, second.relative_update),
        jnp.maximum(first.residual_norm, second.residual_norm),
        first.converged & second.converged,
        first.admissible & second.admissible,
        first.rejected_trials + second.rejected_trials,
        jnp.where(~first.converged, first.status, second.status),
    )


class StepWork(NamedTuple):
    """Observed full implicit Jacobian work, excluding recovery solves."""

    jacobian_builds: object
    frozen_attempts: object
    refreshes: object
    supplied_jacobian_uses: object


class InexactWork(NamedTuple):
    """Work in a capped implicit smoother, including its exact fallback."""

    jacobian_builds: object
    accepted_inexact: object
    exact_converged: object
    fallbacks: object
    recovery_iterations: object
    recovery_calls: object
    recovery_rejected_trials: object


@dataclass
class Level:
    """One immutable model/operator binding and its forced numerical kernels."""

    model: object
    scheme: Scheme
    explicit: bool = False
    jacobian_reuse: str = "none"
    smoother_newton_iterations: int = 2
    smoother_residual_ratio: float = 0.1
    smoother_boundary_tolerance: float = 1e-9
    smoother_constraint_tolerance: float = 1e-9

    def __post_init__(self):
        if self.jacobian_reuse not in ("none", "step"):
            raise ValueError("jacobian_reuse must be none or step")
        if bool(
            getattr(
                self.model,
                "has_equation_pins",
                callable(getattr(self.model, "transform_cell_equations", None)),
            )
        ):
            raise ValueError("FAS does not support equation-changing model policies")
        self.operator = op = Operator(self.model, self.scheme)
        self.volumes = jnp.asarray(self.model.mesh.volumes)
        self.content_scale = self.volumes[:, None] * jnp.asarray(
            self.model.conservative_scales
        )
        self.reference_time = float(self.model.reference_time)
        n, nv, nt = self.model.mesh.n_cells, self.model.nvar, op.ntransport
        shape = op.shape
        variable_scale = jnp.broadcast_to(
            jnp.asarray(self.model.variable_scales), shape
        ).ravel()

        def forced_residual(new_flat, old_flat, dt, time, forcing):
            # Exact production residual at f=0, including large-step conditioning.
            rows = op.residual(new_flat, old_flat, dt, time).reshape(shape)
            factor = dt / jnp.maximum(1.0, dt / self.reference_time)
            return (
                rows.at[1:-1, :nt].add(-factor * forcing / self.content_scale).ravel()
            )

        self.forced_residual = jax.jit(forced_residual)
        self.forced_blocks = make_block_operators(
            forced_residual,
            n + 2,
            nv,
            op.radius,
            jacobian_mode=self.scheme.jacobian_mode,
            linear_solver=self.scheme.linear_solver,
        )
        newton = make_newton_solver(
            forced_residual,
            self.forced_blocks["direction"],
            lambda _: variable_scale,
            max_iterations=self.scheme.newton_iterations,
            tolerance=self.scheme.newton_update_tolerance,
            residual_tolerance=self.scheme.newton_residual_tolerance,
            admissible=lambda flat: op.is_admissible(flat.reshape(shape)),
            backtracking=self.scheme.newton_backtracking,
            max_backtracks=self.scheme.max_backtracks,
        )

        @jax.jit
        def implicit_step(full, forcing, dt, time=0.0):
            result, info = newton(full.ravel(), full.ravel(), dt, time, forcing)
            return result.reshape(shape), info

        self.implicit_step = implicit_step

        # Modified Newton changes the direction only: every line search and
        # stopping decision still evaluates the original nonlinear residual.
        # A stale matrix that fails is followed by ordinary fresh Newton from
        # the last valid iterate, with the original old state and timestep.
        def frozen_residual(point, old, dt, time, forcing, blocks):
            return forced_residual(point, old, dt, time, forcing)

        def frozen_direction(point, old, dt, time, forcing, blocks):
            value = forced_residual(point, old, dt, time, forcing)
            return self.forced_blocks["solve_blocks"](blocks, -value)

        frozen_newton = make_newton_solver(
            frozen_residual,
            frozen_direction,
            lambda _: variable_scale,
            max_iterations=self.scheme.newton_iterations,
            tolerance=self.scheme.newton_update_tolerance,
            residual_tolerance=self.scheme.newton_residual_tolerance,
            admissible=lambda flat: op.is_admissible(flat.reshape(shape)),
            backtracking=self.scheme.newton_backtracking,
            max_backtracks=self.scheme.max_backtracks,
        )

        @jax.jit
        def implicit_step_frozen(full, forcing, dt, blocks, time=0.0):
            """Try supplied block Jacobian, then refresh on exact-root failure.

            ``blocks`` has the native conditioned backward-Euler row units and
            band layout. Its construction is charged by the caller. Block
            factorization is repeated by the native linear solver per solve;
            this prototype reuses AD Jacobians, not LU factorizations.
            """
            old = full.ravel()
            point, first = frozen_newton(old, old, dt, time, forcing, blocks)

            def refresh(_):
                result, last = newton(point, old, dt, time, forcing)
                # Failed frozen work is retained, while the final exact solve
                # determines acceptance. AND-combining statuses would wrongly
                # reject a successful refresh.
                combined = last._replace(
                    iterations=first.iterations + last.iterations,
                    rejected_trials=first.rejected_trials + last.rejected_trials,
                )
                return result, combined, StepWork(last.iterations, 1, 1, 1)

            def done(_):
                return point, first, StepWork(jnp.asarray(0), 1, 0, 1)

            result, info, work = jax.lax.cond(
                first.converged & first.admissible, done, refresh, operand=None
            )
            return result.reshape(shape), info, work

        self.implicit_step_frozen = implicit_step_frozen

        @jax.jit
        def implicit_step_with_work(full, forcing, dt, time=0.0):
            if self.jacobian_reuse == "step":
                blocks = self.forced_blocks["jacobian_blocks"](
                    full.ravel(), full.ravel(), dt, time, forcing
                )
                result, info, work = implicit_step_frozen(
                    full, forcing, dt, blocks, time
                )
                return result, info, work._replace(
                    jacobian_builds=work.jacobian_builds + 1,
                    supplied_jacobian_uses=jnp.asarray(0),
                )
            result, info = implicit_step(full, forcing, dt, time)
            return result, info, StepWork(info.iterations, 0, 0, 0)

        self.implicit_step_with_work = implicit_step_with_work

        limited_newton = make_newton_solver(
            forced_residual,
            self.forced_blocks["direction"],
            lambda _: variable_scale,
            max_iterations=self.smoother_newton_iterations,
            tolerance=self.scheme.newton_update_tolerance,
            residual_tolerance=self.scheme.newton_residual_tolerance,
            admissible=lambda flat: op.is_admissible(flat.reshape(shape)),
            backtracking=True,
            max_backtracks=self.scheme.max_backtracks,
        )

        @jax.jit
        def implicit_smoother_step(full, forcing, dt, time=0.0):
            """Take capped Newton work under a separate smoothing criterion.

            Native Newton convergence is never asserted for an inexact step.
            A boundary-completed, admissible trial must reduce the actual BE
            residual by the configured factor and meet the unchanged algebraic
            targets. Otherwise ordinary fresh Newton continues for the same
            old state, forcing and timestep. Constrained models currently use
            the exact path selected by the workspace instead of this kernel.
            """
            old = full.ravel()
            initial_norm = jnp.sqrt(jnp.mean(forced_residual(old, old, dt, time, forcing) ** 2))
            point, first = limited_newton(old, old, dt, time, forcing)
            provisional = point.reshape(shape)
            projected, boundary = op.complete_boundaries(provisional, time)
            projected_valid = boundary.converged & boundary.admissible & op.is_admissible(projected)
            def assess_projection(_):
                norm = jnp.sqrt(jnp.mean(forced_residual(projected.ravel(), old, dt, time, forcing) ** 2))
                bc = jnp.max(jnp.abs(op.boundary_residual(projected, time)))
                constraint = jnp.max(jnp.abs(op.cell_constraints(projected[1:-1])), initial=0.0)
                return norm, bc, constraint

            def invalid_projection(_):
                infinity = jnp.asarray(jnp.inf, full.dtype)
                return infinity, infinity, infinity

            current_norm, bc_max, constraint_max = jax.lax.cond(
                projected_valid, assess_projection, invalid_projection, operand=None
            )
            projection_update = jnp.max(jnp.abs(projected.ravel() - point) / variable_scale)
            relative_update = jnp.maximum(first.relative_update, projection_update)
            residual_floor = self.scheme.newton_residual_tolerance
            target = jnp.maximum(residual_floor, self.smoother_residual_ratio * initial_norm)
            accepted = (
                ((first.status == 0) | (first.status == 1))
                & projected_valid & jnp.isfinite(current_norm)
                & (current_norm <= target)
                & (bc_max <= self.smoother_boundary_tolerance)
                & (constraint_max <= self.smoother_constraint_tolerance)
            )
            exact = (
                first.converged & projected_valid
                & ((self.scheme.newton_residual_tolerance <= 0)
                   | (current_norm <= self.scheme.newton_residual_tolerance))
                & ((self.scheme.newton_update_tolerance <= 0)
                   | (relative_update <= self.scheme.newton_update_tolerance))
            )

            def done(_):
                info = first._replace(
                    residual_norm=current_norm, relative_update=relative_update,
                    converged=exact, admissible=projected_valid,
                )
                return projected, info, accepted, InexactWork(
                    first.iterations, (~exact).astype(jnp.int32),
                    exact.astype(jnp.int32), jnp.asarray(0, jnp.int32),
                    boundary.iterations, jnp.asarray(1, jnp.int32), boundary.rejected_trials,
                )

            def fallback(_):
                seed = jnp.where(projected_valid, projected.ravel(), point)
                result, last = newton(seed, old, dt, time, forcing)
                info = last._replace(
                    iterations=first.iterations + last.iterations,
                    rejected_trials=first.rejected_trials + last.rejected_trials,
                )
                valid = last.converged & last.admissible
                return result.reshape(shape), info, valid, InexactWork(
                    info.iterations, jnp.asarray(0, jnp.int32),
                    last.converged.astype(jnp.int32), jnp.asarray(1, jnp.int32),
                    boundary.iterations, jnp.asarray(1, jnp.int32), boundary.rejected_trials,
                )

            return jax.lax.cond(accepted, done, fallback, operand=None)

        self.implicit_smoother_step = implicit_smoother_step

        @jax.jit
        def recover_contents(contents, full_seed, time=0.0):
            cells, recovery = op.recover_storage(contents, full_seed[1:-1])
            provisional = full_seed.at[1:-1].set(cells)

            def complete(_):
                candidate, boundary = op.complete_boundaries(provisional, time)
                combined = _combine(recovery, boundary)
                valid = op.is_admissible(candidate)
                return candidate, combined._replace(
                    converged=combined.converged & valid,
                    admissible=combined.admissible & valid,
                )

            def failed(_):
                return full_seed, recovery._replace(converged=jnp.asarray(False))

            return jax.lax.cond(
                recovery.converged & recovery.admissible, complete, failed, operand=None
            )

        self.recover_contents = recover_contents

        @jax.jit
        def forced_norm(full, forcing):
            value = (
                (op.spatial_residual(full) - forcing)
                * self.reference_time
                / self.content_scale
            )
            weight = self.volumes / jnp.sum(self.volumes)
            return jnp.max(jnp.sqrt(jnp.sum(weight[:, None] * value**2, axis=0)))

        self.forced_norm = forced_norm

        if self.explicit:

            @jax.jit
            def explicit_step(full, forcing, dt, time=0.0):
                # Build the stage closure during tracing, not on each host cycle.
                # Physical time is fixed during steady pseudo-time smoothing.
                advance = make_explicit_step(
                    op.storage,
                    lambda state: op.spatial_residual(state) - forcing,
                    op.recover_storage,
                    lambda state, stage_time: op.complete_boundaries(state, time),
                    lambda state: op.balance_rate(state) + jnp.sum(forcing, axis=0),
                    op.is_admissible,
                    method="runge-kutta-3",
                    recover_initial=op.nconstraints > 0,
                )
                candidate, info, _ = advance(full, dt, time)
                return candidate, info

            self.forced_step = explicit_step
        else:
            self.forced_step = implicit_step

        self.density = jax.jit(
            lambda state: op.storage(state[1:-1]) / self.volumes[:, None]
        )


def _metric_success(metrics):
    return bool(
        metrics["finite"]
        and metrics["admissible"]
        and metrics["all_equations_passed"]
        and metrics["boundary_passed"]
        and metrics["constraints_passed"]
        and metrics["global_passed"]
        and metrics.get("invariants_passed", True)
    )


class FASWorkspace:
    """Reusable host V-cycle with dynamic transported-balance forcing.

    Models are ordered coarse to fine, with exact 2:1 nesting. ``mode`` is
    ``implicit`` or ``explicit``; the latter keeps the coarsest implicit and
    binds SSPRK3 explicit timestep policies on other levels. Every ``solve``
    resets states, CFLs, histories and counters while retaining compiled kernels.
    """

    DEFAULTS = dict(
        cfl=0.2,
        cfl_growth=1.5,
        cfl_shrink=0.5,
        cfl_max=1e6,
        max_cycles=300,
        max_wall_s=120.0,
        pre_sweeps=1,
        post_sweeps=1,
        coarse_steps=4,
        coarse_reduction=0.1,
        coarse_floor=1e-10,
        coarse_cfl_max=None,
        max_retries=10,
        correction_backtracks=12,
        correction_damping=1.0,
        min_damping=1.0 / 4096,
        transfer_tolerance=1e-9,
        correction_growth_limit=2.0,
        stagnation_cycles=20,
        jacobian_reuse="none",
        coarse_reconstruction=None,
        initialization="cold",
        fmg_cycles=2,
        fmg_carry_cfl=True,
        fmg_coarse_tolerance=None,
        fmg_min_cfl=0.0,
        fmg_prolongation="constant",
        fmg_inexact_initialization=False,
        smoother="exact",
        smoother_newton_iterations=2,
        smoother_residual_ratio=0.1,
        smoother_levels="coarse",
        finest_early_stop=False,
    )

    def __init__(
        self, models, scheme, residual_policy=None, settings=None, mode="implicit",
        coarse_jacobian_factory=None,
        level_cache=None,
    ):
        if mode not in ("implicit", "explicit"):
            raise ValueError("mode must be implicit or explicit")
        self.settings = dict(self.DEFAULTS)
        self.settings.update(settings or {})
        unknown = set(self.settings) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FAS settings: {sorted(unknown)}")
        if self.settings["jacobian_reuse"] not in ("none", "step"):
            raise ValueError("jacobian_reuse must be none or step")
        if self.settings["coarse_reconstruction"] not in (None, "constant"):
            raise ValueError("coarse_reconstruction must be None or constant")
        if self.settings["initialization"] not in ("cold", "fmg"):
            raise ValueError("initialization must be cold or fmg")
        if int(self.settings["fmg_cycles"]) < 0:
            raise ValueError("fmg_cycles must be nonnegative")
        if self.settings["coarse_cfl_max"] is not None and (
            not np.isfinite(self.settings["coarse_cfl_max"])
            or self.settings["coarse_cfl_max"] <= 0
        ):
            raise ValueError("coarse_cfl_max must be positive and finite")
        if self.settings["fmg_prolongation"] not in ("constant", "linear"):
            raise ValueError("fmg_prolongation must be constant or linear")
        if self.settings["smoother"] not in ("exact", "inexact"):
            raise ValueError("smoother must be exact or inexact")
        if self.settings["smoother_levels"] not in ("coarse", "all"):
            raise ValueError("smoother_levels must be coarse or all")
        if self.settings["smoother_newton_iterations"] not in (1, 2):
            raise ValueError("smoother_newton_iterations must be 1 or 2")
        if not 0 < float(self.settings["smoother_residual_ratio"]) < 1:
            raise ValueError("smoother_residual_ratio must lie between zero and one")
        coarse_tolerance = self.settings["fmg_coarse_tolerance"]
        if coarse_tolerance is not None and (
            not np.isfinite(coarse_tolerance) or float(coarse_tolerance) <= 0
        ):
            raise ValueError("fmg_coarse_tolerance must be positive and finite")
        if not np.isfinite(self.settings["fmg_min_cfl"]) or self.settings["fmg_min_cfl"] < 0:
            raise ValueError("fmg_min_cfl must be nonnegative and finite")
        if self.settings["smoother"] == "inexact" and (
            mode != "implicit" or coarse_jacobian_factory is not None
            or self.settings["jacobian_reuse"] != "none"
        ):
            raise ValueError("Inexact smoothing currently requires implicit fresh Jacobians")
        if self.settings["fmg_inexact_initialization"] and self.settings["smoother"] != "inexact":
            raise ValueError("fmg_inexact_initialization requires smoother='inexact'")
        if mode != "implicit" and coarse_jacobian_factory is not None:
            raise ValueError("Supplied coarse Jacobians require implicit mode")
        self.mode, self.scheme = mode, scheme
        self.residual_policy = residual_policy or ResidualPolicy(criterion="absolute")
        models = tuple(models)
        if len(models) < 2:
            raise ValueError("FAS needs at least two levels")
        for coarse, fine in zip(models, models[1:]):
            if (coarse.nvar, coarse.ntransport, coarse.nconstraints) != (
                fine.nvar,
                fine.ntransport,
                fine.nconstraints,
            ) or tuple(coarse.equation_names) != tuple(fine.equation_names):
                raise ValueError(
                    "All levels must retain variable and transport meanings"
                )
            if fine.mesh.n_cells != 2 * coarse.mesh.n_cells:
                raise ValueError("Use exactly 2:1 nested cell counts")
            if not np.allclose(
                coarse.mesh.faces, fine.mesh.faces[::2], rtol=0, atol=1e-14
            ):
                raise ValueError("Coarse faces must be alternate fine faces")
            if not np.allclose(
                coarse.mesh.volumes,
                np.asarray(fine.mesh.volumes).reshape(-1, 2).sum(1),
                rtol=5e-13,
                atol=1e-20,
            ):
                raise ValueError("Nested physical cell volumes must add")
            if not np.allclose(
                coarse.mesh.area_faces,
                fine.mesh.area_faces[::2],
                rtol=5e-13,
                atol=1e-20,
            ):
                raise ValueError("Shared face areas must agree")
        self.levels = []
        for i, model in enumerate(models):
            explicit = mode == "explicit" and i > 0
            level_scheme = replace(
                scheme,
                time_integrator="runge-kutta-3" if explicit else "euler-implicit",
                boundary_treatment="segregated" if explicit else "coupled",
                reconstruction=(
                    self.settings["coarse_reconstruction"]
                    if i < len(models) - 1
                    and self.settings["coarse_reconstruction"] is not None
                    else scheme.reconstruction
                ),
            )
            options = (
                explicit, self.settings["jacobian_reuse"],
                int(self.settings["smoother_newton_iterations"]),
                float(self.settings["smoother_residual_ratio"]),
                float(self.residual_policy.boundary_tolerance),
                float(self.residual_policy.constraint_tolerance),
            )
            key = (id(model), level_scheme, *options)
            if level_cache is None:
                level = Level(model, level_scheme, *options)
            else:
                if key not in level_cache:
                    level_cache[key] = Level(model, level_scheme, *options)
                level = level_cache[key]
            self.levels.append(level)
        self.aggregate_operators = (
            {
                i: coarse_jacobian_factory(self.levels[i + 1], self.levels[i])
                for i in range(len(self.levels) - 1)
            }
            if coarse_jacobian_factory is not None
            else {}
        )
        self.operator = self.levels[-1].operator
        self.samples = [
            sampler(level.operator, self.residual_policy) for level in self.levels
        ]
        self.sample = self.samples[-1]
        self._reset()

    def _reset(self):
        self.cfls = [float(self.settings["cfl"]) for _ in self.levels]
        self.work = dict(
            levels=[
                dict(
                    cells=level.model.mesh.n_cells,
                    accepted_steps=0,
                    rejected_steps=0,
                    newton_or_recovery_iterations=0,
                    implicit_newton_iterations=0,
                    recovery_iterations=0,
                    jacobian_builds=0,
                    frozen_attempts=0,
                    jacobian_refreshes=0,
                    aggregate_spatial_builds=0,
                    aggregate_fine_spatial_jacobian_builds=0,
                    aggregate_implicit_builds=0,
                    aggregate_jacobian_uses=0,
                    aggregate_spatial_s=0.0,
                    aggregate_implicit_s=0.0,
                    rejected_newton_trials=0,
                    implicit_steps=0,
                    explicit_steps=0,
                    recovery_calls=0,
                    inexact_attempts=0,
                    inexact_accepted=0,
                    inexact_exact_converged=0,
                    inexact_fallbacks=0,
                    inexact_boundary_rejected_trials=0,
                    inexact_constrained_exact_steps=0,
                    initial_prolongation_attempts=0,
                    initial_prolongation_rejected_trials=0,
                    initial_prolongation_constant_fallbacks=0,
                    coarse_visits=0,
                    corrections_attempted=0,
                    corrections_accepted=0,
                    corrections_rejected=0,
                    correction_backtracks=0,
                    transfer_failures=0,
                    forced_norm_calls=0,
                    coarse_caps=0,
                    forced_coarse_cap_steps=0,
                    max_effective_coarse_cfl=0.0,
                    cycle_visits=0,
                    smoother_s=0.0,
                    transfer_s=0.0,
                    correction_s=0.0,
                )
                for level in self.levels
            ],
            cycles=0,
            coarse_failures=0,
            stopped_by_wall_budget=False,
            initialization_history=[],
            initialization_transfers=[],
            early_stop_checks=0,
            early_converged_cycles=0,
        )
        self._aggregate_spatial = {}
        self._fine_cycle_context = False
        self._fine_initial_norm = None
        self._early_cycle_metrics = None
        self._started = perf_counter()

    def _expired(self):
        return perf_counter() - self._started > float(self.settings["max_wall_s"])

    def _norm(self, index, state, forcing):
        self.work["levels"][index]["forced_norm_calls"] += 1
        return float(self.levels[index].forced_norm(state, forcing))

    def _smooth(self, index, state, forcing, sweeps, *, exact=False, forced_coarse=False):
        level = self.levels[index]
        counts = self.work["levels"][index]
        started = perf_counter()
        okay = True
        for _ in range(int(sweeps)):
            if self._expired():
                okay = False
                break
            accepted = False
            cap = (
                min(float(self.settings["cfl_max"]), 0.45)
                if level.explicit
                else float(self.settings["cfl_max"])
            )
            cap_forced_coarse = (
                forced_coarse and self.settings["coarse_cfl_max"] is not None
            )
            if cap_forced_coarse:
                cap = min(cap, float(self.settings["coarse_cfl_max"]))
            for _retry in range(int(self.settings["max_retries"]) + 1):
                cfl = min(self.cfls[index], cap)
                if cap_forced_coarse:
                    counts["max_effective_coarse_cfl"] = max(
                        counts["max_effective_coarse_cfl"], cfl
                    )
                limit = jax.device_get(
                    level.operator.timestep(state, jnp.float64(cfl), jnp.float64(0.0))
                )
                dt = float(limit.dt)
                if not bool(limit.valid) or not np.isfinite(dt) or dt <= 0:
                    counts["rejected_steps"] += 1
                    self.cfls[index] *= float(self.settings["cfl_shrink"])
                    continue
                if level.explicit:
                    candidate, info = level.forced_step(
                        state, forcing, jnp.float64(dt), jnp.float64(0.0)
                    )
                else:
                    use_inexact = (
                        not exact and self.settings["smoother"] == "inexact"
                        and (self.settings["smoother_levels"] == "all" or index < len(self.levels) - 1)
                    )
                    if use_inexact and level.operator.nconstraints:
                        counts["inexact_constrained_exact_steps"] += 1
                        use_inexact = False
                    if use_inexact:
                        candidate, info, smoothing_accepted, inexact_work = level.implicit_smoother_step(
                            state, forcing, jnp.float64(dt), jnp.float64(0.0)
                        )
                        inexact_work = jax.device_get(inexact_work)
                        counts["inexact_attempts"] += 1
                        counts["inexact_accepted"] += int(inexact_work.accepted_inexact)
                        counts["inexact_exact_converged"] += int(inexact_work.exact_converged)
                        counts["inexact_fallbacks"] += int(inexact_work.fallbacks)
                        counts["jacobian_builds"] += int(inexact_work.jacobian_builds)
                        counts["recovery_calls"] += int(inexact_work.recovery_calls)
                        counts["recovery_iterations"] += int(inexact_work.recovery_iterations)
                        counts["newton_or_recovery_iterations"] += int(inexact_work.recovery_iterations)
                        counts["inexact_boundary_rejected_trials"] += int(inexact_work.recovery_rejected_trials)
                    elif index in self._aggregate_spatial:
                        assembly_started = perf_counter()
                        blocks = self.aggregate_operators[index].implicit_blocks(
                            state, jnp.float64(dt), jnp.float64(0.0),
                            self._aggregate_spatial[index],
                        )
                        jax.block_until_ready(blocks)
                        counts["aggregate_implicit_builds"] += 1
                        counts["aggregate_implicit_s"] += (
                            perf_counter() - assembly_started
                        )
                        candidate, info, step_work = level.implicit_step_frozen(
                            state, forcing, jnp.float64(dt), blocks, jnp.float64(0.0)
                        )
                    else:
                        candidate, info, step_work = level.implicit_step_with_work(
                            state, forcing, jnp.float64(dt), jnp.float64(0.0)
                        )
                    if not use_inexact:
                        step_work = jax.device_get(step_work)
                        counts["jacobian_builds"] += int(step_work.jacobian_builds)
                        counts["frozen_attempts"] += int(step_work.frozen_attempts)
                        counts["jacobian_refreshes"] += int(step_work.refreshes)
                        counts["aggregate_jacobian_uses"] += int(step_work.supplied_jacobian_uses)
                if not level.explicit and use_inexact:
                    info, smoothing_accepted = jax.device_get((info, smoothing_accepted))
                else:
                    info = jax.device_get(info)
                    smoothing_accepted = bool(info.converged and info.admissible)
                counts["newton_or_recovery_iterations"] += int(info.iterations)
                counts[
                    "recovery_iterations" if level.explicit
                    else "implicit_newton_iterations"
                ] += int(info.iterations)
                counts["rejected_newton_trials"] += int(info.rejected_trials)
                if bool((info.converged and info.admissible) if level.explicit else smoothing_accepted):
                    accepted = bool(level.operator.is_admissible(candidate))
                if accepted:
                    state = candidate
                    counts["accepted_steps"] += 1
                    if cap_forced_coarse:
                        counts["forced_coarse_cap_steps"] += 1
                    counts[
                        "explicit_steps" if level.explicit else "implicit_steps"
                    ] += 1
                    self.cfls[index] = min(
                        cap, cfl * float(self.settings["cfl_growth"])
                    )
                    break
                counts["rejected_steps"] += 1
                self.cfls[index] = cfl * float(self.settings["cfl_shrink"])
            if not accepted:
                okay = False
                break
        counts["smoother_s"] += perf_counter() - started
        return state, okay

    def _restrict(self, index, state):
        """Recover the coarse restriction and return its actual storage base."""
        fine, coarse = self.levels[index], self.levels[index - 1]
        counts = self.work["levels"][index - 1]
        started = perf_counter()
        target = restrict_contents(fine.operator.storage(state[1:-1]))
        weights = fine.volumes.reshape(-1, 2)
        seed_cells = (
            state[1:-1].reshape(-1, 2, fine.model.nvar) * weights[:, :, None]
        ).sum(1) / coarse.volumes[:, None]
        seed = jnp.concatenate((state[:1], seed_cells, state[-1:]), axis=0)
        candidate, info = coarse.recover_contents(target, seed, jnp.float64(0.0))
        info = jax.device_get(info)
        counts["recovery_calls"] += 1
        counts["newton_or_recovery_iterations"] += int(info.iterations)
        counts["recovery_iterations"] += int(info.iterations)
        valid = bool(info.converged and info.admissible)
        if valid:
            discrepancy = np.asarray(
                (coarse.operator.storage(candidate[1:-1]) - target)
                / coarse.content_scale
            )
            valid = bool(
                np.isfinite(discrepancy).all()
                and np.max(np.abs(discrepancy)) <= self.settings["transfer_tolerance"]
            )
        if not valid:
            counts["transfer_failures"] += 1
        counts["transfer_s"] += perf_counter() - started
        return candidate, valid

    def _correct(self, index, state, forcing, density_correction):
        level = self.levels[index]
        counts = self.work["levels"][index]
        started = perf_counter()
        counts["corrections_attempted"] += 1
        base_contents = level.operator.storage(state[1:-1])
        delta = level.volumes[:, None] * prolong_density(density_correction)
        before = self._norm(index, state, forcing)
        theta = float(self.settings["correction_damping"])
        accepted = False
        candidate = state
        for _ in range(int(self.settings["correction_backtracks"]) + 1):
            if theta < self.settings["min_damping"] or self._expired():
                break
            trial, info = level.recover_contents(
                base_contents + theta * delta, state, jnp.float64(0.0)
            )
            info = jax.device_get(info)
            counts["recovery_calls"] += 1
            counts["newton_or_recovery_iterations"] += int(info.iterations)
            counts["recovery_iterations"] += int(info.iterations)
            accepted = bool(info.converged and info.admissible)
            if accepted:
                after = self._norm(index, trial, forcing)
                accepted = bool(
                    np.isfinite(after)
                    and after
                    <= float(self.settings["correction_growth_limit"])
                    * max(before, 1e-14)
                )
            if accepted:
                candidate = trial
                counts["corrections_accepted"] += 1
                break
            theta *= 0.5
            counts["correction_backtracks"] += 1
        if not accepted:
            counts["corrections_rejected"] += 1
        counts["correction_s"] += perf_counter() - started
        return candidate

    def _cycle(self, index, state, forcing, root_index=None):
        # The active original problem is the root of this visit. Only its
        # recursive, forced descendants receive the optional coarse CFL cap.
        root_index = index if root_index is None else root_index
        forced_coarse = index < root_index
        self.work["levels"][index]["cycle_visits"] += 1
        if self._expired():
            return state, False
        if index == 0:
            counts = self.work["levels"][0]
            counts["coarse_visits"] += 1
            initial = self._norm(0, state, forcing)
            target = max(
                float(self.settings["coarse_floor"]),
                initial * float(self.settings["coarse_reduction"]),
            )
            if initial <= float(self.settings["coarse_floor"]):
                return state, True
            okay = True
            for _ in range(int(self.settings["coarse_steps"])):
                state, okay = self._smooth(0, state, forcing, 1, forced_coarse=forced_coarse)
                if not okay or self._norm(0, state, forcing) <= target:
                    return state, okay
            counts["coarse_caps"] += 1
            return state, okay
        state, okay = self._smooth(
            index, state, forcing, self.settings["pre_sweeps"],
            forced_coarse=forced_coarse,
        )
        if not okay:
            return state, False
        if (
            self.settings["finest_early_stop"] and self._fine_cycle_context
            and int(self.settings["pre_sweeps"]) > 0
            and index == len(self.levels) - 1
        ):
            self.work["early_stop_checks"] += 1
            metrics = self.sample(state, self._fine_initial_norm)
            if _metric_success(metrics):
                self.work["early_converged_cycles"] += 1
                self._early_cycle_metrics = metrics
                return state, True
        coarse_state, transferred = self._restrict(index, state)
        if transferred:
            fine, coarse = self.levels[index], self.levels[index - 1]
            coarse_base = coarse.density(coarse_state)
            defect = forcing - fine.operator.spatial_residual(state)
            coarse_forcing = coarse.operator.spatial_residual(
                coarse_state
            ) + restrict_integrated(defect)
            if index - 1 in self.aggregate_operators:
                assembly_started = perf_counter()
                aggregate = self.aggregate_operators[index - 1]
                if index in self._aggregate_spatial:
                    spatial = aggregate.aggregate_blocks(
                        state, coarse_state, self._aggregate_spatial[index]
                    )
                else:
                    spatial = aggregate.spatial_blocks(state, coarse_state)
                    self.work["levels"][index][
                        "aggregate_fine_spatial_jacobian_builds"
                    ] += 1
                jax.block_until_ready(spatial)
                self._aggregate_spatial[index - 1] = spatial
                # Each projection is counted, but only the first descent builds
                # a new physical spatial Jacobian. Deeper levels reuse blocks
                # inherited from their parent; exact residuals remain unchanged.
                counts = self.work["levels"][index]
                counts["aggregate_spatial_builds"] += 1
                counts["aggregate_spatial_s"] += perf_counter() - assembly_started
            corrected, coarse_ok = self._cycle(
                index - 1, coarse_state, coarse_forcing, root_index=root_index
            )
            if coarse_ok:
                state = self._correct(
                    index, state, forcing, coarse.density(corrected) - coarse_base
                )
            else:
                self.work["coarse_failures"] += 1
        else:
            self.work["coarse_failures"] += 1
        # A failed coarse correction leaves the current valid fine iterate;
        # the skipped coarse work is retained in counters rather than hidden.
        return self._smooth(
            index, state, forcing, self.settings["post_sweeps"],
            forced_coarse=forced_coarse,
        )

    def _coarse_initial_success(self, metrics):
        """An explicitly auxiliary coarse target; original flags stay intact."""
        target = self.settings["fmg_coarse_tolerance"]
        if target is None:
            return _metric_success(metrics)
        return bool(
            metrics["finite"] and metrics["admissible"]
            and metrics["boundary_passed"] and metrics["constraints_passed"]
            and metrics.get("invariants_passed", True)
            and np.max(metrics["equation_norm"]) <= float(target)
            and np.max(metrics["global_defect"]) <= float(target)
        )

    def _initialization_record(self, index, state, initial_norm, stage, step, okay, metrics=None):
        metrics = self.samples[index](state, initial_norm) if metrics is None else metrics
        counts = self.work["levels"][index]
        self.work["initialization_history"].append(
            dict(
                level_index=index,
                cells=self.levels[index].model.mesh.n_cells,
                stage=stage,
                step=step,
                accepted=okay,
                elapsed_s=perf_counter() - self._started,
                iterations=sum(
                    row["implicit_newton_iterations"] for row in self.work["levels"]
                ),
                level_iterations=counts["implicit_newton_iterations"],
                smoother_steps=sum(
                    row["accepted_steps"] for row in self.work["levels"]
                ),
                level_smoother_steps=counts["accepted_steps"],
                cfl=self.cfls[index],
                auxiliary_target=(self.settings["fmg_coarse_tolerance"] if index == 0 else None),
                auxiliary_target_passed=(self._coarse_initial_success(metrics) if index == 0 else None),
                **metrics,
            )
        )
        return metrics

    def _prolong_state(self, index, coarse_state):
        """Conservatively prolong a physical state for nested initialization."""
        coarse, fine = self.levels[index - 1], self.levels[index]
        counts = self.work["levels"][index]
        started = perf_counter()
        coarse_density = coarse.density(coarse_state)
        seed = jnp.concatenate(
            (coarse_state[:1], prolong_density(coarse_state[1:-1]), coarse_state[-1:]),
            axis=0,
        )
        linear = self.settings["fmg_prolongation"] == "linear"
        if linear:
            from .transfers import prolong_initial_density

            damping = [
                0.5 ** attempt
                for attempt in range(int(self.settings["correction_backtracks"]) + 1)
                if 0.5 ** attempt >= float(self.settings["min_damping"])
            ] + [0.0]
        else:
            damping = [0.0]
        state, okay = seed, False
        for theta in damping:
            density = (
                prolong_initial_density(
                    coarse_density,
                    jnp.asarray(coarse.model.mesh.centers),
                    jnp.asarray(fine.model.mesh.centers),
                    fine.volumes, theta,
                )
                if linear and theta > 0 else prolong_density(coarse_density)
            )
            target = fine.volumes[:, None] * density
            state, info = fine.recover_contents(target, seed, jnp.float64(0.0))
            info = jax.device_get(info)
            counts["initial_prolongation_attempts"] += 1
            counts["recovery_calls"] += 1
            counts["newton_or_recovery_iterations"] += int(info.iterations)
            counts["recovery_iterations"] += int(info.iterations)
            okay = bool(info.converged and info.admissible)
            error = None
            if okay:
                discrepancy = np.asarray(
                    (fine.operator.storage(state[1:-1]) - target) / fine.content_scale
                )
                error = float(np.max(np.abs(discrepancy)))
                okay = bool(
                    np.isfinite(discrepancy).all()
                    and error <= self.settings["transfer_tolerance"]
                )
            self.work["initialization_transfers"].append(dict(
                cells=fine.model.mesh.n_cells, method=self.settings["fmg_prolongation"],
                theta=theta, accepted=okay, recovery_iterations=int(info.iterations),
                recovery_status=int(info.status), scaled_content_error=error,
            ))
            if okay:
                if linear and theta == 0:
                    counts["initial_prolongation_constant_fallbacks"] += 1
                break
            counts["initial_prolongation_rejected_trials"] += 1
        if not okay:
            counts["transfer_failures"] += 1
        counts["transfer_s"] += perf_counter() - started
        return state, okay

    def _initialize_fmg(self, fine_initial, fine_initial_norm):
        """Solve the original coarse problem, then ascend with actual V-cycles.

        A failed stage stops initialization and is reported. It does not launch
        an uncharged cold fine-grid fallback. Every stage uses original spatial,
        boundary, constraint and global residual sampling.
        """
        state = jnp.asarray(self.levels[0].model.initial_state(), dtype=jnp.float64)
        if not bool(self.levels[0].operator.is_admissible(state)):
            return fine_initial, False, "fmg_invalid_coarse_initial_state", None
        current = self.samples[0](state)
        initial_norm = np.asarray(current["equation_norm"])
        self._initialization_record(0, state, initial_norm, "initial", 0, True, metrics=current)
        forcing = jnp.zeros(
            (self.levels[0].model.mesh.n_cells, self.levels[0].operator.ntransport),
            dtype=state.dtype,
        )
        okay = True
        for step in range(1, int(self.settings["max_cycles"]) + 1):
            if self._coarse_initial_success(current):
                break
            state, okay = self._smooth(
                0, state, forcing, 1,
                exact=not self.settings["fmg_inexact_initialization"],
            )
            current = self._initialization_record(
                0, state, initial_norm, "coarse_solve", step, okay
            )
            if not okay or self._expired():
                break
        if not okay or not self._coarse_initial_success(current):
            reason = "wall_budget" if self._expired() else "fmg_coarse_failed"
            return fine_initial, False, reason, None
        for index in range(1, len(self.levels)):
            if self._expired():
                return fine_initial, False, "wall_budget", None
            state, okay = self._prolong_state(index, state)
            if self.settings["fmg_carry_cfl"]:
                self.cfls[index] = min(
                    max(self.cfls[index - 1], float(self.settings["fmg_min_cfl"])),
                    float(self.settings["cfl_max"]),
                )
            current = self.samples[index](
                state, fine_initial_norm if index == len(self.levels) - 1 else None
            )
            initial_norm = (
                fine_initial_norm if index == len(self.levels) - 1
                else np.asarray(current["equation_norm"])
            )
            self._initialization_record(
                index, state, initial_norm, "prolongation", 0, okay, metrics=current
            )
            if not okay:
                return fine_initial, False, "fmg_prolongation_failed", None
            # The finest level continues to the common original fine target in
            # solve(); intermediate levels receive a fixed number of V-cycles.
            if index == len(self.levels) - 1:
                break
            forcing = jnp.zeros(
                (self.levels[index].model.mesh.n_cells,
                 self.levels[index].operator.ntransport), dtype=state.dtype,
            )
            for step in range(1, int(self.settings["fmg_cycles"]) + 1):
                state, okay = self._cycle(index, state, forcing)
                self._initialization_record(
                    index, state, initial_norm, "fmg_cycle", step, okay
                )
                if not okay or self._expired():
                    reason = "wall_budget" if self._expired() else "fmg_smoother_failed"
                    return fine_initial, False, reason, None
        return state, True, "initialized", current

    def solve(self, initial_state=None):
        self._reset()
        level = self.levels[-1]
        initial = (
            level.model.initial_state() if initial_state is None else initial_state
        )
        state = jnp.asarray(initial, dtype=jnp.float64)
        if state.shape != self.operator.shape:
            raise ValueError(f"Expected initial state shape {self.operator.shape}")
        if not bool(self.operator.is_admissible(state)):
            result = Solution(
                np.asarray(state),
                False,
                "invalid_initial_state",
                [],
                {},
                operator=self.operator,
            )
            result.work = self.work
            result.initialization_history = self.work["initialization_history"]
            return result
        current = self.sample(state)
        initial_norm = np.asarray(current["equation_norm"])
        history = [
            dict(
                step=0,
                phase="initial",
                accepted=True,
                elapsed_s=perf_counter() - self._started,
                **current,
            )
        ]
        initialized = True
        initialization_reason = None
        if self.settings["initialization"] == "fmg" and not _metric_success(current):
            state, initialized, initialization_reason, initialized_metrics = self._initialize_fmg(
                state, initial_norm
            )
            if initialized_metrics is not None:
                current = initialized_metrics
            history.append(
                dict(
                    step=0,
                    phase="fmg_initialization",
                    accepted=initialized,
                    elapsed_s=perf_counter() - self._started,
                    **current,
                )
            )
        forcing = jnp.zeros(
            (level.model.mesh.n_cells, self.operator.ntransport), dtype=state.dtype
        )
        reason = "max_cycles" if initialized else initialization_reason
        unchanged = 0
        previous = np.asarray(state)
        for cycle in range(1, int(self.settings["max_cycles"]) + 1):
            if not initialized:
                break
            if _metric_success(current):
                reason = "converged"
                break
            self._fine_cycle_context = True
            self._fine_initial_norm = initial_norm
            self._early_cycle_metrics = None
            try:
                state, okay = self._cycle(len(self.levels) - 1, state, forcing)
            finally:
                self._fine_cycle_context = False
            self.work["cycles"] += 1
            current = (
                self._early_cycle_metrics if self._early_cycle_metrics is not None
                else self.sample(state, initial_norm)
            )
            snapshot = np.asarray(state)
            unchanged = unchanged + 1 if np.array_equal(snapshot, previous) else 0
            previous = snapshot
            history.append(
                dict(
                    step=cycle,
                    phase="accepted" if okay else "rejected",
                    accepted=okay,
                    kind="fas_cycle",
                    elapsed_s=perf_counter() - self._started,
                    cfl=self.cfls[-1],
                    coarse_failures=self.work["coarse_failures"],
                    **current,
                )
            )
            if self._expired():
                self.work["stopped_by_wall_budget"] = True
                reason = "wall_budget"
                break
            if not okay:
                reason = "smoother_failed"
                break
            if unchanged >= int(self.settings["stagnation_cycles"]):
                reason = "stagnation"
                break
        final = self.sample(state, initial_norm)
        success = initialized and _metric_success(final)
        if self._expired():
            self.work["stopped_by_wall_budget"] = True
        audit = getattr(level.model, "solution_audit", None)
        if success and callable(audit) and not bool(audit(state)):
            success, reason = False, "solution_audit_failed"
        elif success:
            reason = "converged"
        result = Solution(
            np.asarray(state),
            success,
            reason,
            history,
            final,
            elapsed_s=perf_counter() - self._started,
            operator=self.operator,
        )
        result.work = self.work
        result.cycle_history = history
        result.initialization_history = self.work["initialization_history"]
        return result
