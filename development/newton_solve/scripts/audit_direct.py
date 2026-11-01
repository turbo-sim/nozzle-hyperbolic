"""Executable adapter audit, with no nozzle target solve or reference seeding.

Run from the repository root with ``python -B -m
development.newton_solve.scripts.audit_direct``. The linear model checks
the actual custom backend and callback stopping. Euler checks only residuals,
Jacobians, scaling and initial rank; they do not advance a flow solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from nozzle_hyperbolic.verification import build_case

from .direct import DirectProblem, _metrics_from_rows, solve_direct, state_hash


def linear_check():
    """A known square root verifies exact first state, fractions and stopping."""
    volumes = np.array([1.0, 2.0])
    model = SimpleNamespace(variable_scales=np.array([2.0]),
                            conservative_scales=np.array([3.0]),
                            global_flux_scales=np.array([7.0]), reference_time=5.0)
    roots = jnp.array([[12.0], [13.0], [14.0], [15.0]])
    matrix = jnp.array([[2., 0., 0., 0.], [-.2, 2., -.1, 0.],
                        [0., -.1, 2., -.2], [0., 0., 0., 2.]])

    def steady(flat, time=0.0):
        return matrix @ (flat-roots.ravel())

    def spatial(full):
        scaled = steady(full.ravel()).reshape((4, 1))[1:-1]
        return scaled*jnp.asarray(volumes)[:, None]*3./5.

    def boundary(full, time=0.0):
        return steady(full.ravel()).reshape((4, 1))[jnp.array([0, 3]), :]

    operator = SimpleNamespace(
        shape=(4, 1), model=model,
        mesh=SimpleNamespace(volumes=volumes),
        steady_residual=jax.jit(steady), spatial_residual=jax.jit(spatial),
        boundary_residual=jax.jit(boundary),
        is_admissible=jax.jit(lambda full: jnp.all(jnp.isfinite(full) & (full > 0))),
    )
    initial = np.full((4, 1), 4.0)
    result = solve_direct(SimpleNamespace(operator=operator), initial, ResidualPolicy())
    assert result["success"], result["native_message"]
    assert result["native_status"] == 5 and not result["native_success"]
    assert result["reason"] == "common_target_reached"
    assert np.max(np.abs(result["full_state"]-np.asarray(roots))) < 1e-10
    assert result["initial_state_sha256"] == result["actual_initial_state_sha256"] == state_hash(initial)
    fractions = [row["step_fraction"] for row in result["history"]][1:]
    assert np.allclose(fractions, [.25, .375, .5625, .84375, 1.0]), fractions
    assert result["counters"]["ntrials"] == len(fractions)
    limited = solve_direct(SimpleNamespace(operator=operator), initial, ResidualPolicy(),
                           options={"max_accepted_steps": 2})
    assert not limited["success"] and not limited["common_success"]
    assert limited["native_status"] == 5 and not limited["native_success"]
    assert limited["reason"] == "newton_step_limit_2"
    assert limited["counters"]["nit"] == 2 and len(limited["history"]) == 3
    assert limited["options"]["max_accepted_steps"] == limited["limits"]["max_accepted_steps"] == 2
    expected_limited = initial+(1-.75*.625)*(np.asarray(roots)-initial)
    assert np.allclose(limited["full_state"], expected_limited, rtol=0., atol=1e-13)
    at_cap = solve_direct(SimpleNamespace(operator=operator), initial, ResidualPolicy(),
                         options={"max_accepted_steps": 5})
    assert at_cap["success"] and at_cap["reason"] == "common_target_reached"
    assert at_cap["counters"]["nit"] == 5
    return dict(passed=True, native_status=result["native_status"],
                reason=result["reason"], accepted_fractions=fractions,
                ntrials=result["counters"]["ntrials"],
                actual_backend=result["backend"], effective_options=result["options"],
                accepted_step_cap=dict(passed=True, failed_cap=2,
                                       failure_reason=limited["reason"],
                                       accepted_updates=limited["counters"]["nit"],
                                       root_at_cap_allowed=bool(at_cap["success"]), root_cap=5))


def euler_checks(cells=8, reconstructions=("constant",), outlets=("acoustic",)):
    """Measure actual Euler Jacobians, without assuming stagnation is singular."""
    rows = []
    policy = ResidualPolicy()
    for flux in ("rusanov", "hll", "stadtke"):
        for reconstruction in reconstructions:
            for outlet in outlets:
                physical = build_case("euler_nozzle/shock_35", cells)
                original = physical.model
                model = type(original)(
                    original.mesh, original.fluid, p0=original.p0, T0=original.T0,
                    back_pressure=original.back_pressure, inlet_regime="subsonic",
                    outlet_regime="subsonic", outlet_formulation=outlet,
                )
                scheme = Scheme(flux=flux, reconstruction=reconstruction,
                                boundary_treatment="coupled", newton_backtracking=True,
                                custom_jump_correction=True, custom_endpoint_envelope=True)
                operator = Operator(model, scheme)
                rest = np.zeros(operator.shape)
                rest[:, 0], rest[:, -1] = model.p0, model.h0
                problem = DirectProblem(operator, rest)
                compilation = problem.compile()
                residual = problem.residual(problem.x0)
                jacobian = problem.gradient(problem.x0)
                singular = np.linalg.svd(jacobian, compute_uv=False)
                cutoff = np.finfo(float).eps*max(jacobian.shape)*singular[0]
                rank = int(np.count_nonzero(singular > cutoff))
                host = _metrics_from_rows(residual, operator, policy)
                raw = np.asarray(operator.spatial_residual(jnp.asarray(rest)))
                expected_l2 = np.sqrt(np.sum(
                    (np.asarray(model.mesh.volumes)/np.sum(model.mesh.volumes))[:, None]
                    *(raw*model.reference_time/(np.asarray(model.mesh.volumes)[:, None]
                      *np.asarray(model.conservative_scales)[None, :]))**2, axis=0))
                assert np.allclose(host["equation_l2"], expected_l2, rtol=1e-11, atol=1e-12)
                # A deliberately smooth, admissible field is audit input only,
                # not an analytical or numerical start for any target solve.
                x = np.linspace(0., 1., cells+2)
                smooth = rest.copy()
                smooth[:, 0] *= .96-.14*x+.013*x*x
                smooth[:, 1] = model.a0*(.08+.06*x+.007*x*x)
                smooth[:, -1] -= model.a0**2*(.025+.02*x+.003*x*x)
                z = ((smooth-rest)/problem.scales).ravel()
                assert bool(operator.is_admissible(jnp.asarray(smooth)))
                rng = np.random.default_rng(217)
                direction = rng.normal(size=z.size)
                direction /= np.linalg.norm(direction)
                dense_product = problem.gradient(z)@direction
                jvp = np.asarray(problem._jvp(jnp.asarray(z), jnp.asarray(direction)))
                jvp_error = float(np.linalg.norm(dense_product-jvp)/max(np.linalg.norm(jvp), 1e-300))
                steps = [1e-4, 1e-5, 1e-6]
                fd_errors = []
                for step in steps:
                    fd = (problem.residual(z+step*direction)-problem.residual(z-step*direction))/(2*step)
                    fd_errors.append(float(np.linalg.norm(fd-dense_product)/max(np.linalg.norm(dense_product), 1e-300)))
                assert jvp_error < 1e-12, jvp_error
                assert min(fd_errors) < 2e-5, fd_errors
                # Rest is at |u|, reconstruction and branch switches. A symmetric
                # finite difference is diagnostic there, not an AD pass target.
                rest_direction = rng.normal(size=z.size)
                rest_direction /= np.linalg.norm(rest_direction)
                rest_ad = jacobian@rest_direction
                step = 1e-6
                rest_fd = (problem.residual(step*rest_direction)-problem.residual(-step*rest_direction))/(2*step)
                rest_fd_defect = float(np.linalg.norm(rest_fd-rest_ad)/max(np.linalg.norm(rest_ad), 1e-300))
                row = dict(flux=flux, reconstruction=reconstruction, outlet=outlet,
                           cells=cells, unknowns=jacobian.shape[0],
                           stagnant_admissible=True, stagnant_rank=rank,
                           rank_cutoff=float(cutoff), largest_singular_value=float(singular[0]),
                           smallest_singular_value=float(singular[-1]),
                           condition_number=float(singular[0]/singular[-1]) if singular[-1] else None,
                           stagnant_centered_fd_relative_defect=rest_fd_defect,
                           smooth_jvp_relative_error=jvp_error,
                           smooth_fd_steps=steps, smooth_fd_relative_errors=fd_errors,
                           compile_seconds=compilation,
                           interpretation="Rest directional mismatch is diagnostic at nonsmooth switches; smooth-state scaling and AD checks are asserted")
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--full", action="store_true", help="Both reconstructions and outlet laws")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    linear = linear_check()
    print(json.dumps(dict(linear=linear)), flush=True)
    rows = [] if args.linear_only else euler_checks(
        cells=args.cells,
        reconstructions=("constant", "muscl_van_leer") if args.full else ("constant",),
        outlets=("acoustic", "finite_wave") if args.full else ("acoustic",),
    )
    result = dict(linear=linear, euler=rows, all_checks_passed=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
