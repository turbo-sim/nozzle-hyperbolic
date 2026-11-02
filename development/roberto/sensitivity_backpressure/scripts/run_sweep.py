"""Implicit nozzle backpressure sweep; source kernels and BCs remain unchanged.

Run from the repository root:
  poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep

Each operating point is a steady pseudo-time solve. Pressure is a dynamic
JAX argument to the unchanged source factory, allowing one compiled kernel
to be reused. The previous converged numerical solution seeds the next point.
This is parameter continuation, not a prescribed physical pressure ramp.
"""

import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import PerfectGasReference
from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
from nozzle_hyperbolic.models.euler_nozzle.fvm import make_jax_stadtke_step
from nozzle_hyperbolic.models.euler_nozzle.initialization import stadtke_puh_initialization_simple

STUDY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
RMS_TOL = 1e-8


class ImplicitNozzle:
    """Thin experimental controller around the production implicit kernel."""
    def __init__(self, n_points):
        self.fluid = jxp.FluidPerfectGas(name='Air')
        self.x = np.linspace(0, baseline.NOZZLE['L'], n_points)
        self.geo = get_symmetric_nozzle(self.x, **baseline.NOZZLE)
        self.area = np.asarray(self.geo.A)
        face = (self.area[:-1] + self.area[1:]) / 2
        self.dx = float(self.x[1] - self.x[0])
        gradient = np.asarray(self.geo.dAdx).copy()
        gradient[1:-1] = np.diff(face) / self.dx
        stag = self.fluid.get_state(jxp.PT_INPUTS, baseline.P_STAGNATION, baseline.T_STAGNATION)
        self.h0, self.s0, self.a0, self.rho0 = (float(stag[k]) for k in ('h', 's', 'a', 'rho'))
        self.settings = dict(baseline.SOLVER, flux_variant='regularized')
        s = self.settings

        def factory(pb):
            return make_jax_stadtke_step(
                self.fluid, self.area, face, gradient, self.geo.D_h, self.geo.roughness,
                self.dx, s['CFL'], pb, self.h0, self.s0,
                delta_factor=s['delta_factor'], newton_iter=s['newton_iter'],
                wall_friction=False, recon=s['reconstruction'], jacobian_mode=s['jacobian'],
                eps_filter=s['eps_filter'], linear_solver=s['linear_solver'],
                newton_tol=s['newton_tol'], flux_local_jac=s['flux_local_jac'],
                cell_local_jac=s['cell_local_jac'], flux_variant=s['flux_variant'],
                return_operators=True)

        self.factory = factory

        self.complete = jax.jit(lambda interior, pb: factory(pb)[4]['state_from_interior'](interior.ravel()))
        self.step = jax.jit(lambda full, dt, pb: factory(pb)[0](full, dt))
        # compute_dt does not depend on backpressure.
        self.compute_dt = factory(baseline.P_STAGNATION)[1]
        scales = (self.area[1:-1, None] * self.rho0
                  * np.array([1.0, self.a0, self.h0])[None, :] * self.a0)
        scales = jnp.asarray(scales)

        @jax.jit
        def metrics(full, pb):
            ops = factory(pb)[4]
            interior = full[1:-1].ravel()
            residual = ops['steady_residual'](interior).reshape(-1, 3)
            return jnp.sqrt(jnp.mean((residual / scales)**2)), ops['steady_residual_norm'](interior)
        self.metrics = metrics
        self.face_flux = jax.jit(lambda full, pb: factory(pb)[3](full))

    def cold_initial(self, pb):
        return stadtke_puh_initialization_simple(self.x, self.area, dict(
            fluid=self.fluid, fluid_name='Air', p_out=pb,
            pT_stagnation=np.array([baseline.P_STAGNATION, baseline.T_STAGNATION])))

    def integrate(self, initial, pb, max_updates):
        full = self.complete(jnp.asarray(initial)[1:-1], jnp.float64(pb))
        dt = float(self.compute_dt(full)[0])
        reference_ssr, ssr = None, np.inf
        history, trials = [], []
        start = perf_counter()
        reason = 'maximum updates'
        for n in range(max_updates + 1):
            rms, native = (float(v) for v in self.metrics(full, jnp.float64(pb)))
            history.append(dict(update=n, spatial_rms=rms, source_ssr=native,
                                elapsed_s=perf_counter() - start, dt_s=dt))
            # The fixed residual scale remains defined at u=0; native SSR does not.
            stagnant = bool(np.max(np.abs(np.asarray(full)[:, 1])) < 1e-12)
            if rms <= RMS_TOL and (native < RMS_TOL or stagnant):
                reason = 'original spatial residual converged'
                break
            if n == max_updates or not np.isfinite(np.asarray(full)).all():
                break
            cap, sound = self.compute_dt(full)
            cap = float(cap)
            if reference_ssr is not None and np.isfinite(ssr):
                cap *= np.clip((reference_ssr / max(ssr, 1e-300)) ** self.settings['ser_beta'],
                               1.0, self.settings['ser_cfl_max'] / self.settings['CFL'])
            dt = min(dt * self.settings['dt_growth_factor'], cap)
            old, sound = np.asarray(full), np.asarray(sound)
            accepted = False
            for retry in range(9):
                candidate, next_ssr = self.step(full, jnp.float64(dt), jnp.float64(pb))
                new = np.asarray(candidate)  # synchronize JAX
                increments = [np.max(np.abs(new[:, 0] - old[:, 0]) / (old[:, 0] + 1e-10)),
                              np.max(np.abs(new[:, 2] - old[:, 2]) / (np.abs(old[:, 2]) + 1e-10)),
                              np.max(np.abs(new[:, 1] - old[:, 1]) / (np.abs(old[:, 1]) + sound + 1e-10))]
                accepted = bool(np.isfinite(new).all() and np.all(new[:, [0, 2]] > 0)
                                and all(v < limit for v, limit in zip(increments, (0.2, 0.1, 0.2))))
                trials.append(dict(update=n + 1, retry=retry, dt_s=dt, accepted=accepted,
                                   dp=increments[0], dh=increments[1], du=increments[2]))
                if accepted:
                    break
                if retry < 8:
                    dt *= 0.5
            if not accepted:
                reason = 'all nine implicit trials rejected'
                break
            full, ssr = candidate, float(next_ssr)
            if reference_ssr is None:
                reference_ssr = ssr
            if (n + 1) % 500 == 0:
                print(f'    {n+1} updates, RMS={rms:.2e}, SSR={native:.2e}', flush=True)
        return np.asarray(full), dict(history=history, trials=trials, reason=reason,
                                    solve_s=perf_counter() - start,
                                    converged=reason == 'original spatial residual converged')


def pressure_schedule(ref):
    """Dense around physical transitions; broad coverage elsewhere."""
    values = [1.0, .9999, .999, .997, .99, .98, .97, .96, .95, .945,
              .94, .939, .938, .9375, .937, .936, .935, .93, .925, .92,
              .91, .90, .88, .86, .84, .82, .80, .77, .74, .71, .68,
              .65, .62, .60, .58, .56, .54, .52, .50, .48, .45, .40,
              .35, .30, .25, .20, .15, .12, .10, .08, .06, .04, .02]
    # Refinement prompted by the first sweep: delayed numerical choking and
    # a captured shock persisting below the analytical shock-at-exit pressure.
    values.extend([.9348, .9346, .9344, .9342, .9340, .9338, .9335, .9330, .9325, .9320,
                   .445, .440, .435, .430, .425, .420, .415, .410, .405])
    for ratio in (ref.critical, ref.shock_exit, ref.adapted):
        values.extend([ratio + .002, ratio + .0005, ratio, ratio - .0005, ratio - .002])
    return sorted(set(round(v, 12) for v in values if 0 < v <= 1), reverse=True)


def summarize(model, ref, full, ratio, exact):
    state = model.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
    props = {k: np.asarray(state[k]) for k in ('rho', 'a', 'T', 's')}
    mach = full[:, 1] / props['a']
    mass = np.asarray(model.face_flux(jnp.asarray(full), jnp.float64(ratio * ref.p0)))
    mdot = float(np.mean(mass))
    spread = float(np.ptp(mass) / max(abs(mdot), ref.mdot_star * 1e-12))
    entropy_rise = float(props['s'][-2] - props['s'][1])
    pressure = full[:, 0]
    faces = np.flatnonzero((model.x[:-1] >= ref.length / 2) & (model.x[1:] < ref.length))
    k = faces[np.argmax(np.diff(pressure)[faces])]
    compression = float((pressure[k+1] - pressure[k]) / ref.p0)
    shock_est = float((model.x[k] + model.x[k+1]) / 2) if compression > .005 else None
    exact_mdot = float(np.mean(exact['mdot']))
    return dict(
        p_back_ratio=ratio, p_back_Pa=ratio * ref.p0, regime_exact=exact['regime'],
        mdot_kg_s=mdot, mdot_exact_kg_s=exact_mdot,
        mdot_relative_error=(mdot - exact_mdot) / max(exact_mdot, ref.mdot_star * 1e-12),
        face_mass_spread_relative=spread,
        entropy_rise_J_kg_K=entropy_rise, entropy_generation_W_K=mdot * entropy_rise,
        entropy_generation_exact_W_K=exact_mdot * float(exact['ds'][-2] - exact['ds'][1]),
        shock_entropy_generation_including_exit_W_K=exact_mdot * exact['shock_ds'],
        exit_T_K=float(props['T'][-2]), exit_T_exact_K=float(exact['T'][-2]),
        exit_p_ratio=float(pressure[-2] / ref.p0), exit_p_exact_ratio=float(exact['p'][-2] / ref.p0),
        ghost_p_ratio=float(pressure[-1] / ref.p0), exit_M=float(mach[-2]),
        throat_M=float(mach[np.argmin(model.area)]), max_M=float(np.max(mach[1:-1])),
        outlet_pressure_weight=float(.5 * (1 + np.tanh((1 - abs(mach[-2])) / .004))),
        shock_exact_m=exact['shock_x'], shock_estimate_m=shock_est,
        pressure_L1_p0=float(np.mean(np.abs(pressure[1:-1] - exact['p'][1:-1])) / ref.p0),
        mach_L1=float(np.mean(np.abs(mach[1:-1] - exact['M'][1:-1]))),
        total_enthalpy_error_relative=float(np.max(np.abs(full[1:-1, 2] + full[1:-1, 1]**2 / 2 - model.h0)) / abs(model.h0)),
        inlet_entropy_error_J_kg_K=float(props['s'][0] - model.s0),
        flow_state=dict(p=pressure.tolist(), T=props['T'].tolist(), M=mach.tolist(),
                        ds=(props['s'] - model.s0).tolist(), mdot_face=mass.tolist()))


def hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'src').rglob('*.py')}


def save(report, destination):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    (destination / 'summary.json').write_text(json.dumps(clean(report), indent=2), encoding='utf-8')
    rows = [{k: v for k, v in row.items() if k not in {'flow_state', 'history', 'trials'}} for row in report['runs']]
    if rows:
        with (destination / 'operating_map.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'outputs' / 'baseline')
    parser.add_argument('--max-updates', type=int, default=5000)
    parser.add_argument('--n-points', type=int, default=baseline.N_POINTS)
    parser.add_argument('--smoke', action='store_true', help='Six representative pressures, same source kernel')
    parser.add_argument('--no-cold-checks', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        from development.roberto.historical.backpressure_sensitivity.scripts.plot_results import render
        render(json.loads((args.output / 'summary.json').read_text(encoding='utf-8')), args.output)
        return
    model = ImplicitNozzle(args.n_points)
    ref = PerfectGasReference(model.fluid)
    schedule = ([1.0, .97, ref.critical, .8, ref.shock_exit, .04] if args.smoke else pressure_schedule(ref))
    # Equal reservoir/backpressure gives the trivial stagnant field. No analytical
    # nozzle profile or shock position is ever supplied as a numerical seed.
    seed = np.tile([ref.p0, 0.0, model.h0], (args.n_points, 1))
    report = dict(landmarks=ref.landmarks(), settings=model.settings,
                  mesh_points=args.n_points, x_m=model.x.tolist(), area_m2=model.area.tolist(),
                  p0_Pa=ref.p0, T0_K=ref.T0, source_before=hashes(), runs=[],
                  protocol='descending numerical continuation, independent simple starts, and supersonic-restart branch probes',
                  versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxprop', 'numpy', 'scipy')})
    print('Physical thresholds:', report['landmarks'], flush=True)
    start = perf_counter()
    model.step(jnp.asarray(model.cold_initial(.97 * ref.p0)), jnp.float64(1e-5), jnp.float64(.97 * ref.p0))[0].block_until_ready()
    report['step_first_call_including_compile_s'] = perf_counter() - start
    # Guard the dynamic-backpressure wrapper against changing the source step.
    check_initial = jnp.asarray(model.cold_initial(.97 * ref.p0))
    dynamic, _ = model.step(check_initial, jnp.float64(1e-5), jnp.float64(.97 * ref.p0))
    static, _ = model.factory(.97 * ref.p0)[0](check_initial, jnp.float64(1e-5))
    parity = float(np.max(np.abs(np.asarray(dynamic) - np.asarray(static))
                          / np.array([ref.p0, model.a0, model.h0])[None, :]))
    report['dynamic_vs_static_source_step_scaled_max_error'] = parity
    assert parity < 1e-11, 'Dynamic wrapper changed the source implicit update'
    source_ratio = 1.0
    groups = [('descending', schedule)]
    if not args.no_cold_checks and not args.smoke:
        groups.append(('cold', [.97, ref.critical, .90, .70, ref.shock_exit + .002,
                                ref.shock_exit, ref.shock_exit - .002, .48, .45, .43, .42,
                                .40, ref.adapted, .04]))
        # Deliberately probe whether the local Mach-based outlet can admit the
        # already-supersonic branch at a higher imposed pressure. This is not a
        # physical exterior-domain pressure-ramp simulation.
        groups.append(('supersonic_restart', [.48, .70, .95]))
    for mode, ratios in groups:
        for index, ratio in enumerate(ratios):
            exact = ref.profile(model.x, ratio)
            pb = ratio * ref.p0
            initial = model.cold_initial(pb) if mode == 'cold' else seed
            print(f'{mode} {index+1}/{len(ratios)} pb/p0={ratio:.8f} [{exact["regime"]}]', flush=True)
            full, result = model.integrate(initial, pb, args.max_updates)
            row = dict(mode=mode, seed_back_ratio=source_ratio if mode != 'cold' else None,
                       **summarize(model, ref, full, ratio, exact), **result)
            row.update(accepted_updates=len(result['history']) - 1,
                       rejected_trials=sum(not t['accepted'] for t in result['trials']))
            report['runs'].append(row)
            if mode == 'descending' and result['converged']:
                seed, source_ratio = full.copy(), ratio
            print(f'  converged={row["converged"]} updates={row["accepted_updates"]} '
                  f'RMS={row["history"][-1]["spatial_rms"]:.2e} M_exit={row["exit_M"]:.4f} '
                  f'mdot={row["mdot_kg_s"]:.6f}', flush=True)
            save(report, args.output)
    report['source_after'] = hashes()
    report['source_unchanged'] = report['source_before'] == report['source_after']
    assert report['source_unchanged'], 'Production source changed during the study'
    save(report, args.output)
    from development.roberto.historical.backpressure_sensitivity.scripts.plot_results import render
    render(report, args.output)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
