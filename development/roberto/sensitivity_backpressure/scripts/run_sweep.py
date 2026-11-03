"""Nozzle backpressure continuation using the current shared FVM solver.

Run from the repository root:
  poetry run python -m development.roberto.sensitivity_backpressure.scripts.run_sweep

Each operating point is a steady pseudo-time solve. Each pressure binds an immutable production model and reusable workspace.
The finite-volume mesh, strict Newton solve and automatic acoustic outlet
are the current production implementations. The previous converged numerical solution seeds the next point.
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

from .problem import ImplicitNozzle, N_POINTS, RMS_TOL, SCHEMA

STUDY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]


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
        outlet_direct_flux=bool(model.workspace(ratio * ref.p0).model.outlet.direct_flux(jnp.asarray(full[1:-1]))),
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
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        return value
    (destination / 'summary.json').write_text(json.dumps(clean(report), indent=2), encoding='utf-8')
    rows = [{k: v for k, v in row.items() if k not in {'flow_state', 'history', 'trials', 'metrics'}} for row in report['runs']]
    if rows:
        with (destination / 'operating_map.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'outputs' / 'shared_fvm')
    parser.add_argument('--max-updates', type=int, default=5000)
    parser.add_argument('--n-points', type=int, default=N_POINTS)
    parser.add_argument('--smoke', action='store_true', help='Six representative pressures using the shared FVM engine')
    parser.add_argument('--ratios', type=float, nargs='+', help='Explicit descending pressure/p0 values for a bounded sweep')
    parser.add_argument('--no-cold-checks', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.n_points < 4 or args.max_updates < 1:
        parser.error('Use at least four points and a positive update budget')
    if args.ratios is not None and (not all(0 < r <= 1 for r in args.ratios)
                                  or any(a <= b for a, b in zip(args.ratios, args.ratios[1:]))):
        parser.error('Pressure ratios must decrease strictly within (0, 1]')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only and (args.output / 'summary.json').exists():
        parser.error('Select a fresh --output directory for a new numerical campaign')
    if args.plot_only:
        from .plot_results import render
        render(json.loads((args.output / 'summary.json').read_text(encoding='utf-8')), args.output)
        return
    model = ImplicitNozzle(args.n_points)
    ref = model.reference
    schedule = ([1.0, .97, ref.critical, .8, ref.shock_exit, .04] if args.smoke else pressure_schedule(ref))
    if args.ratios is not None:
        schedule = args.ratios
    # Equal reservoir/backpressure gives the trivial stagnant field. No analytical
    # nozzle profile or shock position is ever supplied as a numerical seed.
    seed = np.tile([ref.p0, 0.0, model.h0], (args.n_points, 1))
    report = dict(landmarks=ref.landmarks(), settings=model.settings,
                  mesh_points=args.n_points, x_m=model.x.tolist(), area_m2=model.area.tolist(),
                  p0_Pa=ref.p0, T0_K=ref.T0, source_before=hashes(), runs=[],
                  protocol='descending numerical continuation, independent simple starts, and supersonic-restart branch probes',
                  versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxprop', 'numpy', 'scipy')})
    print('Physical thresholds:', report['landmarks'], flush=True)
    report['schema'] = SCHEMA
    report['step_first_call_including_compile_s'] = model.warm_up(
        model.cold_initial(.97 * ref.p0), .97 * ref.p0)
    source_ratio = 1.0
    groups = [('descending', schedule)]
    if not args.no_cold_checks and not args.smoke and args.ratios is None:
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
    from .plot_results import render
    render(report, args.output)
    print('Saved', args.output, flush=True)
    if not all(row['converged'] for row in report['runs']):
        raise SystemExit('Some solves failed; all requested attempts were retained')


if __name__ == '__main__':
    main()
