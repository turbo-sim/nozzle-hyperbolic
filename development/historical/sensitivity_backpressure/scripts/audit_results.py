"""Independent reference and saved-field residual audits, without new timesteps.

Run: poetry run python -m development.historical.sensitivity_backpressure.scripts.audit_results
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from .run_sweep import ImplicitNozzle, STUDY, hashes
from .problem import SCHEMA
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import subsonic_reference
from nozzle_hyperbolic.models.euler_nozzle.analytical.general_eos import nozzle_exact_real_gas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'outputs' / 'shared_fvm')
    args = parser.parse_args()
    report = json.loads((args.output / 'summary.json').read_text(encoding='utf-8'))
    if report.get('schema') != SCHEMA:
        raise ValueError('This audit evaluates current FVM states; use the saved report for pre-migration results')
    model = ImplicitNozzle(report['mesh_points'])
    ref = model.reference
    sub = subsonic_reference(model.area, model.fluid, .6,
                             throat_area=ref.At, p0=ref.p0, T0=ref.T0)
    ours = ref.profile(model.x, float(sub['p'][-1] / ref.p0))
    checks = dict(
        subsonic_pressure_max_relative_error=float(np.max(abs(ours['p'] - sub['p'])) / ref.p0),
        subsonic_mach_max_error=float(np.max(abs(ours['M'] - sub['M']))))
    shock_x = .5 * ref.length + .35 * (.5 * ref.length)
    # The independent general-EOS reference locates the throat by argmin(A)
    # and interpolates shock area. Include both exact geometry stations even
    # when a finite-volume center grid does not contain either one.
    reference_x = np.unique(np.r_[model.x, .5 * ref.length, shock_x])
    reference_area = model.mesh.geometry.sample(reference_x)['area']
    ours = ref.profile(reference_x, ref.shock_data(shock_x)[0])
    existing = nozzle_exact_real_gas(reference_x, reference_area, model.fluid, ref.p0, ref.T0,
                                     x_shock=shock_x, A_throat=ref.At)['shock']
    mask = np.abs(reference_x - shock_x) > 1e-8  # exclude the jump's sampling convention
    checks.update(
        shocked_pressure_max_relative_error=float(np.max(abs(ours['p'][mask] - existing['p'][mask])) / ref.p0),
        shocked_mach_max_error=float(np.max(abs(ours['M'][mask] - existing['M'][mask]))),
        reference_mass_flow_relative_range=float(np.ptp(ours['mdot']) / ref.mdot_star))
    assert max(checks.values()) < 1e-5, checks
    audits = []
    for row in report['runs']:
        # p, T and M are sufficient to recover primitives for this backend.
        # Recompute the original spatial residual from these saved physical data.
        field = row['flow_state']
        st = model.fluid.get_state(jxp.PT_INPUTS, np.array(field['p']), np.array(field['T']))
        full = jnp.asarray(np.column_stack([field['p'], np.array(field['M']) * np.asarray(st['a']), st['h']]))
        rms = float(model.metrics(full, jnp.float64(row['p_back_Pa']))[0])
        face_flux = np.asarray(model.face_flux(full, jnp.float64(row['p_back_Pa'])))
        flux_difference = float(np.max(abs(face_flux - np.asarray(field['mdot_face']))))
        audits.append(dict(mode=row['mode'], p_back_ratio=row['p_back_ratio'],
                           recomputed_spatial_rms=rms, recorded_converged=row['converged'],
                           face_flux_max_difference_kg_s=flux_difference))
        if row['converged']:
            assert rms < 1.01e-8, audits[-1]
            checks_now = model.workspace(row['p_back_Pa']).sample(full)
            assert checks_now['boundary_passed'] and checks_now['global_passed'], audits[-1]
        assert flux_difference < 1e-12, audits[-1]
    unchanged = hashes() == report['source_before'] == report['source_after']
    assert unchanged, 'Current source differs from the source used for the saved study'
    result = dict(reference_checks=checks, source_matches_saved_run=unchanged,
                  saved_field_audits=audits, audited_runs=len(audits),
                  maximum_recomputed_spatial_rms=max(r['recomputed_spatial_rms'] for r in audits))
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'saved_field_audits'}, indent=2))


if __name__ == '__main__':
    main()
