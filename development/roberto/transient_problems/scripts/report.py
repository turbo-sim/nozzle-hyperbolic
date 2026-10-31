"""Build a shareable offline animation from saved production transient fields.

This module only validates, summarizes and presents saved arrays. It imports no
solver, physical model or thermodynamic backend and performs no numerical run.
"""

import argparse
from base64 import b64encode
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('report_assets')
FIELDS = ('p', 'rho', 'u', 'M', 'T')


def validate(data):
    errors = []
    if data.get('schema_version') != 1:
        errors.append('Unsupported report data schema; expected 1')
    for case in data.get('cases', []):
        x, times = case.get('x', []), case.get('times', [])
        if len(x) < 2 or not times or any(not math.isfinite(v) for v in x+times):
            errors.append(f'{case.get("id")}: missing/nonfinite coordinates or times')
            continue
        if any(b <= a for a, b in zip(x, x[1:])) or any(b <= a for a, b in zip(times, times[1:])):
            errors.append(f'{case["id"]}: coordinates/times are not strictly increasing')
        area = case.get('area_centers')
        if area is not None and (len(area) != len(x) or any(not math.isfinite(v) or v <= 0 for v in area)):
            errors.append(f'{case["id"]}: invalid fixed area samples')
        faces = case.get('faces')
        if faces is not None and (len(faces) != len(x)+1 or any(not math.isfinite(v) for v in faces)
                                  or any(b <= a for a, b in zip(faces, faces[1:]))):
            errors.append(f'{case["id"]}: invalid physical faces')
        for method in case.get('methods', []):
            frames = method.get('times', times)
            for field in FIELDS:
                values = method.get('fields', {}).get(field, [])
                if len(values) != len(frames) or any(len(row) != len(x) for row in values):
                    errors.append(f'{case["id"]}/{method["id"]}/{field}: invalid frame shape')
                elif any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in values for v in row):
                    errors.append(f'{case["id"]}/{method["id"]}/{field}: nonfinite field')
        reference = case.get('reference', {}).get('fields', {})
        for field, values in reference.items():
            if field not in FIELDS:
                continue
            if len(values) != len(times) or any(len(row) != len(x) for row in values):
                errors.append(f'{case["id"]}/reference/{field}: invalid frame shape')
            elif any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in values for v in row):
                errors.append(f'{case["id"]}/reference/{field}: nonfinite field')
    if not data.get('cases'):
        errors.append('No cases in saved dataset')
    if errors:
        raise ValueError('\n'.join(errors))


def method_summary(case, method):
    metrics = method.get('metrics', {})
    boundary = metrics.get('boundary_max', [])
    budgets = metrics.get('conservation_budget', [])
    times = method.get('times', case['times'])
    return dict(case=case['id'], method=method['id'], label=method.get('label', method['id']),
        success=bool(method.get('success', False)), reason=method.get('reason', ''),
        frames=len(times), final_time=times[-1] if times else None,
        boundary_max=max((float(v) for v in boundary if v is not None), default=None),
        conservation_budget_max=max((abs(float(v)) for row in budgets for v in row if v is not None), default=None),
        summary=method.get('summary', {}))


def implicit_comparisons(case):
    methods = {method['id']: method for method in case['methods']}
    pair = [methods.get(name) for name in ('implicit_coupled', 'implicit_segregated')]
    if any(method is None for method in pair):
        return None
    left, right = pair
    if left.get('times', case['times']) != right.get('times', case['times']):
        return dict(available=False, reason='Implicit methods did not reach identical output times')
    maxima = {}
    for field in FIELDS:
        maxima[field] = max((abs(a-b) for L, R in zip(left['fields'][field], right['fields'][field])
                             for a, b in zip(L, R)), default=0.)
    return dict(available=True, maximum_absolute_field_difference=maxima,
                units=dict(p='Pa', rho='kg/m^3', u='m/s', M='1', T='K'))


def build(data):
    validate(data)
    summaries = [method_summary(case, method) for case in data['cases'] for method in case['methods']]
    display = dict(methods=summaries,
                   implicit_comparisons={case['id']: implicit_comparisons(case) for case in data['cases']})
    payload = dict(data, presentation_summary=display)
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    nav = ''.join(f'<button data-page="{escape(case["id"], quote=True)}">{escape(case["label"])}</button>'
                  for case in data['cases'])
    body = (ASSETS/'template.html').read_text(encoding='utf-8')
    animations = []
    manifest = data.get('report_animation_manifest', {})
    animation_records = manifest.get('records', []) if manifest.get('campaign_sha256') == data.get('report_input_sha256') else []
    for i, record in enumerate(animation_records):
        path = STUDY/'outputs/animations'/record['file']
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'Animation hash differs from saved manifest: {path}')
        encoded_gif = b64encode(raw).decode('ascii')
        animations.append(f'<article><h3>{escape(path.stem.replace("_", " ").title())}</h3>'
            f'<img id="gif-{i}" loading="lazy" src="data:image/gif;base64,{encoded_gif}" alt="Computed transient comparison animation">'
            f'<a href="#gif-{i}" data-gif-target="gif-{i}" data-gif-name="{escape(path.name, quote=True)}">'
            f'Download GIF ({path.stat().st_size/1.e6:.1f} MB)</a></article>')
    animation_block = ('<details class="panel shareable"><summary>Shareable GIF animations</summary>'
        '<p>These self-contained animated images use the computed snapshots and can be downloaded without browser video-codec support.</p>'
        '<div class="gif-grid">'+''.join(animations)+'</div></details>') if animations else ''
    return (body.replace('__NAVIGATION__', nav)
                .replace('__ANIMATIONS__', animation_block)
                .replace('__STYLE__', (ASSETS/'style.css').read_text(encoding='utf-8'))
                .replace('__DATA__', encoded)
                .replace('__SCRIPT__', (ASSETS/'app.js').read_text(encoding='utf-8'))), display


def markdown(data, display, source):
    rows = []
    for row in display['methods']:
        details = row['summary']
        steps = details.get('accepted_steps', details.get('steps', 'n/a'))
        boundary = f"{row['boundary_max']:.3e}" if row['boundary_max'] is not None else 'n/a'
        budget = f"{row['conservation_budget_max']:.3e}" if row['conservation_budget_max'] is not None else 'n/a'
        rows.append(f"| {row['case']} | {row['label']} | {row['success']} | {steps} | {row['frames']} | {boundary} | {budget} |")
    text = ('# Transient boundary-treatment verification\n\n'
        'The animations display computed physical-cell snapshots from the production solver. '
        'Backward Euler is compared with coupled and segregated boundary treatment; forward Euler '
        'and SSPRK3 use nonlinear segregated boundary completion at every stage.\n\n'
        '| Case | Method | Success | Accepted steps | Frames | Max BC defect | Max scaled budget |\n'
        '|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n')
    evidence = data.get('report_evidence', {})
    independent = evidence.get('independent_audit')
    if independent:
        checks = independent.get('records', [])
        pairs = independent.get('implicit_comparisons', [])
        text += (f"Independent numerical audits: {sum(row['passed'] for row in checks)}/{len(checks)}; "
                 f"implicit-treatment comparisons: {sum(row['passed'] for row in pairs)}/{len(pairs)}. "
                 f"Overall audit passed: {independent.get('passed')}.\n\n")
        if pairs:
            text += '| Implicit-treatment comparison | Maximum normalized field difference | Target | Passed |\n|---|---|---|---|\n'
            for pair in pairs:
                maximum = max(pair['normalized_maximum_field_difference'].values())
                text += f"| {pair['case']} | {maximum:.3e} | {pair['tolerance']:.1e} | {pair['passed']} |\n"
            text += '\n'
    tests = evidence.get('production_tests')
    if tests and 'counts' in tests:
        counts = tests['counts']
        text += (f"Full production suite: {counts['passed']} passed, {counts['skipped']} expected failures/skips; "
                 f"test execution passed: {tests.get('passed')}.\n\n")
    provenance = data.get('provenance', {})
    text += (f"Numerical source unchanged during execution: {provenance.get('source_unchanged')}; "
             f"numerical clients unchanged: {provenance.get('client_unchanged')}.\n\n")
    for case in data['cases']:
        item = display['implicit_comparisons'][case['id']]
        if item and item.get('available'):
            values = item['maximum_absolute_field_difference']
            text += (f"For {case['id']}, maximum coupled-versus-segregated backward-Euler differences over all saved frames: "
                     f"pressure {values['p']:.3e} Pa, density {values['rho']:.3e} kg/m³, "
                     f"velocity {values['u']:.3e} m/s, Mach {values['M']:.3e}, temperature {values['T']:.3e} K.\n\n")
    sod = next((case for case in data['cases'] if case['id'] == 'sod'), None)
    if sod:
        text += '| Sod final normalized mean error | p | rho | u | M | T |\n|---|---|---|---|---|---|\n'
        for method in sod['methods']:
            errors = method.get('summary', {}).get('final_exact_L1', {})
            values = [f'{errors[field]:.5f}' if field in errors else 'n/a' for field in FIELDS]
            text += '| '+method.get('label', method['id'])+' | '+' | '.join(values)+' |\n'
        text += '\n'
    text += ('These are finite-CFL transient calculations. Backward Euler has first-order temporal diffusion; '
             'differences from explicit Euler/SSPRK3 do not by themselves identify a boundary-condition error. '
             'Forward Euler can look sharper here because its temporal error partly offsets first-order spatial diffusion; '
             'this does not establish general superiority over SSPRK3. '
             'Coupled and segregated backward Euler impose the same nonlinear boundary equations. '
             'Segregated elimination includes the implicit derivative of the completed trace.\n\n'
             'At the nozzle initial frame, physical cells are stagnant. Zero-storage endpoint traces are '
             'completed nonlinearly at $t=0^+$ after the pressure step, without changing physical-cell storage; '
             'endpoint traces are not plotted as cells. '
             'Sod uses transmissive boundaries and an independent exact solution before waves reach the ends.\n\n'
             'Open [REPORT.html](REPORT.html) for offline play/pause, time scrubbing, method selection and '
             'five physical fields. WebM export is available when the browser supports canvas capture. '
             'The HTML embeds its numeric data and requires no network connection.\n\n'
             f'Numerical data: `{source}`. Settings, histories and source provenance remain in `outputs/`. '
             'Reproduction commands and scope are in [README.md](README.md).\n')
    if data.get('report_animation_manifest', {}).get('campaign_sha256') == data.get('report_input_sha256'):
        text += '\nShareable animations: '+', '.join(
            f"[{row['case']} GIF](outputs/animations/{row['file']})" for row in data['report_animation_manifest']['records'])+'. '
        text += 'The same directory contains initial/middle/final PNG and PDF keyframes. Both GIFs are embedded in the HTML download appendix.\n'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STUDY/'outputs/verification/summary.json')
    parser.add_argument('--output', type=Path, default=STUDY/'REPORT.html')
    parser.add_argument('--markdown', type=Path, default=STUDY/'VERIFICATION_REPORT.md')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    data['report_input_sha256'] = sha256(args.input.read_bytes()).hexdigest()
    manifest_path = STUDY/'outputs/animations/manifest.json'
    if manifest_path.exists():
        data['report_animation_manifest'] = json.loads(manifest_path.read_text(encoding='utf-8'))
    evidence = {}
    audit_path = args.input.with_name('audit.json')
    if audit_path.exists():
        evidence['independent_audit'] = json.loads(audit_path.read_text(encoding='utf-8'))
    tests_path = STUDY/'outputs/production_tests_provenance.json'
    if tests_path.exists():
        evidence['production_tests'] = json.loads(tests_path.read_text(encoding='utf-8'))
    data['report_evidence'] = evidence
    data['report_evidence_source_matches_campaign'] = {
        name: record.get('provenance', record).get('source_before') == data.get('provenance', {}).get('source_before')
        for name, record in evidence.items()}
    html, display = build(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    source = args.input.relative_to(STUDY).as_posix() if args.input.is_relative_to(STUDY) else str(args.input)
    args.markdown.write_text(markdown(data, display, source), encoding='utf-8')
    print(f'Wrote {args.output} ({args.output.stat().st_size:,} bytes)')
    print(f'HTML sha256 {sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
