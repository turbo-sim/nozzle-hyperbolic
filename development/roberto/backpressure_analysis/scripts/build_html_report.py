"""Build the self-contained Plotly report from saved fields; never run a solver.

Run from the repository root:
  poetry run python -m development.roberto.backpressure_analysis.scripts.build_html_report

Plotly's JavaScript runtime is embedded once. It can come from an installed
Plotly package, the neighboring TurboDash environment, or --plotly-js. This
renderer does not install packages or import the TurboDash application.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import jaxprop as jxp
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np

from development.roberto.backpressure_analysis.scripts.reference import PerfectGasReference

STUDY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
ASSETS = Path(__file__).resolve().parent / 'report_assets'
EXPECTED_SUMMARY_SHA256 = '448d62d5bac6083d5367b5b2a0c8cb5c75eefcc01e6d9b4e7a4aa2a360afff32'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_runtime(explicit):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    spec = importlib.util.find_spec('plotly')
    candidates = []
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent / 'package_data' / 'plotly.min.js')
    turbo = ROOT.parents[1] / 'turbodash' / '.venv'
    candidates.append(turbo / 'Lib' / 'site-packages' / 'plotly' / 'package_data' / 'plotly.min.js')
    candidates.extend(turbo.glob('lib/python*/site-packages/plotly/package_data/plotly.min.js'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Plotly JavaScript not found. Supply --plotly-js PATH or install Plotly in the rendering environment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STUDY / 'outputs' / 'baseline')
    parser.add_argument('--output', type=Path, default=STUDY / 'report_backpressure_sensitivity.html')
    parser.add_argument('--plotly-js', type=Path)
    args = parser.parse_args()
    summary = args.input / 'summary.json'
    audit_path = args.input / 'audit.json'
    data = json.loads(summary.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    input_hashes = {path.name: sha256(path) for path in [summary, audit_path, args.input / 'operating_map.csv']}
    # The editorial template describes this audited baseline. Fail clearly if
    # a different study is passed instead of publishing stale narrative values.
    counts = {mode: sum(r['mode'] == mode for r in data['runs'])
              for mode in ('descending', 'cold', 'supersonic_restart')}
    if input_hashes['summary.json'] != EXPECTED_SUMMARY_SHA256:
        raise ValueError('Saved results changed. Review the editorial narrative and update its dataset fingerprint before publishing.')
    if counts != {'descending': 87, 'cold': 14, 'supersonic_restart': 3}:
        raise ValueError(f'Update the report narrative for a changed dataset: {counts}')
    assert audit['audited_runs'] == len(data['runs']) == 104
    assert data['source_unchanged'] and audit['source_matches_saved_run']
    assert all(row['converged'] for row in data['runs'])
    reference = PerfectGasReference(jxp.FluidPerfectGas(name='Air'))
    profiles = {}
    for row in data['runs']:
        key = f'{row["p_back_ratio"]:.12f}'
        if key in profiles:
            continue
        curve = reference.profile(np.asarray(data['x_m']), row['p_back_ratio'])
        profiles[key] = {name: curve[name].tolist() for name in ('p', 'M', 'T', 'ds')}
    runtime_path = locate_runtime(args.plotly_js)
    runtime = runtime_path.read_text(encoding='utf-8')
    match = re.search(r'plotly\.js v([\w.]+)', runtime[:500])
    runtime_version = match.group(1) if match else 'bundled distribution'
    payload = dict(
        report=data, audit=audit, references=profiles,
        colors=[to_hex(colormaps['Blues'](v)) for v in np.linspace(.45, .95, 121)],
        csv=(args.input / 'operating_map.csv').read_text(encoding='utf-8'),
        provenance=dict(study_date='2026-10-28', source_files=len(data['source_before']),
                        input_sha256=input_hashes, plotly_js_version=runtime_version,
                        style_sources=['turbodash/visualization/matplotlib_style.py:set_plot_options',
                                       'turbodash/visualization/plotly_turbine.py:apply_plot_theme,_style_axes',
                                       'turbodash/development/app_design/visual-style.md'],
                        renderer='scripts/build_html_report.py', numerical_solves_rerun=False,
                        palette='Blues', palette_range=[.45, .95], presentation='desktop technical report'))
    encoded = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (ASSETS / 'report.html').read_text(encoding='utf-8')
    substitutions = {
        '__STYLE__': (ASSETS / 'report.css').read_text(encoding='utf-8'),
        '__PLOTLY_RUNTIME__': runtime.replace('</script', '<\\/script'),
        '__REPORT_DATA__': encoded,
        '__REPORT_APP__': (ASSETS / 'report.js').read_text(encoding='utf-8'),
    }
    for token, value in substitutions.items():
        assert template.count(token) == 1, f'Missing/duplicated template token: {token}'
        template = template.replace(token, value)
    assert not re.search(r'<(?:script|img)[^>]+src\s*=', template, re.IGNORECASE), 'Report must be self-contained'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding='utf-8')
    assert input_hashes == {path.name: sha256(path) for path in [summary, audit_path, args.input / 'operating_map.csv']}
    manifest = dict(input_sha256=input_hashes, output_sha256=sha256(args.output),
                    plotly_runtime_sha256=sha256(runtime_path), plotly_js_version=runtime_version,
                    assets_sha256={p.name: sha256(p) for p in ASSETS.iterdir() if p.is_file()},
                    output_bytes=args.output.stat().st_size,
                    numerical_solves_rerun=False)
    (args.input / 'html_report_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Created {args.output} ({manifest["output_bytes"] / 1e6:.2f} MB), Plotly.js {runtime_version}')
    print(f'Embedded {len(data["runs"])} states, {len(profiles)} analytical profiles and the independent audit.')
    print('Saved numerical inputs unchanged; no integration or source changes.')


if __name__ == '__main__':
    main()
