"""Prepare a compact full-time, spatially sampled input for the JavaScript audit.

The separate static audit validates every full-resolution value. This helper
keeps every time frame and samples three coordinates to exercise interactive
control logic without sending the full embedded GIFs through a JS tool bridge.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .report import STUDY


def prepare(path):
    html = path.read_text(encoding='utf-8')
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    for case in data['cases']:
        count = len(case['x']); indices = [0, count//2, count-1]
        case['x'] = [case['x'][i] for i in indices]
        case['area_centers'] = [case['area_centers'][i] for i in indices]
        faces = case['faces']; case['faces'] = [faces[0], faces[indices[1]], faces[indices[2]], faces[-1]]
        for method in case['methods']:
            method['fields'] = {key: [[row[i] for i in indices] for row in rows] for key, rows in method['fields'].items()}
            method['metrics'] = {key: value for key, value in method['metrics'].items() if key in ('boundary_max', 'conservation_budget')}
        if 'reference' in case:
            case['reference']['fields'] = {key: [[row[i] for i in indices] for row in rows]
                                           for key, rows in case['reference']['fields'].items()}
    data['provenance'] = {key: value for key, value in data['provenance'].items()
                          if key not in ('source_before', 'source_after', 'client_before', 'client_after')}
    evidence = data.get('report_evidence', {})
    audit = evidence.get('independent_audit')
    if audit:
        evidence['independent_audit'] = {key: audit[key] for key in ('passed', 'implicit_comparisons',
            'campaign_source_matches_current', 'campaign_client_matches_current')}
        evidence['independent_audit']['records'] = [{key: row.get(key) for key in ('case', 'method', 'passed')}
                                                  for row in audit['records']]
    tests = evidence.get('production_tests')
    if tests:
        evidence['production_tests'] = dict(passed=tests['passed'], counts=tests['counts'],
            provenance=dict(source_unchanged=tests.get('provenance', tests)['source_unchanged']))
    code = re.search(r'</script><script>(.*?)</script>', html, re.S).group(1)
    links = ''.join('<a '+match.group(0)+'></a>' for match in re.finditer(
        r'data-gif-target="[^"]+" data-gif-name="[^"]+"', html))
    compact = (links+'<script type="application/json" id="data">'+json.dumps(data, separators=(',', ':'))
               +'</script><script>'+code+'</script>')
    return dict(html=compact, code=Path(__file__).with_name('runtime_audit.js').read_text(encoding='utf-8'),
                report_sha256=sha256(path.read_bytes()).hexdigest(), spatial_samples_per_series=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STUDY/'REPORT.html')
    parser.add_argument('--output', type=Path, default=STUDY/'outputs/presentation_qa/runtime_input.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepare(args.input), separators=(',', ':')), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
