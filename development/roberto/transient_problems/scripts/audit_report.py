"""Audit saved animations for self-containment and valid full-resolution data."""

import argparse
from base64 import b64decode
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

from .report import STUDY, FIELDS, validate


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.pages, self.external = set(), [], []
        self.gifs = []
        self.data, self.collect = [], False

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if 'id' in attributes:
            self.ids.add(attributes['id'])
        if 'data-page' in attributes:
            self.pages.append(attributes['data-page'])
        for attribute in ('src', 'href'):
            value = attributes.get(attribute, '')
            if value and not value.startswith(('#', 'data:', 'blob:')):
                self.external.append(dict(tag=tag, attribute=attribute, value=value))
        if tag == 'script' and attributes.get('id') == 'data':
            self.collect = True
        if tag == 'img' and attributes.get('src', '').startswith('data:image/gif;base64,'):
            raw = b64decode(attributes['src'].split(',', 1)[1], validate=True)
            self.gifs.append(dict(id=attributes.get('id'), bytes=len(raw),
                width=int.from_bytes(raw[6:8], 'little'), height=int.from_bytes(raw[8:10], 'little'),
                sha256=sha256(raw).hexdigest(), valid=raw[:6] in (b'GIF87a', b'GIF89a')))

    def handle_endtag(self, tag):
        if tag == 'script':
            self.collect = False

    def handle_data(self, value):
        if self.collect:
            self.data.append(value)


def audit(path):
    html = path.read_text(encoding='utf-8')
    parser = Parser(); parser.feed(html)
    data = json.loads(''.join(parser.data))
    validate(data)
    expected = {'comparison', 'timeline', 'play', 'restart', 'previous', 'next', 'speed', 'loop', 'methods',
                'exact', 'autoscale', 'theme', 'download', 'record', 'simulation', 'theory', 'evidence'}
    missing = sorted(expected-parser.ids)
    methods = [method for case in data['cases'] for method in case['methods']]
    frames = sum(len(method.get('times', case['times'])) for case in data['cases'] for method in case['methods'])
    series = frames*len(FIELDS) + sum(len(case['times'])*len(case.get('reference', {}).get('fields', {})) for case in data['cases'])
    expected_pages = {case['id'] for case in data['cases']} | {'theory', 'evidence'}
    expected_methods = {'euler_explicit', 'rk3', 'implicit_coupled', 'implicit_segregated'}
    missing_methods = {case['id']: sorted(expected_methods-{method['id'] for method in case['methods']})
                       for case in data['cases']}
    return dict(passed=not parser.external and not missing and set(parser.pages) == expected_pages
                and all(gif['valid'] and gif['width'] > 0 and gif['height'] > 0 for gif in parser.gifs),
        file=str(path), sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
        cases=len(data['cases']), methods=len(methods), numerical_successes=sum(bool(method.get('success')) for method in methods),
        method_frames=frames, full_resolution_field_frames_checked=series,
        external_assets=parser.external, missing_controls=missing, pages=parser.pages,
        missing_method_records=missing_methods,
        embedded_gifs=parser.gifs,
        browser_layout_verified=False, video_codec_export_verified=False,
        note='Static full-resolution data and offline-asset audit; browser layout and encoded WebM require a browser check.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STUDY/'REPORT.html')
    parser.add_argument('--output', type=Path, default=STUDY/'outputs/report_audit.json')
    args = parser.parse_args()
    data = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    assert data['passed'], data
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
