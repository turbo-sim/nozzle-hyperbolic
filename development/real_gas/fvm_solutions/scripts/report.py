"""Build one offline Plotly report from saved real-gas FVM campaign data.

No thermodynamic call or numerical solve is performed by this module.
"""

from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
ASSETS = Path(__file__).resolve().parent / "report_assets"
PLOTLY = REPO / "development/roberto/benchmarking/scripts/report_assets/vendor/plotly-2.35.2.min.js.gz"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return (json.dumps(_clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def read_campaign(path):
    """Materialize explicitly linked result JSONs inside the output directory."""
    path = Path(path).resolve()
    campaign_bytes = path.read_bytes()
    data = json.loads(campaign_bytes)
    rows = []
    for item in data.get("results", []):
        item = {"file": item} if isinstance(item, str) else item
        if "file" in item:
            linked = (path.parent / item["file"]).resolve()
            if not linked.is_relative_to(path.parent) or linked.suffix != ".json":
                raise ValueError(f"Result reference outside campaign: {item['file']}")
            item = json.loads(linked.read_text(encoding="utf-8")) | item
        rows.append(item)
    data["results"] = rows
    data["report_provenance"] = {
        "campaign_sha256": sha256(campaign_bytes).hexdigest(),
        "report_generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_assets_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ASSETS.iterdir()) if p.is_file()},
        "plotly_compressed_sha256": sha256(PLOTLY.read_bytes()).hexdigest(),
        "plotly_version": "2.35.2", "external_assets": False,
        "source": path.name,
    }
    return data


def validate(data):
    if not isinstance(data.get("cases"), list) or not isinstance(data.get("results"), list):
        raise ValueError("Campaign must provide cases and results lists")
    ids = {case["id"] for case in data["cases"]}
    if len(ids) != len(data["cases"]):
        raise ValueError("Duplicate physical case IDs")
    for row in data["results"]:
        if row.get("case_id") not in ids:
            raise ValueError(f"Unknown physical case: {row.get('case_id')}")
        fields = row.get("fields", {})
        x = fields.get("x", row.get("x", []))
        for key, values in fields.items():
            if isinstance(values, list) and key != "faces" and x and len(values) != len(x):
                raise ValueError(f"Profile length mismatch in {row.get('id')}/{key}")
    return True


class _AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
        self.ids = []
        self.data_attributes = {}
        self.data_chunks = []
        self.collect_data = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("id") == "data":
            self.data_attributes = attrs
            self.collect_data = True
        for key in ("src", "href"):
            value = attrs.get(key, "")
            if value and not value.startswith(("#", "data:", "blob:")):
                self.external.append({"tag": tag, "attribute": key, "value": value})

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect_data = False

    def handle_data(self, value):
        if self.collect_data:
            self.data_chunks.append(value)


def audit_html(path, data, expected_json=None):
    parser = _AuditParser()
    parser.feed(path.read_text(encoding="utf-8"))
    controls = {"physical-case", "regime", "mesh", "reconstruction", "methods", "theme",
                "download-data", "download-fields", "download-history", "residual-mode",
                "solution", "convergence", "refinement", "methodology", "data"}
    missing = sorted(controls - set(parser.ids))
    duplicate = sorted({name for name in parser.ids if parser.ids.count(name) > 1})
    payload = "".join(parser.data_chunks)
    encoding = parser.data_attributes.get("data-encoding", "plain-json")
    decoded = (gzip.decompress(base64.b64decode(payload, validate=True))
               if encoding == "gzip-base64" else payload.encode("utf-8"))
    expected = expected_json if expected_json is not None else _json(data).encode("utf-8")
    identical = decoded == expected
    result = dict(passed=not parser.external and not missing and not duplicate and identical,
                  html_sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                  external_assets=parser.external, missing_controls=missing,
                  duplicate_ids=duplicate, cases=len(data["cases"]), results=len(data["results"]),
                  data_encoding=encoding, raw_json_bytes=len(decoded),
                  embedded_payload_bytes=len(payload.encode("utf-8")),
                  embedded_data_sha256=sha256(decoded).hexdigest(),
                  complete_data_roundtrip_identical=identical,
                  final_attempt_history_rows=sum(len(r.get("history", [])) for r in data["results"]),
                  retained_attempt_history_rows=sum(len(a.get("history", [])) for r in data["results"]
                                                    for a in r.get("attempts", [])),
                  browser_verified=False,
                  note="Static offline/data/control audit; browser interactions require separate verification.")
    return result


def build(input_path=STUDY / "outputs/campaign.json", output_path=STUDY / "REPORT.html", data_encoding="auto"):
    input_path, output_path = Path(input_path), Path(output_path)
    data = read_campaign(input_path)
    validate(data)
    if data_encoding not in ("auto", "plain", "gzip"):
        raise ValueError("data_encoding must be auto, plain or gzip")
    raw = _json(data).encode("utf-8")
    compressed = data_encoding == "gzip" or (data_encoding == "auto" and len(raw) >= 20_000_000)
    encoding = "gzip-base64" if compressed else "plain-json"
    data["report_provenance"]["data_encoding"] = encoding
    raw = _json(data).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii") if compressed else raw.decode("utf-8")
    data_type = "application/octet-stream" if compressed else "application/json"
    history_rows = sum(len(r.get("history", [])) for r in data["results"])
    retained_rows = sum(len(a.get("history", [])) for r in data["results"]
                        for a in r.get("attempts", []))
    embedded = (f'<script id="data" type="{data_type}" data-encoding="{encoding}" '
                f'data-json-bytes="{len(raw)}" data-json-sha256="{sha256(raw).hexdigest()}" '
                f'data-history-rows="{history_rows}" data-retained-history-rows="{retained_rows}">{payload}</script>')
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    plotly = gzip.decompress(PLOTLY.read_bytes()).decode("utf-8")
    plotly = re.sub(r"</script", r"<\\/script", plotly, flags=re.I)
    page = (template.replace("<!--STYLE-->", f"<style>{css}</style>")
            .replace("<!--PLOTLY-->", f"<script>{plotly}</script>")
            .replace("<!--DATA-->", embedded)
            .replace("<!--APP-->", f"<script>{app}</script>"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8", newline="\n")
    audit = audit_html(output_path, data, expected_json=raw)
    audit_file = input_path.parent / "report_static_audit.json"
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"Report structural audit failed: {audit}")
    print(f"Built {output_path} ({audit['bytes']:,} bytes; {audit['results']} results; "
          f"{encoding}; {len(raw):,} raw JSON bytes; lossless roundtrip verified)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/campaign.json")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--data-encoding", choices=("auto", "plain", "gzip"), default="auto",
                        help="auto embeds lossless gzip/base64 when JSON exceeds 20 MB; no data are downsampled")
    args = parser.parse_args()
    build(args.input, args.output, args.data_encoding)


if __name__ == "__main__":
    main()
