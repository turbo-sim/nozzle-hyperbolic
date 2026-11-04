"""Build the standalone multigrid report from saved JSON; never run a solver."""

from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
ASSETS = Path(__file__).with_name("report_assets")
VENDOR = ROOT / "development/flux_methods/shared/assets/vendor"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return (
        json.dumps(_clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _resolve_record(value, directory, inputs):
    """Materialize explicitly linked JSON records within the campaign root."""
    if not isinstance(value, dict):
        return value
    if "file" in value and isinstance(value["file"], str):
        candidate = (directory / value["file"]).resolve()
        if candidate.suffix == ".json":
            if not candidate.is_relative_to(directory):
                raise ValueError(f"Record outside campaign: {candidate}")
            if not candidate.is_file():
                raise ValueError(f"Missing linked record: {candidate}")
            raw = candidate.read_bytes()
            inputs[str(candidate.relative_to(directory))] = sha256(raw).hexdigest()
            value = json.loads(raw) | value
    return value


def read_campaign(input_path):
    source = Path(input_path).resolve()
    if source.is_dir():
        source = source / "summary.json"
    raw = source.read_bytes()
    data = json.loads(raw)
    inputs = {source.name: sha256(raw).hexdigest()}
    if not isinstance(data.get("groups"), list):
        raise ValueError("The report requires a groups list in the saved summary")
    for group in data["groups"]:
        if not group.get("case_id") or not group.get("reconstruction"):
            raise ValueError("Each group needs case_id and reconstruction")
        methods = group.get("methods", [])
        if not isinstance(methods, list):
            raise ValueError("Each group needs a methods list")
        for method in methods:
            if not method.get("method_id"):
                raise ValueError("Each method needs method_id")
            method["warmup"] = _resolve_record(method.get("warmup"), source.parent, inputs)
            method["samples"] = [
                _resolve_record(record, source.parent, inputs)
                for record in method.get("samples", [])
            ]
            if method.get("representative") is not None:
                method["representative"] = _resolve_record(
                    method["representative"], source.parent, inputs
                )
    startup_path = source.parent / "startup.json"
    if startup_path.is_file():
        startup_bytes = startup_path.read_bytes()
        startup = json.loads(startup_bytes)
        if not isinstance(startup.get("records"), list):
            raise ValueError("startup.json must contain a records list")
        data["startup_probes"] = startup
        inputs["startup.json"] = sha256(startup_bytes).hexdigest()
    tuning_path = source.parent / "tuning.json"
    if tuning_path.is_file():
        tuning_bytes = tuning_path.read_bytes()
        tuning = json.loads(tuning_bytes)
        if not isinstance(tuning, dict) or not isinstance(tuning.get("records"), list):
            raise ValueError("tuning.json must contain a records list")
        data["tuning"] = tuning
        inputs["tuning.json"] = sha256(tuning_bytes).hexdigest()
    core_audit_path = source.parent / "core_algebra_audit.json"
    if core_audit_path.is_file():
        core_audit_bytes = core_audit_path.read_bytes()
        core_audit = json.loads(core_audit_bytes)
        if not isinstance(core_audit, dict) or not isinstance(core_audit.get("checks"), dict):
            raise ValueError("core_algebra_audit.json must contain a checks object")
        audits = data.setdefault("audits", {})
        if not isinstance(audits, dict):
            raise ValueError("Campaign audits must be an object")
        audits["core_algebra"] = core_audit
        inputs["core_algebra_audit.json"] = sha256(core_audit_bytes).hexdigest()
    for filename, key in (
        ("transfer_algebra_audit.json", "initialization_transfer_algebra"),
        ("smoother_smoke.json", "implicit_smoother_smoke"),
        ("coarse_cap_smoke.json", "coarse_controller_smoke"),
    ):
        candidate = source.parent / filename
        if candidate.is_file():
            raw_audit = candidate.read_bytes()
            audit = json.loads(raw_audit)
            if not isinstance(audit, dict):
                raise ValueError(f"{filename} must contain an audit object")
            data.setdefault("audits", {})[key] = audit
            inputs[filename] = sha256(raw_audit).hexdigest()
    data["report_provenance"] = {
        "source_summary": source.name,
        "input_sha256": inputs,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets_sha256": {
            file.name: sha256(file.read_bytes()).hexdigest()
            for file in sorted(ASSETS.iterdir()) if file.is_file()
        },
        "plotly_version": "2.35.2",
        "mathjax_version": "3.2.2",
        "numerical_solves_during_render": False,
        "external_runtime_assets": False,
    }
    return data, source


class _AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.resources = []
        self.data = []
        self.collect = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("id") == "report-data":
            self.collect = True
        if tag in ("script", "img", "iframe", "link"):
            for attr in ("src", "href"):
                value = attrs.get(attr)
                if value and not value.startswith(("#", "data:", "blob:")):
                    self.resources.append(value)

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect = False

    def handle_data(self, data):
        if self.collect:
            self.data.append(data)


def audit_html(path, expected):
    parser = _AuditParser()
    content = Path(path).read_text(encoding="utf-8")
    parser.feed(content)
    payload = gzip.decompress(base64.b64decode("".join(parser.data), validate=True))
    ids = set(parser.ids)
    controls = {
        "report-data", "theme", "physical-case", "reconstruction", "methods",
        "overview", "explorer", "theory", "protocol", "all-timings", "case-timing",
        "profile-p", "profile-u", "profile-rho", "profile-M", "residual-chart",
        "work-chart", "compilation-chart", "download-data", "download-csv",
        "download-fields", "download-history", "residual-mode", "residual-axis",
        "startup-process-chart", "startup-compile-chart", "startup-table",
        "startup-protocol", "history-scope", "sequence-stage-table",
        "work-metric", "work-counter-head", "work-counter-table",
        "fmg-settings", "initialization-table", "tuning-panel", "tuning-table",
        "matched-control", "matched-summary", "matched-table",
    }
    duplicate = sorted({key for key in ids if parser.ids.count(key) > 1})
    missing = sorted(controls - ids)
    return {
        "passed": not parser.resources and not duplicate and not missing and payload == expected,
        "external_runtime_assets": parser.resources,
        "duplicate_ids": duplicate,
        "missing_controls": missing,
        "embedded_payload_identical": payload == expected,
        "embedded_json_sha256": sha256(payload).hexdigest(),
        "embedded_json_bytes": len(payload),
        "html_sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "html_bytes": Path(path).stat().st_size,
        "browser_verified": False,
        "note": "Structural/offline/data audit only; browser interactions require separate verification.",
    }


def build(input_path, output_path=None):
    """Render one portable HTML and save a structural audit next to the campaign."""
    data, source = read_campaign(input_path)
    output = Path(output_path or STUDY / "REPORT.html").resolve()
    plotly_bytes = gzip.decompress((VENDOR / "plotly-2.35.2.min.js.gz").read_bytes())
    expected_plotly = "6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603"
    if sha256(plotly_bytes).hexdigest() != expected_plotly:
        raise ValueError("The vendored Plotly runtime failed its recorded byte audit")
    mathjax = gzip.decompress((VENDOR / "mathjax-3.2.2-tex-svg.js.gz").read_bytes())
    licenses = "\n\n".join(
        (VENDOR / name).read_text(encoding="utf-8")
        for name in ("PLOTLY_LICENSE.txt", "MATHJAX_LICENSE.txt")
    )
    data["report_provenance"]["plotly_sha256"] = sha256(plotly_bytes).hexdigest()
    data["report_provenance"]["mathjax_sha256"] = sha256(mathjax).hexdigest()
    payload = _json(data).encode("utf-8")
    embedded = base64.b64encode(gzip.compress(payload, compresslevel=9, mtime=0)).decode("ascii")
    replacements = {
        "__STYLE__": (ASSETS / "style.css").read_text(encoding="utf-8"),
        "__APP__": (ASSETS / "app.js").read_text(encoding="utf-8"),
        "__PLOTLY__": plotly_bytes.decode("utf-8"),
        "__MATHJAX__": mathjax.decode("utf-8"),
        "__DATA__": embedded,
        "__DATA_HASH__": sha256(payload).hexdigest(),
        "__LICENSES__": escape(licenses),
    }
    html = (ASSETS / "template.html").read_text(encoding="utf-8")
    for token, value in replacements.items():
        html = html.replace(token, value)
    if re.search(r"__(?:STYLE|APP|PLOTLY|MATHJAX|DATA|DATA_HASH|LICENSES)__", html):
        raise ValueError("Unexpanded report template token")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    audit = audit_html(output, payload)
    folder = source.parent / "presentation"
    folder.mkdir(exist_ok=True)
    (folder / "report_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise ValueError(f"Report audit failed: {audit}")
    return {"output": str(output), "groups": len(data["groups"]), **audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/campaign/summary.json")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
