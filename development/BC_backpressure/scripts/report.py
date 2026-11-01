"""Build the offline backpressure report from saved numerical records only.

No solver or thermodynamic backend is imported. Linked JSON records are
materialized losslessly, including unsuccessful outcomes and trial histories.
"""
from __future__ import annotations

import argparse
import ast
import base64
import gzip
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

import yaml

from development.flux_methods.shared.presentation import math_runtime, render_document

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
VENDOR = REPO / "development/flux_methods/shared/assets/vendor"
PLOTLY = VENDOR / "plotly-2.35.2.min.js.gz"
PG_CASE_FILES = {
    "nozzle_shock35": "src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_35.yaml",
    "nozzle_shock70": "src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_70.yaml",
    "annular_shock35": "src/nozzle_hyperbolic/models/euler_annular/cases/shock35.yaml",
    "annular_shock70": "src/nozzle_hyperbolic/models/euler_annular/cases/shock70.yaml",
}


def frozen_physical_inputs(data, campaign_path):
    """Recover physical inputs only from manifest-verified execution snapshots.

    This presentation enrichment neither imports physical solvers nor reads
    their live case definitions. Exact source text is embedded as evidence.
    """
    provenance = data.get("provenance") or {}
    snapshot_name = provenance.get("snapshot")
    if not snapshot_name:
        for case in data["cases"]:
            case["physical_inputs"] = dict(available=False, reason="No frozen execution snapshot registered")
        return
    root = campaign_path.parent.resolve()
    snapshot = (root / snapshot_name).resolve()
    if not snapshot.is_relative_to(root) or snapshot == root:
        raise ValueError("Frozen input snapshot must be a subdirectory of the campaign directory")
    cache = {}

    def read(relative):
        if relative in cache:
            return cache[relative]
        registered = [(name, provenance.get(name, {}).get(relative)) for name in ("source_files", "study_files")]
        registered = [(name, digest) for name, digest in registered if digest is not None]
        if not registered or len({digest for _, digest in registered}) != 1:
            raise ValueError(f"Frozen physical input is missing or conflicts in execution manifests: {relative}")
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError(f"Frozen input path escapes its execution snapshot: {relative}")
        raw = path.read_bytes()
        actual = sha256(raw).hexdigest()
        if actual != registered[0][1]:
            raise ValueError(f"Frozen physical input byte hash differs from execution provenance: {relative}")
        evidence = dict(path=relative, snapshot_path=f"{snapshot_name}/{relative}", sha256=actual,
                        manifests=[name for name, _ in registered], manifest_validated=True,
                        source_text=raw.decode("utf-8"))
        cache[relative] = evidence
        return evidence

    units = dict(pressure="Pa", temperature="K", length="m", area="m²", enthalpy="J/kg",
                 M_m="meridional Mach number, dimensionless",
                 alpha_tangent="tan(inlet flow angle) = tangential / meridional velocity",
                 inlet_ratio="inlet area / throat area", exit_ratio="exit area / throat area")
    for case in data["cases"]:
        key = case["id"]
        sources = []
        if key in PG_CASE_FILES:
            source = read(PG_CASE_FILES[key])
            definition = yaml.safe_load(source["source_text"])
            if definition["model"] != case["model"]:
                raise ValueError(f"Frozen physical model differs from saved case: {key}")
            effective = dict(model=definition["model"], fluid=definition["fluid"],
                             geometry=definition["geometry"], inlet=definition["inlet"],
                             walls=definition["walls"], reference_selection=definition["reference"],
                             reference_derived_back_pressure_Pa=case["back_pressure"],
                             reference_shock_position_m=case["shock_position"],
                             coordinate_domain_m=case["domain"],
                             boundary_definition_defaults=definition["boundaries"],
                             outlet_formulations_compared=data["settings"]["outlets"],
                             accuracy_targets=definition["accuracy"])
            sources.append(source)
        elif case.get("family") in ("supercritical", "lettieri_equilibrium"):
            physical = read("development/real_gas/cases.json")
            numerical = read("development/real_gas/settings.json")
            adapter = read("development/real_gas/scripts/thermodynamics.py")
            geometry_code = read("development/real_gas/scripts/geometry.py")
            definitions = json.loads(physical["source_text"])
            settings = json.loads(numerical["source_text"])
            matches = [d for d in definitions["cases"] if d["id"] == case["family"]]
            if len(matches) != 1:
                raise ValueError(f"Frozen CO2 physical definition is ambiguous: {key}")
            definition = matches[0]
            table = definition["table"] | settings.get("table_overrides", {}).get(case["family"], {})
            geometry = definitions["geometry"] | definition.get("geometry_override", {})
            calls = [node for node in ast.walk(ast.parse(adapter["source_text"]))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "FluidBicubic"]
            if len(calls) != 1:
                raise ValueError("Frozen bicubic adapter does not identify one table constructor")
            constants = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords
                         if kw.arg in ("backend", "N_h", "N_p")}
            if not math.isclose(definition["backpressure"], case["back_pressure"], rel_tol=1e-12):
                raise ValueError(f"Frozen CO2 backpressure differs from saved case: {key}")
            if [geometry["x_in"], geometry["x_out"]] != case["domain"]:
                raise ValueError(f"Frozen CO2 geometry domain differs from saved case: {key}")
            effective = dict(model=case["model"],
                             fluid=dict(name="CO2", evaluator="jaxprop.FluidBicubic", **constants),
                             inlet=dict(p0=definition["p0"], T0=definition["T0"]),
                             geometry=dict(kind="piecewise quadratic area law", **geometry),
                             back_pressure_Pa=case["back_pressure"], reference_shock_position_m=case["shock_position"],
                             coordinate_domain_m=case["domain"], table=table,
                             actual_table_bytes_sha256=case["table_sha256"],
                             outlet_formulations_compared=data["settings"]["outlets"])
            sources.extend((physical, numerical, adapter, geometry_code))
        else:
            raise ValueError(f"No frozen physical-input reader is defined for case: {key}")
        case["physical_inputs"] = dict(available=True, effective=effective, units=units,
                                       validation="Each embedded source is byte-verified against its execution source_files/study_files manifest",
                                       sources=sources)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_text(value):
    return (json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def read_campaign(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    frozen_physical_inputs(data, path)
    linked_hashes = {}
    for population in ("results", "seeds"):
        materialized = []
        for item in data.get(population, []):
            item = {"file": item} if isinstance(item, str) else item
            if "file" in item:
                link = (path.parent / item["file"]).resolve()
                if not link.is_relative_to(path.parent) or link.suffix != ".json":
                    raise ValueError(f"Record outside campaign directory: {item['file']}")
                source = link.read_bytes()
                linked_hashes[item["file"]] = sha256(source).hexdigest()
                item = json.loads(source) | item
            materialized.append(item)
        data[population] = materialized
    data["documents"] = {p.name: p.read_text(encoding="utf-8")
                         for p in (STUDY / "README.md", STUDY / "THEORY.md") if p.is_file()}
    audit_files = [path.parent / name for name in ("campaign_audit.json", "sensitivity_audit.json", "test_evidence.json")
                   if (path.parent / name).is_file()]
    data["saved_audits"] = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in audit_files}
    data["report_provenance"] = {
        "campaign_sha256": sha256(raw).hexdigest(), "linked_json_sha256": linked_hashes,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                          for p in sorted(ASSETS.iterdir()) if p.is_file()},
        "plotly_version": "2.35.2", "plotly_compressed_sha256": sha256(PLOTLY.read_bytes()).hexdigest(),
        "external_assets": False,
        "saved_audits_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in audit_files},
    }
    return data


def validate(data):
    if not isinstance(data.get("cases"), list) or not isinstance(data.get("results"), list):
        raise ValueError("A campaign needs cases and results lists")
    case_ids = [c["id"] for c in data["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs")
    run_ids = [r["id"] for r in data["results"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run IDs")
    for row in data["results"]:
        if row.get("case_id") not in case_ids:
            raise ValueError(f"Unknown case for {row['id']}")
        f = row.get("fields", {})
        x = f.get("x", [])
        for key, value in f.items():
            if isinstance(value, list) and key not in ("faces", "x_faces") and x and len(value) != len(x):
                raise ValueError(f"Field length mismatch: {row['id']}/{key}")
    return True


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.external, self.payload = [], [], []
        self.attributes, self.collect = {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("id") == "data":
            self.collect, self.attributes = True, attrs
        for attribute in ("src", "href"):
            value = attrs.get(attribute, "")
            if value and not value.startswith(("#", "data:", "blob:", "http:", "https:")):
                self.external.append({"tag": tag, "attribute": attribute, "value": value})
            if value.startswith(("http:", "https:")) and tag != "a":
                self.external.append({"tag": tag, "attribute": attribute, "value": value})

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect = False

    def handle_data(self, value):
        if self.collect:
            self.payload.append(value)


def audit_html(path, data, expected):
    parser = AuditParser()
    parser.feed(path.read_text(encoding="utf-8"))
    controls = {"case", "mesh", "outlet", "start", "flux", "reconstruction", "theme",
                "overview", "explorer", "theory", "evidence", "view-fields", "view-convergence",
                "view-sequence", "download-data", "download-fields", "download-history", "data",
                "sensitivity-table", "download-sensitivity", "selected-physical-inputs"}
    payload = "".join(parser.payload)
    encoding = parser.attributes.get("data-encoding", "plain-json")
    actual = gzip.decompress(base64.b64decode(payload, validate=True)) if encoding == "gzip-base64" else payload.encode()
    missing = sorted(controls - set(parser.ids))
    duplicate = sorted({x for x in parser.ids if parser.ids.count(x) > 1})
    same = actual == expected
    return dict(passed=not missing and not duplicate and not parser.external and same,
                html_sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                cases=len(data["cases"]), results=len(data["results"]), seeds=len(data.get("seeds", [])),
                missing_controls=missing, duplicate_ids=duplicate, external_assets=parser.external,
                complete_data_roundtrip_identical=same, raw_json_bytes=len(actual),
                history_rows=sum(len(r.get("history", [])) for r in data["results"]),
                embedded_data_sha256=sha256(actual).hexdigest(), data_encoding=encoding,
                browser_verified=False)


def build(input_path=STUDY / "outputs/campaign.json", output_path=STUDY / "REPORT.html", data_encoding="auto", audit_output=None):
    input_path, output_path = Path(input_path), Path(output_path)
    data = read_campaign(input_path)
    validate(data)
    raw = json_text(data).encode("utf-8")
    compressed = data_encoding == "gzip" or (data_encoding == "auto" and len(raw) >= 12_000_000)
    encoding = "gzip-base64" if compressed else "plain-json"
    data["report_provenance"]["data_encoding"] = encoding
    raw = json_text(data).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode() if compressed else raw.decode()
    embedded = (f'<script id="data" type="application/{"octet-stream" if compressed else "json"}" '
                f'data-encoding="{encoding}" data-json-bytes="{len(raw)}" '
                f'data-json-sha256="{sha256(raw).hexdigest()}" '
                f'data-history-rows="{sum(len(r.get("history", [])) for r in data["results"])}">{payload}</script>')
    theory = data["documents"].get("THEORY.md", "# Boundary theory\n\nThe theory document has not yet been saved. This partial report does not establish final coverage.")
    theory_html = render_document(theory, prefix="bc-theory")["html"]
    plotly = re.sub(r"</script", r"<\\/script", gzip.decompress(PLOTLY.read_bytes()).decode(), flags=re.I)
    license_text = (VENDOR / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8").replace("</script", "<\\/script")
    page = (ASSETS / "template.html").read_text(encoding="utf-8")
    replacements = {
        "<!--STYLE-->": "<style>" + (ASSETS / "style.css").read_text(encoding="utf-8") + "</style>",
        "<!--PLOTLY-->": "<script>" + plotly + '</script><script type="text/plain" id="plotly-license">' + license_text + "</script>",
        "<!--MATH-->": math_runtime(), "<!--THEORY-->": theory_html,
        "<!--DATA-->": embedded, "<!--APP-->": "<script>" + (ASSETS / "app.js").read_text(encoding="utf-8") + "</script>",
    }
    for token, value in replacements.items():
        page = page.replace(token, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8", newline="\n")
    audit = audit_html(output_path, data, raw)
    audit_file = Path(audit_output) if audit_output is not None else input_path.parent / "report_static_audit.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"Report structural audit failed: {audit}")
    print(f"Built {output_path}: {len(data['results'])} saved stages, {audit['bytes']:,} bytes; lossless data roundtrip verified")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/campaign.json")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--data-encoding", choices=("auto", "plain", "gzip"), default="auto")
    parser.add_argument("--audit-output", type=Path, help="Optional path for preview/report structural audit JSON")
    args = parser.parse_args()
    build(args.input, args.output, args.data_encoding, args.audit_output)


build_report = build


if __name__ == "__main__":
    main()
