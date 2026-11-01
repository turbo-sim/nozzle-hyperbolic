"""Build a standalone Newton study report from saved evidence, without solves."""
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
import struct

from development.flux_methods.shared.presentation import math_runtime, render_document

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
VENDOR = REPO / "development/flux_methods/shared/assets/vendor"
PLOTLY = VENDOR / "plotly-2.35.2.min.js.gz"
CASE_FILES = {
    "nozzle_shock35": "src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_35.yaml",
    "nozzle_shock70": "src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_70.yaml",
    "adapted": "src/nozzle_hyperbolic/models/euler_nozzle/cases/adapted.yaml",
}


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


def contained(root, relative):
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Evidence path escapes campaign: {relative}")
    return resolved


def decode_record(raw, path):
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)


def initial_bytes_hash(row):
    """Independent hash of serialized initial physical float64 state bytes."""
    state = row.get("initial_state")
    if not isinstance(state, list) or not state or not all(isinstance(v, list) for v in state):
        return None
    values = [float(value) for point in state for value in point]
    if not all(math.isfinite(value) for value in values):
        return None
    return sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def report_record(record, source_relative=None, source_raw=None):
    """Retain all outcome/scalar work evidence; native event vectors stay on disk.

    Initial/final states and the final original residual vectors remain in the
    report. Hundreds-dimensional coordinates at every native event are useful
    raw archives, but would make a complete 96-case offline report impractical.
    """
    record = dict(record)
    omitted = {}
    vector_keys = {"x", "x_base", "fun", "coordinates", "physical_state"}

    def scalar_events(value, path):
        if isinstance(value, dict):
            result = {}
            for key, content in value.items():
                if key in vector_keys and isinstance(content, list):
                    location = f"{path}.{key}"
                    omitted[location] = omitted.get(location, 0)+1
                else:
                    result[key] = scalar_events(content, f"{path}.{key}")
            return result
        if isinstance(value, list):
            return [scalar_events(item, f"{path}[]") for item in value]
        return value

    if "trials" in record:
        record["trials"] = scalar_events(record["trials"], "trials")
    if isinstance(record.get("direct"), dict):
        direct = dict(record["direct"])
        for name in ("native_history", "evaluations", "domain_failures"):
            if name in direct:
                direct[name] = scalar_events(direct[name], f"direct.{name}")
        if "trials" in direct:
            if "trials" not in record:
                record["trials"] = scalar_events(direct["trials"], "trials")
            direct["report_trials_reference"] = "The identical scalar trial sequence is stored once in this record's top-level trials"
            del direct["trials"]
        record["direct"] = direct
    record["report_initial_state_bytes_sha256"] = initial_bytes_hash(record)
    record["raw_record_evidence"] = dict(
        relative_path=source_relative, sha256=sha256(source_raw).hexdigest() if source_raw is not None else None,
        bytes=len(source_raw) if source_raw is not None else None,
        embedded=False, omitted_native_event_vectors=omitted,
        embedded_scalar_record_counts=dict(trials=len(record.get("trials", [])),
            native_history=len(record.get("direct", {}).get("native_history", [])),
            evaluations=len(record.get("direct", {}).get("evaluations", [])),
            domain_failures=len(record.get("direct", {}).get("domain_failures", []))),
        retention="Full native coordinates and event residual vectors remain in the raw run JSON on disk. All scalar event diagnostics, initial/final physical fields and final residual proofs are embedded.")
    return record


def frozen_inputs(data, root):
    """Embed exact execution YAMLs, never live physical case definitions."""
    provenance = data.get("provenance") or {}
    name = provenance.get("snapshot")
    if not name:
        return {"available": False, "reason": "No frozen execution snapshot is registered"}
    snapshot = contained(root, name)
    sources = []
    for case_id, relative in CASE_FILES.items():
        digest = provenance.get("source_files", {}).get(relative)
        if digest is None:
            continue
        raw = contained(snapshot, relative).read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"Frozen physical input differs from execution manifest: {relative}")
        sources.append(dict(case_id=case_id, path=relative, sha256=digest, source_text=raw.decode("utf-8")))
    return dict(available=bool(sources), sources=sources,
                validation="Exact source bytes verified against the execution source_files manifest")


def read_campaign(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    hashes = {}
    for population in ("results", "seeds"):
        materialized = []
        for item in data.get(population, []):
            item = {"file": item} if isinstance(item, str) else dict(item)
            record_raw, relative = None, None
            if "file" in item:
                linked = contained(path.parent, item["file"])
                if not str(linked).endswith((".json", ".json.gz")):
                    raise ValueError("Only JSON result records may be materialized")
                record_raw, relative = linked.read_bytes(), item["file"]
                hashes[relative] = sha256(record_raw).hexdigest()
                if item.get("file_sha256") and item["file_sha256"] != hashes[relative]:
                    raise ValueError(f"Saved result differs from campaign index: {relative}")
                item = decode_record(record_raw, linked) | item
            else:
                candidate = f"{'runs' if population == 'results' else 'seeds'}/{item.get('id', '')}.json"
                linked = contained(path.parent, candidate)
                if not linked.is_file():
                    candidate += ".gz"
                    linked = contained(path.parent, candidate)
                if linked.is_file():
                    record_raw, relative = linked.read_bytes(), candidate
                    hashes[relative] = sha256(record_raw).hexdigest()
            materialized.append(report_record(item, relative, record_raw))
        data[population] = materialized
    for seed in data.get("seeds", []):
        stages = []
        for cells in data.get("settings", {}).get("adapted_seed_grids", [50, 100, 200]):
            relative = f"seed_stages/{seed['id']}/N{cells}.json"
            stage_path = contained(path.parent, relative)
            if not stage_path.is_file():
                relative += ".gz"
                stage_path = contained(path.parent, relative)
            if stage_path.is_file():
                stage_raw = stage_path.read_bytes()
                hashes[relative] = sha256(stage_raw).hexdigest()
                stages.append(report_record(decode_record(stage_raw, stage_path), relative, stage_raw))
        seed["stage_records"] = stages
    return enrich_campaign(data, path, raw, hashes)


def enrich_campaign(data, path, raw, hashes):
    """Refresh documents/audits without reading or recomputing numerical states."""
    path = Path(path).resolve()
    data["frozen_physical_inputs"] = frozen_inputs(data, path.parent)
    data["report_dataset_scope"] = {
        "embedded": "Every requested outcome, initial/final physical state, accepted metric history, scalar native trial/evaluation/domain-failure record, solver option, counter, independent audit and source provenance",
        "raw_on_disk": "Native event coordinate/residual vectors remain in the SHA-identified full run files under outputs/runs; they are not embedded in this HTML",
        "duplicate_trials": "Identical nested direct.trials are represented once at each result's top-level trials",
        "downloads": "All offered downloads are generated entirely from this embedded report dataset and require no external files"}
    data["documents"] = {p.name: p.read_text(encoding="utf-8")
                         for p in (STUDY / "README.md", STUDY / "THEORY.md", STUDY / "RESULTS.md") if p.is_file()}
    audits = [p for p in set(path.parent.glob("*audit*.json")) | set(path.parent.glob("*diagnostics*.json")) | {path.parent / "verify_saved.json"}
              if p.is_file()
              and p.name not in ("report_static_audit.json", "report_browser_audit.json")]
    data["saved_audits"] = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(audits)}
    data["report_provenance"] = dict(
        campaign_sha256=sha256(raw).hexdigest(), linked_json_sha256=hashes,
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        assets_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(ASSETS.iterdir()) if p.is_file()},
        plotly_version="2.35.2", plotly_compressed_sha256=sha256(PLOTLY.read_bytes()).hexdigest(),
        saved_audits_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(audits)},
        documents_sha256={name: sha256(text.encode("utf-8")).hexdigest() for name, text in data["documents"].items()},
        external_assets=False)
    return data


def validate(data):
    if not isinstance(data.get("cases"), list) or not isinstance(data.get("results"), list):
        raise ValueError("A campaign needs cases and results lists")
    cases = [c["id"] for c in data["cases"]]
    ids = [r["id"] for r in data["results"]]
    if len(cases) != len(set(cases)) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate physical case or run IDs")
    for row in data["results"]:
        if row.get("case_id") not in cases:
            raise ValueError(f"Unknown case in {row['id']}")
        fields = row.get("fields") or {}
        x = fields.get("x", [])
        for name, values in fields.items():
            if isinstance(values, list) and name not in ("x_faces", "faces") and x and len(values) != len(x):
                raise ValueError(f"Field length mismatch: {row['id']}/{name}")


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
        for key in ("src", "href"):
            value = attrs.get(key, "")
            if value and not value.startswith(("#", "data:", "blob:", "http:", "https:")):
                self.external.append(dict(tag=tag, attribute=key, value=value))
            if value.startswith(("http:", "https:")) and tag != "a":
                self.external.append(dict(tag=tag, attribute=key, value=value))

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect = False

    def handle_data(self, value):
        if self.collect:
            self.payload.append(value)


def audit_html(path, data, expected):
    parser = AuditParser()
    parser.feed(path.read_text(encoding="utf-8"))
    controls = {"case", "flux", "reconstruction", "outlet", "start", "strategy", "theme",
                "overview", "explorer", "theory", "evidence", "view-fields", "view-convergence",
                "view-work", "run-focus", "download-data", "download-fields", "download-history",
                "download-trials", "download-theory", "data", "run-table", "pair-table"}
    payload = "".join(parser.payload)
    actual = gzip.decompress(base64.b64decode(payload, validate=True)) if parser.attributes.get("data-encoding") == "gzip-base64" else payload.encode()
    missing = sorted(controls - set(parser.ids))
    duplicate = sorted({x for x in parser.ids if parser.ids.count(x) > 1})
    return dict(passed=not missing and not duplicate and not parser.external and actual == expected,
                html_sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                cases=len(data["cases"]), results=len(data["results"]), seeds=len(data.get("seeds", [])),
                missing_controls=missing, duplicate_ids=duplicate, external_assets=parser.external,
                complete_data_roundtrip_identical=actual == expected, raw_json_bytes=len(actual),
                embedded_data_sha256=sha256(actual).hexdigest(), browser_verified=False)


def build(input_path=STUDY / "outputs/campaign.json", output_path=STUDY / "REPORT.html", data_encoding="gzip", audit_output=None, *, prepared_data=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if prepared_data is None:
        data = read_campaign(input_path)
    else:
        campaign_raw = input_path.read_bytes()
        if prepared_data["report_provenance"]["campaign_sha256"] != sha256(campaign_raw).hexdigest():
            raise ValueError("Prepared report data belongs to a different campaign index")
        data = enrich_campaign(prepared_data, input_path, campaign_raw, prepared_data["report_provenance"]["linked_json_sha256"])
    validate(data)
    encoding = "plain-json" if data_encoding == "plain" else "gzip-base64"
    data["report_provenance"]["data_encoding"] = encoding
    raw = json_text(data).encode()
    compressed = encoding == "gzip-base64"
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode() if compressed else raw.decode()
    embedded = (f'<script id="data" type="application/{"octet-stream" if compressed else "json"}" '
                f'data-encoding="{encoding}" data-json-bytes="{len(raw)}" '
                f'data-json-sha256="{sha256(raw).hexdigest()}" '
                f'data-history-rows="{sum(len(r.get("history", [])) for r in data["results"])}">{payload}</script>')
    theory = data["documents"].get("THEORY.md", "# Theory\n\nThe theory document has not yet been saved. This is a partial report. The direct steady system is\n\n$$R(U)=0.$$\n")
    plotly = re.sub(r"</script", r"<\\/script", gzip.decompress(PLOTLY.read_bytes()).decode(), flags=re.I)
    license_text = (VENDOR / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8").replace("</script", "<\\/script")
    replacements = {
        "<!--STYLE-->": "<style>" + (ASSETS / "style.css").read_text(encoding="utf-8") + "</style>",
        "<!--PLOTLY-->": "<script>" + plotly + '</script><script type="text/plain" id="plotly-license">' + license_text + "</script>",
        "<!--MATH-->": math_runtime(), "<!--THEORY-->": render_document(theory, prefix="newton-theory")["html"],
        "<!--FINDINGS-->": ('<article class="document rendered-document measured-findings">'
                            +render_document(data["documents"]["RESULTS.md"], prefix="newton-findings")["html"]+"</article>"
                            if "RESULTS.md" in data["documents"] else ""),
        "<!--DATA-->": embedded, "<!--APP-->": "<script>" + (ASSETS / "app.js").read_text(encoding="utf-8") + "</script>",
    }
    page = (ASSETS / "template.html").read_text(encoding="utf-8")
    for token, value in replacements.items():
        page = page.replace(token, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8", newline="\n")
    audit = audit_html(output_path, data, raw)
    audit_file = Path(audit_output) if audit_output else input_path.parent / "report_static_audit.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"Report structural audit failed: {audit}")
    print(f"Built {output_path}: {len(data['results'])} saved results, {audit['bytes']:,} bytes; lossless payload verified")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/campaign.json")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--data-encoding", choices=("plain", "gzip"), default="gzip")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    build(args.input, args.output, args.data_encoding, args.audit_output)


build_report = build


if __name__ == "__main__":
    main()
