"""Build an offline Plotly report from saved NICFD fields, without flow solves."""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ASSETS = Path(__file__).parent / "report_assets"
VENDOR = REPO / "development/flux_methods/shared/assets/vendor"
REGIMES = ("subsonic", "supersonic_shock", "supersonic_adapted")
LABELS = dict(subsonic="Subsonic", supersonic_shock="Supersonic with shock", supersonic_adapted="Supersonic adapted")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_json(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def split_fields(raw):
    return ({key: values[1:-1] for key, values in raw.items() if key != "state"},
            {key: [values[0], values[-1]] for key, values in raw.items() if key != "state"})


def profile_regime(fields, regime):
    if not fields:
        return False
    maximum, outlet = max(fields["Ma_mix"]), fields["Ma_mix"][-1]
    forward = min(fields["u_g"]) > 0 and min(fields["u_l"]) > 0
    branch = (maximum < 1 if regime == "subsonic" else
              maximum > 1 and outlet < 1 if regime == "supersonic_shock" else outlet > 1)
    return bool(forward and branch)


def read_data(input_path=None):
    settings = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
    path = Path(input_path) if input_path else ROOT / settings["campaign"]
    if path.is_dir():
        path = path / "result.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record["arguments"].get("independent_meshes") or record["arguments"]["initialization"] != "reference":
        raise ValueError("This comparison requires independent published-profile initialization on every mesh")
    if any("continuation_preparation" in stage for stage in record["stages"]):
        raise ValueError("Unexpected mesh continuation in the independent-mesh comparison")
    verification_path = ROOT / settings["verification"]
    audits = json.loads(verification_path.read_text(encoding="utf-8")).get("stages", []) if verification_path.exists() else []
    audits = {(a["regime"], a["cells"]): a for a in audits if Path(a["run"]).name == path.parent.name}
    baseline = json.loads((ROOT / "baseline_numerics.json").read_text(encoding="utf-8"))["effective"]
    cases = []
    for regime in REGIMES:
        directory = ROOT / "upstream/data/two_component" / regime
        source = np.genfromtxt(directory / "solution.csv", delimiter=",", names=True)
        raw = {key:source[key].tolist() for key in source.dtype.names}
        fields, endpoints = split_fields(raw)
        cases.append(dict(id=regime, label=LABELS[regime],
            backpressure_bar=dict(subsonic=16., supersonic_shock=12., supersonic_adapted=.986)[regime],
            reference=dict(fields=fields, endpoints=endpoints, total_rows=len(raw["x"]),
                physical_rows=len(fields["x"]), sha256=digest(directory / "solution.csv"),
                settings_yaml=(directory / "settings.yaml").read_text(encoding="utf-8"))))
    stages = {(s["regime"], s["cells"]): s for s in record["stages"]}
    rows = []
    unchanged_keys = ("scheme", "steady", "boundaries", "closures", "compatibility", "stabilization")
    for regime in REGIMES:
        for cells in settings["meshes"]:
            stage = stages.get((regime, cells))
            if stage is None:
                rows.append(dict(id=f"{regime}_N{cells}", regime=regime, cells=cells,
                    success=False, numerical_success=False, verified=False, reason="not_run",
                    fields={}, history=[], errors={}, metrics={}, audit=None))
                continue
            fields, endpoints, raw_state = {}, {}, None
            if "fields" in stage:
                with np.load(path.parent / stage["fields"]) as saved:
                    raw = {key:saved[key].tolist() for key in saved.files}
                fields, endpoints = split_fields(raw)
                raw_state = raw.get("state")
            comparison = {key:stage["effective"][key] == baseline[regime][key] for key in unchanged_keys}
            if not all(comparison.values()):
                raise ValueError(f"Unexpected setting changes for {regime}/N{cells}: {comparison}")
            branch = profile_regime(fields, regime) and stage.get("regime_audit", {}).get("passed", True)
            numerical = bool(stage["success"])
            success = numerical and branch
            audit = audits.get((regime, cells))
            rows.append(dict(id=f"{regime}_N{cells}", regime=regime, cells=cells,
                success=success, numerical_success=numerical, intended_regime=branch,
                verified=bool(success and audit and audit["fresh_case_success"]),
                reason=stage["reason"] if not numerical or branch else "case_regime_mismatch",
                fields=fields, endpoints=endpoints, raw_state=raw_state,
                history=stage.get("history", []), metrics=stage.get("metrics", {}),
                fresh_metrics=stage.get("fresh_metrics", {}), errors=stage.get("errors", {}),
                audit=audit, effective=stage["effective"], unchanged_settings=comparison,
                initialization=stage.get("initialization", stage["effective"]["initialization"]),
                solve_s=stage.get("solve_s"), preparation_s=stage.get("preparation_s"),
                audit_s=stage.get("audit_s"), source_file=str(path.relative_to(ROOT))))
    archive = []
    for old_path in sorted((ROOT / "outputs").glob("run_0[1-8]_*/result.json")):
        old = json.loads(old_path.read_text(encoding="utf-8"))
        for s in old.get("stages", []):
            branch = None
            if "fields" in s:
                with np.load(old_path.parent / s["fields"]) as saved:
                    branch = profile_regime({key:saved[key][1:-1].tolist() for key in ("Ma_mix", "u_g", "u_l")}, s["regime"])
            archive.append(dict(run=old_path.parent.name, regime=s["regime"], cells=s["cells"],
                area_law=s["effective"]["closures"]["area_law"], success=s["success"],
                reason=s["reason"], intended_regime=branch,
                pressure_rmse_bar=s.get("errors", {}).get("p", {}).get("rmse", 0) / 1e5))
    return dict(title="NICFD nitrogen–water mesh comparison", settings=settings, cases=cases,
        results=rows, archive=archive, environment=record["environment"],
        provenance=dict(reference_revision=record["reference_repository_revision"],
            source_hashes=record["source_hashes"], campaign_sha256=digest(path),
            campaign_file=str(path.relative_to(ROOT)), property_preparation_s=record.get("property_preparation_s"),
            baseline_numerics_sha256=digest(ROOT / "baseline_numerics.json"),
            properties=record.get("properties"), numerical_settings_unchanged=all(
                all(r.get("unchanged_settings", {}).values()) for r in rows),
            initialization="Independent published-profile seed on every mesh; no grid sequencing",
            generated_utc=datetime.now(timezone.utc).isoformat()),
        plotly_license=(VENDOR / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8"))


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.external, self.data, self.collect, self.encoding = [], [], [], False, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag in ("script", "link", "img"):
            for attr in ("src", "href"):
                value = attrs.get(attr, "")
                if value and not value.startswith(("data:", "#", "blob:")):
                    self.external.append(value)
        if tag == "script" and attrs.get("id") == "data":
            self.collect, self.encoding = True, attrs["data-encoding"]

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect = False

    def handle_data(self, data):
        if self.collect:
            self.data.append(data)


def audit_html(path, expected=None):
    doc = Document()
    doc.feed(path.read_text(encoding="utf-8"))
    assert not doc.external, doc.external
    assert len(set(doc.ids)) == len(doc.ids)
    required = {"regime", "mesh", "data-source", "field-group", "theme", "solution", "differences",
                "convergence", "methodology", "download-data", "download-fields", "download-history", "data"}
    assert required.issubset(doc.ids), required - set(doc.ids)
    payload = "".join(doc.data)
    raw = gzip.decompress(base64.b64decode(payload)) if doc.encoding == "gzip-base64" else payload.encode()
    if expected is not None:
        assert raw == expected, "Embedded data are not lossless"
    data = json.loads(raw)
    assert data["settings"]["meshes"] == [50, 100, 200, 400]
    assert len(data["results"]) == 12
    assert {r["cells"] for r in data["results"]} == {50, 100, 200, 400}
    return dict(passed=True, external_assets=[], data_roundtrip_identical=True,
                data_bytes=len(raw), html_bytes=path.stat().st_size,
                history_rows=sum(len(r["history"]) for r in data["results"]))


def write_markdown(data):
    lines = ["# NICFD nitrogen–water: 50–400-cell comparison", "",
        "Open [REPORT.html](REPORT.html) for interactive Plotly profiles, published/new/both",
        "visibility controls, mesh/regime selectors, difference plots, histories and downloads.", "",
        "Every new mesh starts independently from the published profile. The physical model,",
        "MUSCL Van Albada reconstruction, Städtke compatibility flux, wb NC pairing, void filter,",
        "coupled backward-Euler solve, 20-correction Newton cap and stopping criteria match",
        "the previous fine-grid comparison. Only the mesh count changes.", "",
        "| Regime | Physical cells | Result | Pressure RMSE [bar] | Gas / liquid velocity RMSE [m/s] | Accepted / rejected |",
        "|---|---:|---|---:|---:|---:|"]
    for row in data["results"]:
        accepted = sum(h.get("phase") == "accepted" for h in row["history"])
        rejected = sum(h.get("phase") == "rejected" for h in row["history"])
        e = row["errors"]
        pressure = f"{e['p']['rmse']/1e5:.6g}" if "p" in e else "—"
        velocity = f"{e['u_g']['rmse']:.4g} / {e['u_l']['rmse']:.4g}" if "u_g" in e else "—"
        status = "verified" if row["verified"] else row["reason"]
        lines.append(f"| {row['regime']} | {row['cells']} | {status} | {pressure} | {velocity} | {accepted} / {rejected} |")
    lines += ["", "The published reference has 500 total rows, including two endpoint ghosts. Its",
        "498 interior samples are comparison data; new meshes contain the stated number of",
        "physical cells plus two endpoint traces. Pressure errors compare point profiles",
        "using bounded linear interpolation, not exact cell averages. Errors of unfinished",
        "states are retained as diagnostics. No grid-independence claim is made.", "",
        "Previous investigation records and the former 498-cell report remain in `outputs/`.",
        "The original cold-to-fine subsonic test found a numerically converged root with a",
        "supersonic pocket. Independent cold-start recovery of the published branches remains",
        "unverified; this mesh comparison uses explicit published-profile seeding.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "REPORT.html")
    parser.add_argument("--data-encoding", choices=("gzip", "plain"), default="gzip")
    args = parser.parse_args()
    data = read_data(args.input)
    data["provenance"].update(plotly_version="2.35.2", plotly_sha256=digest(VENDOR / "plotly-2.35.2.min.js.gz"),
        report_generator_sha256=digest(Path(__file__)),
        report_assets_sha256={p.name:digest(p) for p in ASSETS.iterdir() if p.is_file()})
    raw = encoded_json(data).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode() if args.data_encoding == "gzip" else raw.decode()
    encoding = "gzip-base64" if args.data_encoding == "gzip" else "plain-json"
    embedded = f'<script id="data" type="application/octet-stream" data-encoding="{encoding}" data-json-bytes="{len(raw)}" data-json-sha256="{hashlib.sha256(raw).hexdigest()}">{payload}</script>'
    plotly = gzip.decompress((VENDOR / "plotly-2.35.2.min.js.gz").read_bytes()).decode("utf-8")
    plotly = re.sub(r"</script", r"<\\/script", plotly, flags=re.I)
    page = ((ASSETS / "template.html").read_text(encoding="utf-8")
        .replace("<!--STYLE-->", "<style>" + (ASSETS / "style.css").read_text(encoding="utf-8") + "</style>")
        .replace("<!--PLOTLY-->", "<script>" + plotly + "</script>")
        .replace("<!--DATA-->", embedded)
        .replace("<!--APP-->", "<script>" + (ASSETS / "app.js").read_text(encoding="utf-8") + "</script>"))
    args.output.write_text(page, encoding="utf-8")
    audit = audit_html(args.output, raw)
    output = ROOT / "outputs/plotly_report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "static_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "data.json").write_bytes(raw)
    write_markdown(data)
    print(f"Built {args.output}: {len(data['results'])} mesh cases; {audit['html_bytes']:,} bytes; offline data roundtrip passed.")


if __name__ == "__main__":
    main()
