"""Render the offline Plotly explorer from saved data; never import a solver."""

import argparse
import gzip
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

from .report_data import NAMES, csv_data, embedded, measured_findings, number, same_flux_comparison

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
SHARED = ROOT / "development/flux_methods/shared/assets"
DEFAULT_INPUT = STUDY / "outputs/cfl_carryover_20261103/summary.json"
COLORS = ["#6a00a8", "#9410a2", "#b52f8c", "#d5536f", "#ed7953", "#fca636"]


def packed_json(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def payload(summary, source):
    """Select plotted columns without resampling any measured trajectory."""
    references = {}
    cases = []
    with ZipFile(source.parent / summary["source_archive"]["file"]) as archive:
        for case in summary["cases"]:
            definition = yaml.safe_load(archive.read(case["path"]))
            cases.append(dict(case, label=definition["label"], description=definition["description"],
                              model=definition["model"], physical=definition))
    rows = []
    history_keys = ("step", "phase", "cfl", "dt", "inner_iterations", "equation_max_ratio",
                    "overall_relative", "boundary_max", "global_defect", "equation_norm")
    for record in summary["results"]:
        assessment = record.get("assessment", {})
        references.setdefault(record["case_id"], assessment.get("reference"))
        row = {key: record.get(key) for key in (
            "id", "case_id", "reconstruction", "cells", "success", "reason", "all_stages_success",
            "fallback_count", "timing", "fine_timing", "kernel_timing", "kernel_metadata",
            "repeat_consistency", "representative_file", "representative_stages", "timing_environment",
            "source_unchanged", "failures")}
        row["method"] = record["method"]["id"]
        row["samples"] = [{key: sample[key] for key in (
            "elapsed_s", "fine_total_s", "coarse_total_s", "success", "total_work", "file", "sha256")}
            for sample in record["samples"]]
        row["assessment"] = {key: value for key, value in assessment.items() if key not in ("state", "reference")}
        state = assessment.get("state", {})
        row["fields"] = dict(x=state.get("x", []), values=state.get("fields", {}))
        row["history"] = [{key: h[key] for key in history_keys if key in h} for h in record.get("fine_history", [])]
        row["stage_timings"] = [dict(cells=n, samples_s=[next(stage["stage_total_s"] for stage in sample["stages"] if stage["cells"] == n)
                                                       for sample in record["samples"]]) for n in record["method"]["grids"]]
        rows.append(row)
    comparisons = same_flux_comparison(summary["results"])
    methods = [dict(method, name=NAMES[method["id"]], color=COLORS[i])
               for i, method in enumerate(summary["protocol"]["methods"])]
    return dict(campaign_id=summary["campaign_id"], cases=cases, methods=methods, rows=rows,
                references=references, comparisons=comparisons,
                findings=measured_findings(summary["results"], comparisons, source.parent),
                protocol=summary["protocol"], provenance=summary["provenance"],
                source_archive=summary["source_archive"], measurement_summary_sha256=sha256(source.read_bytes()).hexdigest(),
                counts=dict(configurations=len(rows), measured_solves=sum(len(r["samples"]) for r in rows),
                            excluded_solves=sum(len(r["warmups"]) for r in summary["results"]),
                            converged=sum(bool(r["success"]) for r in rows),
                            accurate=sum(bool(r["assessment"].get("analytical_passed")) for r in rows)))


def markdown(summary, data):
    c = data["counts"]
    lines = ["# Solver time benchmarking", "",
             f"{c['configurations']} configurations and {c['measured_solves']} measured complete solves. "
             f"{c['converged']} fine-grid configurations converge; {c['accurate']} meet original analytical targets.", "",
             "Open [REPORT.html](REPORT.html) for the Plotly case explorer. Select the model, physical case, "
             "reconstruction and methods. Timing, field and convergence views share the selection.", "",
             "The report is generated from the saved campaign; rendering performs no solves or timing measurements.", ""]
    lines.extend(["Grid sequencing carries the preceding successful solve's last accepted CFL, capped at "
                  "the configured maximum, and recomputes the timestep on each grid. Cold starts and retries "
                  "use the configured initial CFL; residual normalization is fresh on every grid.", ""])
    for paragraph in data["findings"]:
        lines.extend([paragraph, ""])
    lines.extend(["## Cost per correction", "",
                  "Means from twenty synchronized samples. A correction is the fixed twenty-correction block "
                  "divided by twenty. Standalone operations include dispatch and are not additive to a fused correction.", "",
                  "| Case | Reconstruction | Method | Residual [ms] | Jacobian [ms] | Linear [ms] | Direction [ms] | Correction [ms] |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in summary["results"]:
        k = r.get("kernel_timing", {})
        costs = [number(k[name]["mean_s"] * factor) if name in k else "—" for name, factor in
                 (("residual", 1000), ("jacobian_blocks", 1000), ("linear_solve", 1000), ("direction", 1000), ("fixed20", 50))]
        lines.append("| " + " | ".join([r["case_id"], r["reconstruction"], NAMES[r["method"]["id"]], *costs]) + " |")
    lines.extend(["", "## Complete-solve timing", "",
                  "Totals include every coarse solve, prolongation, rejected work and cold retry. "
                  "Production multigrid is 50→100→200→400 grid sequencing.", "",
                  "| Case | Reconstruction | Method | n | Mean [s] | Min [s] | Max [s] | Accuracy |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for r in summary["results"]:
        t = r["timing"]
        lines.append(f"| {r['case_id']} | {r['reconstruction']} | {NAMES[r['method']['id']]} | {len(r['samples'])} | "
                     f"{number(t['mean_s'])} | {number(t['min_s'])} | {number(t['max_s'])} | {r['assessment']['analytical_passed']} |")
    lines.extend(["", "## Single grid versus sequencing", "",
                  "Each row compares the same Stadtke variant. A timing ratio is descriptive when original "
                  "checks or the separate field-agreement check fail.", "",
                  "| Case | Reconstruction | Variant | Single [s] | Sequence [s] | Sequence / single | Original checks | Field agreement |",
                  "|---|---|---|---:|---:|---:|---|---|"])
    for pair in data["comparisons"]:
        lines.append(f"| {pair['case']} | {pair['reconstruction']} | {pair['variant']} | "
                     f"{number(pair['single_mean_s'])} | {number(pair['sequence_mean_s'])} | "
                     f"{number(pair['sequence_over_single_total'])} | {pair['original_checks_passed']} | "
                     f"{pair['comparison_field_agreement_passed']} |")
    lines.extend(["", f"Measured source SHA-256: `{data['provenance']['source']['sha256']}`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    source = parser.parse_args().input.resolve()
    before = sha256(source.read_bytes()).hexdigest()
    summary = json.loads(source.read_text(encoding="utf-8"))
    data = payload(summary, source)
    csv_bytes = csv_data(summary["results"])
    (source.parent / "timings.csv").write_bytes(csv_bytes)
    comparison_bytes = (json.dumps(data["comparisons"], indent=2) + "\n").encode()
    (source.parent / "same_flux_comparison.json").write_bytes(comparison_bytes)
    assets = [Path(__file__), Path(__file__).with_name("report_data.py"), Path(__file__).with_name("audit.py"),
              Path(__file__).with_name("audit_interactions.mjs"),
              STUDY / "README.md", SHARED / "report_theme.css", SHARED / "report_theme.js", *sorted(ASSETS.glob("*"))]
    data["presentation"] = dict(assets={p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in assets if p.is_file()},
                                plotly_version="2.35.2", regenerated_from_saved_results=True)
    plotted = packed_json(data).encode("utf-8")
    package = BytesIO()
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", source.read_bytes())
        archive.writestr("plotted_data.json", plotted)
        archive.writestr("timings.csv", csv_bytes)
        archive.writestr("same_flux_comparison.json", comparison_bytes)
        validation = source.parent / "api_removal_validation.json"
        if validation.is_file():
            archive.write(validation, validation.name)
        for name in ("application_audit.json", "runtime_relocation_audit.json", "browser_audit.json", "numerical_input_manifest.json"):
            evidence = source.parent / "presentation" / name
            if evidence.is_file():
                archive.write(evidence, "checks/" + name)
        measurement_audit = source.parent / "audit.json"
        if measurement_audit.is_file():
            archive.write(measurement_audit, "checks/measurement_audit.json")
        for p in assets:
            if p.is_file():
                archive.write(p, "presentation/" + p.relative_to(ROOT).as_posix())
    downloads = "\n".join((
        embedded("production_results.zip", package.getvalue(), "application/zip"),
        embedded("timings.csv", csv_bytes, "text/csv"),
        embedded("source_and_cases.zip", (source.parent / "source_and_cases.zip").read_bytes(), "application/zip"),
        embedded("PLOTLY_LICENSE.txt", (SHARED / "vendor/PLOTLY_LICENSE.txt").read_bytes(), "text/plain")))
    values = dict(THEME_CSS=(SHARED / "report_theme.css").read_text(encoding="utf-8"),
                  THEME_JS=(SHARED / "report_theme.js").read_text(encoding="utf-8"),
                  STYLE=(ASSETS / "style.css").read_text(encoding="utf-8"),
                  APP=(ASSETS / "app.js").read_text(encoding="utf-8"), PAYLOAD=plotted.decode("utf-8"),
                  PLOTLY=gzip.decompress((SHARED / "vendor/plotly-2.35.2.min.js.gz").read_bytes()).decode("utf-8").replace("</script", "<\\/script"),
                  DOWNLOADS=downloads, CAMPAIGN=escape(data["campaign_id"]))
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m[1]], template)
    (STUDY / "REPORT.html").write_text(html, encoding="utf-8")
    (STUDY / "RESULTS.md").write_text(markdown(summary, data), encoding="utf-8")
    assert sha256(source.read_bytes()).hexdigest() == before, "Measurement summary was modified"
    qa = source.parent / "presentation"
    qa.mkdir(exist_ok=True)
    (qa / "render.json").write_text(json.dumps(dict(summary_sha256=before, html_sha256=sha256(html.encode()).hexdigest(),
        html_bytes=len(html.encode()), payload_sha256=sha256(plotted).hexdigest(), presentation=data["presentation"],
        timings_rerun=False), indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {STUDY / 'REPORT.html'} from {len(data['rows'])} saved configurations; no solves.")


if __name__ == "__main__":
    main()
