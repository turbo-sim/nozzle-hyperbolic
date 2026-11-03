"""Build finished, standalone case reports from saved NICFD results only."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ASSETS = Path(__file__).parent / "report_assets"
VENDOR = REPO / "development/flux_methods/shared/assets/vendor"
FAMILIES = {
    "single_component": "Single-component flashing water",
    "two_component": "Two-component nitrogen–water",
}
REGIMES = {
    "subsonic": "Subsonic",
    "supersonic_shock": "Supersonic with shock",
    "supersonic_adapted": "Supersonic adapted",
}
EQUATIONS = [
    "Gas mass",
    "Liquid mass",
    "Gas momentum",
    "Liquid momentum",
    "Gas energy",
    "Liquid energy",
]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_safe(value):
    return (
        json.dumps(
            clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=3):
    return (
        "Not available"
        if value is None or not np.isfinite(value)
        else f"{value:.{digits}g}"
    )


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(
                (
                    f'<td class="wrap">{esc(v)}</td>'
                    if len(str(v)) > 80
                    else f"<td>{esc(v)}</td>"
                )
                for v in row
            )
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def load_case(family, regime):
    directory = ROOT / family / regime
    path = directory / "outputs/result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    property_path = directory / "outputs/property_audit.json"
    data["property_audit"] = json.loads(property_path.read_text(encoding="utf-8"))
    if "verdict" not in data:
        raise ValueError(f"Case execution has not finished: {path}")
    if not data["property_audit"]["usable_on_published_profile"]:
        data["verdict"]["detail"] = data["property_audit"]["explanation"]
    old = np.genfromtxt(directory / "reference.csv", delimiter=",", names=True)
    data.update(
        kind="case",
        equation_names=EQUATIONS,
        reference={key: old[key][1:-1].tolist() for key in old.dtype.names},
        reference_endpoints={
            key: old[key][[0, -1]].tolist() for key in old.dtype.names
        },
        published_yaml=(directory / "settings.yaml").read_text(encoding="utf-8"),
        reference_sha256=hashlib.sha256(
            (directory / "reference.csv").read_bytes()
        ).hexdigest(),
        source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_result=str(path.relative_to(ROOT)),
    )
    for run in data["runs"]:
        attempted = [
            stage
            for stage in run["stages"]
            if stage.get("reason") != "not_attempted_after_coarse_failure"
        ]
        run["measured_state_replay_valid"] = bool(
            run.get("warm_replay_valid")
            and attempted
            and all(stage.get("cold_warm_agree", False) for stage in attempted)
        )
        for stage in run["stages"]:
            if "fields_file" not in stage:
                continue
            fields_path = directory / "outputs" / stage["fields_file"]
            with np.load(fields_path, allow_pickle=False) as saved:
                stage["fields"] = {
                    key: saved[key][1:-1].tolist()
                    for key in saved.files
                    if key != "state"
                }
                stage["trace_fields"] = {
                    key: saved[key][[0, -1]].tolist()
                    for key in saved.files
                    if key != "state"
                }
                stage["raw_state"] = saved["state"].tolist()
            stage["fields_sha256"] = hashlib.sha256(
                fields_path.read_bytes()
            ).hexdigest()
    return data


def reported_stage(data):
    return data["runs"][-1]["stages"][-1] if data["runs"] else {}


def content(data):
    family, regime, verdict = data["family"], data["regime"], data["verdict"]
    final, config, protocol = (
        reported_stage(data),
        data["specification"],
        data["protocol"],
    )
    errors = final.get("errors", {})
    pressure = errors.get("p", {}).get("normalized_rmse")
    mass = errors.get("mass_flow", {}).get("relative_difference")
    cold = next((run for run in data["runs"] if run["mode"] == "cold"), {})
    warm = next((run for run in data["runs"] if run["mode"] == "warm"), {})
    start = f"""<div class="eyebrow">{esc(FAMILIES[family])} · published-profile reproduction</div>
<h1>{esc(REGIMES[regime])}</h1><section class="verdict {esc(verdict['status'])}" id="overall-verdict">
<h2>{esc(verdict['title'])}</h2><p>{esc(verdict['detail'])}</p></section>
<p class="muted">The overall verdict uses the final requested {protocol['meshes'][-1]}-cell attempt in the replayed execution. Numerical convergence, the intended regime and agreement with the published profiles are separate checks. “Cold” means compilation-inclusive execution; every first-grid field is explicitly seeded from the published numerical solution.</p>
<div class="metrics"><div class="metric"><span>Pressure normalized RMSE</span><strong>{'Not available' if pressure is None else fmt(pressure*100)+'%'}</strong><small>Scale: reservoir pressure</small></div>
<div class="metric"><span>Mean mass-flow difference</span><strong>{'Not available' if mass is None else fmt(mass*100)+'%'}</strong><small>Relative to the published interior profile</small></div>
<div class="metric"><span>Compilation-inclusive total</span><strong>{fmt(cold.get('total_s'),4)} s</strong><small>Preparation + all attempted mesh solves</small></div>
<div class="metric"><span>Replay total</span><strong>{fmt(warm.get('total_s'),4)} s</strong><small>{'Zero new compilation events; states agree' if warm.get('measured_state_replay_valid') else 'No verified replay comparison; see stage outcomes'}</small></div></div>
<nav class="tabs" role="tablist" aria-label="Report sections"><button data-page="profiles" role="tab" aria-selected="true">Profiles & differences</button><button data-page="convergence" role="tab" aria-selected="false">Convergence & work</button><button data-page="timings" role="tab" aria-selected="false">Timings & mesh outcomes</button><button data-page="method" role="tab" aria-selected="false">Method & evidence</button></nav>
<section class="panel controls"><label>Execution<select id="mode"></select></label><label>Physical cells<select id="grid"></select></label><label>Profile visibility<select id="visibility"><option value="both">Published + current</option><option value="published">Published only</option><option value="current">Current only</option></select></label><label>Variables<select id="field-group"><option value="flow">Flow variables</option><option value="thermal">Thermodynamic variables</option><option value="mixture">Composition & acoustic variables</option></select></label><label>Points skipped between markers<input id="skip" type="number" min="0" max="999" value="3"></label></section>
<p class="note" id="selected-note"></p><div class="actions"><button id="download-profile">Current profile CSV</button><button id="download-reference">Published profile CSV</button><button id="download-history">Selected history CSV</button><button id="download-json">Full case data JSON</button></div>
<section class="page" id="profiles"><h2>Geometry and profile comparison</h2><p class="small muted">Published curves are continuous lines. Current results use white-face markers. Statistics exclude both endpoint traces. Every current cell enters the absolute-difference curves, regardless of marker skipping; zero differences remain zero in downloads and are omitted on logarithmic axes.</p><div class="chart geometry" id="geometry"></div><div class="plots" id="profile-plots"></div></section>
<section class="page" id="convergence" hidden><h2>Original spatial residuals and actual work</h2><p class="small muted">Each integrated spatial row is normalized as <code>Rᵢ t_ref / (Vᵢ Q_ref)</code>, where <code>t_ref = L_ref / U_ref</code>; the reported per-equation L2 uses physical-volume weights. Cell EOS and endpoint equations have separate fixed scales and absolute infinity-norm checks. Rejected attempts contribute to the CFL and Newton-work plots. A small Newton update is not used as proof of outer convergence. Effective stopping targets are included under Method & evidence.</p><p class="note" id="history-note" hidden></p><label>History coordinate <select id="history-axis"><option value="steps">Accepted steps</option><option value="time">Solve time</option><option value="work">Actual Newton corrections</option></select></label><div class="plots"><div class="chart" id="residual"></div><div class="chart" id="defects"></div><div class="chart" id="cfl"></div><div class="chart" id="work"></div></div>"""
    audit_rows = []
    for run in data["runs"]:
        for stage in run["stages"]:
            audit = stage.get("audit", {})
            audit_rows.append(
                [
                    run["mode"],
                    stage["cells"],
                    stage.get("reason"),
                    stage.get("case_success", False),
                    fmt(max(audit.get("equation_norm", [float("nan")]))),
                    fmt(audit.get("constraint_max")),
                    fmt(audit.get("boundary_max")),
                    fmt(audit.get("global_max")),
                    fmt(audit.get("invariant_max")),
                ]
            )
    start += table(
        [
            "Execution",
            "Cells",
            "Termination",
            "Case checks pass",
            "Max equation norm",
            "Cell EOS",
            "Boundary",
            "Global selected",
            "Total-mass invariant",
        ],
        audit_rows,
    )
    if family == "single_component":
        start += '<p class="note">The published single-component configuration enables the saturation-energy pin. This intentionally changes selected energy equations. The independent audit also retains unmodified physical equation and global defects in the downloadable data; passing selected pinned equations is not a claim that unmodified energy balances vanish.</p>'
    start += (
        '</section><section class="page" id="timings" hidden><h2>What was timed</h2><p>'
        + esc(protocol["timing_note"])
        + '</p><p class="note">Old-code runtime and convergence histories are not supplied with the published CSVs. They are unavailable here, rather than estimated or shown as zero. No old/new speedup is claimed.</p>'
    )
    stage_rows = []
    for run in data["runs"]:
        for stage in run["stages"]:
            history = stage.get("history", [])
            trials = [row for row in history if row.get("phase") != "initial"]
            stage_rows.append(
                [
                    run["mode"],
                    stage["cells"],
                    stage.get("initialization", "Not attempted"),
                    stage.get("case_success", False),
                    fmt(stage.get("preparation_s"), 5),
                    fmt(stage.get("solve_s"), 5),
                    fmt(stage.get("total_s"), 5),
                    sum(row.get("accepted", False) for row in trials),
                    sum(not row.get("accepted", False) for row in trials),
                    sum(row.get("inner_iterations", 0) for row in trials),
                    stage.get("compilation_events", 0),
                ]
            )
    start += table(
        [
            "Execution",
            "Cells",
            "Initialization",
            "Case checks pass",
            "Preparation [s]",
            "Solve [s]",
            "Total [s]",
            "Accepted",
            "Rejected",
            "Newton corrections",
            "Compilation events",
        ],
        stage_rows,
    )
    start += table(
        ["Execution", "Total [s]", "All mesh stages pass", "Warm replay eligible"],
        [
            [
                run["mode"],
                fmt(run["total_s"], 6),
                run["complete"],
                (
                    run.get("measured_state_replay_valid", False)
                    if run["mode"] == "warm"
                    else "Compilation included"
                ),
            ]
            for run in data["runs"]
        ],
    )
    start += (
        '<p class="small muted">Property-table loading/preparation: '
        + fmt(data.get("property_preparation_s"), 5)
        + " s, outside these totals. Property-cache generation, process startup, field extraction, independent final audits and output are excluded. Totals retain all failed coarse work before any independent finest-grid check. These single observations are not repeated-sample performance distributions.</p>"
    )
    start += "<h2>Declared agreement criteria</h2>"
    thresholds = protocol["agreement"]
    start += table(
        ["Quantity", "Criterion"],
        [
            [
                "Pressure normalized RMSE",
                f"≤ {thresholds['pressure_normalized_rmse']*100:g}% of reservoir pressure",
            ],
            [
                "Each phase velocity normalized RMSE",
                f"≤ {thresholds['phase_velocity_normalized_rmse']*100:g}% of published peak speed",
            ],
            ["Void-fraction RMSE", f"≤ {thresholds['void_absolute_rmse']:g} absolute"],
            [
                "Each phase temperature normalized RMSE",
                f"≤ {thresholds['temperature_normalized_rmse']*100:g}% of reservoir temperature",
            ],
            [
                "Mean mass flow",
                f"≤ {thresholds['mass_flow_relative_difference']*100:g}% relative difference",
            ],
        ],
    )
    start += '<p class="small muted">' + esc(thresholds["note"]) + "</p>"
    error_rows = [
        [
            key,
            fmt(value.get("rmse"), 6),
            fmt(value.get("max_absolute"), 6),
            fmt(value.get("normalized_rmse", 0) * 100, 5),
        ]
        for key, value in errors.items()
        if isinstance(value, dict) and "rmse" in value
    ]
    start += (
        table(
            [
                "Reported finest field",
                "RMSE [SI]",
                "Maximum difference [SI]",
                "Normalized RMSE [%]",
            ],
            error_rows,
        )
        + "</section>"
    )
    start += '<section class="page" id="method" hidden><h2>Physical case and numerical mapping</h2>'
    start += table(
        ["Input", "Value"],
        [
            ["Family", FAMILIES[family]],
            ["Geometry", config["geometry"]["function"]],
            ["Reservoir pressure [bar]", config["bc"]["p0"] / 1e5],
            [
                "Reservoir gas / liquid temperature [K]",
                f"{config['bc']['T_g0']} / {config['bc']['T_l0']}",
            ],
            ["Backpressure [bar]", config["bc"]["p_out"] / 1e5],
            [
                "Published total / physical rows",
                f"{config['grid']['Nx']} / {config['grid']['Nx']-2}",
            ],
            ["Current physical cells", ", ".join(map(str, protocol["meshes"]))],
            ["Mass transfer", config["sources"]["mass_transfer"]],
            [
                "Initial composition",
                (
                    f"Gas mass-flow fraction {config['bc']['quality']}"
                    if "quality" in config["bc"]
                    else f"Vapour void fraction {config['bc']['void_fraction']}"
                ),
            ],
            [
                "Solver-used source unchanged during execution",
                data["property_audit"]["source_audit"]["solver_sources_unchanged"],
            ],
        ],
    )
    start += '<p class="note">This is a reference-seeded numerical reproduction, not a cold physical-start robustness test. Successful grids transfer transported densities with EOS recovery; failed states are never used for continuation. A failed coarse stage triggers an explicitly labelled independent 400-cell published-profile start. Resource-limit exits remain failed outcomes.</p>'
    if not data.get("source_unchanged"):
        start += (
            '<p class="note">The broad repository snapshot changed during this execution in files outside this solver’s import graph: '
            + esc(
                ", ".join(
                    data["property_audit"]["source_audit"][
                        "changed_files_outside_solver_import_graph"
                    ]
                )
            )
            + ". The solver-used source hashes remained unchanged. Both endpoint snapshots and the import-graph audit are retained in the downloadable data.</p>"
        )
    start += "<p>Published and current meshes use different interior locations and geometry integration. Profile errors use arclength-weighted point comparisons with bounded interpolation of published interior samples. They do not compare exact finite-volume averages. The frozen published fields use JAXProp 0.5.6; the actual installed versions and validated property definitions are recorded below.</p>"
    differences = final.get("effective", {}).get("differences", [])
    start += (
        "<ul>"
        + "".join("<li>" + esc(value) + "</li>" for value in differences)
        + "</ul>"
    )
    start += "<p>Current later pressure-work corrections remain active. The upstream supplementary repository supplies solutions and settings, not its executable solver or runtime logs. This report compares those published outputs with the present implementation; it does not claim bitwise equivalence of the two algorithms.</p>"
    start += table(
        ["Aspect", "Publication-era setup / saved output", "Current implementation"],
        [
            [
                "Physical equations",
                "Six non-equilibrium phase balances; virtual mass, drag and heat/mass exchange",
                "Eight constrained unknowns: six balances plus two EOS equations; later pressure-work corrections retained",
            ],
            [
                "Closures",
                "Number-density interfacial-area law from September e185f571; flashing pins enabled only for water",
                "Same explicit number-density law, phase settings and pin selection; no substitution of the later polynomial default",
            ],
            [
                "Reservoir",
                "FluidJAX PT; ghost vapor at saturation when subcooled PT is below vapor-table floor",
                "Same physical reservoir rule; coupled algebraic inlet equations replace fixed-point/isentrope lookup completion",
            ],
            [
                "Spatial method",
                "Compatibility FVS, well-balanced nonconservative form, regularized global Van Albada formula",
                "Compatibility FVS and well-balanced form with generic local MUSCL Van Albada; integrated geometry and shared mesh convention",
            ],
            [
                "Nonlinear step",
                "Normal-equation Newton with adaptive column scales; published cap 7",
                "Square banded constrained Newton, fixed physical scales, backtracking; cap 20",
            ],
            [
                "Continuation / stopping",
                "Saved solutions only; original restart identities and residual histories absent",
                "100 → 200 → 400 cells, successful transfers only; independent final attempt after coarse failure; original selected residual, EOS, BC, balance, mass and regime gates",
            ],
            [
                "Properties",
                "Published settings; JAXProp 0.5.6 stated by upstream",
                "Configured bicubic/metastable tables with installed dependency version and source/table hashes; invalid stencils rejected",
            ],
            [
                "Timing evidence",
                "Not supplied",
                "One compilation-inclusive hierarchy and identical replay; all failed work retained; no old/new speedup claim",
            ],
        ],
    )
    diagnostic = data["property_audit"]
    start += (
        "<h2>Independent property-table diagnostic</h2><p>"
        + esc(diagnostic["explanation"])
        + "</p>"
    )
    start += table(
        [
            "Phase",
            "Published rows rejected by stencil audit",
            "Current EOS temperature range [K]",
            "Published temperature range [K]",
            "Temperature RMSE [K]",
        ],
        [
            [
                phase,
                f"{item['invalid_stencil_rows']} / {item['reference_rows']}",
                " – ".join(fmt(v, 6) for v in item["temperature_range_K"]),
                " – ".join(fmt(v, 6) for v in item["reference_temperature_range_K"]),
                fmt(item["temperature_rmse_K"], 6),
            ]
            for phase, item in diagnostic["phases"].items()
        ],
    )
    for label, value in [
        ("Unchanged published settings.yaml", data["published_yaml"]),
        ("Effective current solver settings", final.get("effective", {})),
        (
            "Independent final audit, including unmodified physical defects",
            final.get("audit", {}),
        ),
        (
            "Environment and property definitions",
            dict(
                environment=data["environment"],
                properties=data.get("properties"),
                resolution=data.get("property_resolution"),
            ),
        ),
        ("Property audit and dependency source identity", diagnostic),
        (
            "Failed attempt details",
            [
                dict(mode=run["mode"], cells=stage["cells"], error=stage["error"])
                for run in data["runs"]
                for stage in run["stages"]
                if "error" in stage
            ],
        ),
        (
            "Source and data provenance",
            dict(
                upstream_revision=data["upstream_revision"],
                reference_sha256=data["reference_sha256"],
                source_result_sha256=data["source_result_sha256"],
                numerical_source_hashes=data["source_before"],
                source_unchanged=data.get("source_unchanged"),
            ),
        ),
    ]:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, indent=2, ensure_ascii=False)
        )
        start += (
            f"<details><summary>{esc(label)}</summary><pre>{esc(text)}</pre></details>"
        )
    start += "</section>"
    return start


def render_document(title, body, data, destination, runtime):
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    substitutions = dict(
        TITLE=esc(title),
        CSS=(ASSETS / "style.css").read_text(encoding="utf-8"),
        CONTENT=body,
        DATA=json_safe(data),
        PLOTLY=runtime,
        APP=(ASSETS / "app.js").read_text(encoding="utf-8"),
    )
    for key, value in substitutions.items():
        template = template.replace("@@" + key + "@@", value)
    destination.write_text(template, encoding="utf-8", newline="\n")
    assert all("@@" + key + "@@" not in template for key in substitutions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    runtime = (
        gzip.decompress((VENDOR / "plotly-2.35.2.min.js.gz").read_bytes())
        .decode("utf-8")
        .replace("</script", "<\\/script")
    )
    reports = []
    for family, family_name in FAMILIES.items():
        for regime, label in REGIMES.items():
            data = clean(load_case(family, regime))
            destination = ROOT / family / regime / "REPORT.html"
            render_document(
                family_name + " · " + label, content(data), data, destination, runtime
            )
            reports.append(
                dict(
                    family=family,
                    regime=regime,
                    verdict=data["verdict"],
                    path=str(destination.relative_to(ROOT)).replace("\\", "/"),
                    current_field_available="fields" in reported_stage(data),
                    cold_s=next(
                        (
                            run["total_s"]
                            for run in data["runs"]
                            if run["mode"] == "cold"
                        ),
                        None,
                    ),
                    replay_s=next(
                        (
                            run["total_s"]
                            for run in data["runs"]
                            if run["mode"] == "warm"
                        ),
                        None,
                    ),
                )
            )
            print("Wrote", destination, flush=True)
    body = '<div class="eyebrow">Six published cases · two physical families</div><h1>NICFD 2026 reproduction</h1><p class="muted">Every case report begins with a verdict based on independent convergence checks, the intended flow regime and declared profile-error criteria. Published solutions are the old-code reference; current-solver timings are newly measured.</p>'
    for family, label in FAMILIES.items():
        body += "<h2>" + esc(label) + '</h2><div class="cards">'
        for report in (r for r in reports if r["family"] == family):
            verdict = report["verdict"]
            timing = (
                f"Current hierarchy: {fmt(report['cold_s'],4)} s including compilation; {fmt(report['replay_s'],4)} s replay."
                if report["current_field_available"]
                else "No current CFD field: property-table preparation was rejected."
            )
            body += f'<article class="case-card"><span class="badge {esc(verdict["status"])}">{esc(verdict["status"].replace("_"," "))}</span><h3>{esc(REGIMES[report["regime"]])}</h3><p>{esc(verdict["title"])}</p><p>{esc(verdict["detail"])}</p><p>{esc(timing)}</p><a href="{esc(report["path"])}">Open complete case report →</a></article>'
        body += "</div>"
    render_document(
        "All six cases",
        body,
        dict(kind="index", cases=reports),
        ROOT / "REPORT.html",
        "",
    )
    (ROOT / "report_manifest.json").write_text(
        json.dumps(
            dict(
                generated_utc=datetime.now(timezone.utc).isoformat(),
                cases=reports,
                reports={
                    r["path"]: hashlib.sha256(
                        (ROOT / r["path"]).read_bytes()
                    ).hexdigest()
                    for r in reports
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
