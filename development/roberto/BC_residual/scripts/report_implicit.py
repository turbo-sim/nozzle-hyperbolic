"""Build one standalone HTML from saved records and retained background notes.

This path imports no numerical solver and does not rerun historical timings.
"""

import argparse
import csv
import gzip
import hashlib
import html
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import statistics

import markdown

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
LABELS = {"production": "Source: internal BCs", "coupled": "Coupled BC equations"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_safe(value):
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )


def script_safe(value):
    return re.sub(r"</script", r"<\/script", value, flags=re.I)


def document(name, prefix):
    """Typeset preserved prose/math, embedding images and resolving local links."""
    source = (ASSETS / "notes" / name).read_text(encoding="utf-8")
    math = []

    def equation(match, block=False):
        key = f"LATEXPLACEHOLDER{len(math)}END"
        tag, cls, delimiters = (
            ("div", "math-display", "$$") if block else ("span", "math-inline", "$")
        )
        math.append(
            (
                key,
                f'<{tag} class="{cls}">{delimiters}{html.escape(match[1])}{delimiters}</{tag}>',
            )
        )
        return f"\n\n{key}\n\n" if block else key

    protected = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: equation(m, True), source)
    protected = re.sub(r"(?<![\\$])\$(?!\$)([^\n$]+?)\$(?!\$)", equation, protected)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = re.sub(
        r"^<h1[^>]*>.*?</h1>\s*", "", md.convert(protected), count=1, flags=re.S
    )
    for key, value in math:
        body = body.replace(f"<p>{key}</p>", value).replace(key, value)
    ids = set(re.findall(r'\bid="([^"]+)"', body))
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}-{m[1]}"', body)

    def link(match):
        href, label = html.unescape(match[1]), match[2]
        if href.startswith(("https://", "http://")):
            return match[0]
        if href.startswith("#") and href[1:] in ids:
            return f'<a href="#{prefix}-{href[1:]}">{label}</a>'
        tab = {
            "BC_IMPLEMENTATION.md": "theory",
            "BOUNDARY_DESIGN_RECOMMENDATIONS.md": "recommendations",
            "IMPLICIT_REPORT.md": "results",
            "README.md": "results",
        }.get(Path(href).name)
        if tab:
            return f'<a href="#{tab}">{label}</a>'
        return f'<span title="Repository provenance: {html.escape(href, quote=True)}">{label} <code>(repository artifact)</code></span>'

    body = re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>', link, body, flags=re.S)

    def image(match):
        path = (STUDY / html.unescape(match[1])).resolve()
        if path.is_file() and path.suffix.lower() == ".png":
            import base64

            return (
                '<img class="archive-image" alt="Retained historical figure" src="data:image/png;base64,'
                + base64.b64encode(path.read_bytes()).decode()
                + '">'
            )
        return (
            '<p class="small">Historical figure path: <code>'
            + html.escape(match[1])
            + "</code>. See the interactive results above.</p>"
        )

    body = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', image, body)
    headings = [
        (target, re.sub("<.*?>", "", label))
        for target, label in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    ]
    return body, headings


def saved_record_audit(data, summary_hash):
    """Check saved arithmetic and selection; this is not a fresh equation audit."""
    rows = []
    for case in data["cases"]:
        for mode in data["methods"]:
            measured = [r for r in case["measured_runs"] if r["mode"] == mode]
            excluded = [r for r in case["excluded_runs"] if r["mode"] == mode]
            assert len(excluded) == 1 and len(measured) == data["settings"]["runs"]
            times = [r["solve_s"] for r in measured]
            stats = case["timings"][mode]
            for key, actual in (
                ("min_s", min(times)),
                ("max_s", max(times)),
                ("mean_s", statistics.mean(times)),
            ):
                assert abs(actual - stats[key]) <= 1e-12
            assert stats["all_success"] == all(r["success"] for r in measured)
            representative = next(r for r in case["runs"] if r["mode"] == mode)
            assert (
                representative
                == sorted(measured, key=lambda r: r["solve_s"])[len(measured) // 2]
            )
            for run in measured + excluded:
                assert len(run["history"]) == run["accepted_updates"] + 1
                assert (
                    sum(t["accepted"] for t in run["trials"]) == run["accepted_updates"]
                )
                assert (
                    sum(not t["accepted"] for t in run["trials"])
                    == run["rejected_trials"]
                )
                for item in run["history"]:
                    assert item["residual_rms"] == max(
                        item["spatial_rms"], item["boundary_rms"]
                    )
            rows.append(
                dict(
                    case=case["id"],
                    mode=mode,
                    count=len(measured),
                    all_reported_roots=stats["all_success"],
                )
            )
    return dict(
        passed=True,
        summary_sha256=summary_hash,
        scope="Saved-record consistency only; no fresh equation/source audit or timing runs",
        runs=rows,
    )


def timing_csv(data):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "case",
            "method",
            "measurement",
            "order",
            "solve_s",
            "success",
            "analytic_pass",
            "updates",
            "rejections",
        ]
    )
    for case in data["cases"]:
        for run in case["measured_runs"]:
            writer.writerow(
                [
                    case["id"],
                    run["mode"],
                    run["measurement"],
                    run["order"],
                    run["solve_s"],
                    run["success"],
                    run["diagnostics"]["analytical_pass"],
                    run["accepted_updates"],
                    run["rejected_trials"],
                ]
            )
    return buffer.getvalue()


def table(headings, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join("<th>" + h + "</th>" for h in headings)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def results_body(data, audit):
    measured = [r for c in data["cases"] for r in c["measured_runs"]]
    excluded = [r for c in data["cases"] for r in c["excluded_runs"]]
    root_count = sum(r["success"] for r in measured)
    analytic_count = sum(r["diagnostics"]["analytical_pass"] for r in measured)
    time_rows, final_rows, field_rows = [], [], []
    for c in data["cases"]:
        for mode in data["methods"]:
            t = c["timings"][mode]
            time_rows.append(
                [
                    html.escape(c["name"]),
                    LABELS[mode],
                    t["count"],
                    f'{t["min_s"]:.4f}',
                    f'{t["max_s"]:.4f}',
                    f'{t["mean_s"]:.4f}',
                    str(t["all_success"]),
                ]
            )
        for r in c["runs"]:
            h = r["history"][-1]
            final_rows.append(
                [
                    c["id"],
                    LABELS[r["mode"]],
                    r["accepted_updates"],
                    r["rejected_trials"],
                    f'{h["residual_rms"]:.3e}',
                    f'{h["boundary_rms"]:.3e}',
                    "Pass" if r["diagnostics"]["analytical_pass"] else "Fail",
                ]
            )
        ratio = c["timings"]["coupled"]["mean_s"] / c["timings"]["production"]["mean_s"]
        field_rows.append(
            [
                c["id"],
                f"{ratio:.3f}",
                f'{c["maximum_pressure_difference_Pa"]:.3e}',
                f'{c["scaled_state_difference_rms"]:.3e}',
            ]
        )

    def panels(kind):
        return (
            '<div class="five-grid">'
            + "".join(
                f'<div class="panel"><div class="panel-head"><h3>{html.escape(c["name"])} &middot; p<sub>b</sub>/p<sub>0</sub> = {c["back_pressure_ratio"]:.6f}</h3></div><div class="plot" id="{kind}-{i}"></div></div>'
                for i, c in enumerate(data["cases"])
            )
            + "</div>"
        )

    body = f"""<div class="hero"><div class="eyebrow">EulerNozzle / matched implicit boundary formulations</div><h1>Internal boundary maps or coupled residual equations?</h1><p>Two representations of the same physical boundary laws, compared with matched implicit Euler, JAX Newton, colored AD and banded linear algebra. The retained benchmark covers five nozzle regimes and ten measured cold-start solves per treatment and case.</p><div class="hero-meta"><span>{data['settings']['points']} points</span><span>{len(measured)} measured solves + {len(excluded)} excluded solves</span><span>Perfect-gas &middot; corrected FVS &middot; ordinary Van Leer</span><span>Historical evidence, presentation rebuilt</span></div><p class="status-line" id="report-status">Loading saved numerical evidence...</p></div>
<section id="results-overview"><span class="section-index">01 / FINDINGS</span><h2>Equivalent trajectories, a modest coupled cost</h2><div class="finding"><strong>{root_count}/{len(measured)} measured solves</strong> satisfy the recorded root checks. <strong>{analytic_count}/{len(measured)}</strong> pass the separate analytical check: both methods retain the sonic-limit discrepancy. Coupled/source mean-time ratios are about 1.045&ndash;1.066 on this machine.</div><p>The representative histories and fields agree to very small differences. This does not establish universally greater robustness; it shows preservation of the tested Euler solutions when the local boundary map is regular.</p><p class="note">These timings belong to the original matched benchmark with a read-only closure adapter. The current adapter later migrated to public physical-residual and solver-metadata operators, checked by a small compatibility smoke. The 100 timing samples were <strong>not rerun</strong>. This reporting change runs no numerical solver.</p></section>
<section id="results-setup"><span class="section-index">02 / SETUP</span><h2>Five regimes and a controlled comparison</h2><p>Perfect-gas air flows through the shared symmetric 0.2 m nozzle, exit/throat area ratio two, with a 200 kPa, 300 K reservoir. Cases cover supersonic discharge, shocks at 35% and 70% of the divergent section, the sonic-throat/subsonic-exit limit, and fully subsonic flow. Analytical solutions are checks, never initial guesses. Each repeat restores the same cold full field and fresh controller state.</p><p>Internal treatment has 597 unknowns on the default mesh. Coupled treatment adds six ghost unknowns and six algebraic equations, with zero ghost storage. Exact coupled inlet constraints replace three finite thermodynamic inversion updates; equivalence is restricted to the admissible, unclipped branch.</p>"""
    body += table(
        [
            "Treatment",
            "CFL / maximum",
            "SER exponent / growth",
            "Newton cap / relative tolerance",
            "Steady stop",
        ],
        [
            [
                LABELS[m],
                "20 / 500",
                "1 / 1.2",
                "8 / 1e-8",
                "max(spatial RMS, boundary RMS) &le; 1e-8",
            ]
            for m in data["methods"]
        ],
    )
    body += '<p class="small">These are the documented measured controls, not values inferred from current defaults. Eight corrections are a cap, not proof of exact nonlinear timestep convergence. The coupled relative-update check additionally includes ghosts. Both paths retain the same retry/increment policy and monitor.</p></section>'
    body += (
        '<section id="results-convergence"><span class="section-index">03 / CONVERGENCE</span><h2>Actual residual histories at five backpressures</h2><div class="controls"><button id="reset-plots">Reset plots</button><span class="small">Drag to zoom; click a legend to hide a line. Each history is an upper median-duration measured run, never an averaged curve.</span></div>'
        + panels("residual")
        + '<p class="small">Residual = max(fixed-scale spatial RMS, dimensionless boundary RMS). Rejected trials are counted separately. Theory explains why reduced and coupled Jacobians can generate matching trajectories.</p></section>'
    )
    body += (
        '<section id="results-pressure"><span class="section-index">04 / FIELDS</span><h2>Final pressure against analytical references</h2>'
        + panels("pressure")
        + '<p class="small">Ghost cells are excluded. Fields and histories use the same representative runs. The sonic-limit analytical mismatch is preserved.</p></section>'
    )
    body += (
        '<section id="results-timing"><span class="section-index">05 / TIMINGS</span><h2>All samples, means and min&ndash;max ranges</h2><div class="panel"><div class="plot timing-plot" id="timings"></div></div><p>One complete solve per case/method is excluded, followed by ten measured cold starts. Only compiled kernels are reused. Order alternates; solves run sequentially with synchronized JAX work and one BLAS thread. Timing includes monitors, controller and retries, and excludes setup, final audits, files and plotting. No timing samples are discarded.</p>'
        + table(
            [
                "Case",
                "Treatment",
                "N",
                "Min [s]",
                "Max [s]",
                "Mean [s]",
                "All reported roots?",
            ],
            time_rows,
        )
        + "</section>"
    )
    body += (
        '<section id="results-verification"><span class="section-index">06 / VERIFICATION</span><h2>Root convergence, accuracy and agreement</h2>'
        + table(
            [
                "Case",
                "Treatment",
                "Updates",
                "Rejected",
                "Combined RMS",
                "Boundary RMS",
                "Analytical check",
            ],
            final_rows,
        )
        + table(
            [
                "Case",
                "Coupled/source mean time",
                "Max pressure difference [Pa]",
                "Scaled field RMS difference",
            ],
            field_rows,
        )
        + "</section>"
    )
    body += (
        '<section id="results-provenance"><span class="section-index">07 / PROVENANCE</span><h2>Historical benchmark, current documentation</h2><p>The embedded JSON contains all measured and excluded records. Original source hashes identify the measured revision. Its recorded unchanged-source flag is not a fresh validation of today\'s solver.</p><div class="controls"><button id="download-csv">Download timing samples</button><button id="download-json">Download complete evidence</button></div><details><summary>Saved-record consistency audit</summary><div class="details-body"><pre>'
        + html.escape(json.dumps(audit, indent=2))
        + '</pre></div></details><details><summary>Recorded metadata and source fingerprints</summary><div class="details-body"><pre>'
        + html.escape(
            json.dumps({k: v for k, v in data.items() if k != "cases"}, indent=2)
        )
        + "</pre></div></details>"
    )
    for filename, title in [
        ("study_overview.md", "Original study overview and protocol"),
        ("benchmark_record.md", "Original benchmark narrative"),
        (
            "historical_smoke.md",
            "Historical one-case coarse commit smoke (not a timing benchmark)",
        ),
    ]:
        rendered, _ = document(filename, "archive-" + filename.split(".")[0])
        body += (
            "<details><summary>"
            + title
            + '</summary><div class="details-body theory-content">'
            + rendered
            + "</div></details>"
        )
    return body + "</section>"


def write_report(output, destination=None):
    output = Path(output).resolve()
    if not output.is_relative_to(STUDY):
        raise ValueError("Output outside BC_residual")
    destination = (
        Path(destination).resolve()
        if destination
        else (
            STUDY / "REPORT.html" if output.name == "steady" else output / "REPORT.html"
        )
    )
    if not destination.is_relative_to(STUDY):
        raise ValueError("HTML destination outside study")
    raw = (output / "summary.json").read_bytes()
    data = json.loads(raw)
    audit = saved_record_audit(data, digest(raw))
    (output / "report_data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    csv_text = timing_csv(data)
    theory, theory_nav = document("implementation.md", "theory")
    recommendations, recommendations_nav = document(
        "recommendations.md", "recommendations"
    )
    theory = (
        '<div class="hero"><div class="eyebrow">Theory / equations / implementation history</div><h1>Eliminated and coupled boundary systems</h1><p>One set of physical constraints, two algebraic representations. The background distinguishes the measured closure adapter from the current public-operator adapter.</p></div><article class="theory-content">'
        + theory
        + "</article>"
    )
    recommendations = (
        '<div class="hero"><div class="eyebrow">Recommendations / future infrastructure</div><h1>One physical boundary model, two assembly strategies</h1><p>Retain elimination for the tested Euler model; validate coupled assembly for difficult implicit closures. These are design recommendations, not new two-fluid results.</p></div><article class="theory-content">'
        + recommendations
        + "</article>"
    )
    bodies = "".join(
        f'<div id="{name}" data-report-page="{name}"'
        + ("" if name == "results" else " hidden")
        + ">"
        + body
        + "</div>"
        for name, body in [
            ("results", results_body(data, audit)),
            ("theory", theory),
            ("recommendations", recommendations),
        ]
    )
    result_nav = [
        ("results-" + key, label)
        for key, label in [
            ("overview", "Findings"),
            ("setup", "Setup & controls"),
            ("convergence", "Residual histories"),
            ("pressure", "Pressure profiles"),
            ("timing", "Repeated timings"),
            ("verification", "Verification"),
            ("provenance", "Provenance & archives"),
        ]
    ]
    nav = "".join(
        '<div data-report-nav="'
        + name
        + '"'
        + ("" if name == "results" else " hidden")
        + ">"
        + "".join(f'<a href="#{key}">{html.escape(label)}</a>' for key, label in items)
        + "</div>"
        for name, items in [
            ("results", result_nav),
            ("theory", theory_nav),
            ("recommendations", recommendations_nav),
        ]
    )
    build = dict(
        summary_sha256=digest(raw),
        renderer_sha256=digest(Path(__file__).read_bytes()),
        renderer_executes_solves=False,
        markdown_version=version("Markdown"),
        preserves_historical_measurements=True,
    )
    payload = dict(benchmark=data, audit=audit, timing_csv=csv_text, provenance=build)
    vendor = ASSETS / "vendor"
    runtime = lambda name: script_safe(
        gzip.decompress((vendor / name).read_bytes()).decode("utf-8")
    )
    scripts = f'<script id="report-data" type="application/json">{json_safe(payload)}</script><script>{runtime("plotly-2.35.2.min.js.gz")}</script><script>{script_safe((ASSETS/"report.js").read_text(encoding="utf-8"))}</script><script>{runtime("mathjax-3.2.2-tex-svg.js.gz")}</script>'
    config = """<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{document.body.dataset.mathReady='true';window.dispatchEvent(new Event('report-math-ready'));})}};</script>"""
    template = (ASSETS / "shell.html").read_text(encoding="utf-8")
    values = dict(
        TITLE="Internal versus coupled boundary residuals",
        BUILD=json_safe(build),
        CSS=(ASSETS / "report.css").read_text(encoding="utf-8"),
        HEAD_EXTRA=config,
        NAV=nav,
        BODY=bodies,
        SCRIPTS=scripts,
        LICENSES=html.escape(
            (vendor / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8")
            + "\n\n"
            + (vendor / "MATHJAX_LICENSE.txt").read_text(encoding="utf-8")
        ),
    )
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", value)
    assert not re.search(r"@@[A-Z_]+@@", template)
    assert not re.search(r"<script[^>]+src\s*=", template, re.I)
    destination.write_bytes(template.encode("utf-8"))
    assert (output / "summary.json").read_bytes() == raw
    manifest = dict(
        **build,
        standalone_html=True,
        measured_runs=sum(len(c["measured_runs"]) for c in data["cases"]),
        excluded_runs=sum(len(c["excluded_runs"]) for c in data["cases"]),
        report_sha256=digest(destination.read_bytes()),
        assets_sha256={
            p.relative_to(ASSETS).as_posix(): digest(p.read_bytes())
            for p in ASSETS.rglob("*")
            if p.is_file()
        },
    )
    (output / "html_report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {destination.name}: {manifest['measured_runs']} measured records retained; no numerical solves."
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/steady")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    write_report(args.output, args.destination)
