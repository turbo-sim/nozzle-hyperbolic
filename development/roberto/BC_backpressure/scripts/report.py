"""Build the three-page, offline implicit-Euler boundary-condition report.

Only current/source and pressure-wave outlet records enter the presentation.
Numerical arrays and histories are read-only. Plotly and MathJax are vendored
locally and embedded into the generated HTML; no installation is required.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re


STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
VARIANTS = ("source", "pressure_wave")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_safe(value) -> str:
    """Embed data as JSON without allowing an HTML script terminator."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def script_safe(value: str) -> str:
    return re.sub(r"</script", r"<\/script", value, flags=re.I)


def selected_report(report, initialization):
    """Restrict embedded data to the two implicit formulations in this report."""
    metadata_keys = (
        "mesh",
        "settings",
        "controller",
        "precision",
        "initialization",
        "source_baseline",
        "timing_note",
        "solution_and_history_sha256",
        "pressure_wave_flux",
        "backflow_safeguard",
    )
    metadata = {
        k: report["metadata"][k] for k in metadata_keys if k in report["metadata"]
    }
    metadata["versions"] = {
        k: v
        for k, v in report["metadata"].get("versions", {}).items()
        if k in ("jax", "jaxlib", "jaxprop", "numpy", "scipy")
    }
    adapted_keys = ("iterations", "wall_s", "initial_point_sha256", "diagnostics")
    result = {
        "metadata": dict(metadata, selected_initialization=initialization),
        "adapted": {k: report["adapted"][k] for k in adapted_keys},
        "source_unchanged": report.get("source_unchanged"),
        "cases": [],
    }
    run_keys = (
        "variant",
        "solver",
        "x",
        "history",
        "wall_s",
        "iterations",
        "message",
        "root_success",
        "physical_success",
        "diagnostics",
        "backend",
        "solver_settings",
        "initialization",
        "initial_diagnostics",
        "initial_point_sha256",
        "assembly_audit",
        "compilation",
        "trial_history",
    )
    for case in report["cases"]:
        runs = [
            r
            for r in case["runs"]
            if r.get("solver") == "implicit"
            and r["variant"] in VARIANTS
            and r["initialization"] == initialization
        ]
        if len(runs) != 2 or {r["variant"] for r in runs} != set(VARIANTS):
            raise ValueError(
                "Each case must contain exactly one implicit record per compared outlet"
            )
        runs.sort(key=lambda r: VARIANTS.index(r["variant"]))
        for run in runs:
            for point in run["history"]:
                if point["accepted_index"] > 0:
                    cfl = point.get("effective_cfl")
                    if not isinstance(cfl, (int, float)) or not 0 < cfl < float("inf"):
                        raise ValueError(
                            "Each accepted update must retain a finite positive actual CFL"
                        )
        result["cases"].append(
            {
                "id": case["id"],
                "label": case["label"],
                "x_m": case["x_m"],
                "p_back_ratio": case["p_back_ratio"],
                "reference": case["reference"],
                "runs": [{k: r[k] for k in run_keys if k in r} for r in runs],
            }
        )
    if sum(len(c["runs"]) for c in result["cases"]) != 4:
        raise ValueError(
            "This report expects the two-case, four-record implicit comparison"
        )
    return result


def outcome_csv(report):
    stream = io.StringIO(newline="")
    rows = []
    for case in report["cases"]:
        for run in case["runs"]:
            d = run["diagnostics"]
            cfl = [
                h["effective_cfl"] for h in run["history"] if h["accepted_index"] > 0
            ]
            rows.append(
                {
                    "case": case["id"],
                    "variant": run["variant"],
                    "initialization": run["initialization"],
                    "accepted_updates": run["iterations"],
                    "residual_rms": d["residual_rms"],
                    "root_success": run["root_success"],
                    "physical_success": run["physical_success"],
                    "exit_M": d["exit_M"],
                    "mass_imbalance": d["mass_imbalance"],
                    "boundary_pressure_relative_error": d[
                        "boundary_pressure_relative_error"
                    ],
                    "shock_location_m": d["shock_location_m"],
                    "normalized_L1_p": d["normalized_L1"]["p"],
                    "normalized_L1_rho": d["normalized_L1"]["rho"],
                    "normalized_L1_M": d["normalized_L1"]["M"],
                    "actual_cfl_samples": len(cfl),
                    "actual_cfl_min": min(cfl) if cfl else None,
                    "actual_cfl_max": max(cfl) if cfl else None,
                    "wall_s": run["wall_s"],
                    "settings": json.dumps(run["solver_settings"], sort_keys=True),
                }
            )
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def audit_bundle(output, summary_hash):
    verification = {}
    fingerprints = {}
    for name, path in (
        ("boundary", STUDY / "outputs" / "boundary_audit.json"),
        ("saved_state", output / "audit.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required verification artifact is missing: {path}"
            )
        verification[name] = json.loads(path.read_text(encoding="utf-8"))
        fingerprints[name] = digest(path.read_bytes())
    verification["saved_state_matches_summary"] = (
        verification["saved_state"].get("summary_sha256") == summary_hash
    )
    if not verification["saved_state_matches_summary"]:
        raise ValueError("Saved-state audit does not match the numerical summary")
    # Audit records have the same narrow scope as the displayed comparison.
    for item in verification["saved_state"].get("runs", []):
        if (
            item.get("variant") not in VARIANTS
            or item.get("solver", "implicit") != "implicit"
        ):
            raise ValueError(
                "The state audit includes a formulation outside this report's scope"
            )
    for item in verification["boundary"].get("local_cases", []):
        if item.get("variant") not in VARIANTS:
            raise ValueError(
                "The boundary audit includes a formulation outside this report's scope"
            )
    return verification, fingerprints


def navigation(items):
    return "\n".join(
        f'<a href="#{target}">{html.escape(label)}</a>' for target, label in items
    )


def panels(report, kind):
    return "\n".join(
        f'<div class="panel"><div class="panel-head"><h3>{html.escape(c["label"])} &middot; '
        f'p<sub>b</sub>/p<sub>0</sub> = {c["p_back_ratio"]:.6f}</h3></div>'
        f'<div class="plot" id="{kind}-{index}" aria-label="{kind} for {html.escape(c["id"])}"></div></div>'
        for index, c in enumerate(report["cases"])
    )


def shell(page, body, build, css, scripts, licenses, head=""):
    template = (ASSETS / "shell.html").read_text(encoding="utf-8")
    results = page != "theory"
    nav = navigation(
        [
            ("case", "Case & setup"),
            ("method", "Implicit solver settings"),
            ("convergence", "Residual histories"),
            ("cfl", "Actual CFL histories"),
            ("pressure", "Pressure profiles"),
            ("mach", "Mach profiles"),
            ("outcomes", "Verification outcomes"),
            ("audit", "Audit & provenance"),
        ]
        if results
        else [
            ("balances", "Euler balances"),
            ("current", "Current outlet"),
            ("wave", "Pressure-wave outlet"),
            ("wall", "One-way wall treatment"),
            ("flux", "Boundary-face flux"),
            ("implicit", "Implicit corrections"),
            ("controller", "Timestep & CFL"),
            ("verification", "Verification & limits"),
        ]
    )
    values = {
        "TITLE": (
            (
                "Uniform initialization"
                if page == "uniform"
                else "Adapted initialization"
            )
            if results
            else "Theory and equations"
        ),
        "PAGE": page,
        "BUILD": json_safe(build),
        "CSS": css,
        "HEAD_EXTRA": head,
        "UNIFORM_CURRENT": 'aria-current="page"' if page == "uniform" else "",
        "ADAPTED_CURRENT": 'aria-current="page"' if page == "adapted" else "",
        "THEORY_CURRENT": "" if results else 'aria-current="page"',
        "SIDEBAR_LABEL": "Case & results" if results else "Theory & equations",
        "NAV": nav,
        "MAIN_CLASS": "results-main" if results else "theory-main",
        "BODY": body,
        "SCRIPTS": scripts,
        "LICENSES": html.escape(licenses),
    }
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("An unresolved template marker remains")
    return template


def render(report, output: Path, destination: Path):
    """Create REPORT.html, ADAPTED.html, THEORY.html, outcome CSVs and rendering manifest."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if not output.is_relative_to(STUDY) or not destination.is_relative_to(STUDY):
        raise ValueError("All report artifacts must remain inside BC_backpressure")
    output.mkdir(parents=True, exist_ok=True)
    raw_summary = (output / "summary.json").read_bytes()
    summary_hash = digest(raw_summary)
    if json.loads(raw_summary) != report:
        raise ValueError("Passed report differs from the saved numerical summary")
    data = selected_report(report, "uniform")
    verification, audit_hashes = audit_bundle(output, summary_hash)
    csv_text = outcome_csv(data)
    css = (ASSETS / "report.css").read_text(encoding="utf-8")
    app = (ASSETS / "report.js").read_text(encoding="utf-8")
    vendor = ASSETS / "vendor"
    plotly = gzip.decompress((vendor / "plotly-2.35.2.min.js.gz").read_bytes()).decode(
        "utf-8"
    )
    mathjax = gzip.decompress(
        (vendor / "mathjax-3.2.2-tex-svg.js.gz").read_bytes()
    ).decode("utf-8")
    provenance = {
        "summary_sha256": summary_hash,
        "audit_sha256": audit_hashes,
        "retained_solution_and_history_sha256": data["metadata"].get(
            "solution_and_history_sha256"
        ),
        "renderer_executes_solves": False,
        "source_code_read_only": True,
        "production_source_sha256": report["source_before"],
        "shared_imports_sha256": report["shared_before"],
    }
    build = {
        "summary_sha256": summary_hash,
        "renderer_sha256": digest(Path(__file__).read_bytes()),
        "audit_sha256": audit_hashes,
        "scope": "Two outlet laws; uniform and adapted starts; implicit Euler only",
        "library_versions": {"plotly_js": "2.35.2", "mathjax": "3.2.2"},
    }
    common_script = f"<script>{script_safe(app)}</script>"
    pages = []
    for initialization, filename in (
        ("uniform", destination.name),
        ("adapted", "ADAPTED.html"),
    ):
        data = selected_report(report, initialization)
        csv_text = outcome_csv(data)
        (output / f"outcomes_{initialization}.csv").write_text(
            csv_text, encoding="utf-8"
        )
        bundle = {
            "report": data,
            "verification": verification,
            "csv": csv_text,
            "provenance": provenance,
        }
        result_body = (ASSETS / "results.html").read_text(encoding="utf-8")
        for kind in ("residual", "cfl", "pressure", "mach"):
            result_body = result_body.replace(
                f"@@{kind.upper()}_PANELS@@", panels(data, kind)
            )
        uniform = initialization == "uniform"
        replacements = {
            "HEADING": (
                "Uniform stagnation initialization"
                if uniform
                else "From an adapted expansion to an internal shock"
            ),
            "INTRO": (
                "A uniform field at reservoir pressure and enthalpy, with zero velocity, initializes both backpressure cases. Algebraic ghost values are completed by each boundary law."
                if uniform
                else "The same numerically converged adapted field initializes both backpressure cases. This isolates whether each outlet can admit an upstream-moving compression from a supersonic exit."
            ),
            "INITIAL_LABEL": "Uniform stagnation" if uniform else "Adapted",
            "INITIAL_SETUP": (
                "Every solve receives identical uniform interior coordinates: p = 200 kPa, T = 300 K and u = 0 m/s."
                if uniform
                else "Every solve receives the same copied numerical adapted interior coordinates."
            ),
            "HISTORY_NOTE": (
                "Each curve includes the common uniform initial state and all accepted updates. Compare residual reduction together with the verified final shock branch."
                if uniform
                else "The source control is a single point at index zero: it stops at the initial Euler root without developing a shock. No source CFL curve exists because no timestep is accepted."
            ),
        }
        for key, value in replacements.items():
            result_body = result_body.replace(f"@@{key}@@", value)
        result_scripts = (
            f'<script id="report-data" type="application/json">{json_safe(bundle)}</script>'
            f"<script>{script_safe(plotly)}</script>{common_script}"
        )
        content = shell(
            initialization,
            result_body,
            build,
            css,
            result_scripts,
            "Plotly.js 2.35.2\n\n"
            + (vendor / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8"),
        )
        pages.append((destination.with_name(filename), content))
    math_config = """<script>
window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{document.body.dataset.theoryReady='true';window.dispatchEvent(new CustomEvent('backpressure-theory-ready'));})}};
</script>"""
    theory_body = (ASSETS / "theory.html").read_text(encoding="utf-8")
    theory = shell(
        "theory",
        theory_body,
        build,
        css,
        common_script + f'<script id="MathJax-script">{script_safe(mathjax)}</script>',
        "MathJax 3.2.2\n\n"
        + (vendor / "MATHJAX_LICENSE.txt").read_text(encoding="utf-8"),
        math_config,
    )
    theory_path = destination.with_name("THEORY.html")
    pages.append((theory_path, theory))
    for path, content in pages:
        if re.search(r"<(?:script|img)[^>]+src\s*=", content, re.I):
            raise ValueError("Report pages must not load external scripts or images")
        path.write_bytes(content.encode("utf-8"))
    if (output / "summary.json").read_bytes() != raw_summary:
        raise ValueError("Numerical data changed while the report was rendered")
    manifest = {
        **build,
        "input_summary_matches_retained_runs": True,
        "n_cases": len(data["cases"]),
        "n_saved_implicit_runs": 8,
        "renderer_executes_solves": False,
        "assets_sha256": {
            p.relative_to(ASSETS).as_posix(): digest(p.read_bytes())
            for p in ASSETS.rglob("*")
            if p.is_file()
        },
        "artifacts": {
            p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
            for p, _ in pages
        },
        "outcome_csv_sha256": digest(csv_text.encode("utf-8")),
    }
    manifest_path = output / "html_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "report": destination,
        "adapted": destination.with_name("ADAPTED.html"),
        "theory": theory_path,
        "csv": [output / "outcomes_uniform.csv", output / "outcomes_adapted.csv"],
        "manifest": manifest_path,
    }
