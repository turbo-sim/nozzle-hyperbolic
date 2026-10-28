"""Build the single-file, offline implicit-Euler boundary-condition report.

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


def result_body(data, initialization):
    """Reuse one results template for both initializations."""
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
    return result_body


def scoped_body(body, prefix):
    """Give every panel a unique ID and keep local navigation inside this file."""
    body = body.replace('href="THEORY.html"', 'href="#theory"')
    body = body.replace('href="REPORT.html"', 'href="#uniform"')
    body = re.sub(r'<a href="THEORY.md">.*?</a> or ', "", body)
    ids = set(re.findall(r'\bid="([^"]+)"', body))
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}-{m[1]}"', body)
    return re.sub(
        r'href="#([^"]+)"',
        lambda m: f'href="#{prefix}-{m[1]}"' if m[1] in ids else m[0],
        body,
    )


def render(report, output: Path, destination: Path):
    """Generate one offline REPORT.html containing both studies and all theory."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if not output.is_relative_to(STUDY) or not destination.is_relative_to(STUDY):
        raise ValueError("All report artifacts must remain inside BC_backpressure")
    raw_summary = (output / "summary.json").read_bytes()
    if json.loads(raw_summary) != report:
        raise ValueError("Passed report differs from the saved numerical summary")
    summary_hash = digest(raw_summary)
    verification, audit_hashes = audit_bundle(output, summary_hash)
    provenance = {
        "summary_sha256": summary_hash,
        "audit_sha256": audit_hashes,
        "renderer_executes_solves": False,
        "source_code_read_only": True,
        "production_source_sha256": report["source_before"],
        "shared_imports_sha256": report["shared_before"],
    }
    build = {
        **provenance,
        "renderer_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Single offline document; two outlet laws; uniform and adapted starts; implicit Euler only",
        "library_versions": {"plotly_js": "2.35.2", "mathjax": "3.2.2"},
    }
    bundles, bodies = {}, []
    csv_hashes = {}
    for initialization in ("uniform", "adapted"):
        data = selected_report(report, initialization)
        csv_text = outcome_csv(data)
        (output / f"outcomes_{initialization}.csv").write_text(
            csv_text, encoding="utf-8"
        )
        csv_hashes[initialization] = digest(csv_text.encode("utf-8"))
        bundles[initialization] = dict(
            report=data, verification=verification, csv=csv_text, provenance=provenance
        )
        body = scoped_body(result_body(data, initialization), initialization)
        hidden = "" if initialization == "uniform" else " hidden"
        bodies.append(
            f'<div id="{initialization}" data-report-page="{initialization}"{hidden}>{body}</div>'
        )
    theory = scoped_body((ASSETS / "theory.html").read_text(encoding="utf-8"), "theory")
    note = (ASSETS / "theory_notes.md").read_text(encoding="utf-8")
    theory += (
        '<details class="supplementary-note"><summary>Supplementary implementation note (Markdown source)</summary>'
        f'<div class="details-body"><pre>{html.escape(note)}</pre></div></details>'
    )
    bodies.append(f'<div id="theory" data-report-page="theory" hidden>{theory}</div>')
    result_nav = [
        ("case", "Case & setup"),
        ("method", "Implicit solver settings"),
        ("convergence", "Residual histories"),
        ("cfl", "Actual CFL histories"),
        ("pressure", "Pressure profiles"),
        ("mach", "Mach profiles"),
        ("outcomes", "Verification outcomes"),
        ("audit", "Audit & provenance"),
    ]
    theory_nav = [
        ("balances", "Euler balances"),
        ("current", "Current outlet"),
        ("wave", "Pressure-wave outlet"),
        ("wall", "One-way wall treatment"),
        ("flux", "Boundary-face flux"),
        ("implicit", "Implicit corrections"),
        ("controller", "Timestep & CFL"),
        ("verification", "Verification & limits"),
    ]
    nav = "".join(
        f'<div data-report-nav="{page}"'
        + ("" if page == "uniform" else " hidden")
        + ">"
        + navigation([(f"{page}-{key}", label) for key, label in items])
        + "</div>"
        for page, items in (
            ("uniform", result_nav),
            ("adapted", result_nav),
            ("theory", theory_nav),
        )
    )
    vendor = ASSETS / "vendor"

    def runtime(name):
        return script_safe(
            gzip.decompress((vendor / name).read_bytes()).decode("utf-8")
        )

    math_config = """<script>
window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{document.body.dataset.theoryReady='true';window.dispatchEvent(new CustomEvent('backpressure-theory-ready'));})}};
</script>"""
    app = (ASSETS / "report.js").read_text(encoding="utf-8")
    scripts = (
        f'<script id="report-data" type="application/json">{json_safe(bundles)}</script>'
        f'<script>{runtime("plotly-2.35.2.min.js.gz")}</script>'
        f"<script>{script_safe(app)}</script>"
        f'<script id="MathJax-script">{runtime("mathjax-3.2.2-tex-svg.js.gz")}</script>'
    )
    values = {
        "TITLE": "Backpressure boundary-condition verification",
        "BUILD": json_safe(build),
        "CSS": (ASSETS / "report.css").read_text(encoding="utf-8"),
        "HEAD_EXTRA": math_config,
        "NAV": nav,
        "BODY": "\n".join(bodies),
        "SCRIPTS": scripts,
        "LICENSES": html.escape(
            "Plotly.js 2.35.2\n\n"
            + (vendor / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8")
            + "\n\nMathJax 3.2.2\n\n"
            + (vendor / "MATHJAX_LICENSE.txt").read_text(encoding="utf-8")
        ),
    }
    content = (ASSETS / "shell.html").read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", content):
        raise ValueError("Unresolved report template marker")
    if re.search(r"<(?:script|img)[^>]+src\s*=", content, re.I):
        raise ValueError("Standalone report must not load external scripts or images")
    destination.write_bytes(content.encode("utf-8"))
    if (output / "summary.json").read_bytes() != raw_summary:
        raise ValueError("Numerical data changed while rendering")
    manifest = {
        **build,
        "input_summary_matches_retained_runs": True,
        "n_cases": 2,
        "n_saved_implicit_runs": 8,
        "standalone_html": True,
        "assets_sha256": {
            p.relative_to(ASSETS).as_posix(): digest(p.read_bytes())
            for p in ASSETS.rglob("*")
            if p.is_file()
        },
        "artifacts": {
            destination.name: dict(
                sha256=digest(destination.read_bytes()),
                bytes=destination.stat().st_size,
            )
        },
        "outcome_csv_sha256": csv_hashes,
    }
    manifest_path = output / "html_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dict(report=destination, manifest=manifest_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs" / "recommended_cfl"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    render(
        json.loads((output / "summary.json").read_text(encoding="utf-8")),
        output,
        STUDY / "REPORT.html",
    )
