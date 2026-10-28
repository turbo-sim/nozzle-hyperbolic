"""Repackage unchanged backpressure results into one offline report and theory tab.

No analytical reference or numerical solution is recomputed. Original embedded
profiles and colors are retained exactly; the dataset fingerprint remains fixed.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from .markdown_notes import document

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
EXPECTED_SUMMARY_SHA256 = (
    "448d62d5bac6083d5367b5b2a0c8cb5c75eefcc01e6d9b4e7a4aa2a360afff32"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script[^>]*id="report-data"[^>]*>(.*?)</script>', text, re.S)
    if not match:
        raise ValueError(f"No embedded study payload in {path}")
    return json.loads(match[1])


def script_safe(value):
    return re.sub(r"</script", r"<\/script", value, flags=re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/baseline")
    parser.add_argument("--output", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument(
        "--plotly-js",
        type=Path,
        help="Optional compatible local Plotly runtime; default is vendored",
    )
    args = parser.parse_args()
    folder = args.input.resolve()
    destination = args.output.resolve()
    if not folder.is_relative_to(STUDY) or not destination.is_relative_to(STUDY):
        raise ValueError("Report paths must remain inside the study")
    inputs = [
        folder / name for name in ["summary.json", "audit.json", "operating_map.csv"]
    ]
    hashes = {p.name: sha256(p) for p in inputs}
    if hashes["summary.json"] != EXPECTED_SUMMARY_SHA256:
        raise ValueError(
            "Saved results changed; review the existing narrative before rendering another dataset"
        )
    summary = json.loads(inputs[0].read_text(encoding="utf-8"))
    audit = json.loads(inputs[1].read_text(encoding="utf-8"))
    payload_file = folder / "report_payload.json"
    if payload_file.exists():
        bundle = json.loads(payload_file.read_text(encoding="utf-8"))
    else:
        candidates = [
            destination,
            STUDY / "REPORT.html",
            STUDY / "report_backpressure_sensitivity.html",
            STUDY / "outputs/presentation_archive/original_report.html",
        ]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "Original embedded analytical profiles required. Keep REPORT.html or supply outputs/baseline/report_payload.json."
            )
        bundle = extract(source)
    assert bundle["report"] == summary and bundle["audit"] == audit
    assert bundle["csv"] == inputs[2].read_text(encoding="utf-8")
    assert len(summary["runs"]) == 104 and len(bundle["references"]) == 88
    theory, headings = document("physics_and_boundaries.md", "theory")
    theory = (
        '<div class="hero"><div class="eyebrow">Existing physical background / numerical outlet</div><h1>Physical regimes and the numerical outlet</h1></div><article class="theory-content">'
        + theory
        + "</article>"
    )
    for name, title in [
        ("study_overview.md", "Original study setup, diagnostics and execution notes"),
        ("results_record.md", "Original detailed results record"),
    ]:
        rendered, _ = document(name, "theory-archive-" + name.split(".")[0])
        theory += (
            "<details><summary>"
            + title
            + '</summary><div class="details-body theory-content">'
            + rendered
            + "</div></details>"
        )
    body = (
        '<div id="results" data-report-page="results">'
        + (ASSETS / "results.html").read_text(encoding="utf-8")
        + '</div><div id="theory" data-report-page="theory" hidden>'
        + theory
        + "</div>"
    )
    labels = [
        ("summary", "Abstract"),
        ("setup", "Model & numerical method"),
        ("physics", "Analytical regimes"),
        ("operating-map", "Continuation results"),
        ("transitions", "Critical transitions"),
        ("profiles", "Field distributions"),
        ("branches", "Steady-state multiplicity"),
        ("convergence", "Convergence"),
        ("quality", "Accuracy & conservation"),
        ("data", "Operating-point data"),
        ("reproducibility", "Limitations & reproducibility"),
    ]
    import html

    nav = (
        '<div data-report-nav="results" class="contents">'
        + "".join(f'<a href="#{key}">{html.escape(label)}</a>' for key, label in labels)
        + "</div>"
    )
    nav += (
        '<div data-report-nav="theory" hidden>'
        + "".join(
            f'<a href="#{key}">{html.escape(label)}</a>' for key, label in headings
        )
        + "</div>"
    )
    vendor = ASSETS / "vendor"
    runtime = lambda name: gzip.decompress((vendor / name).read_bytes()).decode("utf-8")
    plotly = (
        args.plotly_js.read_text(encoding="utf-8")
        if args.plotly_js
        else runtime("plotly-2.35.2.min.js.gz")
    )
    encoded = (
        json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    scripts = f'<script type="application/json" id="report-data">{encoded}</script><script>{script_safe(plotly)}</script><script>{script_safe((ASSETS/"shell.js").read_text(encoding="utf-8"))}</script><script>{script_safe((ASSETS/"report.js").read_text(encoding="utf-8"))}</script><script>{script_safe(runtime("mathjax-3.2.2-tex-svg.js.gz"))}</script>'
    config = """<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{document.body.dataset.mathReady='true';window.dispatchEvent(new Event('report-math-ready'));})}};</script>"""
    build = dict(
        input_sha256=hashes,
        numerical_solves_rerun=False,
        analytical_profiles_recomputed=False,
        original_payload_preserved=True,
    )
    values = dict(
        TITLE="Nozzle backpressure sensitivity",
        BUILD=json.dumps(build),
        CSS=(ASSETS / "shell.css").read_text(encoding="utf-8")
        + "\n"
        + (ASSETS / "report.css").read_text(encoding="utf-8"),
        HEAD_EXTRA=config,
        NAV=nav,
        BODY=body,
        SCRIPTS=scripts,
        LICENSES=html.escape(
            (vendor / "PLOTLY_LICENSE.txt").read_text(encoding="utf-8")
            + "\n\n"
            + (vendor / "MATHJAX_LICENSE.txt").read_text(encoding="utf-8")
        ),
    )
    template = (ASSETS / "shell.html").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", value)
    assert not re.search(r"@@[A-Z_]+@@", template)
    destination.write_bytes(template.encode("utf-8"))
    assert extract(destination) == bundle
    assert hashes == {p.name: sha256(p) for p in inputs}
    manifest = dict(
        **build,
        output_sha256=sha256(destination),
        output_bytes=destination.stat().st_size,
        renderer_sha256=sha256(Path(__file__)),
        assets_sha256={
            p.relative_to(ASSETS).as_posix(): sha256(p)
            for p in ASSETS.rglob("*")
            if p.is_file()
        },
    )
    (folder / "html_report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Created {destination.name}: exact original 104 states, 88 analytical profiles, colors, CSV, audit and provenance retained."
    )


if __name__ == "__main__":
    main()
