"""Build one offline REPORT.html with internal results, theory, use and notes pages.

Reads saved evidence only. All libraries, plot data and offered downloads are
embedded; copying this one file is sufficient for offline reading.
"""

import argparse
import base64
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path
import re

from .html_report_data import FIELD_METADATA, build_report_data
from .channel_visualization_data import build_channel_data

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"


def digest(data):
    return sha256(data).hexdigest()


def script_safe(text):
    return re.sub(r"</script", r"<\\/script", text, flags=re.IGNORECASE)


def json_safe(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")


def vendor(name):
    return gzip.decompress((ASSETS / "vendor" / name).read_bytes()).decode("utf-8")


def headings(body):
    return [(target, re.sub(r"<.*?>", "", html.unescape(title)))
            for target, title in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, re.DOTALL)]


def sidebar(items):
    return "".join(f'<a href="#{html.escape(target, quote=True)}"><span>{i:02d}</span>{html.escape(label)}</a>'
                   for i, (target, label) in enumerate(items, 1))


def table(columns, rows):
    return ('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)
            +'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)
            +'</tbody></table></div>')


def collapsible_results(body):
    """Keep section anchors stable while putting their contents in native boxes."""
    first = body.index('<section id=')
    overview, sections = body[:first], body[first:]
    def wrap(match):
        opening, section_id, contents, closing = match.groups()
        heading, content = contents.strip().split('\n', 1)
        heading = heading.rstrip('\r')
        if not heading.startswith('<div class="section-head">') or not heading.endswith('</div>'):
            raise ValueError(f"Missing first-line results section heading: {section_id}")
        summary = heading.replace('<div class="section-head">', '<summary class="section-head">', 1)
        summary = summary[:-6]+'</summary>'
        return (opening+f'<details class="result-section" id="box-{section_id}" open>'+
                summary+'<div class="result-section-body">'+content+
                '</div></details>'+closing)
    sections, count = re.subn(r'(<section id="([^"]+)"[^>]*>)(.*?)(</section>)', wrap, sections, flags=re.DOTALL)
    if count != 7:
        raise ValueError(f"Expected seven results sections, found {count}")
    return ('<div class="section-controls"><span>Results sections</span>'
            '<button id="results-expand" type="button">Expand all</button>'
            '<button id="results-collapse" type="button">Collapse all</button></div>'
            '<details class="result-section overview-box" id="box-overview" open>'
            '<summary><h2>Campaign overview</h2></summary><div class="result-section-body">'+overview+
            '</div></details>'+sections)


def notes_tables(body, data):
    sweep = (data.get("controller_sweep") or {}).get("runs", [])
    sweep_rows = []
    for r in sweep:
        c = r["controller"]
        sweep_rows.append([r["phase"], r["cells"], r["reconstruction"], r["name"],
            str(r["options"]["newton_iterations"])+" / "+str(r.get("inner_acceptance", "strict")),
            c["initial_cfl"], c["growth"], c["shrink"], c["policy"], r["success"], r["accepted"], r["rejected"],
            r["newton_corrections"], r["rejected_line_trials"],
            r.get("shock_error_cells"), r.get("baseline_scaled_state_difference"),
            bool(r["success"] and r.get("shock_error_cells", 1e99)<=2 and r.get("baseline_scaled_state_difference", 1e99)<1e-5)])
    body = body.replace("@@SWEEP_TABLE@@", table(["Phase", "Cells", "Reconstruction", "Configuration", "Cap / stop", "CFL", "Growth", "Shrink", "Policy", "Root", "Accepted", "Rejected", "Newton", "Backtracks", "Shock error [cells]", "Scaled profile difference", "Matched"], sweep_rows))
    audit = data.get("controller_audit") or {}
    rows = []
    for record in audit.get("records", []):
        if record["cells"] != 400:
            continue
        for method, run in record["methods"].items():
            rows.append([record["case"], method, str(run["old_accepted"])+" / "+str(run["accepted"]),
                str(run["old_rejected"])+" / "+str(run["rejected"]), run["newton_corrections"], run["backtracks"],
                f'{run["final_original_defect"]:.5e}', f'{run["maximum_scaled_state_difference"]:.5e}', run["matched_archived_discrete_state"]])
    body = body.replace("@@CAMPAIGN_TABLE@@", table(["Case, N=400", "Method", "Old / new accepted", "Old / new rejected", "Newton", "Backtracks", "Original defect", "Scaled profile difference", "Matched"], rows))
    summary = data["summary"]
    matched = sum(r["matched_archived_discrete_state"] for d in audit.get("records", []) for r in d["methods"].values())
    misses=summary["finest_method_entries"]-summary["finest_accuracy_passes"]
    body = body.replace("@@CAMPAIGN_SUMMARY@@", f'<p><strong>{summary["root_passes"]}/{summary["method_entries"]} solves meet every configured convergence target; {summary["finest_accuracy_passes"]}/{summary["finest_method_entries"]} finest-grid accuracy assessments pass.</strong> All {misses} accuracy misses and every convergence failure remain visible. The selected data directory is {html.escape(data["provenance"]["primary_directory"])}. The relative transport-equation target is 10<sup>−9</sup>, with separate absolute/BC/global safeguards and monitor stationarity. Histories belong to the selected mesh solve; preceding initialization stages are saved separately. Archived controller/profile comparisons below are historical evidence.</p>')
    return body


def build_report(study, output, *, verification_folder=None):
    data = build_report_data(study, verification_folder=verification_folder)
    data["channels"] = build_channel_data(study, verification_folder=verification_folder)
    payload = json_safe(data)
    sources = {name: (ASSETS/name).read_bytes() for name in ("results.html", "theory.html", "implementation.html", "notes.html")}
    chapters = {name[:-5]: raw.decode("utf-8") for name, raw in sources.items()}
    summary = data["summary"]
    settings = data["cases"][0]["meshes"][0]["settings"]
    options=data["cases"][0]["meshes"][0]["methods"]["constant"]["effective_options"]
    values = dict(CONTROLLER_LABEL=f'CFL {settings["cfl"]} / grow {settings.get("cfl_growth", settings.get("growth", "work-based"))} / shrink {settings.get("cfl_shrink", settings.get("shrink", .5))} / Newton cap {options["newton_iterations"]}',
        ROOT_PASS=summary["root_passes"], RUN_COUNT=summary["method_entries"], ACCURACY_PASS=summary["finest_accuracy_passes"],
        ACCURACY_COUNT=summary["finest_method_entries"], ACCURACY_FAILURES=summary["finest_method_entries"]-summary["finest_accuracy_passes"], REPRODUCED=summary["audited_array_reproduction_passes"],
        ROOT_FAILURES=summary["method_entries"]-summary["root_passes"],
        ROOT_STATUS_CLASS="success" if summary["root_passes"]==summary["method_entries"] else "warning")
    for key, value in values.items():
        chapters["results"] = chapters["results"].replace(f"@@{key}@@", html.escape(str(value)))
    monitoring_rows=[]
    for run in data.get("monitoring_validation",{}).get("runs",[]):
        if run["root_pass"]:
            continue
        monitors=run["monitors"]
        monitoring_rows.append([run["case"],run["cells"],run["method"],run["root_pass"],
            run["accepted_updates"],f'{run["max_absolute"]:.4e}',
            f'{run["max_relative"]:.4e}' if run["max_relative"] is not None else "absolute only",
            "; ".join(f'{name}: {"stable" if value["stable"] else "not stable"}' for name,value in monitors.items()),
            run["reason"]])
    chapters["results"]=chapters["results"].replace("@@MONITORING_TABLE@@",table(
        ["Case","Cells","Method","All targets","Accepted updates","Max equation norm",
         "Worst active R/R(initial)","Engineering monitors","Final reason"],monitoring_rows)
        if monitoring_rows else "<p>Every selected run met all configured convergence targets.</p>")
    benchmark_rows=[]
    for case in data["cases"]:
        record=case["meshes"][-1];meta=record["metadata"];g=data["channels"]["cases"][case["id"]]["geometry"]
        reservoir=meta["reservoir"]
        benchmark_rows.append([case["id"],f'{g["length"]:.6g}',
            f'{g["r"][0]:.6g} / {g["r"][-1]:.6g}',f'{1000*min(g["b"]):.6g} / {1000*max(g["b"]):.6g}',
            reservoir["p0"],reservoir["T0"],f'{meta["alpha_in"]*180/3.141592653589793:.6g}',
            f'{meta["backpressure"]:.8g}',meta["inlet_regime"]+" / "+meta["outlet_regime"],meta["shock_m"]])
    chapters["implementation"]=chapters["implementation"].replace("@@BENCHMARK_TABLE@@",table(
        ["Case","L [m]","r inlet / outlet [m]","Width min / max [mm]","p0 [Pa]","T0 [K]","Inlet angle [deg]","Reference outlet p [Pa]","Inlet / outlet mode","Reference shock m [m]"],benchmark_rows))
    constants=data["cases"][0]["meshes"][0]["metadata"]["reference_invariants"]
    chapters["implementation"]=chapters["implementation"].replace("@@REFERENCE_CONSTANTS@@",
        '<p>Independent perfect-gas reference constants (not FVM closures): '+html.escape(
            f'gamma = {constants["gamma"]:.9g}, R = {constants["R"]:.9g} J/(kg K), enthalpy shift = {constants["h_shift"]:.9g} J/kg, entropy shift = {constants["entropy_shift"]:.9g} J/(kg K).')+'</p>')
    chapters["notes"] = notes_tables(chapters["notes"], data)
    chapters["theory"] = ('<div class="hero" id="theory-overview"><div class="hero-copy"><div class="eyebrow">Mathematical model / discrete algorithm</div><h1>Theory &amp; numerical implementation</h1><p>General-EOS balances, eigenanalysis, finite volumes, full implicit residual and coupled boundary equations.</p></div></div><article class="theory-content">'+chapters["theory"]+'</article>')
    page_names = dict(results="Results", theory="Theory", implementation="Implementation", notes="Notes & findings")
    roots = dict(results="overview", theory="theory-overview", implementation="use-overview", notes="notes-overview")
    nav_items = {name: headings(body) for name, body in chapters.items()}
    nav_items["results"] = [(key, label) for key, label in [("overview","Overview"),("fields","Flow fields"),("conservation","Convergence & conservation"),("refinement","Grid refinement"),("shocks","Shock capture"),("audit","Numerical audit"),("provenance","Provenance"),("assessment","Case evaluation")]]
    chapters["results"] = collapsible_results(chapters["results"])
    evidence_path = study/"outputs/controller_sweep/evidence.zip"
    evidence = evidence_path.read_bytes()
    downloads = dict(controller_evidence=dict(filename="EulerAnnular_Controller_Evidence.zip", mime="application/zip", sha256=digest(evidence), base64=base64.b64encode(evidence).decode("ascii")))
    build = dict(schema_version=2, generated_at_utc=datetime.now(timezone.utc).isoformat(), display_timezone="Europe/Paris",
        payload_sha256=digest(payload.encode("utf-8")), generator_sha256=digest(Path(__file__).read_bytes()),
        chapter_sources={name:digest(raw) for name,raw in sources.items()},
        assets={path.name:digest(path.read_bytes()) for path in ASSETS.iterdir() if path.is_file()},
        internal_pages=list(page_names), numerical_sources_and_saved_data_read_only=True, network_required=False,
        solution_field_markers=False, solution_fields=[entry[0] for entry in FIELD_METADATA],
        channel_visualizations=dict(cases=len(data["channels"]["cases"]),views=["meridional","frontal","annular"],layout="side_by_side",control_points_default=False),
        embedded_downloads={key:{k:v for k,v in value.items() if k!="base64"} for key,value in downloads.items()},
        library_versions=dict(plotly_js="2.35.2",mathjax="3.2.2"))
    switches = ''.join(f'<a href="#{roots[name]}" data-page-choice="{name}"'+(' aria-current="page"' if name=="results" else '')+f'>{i:02d} &middot; {label}</a>' for i,(name,label) in enumerate(page_names.items(),1))
    navigation = ''.join(f'<nav data-page-nav="{name}" aria-label="{label} sections"'+(' hidden' if name!="results" else '')+'>'+sidebar(nav_items[name])+'</nav>' for name,label in page_names.items())
    body = ''.join(f'<section class="report-page" data-report-page="{name}" id="report-page-{name}"'+(' hidden' if name!="results" else '')+'>'+chapters[name]+'</section>' for name in page_names)
    math_config = """<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{document.body.dataset.theoryReady='true';window.dispatchEvent(new CustomEvent('annular-theory-ready'));})}};</script>"""
    scripts = (f'<script id="report-data" type="application/json">{payload}</script>'
        +f'<script id="embedded-downloads" type="application/json">{json_safe(downloads)}</script>'
        +f'<script>{script_safe(vendor("plotly-2.35.2.min.js.gz"))}</script>'
        +f'<script id="MathJax-script">{script_safe(vendor("mathjax-3.2.2-tex-svg.js.gz"))}</script>'
        +f'<script>{script_safe((ASSETS/"report.js").read_text(encoding="utf-8"))}</script>')
    licenses = '\n\n'.join((ASSETS/"vendor"/name).read_text(encoding="utf-8") for name in ("PLOTLY_LICENSE.txt","MATHJAX_LICENSE.txt"))
    template = (ASSETS/"document.html").read_text(encoding="utf-8")
    for key, value in dict(BUILD=json_safe(build), CSS=(ASSETS/"report.css").read_text(encoding="utf-8"), HEAD_EXTRA=math_config,
                           SWITCHES=switches, NAV=navigation, BODY=body, LICENSES=html.escape(licenses), SCRIPTS=scripts).items():
        template = template.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("Unresolved report placeholder")
    for name, raw in sources.items():
        if (ASSETS/name).read_bytes()!=raw:
            raise ValueError("Chapter source changed during build")
    output.mkdir(parents=True,exist_ok=True)
    raw = template.encode("utf-8")
    (output/"REPORT.html").write_bytes(raw)
    equations = len(re.findall('class="math-(?:block|inline)"', chapters["theory"]))
    manifest = dict(**build, equations=equations, theory_sections=len(nav_items["theory"]), summary=summary,
        input_files=data["provenance"]["input_files"], artifacts={"REPORT.html":dict(bytes=len(raw),sha256=digest(raw))})
    log = output/"outputs/html_report";log.mkdir(parents=True,exist_ok=True)
    (log/"channel_geometry.json").write_text(json_safe(data["channels"])+"\n",encoding="utf-8")
    (log/"build.json").write_text(json.dumps(manifest,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(dict(report=str(output/"REPORT.html"),internal_pages=list(page_names),equations=equations,summary=summary,bytes=len(raw)),indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=STUDY)
    parser.add_argument("--output",type=Path,default=STUDY)
    parser.add_argument("--verification-folder",type=Path,default=None)
    args=parser.parse_args()
    build_report(args.study.resolve(),args.output.resolve(),verification_folder=args.verification_folder)


if __name__=="__main__":
    main()
