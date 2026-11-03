"""Build the single-file HTML from retained implicit-integration results.

No numerical solver is imported, no simulations are advanced, and historical
timings, source fingerprints and run records are read-only.
"""

import argparse
import base64
import csv
import gzip
import hashlib
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys

import markdown
from matplotlib import colormaps
from matplotlib.colors import to_hex

from .report_data import (CASE_ORDER, PAGES, STUDY, TITLES, audit_saved, digest,
                          fingerprints, load_saved, payload, study_runs)

ASSETS = Path(__file__).resolve().parent/"report_assets"
SHARED = STUDY.parent/"sensitivity_backpressure/scripts/report_assets"


def json_safe(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")


def script_safe(value):
    return re.sub(r"</script", r"<\/script", value, flags=re.I)


def csv_bytes(rows):
    handle=io.StringIO(newline="")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def download(label, name, raw, mime):
    uri=f"data:{mime};base64,"+base64.b64encode(raw).decode()
    return f'<a href="{uri}" download="{html.escape(name,quote=True)}">{html.escape(label)}</a>'


def table(headers, rows):
    return '<div class="panel table-scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def fmt(value):
    return f"{value:.4g}" if isinstance(value,float) else html.escape(str(value))


def flag(value):
    return '<span class="pass">Pass</span>' if value else '<span class="fail">Fail</span>'


def notes_html():
    """Protect math before Markdown conversion; keep all theory inside the HTML."""
    source=(ASSETS/"theory.md").read_text(encoding="utf-8")
    equations=[]
    def protect(match,block=False):
        key=f"MATHPLACEHOLDER{len(equations)}END"
        tag,cls,delim=("div","math-display","$$") if block else ("span","math-inline","$")
        equations.append((key,f'<{tag} class="{cls}">{delim}{html.escape(match[1])}{delim}</{tag}>'))
        return f"\n\n{key}\n\n" if block else key
    source=re.sub(r"\$\$([\s\S]*?)\$\$",lambda m:protect(m,True),source)
    source=re.sub(r"(?<![\\$])\$(?!\$)([^\n$]+?)\$(?!\$)",protect,source)
    body=markdown.markdown(source,extensions=["tables","fenced_code","toc"])
    body=re.sub(r'^<h1[^>]*>.*?</h1>\s*','',body,count=1,flags=re.S)
    for key,value in equations:body=body.replace(f"<p>{key}</p>",value).replace(key,value)
    body=re.sub(r'\bid="([^"]+)"',lambda m:f'id="theory-{m[1]}"',body)
    body=re.sub(r'href="#([^"]+)"',lambda m:f'href="#theory-{m[1]}"',body)
    return body


def result_page(page,title,summary,loaded):
    rows=study_runs(summary["runs"],page)
    variants=study_runs(summary["runs"],page,CASE_ORDER[0])
    root_count=sum(r["root_success"] for r in rows)
    accurate=sum(r["analytic_pass"] for r in rows)
    description=("Maximum corrections per timestep trial: 1, 3, 5, 10 and 100. The CFL controller is held fixed."
                 if page=="newton" else f"Nine one-parameter controller settings at a fixed Newton cap of {100 if page=='cfl100' else 1}.")
    look=lambda case,cap:next(r for r in rows if r["case"]==case and r["cap"]==cap)
    if page=="newton":
        a,b=look("shock35",1),look("shock35",100)
        finding=f"For shock35, cap 1 takes {a['updates']} updates and {a['linear_solves']} Newton corrections; cap 100 takes {b['updates']} updates and {b['linear_solves']} corrections. Fewer outer updates need not mean less work."
    else:
        base=[r for r in rows if r["cfl_variant"]=="baseline"]
        slow=[r for r in rows if r["cfl_variant"]=="no_ser"]
        worse=sum(next(r for r in slow if r["case"]==b["case"])["updates"]>b["updates"] for b in base)
        finding=f"Disabling SER increases accepted updates in {worse}/{len(base)} saved comparisons. Configured CFL limits and actual accepted values differ; inspect both work and convergence histories."
    metric_options=[("updates","Residual vs accepted updates"),("work","Residual vs total Newton work"),
                    ("time","Residual vs recorded solve time"),("cfl","Actual CFL histories"),
                    ("pressure","Final pressure profiles"),("inner","Actual inner corrections per accepted step")]
    options=''.join(f'<option value="{key}">{name}</option>' for key,name in metric_options)
    records=[]
    for r in rows:
        label=f"Cap {r['cap']}" if page=="newton" else r["label"]
        records.append([TITLES[r["case"]],html.escape(label),str(r["updates"]),str(r["linear_solves"]),
            str(r["rejected_steps"]),f"{r['inner_converged_accepted']}/{r['updates']}",fmt(r["solve_s"]),
            fmt(r["residual_rms"]),fmt(r["mass_imbalance"]),flag(r["same_baseline_root"]),flag(r["analytic_pass"])])
    body=f'''<section class="report-page" id="{page}" data-report-page="{page}" {'hidden' if page!='newton' else ''}>
<div class="hero"><div class="hero-copy"><div class="eyebrow">Implicit integration · retained sensitivity evidence</div><h1>{html.escape(title)}</h1><p>{description}</p><div class="hero-meta"><span>{summary['settings']['points']} points including two ghosts</span><span>Perfect-gas Air backend</span><span>Implicit Euler / Newton / banded</span><span>No simulations rerun</span></div></div></div>
<div class="study-stats"><span><strong>{root_count}/{len(rows)}</strong> spatial roots</span><span><strong>{sum(r['same_baseline_root'] for r in rows)}/{len(rows)}</strong> baseline agreements</span><span><strong>{accurate}/{len(rows)}</strong> analytical checks</span><span><strong>1</strong> timed sample per configuration</span></div>
<div class="callout"><strong>Observed finding.</strong>&nbsp;{finding}</div>
<div class="method-grid"><div class="method-card"><h3>Caps, not forced iteration counts</h3><p>The inner relative-update tolerance is 10<sup>−8</sup>. The saved counts show actual corrections, including rejected attempts. A steady root can be reached while some accepted timesteps remain cap-limited.</p></div><div class="method-card"><h3>Single recorded timings</h3><p>Time includes controller, counter monitoring and rejected work, excluding compilation and final diagnostics. These single measurements are descriptive; they are not repeated timing distributions or validated speedup estimates.</p></div></div>
<section id="{page}-convergence"><div class="section-head"><div><div class="eyebrow">01 / CONVERGENCE AND FINAL FIELDS</div><h2>Five operating points, one spatial method</h2><p>Each line is an actual saved solve. Legend entries toggle a variant across all panels. The pressure view overlays the retained analytical reference and large hollow numerical markers.</p></div></div>
<div class="panel"><div class="report-controls"><div><label for="{page}-metric">View</label><select id="{page}-metric">{options}</select></div><div><label for="{page}-skip">Points skipped between pressure markers</label><input id="{page}-skip" type="number" min="0" step="1" value="5"></div><div class="inline-check"><input id="{page}-rejects" type="checkbox" checked><label for="{page}-rejects">Show rejected trials in CFL/inner views</label></div><button id="{page}-reset" type="button">Reset plots</button></div><div id="{page}-histories" class="five-case-plot"></div><div class="caption">Residual: original fixed-scale spatial RMS; dotted line: 10<sup>−8</sup>. Outer updates exclude rejected trials. Work axes include all attempted Newton corrections. Pressure markers: zero skipped points marks every physical point. All numerical curves use the same MUSCL reconstruction; no constant-reconstruction data are present in this sensitivity dataset.</div></div></section>
<section id="{page}-timing"><div class="section-head"><div><div class="eyebrow">02 / TOTAL WORK AND RECORDED TIME</div><h2>Compare all corrections with elapsed solve time</h2><p>One bar per saved configuration, without invented error bars or repeat statistics. Compilation/first-call measurements are retained separately in the provenance metadata.</p></div></div><div class="panel"><div id="{page}-work" class="work-plot"></div></div></section>
<section id="{page}-outcomes"><div class="section-head"><div><div class="eyebrow">03 / SAVED OUTCOMES</div><h2>Root convergence and analytical accuracy are separate</h2><p>The sonic-limit analytical discrepancy is retained. A small equation residual does not remove that error. Baseline agreement uses the recorded 10<sup>−4</sup> RMS distance in scaled primitive variables.</p></div></div>'''
    body+=table(["Physical case","Variant","Updates","Newton corrections","Rejected trials","Inner criterion met","Time [s], n=1","Spatial RMS","Mass imbalance","Same baseline","Analytical"],records)
    body+='<div class="download-links">'+download('Download this page’s saved outcomes CSV',page+'_outcomes.csv',csv_bytes(rows),'text/csv')+'</div></section></section>'
    return body


def audit_html(report,output):
    class Inspector(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.links=[];self.deps=[];self.downloads=[];self.pages=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if 'href' in a:self.links.append(a['href'])
            if a.get('data-report-page'):self.pages.append(a['data-report-page'])
            if 'download' in a:
                raw=base64.b64decode(a['href'].split(',',1)[1],validate=True)
                self.downloads.append(dict(name=a['download'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            for key in ('src','srcset','poster'):
                if a.get(key) and not a[key].startswith(('data:','#')):self.deps.append(a[key])
            if tag=='link' and a.get('href') and not a['href'].startswith('data:'):self.deps.append(a['href'])
    doc=Inspector();doc.feed(report.read_text(encoding='utf-8'))
    broken=[u for u in doc.links if (u.startswith('#') and u[1:] not in doc.ids)
            or not u.startswith(('#','data:','https:','http:','mailto:'))]
    result=dict(pass_=len(doc.ids)==len(set(doc.ids)) and not broken and not doc.deps
                and doc.pages==['newton','cfl100','cfl1','theory'],
                standalone_html=True,internal_pages=doc.pages,unique_ids=len(doc.ids)==len(set(doc.ids)),
                ids=len(doc.ids),broken_links=broken,external_resources=doc.deps,embedded_downloads=doc.downloads,
                browser_runtime_checked=False,note='Structural portability audit, not a claim of browser rendering.')
    (output/'report_structure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['pass_']:raise ValueError('Standalone report structural audit failed')
    return result


def render(output,destination):
    output,destination=output.resolve(),destination.resolve()
    if not output.is_relative_to(STUDY.resolve()) or destination.parent!=STUDY.resolve():
        raise ValueError('Read study-local output data and place the HTML directly at the study root')
    if json.loads((output/'summary.json').read_text(encoding='utf-8')).get('schema') == 'shared_fvm_sensitivity_v1':
        from .current_report import render as render_current
        return render_current(output)
    before=fingerprints(output)
    summary,loaded=load_saved(output)
    evidence=audit_saved(summary,loaded)
    if not evidence['pass_']:raise ValueError('Saved-record consistency audit failed; inspect the original files')
    (output/'saved_results_audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    data=payload(summary,loaded)
    data['palettes']={page:[to_hex(colormaps['plasma'](.2+.6*i/max(1,len(study_runs(summary['runs'],page,CASE_ORDER[0]))-1)))
                           for i in range(len(study_runs(summary['runs'],page,CASE_ORDER[0])))] for page,_ in PAGES}
    bodies=[result_page(page,title,summary,loaded) for page,title in PAGES]
    audit_bytes=json.dumps(evidence,indent=2).encode('utf-8')
    provenance=dict(settings=summary['settings'],versions=summary['versions'],source_sha256=summary['source_sha256'],
                    compilation=summary['compilation'],render_source_sha256=summary.get('render_source_sha256'))
    bodies.append('''<section id="theory" class="report-page theory-content" data-report-page="theory" hidden><div class="hero"><div class="hero-copy"><div class="eyebrow">Method, interpretation and provenance</div><h1>What the implicit iterations solve</h1><p>Physics, discretization, controller and verification definitions for the saved campaign. This report is a presentation of retained evidence, not a new numerical validation of the current source.</p></div></div>'''+notes_html())
    bodies.append(f'''<h2 id="theory-saved-audit">Saved-record consistency audit</h2><p>{evidence['unique_runs']} unique runs; {evidence['original_spatial_roots']} recorded spatial roots; {evidence['baseline_agreements']} baseline agreements. Analytical accuracy passes in {evidence['analytical_passes']} runs and fails in {len(evidence['analytical_failures'])}. Shared baselines appear on multiple pages but are counted only once.</p><div class="download-links">'''+
        download('Original summary JSON','summary.json',(output/'summary.json').read_bytes(),'application/json')+
        download('All original outcomes CSV','results.csv',(output/'results.csv').read_bytes(),'text/csv')+
        download('Saved-record audit JSON','saved_results_audit.json',audit_bytes,'application/json')+'</div>')
    bodies.append('<details><summary>Original recorded software, source hashes and compilation costs</summary><pre class="record-pre">'+html.escape(json.dumps(provenance,indent=2))+'</pre></details>')
    bodies.append('<details><summary>Retained historical setup note</summary><pre class="record-pre">'+html.escape((ASSETS/'historical_setup.md').read_text(encoding='utf-8'))+'</pre></details></section>')
    nav=[]
    for page,title in PAGES:
        nav.append(f'<div data-report-nav="{page}" {"hidden" if page!="newton" else ""}><a href="#{page}"><span>01</span>{title}</a><a href="#{page}-convergence"><span>02</span>Convergence and pressure</a><a href="#{page}-timing"><span>03</span>Work and timing</a><a href="#{page}-outcomes"><span>04</span>All saved outcomes</a></div>')
    nav.append('<div data-report-nav="theory" hidden>'+''.join(f'<a href="#theory-{key}"><span>{i+1:02}</span>{label}</a>' for i,(key,label) in enumerate((('physical-problem','Physical setup'),('conservative-equations-and-wave-speeds','Euler equations'),('implicit-pseudo-time-integration','Implicit integration'),('cfl-and-ser-controller','CFL / SER controller'),('original-residual-root-agreement-and-analytical-accuracy','Verification definitions'),('saved-audit','Audit and provenance'))))+'</div>')
    css=(SHARED/'report.css').read_text(encoding='utf-8')+'\n'+(ASSETS/'report.css').read_text(encoding='utf-8')
    vendor=SHARED/'vendor'
    runtime=lambda name:script_safe(gzip.decompress((vendor/name).read_bytes()).decode())
    math_config="""<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{typeset:false}};</script>"""
    app=(ASSETS/'report.js').read_text(encoding='utf-8')
    scripts='<script id="report-data" type="application/json">'+json_safe(data)+'</script>'+math_config+'<script>'+runtime('plotly-2.35.2.min.js.gz')+'</script><script>'+runtime('mathjax-3.2.2-tex-svg.js.gz')+'</script><script>'+script_safe(app)+'</script>'
    links=''.join(f'<a href="#{page}" data-report-tab="{page}" {"aria-current=page" if page=="newton" else ""}>{i+1:02} · {title}</a>' for i,(page,title) in enumerate([*PAGES,('theory','Theory')]))
    licenses='\n\n'.join((vendor/name).read_text(encoding='utf-8') for name in ('PLOTLY_LICENSE.txt','MATHJAX_LICENSE.txt'))
    text='''<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; worker-src blob:; object-src 'none'; base-uri 'none'"><title>Implicit integration · Euler nozzle</title><script>try{document.documentElement.dataset.theme=new URLSearchParams(location.search).get('theme')||localStorage.getItem('implicit-integration-report-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');}catch{}</script><style>'''+css+'''</style></head><body><header class="header"><a class="brand" href="#newton">Euler nozzle <small>Implicit integration</small></a><nav class="page-switch" aria-label="Report pages">'''+links+'''</nav><div class="header-actions"><button id="print-report">Print / PDF</button><div class="theme-switch"><button data-theme-choice="light" aria-pressed="true">Light</button><button data-theme-choice="dark" aria-pressed="false">Dark</button></div></div></header><aside class="sidebar"><div class="eyebrow" id="sidebar-label">Newton caps</div><nav>'''+''.join(nav)+'''</nav><div class="sidebar-foot">Single-file offline report<br>105 unique saved solves<br>No simulations rerun</div></aside><main><noscript>Enable JavaScript for interactive plots and equations; no network connection is required.</noscript>'''+''.join(bodies)+'''<footer><span>Implicit integration · retained EulerNozzle sensitivity evidence</span><span>Share this HTML file · resources and downloads embedded</span></footer><details><summary>Embedded library licenses</summary><pre class="record-pre">'''+html.escape(licenses)+'''</pre></details></main>'''+scripts+'''</body></html>'''
    destination.write_bytes(text.encode('utf-8'))
    structure=audit_html(destination,output)
    after=fingerprints(output)
    if before!=after:raise ValueError('Numerical records changed during report generation')
    numerical_imports=[name for name in sys.modules if name=='jax' or name.startswith(('jax.','nozzle_hyperbolic.'))]
    if numerical_imports:raise ValueError('Report generation unexpectedly imported numerical solver modules')
    manifest=dict(report=destination.name,report_sha256=digest(destination),standalone_html=True,
        internal_pages=['newton','cfl100','cfl1','theory'],numerical_files_checked=len(before),numerical_data_unchanged=before==after,
        input_sha256=before,unique_runs=len(summary['runs']),saved_consistency_pass=evidence['pass_'],
        structural_audit_pass=structure['pass_'],numerical_solver_imports=numerical_imports,
        renderer_sha256=digest(Path(__file__)),report_data_sha256=digest(Path(__file__).with_name('report_data.py')),
        assets_sha256={p.relative_to(ASSETS).as_posix():digest(p) for p in ASSETS.rglob('*') if p.is_file()},
        reused_assets=str(SHARED.relative_to(STUDY.parent)),timing_note='One recorded solve per configuration; no repeated speed benchmark')
    (output/'html_report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Wrote {destination}: {len(summary["runs"])} unique saved runs, {len(before)} unchanged numerical files; no solver imports or simulations.')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=STUDY/'outputs')
    parser.add_argument('--destination',type=Path,default=STUDY/'REPORT.html')
    args=parser.parse_args();render(args.output,args.destination)
