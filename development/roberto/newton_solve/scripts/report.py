"""Build the standalone report from retained Newton/PTC/custom solver evidence.

No numerical solver or current source model is imported. Historical comparison,
performance and custom adapted-start records remain read-only during rendering.
"""

import argparse
import base64
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

import markdown
from development.roberto.implicit_integration.scripts.report import (
    csv_bytes, download, flag, json_safe, script_safe, table,
)
from .report_data import STUDY, NAMES, audit, cold_runs, load, make_payload, numerical_files, sha

ASSETS=Path(__file__).resolve().parent/'report_assets'
SHARED=STUDY.parent/'BC_residual/scripts/report_assets'
IMPLICIT=STUDY.parent/'implicit_integration/scripts/report_assets'
PAGES=(('cold','Cold starts'),('performance','PTC cost'),('adapted','Adapted starts'),
       ('theory','Mechanism'),('notes','Earlier notes'))


def f(value):
    if value is None:return '—'
    return f'{value:.4g}' if isinstance(value,float) else html.escape(str(value))


def note(name,prefix):
    text=(ASSETS/'notes'/name).read_text(encoding='utf-8');math=[]
    def protect(match,block=False):
        key=f'MATHPLACEHOLDER{len(math)}END';tag,cls,delim=('div','math-display','$$') if block else ('span','math-inline','$')
        math.append((key,f'<{tag} class="{cls}">{delim}{html.escape(match[1])}{delim}</{tag}>'))
        return '\n\n'+key+'\n\n' if block else key
    text=re.sub(r'\$\$([\s\S]*?)\$\$',lambda m:protect(m,True),text)
    text=re.sub(r'(?<![\\$])\$(?!\$)([^\n$]+?)\$(?!\$)',protect,text)
    body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
    body=re.sub(r'^<h1[^>]*>.*?</h1>\s*','',body,count=1,flags=re.S)
    for key,value in math:body=body.replace(f'<p>{key}</p>',value).replace(key,value)
    ids=set(re.findall(r'\bid="([^"]+)"',body))
    body=re.sub(r'\bid="([^"]+)"',lambda m:f'id="{prefix}-{m[1]}"',body)
    def link(match):
        href,label=html.unescape(match[1]),match[2]
        if href.startswith(('https:','http:','data:')):return match[0]
        if href.startswith('#') and href[1:] in ids:return f'<a href="#{prefix}-{href[1:]}">{label}</a>'
        page={'CONVERGENCE_MECHANISM.md':'theory','convergence_mechanism.md':'theory',
              'PTC_PERFORMANCE.md':'performance','ptc_performance.md':'performance',
              'README.md':'cold','REPORT.md':'cold'}.get(Path(href).name)
        if page:return f'<a href="#{page}">{label}</a>'
        return f'<span title="Retained repository artifact: {html.escape(href,quote=True)}">{label}</span>'
    body=re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>',link,body,flags=re.S)
    def image(match):
        path=(ASSETS/'notes'/html.unescape(match[1])).resolve()
        if not path.is_file():path=(STUDY/html.unescape(match[1])).resolve()
        if not path.is_relative_to(STUDY) or not path.is_file():raise ValueError(f'Missing retained note figure: {path}')
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    return re.sub(r'src="([^"]+\.png)"',image,body)


def controls(prefix,options,skip=True,seed=False):
    choices=''.join(f'<option value="{key}">{label}</option>' for key,label in options)
    text=f'<div class="report-controls"><div><label for="{prefix}-metric">View</label><select id="{prefix}-metric">{choices}</select></div>'
    if seed:text+='<div><label for="adapted-seed">Numerical starting field</label><select id="adapted-seed"><option value="adapted">Unmodified adapted expansion</option><option value="outlet_adjusted">Outlet-adjusted layer</option></select></div>'
    if skip:text+=f'<div><label for="{prefix}-skip">Points skipped between pressure markers</label><input id="{prefix}-skip" type="number" min="0" step="1" value="5"></div>'
    return text+'</div>'


def header(page,title,intro,stats=''):
    return f'<section id="{page}" class="report-page" data-report-page="{page}" {"hidden" if page!="cold" else ""}><div class="hero"><div class="hero-copy"><div class="eyebrow">Newton/PTC · retained numerical evidence</div><h1>{title}</h1><p>{intro}</p></div></div>'+stats


def build_bodies(cold,perf,start,evidence):
    bodies=[];outcome=[]
    rows=[]
    for case in cold['cases']:
        for run in cold_runs(case):
            d=run['diagnostics'];method=run['method'];same=run.get('same_root',run['root_success'])
            rows.append([case['name'],NAMES[method],f(run['iterations']),f(d['residual_rms']),flag(run['root_success']),flag(d['analytical_pass']),flag(same),f(run['wall_s'])])
            outcome.append(dict(campaign='cold',case=case['id'],method=method,initialization='simple',
                                iterations=run['iterations'],residual_rms=d['residual_rms'],root=run['root_success'],
                                intended_root=same,analytical_pass=d['analytical_pass'],wall_s=run['wall_s']))
    body=header('cold','Cold starts: the correction path matters',
        'Implicit marching, physical-storage pseudo-transient continuation and controlled custom Newton start independently from identical simple fields. All use the same original spatial equations in this archived comparison.',
        '<div class="study-stats"><span><strong>5/5</strong> implicit spatial roots</span><span><strong>5/5</strong> PTC spatial roots</span><span><strong>3/5</strong> direct Newton spatial roots</span><span>201 points · 597 interior unknowns</span></div>')
    body+='<div class="callout"><strong>Measured result.</strong> PTC reaches both shocked cold-start solutions, while controlled Newton stalls. The retained implicit implementation is much faster. This is evidence for a useful globalization direction, not a matched-implementation performance ranking.</div>'
    body+='<section id="cold-figures"><h2>Five operating points</h2><p>Actual histories are shown without a running-minimum envelope. A cross marks an unconverged endpoint; its pressure curve is a returned diagnostic field, not a steady solution.</p><div class="panel">'+controls('cold',[('updates','Residual vs accepted updates'),('time','Residual vs recorded solve time'),('pressure','Final pressure profiles')])+'<div id="cold-plot" class="five-case-plot"></div><div class="caption">Original fixed-scale spatial RMS target: 10<sup>−8</sup>. Outer updates do not count inner Newton corrections or rejected trials. Numerical pressure curves exclude ghost slots; marker skipping is configurable.</div></div></section>'
    body+='<section id="cold-outcomes"><h2>Root convergence, accuracy and recorded time</h2><p>Single serial observations, not repeated timing benchmarks. The sonic-limit analytical discrepancy remains a failure for all three methods.</p>'+table(['Case','Method','Native/accepted iterations','Original RMS','Spatial root','Analytical','Same implicit root','Solve [s]'],rows)+'</section></section>'
    bodies.append(body)
    work=[]
    for case in cold['cases']:
        cid=case['id'];default=perf['replay'][cid]['default'];one=perf['replay'][cid]['one_inner']
        ptc=next(r for r in case['runs'] if r['method']=='ptc');implicit=case['implicit_references']['simple']
        work.append([case['name'],f(default['updates']),f(default['total_linear_solves']),f(one['updates']),
                     f(one['total_linear_solves']),f(ptc['iterations']),f(ptc['wall_s']/implicit['wall_s'])])
    body=header('performance','Why this PTC prototype costs more',
        'The saved cost investigation separates correction counts, controller effects and dense backend linear algebra. Counter replays and a budget-limited prefix are diagnostic measurements, not new repeated speed benchmarks.')
    body+='<div class="callout">At a matched state and timestep, the first implicit Newton correction and the undamped physical-storage PTC correction agree to a scaled maximum difference of '+f(perf['first_correction']['scaled_max_difference'])+'. The subsequent controllers and nonlinear work differ.</div>'
    body+='<section id="performance-work"><h2>Accepted updates and actual inner work</h2><div class="panel">'+controls('performance',[('inner','Actual implicit inner Newton counts'),('cfl','Accepted CFL: implicit vs PTC'),('residual','Residual trajectories')],skip=False)+'<div id="performance-plot" class="five-case-plot"></div><div class="caption">Implicit replay traces begin at the first accepted timestep. Counts include rejected-trial work in the table; replay timings are not used as benchmarks. The PTC trace belongs to the original cold comparison.</div></div>'+table(['Case','Implicit updates','Implicit Newton solves','One-inner updates','One-inner solves','PTC accepted updates','PTC/implicit recorded time ratio'],work)+'</section>'
    body+='<section id="performance-ablations"><h2>Upstream-shock PTC controller diagnostics</h2><p>The three saved ablations reach the original target. They test selected settings, not a universal controller optimum. Their additional timings are single observations.</p><div class="panel"><div id="ablation-plot" class="single-plot"></div></div></section>'
    body+='<section id="performance-note" class="notes-content"><h2>Retained cost analysis and measurements</h2>'+note('ptc_performance.md','cost')+'</section></section>';bodies.append(body)
    branch_rows=[]
    for case in start['cases']:
        for run in case['runs']:
            d=run['diagnostics'];branch_rows.append([case['id'],run['initialization'],NAMES[run['method']],flag(run['root_success']),flag(run['same_shocked_root']),f(d['exit_M']),f(d['residual_rms']),f(run['wall_s']),f(run.get('worker_elapsed_s'))])
            outcome.append(dict(campaign='adapted',case=case['id'],method=run['method'],initialization=run['initialization'],
                iterations=run['iterations'],residual_rms=d['residual_rms'],root=run['root_success'],
                intended_root=run['same_shocked_root'],analytical_pass=run['analytical_pass'],wall_s=run['wall_s'],worker_elapsed_s=run.get('worker_elapsed_s')))
    body=header('adapted','An adapted expansion can be the wrong root',
        'Retained custom Newton, HYBR, LM and SLSQP calls use an implicitly generated adapted expansion, or a separately labeled outlet-layer adjustment. No analytical profile or imposed shock station initializes a solve.')
    body+='<div class="callout"><strong>Branch selection matters.</strong> Eight unmodified-adapted calls satisfy the equation-residual target while retaining a supersonic outlet and missing the requested shocked solution. The eight outlet-adjusted custom calls do not reach the target either. None of these sixteen calls matches the independent shocked implicit reference.</div>'
    body+='<p class="provenance">These are a separate frozen-baseline experiment, not the same seed/version campaign as the cold comparison. The shared adapted seed required '+f(start['adapted']['implicit']['iterations'])+' accepted implicit updates and '+f(start['adapted']['implicit']['wall_s'])+' s of recorded solve work. Native solve/callback time and whole worker time (including setup/JIT) are shown separately; the seed cost is not silently zero.</p>'
    body+='<section id="adapted-figures"><h2>Two backpressures, two numerical seed rules</h2><div class="panel">'+controls('adapted',[('pressure','Final pressure profiles'),('updates','Original residual vs accepted states'),('time','Original residual vs native solve time')],seed=True)+'<div id="adapted-plot" class="two-case-plot"></div><div class="caption">Pressure references show the requested shocked branch. A residual root with the wrong outlet branch is not counted as success for that target. Unconverged fields remain explicitly labeled.</div></div></section>'
    body+='<section id="adapted-outcomes"><h2>Equation root versus requested shocked root</h2>'+table(['Case','Seed rule','Custom method','Equation root','Requested root','Exit Mach','Original RMS','Native solve/callback [s]','Whole worker [s]'],branch_rows)+'</section></section>';bodies.append(body)
    body=header('theory','Conservative storage changes the direction',
        'A physical and mathematical explanation of the retained Newton, implicit marching and PTC outcomes. Archived equations use u for the same meridional velocity denoted v_m in the report controls.')
    body+='<div class="notes-content">'+note('convergence_mechanism.md','mechanism')+'</div></section>';bodies.append(body)
    body=header('notes','Earlier approaches and useful limits',
        'The retained notes summarize direct-solver, damping, continuation, flux/limiter and initialization investigations. They preserve useful negative results without presenting them as current-source reruns.')
    body+='<div class="callout">Some earlier narrative timings describe separate exploratory runs and differ from the plotted five-case archive. The Results and PTC cost pages identify the records used for their figures. Root convergence, branch choice and analytical accuracy remain separate.</div><div class="notes-content">'+note('earlier_investigations.md','earlier')+'</div>'
    provenance=dict(cold=dict(settings=cold['settings'],versions=cold['versions'],source_hashes=cold['core_source_sha256']),performance=perf['metadata'],adapted=start['metadata'])
    body+='<h2 id="notes-provenance">Saved evidence and reproducibility</h2><p>No numerical solver is imported to build this document. The saved-data audit checks recorded flags, state distances, counters, common seeds and preservation of custom records; it does not recompute current-source PDE residuals.</p><div class="download-links">'
    body+=download('All retained outcomes CSV','newton_outcomes.csv',csv_bytes(outcome),'text/csv')
    body+=download('Saved-data audit JSON','saved_results_audit.json',json.dumps(evidence,indent=2).encode(),'application/json')
    body+='</div><details><summary>Recorded settings, versions and source fingerprints</summary><pre class="record-pre">'+html.escape(json.dumps(provenance,indent=2))+'</pre></details><h3>Regenerate this document without solving</h3><pre>poetry run python -B -m development.roberto.newton_solve.scripts.report</pre></section>';bodies.append(body)
    return ''.join(bodies)


def structural_audit(report,output):
    class Inspector(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.hrefs=[];self.deps=[];self.pages=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if 'href' in a:self.hrefs.append(a['href'])
            if a.get('data-report-page'):self.pages.append(a['data-report-page'])
            for attr in ('src','poster','srcset'):
                if a.get(attr) and not a[attr].startswith(('data:','#')):self.deps.append(a[attr])
            if tag=='link' and a.get('href') and not a['href'].startswith('data:'):self.deps.append(a['href'])
    doc=Inspector();doc.feed(report.read_text(encoding='utf-8'))
    broken=[h for h in doc.hrefs if h.startswith('#') and h[1:] not in doc.ids or not h.startswith(('#','data:','http:','https:','mailto:'))]
    result=dict(pass_=len(doc.ids)==len(set(doc.ids)) and not broken and not doc.deps and doc.pages==[p for p,_ in PAGES],
        standalone_html=True,internal_pages=doc.pages,unique_ids=len(doc.ids)==len(set(doc.ids)),ids=len(doc.ids),broken_links=broken,external_resources=doc.deps)
    (output/'report_structure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['pass_']:raise ValueError('Standalone report structure failed: '+str(result))


def render(output,destination,adapted_output=None):
    preexisting_modules=set(sys.modules)
    output,destination=Path(output).resolve(),Path(destination).resolve();adapted=Path(adapted_output or output/'adapted_initialization').resolve()
    if not output.is_relative_to(STUDY) or not adapted.is_relative_to(STUDY) or destination.parent!=STUDY:
        raise ValueError('Keep input data inside this study and the HTML directly at its root')
    before=numerical_files(output,adapted);cold,perf,start=load(output,adapted)
    cleanup=json.loads((output/'cleanup_manifest.json').read_text(encoding='utf-8'))
    protected={path.replace('\\','/'):value for path,value in cleanup['protected_files_before'].items()}
    for key,path in [('outputs/summary.json',output/'summary.json'),('outputs/performance_analysis/analysis.json',output/'performance_analysis/analysis.json')]:
        if sha(path)!=protected[key]:
            raise ValueError('A different numerical campaign requires review of the retained report narrative')
    for path,value in protected.items():
        if sha(STUDY/path)!=value:raise ValueError('Protected cold/PTC data changed: '+path)
    evidence=audit(cold,perf,start,cleanup)
    if not evidence['pass_']:raise ValueError('Saved-data consistency audit failed')
    (output/'saved_results_audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    data=make_payload(cold,perf,start);body=build_bodies(cold,perf,start,evidence)
    css=(SHARED/'report.css').read_text(encoding='utf-8')+'\n'+(IMPLICIT/'report.css').read_text(encoding='utf-8')+'\n'+(ASSETS/'report.css').read_text(encoding='utf-8')
    vendor=SHARED/'vendor';runtime=lambda name:script_safe(__import__('gzip').decompress((vendor/name).read_bytes()).decode())
    links=''.join(f'<a href="#{p}" data-report-tab="{p}" {"aria-current=page" if p=="cold" else ""}>{i+1:02} · {name}</a>' for i,(p,name) in enumerate(PAGES))
    class Contents(HTMLParser):
        def __init__(self):super().__init__();self.sections=[];self.heading=None;self.items={p:[] for p,_ in PAGES}
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='section':self.sections.append(a.get('data-report-page') or (self.sections[-1] if self.sections else None))
            if tag=='h2' and a.get('id') and self.sections and self.sections[-1]:self.heading=[self.sections[-1],a['id'],'']
        def handle_data(self,data):
            if self.heading:self.heading[2]+=data
        def handle_endtag(self,tag):
            if tag=='h2' and self.heading:
                page,key,label=self.heading;self.items[page].append((key,label));self.heading=None
            if tag=='section' and self.sections:self.sections.pop()
    contents=Contents();contents.feed(body)
    nav=''.join(f'<div data-report-nav="{p}" {"hidden" if p!="cold" else ""}>'+
        ''.join(f'<a href="#{key}"><span>{i+1:02}</span>{html.escape(label)}</a>'
                for i,(key,label) in enumerate([(p,name),*contents.items[p]]))+'</div>' for p,name in PAGES)
    scripts='<script id="report-data" type="application/json">'+json_safe(data)+'</script><script>'+runtime('plotly-2.35.2.min.js.gz')+'</script>'
    scripts+="""<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{typeset:false}};</script>"""
    scripts+='<script>'+runtime('mathjax-3.2.2-tex-svg.js.gz')+'</script><script>'+script_safe((ASSETS/'report.js').read_text(encoding='utf-8'))+'</script>'
    licenses='\n\n'.join((vendor/name).read_text(encoding='utf-8') for name in ('PLOTLY_LICENSE.txt','MATHJAX_LICENSE.txt'))
    text='''<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; worker-src blob:; object-src 'none'; base-uri 'none'"><title>Newton and physical-storage PTC</title><script>try{document.documentElement.dataset.theme=new URLSearchParams(location.search).get('theme')||localStorage.getItem('newton-study-report-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');}catch{}</script><style>'''+css+'''</style></head><body><header class="header"><a class="brand" href="#cold">Euler nozzle <small>Newton and PTC</small></a><nav class="page-switch">'''+links+'''</nav><div class="header-actions"><button id="print-report">Print / PDF</button><div class="theme-switch"><button data-theme-choice="light" aria-pressed="true">Light</button><button data-theme-choice="dark" aria-pressed="false">Dark</button></div></div></header><aside class="sidebar"><div class="eyebrow" id="sidebar-label">Cold starts</div><nav>'''+nav+'''</nav><div class="sidebar-foot">Single-file offline report<br>Retained numerical evidence<br>No simulations rerun</div></aside><main><noscript>Enable JavaScript to use the embedded plots and equations; no internet connection is required.</noscript>'''+body+'''<footer><span>Newton/PTC · archived nozzle investigations</span><span>Share this single HTML file</span></footer><details><summary>Embedded library licenses</summary><pre class="record-pre">'''+html.escape(licenses)+'''</pre></details></main>'''+scripts+'</body></html>'
    destination.write_bytes(text.encode('utf-8'));structural_audit(destination,output)
    after=numerical_files(output,adapted)
    if before!=after:raise ValueError('Retained numerical records changed during report generation')
    imports=[n for n in set(sys.modules)-preexisting_modules if n=='jax' or n.startswith(('jax.','nozzle_hyperbolic.'))]
    if imports:raise ValueError('Unexpected numerical solver imports during report generation')
    manifest=dict(report=destination.name,report_sha256=sha(destination),standalone_html=True,numerical_data_unchanged=True,
        numerical_files_checked=len(before),numerical_input_sha256={str(p.relative_to(STUDY)):v for p,v in before.items()},
        saved_data_audit_pass=evidence['pass_'],numerical_solver_imports=imports,
        renderer_sha256=sha(Path(__file__)),report_data_sha256=sha(Path(__file__).with_name('report_data.py')),
        assets_sha256={p.relative_to(ASSETS).as_posix():sha(p) for p in ASSETS.rglob('*') if p.is_file()})
    (output/'html_report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Wrote {destination}; 15 cold runs and 16 custom adapted runs retained; {len(before)} numerical files unchanged; no simulations.')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=STUDY/'outputs')
    parser.add_argument('--destination',type=Path,default=STUDY/'REPORT.html')
    parser.add_argument('--adapted-output',type=Path)
    args=parser.parse_args();render(args.output,args.destination,args.adapted_output)
