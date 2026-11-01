"""Opt-in presentation adapter for the Roe flux-method report.

The maintained benchmark renderer and its numerical/evidence schema keep their
original defaults. This adapter only changes Roe's shell and saved-data plots.
"""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import re

from development.flux_methods.shared.presentation import theme_css, theme_script

ASSETS = Path(__file__).resolve().parent / "report_assets"


def _replace_once(text, old, new, description):
    if text.count(old) != 1:
        raise ValueError(f"Roe presentation anchor changed: {description}")
    return text.replace(old, new, 1)


def _application(source):
    source = _replace_once(source,
        "  let activePage='summary', selectedCase=cases[0]?.id||'', theme='light';\n"
        "  try{theme=localStorage.getItem('nh-benchmark-theme')||'light';}catch(_error){}\n"
        "  document.documentElement.dataset.theme=theme;\n"
        "  $('theme-toggle').textContent=theme==='dark'?'Light theme':'Dark theme';",
        "  let activePage='summary', selectedCase=cases[0]?.id||'';",
        "legacy theme initialization")
    source = _replace_once(source,
        "  $('theme-toggle').addEventListener('click',()=>{theme=theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;$('theme-toggle').textContent=theme==='dark'?'Light theme':'Dark theme';try{localStorage.setItem('nh-benchmark-theme',theme);}catch(_error){}if(activePage==='summary')renderSummary();if(activePage==='case')renderCase();});",
        "  const fluxTheme=window.FluxReportUI.init({onThemeChange:()=>{if(!window.benchmarkReportReady)return;if(activePage==='summary')renderSummary();else if(activePage==='case')renderCase();else if(activePage==='protocol')renderSodProfiles();}});",
        "legacy theme event")
    source = _replace_once(source,"const pages=['summary','case','protocol','theory'];",
        "const pages=['summary','case','theory','protocol'];", "page order")
    source = _replace_once(source,
        "if(name==='summary')renderSummary();else if(name==='case')renderCase();if(focus)",
        "if(name==='summary')renderSummary();else if(name==='case')renderCase();else if(name==='protocol')renderSodProfiles();if(focus)",
        "protocol plot refresh")
    source = _replace_once(source,
        "      if(data.sod_figure){$('roe-sod-figure').hidden=false;$('roe-sod-image').src=data.sod_figure;}",
        "      const grids=sortedNumbers(sodRows.map(row=>row.cells));setOptions($('roe-sod-grid'),grids,Math.max(...grids),n=>n+' cells');$('roe-sod-grid').addEventListener('change',renderSodProfiles);",
        "saved Sod figure")
    source = _replace_once(source,
        "  function notice(text,kind=''){",
        (ASSETS/"sod_profiles.js").read_text(encoding="utf-8-sig")+"\n  function notice(text,kind=''){",
        "saved Sod plotting helper")
    old_base = "    const base={autosize:true,paper_bgcolor:p.paper,plot_bgcolor:p.paper,font:{family:'Segoe UI, Arial, sans-serif',size:11,color:p.ink},margin:{l:67,r:25,t:26,b:80},showlegend:true,legend:{orientation:'h',y:-.25,x:0,font:{size:10}},hovermode:'closest',xaxis:{gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,title:{font:{size:11}}},yaxis:{gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,title:{font:{size:11}}},...layout};\n    base.xaxis={gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,...layout.xaxis};base.yaxis={gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,...layout.yaxis};"
    source = _replace_once(source,old_base,
        "    const common=window.FluxReportUI.layout('','');\n"
        "    const base={...common,autosize:true,showlegend:true,...layout};\n"
        "    base.xaxis={...common.xaxis,...layout.xaxis};base.yaxis={...common.yaxis,...layout.yaxis};",
        "Plotly layout")
    source=source.replace("'Roe numerical verification'","'Roe flux verification'")
    source=source.replace("sod_figure:data.sod_figure?'Inline PNG; exact original in evidence ZIP':null", "sod_figure:data.sod_figure?'Original PNG retained in evidence ZIP; interactive plots use saved numerical arrays':null")
    source=source.replace("current:{constant:'#dc5e66',muscl_van_leer:'#fca636',muscl_van_leer_smooth:'#fca636'}", "current:{constant:'#6a00a8',muscl_van_leer:'#b12a90',muscl_van_leer_smooth:'#b12a90'}")
    return source


def _evidence_signatures(html):
    data=re.search(r'<script id="benchmark-data" type="application/json">(.*?)</script>',html,re.S)
    if data is None:raise ValueError("Missing original Roe data payload")
    return (sha256(data[1].encode("utf-8")).hexdigest(),
            sorted(sha256(match[1].encode("utf-8")).hexdigest()
                   for match in re.finditer(r'href="(data:[^"]+)"',html)))


def _sod_packet(raw):
    """Copy plotted values exactly from the same original JSON bound to the ZIP."""
    data=json.loads(raw)
    fields=("cells","method","time","target_time","success","passed","classification",
            "final_state","exact_final_conservative_cell_averages","physical_cell_centers")
    rows=[]
    for original in data["records"]:
        n=original["cells"]
        if (len(original["final_state"])!=n+2 or len(original["exact_final_conservative_cell_averages"])!=n
                or len(original["physical_cell_centers"])!=n):
            raise ValueError("Saved Sod profile arrays have inconsistent lengths")
        rows.append({key:original[key] for key in fields})
    packet=dict(schema_version=1,source_sha256=sha256(raw).hexdigest(),
                archive_entry="campaign/validation/roe_sod.json",settings=data["settings"],records=rows,
                interpretation="Exact saved profile arrays; plotting-only perfect-gas conversion, no numerical or EOS evaluation")
    return json.dumps(packet,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c").replace("&","\\u0026")


def unify_report(html, *, sod_raw):
    """Apply the shared report theme without modifying data or archive bytes."""
    original=_evidence_signatures(html)
    html=_replace_once(html,"<body>",'<body data-flux-study="roe">',"body")
    html=_replace_once(html,"NOZZLE-HYPERBOLIC / VERIFICATION LAB", "NOZZLE-HYPERBOLIC / FLUX METHODS", "eyebrow")
    html=_replace_once(html,
        '<button id="theme-toggle" class="button quiet" type="button" aria-label="Switch color theme">Dark theme</button>',
        '<label class="theme-control" for="report-theme"><span>Theme</span><select id="report-theme" aria-label="Color theme"><option value="auto">Automatic</option><option value="light">Light</option><option value="dark">Dark</option></select></label>',
        "theme control")
    nav=re.search(r'<nav class="page-tabs"[^>]*>.*?</nav>',html,re.S)
    if nav is None:raise ValueError("Missing Roe report navigation")
    tabs={key:re.search(r'<button\b[^>]*data-page="'+key+r'"[^>]*>.*?</button>',nav[0],re.S)[0] for key in ("summary","case","theory","protocol")}
    labels={"summary":"Overview","case":"Case explorer","theory":"Theory","protocol":"Protocol &amp; evidence"}
    ordered=[]
    for key,button in tabs.items():
        button=re.sub(r'(?<=>)[^<>]+(?=</button>)',labels[key],button)
        ordered.append(button)
    html=html[:nav.start()]+re.sub(r'>.*</nav>$','>'+''.join(ordered)+'<span id="campaign-stamp"></span></nav>',nav[0],flags=re.S)+html[nav.end():]
    figure=re.search(r'<figure id="roe-sod-figure".*?</figure>',html,re.S)
    if figure is None:raise ValueError("Missing original Sod presentation anchor")
    sod='''<article id="roe-sod-profiles" class="panel prose"><div class="panel-heading"><h3>Saved Sod profiles</h3><p id="roe-sod-profile-caption">Numerical final states compared with exact conservative cell averages.</p></div><div class="filterbar"><label>Physical cells<select id="roe-sod-grid"></select></label></div><div class="chart-grid"><div id="roe-sod-pressure" class="plot"></div><div id="roe-sod-density" class="plot"></div><div id="roe-sod-velocity" class="plot"></div></div></article>'''
    html=html[:figure.start()]+sod+html[figure.end():]
    start=html.index('<section id="page-protocol"')
    theory=html.index('<section id="page-theory"')
    end=html.index('</main>',theory)
    html=html[:start]+html[theory:end]+html[start:theory]+html[end:]
    html=html.replace("04 / THE IMPLEMENTED NUMERICAL METHOD","03 / THE IMPLEMENTED NUMERICAL METHOD")
    html=html.replace("03 / INTERPRET THE EVIDENCE","04 / INTERPRET THE EVIDENCE")
    html=html.replace("nozzle-hyperbolic · Maintained steady verification and performance suite","nozzle-hyperbolic · Numerical flux studies")
    html=html.replace("development/roe_method/roe_theory.md","development/flux_methods/roe/roe_theory.md")
    css='<style id="report-theme-css">'+theme_css()+'</style>'
    css+='<style id="roe-report-css">'+(ASSETS/"roe.css").read_text(encoding="utf-8-sig")+'</style>'
    html=_replace_once(html,"</head>",css+"</head>","common style")
    app=re.search(r'<script id="benchmark-app">(.*?)</script>',html,re.S)
    if app is None:raise ValueError("Missing Roe report application")
    replacement='<script id="roe-sod-data" type="application/json">'+_sod_packet(sod_raw)+'</script>\n<script id="report-theme-runtime">'+theme_script()+'</script>\n<script id="benchmark-app">'+_application(app[1])+'</script>'
    result=html[:app.start()]+replacement+html[app.end():]
    if _evidence_signatures(result)!=original:
        raise ValueError("Roe presentation changed the data payload or original evidence download")
    return result
