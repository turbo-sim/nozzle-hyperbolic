"""Audit offline report packaging and prepare an isolated browser runtime check.

The optional browser copy exercises presentation controls only. It does not
load numerical solvers, change campaign results, or rerun a simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import STUDY, audit_html


RUNTIME_AUDIT = r"""<script>
window.addEventListener('load',async()=>{
 const tests=[], check=(name,pass,detail='')=>tests.push({name,pass:!!pass,detail});
 const wait=async()=>{for(let i=0;i<200&&!window.benchmarkPlotsReady;i++)await new Promise(r=>setTimeout(r,20));};
 const id=name=>document.getElementById(name);
 const params=new URLSearchParams(location.search);
 if(params.get('capture')==='1'){
  const style=document.createElement('style');style.textContent='html{scroll-behavior:auto!important}.page{animation:none!important;opacity:1!important;transform:none!important}';document.head.append(style);
  await wait();window.benchmarkReport.showPage(params.get('page')||'summary');await wait();
  await new Promise(r=>setTimeout(r,300));
  const target={flow:'roe-flow-validation','real-gas':'roe-real-gas-section',algebra:'roe-validation'}[params.get('section')];
  if(target&&id(target)&&!id(target).hidden)window.scrollTo({top:id(target).getBoundingClientRect().top+window.scrollY,behavior:'instant'});
  await new Promise(r=>setTimeout(r,300));
  const pre=document.createElement('pre');pre.id='capture-metrics';pre.hidden=true;pre.textContent=JSON.stringify({scrollY:window.scrollY,target,targetTop:target?id(target).getBoundingClientRect().top:null,page:params.get('page')||'summary',errors:window.__benchmarkErrors});document.body.append(pre);
  return;
 }
 await wait();
 check('application initialized',window.benchmarkReportReady===true);
 check('summary visible',!id('page-summary').hidden&&id('page-case').hidden);
 check('four report pages',document.querySelectorAll('[data-page]').length===4);
 check('summary table populated',id('summary-table').querySelectorAll('tr').length>0);
 check('summary timing plot present',!!id('speed-chart').data||!!id('speed-chart').querySelector('.plot-empty'));
 const cases=[...id('case-select').options].map(o=>o.value),visited=[];
 for(const value of cases){
  window.benchmarkReport.openCase(value);await wait();
  const rows=window.benchmarkReport.selectedRows();
  const expectedFields=rows.some(r=>Object.keys(r.state.fields).length);
  visited.push({case_id:value,selection:id('case-select').value===value,field_plot:!expectedFields||!!id('field-chart').data,
                reference_resolved:rows.every(r=>!r.raw.reference?.shared_reference_sha256||(r.reference.x.length>0&&Object.keys(r.reference.fields).length>0))});
 }
 check('every case can be selected',visited.every(x=>x.selection&&x.field_plot),visited);
 check('shared analytical references resolve',visited.every(x=>x.reference_resolved));
 check('case page activated',!id('page-case').hidden&&id('tab-case').getAttribute('aria-selected')==='true');
 if(cases.length){
  const choice=id('field-select');if(choice.options.length>1){choice.selectedIndex=1;choice.dispatchEvent(new Event('change'));await wait();check('field selector updates plot title',id('field-chart-title').textContent.length>0);}
  const mesh=id('case-grid');if(mesh.options.length>1){mesh.selectedIndex=0;mesh.dispatchEvent(new Event('change'));await wait();check('mesh selector updates selected results',window.benchmarkReport.selectedRows().every(r=>String(r.cells)===mesh.value));}
  const flux=id('case-flux');if(flux.options.length>1){
   flux.selectedIndex=1;flux.dispatchEvent(new Event('change'));await wait();
   check('flux selector keeps one flux',window.benchmarkReport.selectedRows().every(r=>r.flux===flux.value));
   check('refinement curves distinguish flux',!id('refinement-chart').data||id('refinement-chart').data.every(t=>new Set(t.x).size===t.x.length));
   flux.value='all';flux.dispatchEvent(new Event('change'));await wait();
  }
  check('flux options match saved data',id('case-flux').options.length===window.benchmarkReport.fluxes.length+1);
  const selected=window.benchmarkReport.selectedRows()[0];
  const initial=window.benchmarkReport.baselineComparisons().find(r=>r.case_id===selected?.caseId&&r.backend===selected?.backend&&r.matched);
  if(initial){
   const target=window.benchmarkReport.selectedRows().find(r=>r.caseId===initial.case_id&&r.backend===initial.backend);
   const previous=target.raw.scheme;target.raw.scheme={report_audit_changed:true};
   check('baseline rejects a changed scheme even when settings exist',!window.benchmarkReport.baselineComparisons().find(r=>r.case_id===target.caseId&&r.backend===target.backend&&r.match?.index===initial.match?.index)?.matched);
   if(previous===undefined)delete target.raw.scheme;else target.raw.scheme=previous;
  }
 }
 const before=document.documentElement.dataset.theme;id('theme-toggle').click();await wait();
 check('theme toggle updates layout',document.documentElement.dataset.theme!==before);
 id('theme-toggle').click();await wait();
 id('tab-theory').click();
 check('Roe theory page activated',!id('page-theory').hidden&&id('tab-theory').getAttribute('aria-selected')==='true');
 check('Roe theory equations render natively offline',id('page-theory').querySelectorAll('math').length>=7&&[...id('page-theory').querySelectorAll('math')].every(node=>node.getBoundingClientRect().width>0));
 check('Roe theory includes implementation policies',id('roe-theory-secant').textContent.includes('4096')&&id('roe-theory-secant').textContent.includes('C²')&&id('roe-theory-entropy').textContent.includes('does not clip'));
 id('tab-theory').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
 check('tab keyboard wraps across four pages',!id('page-summary').hidden&&id('tab-summary').getAttribute('aria-selected')==='true');
 id('tab-protocol').click();
 check('protocol page and data visible',!id('page-protocol').hidden&&id('protocol-json').textContent.length>2&&id('provenance-json').textContent.length>2);
 const validation=JSON.parse(id('benchmark-data').textContent).roe_validation;
 if(validation){
  check('Roe validation section visible',!id('roe-validation').hidden);
  check('Roe pair summary matches saved audit',id('roe-validation-cards').querySelector('.stat-value').textContent.replace(/[^0-9]/g,'')===String(validation.summary.pair_checks_passed));
  check('Roe model audits retained',id('roe-validation-table').querySelectorAll('tr').length===validation.pair_audits.length);
  check('Roe host finite-difference limitation visible',id('roe-validation-jvp').textContent.includes('one-sided')&&id('roe-validation-jvp').textContent.includes('finite differences'));
  check('Roe exact audit archive declared',validation.evidence.archive_entry==='campaign/validation/roe_audit.json'&&validation.evidence.sha256.length===64);
  check('Roe audit download control available',!!id('roe-audit-download'));
 }
 const flow=JSON.parse(id('benchmark-data').textContent).roe_flow_validation;
 if(flow){
  check('flow verification section visible',!id('roe-flow-validation').hidden);
  if(flow.sod){
   check('Sod saved rows retained',id('roe-sod-table').querySelectorAll('tr').length===flow.sod.records.length);
   const numerical=flow.sod.assessment.numerical_checks_passed,strict=flow.sod.assessment.analytic_boundary_transfer_checks_passed,total=flow.sod.records.length;
   check('Sod numerical and strict criteria distinguished',id('roe-sod-summary').textContent.includes(`${numerical}/${total}`)&&id('roe-sod-summary').textContent.includes(`${strict}/${total}`));
   check('Sod strict failures visible',[...id('roe-sod-table').querySelectorAll('tr')].filter(row=>row.textContent.includes('failed; retained')).length===total-strict);
   if(flow.sod_figure)check('Sod figure embedded offline',id('roe-sod-image').src.startsWith('data:image/png;base64,')&&!id('roe-sod-figure').hidden);
  }
  if(flow.real_gas){
   check('all real-gas attempts retained',id('roe-real-gas-table').querySelectorAll('tr').length===flow.real_gas.length);
   const rows=[...id('roe-real-gas-table').querySelectorAll('tr')];
   check('real-gas budget stops preserved',rows.filter(row=>row.textContent.includes('budget stop; incomplete')).length===flow.real_gas.filter(row=>row.budget_incomplete).length);
   check('real-gas steady passes distinct',rows.filter(row=>row.textContent.includes('steady verified')).length===flow.real_gas.filter(row=>row.steady_verified).length);
  }
  check('flow evidence metadata retained',flow.evidence.every(row=>row.sha256.length===64)&&!!id('roe-flow-download'));
 }
 id('download-toggle').click();check('download menu opens',!id('download-menu').hidden);
 document.body.click();check('download menu closes',id('download-menu').hidden);
 check('no external script or stylesheet',!document.querySelector('script[src],link[rel="stylesheet"]'));
 check('no JavaScript errors',!window.__benchmarkErrors.length,window.__benchmarkErrors);
 window.benchmarkReport.showPage(new URLSearchParams(location.search).get('page')||'summary');await wait();
 const section=new URLSearchParams(location.search).get('section');
 const target={flow:'roe-flow-validation','real-gas':'roe-real-gas-section',algebra:'roe-validation'}[section];
 if(target&&id(target)&&!id(target).hidden)id(target).scrollIntoView({block:'start'});
 check('no horizontal document overflow',document.documentElement.scrollWidth<=innerWidth+1,{document:document.documentElement.scrollWidth,viewport:innerWidth});
 const pre=document.createElement('pre');pre.id='benchmark-runtime-audit';pre.hidden=true;pre.textContent=JSON.stringify(tests);document.body.append(pre);
 document.body.dataset.runtimeAuditPassed=String(tests.every(t=>t.pass));
 if(new URLSearchParams(location.search).get('page')==='case'&&cases.length)window.benchmarkReport.openCase(cases[0]);
});
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--prepare-browser", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    html = args.input.read_text(encoding="utf-8")
    audit = audit_html(html)
    if args.prepare_browser:
        capture = "<script>window.__benchmarkErrors=[];addEventListener('error',e=>window.__benchmarkErrors.push(e.message));addEventListener('unhandledrejection',e=>window.__benchmarkErrors.push(String(e.reason)));</script>"
        browser = html.replace("</head>", capture + "</head>").replace("</body>", RUNTIME_AUDIT + "</body>")
        args.prepare_browser.parent.mkdir(parents=True, exist_ok=True)
        args.prepare_browser.write_text(browser, encoding="utf-8", newline="\n")
        audit["browser_copy"] = str(args.prepare_browser)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
