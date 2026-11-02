"""Exercise an isolated report in installed headless Edge, without Playwright.

This copies only the HTML into outputs/html_audit/, disables DNS resolution,
uses a fresh browser profile, and never imports or calls a numerical solver.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

STUDY=Path(__file__).resolve().parents[1]

HARNESS=r'''
(async()=>{
 const failures=[],records=[];
 const assert=(value,label)=>{if(!value)throw new Error(label);};
 const finish=()=>{const el=document.createElement('script');el.id='implicit-browser-audit';el.type='application/json';el.textContent=JSON.stringify({passed:failures.length===0,failures,records,errors:window.__reportErrors});document.body.appendChild(el);};
 try{
  for(let i=0;i<300&&!window.implicitIntegrationReport;i++)await new Promise(r=>setTimeout(r,20));
  const api=window.implicitIntegrationReport;assert(api,'Report API initialized');await api.ready();
  assert(Object.keys(api.data.runs).length>0,'Embedded numerical records');
  for(const page of ['newton','cfl100','cfl1']){
   location.hash=page;await new Promise(r=>setTimeout(r,40));await api.ready();
   for(const theme of ['light','dark']){
    document.querySelector('[data-theme-choice="'+theme+'"]').click();await api.ready();
    const metrics=theme==='light'?['updates','work','time','cfl','pressure','inner']:['updates','pressure'];
    for(const metric of metrics){
     const select=document.getElementById(page+'-metric');select.value=metric;select.dispatchEvent(new Event('change'));await api.ready();
     const graph=document.getElementById(page+'-histories');assert(graph.data.length>0,page+' rendered '+metric);
     const first=api.data.studies[page].map(id=>api.data.runs[id]).find(r=>r.case===api.data.order[0]);
     const label=page==='newton'?'Cap '+first.cap:first.label;
     const trace=graph.data.find(t=>t.name===label&&t.mode==='lines'&&t.xaxis==='x');
     assert(trace,'Selected source trace '+page+' '+metric);
     const expected=metric==='pressure'?first.pressure_Pa.slice(1,-1).map(v=>v/1000):metric==='cfl'?first.history.effective_cfl:metric==='inner'?first.history.inner_solves:first.history.residual_rms;
     assert(JSON.stringify(trace.y)===JSON.stringify(expected),'Saved plot arrays exact '+page+' '+metric);
     api.data.order.forEach((cid,ci)=>{
      const xa=ci?'x'+(ci+1):'x',meta=api.data.cases[cid];
      api.data.studies[page].map(id=>api.data.runs[id]).filter(r=>r.case===cid).forEach(r=>{
       const label=page==='newton'?'Cap '+r.cap:r.label;
       const line=graph.data.find(t=>t.name===label&&t.mode==='lines'&&t.xaxis===xa);
       const y=metric==='pressure'?r.pressure_Pa.slice(1,-1).map(v=>v/1000):metric==='cfl'?r.history.effective_cfl:metric==='inner'?r.history.inner_solves:r.history.residual_rms;
       const x=metric==='pressure'?meta.x_m.slice(1,-1):metric==='time'?r.history.elapsed_s:metric==='work'?r.history.cumulative_linear_solves:r.history.accepted_index;
       assert(line&&JSON.stringify(line.x)===JSON.stringify(x)&&JSON.stringify(line.y)===JSON.stringify(y),'Every saved trace exact '+r.run_id+' '+metric);
      });
     });
     if(metric==='pressure'){
      const field=document.getElementById(page+'-skip');field.value='0';field.dispatchEvent(new Event('input'));await api.ready();
      const marker=graph.data.find(t=>t.name===label&&t.mode==='markers'&&t.xaxis==='x');
      assert(marker.x.length===first.pressure_Pa.length-2,'Zero-skip pressure markers');
      field.value='11';field.dispatchEvent(new Event('input'));await api.ready();
      assert(graph.data.find(t=>t.name===label&&t.mode==='markers'&&t.xaxis==='x').x.length===Math.ceil((first.pressure_Pa.length-2)/12),'User marker skip');
      field.value='-1';field.dispatchEvent(new Event('input'));await api.ready();assert(field.getAttribute('aria-invalid')==='true','Invalid marker skip rejected');
      field.value='5';field.dispatchEvent(new Event('input'));await api.ready();
     }
     const visible=[...document.querySelectorAll('[data-report-page]')].filter(p=>!p.hidden);
     assert(visible.length===1&&visible[0].id===page,'Internal page visibility');
     assert(document.querySelectorAll('[data-report-tab][aria-current="page"]').length===1,'Single active tab');
     assert(document.documentElement.scrollWidth<=innerWidth+1,'No whole-page horizontal overflow');
     records.push({page,theme,metric,plot_traces:graph.data.length});
    }
   }
  }
  location.hash='theory';await new Promise(r=>setTimeout(r,40));await api.ready();
  assert(document.querySelectorAll('#theory mjx-container').length>20,'Theory equations rendered');
  assert(document.querySelectorAll('[data-mml-node="merror"]').length===0,'No equation errors');
  const downloads=[...document.querySelectorAll('a[download]')];assert(downloads.length===6,'Six embedded downloads');
  assert(downloads.every(a=>a.href.startsWith('data:')),'No neighboring download files');
  assert(window.__reportErrors.length===0,'No JavaScript errors: '+JSON.stringify(window.__reportErrors));
  records.push({page:'theory',math:document.querySelectorAll('#theory mjx-container').length,downloads:downloads.length});
 }catch(e){failures.push(String(e.stack||e));}
 finish();
})();
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=STUDY/'REPORT.html')
    parser.add_argument('--edge',type=Path,default=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'))
    args=parser.parse_args()
    if not args.edge.is_file():raise SystemExit('Installed Edge not found; pass --edge with its executable path')
    report=args.report.resolve()
    if not report.is_relative_to(STUDY.resolve()):raise ValueError('Audit only a report inside this study')
    output=STUDY/'outputs/html_audit';output.mkdir(parents=True,exist_ok=True)
    original=report.read_bytes();document=original.decode('utf-8')
    catcher="<script>window.__reportErrors=[];addEventListener('error',e=>window.__reportErrors.push(String(e.message)));addEventListener('unhandledrejection',e=>window.__reportErrors.push(String(e.reason)));</script>"
    document=document.replace('<head>','<head>'+catcher,1).replace('</body>','<script>'+HARNESS+'</script></body>',1)
    isolated=output/'isolated_report.html';isolated.write_text(document,encoding='utf-8')
    command=[str(args.edge),'--headless=new','--disable-gpu','--no-first-run','--disable-background-networking',
             '--disable-component-update','--disable-extensions','--disable-sync','--host-resolver-rules=MAP * ~NOTFOUND',
             '--user-data-dir='+str((output/'browser_profile').resolve()),'--dump-dom','--virtual-time-budget=60000',isolated.resolve().as_uri()]
    with (output/'dom.html').open('w',encoding='utf-8') as stdout,(output/'browser_stderr.log').open('w',encoding='utf-8') as stderr:
        result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    dom=(output/'dom.html').read_text(encoding='utf-8')
    match=re.search(r'<script id="implicit-browser-audit" type="application/json">(.*?)</script>',dom,re.S)
    if not match:raise RuntimeError(f'Browser audit did not complete (exit {result.returncode}); inspect {output}')
    audit=json.loads(match[1]);audit.update(report_sha256=hashlib.sha256(original).hexdigest(),standalone_copy=True,
                                         network_disabled=True,driver='Installed headless Edge CLI; isolated profile')
    # Capture unmodified presentation states after the scripted interaction audit.
    display=output/'display_report.html';display.write_bytes(original)
    base=command[:-3]
    screenshots=[]
    for page,theme in (('newton','light'),('theory','dark')):
        image=output/f'{page}_{theme}.png'
        image_command=base+['--window-size=1600,1600','--virtual-time-budget=8000',
                           '--screenshot='+str(image.resolve()),display.resolve().as_uri()+f'?theme={theme}#{page}']
        with (output/f'{page}_{theme}_stderr.log').open('w',encoding='utf-8') as errors:
            subprocess.run(image_command,stdout=subprocess.DEVNULL,stderr=errors,timeout=60,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if image.is_file():screenshots.append(image.name)
    audit['screenshots']=screenshots
    assert report.read_bytes()==original,'Browser audit changed the source report'
    (output/'browser_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='records'},indent=2))
    raise SystemExit(0 if audit['passed'] else 1)


if __name__=='__main__':main()
