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
 const records=[],failures=[];const assert=(v,m)=>{if(!v)throw new Error(m);};
 const finish=()=>{const el=document.createElement('script');el.id='newton-browser-audit';el.type='application/json';el.textContent=JSON.stringify({passed:failures.length===0,records,failures,errors:window.__reportErrors});document.body.appendChild(el);};
 try{
  for(let i=0;i<300&&!window.newtonStudyReport;i++)await new Promise(r=>setTimeout(r,20));
  const api=window.newtonStudyReport;assert(api,'Report API initialized');await api.ready();
  const D=api.data;
  for(const page of ['cold','performance','adapted','theory','notes']){
   location.hash=page;await new Promise(r=>setTimeout(r,40));await api.ready();
   for(const theme of ['light','dark']){
    document.querySelector('[data-theme-choice="'+theme+'"]').click();await api.ready();
    const visible=[...document.querySelectorAll('[data-report-page]')].filter(p=>!p.hidden);
    assert(visible.length===1&&visible[0].id===page,'Exactly one internal page visible');
    assert(document.querySelectorAll('[data-report-tab][aria-current="page"]').length===1,'Single active report tab');
    if(document.documentElement.scrollWidth>innerWidth+1){
     const wide=[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&e.getBoundingClientRect().width>0).slice(-12).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right,text:e.textContent.slice(0,80)}));
     throw new Error('Horizontal overflow '+page+' '+theme+' '+JSON.stringify(wide));
    }
    const metrics=['cold','adapted'].includes(page)?['updates','time','pressure']:page==='performance'?['inner','cfl','residual']:[];
    for(const metric of metrics){
     const select=document.getElementById(page+'-metric');select.value=metric;select.dispatchEvent(new Event('change'));await api.ready();
     const seeds=page==='adapted'?['adapted','outlet_adjusted']:[null];
     for(const seed of seeds){
      if(seed){const field=document.getElementById('adapted-seed');field.value=seed;field.dispatchEvent(new Event('change'));await api.ready();}
      const graph=document.getElementById(page==='performance'?'performance-plot':page+'-plot');assert(graph.data.length>0,'Graphs rendered');
      if(page!=='performance'){
       const cases=page==='cold'?D.cases:D.adapted;
       cases.forEach((c,ci)=>{
        const xa=ci?'x'+(ci+1):'x';
        c.runs.filter(r=>!seed||r.initialization===seed).forEach(r=>{
         const line=graph.data.find(t=>t.mode==='lines'&&t.legendgroup===D.names[r.method]&&t.xaxis===xa);
         const expected=metric==='pressure'?r.diagnostics.fields.p.slice(1,-1).map(v=>v/1000):r.history.map(v=>v.residual_rms);
         assert(line&&JSON.stringify(line.y)===JSON.stringify(expected),'Exact saved trace '+page+' '+c.id+' '+r.method+' '+seed+' '+metric);
        });
       });
       if(metric==='pressure'){
        const field=document.getElementById(page+'-skip');field.value='0';field.dispatchEvent(new Event('input'));await api.ready();
        const first=cases[0].runs.find(r=>!seed||r.initialization===seed),group=D.names[first.method];
        const marker=graph.data.find(t=>t.mode==='markers'&&t.legendgroup===group&&t.xaxis==='x');
        assert(marker.x.length===cases[0].x_m.length-2,'Zero skipped points marks every cell');
        field.value='-1';field.dispatchEvent(new Event('input'));await api.ready();assert(field.getAttribute('aria-invalid')==='true','Invalid skip rejected');
        field.value='5';field.dispatchEvent(new Event('input'));await api.ready();
       }
      }else{
       D.cases.forEach((c,ci)=>{
        const xa=ci?'x'+(ci+1):'x';
        for(const [key,name] of [['default','Implicit, original cap 8'],['one_inner','Implicit, one correction']]){
         const h=D.performance.replay[c.id][key].accepted_history;
         const expected=h.map(v=>metric==='inner'?v.inner_solves:metric==='cfl'?v.effective_cfl:v.residual_rms);
         const trace=graph.data.find(t=>t.name===name&&t.xaxis===xa);assert(JSON.stringify(trace.y)===JSON.stringify(expected),'Exact replay arrays');
        }
       });
      }
      records.push({page,theme,metric,seed,plot_traces:graph.data.length});
     }
    }
    if(page==='theory'||page==='notes'||page==='performance'){
     assert(document.querySelectorAll('#'+page+' mjx-container').length>5,'Notes typeset '+page);
     assert(document.querySelectorAll('[data-mml-node="merror"]').length===0,'No equation errors');
     records.push({page,theme,math:document.querySelectorAll('#'+page+' mjx-container').length});
    }
   }
  }
  const downloads=[...document.querySelectorAll('a[download]')];assert(downloads.length===2&&downloads.every(a=>a.href.startsWith('data:')),'Embedded downloads');
  assert(window.__reportErrors.length===0,'No JavaScript errors: '+JSON.stringify(window.__reportErrors));
 }catch(e){failures.push(String(e.stack||e));}
 finish();
})();
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=STUDY/'REPORT.html')
    parser.add_argument('--width',type=int,default=1600)
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
    command=[str(args.edge),'--headless=new','--disable-gpu',f'--window-size={args.width},1000','--no-first-run','--disable-background-networking',
             '--disable-component-update','--disable-extensions','--disable-sync','--host-resolver-rules=MAP * ~NOTFOUND',
             '--user-data-dir='+str((output/'browser_profile').resolve()),'--dump-dom','--virtual-time-budget=60000',isolated.resolve().as_uri()]
    with (output/'dom.html').open('w',encoding='utf-8') as stdout,(output/'browser_stderr.log').open('w',encoding='utf-8') as stderr:
        result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    dom=(output/'dom.html').read_text(encoding='utf-8')
    match=re.search(r'<script id="newton-browser-audit" type="application/json">(.*?)</script>',dom,re.S)
    if not match:raise RuntimeError(f'Browser audit did not complete (exit {result.returncode}); inspect {output}')
    audit=json.loads(match[1]);audit.update(report_sha256=hashlib.sha256(original).hexdigest(),standalone_copy=True,
                                         network_disabled=True,driver='Installed headless Edge CLI; isolated profile')
    # Capture unmodified presentation states after the scripted interaction audit.
    display=output/'display_report.html';display.write_bytes(original)
    base=command[:-3]
    screenshots=[]
    for page,theme in (('cold','light'),('theory','dark')):
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
