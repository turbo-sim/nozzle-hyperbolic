"""Exercise the standalone saved-data report in an isolated headless browser."""

import argparse
import base64
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
from urllib.request import urlopen
import json

from development.multigrid.scripts.report_assets.browser_visual import DebugSocket
from .common import STUDY, save, sha


def audit():
    path=STUDY/'REPORT.html';out=STUDY/'outputs/report/presentation'
    out.mkdir(parents=True,exist_ok=True)
    for previous in out.glob('two-fluid-v2-chrome-*'):
        resolved=previous.resolve()
        if resolved.parent!=out.resolve():raise RuntimeError('Unexpected old browser-profile path')
        shutil.rmtree(resolved,ignore_errors=True)
    profile=Path(tempfile.mkdtemp(prefix='two-fluid-v2-chrome-',dir=out))
    command=['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--disable-gpu',
        '--no-first-run','--no-default-browser-check','--disable-default-apps','--disable-extensions',
        '--disable-background-networking','--disable-component-update','--disable-sync',
        '--allow-file-access-from-files','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
        '--user-data-dir='+str(profile.resolve()),'about:blank']
    debugger=None
    with (out/'chrome.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            end=time.monotonic()+30
            while not (profile/'DevToolsActivePort').exists():
                if time.monotonic()>end or process.poll() is not None:raise RuntimeError('Chrome did not start')
                time.sleep(.1)
            port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
            with urlopen(f'http://127.0.0.1:{port}/json/list',timeout=10) as response:
                target=next(r for r in json.load(response) if r.get('type')=='page')
            debugger=DebugSocket(target['webSocketDebuggerUrl'])
            debugger.socket.settimeout(180)
            debugger.call('Page.enable')
            debugger.call('Page.addScriptToEvaluateOnNewDocument',{'source':"window.QA_ERRORS=[];addEventListener('error',e=>QA_ERRORS.push(e.message));addEventListener('unhandledrejection',e=>QA_ERRORS.push(String(e.reason)));"})
            debugger.call('Emulation.setDeviceMetricsOverride',dict(width=1500,height=1150,deviceScaleFactor=1,mobile=False))
            debugger.call('Page.navigate',{'url':path.resolve().as_uri()})
            end=time.monotonic()+60
            while not debugger.evaluate("document.body?.dataset.ready==='true'"):
                if time.monotonic()>end:raise RuntimeError(str(debugger.evaluate('document.body?.dataset.error')))
                time.sleep(.1)
            result=debugger.evaluate(r'''(async()=>{
              let checks=[];let assert=(name,passed)=>{checks.push({name,passed:!!passed});if(!passed)throw Error(name)};
              assert('all result rows',document.querySelectorAll('#results tbody tr').length===REPORT_DATA.records.length);
              for(let theme=0;theme<2;theme++){
                if(theme)document.getElementById('theme').click();
                for(let regime of [...document.getElementById('regime').options].map(o=>o.value))
                for(let rec of [...document.getElementById('reconstruction').options].map(o=>o.value))
                for(let cells of [...document.getElementById('cells').options].map(o=>o.value))
                for(let field of [...document.getElementById('field').options].map(o=>o.value)){
                  for(let [id,value]of [['regime',regime],['reconstruction',rec],['cells',cells],['field',field]])document.getElementById(id).value=value;
                  document.getElementById('field').dispatchEvent(new Event('change'));
                  await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
                  let rows=REPORT_DATA.records.filter(r=>r.regime===regime&&r.reconstruction===rec&&r.cells===Number(cells));
                  let tag=[theme,regime,rec,cells,field].join('/'),f=document.getElementById('fields'),c=document.getElementById('convergence'),t=document.getElementById('timing');
                  assert('actual fields '+tag,f.data.length>=rows.length);
                  assert('residuals '+tag,c.data.length===rows.length&&c.layout.yaxis.type==='log');
                  assert('timings '+tag,t.data[0].y.length===rows.length);
                  assert('actual values '+tag,rows.every((r,i)=>JSON.stringify(f.data[i].y)===JSON.stringify(r.fields[field])&&t.data[0].y[i]===r.warm_s));
                  for(let trace of c.data)assert('counts '+tag+'/'+trace.name,rows.some(r=>trace.name===REPORT_DATA.labels[r.variant]+' · '+r.cells+' cells · '+r.work.accepted_steps+' accepted / '+r.work.rejected_steps+' rejected'));
                  for(let id of ['fields','convergence','timing']){let node=document.getElementById(id),title=node.querySelector('.gtitle'),legend=node.querySelector('.legend');if(title&&legend){let a=title.getBoundingClientRect(),b=legend.getBoundingClientRect();assert('title separation '+tag+'/'+id,a.bottom<=b.top+2||a.top>=b.bottom-2)}}
                }
              }
              assert('actual MathJax SVG',document.querySelectorAll('mjx-container svg').length>0);
              let create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click,blob;
              URL.createObjectURL=b=>{blob=b;return 'blob:qa'};HTMLAnchorElement.prototype.click=function(){};
              document.getElementById('download').click();URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;
              assert('embedded download',JSON.stringify(JSON.parse(await blob.text()))===JSON.stringify(REPORT_DATA));
              for(let id of ['results','explorer','theory','protocol']){location.hash=id;assert('navigation '+id,!!document.getElementById(id))}
              assert('no script errors',QA_ERRORS.length===0);
              let network=performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name);assert('offline no HTTP',network.length===0);
              document.body.classList.remove('dark');document.getElementById('regime').value='supersonic_shock';document.getElementById('reconstruction').value='muscl_van_albada';let n=document.getElementById('cells');n.value=[...n.options].some(o=>o.value==='200')?'200':n.options[0].value;document.getElementById('field').value='p';document.getElementById('field').dispatchEvent(new Event('change'));window.scrollTo(0,0);
              return {passed:true,checks,records:REPORT_DATA.records.length,download_bytes:blob.size,network,errors:QA_ERRORS};
            })()''')
            for name,anchor in (('overview',None),('fields','explorer')):
                y=0 if anchor is None else debugger.evaluate("document.getElementById('explorer').getBoundingClientRect().top+scrollY")
                screenshot=debugger.call('Page.captureScreenshot',dict(format='png',fromSurface=True,captureBeyondViewport=True,
                    clip=dict(x=0,y=y,width=1500,height=1150,scale=1)))
                (out/(name+'.png')).write_bytes(base64.b64decode(screenshot['data']))
            result.update(report_sha256=sha(path),actual_browser=True)
            save(out/'browser_audit.json',result)
            print({k:v for k,v in result.items() if k!='checks'},'checks',len(result['checks']),flush=True)
        finally:
            if debugger is not None:
                try:debugger.call('Browser.close')
                except (OSError,EOFError):pass
                debugger.socket.close()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=10)
            resolved=profile.resolve()
            if resolved.parent!=out.resolve() or not resolved.name.startswith('two-fluid-v2-chrome-'):
                raise RuntimeError('Unexpected browser-profile cleanup target')
            shutil.rmtree(resolved,ignore_errors=True)


if __name__=='__main__':
    audit()
