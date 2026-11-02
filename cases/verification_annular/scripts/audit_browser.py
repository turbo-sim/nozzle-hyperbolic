"""Render and exercise the local Plotly report in an isolated headless Edge.

Uses the existing real-gas report's CDP test client. The report is copied into
the audit folder to check standalone operation; no CFD or EOS calls occur.
"""

import asyncio
from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

import websockets

from development.real_gas.scripts.audit_report import Browser,EDGE

STUDY=Path(__file__).resolve().parents[1]
OUT=STUDY/"outputs/browser_audit"


async def settled(browser):
    return await browser.evaluate("""(async()=>{
      const end=Date.now()+30000;
      while(Date.now()<end){
        if(window.annularReportLoadError)throw Error(window.annularReportLoadError);
        if(window.annularReport){const s=annularReport.snapshot();if(s.renderNumber===s.rendered)return s;}
        await new Promise(r=>setTimeout(r,30));
      }throw Error('Report did not finish rendering');
    })()""")


async def exercise(url,report):
    checks=[]
    async with websockets.connect(url,max_size=40_000_000) as socket:
        b=Browser(socket)
        for domain in ("Page","Runtime","Network"):
            await b.command(f"{domain}.enable")
        await b.command("Emulation.setDeviceMetricsOverride",dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
        await b.command("Browser.setDownloadBehavior",dict(behavior="allow",downloadPath=str(OUT)))
        await b.command("Page.navigate",dict(url=report.resolve().as_uri()))
        await settled(b)
        checks.append(dict(name="embedded_plotly",passed=await b.evaluate("Plotly.version==='2.35.2'")))
        await b.screenshot(OUT/"desktop_controls.png")
        await b.evaluate("document.getElementById('solution').scrollIntoView();")
        await b.screenshot(OUT/"desktop_profiles.png")
        cases=await b.evaluate("annularReport.data.variants.map(v=>v.id)")
        # Cross every case and grid. Check actual Plotly traces, not just builders.
        expression="""(()=>{
          const a=annularReport,r=a.run(),v=a.variant(),s=a.snapshot(),bad=[];
          for(const key of s.fields){
            const chart=document.getElementById('field-'+key),err=document.getElementById('error-'+key);
            const info=a.INFO[key],traces=chart.data,ref=traces[0],fvm=traces[1],indices=a.markerIndices(r.cells,s.skip);
            if(traces.length!==2||ref.mode!=='lines'||fvm.mode!=='markers'||fvm.marker.color!=='#ffffff')bad.push(key+':traces');
            if(fvm.y.length!==indices.length||fvm.x.at(-1)!==r.m.at(-1)*1000)bad.push(key+':marker-count');
            if(!fvm.y.every((x,j)=>x===r.fields[key][indices[j]]*info.scale))bad.push(key+':fvm-values');
            if(!ref.y.every((x,j)=>x===v.reference.fields[key][j]*info.scale))bad.push(key+':reference-values');
            if(err._fullLayout.yaxis.type!=='log'||err.data[0].y.length!==r.cells)bad.push(key+':log-or-count');
            if(!err.data[0].y.every((x,j)=>{const d=Math.abs(r.fields[key][j]-r.reference_fields[key][j]);return x===(d===0?null:d*info.scale)}))bad.push(key+':errors');
          }
          return {passed:!bad.length,bad,cells:r.cells,case:v.id,fields:s.fields.length};
        })()"""
        for case in cases:
            await b.evaluate(f"document.getElementById('case').value={json.dumps(case)};document.getElementById('case').dispatchEvent(new Event('change'))")
            await settled(b)
            for cells in (50,100,200,400):
                await b.evaluate(f"document.getElementById('grid').value='{cells}';document.getElementById('grid').dispatchEvent(new Event('change'))")
                await settled(b)
                result=await b.evaluate(expression)
                checks.append(dict(name=f"paired_profiles_errors_{case}_{cells}",**result))
        await b.evaluate("document.getElementById('case').value='aungier';document.getElementById('case').dispatchEvent(new Event('change'))")
        await settled(b)
        baseline=await b.evaluate("document.getElementById('error-p').data[0].y")
        for mode in ("fvm","space","both"):
            await b.evaluate(f"document.getElementById('display').value='{mode}';document.getElementById('display').dispatchEvent(new Event('change'))")
            await settled(b)
            actual=await b.evaluate("document.getElementById('field-p').data.map(t=>t.mode)")
            expected={"fvm":["markers"],"space":["lines"],"both":["lines","markers"]}[mode]
            unchanged=await b.evaluate("document.getElementById('error-p').data[0].y")==baseline
            checks.append(dict(name=f"solver_visibility_{mode}",passed=actual==expected and unchanged))
        for skip in (0,1,7,49,999):
            await b.evaluate(f"document.getElementById('skip').value='{skip}';document.getElementById('skip').dispatchEvent(new Event('change'))")
            await settled(b)
            count=await b.evaluate("document.getElementById('field-p').data.at(-1).y.length")
            n=400;step=min(skip,n-1)+1
            expected=len(set(range(0,n,step))|{n-1})
            unchanged=await b.evaluate("document.getElementById('error-p').data[0].y")==baseline
            checks.append(dict(name=f"marker_skip_{skip}",passed=count==expected and unchanged,markers=count))
        await b.evaluate("document.getElementById('skip').value='7';document.getElementById('skip').dispatchEvent(new Event('change'));document.getElementById('field-group').value='all';document.getElementById('field-group').dispatchEvent(new Event('change'))")
        await settled(b)
        result=await b.evaluate(expression)
        checks.append(dict(name="all_twelve_flow_fields",**result))
        zero=await b.evaluate("(()=>{const r=structuredClone(annularReport.run());r.fields.p=[5,7];r.reference_fields.p=[5,6];r.m=[0,1];const t=annularReport.errorTrace(r,'p');return t.y[0]===null&&t.customdata[0]===0&&t.y[1]===.001;})()")
        checks.append(dict(name="exact_zero_not_floored",passed=zero))
        await b.evaluate("(async()=>{await Plotly.relayout(document.getElementById('field-p'),{'xaxis.range':[20,80]});await new Promise(r=>setTimeout(r,60));})()")
        linked=await b.evaluate("document.getElementById('error-p')._fullLayout.xaxis.range")
        checks.append(dict(name="linked_profile_error_zoom",passed=linked==[20,80]))
        await b.evaluate("document.getElementById('theme').click()")
        await settled(b)
        white=await b.evaluate("document.documentElement.dataset.theme==='dark'&&document.getElementById('field-p').data.at(-1).marker.color==='#ffffff'")
        checks.append(dict(name="dark_theme_white_marker_faces",passed=white))
        await b.screenshot(OUT/"profiles_dark.png")
        for page in ("geometry","convergence","model","evidence","solution"):
            await b.evaluate(f"annularReport.switchPage('{page}')")
            await settled(b)
            active=await b.evaluate("[...document.querySelectorAll('.page.active')].map(e=>e.id)")
            checks.append(dict(name=f"page_{page}",passed=active==[page]))
            if page=="model":
                math=await b.evaluate("(()=>{const m=document.querySelector('math');return m.querySelectorAll('mtr').length===4&&m.getBoundingClientRect().height>70;})()")
                checks.append(dict(name="native_mathml_rendered",passed=math))
            if page in ("geometry","model"):
                await b.evaluate(f"document.getElementById('{page}').scrollIntoView()")
                await b.screenshot(OUT/f"{page}.png")
        await b.evaluate("(()=>{window.auditBlobs=[];const original=URL.createObjectURL;URL.createObjectURL=function(blob){auditBlobs.push(blob);return original.call(this,blob)};document.getElementById('download-fields').click();document.getElementById('download-history').click();document.getElementById('download-data').click();})()")
        downloads=await b.evaluate("(async()=>await Promise.all(auditBlobs.map(async b=>({bytes:b.size,type:b.type,lines:(await b.text()).split('\\n').length}))))()")
        checks.append(dict(name="full_resolution_downloads",passed=len(downloads)==3 and downloads[0]["lines"]==401 and all(d["bytes"]>100 for d in downloads),detail=downloads))
        await b.evaluate("document.getElementById('theme').click();document.getElementById('field-group').value='primary';document.getElementById('field-group').dispatchEvent(new Event('change'))")
        await settled(b)
        await b.command("Emulation.setDeviceMetricsOverride",dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
        await b.evaluate("window.scrollTo(0,0);window.dispatchEvent(new Event('resize'));annularReport.render()")
        await settled(b)
        await asyncio.sleep(.2)
        overflow=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")
        checks.append(dict(name="mobile_no_horizontal_overflow",passed=overflow))
        await b.screenshot(OUT/"mobile_controls.png")
        await b.evaluate("document.getElementById('solution').scrollIntoView()")
        await b.screenshot(OUT/"mobile_profiles.png")
        external=sorted({url for url in b.requests if url.startswith(("http:","https:"))})
        checks.append(dict(name="offline_no_external_requests",passed=not external,detail=external))
        checks.append(dict(name="no_javascript_exceptions",passed=not b.exceptions,detail=b.exceptions))
        return dict(passed=all(c["passed"] for c in checks),checks=checks,
                    html_sha256=sha256((STUDY/"REPORT.html").read_bytes()).hexdigest(),
                    browser="Isolated headless Edge via repository CDP audit client",
                    screenshots=[p.name for p in sorted(OUT.glob("*.png"))])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    standalone=OUT/"standalone.html"
    shutil.copyfile(STUDY/"REPORT.html",standalone)
    with socket.socket() as s:
        s.bind(("127.0.0.1",0));port=s.getsockname()[1]
    command=[str(EDGE),"--headless=new","--disable-gpu","--no-first-run",
             "--disable-crash-reporter","--disable-breakpad","--disable-background-networking",
             "--disable-component-update","--disable-sync","--disable-extensions","--no-default-browser-check",
             f"--remote-debugging-port={port}",f"--user-data-dir={OUT/'browser_profile'}","about:blank"]
    with (OUT/"edge.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            targets=[];deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list",timeout=1) as response:targets=json.load(response)
                    if any(t["type"]=="page" for t in targets):break
                except (OSError,TimeoutError):pass
                time.sleep(.15)
            target=next((t for t in targets if t["type"]=="page"),None)
            if target is None:raise RuntimeError(f"Headless Edge unavailable; inspect {OUT/'edge.log'}")
            result=asyncio.run(exercise(target["webSocketDebuggerUrl"],standalone))
            (OUT/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
            print(json.dumps(dict(passed=result["passed"],checks=len(result["checks"]),
                failed=[c["name"] for c in result["checks"] if not c["passed"]]),indent=2))
            assert result["passed"]
        finally:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill()


if __name__=="__main__":main()
