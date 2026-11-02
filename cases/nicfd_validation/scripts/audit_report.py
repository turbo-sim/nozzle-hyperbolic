"""Static and isolated-browser checks of the standalone Plotly report."""

import argparse
import asyncio
import base64
import csv
import inspect
import io
import json
from pathlib import Path
import socket
import subprocess
import uuid
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import websockets

from .report import ROOT, audit_html
# Reuse the real_gas report's small generic CDP transport, not its study logic.
from development.real_gas.scripts.audit_report import Browser

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


async def settled(browser):
    return await browser.evaluate("""(async()=>{
      const end=Date.now()+25000;
      while(Date.now()<end){
        if(window.nicfdReportLoadError)throw Error(window.nicfdReportLoadError);
        if(window.nicfdReport){let s=nicfdReport.snapshot();if(s.rendered===s.renderNumber)return s;}
        await new Promise(r=>setTimeout(r,30));
      }throw Error('Report did not settle');
    })()""")


async def change(browser, control, value):
    await browser.evaluate(f"document.getElementById({json.dumps(control)}).value={json.dumps(str(value))};document.getElementById({json.dumps(control)}).dispatchEvent(new Event('change'))")
    return await settled(browser)


async def downloaded(path):
    previous_size, stable = -1, 0
    for _ in range(100):
        if path.exists() and not path.with_suffix(path.suffix+'.crdownload').exists():
            size = path.stat().st_size
            stable = stable + 1 if size == previous_size else 0
            previous_size = size
            if stable >= 3:
                return path.read_text(encoding="utf-8")
        await asyncio.sleep(.05)
    raise RuntimeError(f"Download missing: {path}")


async def exercise(url, report, out):
    options = {"max_size": 60_000_000}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        options["proxy"] = None
    checks = []
    async with websockets.connect(url, **options) as connection:
        browser = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await browser.command(f"{domain}.enable")
        await browser.command("Emulation.setDeviceMetricsOverride", dict(width=1500,height=1050,deviceScaleFactor=1,mobile=False))
        await browser.command("Browser.setDownloadBehavior", dict(behavior="allow",downloadPath=str(out)))
        await browser.command("Page.navigate",dict(url=report.resolve().as_uri()))
        await settled(browser)
        checks.append(dict(name="embedded_plotly",passed=await browser.evaluate("Plotly.version==='2.35.2'")))
        for source in ("published", "new", "both"):
            await change(browser,"data-source",source)
            detail = await browser.evaluate("""(()=>{
             const traces=document.getElementById('field-0').data;
             return {sources:[...new Set(traces.map(t=>t.meta.dataset))],count:traces.length,
              expectedNew:nicfdReport.selectedRows().filter(r=>r.fields.x?.length).length,
              expectedOld:nicfdReport.selectedCases().length};
            })()""")
            expected = {"published"} if source=="published" else {"new"} if source=="new" else {"published","new"}
            actual = set(detail["sources"])
            expected_count = (detail["expectedOld"] if source!="new" else 0)+(detail["expectedNew"] if source!="published" else 0)
            checks.append(dict(name="visibility_"+source,passed=actual==expected and detail["count"]==expected_count,detail=detail))
        for regime in ("subsonic", "supersonic_shock", "supersonic_adapted"):
            await change(browser,"regime",regime)
            for mesh in (50,100,200,400):
                await change(browser,"mesh",mesh)
                passed = await browser.evaluate("""nicfdReport.selectedRows().length===1&&nicfdReport.selectedRows().every(r=>r.regime===document.getElementById('regime').value&&r.cells===Number(document.getElementById('mesh').value))""")
                checks.append(dict(name=f"selection_{regime}_{mesh}",passed=passed))
        await change(browser,"mesh","all")
        checks.append(dict(name="all_mesh_overlay",passed=await browser.evaluate("nicfdReport.selectedRows().length===4&&document.getElementById('field-0').data.filter(t=>t.meta.dataset==='new').length===nicfdReport.selectedRows().filter(r=>r.fields.x?.length).length")))
        await change(browser,"mesh",400)
        await change(browser,"regime","all")
        await change(browser,"field-group","thermal")
        checks.append(dict(name="thermodynamic_fields",passed=await browser.evaluate("document.getElementById('field-0').data.every(t=>t.meta.field==='T_g')&&document.getElementById('field-5').data.every(t=>t.meta.field==='rho_l')")))
        await change(browser,"field-group","flow")
        await browser.screenshot(out/"solution_light.png")
        await browser.evaluate("window.scrollTo(0,document.getElementById('solution').offsetTop)")
        await browser.screenshot(out/"profiles_light.png")
        await browser.evaluate("nicfdReport.switchPage('differences')")
        await settled(browser)
        differences = await browser.evaluate("""(()=>{
          return nicfdReport.selectedRows().filter(r=>r.fields.x?.length).map(r=>{
            const delta=nicfdReport.difference(r,'p');const rms=Math.sqrt(delta.reduce((s,v)=>s+v*v,0)/delta.length);
            return {id:r.id,computed:rms,saved:r.errors.p.rmse/1e5,passed:Math.abs(rms-r.errors.p.rmse/1e5)<1e-10};
          });
        })()""")
        checks.append(dict(name="difference_values_match_saved_errors",passed=all(x["passed"] for x in differences),detail=differences))
        await change(browser,"error-field","u_l")
        await browser.screenshot(out/"differences.png")
        await browser.evaluate("nicfdReport.switchPage('convergence')")
        await settled(browser)
        await change(browser,"residual-mode","relative")
        checks.append(dict(name="convergence_trajectories",passed=await browser.evaluate("document.getElementById('history-residual').data.length===6*nicfdReport.selectedRows().length")))
        await browser.screenshot(out/"convergence.png")
        await browser.evaluate("nicfdReport.switchPage('methodology')")
        await settled(browser)
        checks.append(dict(name="settings_unchanged",passed=await browser.evaluate("nicfdReport.data.provenance.numerical_settings_unchanged")))
        await browser.evaluate("document.getElementById('download-data').click()")
        payload = json.loads(await downloaded(out/"nicfd_complete_report_data.json"))
        checks.append(dict(name="json_download",passed=len(payload["results"])==12 and payload["settings"]["meshes"]==[50,100,200,400]))
        await browser.evaluate("nicfdReport.switchPage('solution')")
        await settled(browser)
        await change(browser,"data-source","published")
        await browser.evaluate("document.getElementById('download-fields').click()")
        fields = list(csv.DictReader(io.StringIO(await downloaded(out/"nicfd_selected_fields_SI.csv"))))
        checks.append(dict(name="published_only_csv",passed=len(fields)==3*498 and {r["dataset"] for r in fields}=={"published"}))
        (out/"nicfd_selected_fields_SI.csv").rename(out/"published_fields.csv")
        await change(browser,"data-source","new")
        await browser.evaluate("document.getElementById('download-fields').click()")
        newfile=out/"nicfd_selected_fields_SI.csv"
        fields = list(csv.DictReader(io.StringIO(await downloaded(newfile))))
        expected = sum(len(r["fields"].get("x",[])) for r in payload["results"] if r["cells"]==400)
        checks.append(dict(name="new_only_csv",passed=len(fields)==expected and {r["dataset"] for r in fields}=={"new"}))
        await change(browser,"data-source","both")
        await browser.evaluate("document.getElementById('theme').click()")
        await settled(browser)
        checks.append(dict(name="dark_theme",passed=await browser.evaluate("document.documentElement.dataset.theme==='dark'&&document.getElementById('field-0')._fullLayout.paper_bgcolor==='#171f2b'")))
        await browser.screenshot(out/"solution_dark.png")
        await browser.evaluate("document.getElementById('theme').click()")
        await settled(browser)
        await browser.evaluate("(async()=>{await Plotly.relayout('field-0',{'xaxis.range':[100,180]});return true;})()")
        checks.append(dict(name="plotly_zoom",passed=await browser.evaluate("document.getElementById('field-0')._fullLayout.xaxis.range[0]===100")))
        image = await browser.evaluate("Plotly.toImage('field-0',{format:'svg',width:1000,height:650})")
        svg=urllib.parse.unquote(image.split(',',1)[1]);ET.fromstring(svg)
        (out/"pressure_export.svg").write_text(svg,encoding="utf-8")
        checks.append(dict(name="svg_export",passed=True))
        await browser.command("Emulation.setDeviceMetricsOverride",dict(width=430,height=900,deviceScaleFactor=1,mobile=False))
        await browser.evaluate("nicfdReport.render()")
        await settled(browser)
        await asyncio.sleep(.3)
        checks.append(dict(name="mobile_no_horizontal_overflow",passed=await browser.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        await browser.evaluate("window.scrollTo(0,0)")
        await browser.screenshot(out/"mobile.png")
        external=[url for url in browser.requests if not url.startswith(("file:","data:","blob:","about:"))]
        checks.append(dict(name="offline_no_external_requests",passed=not external,detail=external))
        checks.append(dict(name="no_javascript_exceptions",passed=not browser.exceptions,detail=browser.exceptions))
        await browser.command("Browser.close")
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser",action="store_true")
    args=parser.parse_args()
    static=audit_html(ROOT/"REPORT.html")
    print("Offline structure, controls and lossless data checks passed.",flush=True)
    if not args.browser:
        return
    out=ROOT/"outputs/browser_audit"/("plotly_"+uuid.uuid4().hex[:10])
    out.mkdir(parents=True)
    print("Browser evidence:",out,flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    process=subprocess.Popen([str(CHROME),"--headless=new","--disable-gpu","--disable-background-networking",
        "--no-first-run",f"--remote-debugging-port={port}",f"--user-data-dir={out/'profile'}","about:blank"],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            try:
                targets=json.loads(opener.open(f"http://127.0.0.1:{port}/json/list",timeout=1).read())
                target=next(x for x in targets if x["type"]=="page");break
            except Exception:
                time.sleep(.1)
        else:
            raise RuntimeError("Local headless Chrome did not start")
        checks=asyncio.run(exercise(target["webSocketDebuggerUrl"],ROOT/"REPORT.html",out))
        result=dict(passed=all(c["passed"] for c in checks),checks=checks,static=static)
        (out/"checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        (ROOT/"outputs/plotly_report/browser_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        assert result["passed"],[c for c in checks if not c["passed"]]
        print(f"Browser audit passed: {len(checks)} checks. Evidence: {out}",flush=True)
    finally:
        if process.poll() is None:
            process.terminate()


if __name__=="__main__":
    main()
