"""Exercise the offline report in an isolated headless Edge browser via CDP.

Only a local HTML file is opened. Browser profiles, screenshots and audit
evidence are kept under this study's outputs directory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import websockets

from .report import STUDY


EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


class Browser:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.exceptions = []
        self.failed_requests = []
        self.requests = []

    async def command(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        await self.socket.send(json.dumps(dict(id=request_id, method=method, params=params or {})))
        while True:
            message = json.loads(await self.socket.recv())
            event = message.get("method")
            if event == "Runtime.exceptionThrown":
                self.exceptions.append(message["params"])
            elif event == "Network.requestWillBeSent":
                self.requests.append(message["params"]["request"]["url"])
            elif event == "Network.loadingFailed":
                self.failed_requests.append(message["params"])
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    async def evaluate(self, expression):
        result = await self.command("Runtime.evaluate", dict(
            expression=expression, returnByValue=True, awaitPromise=True,
        ))
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    async def settled(self):
        return await self.evaluate("""(async()=>{
          const end=Date.now()+25000;
          while(Date.now()<end){
            if(window.nozzleReportLoadError)throw new Error(window.nozzleReportLoadError);
            if(window.nozzleReport){const s=window.nozzleReport.snapshot();
              if(s.rendered===s.renderNumber)return s;}
            await new Promise(r=>setTimeout(r,30));
          }
          throw new Error('Report render did not settle');
        })()""")

    async def screenshot(self, path):
        result = await self.command("Page.captureScreenshot", dict(format="png", captureBeyondViewport=False))
        path.write_bytes(base64.b64decode(result["data"]))


async def exercise(websocket_url, html, output):
    checks = []
    async with websockets.connect(websocket_url, max_size=40_000_000) as connection:
        browser = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await browser.command(f"{domain}.enable")
        await browser.command("Emulation.setDeviceMetricsOverride", dict(
            width=1440, height=1050, deviceScaleFactor=1, mobile=False,
        ))
        await browser.command("Browser.setDownloadBehavior", dict(behavior="allow", downloadPath=str(output)))
        await browser.command("Page.navigate", dict(url=html.resolve().as_uri()))
        initial = await browser.settled()
        checks.append(dict(name="initial_render", passed=bool(initial), detail=initial))
        checks.append(dict(name="plotly_embedded_loaded", passed=await browser.evaluate("typeof Plotly==='object'&&Plotly.version==='2.35.2'")))
        flux_counts = await browser.evaluate("""(()=>{
          const rows=nozzleReport.campaign.results;
          return [...document.querySelectorAll('#campaign-flux-counts [data-flux]')].map(el=>{
            const selected=rows.filter(r=>r.flux===el.dataset.flux);
            const verified=selected.filter(r=>r.success&&r.audit?.passed===true).length;
            return {flux:el.dataset.flux,verified,total:selected.length,
              passed:verified===Number(el.dataset.verified)&&selected.length===Number(el.dataset.total)};
          });
        })()""")
        expected_fluxes = await browser.evaluate('nozzleReport.campaign.settings.fluxes.length')
        checks.append(dict(name="overall_per_flux_verification_counts", passed=len(flux_counts) == expected_fluxes and all(c["passed"] for c in flux_counts), detail=flux_counts))
        source_status = await browser.evaluate("""(()=>{
          const c=nozzleReport.campaign,m=c.metadata||{},assigned=m.source_by_flux||{};
          return {source_by_flux:assigned,
            passed:c.results.every(r=>r.source_sha256===assigned[r.flux]&&Boolean(m.sources?.[r.source_sha256]))};
        })()""")
        checks.append(dict(name="source_fingerprints_match_results", passed=source_status["passed"], detail=source_status))
        final_results = await browser.evaluate("""(()=>{
          const c=nozzleReport.campaign;
          const expected=c.cases.length*c.settings.fluxes.length*c.settings.reconstructions.length*c.settings.grids.length;
          return c.results.length===expected&&c.results.every(r=>r.reason!=='not_completed'&&r.verified===(r.success&&r.audit?.passed===true));
        })()""")
        checks.append(dict(name="complete_configuration_matrix_and_honest_statuses", passed=bool(final_results)))
        data_status = await browser.evaluate("""(()=>{
          const d=document.getElementById('data'),rows=nozzleReport.campaign.results;
          const final=rows.reduce((s,r)=>s+(r.history?.length||0),0);
          return {encoding:d.dataset.encoding,history_rows:final,
            passed:final===Number(d.dataset.historyRows)};
        })()""")
        checks.append(dict(name="complete_embedded_histories_after_decode", passed=data_status["passed"], detail=data_status))
        # Deterministic light-theme screenshot and full viewport overflow check.
        await browser.evaluate("document.documentElement.dataset.theme='light';nozzleReport.render()")
        await browser.settled()
        await browser.screenshot(output / "solution_light.png")
        await browser.evaluate("window.scrollTo(0,document.getElementById('solution').offsetTop)")
        await browser.screenshot(output / "solution_fields_light.png")
        await browser.evaluate("window.scrollTo(0,0)")
        checks.append(dict(name="desktop_horizontal_overflow", passed=await browser.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        combinations = await browser.evaluate("nozzleReport.campaign.cases.map(c=>({physical:c.fluid_case,regime:c.regime}))")
        for item in combinations:
            for mesh in (50, 100, 200, 400, 800):
                await browser.evaluate(f"""(()=>{{
                  document.getElementById('physical-case').value={json.dumps(item['physical'])};
                  document.getElementById('regime').value={json.dumps(item['regime'])};
                  document.getElementById('mesh').value='{mesh}';
                  nozzleReport.render();
                }})()""")
                snapshot = await browser.settled()
                selected = await browser.evaluate("nozzleReport.selectedRows().every(r=>r.case_id===nozzleReport.selectedCase().id&&r.cells===Number(document.getElementById('mesh').value))")
                checks.append(dict(name=f"selectors_{item['physical']}_{item['regime']}_{mesh}", passed=bool(selected), detail=snapshot))
        for operating in ("adapted", "shock"):
            await browser.evaluate(f"document.getElementById('physical-case').value='lettieri_equilibrium';document.getElementById('regime').value='{operating}';nozzleReport.switchPage('solution')")
            await browser.settled()
            await browser.evaluate("window.scrollTo(0,document.getElementById('solution').offsetTop)")
            await browser.screenshot(output / f"two_phase_{operating}_fields_light.png")
            await browser.evaluate("window.scrollTo(0,0)")
        for reconstruction in ("constant", "muscl_van_leer", "both"):
            await browser.evaluate(f"document.getElementById('reconstruction').value={json.dumps(reconstruction)};nozzleReport.render()")
            await browser.settled()
            passed = await browser.evaluate("document.getElementById('reconstruction').value==='both'||nozzleReport.selectedRows().every(r=>r.reconstruction===document.getElementById('reconstruction').value)")
            checks.append(dict(name=f"reconstruction_{reconstruction}", passed=bool(passed)))
        methods = await browser.evaluate('nozzleReport.campaign.settings.fluxes')
        for method in methods:
            await browser.evaluate(f"document.querySelectorAll('#methods input').forEach(e=>e.checked=e.value==={json.dumps(method)});nozzleReport.render()")
            await browser.settled()
            passed = await browser.evaluate(f"nozzleReport.selectedRows().length>0&&nozzleReport.selectedRows().every(r=>r.flux==={json.dumps(method)})")
            checks.append(dict(name=f'single_flux_filter_{method}', passed=bool(passed)))
        await browser.evaluate("document.querySelectorAll('#methods input').forEach(e=>e.checked=false);nozzleReport.render()")
        await browser.settled()
        checks.append(dict(name="empty_flux_selection", passed=await browser.evaluate("nozzleReport.selectedRows().length===0&&document.getElementById('selection-notice').style.display==='block'")))
        await browser.evaluate("document.querySelectorAll('#methods input').forEach(e=>e.checked=true);nozzleReport.render()")
        await browser.settled()
        for page in ("convergence", "refinement", "methodology", "solution"):
            await browser.evaluate(f"nozzleReport.switchPage({json.dumps(page)})")
            await browser.settled()
            checks.append(dict(name=f"page_{page}", passed=await browser.evaluate(f"document.getElementById({json.dumps(page)}).classList.contains('active')")))
            if page in ("convergence", "refinement"):
                await browser.evaluate(f"window.scrollTo(0,document.getElementById('{page}').offsetTop)")
                await browser.screenshot(output / f"{page}_light.png")
                await browser.evaluate("window.scrollTo(0,0)")
        for mode in ("absolute", "relative", "ratio"):
            await browser.evaluate(f"document.getElementById('residual-mode').value='{mode}';nozzleReport.switchPage('convergence')")
            await browser.settled()
            checks.append(dict(name=f"residual_view_{mode}", passed=True))
        await browser.evaluate("""(()=>{
          const c=nozzleReport.campaign;
          const row=c.results.find(r=>r.attempts?.length);
          if(row){const physical=c.cases.find(v=>v.id===row.case_id);
            document.getElementById('physical-case').value=physical.fluid_case;
            document.getElementById('regime').value=physical.regime;
            document.getElementById('mesh').value=String(row.cells);}
        })()""")
        for scope in ('final', 'all'):
            await browser.evaluate(f"document.getElementById('attempt-mode').value='{scope}';nozzleReport.switchPage('convergence')")
            await browser.settled()
            correct = await browser.evaluate("""(()=>{
              const selected=nozzleReport.selectedRows();
              const expected=selected.length+(document.getElementById('attempt-mode').value==='all'?selected.reduce((n,r)=>n+(r.attempts?.length||0),0):0);
              return nozzleReport.selectedHistoryRows().length===expected;
            })()""")
            checks.append(dict(name=f'history_attempt_scope_{scope}', passed=bool(correct)))
        await browser.evaluate("document.getElementById('attempt-mode').value='final'")
        error_fields = await browser.evaluate("[...document.getElementById('error-field').options].map(o=>o.value)")
        for field in error_fields:
            await browser.evaluate(f"document.getElementById('error-field').value={json.dumps(field)};nozzleReport.switchPage('refinement')")
            await browser.settled()
            checks.append(dict(name=f"grid_error_field_{field}", passed=True))
        scalar_check = await browser.evaluate("""(()=>{
          const flux=nozzleReport.campaign.results.find(r=>Number.isFinite(r.audit?.conservation?.numerical_face_mass_flow_relative_reference_error));
          const shock=nozzleReport.campaign.results.find(r=>Number.isFinite(r.audit?.shock?.position_error_over_divergent_length));
          return {
            mass_flow:flux?nozzleReport.errorValue(flux,'mass_flow')===flux.audit.conservation.numerical_face_mass_flow_relative_reference_error:null,
            shock_position:shock?nozzleReport.errorValue(shock,'shock_position')===Math.abs(shock.audit.shock.position_error_over_divergent_length):null,
            scalar_columns:flux?nozzleReport.errorValue(flux,'mass_flow','l2')===null&&nozzleReport.errorValue(flux,'mass_flow','linf')===null:null
          };
        })()""")
        for name, passed in scalar_check.items():
            checks.append(dict(name=f"scalar_error_definition_{name}", passed=passed is not False,
                               skipped_no_numerical_data=passed is None))
        await browser.evaluate("nozzleReport.switchPage('solution');document.getElementById('theme').click()")
        await browser.settled()
        checks.append(dict(name="dark_theme", passed=await browser.evaluate("document.documentElement.dataset.theme==='dark'")))
        await browser.screenshot(output / "solution_dark.png")
        # Capture generated download bytes in-browser, then allow the actual UI handlers.
        await browser.evaluate("""(()=>{
          window.auditBlobs=[];
          const original=URL.createObjectURL;
          URL.createObjectURL=function(blob){window.auditBlobs.push(blob);return original.call(this,blob)};
          document.getElementById('download-fields').click();
          document.getElementById('download-history').click();
          document.getElementById('download-data').click();
        })()""")
        downloads = await browser.evaluate("(async()=>await Promise.all(auditBlobs.map(async b=>({bytes:b.size,type:b.type,first:(await b.text()).slice(0,120)}))))()")
        checks.append(dict(name="embedded_downloads", passed=len(downloads) == 3 and all(d["bytes"] > 50 for d in downloads), detail=downloads))
        await browser.command("Emulation.setDeviceMetricsOverride", dict(
            width=390, height=844, deviceScaleFactor=1, mobile=True,
        ))
        await browser.evaluate("window.dispatchEvent(new Event('resize'));nozzleReport.render()")
        await browser.settled()
        await asyncio.sleep(0.3)
        checks.append(dict(name="mobile_horizontal_overflow", passed=await browser.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        await browser.screenshot(output / "solution_mobile.png")
        external = sorted(set(url for url in browser.requests if url.startswith(("http:", "https:"))))
        checks.append(dict(name="offline_no_external_requests", passed=not external, detail=external))
        checks.append(dict(name="no_javascript_exceptions", passed=not browser.exceptions, detail=browser.exceptions))
        result = dict(passed=all(c["passed"] for c in checks), checks=checks,
                      javascript_exceptions=browser.exceptions, failed_requests=browser.failed_requests,
                      input=str(html), html_sha256=sha256(html.read_bytes()).hexdigest(),
                      screenshots=[p.name for p in sorted(output.glob("*.png"))])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/browser_audit")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-sandbox", "--disable-crash-reporter", "--disable-breakpad",
               "--disable-background-networking", "--disable-component-update", "--disable-sync",
               "--disable-extensions", "--no-default-browser-check", "--remote-allow-origins=*",
               f"--remote-debugging-port={port}", f"--user-data-dir={output / 'browser_profile'}", "about:blank"]
    with (output / "edge.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            targets = []
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                        targets = json.load(response)
                    if any(target["type"] == "page" for target in targets):
                        break
                except (OSError, TimeoutError):
                    pass
                time.sleep(0.15)
            target = next((target for target in targets if target["type"] == "page"), None)
            if target is None:
                raise RuntimeError(f"Headless Edge did not expose a page; inspect {output / 'edge.log'}")
            result = asyncio.run(exercise(target["webSocketDebuggerUrl"], args.input, output))
            (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                              "failed": [c["name"] for c in result["checks"] if not c["passed"]],
                              "output": str(output)}, indent=2))
            if not result["passed"]:
                raise RuntimeError("Browser report audit failed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
