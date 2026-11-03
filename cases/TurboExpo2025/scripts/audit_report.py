"""Audit saved-data integrity and exercise the standalone report offline.

The browser test copies only REPORT.html into a fresh directory, disables
network access, opens it in an isolated headless Edge profile, exercises every
case/grid selector and checks actual plotted values and the downloaded ZIP.
No CFD or thermodynamic preparation is performed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from hashlib import sha256
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile

import websockets

from development.real_gas.scripts.audit_report import Browser, EDGE

from .report import ROOT, read_json


OUT = ROOT / "outputs/browser_audit"


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_assets = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in ("script", "img", "iframe", "link", "source"):
            url = attrs.get("src", attrs.get("href", ""))
            if url and not url.startswith(("data:", "#")):
                self.external_assets.append(url)


def structural_audit():
    html = (ROOT / "REPORT.html").read_text(encoding="utf-8")
    manifest = read_json(ROOT / "report_manifest.json")
    parser = Document()
    parser.feed(html)
    checks = [
        dict(
            name="standalone_asset_dependencies",
            passed=not parser.external_assets,
            detail=parser.external_assets,
        )
    ]
    checks.append(
        dict(
            name="report_matches_manifest",
            passed=sha256((ROOT / "REPORT.html").read_bytes()).hexdigest()
            == manifest["sha256"],
        )
    )
    changed = [
        name
        for name, digest in manifest["inputs"].items()
        if not (ROOT / name).exists()
        or sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    checks.append(
        dict(
            name="saved_inputs_unchanged_since_report_generation",
            passed=not changed,
            detail=changed,
        )
    )
    current_results = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "outputs").rglob("result.json")
    }
    embedded_results = {
        name for name in manifest["inputs"] if name.endswith("/result.json")
    }
    checks.append(
        dict(
            name="all_current_saved_attempts_embedded",
            passed=current_results == embedded_results,
            detail=sorted(current_results.symmetric_difference(embedded_results)),
        )
    )
    encoded = html.split(
        '<script id="evidence-data" type="application/octet-stream">', 1
    )[1].split("</script>", 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        checks.append(dict(name="embedded_zip_crc", passed=archive.testzip() is None))
        hashes = json.loads(archive.read("SHA256.json"))
        bad = [
            name
            for name, digest in hashes.items()
            if sha256(archive.read(name)).hexdigest() != digest
        ]
        checks.append(dict(name="embedded_zip_hashes", passed=not bad, detail=bad))
        checks.append(
            dict(
                name="embedded_zip_has_inputs_and_licenses",
                passed=all(
                    name in archive.namelist()
                    for name in (
                        "case_definitions.json",
                        "licenses/PLOTLY_LICENSE.txt",
                        "licenses/MATHJAX_LICENSE.txt",
                    )
                ),
            )
        )
    return checks


async def settled(browser):
    return await browser.evaluate(
        """(async()=>{
      const deadline=Date.now()+30000;
      while(Date.now()<deadline){
        if(window.turboReport){const s=turboReport.snapshot();if(s.rendered===s.renderNumber)return s;}
        await new Promise(resolve=>setTimeout(resolve,30));
      }throw Error('Turbo Expo report did not settle: '+JSON.stringify(window.turboReportErrors||[]));
    })()"""
    )


async def exercise(url):
    checks = []
    report_bytes = (ROOT / "REPORT.html").read_bytes()
    download_directory = OUT / "downloads" / sha256(report_bytes).hexdigest()[:12]
    download_directory.mkdir(parents=True, exist_ok=True)
    encoded = (
        report_bytes.decode("utf-8")
        .split('<script id="evidence-data" type="application/octet-stream">', 1)[1]
        .split("</script>", 1)[0]
    )
    expected_zip_sha256 = sha256(base64.b64decode(encoded)).hexdigest()
    async with websockets.connect(url, max_size=100_000_000) as connection:
        browser = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await browser.command(domain + ".enable")
        await browser.command(
            "Network.emulateNetworkConditions",
            dict(offline=True, latency=0, downloadThroughput=0, uploadThroughput=0),
        )
        await browser.command(
            "Browser.setDownloadBehavior",
            dict(behavior="allow", downloadPath=str(download_directory)),
        )
        with tempfile.TemporaryDirectory(prefix="standalone_", dir=OUT) as isolated:
            html = Path(isolated) / "report.html"
            shutil.copyfile(ROOT / "REPORT.html", html)
            await browser.command(
                "Emulation.setDeviceMetricsOverride",
                dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False),
            )
            await browser.command("Page.navigate", dict(url=html.resolve().as_uri()))
            await settled(browser)
            checks.append(
                dict(
                    name="embedded_plotly",
                    passed=await browser.evaluate("Plotly.version==='2.35.2'"),
                )
            )
            checks.append(
                dict(
                    name="desktop_no_horizontal_overflow",
                    passed=await browser.evaluate(
                        "document.documentElement.scrollWidth<=innerWidth+2"
                    ),
                )
            )
            await browser.screenshot(OUT / "overview_light.png")
            case_ids = await browser.evaluate(
                "turboReport.data.definition.cases.map(item=>item.id)"
            )
            for case_id in case_ids:
                await browser.evaluate(f"turboReport.chooseCase({json.dumps(case_id)})")
                await settled(browser)
                grids = await browser.evaluate(
                    "Array.from(document.getElementById('grid').options).map(item=>item.value)"
                )
                for grid in grids:
                    await browser.evaluate(
                        f"document.getElementById('grid').value={json.dumps(grid)};document.getElementById('grid').dispatchEvent(new Event('change'))"
                    )
                    await settled(browser)
                    result = await browser.evaluate(
                        """(()=>{
                      const peers=turboReport.peerRuns(),bad=[],current=turboReport.data.definition.cases.find(item=>item.id===document.getElementById('case').value),xScale=['perfect_gas','spinelli'].includes(current.family)?1:1000;
                      for(const key of ['p','M','quality','u','rho','T','Z','mdot']){
                        const traces=document.getElementById('field-'+key).data;
                        for(const row of peers){
                          const f=row.fields||{},x=f.x_m||f.x||row.x_m;
                          const values=key==='p'?(f.p||f.pressure_Pa):key==='M'?(f.M||f.mach):key==='quality'?(f.quality||f.Q):key==='Z'?(f.Z||f.compressibility_factor):f[key];
                          if(!values||!x)continue;
                          const trace=traces.find(t=>t.name.startsWith(row.flux.toUpperCase()));
                          if(!trace){bad.push(row.id+':'+key+':missing');continue;}
                          const scale=key==='p'?1e-5:1;
                          if(trace.y.length!==values.length||!trace.y.every((value,i)=>value===(Number.isFinite(values[i])&&(key!=='quality'||(values[i]>=0&&values[i]<=1))?values[i]*scale:null)))bad.push(row.id+':'+key+':values');
                          if(!trace.x.every((value,i)=>value===x[i]*xScale))bad.push(row.id+':'+key+':coordinates');
                        }
                      }
                      for(const row of peers){
                        const f=row.fields||{},find=id=>document.getElementById(id).data.find(trace=>trace.name.startsWith(row.flux.toUpperCase()));
                        if(f.s&&f.T){const trace=find('field-Ts');if(!trace||!trace.x.every((value,i)=>value===f.s[i]/1000)||!trace.y.every((value,i)=>value===f.T[i]))bad.push(row.id+':Ts');}
                        if(f.s){const trace=find('field-entropy-rise'),s0=row.reservoir?.s0_J_kgK??row.reservoir?.s0??row.reservoir?.s??row.s0??f.s[0];if(!trace||!trace.y.every((value,i)=>value===f.s[i]-s0))bad.push(row.id+':entropy-rise');}
                      }
                      const references=[...Object.values(turboReport.data.references).flatMap(value=>value[current.id] || value.cases?.[current.id] || []),...(turboReport.selected()?.comparison?.series || [])];
                      for(const ref of references){
                        for(const key of ['p','M','u','rho','T','Z','liquid']){
                          const values=key==='p'?(ref.p_Pa||ref.pressure_Pa||ref.p):key==='Z'?(ref.Z||ref.compressibility_factor):key==='liquid'?(ref.liquid_mass_fraction||ref.liquid):ref[key];
                          if(!values||!ref.x_m)continue;
                          const trace=document.getElementById('field-'+key).data.find(item=>item.name===(ref.label||ref.kind||'Published reference'));
                          const scale=key==='p'?1e-5:1;
                          if(!trace||trace.y.length!==values.length||!trace.y.every((value,i)=>value===values[i]*scale)||!trace.x.every((value,i)=>value===ref.x_m[i]*xScale))bad.push(current.id+':'+(ref.label||ref.kind)+':reference-'+key);
                          if(key==='p'&&ref.kind?.startsWith('digitized_paper')&&trace&&(!trace.error_y.array.every((value,i)=>value===ref.pressure_uncertainty_Pa[i]*scale)||trace.error_x.value!==ref.x_uncertainty_m*xScale||trace.mode!=='markers'))bad.push(current.id+':digitized-uncertainty-or-mode');
                        }
                      }
                      return {passed:!bad.length,bad,peers:peers.length};
                    })()"""
                    )
                    checks.append(
                        dict(name=f"{case_id}:N{grid}:actual_profile_arrays", **result)
                    )
                    if case_id == "B3" and grid in ("100", "200"):
                        checks.append(
                            dict(
                                name=f"{case_id}:N{grid}:converged_wrong_branch_label",
                                passed=await browser.evaluate(
                                    "document.getElementById('selected-status').textContent.includes('Converged root; shock regime absent')"
                                ),
                            )
                        )
                    if (case_id, grid) in (("P1", "600"), ("B3", "100")):
                        await browser.evaluate(
                            "window.scrollTo(0,document.getElementById('profiles').offsetTop)"
                        )
                        await browser.screenshot(
                            OUT / f"{case_id}_N{grid}_profiles_light.png"
                        )
                await browser.evaluate("turboReport.showPage('convergence')")
                await settled(browser)
                checks.append(
                    dict(
                        name=f"{case_id}:convergence_log_axes",
                        passed=await browser.evaluate(
                            "document.getElementById('history-residual')._fullLayout.yaxis.type==='log'&&document.getElementById('history-cfl')._fullLayout.yaxis.type==='log'"
                        ),
                    )
                )
            if case_ids:
                await browser.evaluate(
                    f"turboReport.chooseCase({json.dumps(case_ids[0])});window.scrollTo(0,document.getElementById('profiles').offsetTop)"
                )
                await settled(browser)
                await browser.screenshot(OUT / "profiles_light.png")
                await browser.evaluate("document.getElementById('theme').click()")
                await settled(browser)
                checks.append(
                    dict(
                        name="theme_dark",
                        passed=await browser.evaluate(
                            "document.documentElement.dataset.theme==='dark'&&document.getElementById('field-p')._fullLayout.paper_bgcolor===getComputedStyle(document.documentElement).getPropertyValue('--paper').trim()"
                        ),
                    )
                )
                await browser.screenshot(OUT / "profiles_dark.png")
                await browser.evaluate(
                    "turboReport.showPage('convergence');window.scrollTo(0,document.getElementById('convergence').offsetTop)"
                )
                await settled(browser)
                await browser.screenshot(OUT / "convergence_dark.png")
            await browser.evaluate("turboReport.showPage('method')")
            await settled(browser)
            math = await browser.evaluate(
                "(async()=>{if(MathJax.startup?.promise)await MathJax.startup.promise;return document.querySelectorAll('mjx-container').length;})()"
            )
            checks.append(
                dict(name="offline_equations_typeset", passed=math >= 2, detail=math)
            )
            if await browser.evaluate(
                "Boolean(turboReport.data.references.paper_digitization)"
            ):
                images = await browser.evaluate(
                    "(async()=>{const images=[...document.querySelectorAll('.qa-image')];await Promise.all(images.map(image=>image.decode()));return images.length===2&&images.every(image=>image.naturalWidth===3600);})()"
                )
                checks.append(
                    dict(name="embedded_digitization_QA_panels", passed=images)
                )
            await browser.command(
                "Emulation.setDeviceMetricsOverride",
                dict(width=390, height=844, deviceScaleFactor=1, mobile=True),
            )
            await browser.evaluate(
                "turboReport.showPage('profiles');window.scrollTo(0,0)"
            )
            await settled(browser)
            checks.append(
                dict(
                    name="mobile_no_horizontal_overflow",
                    passed=await browser.evaluate(
                        "document.documentElement.scrollWidth<=innerWidth+2"
                    ),
                )
            )
            await browser.screenshot(OUT / "mobile_dark.png")
            download_started = time.time()
            await browser.evaluate(
                "document.getElementById('download-evidence').click()"
            )
            await asyncio.sleep(0.5)
            checks.append(
                dict(
                    name="offline_no_external_requests",
                    passed=not [
                        request
                        for request in browser.requests
                        if request.startswith(("http:", "https:"))
                    ],
                    detail=[
                        request
                        for request in browser.requests
                        if request.startswith(("http:", "https:"))
                    ],
                )
            )
            checks.append(
                dict(
                    name="no_javascript_exceptions",
                    passed=not browser.exceptions,
                    detail=browser.exceptions,
                )
            )
    complete = False
    for _ in range(50):
        fresh = [
            path
            for path in download_directory.glob("*.zip")
            if path.stat().st_mtime >= download_started
        ]
        if fresh:
            downloaded = max(fresh, key=lambda path: path.stat().st_mtime)
            try:
                with zipfile.ZipFile(downloaded) as archive:
                    complete = (
                        archive.testzip() is None
                        and "case_definitions.json" in archive.namelist()
                        and sha256(downloaded.read_bytes()).hexdigest()
                        == expected_zip_sha256
                    )
                break
            except zipfile.BadZipFile:
                pass
        await asyncio.sleep(0.1)
    checks.append(dict(name="actual_browser_evidence_zip_download", passed=complete))
    return checks


def browser_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "downloads").mkdir(exist_ok=True)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    with tempfile.TemporaryDirectory(
        prefix="turbo_edge_", dir=OUT, ignore_cleanup_errors=True
    ) as profile:
        args = [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--disable-extensions",
            "--disable-crash-reporter",
            "--disable-breakpad",
            "--no-default-browser-check",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ]
        with (OUT / "edge.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                args,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                targets, deadline = [], time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/json/list", timeout=1
                        ) as response:
                            targets = json.load(response)
                        if any(target["type"] == "page" for target in targets):
                            break
                    except (OSError, TimeoutError):
                        pass
                    time.sleep(0.15)
                target = next(
                    (target for target in targets if target["type"] == "page"), None
                )
                if target is None:
                    raise RuntimeError(
                        "Headless Edge unavailable; inspect " + str(OUT / "edge.log")
                    )
                return asyncio.run(exercise(target["webSocketDebuggerUrl"]))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    checks = structural_audit()
    if not args.structural_only:
        checks.extend(browser_audit())
    result = dict(passed=all(check["passed"] for check in checks), checks=checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("structural.json" if args.structural_only else "audit.json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                checks=len(checks),
                failed=[check["name"] for check in checks if not check["passed"]],
            ),
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
