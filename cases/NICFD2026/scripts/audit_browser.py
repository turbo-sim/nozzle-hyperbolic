"""Exercise all six standalone reports offline in isolated headless Edge."""

import asyncio
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import websockets

from development.real_gas.scripts.audit_report import Browser, EDGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/browser_audit"


async def settled(browser):
    return await browser.evaluate(
        """(async()=>{
      const end=Date.now()+30000;
      while(Date.now()<end){
        if(window.nicfdReport){const s=nicfdReport.snapshot();if(s.renderNumber===s.rendered)return s;}
        await new Promise(r=>setTimeout(r,30));
      }throw Error('NICFD report did not settle');
    })()"""
    )


async def exercise(url):
    checks = []
    manifest = json.loads((ROOT / "report_manifest.json").read_text(encoding="utf-8"))
    async with websockets.connect(url, max_size=80_000_000) as connection:
        b = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await b.command(domain + ".enable")
        await b.command(
            "Network.emulateNetworkConditions",
            dict(offline=True, latency=0, downloadThroughput=0, uploadThroughput=0),
        )
        await b.command(
            "Browser.setDownloadBehavior",
            dict(behavior="allow", downloadPath=str(OUT / "downloads")),
        )
        for item in manifest["cases"]:
            tag = item["family"] + "_" + item["regime"]
            # Copy just this HTML, with no supporting files, into a fresh folder.
            with tempfile.TemporaryDirectory(prefix="standalone_", dir=OUT) as isolated:
                html = Path(isolated) / "report.html"
                shutil.copyfile(ROOT / item["path"], html)
                await b.command(
                    "Emulation.setDeviceMetricsOverride",
                    dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False),
                )
                await b.command("Page.navigate", dict(url=html.resolve().as_uri()))
                await settled(b)
                checks.append(
                    dict(
                        name=tag + ":embedded_runtime",
                        passed=await b.evaluate("Plotly.version==='2.35.2'"),
                    )
                )
                checks.append(
                    dict(
                        name=tag + ":opening_verdict",
                        passed=await b.evaluate(
                            "document.getElementById('overall-verdict').innerText.includes(nicfdReport.data.verdict.title)"
                        ),
                    )
                )
                modes = await b.evaluate("nicfdReport.data.runs.map(r=>r.mode)")
                for mode in modes:
                    await b.evaluate(
                        f"document.getElementById('mode').value={json.dumps(mode)};document.getElementById('mode').dispatchEvent(new Event('change'))"
                    )
                    await settled(b)
                    cells = await b.evaluate(
                        "Array.from(document.getElementById('grid').options).map(o=>o.value).filter(Boolean)"
                    )
                    for mesh in cells or [""]:
                        await b.evaluate(
                            f"document.getElementById('grid').value={json.dumps(mesh)}"
                        )
                        for group in ("flow", "thermal", "mixture"):
                            await b.evaluate(
                                f"document.getElementById('field-group').value='{group}';nicfdReport.render()"
                            )
                            await settled(b)
                            actual = await b.evaluate(
                                """(()=>{
                              const a=nicfdReport,s=a.stage(),snapshot=a.snapshot(),bad=[];
                              const scale=k=>k==='p'?1e-5:['h_g','h_l'].includes(k)?.001:1;
                              for(const key of snapshot.fields){
                                const t=document.getElementById('field-'+key).data;
                                const e=document.getElementById('error-'+key);
                                if(t.length!==(s?.fields?2:1))bad.push(key+':trace-count');
                                if(!t[0].y.every((v,i)=>v===a.data.reference[key][i]*scale(key)))bad.push(key+':published-values');
                                if(s?.fields){const idx=a.markerIndices(s.cells);
                                  if(t[1].marker.color!=='#fff'||!t[1].y.every((v,i)=>v===s.fields[key][idx[i]]*scale(key)))bad.push(key+':current-values');
                                  if(e.data[0].y.length!==s.cells||e._fullLayout.yaxis.type!=='log')bad.push(key+':error-dimensions');
                                }
                              }return {passed:!bad.length,bad};
                            })()"""
                            )
                            checks.append(
                                dict(
                                    name=f"{tag}:{mode}:N{mesh or 'none'}:{group}:actual_plot_data",
                                    **actual,
                                )
                            )
                for page in ("profiles", "convergence", "timings", "method"):
                    await b.evaluate(f"nicfdReport.showPage('{page}')")
                    await settled(b)
                    checks.append(
                        dict(
                            name=tag + ":page:" + page,
                            passed=await b.evaluate(
                                f"!document.getElementById('{page}').hidden"
                            ),
                        )
                    )
                await b.evaluate(
                    "nicfdReport.showPage('profiles');document.getElementById('field-group').value='flow';document.getElementById('theme-light').click()"
                )
                await settled(b)
                if item["regime"] == "subsonic":
                    await b.evaluate("window.scrollTo(0,0)")
                    await b.screenshot(OUT / (tag + "_desktop.png"))
                    await b.evaluate(
                        "document.getElementById('profile-plots').scrollIntoView()"
                    )
                    await b.screenshot(OUT / (tag + "_profiles_light.png"))
                for visibility in ("published", "current", "both"):
                    await b.evaluate(
                        f"document.getElementById('visibility').value='{visibility}';nicfdReport.render()"
                    )
                    await settled(b)
                    expected = await b.evaluate(
                        f"(nicfdReport.stage()?.fields?1:0)*Number('{visibility}'!=='published')+Number('{visibility}'!=='current')"
                    )
                    count = await b.evaluate(
                        "document.getElementById('field-p').data.length"
                    )
                    checks.append(
                        dict(
                            name=tag + ":visibility:" + visibility,
                            passed=count == expected,
                        )
                    )
                baseline = await b.evaluate(
                    "JSON.stringify(document.getElementById('error-p').data)"
                )
                for skip in (0, 7, 999):
                    await b.evaluate(
                        f"document.getElementById('skip').value='{skip}';nicfdReport.render()"
                    )
                    await settled(b)
                    marker_ok = await b.evaluate(
                        "(()=>{const s=nicfdReport.stage();if(!s?.fields)return true;const t=document.getElementById('field-p').data.at(-1),idx=nicfdReport.markerIndices(s.cells);return t.y.length===idx.length&&t.x.at(-1)===s.fields.x.at(-1);})()"
                    )
                    unchanged = (
                        await b.evaluate(
                            "JSON.stringify(document.getElementById('error-p').data)"
                        )
                        == baseline
                    )
                    checks.append(
                        dict(
                            name=f"{tag}:marker_skip:{skip}",
                            passed=marker_ok and unchanged,
                        )
                    )
                for axis in ("steps", "time", "work"):
                    await b.evaluate(
                        f"document.getElementById('history-axis').value='{axis}';nicfdReport.showPage('convergence')"
                    )
                    await settled(b)
                    checks.append(
                        dict(
                            name=tag + ":history_axis:" + axis,
                            passed=await b.evaluate(
                                "document.getElementById('residual').data.length===6"
                            ),
                        )
                    )
                await b.evaluate(
                    "document.getElementById('theme-dark').click();nicfdReport.showPage('profiles')"
                )
                await settled(b)
                checks.append(
                    dict(
                        name=tag + ":dark_theme",
                        passed=await b.evaluate(
                            "document.documentElement.dataset.theme==='dark'"
                        ),
                    )
                )
                await b.command(
                    "Emulation.setDeviceMetricsOverride",
                    dict(width=390, height=844, deviceScaleFactor=1, mobile=True),
                )
                await b.evaluate(
                    "document.getElementById('visibility').value='both';document.getElementById('skip').value='3';nicfdReport.render();window.scrollTo(0,0)"
                )
                await settled(b)
                checks.append(
                    dict(
                        name=tag + ":mobile_no_page_overflow",
                        passed=await b.evaluate(
                            "document.documentElement.scrollWidth<=innerWidth+2"
                        ),
                    )
                )
                if item["regime"] == "subsonic":
                    await b.screenshot(OUT / (tag + "_mobile.png"))
                    await b.evaluate(
                        "document.getElementById('profile-plots').scrollIntoView()"
                    )
                    await b.screenshot(OUT / (tag + "_profiles_mobile.png"))
                for button in (
                    "download-reference",
                    "download-json",
                    "download-profile",
                    "download-history",
                ):
                    await b.evaluate(f"document.getElementById('{button}').click()")
        external = sorted(
            {
                request
                for request in b.requests
                if request.startswith(("https:", "http:"))
            }
        )
        checks.append(
            dict(
                name="offline_no_external_requests",
                passed=not external,
                detail=external,
            )
        )
        checks.append(
            dict(
                name="no_javascript_exceptions",
                passed=not b.exceptions,
                detail=b.exceptions,
            )
        )
    # Downloads are real browser downloads, not source-code-only assertions.
    names = [p.name for p in (OUT / "downloads").glob("*")]
    checks.append(
        dict(
            name="actual_embedded_downloads",
            passed=any(
                n.startswith("NICFD2026_") and n.endswith(".json") for n in names
            )
            and "published_reference.csv" in names
            and "current_profile.csv" in names
            and "convergence_history.csv" in names,
        )
    )
    return dict(
        passed=all(item["passed"] for item in checks),
        checks=checks,
        reports=manifest["reports"],
        screenshots=[p.name for p in OUT.glob("*.png")],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "downloads").mkdir(exist_ok=True)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    with tempfile.TemporaryDirectory(
        prefix="nicfd_edge_", ignore_cleanup_errors=True
    ) as profile:
        command = [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-crash-reporter",
            "--disable-breakpad",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--disable-extensions",
            "--no-default-browser-check",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ]
        with (OUT / "edge.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
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
                result = asyncio.run(exercise(target["webSocketDebuggerUrl"]))
                (OUT / "audit.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        dict(
                            passed=result["passed"],
                            checks=len(result["checks"]),
                            failed=[
                                item["name"]
                                for item in result["checks"]
                                if not item["passed"]
                            ],
                        ),
                        indent=2,
                    )
                )
                if not result["passed"]:
                    raise SystemExit(1)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
