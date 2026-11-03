"""Audit saved comparison data and, optionally, the standalone report in Edge.

No flow-model, fluid or numerical-runner imports occur. The browser mode copies
the HTML to its audit folder and exercises the actual embedded Plotly figures.
"""

import argparse
import asyncio
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request
from zipfile import ZipFile

import numpy as np

from development.flux_methods.shared.audit_offline_report import Document, audit
from development.flux_methods.shared.offline_report import clean
from .comparison_report import _fields

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "outputs/report_presentation/comparison_audit"


def structural():
    """Check new evidence without changing the original input manifest."""
    legacy = audit(STUDY)
    document = Document()
    document.feed((STUDY / "REPORT.html").read_text(encoding="utf-8"))
    payload = json.loads("".join(document.payload))
    saved = STUDY / "outputs/solver_comparison/campaign.json"
    if not saved.is_file():
        return dict(passed=True, skipped="No saved solver-comparison campaign", legacy=legacy)
    raw = json.loads(saved.read_text(encoding="utf-8"))
    data = payload["solver_comparison"]
    digest = sha256(saved.read_bytes()).hexdigest()
    expected_manifest = {saved.relative_to(STUDY).as_posix(): digest}
    sensitivity = saved.with_name("reference_sensitivity.json")
    if sensitivity.is_file():
        expected_manifest[sensitivity.relative_to(STUDY).as_posix()] = sha256(sensitivity.read_bytes()).hexdigest()
        assert data["reference_sensitivity"] == json.loads(sensitivity.read_text(encoding="utf-8"))
    if raw.get("previous_campaign"):
        previous=raw["previous_campaign"]
        expected_manifest[(STUDY/previous["file"]).relative_to(STUDY).as_posix()]=previous["sha256"]
        assert sha256((STUDY/previous["file"]).read_bytes()).hexdigest()==previous["sha256"]
    assert data["input_sha256"] == expected_manifest
    manifest = STUDY / "outputs/report_presentation/comparison_input_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8")) == expected_manifest
    assert "cmp-field-p" in payload["comparison_markup"]
    geometry = payload["geometry_profile"]
    assert geometry["saved_geometry_sample_audit"] is True
    assert len(geometry["control_x_m"]) == len(geometry["control_half_height_m"]) == 33
    assert len(geometry["x_m"]) == len(geometry["half_height_m"]) == 501
    assert geometry["depth_m"] == payload["settings"]["geometry"]["width_m"]
    assert geometry["control_half_height_m"][0] == payload["settings"]["geometry"]["inlet_half_height_m"]
    counts = dict(cases=0, fvm=0, successful_space=0, successful_fvm=0,
                  exact_center_references=0, retained_space_attempts=0, retained_refinements=0)
    for physics, variant in raw["variants"].items():
        actual_variant = data["variants"][physics]
        for case_id, entry in variant["cases"].items():
            counts["cases"] += 1
            actual = actual_variant["cases"][case_id]
            space = actual["space"]
            assert space["fields"] == clean(_fields(entry["space"].get("fields", {}))), (physics, case_id)
            for key in ("success", "reason", "critical", "audits", "history", "integration", "source"):
                assert space.get(key) == clean(entry["space"].get(key)), (physics, case_id, key)
            x = np.asarray(space["x_m"], dtype=float)
            assert np.isfinite(x).all() and (not x.size or np.all(np.diff(x) > 0))
            counts["successful_space"] += bool(space.get("success"))
            for key in ("space_attempts", "reference_refinements"):
                assert actual.get(key) == clean(entry.get(key)), (physics, case_id, key)
            counts["retained_space_attempts"] += len(actual.get("space_attempts", []))
            counts["retained_refinements"] += len(actual.get("reference_refinements", []))
            assert len(actual["fvm"]) == len(entry.get("fvm", actual["fvm"]))
            successful={r["cells"]:r for r in entry.get("fvm",[]) if r.get("success")}
            assert set(successful)==set(raw["settings"]["grids"]), (physics,case_id,"incomplete sequence")
            for n,row in successful.items():
                assert row.get("saved_verification",{}).get("verified") is True
                assert all(row["steady"]["residual"][k]==value
                           for k,value in payload["settings"]["residual"].items())
                assert row["steady"]["cfl"] == .2
                previous=[level for level in raw["settings"]["grids"] if level<n]
                if previous:
                    assert successful[max(previous)]["id"] in row["initialization"]
                else:
                    assert row["initialization"] == "model cold Mach ramp"
            for source, row in zip(entry.get("fvm", []), actual["fvm"]):
                counts["fvm"] += 1
                counts["successful_fvm"] += bool(row.get("success"))
                assert row["cells"] in raw["settings"]["grids"]
                assert row["scheme"]["flux"] == "hll"
                assert row["scheme"]["reconstruction"] == "muscl_van_leer"
                assert row["fields"] == clean(_fields(source.get("fields", {})))
                assert row["x_m"] == source.get("x_m", [])
                for key in ("success", "reason", "metrics", "history", "scheme", "initialization", "source_epoch"):
                    assert row.get(key) == clean(source.get(key)), (physics, case_id, row.get("id"), key)
                reference = _fields(source.get("reference_fields", source.get("space_fields", {})))
                if reference:
                    assert row["reference_fields"] == clean(reference)
                    assert row["reference_sampling"] == "Space marcher sampled at the FVM cell centers"
                    counts["exact_center_references"] += 1
                else:
                    for key, values in space["fields"].items():
                        if x.size > 1:
                            expected = np.interp(row["x_m"], x, np.asarray(values, dtype=float),
                                                 left=np.nan, right=np.nan)
                            assert row["reference_fields"][key] == clean(expected)
    with ZipFile(BytesIO(document.downloads["report_data_and_evidence.zip"])) as archive:
        for name, digest in expected_manifest.items():
            assert sha256(archive.read("original/" + name)).hexdigest() == digest
        assert json.loads(archive.read("comparison_input_manifest.json")) == expected_manifest
    render = json.loads((STUDY / "outputs/report_presentation/render.json").read_text())
    assert render["html_sha256"] == sha256((STUDY / "REPORT.html").read_bytes()).hexdigest()
    assert render["comparison_input_sha256"] == expected_manifest
    result = dict(passed=True, legacy=legacy, comparison_input_sha256=expected_manifest,
                  **counts, numerical_cases_rerun=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "structural.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


async def settled(browser):
    return await browser.evaluate("""(async()=>{
      const end=Date.now()+30000;
      while(Date.now()<end){
        if(window.studyReport){const s=studyReport.snapshot();if(s.renderNumber===s.rendered)return s;}
        await new Promise(r=>setTimeout(r,30));
      }throw Error('Report did not finish rendering');
    })()""")


async def exercise(url, report):
    import websockets
    from development.real_gas.scripts.audit_report import Browser
    checks = []
    async with websockets.connect(url, max_size=40_000_000) as connection:
        b = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await b.command(f"{domain}.enable")
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False))
        await b.command("Page.navigate", dict(url=report.resolve().as_uri()))
        initial = await settled(b)
        checks.append(dict(name="comparison_default_page", passed=initial["page"] == "comparison"))
        checks.append(dict(name="embedded_plotly", passed=await b.evaluate("Plotly.version==='2.35.2'")))
        checks.append(dict(name="shared_geometry_contour", passed=await b.evaluate("(()=>{const p=document.getElementById('cmp-geometry'),g=studyReport.data.geometry_profile;return p.data.length===3&&p.data[0].y.length===501&&p.data[2].y.length===33&&p.data[0].y.every((v,i)=>v===g.half_height_m[i]*1000)&&p.data[1].y.every((v,i)=>v===-g.half_height_m[i]*1000);})()")))
        checks.append(dict(name="native_equations_and_reference_sensitivity", passed=await b.evaluate("document.querySelectorAll('#page-comparison math mtr').length===3&&document.querySelector('#page-comparison math').getBoundingClientRect().height>65&&document.querySelectorAll('#cmp-sensitivity-body tr').length===4")))
        await b.screenshot(OUT / "desktop_controls.png")
        await b.evaluate("document.getElementById('cmp-geometry').scrollIntoView()")
        await b.screenshot(OUT / "geometry_and_equations.png")
        physics = await b.evaluate("Object.keys(studyReport.data.solver_comparison.variants)")
        # Check the actual chart arrays against independent arithmetic on saved fields.
        expression = """(()=>{
          const api=studyReport.comparison,r=api.row(),s=api.space(),bad=[];
          const valid=x=>typeof x==='number'&&Number.isFinite(x);
          const field=(o,k)=>k==='liquid'?(o?.quality||[]).map(q=>valid(q)&&q>=0&&q<=1?1-q:null):(o?.[k]||[]);
          for(const k of ['p','u','rho','T','M','liquid']){
            const chart=document.getElementById('cmp-field-'+k),error=document.getElementById('cmp-error-'+k),scale=k==='p'?1e-5:1;
            const ref=chart.data.find(t=>t.name.startsWith('NozzleX')),fvm=chart.data.find(t=>t.name.startsWith('FVM'));
            if(s.success){if(!ref||ref.mode!=='lines'||!ref.y.every((v,i)=>v===(valid(field(s.fields,k)[i])?field(s.fields,k)[i]*scale:null)))bad.push(k+':reference');}
            else if(ref||error.data.length)bad.push(k+':failed-reference-visible');
            if(r?.x_m?.length){
              const ii=api.indices(r.x_m.length,api.snapshot().skip),a=field(r.fields,k),z=field(r.reference_fields,k);
              if(!fvm||chart.data.at(-1)!==fvm||fvm.mode!=='markers'||fvm.marker.color!=='#ffffff')bad.push(k+':marker-style');
              if(fvm&&(!fvm.y.every((v,j)=>v===(valid(a[ii[j]])?a[ii[j]]*scale:null))||fvm.x.at(-1)!==r.x_m.at(-1)*1000))bad.push(k+':marker-values');
              if(s.success){const d=a.map((v,i)=>!valid(v)||!valid(z[i])?null:k==='liquid'?100*Math.abs(v-z[i]):z[i]===0?null:100*Math.abs((v-z[i])/z[i]));
                if(error._fullLayout.yaxis.type!=='log'||error.data.length!==1||!error.data[0].y.every((v,i)=>v===(d[i]===0?null:d[i]))||JSON.stringify(error.data[0].customdata)!==JSON.stringify(d))bad.push(k+':difference');
              }
            }
          }return {passed:!bad.length,bad,cells:r?.cells,case:api.snapshot().caseId,physics:api.snapshot().physics};
        })()"""
        completed = None
        for wall in physics:
            await b.evaluate(f"document.getElementById('cmp-physics').value={json.dumps(wall)};document.getElementById('cmp-physics').dispatchEvent(new Event('change'))")
            cases = await b.evaluate(f"Object.keys(studyReport.data.solver_comparison.variants[{json.dumps(wall)}].cases)")
            for case in cases:
                await b.evaluate(f"document.getElementById('case').value={json.dumps(case)};document.getElementById('case').dispatchEvent(new Event('change'))")
                await settled(b)
                grids = await b.evaluate("[...document.getElementById('cmp-cells').options].map(o=>o.value)")
                assert grids==["50","100","200"], (wall,case,grids)
                for cells in grids:
                    await b.evaluate(f"document.getElementById('cmp-cells').value={json.dumps(cells)};document.getElementById('cmp-cells').dispatchEvent(new Event('change'))")
                    await settled(b)
                    runs = await b.evaluate("[...document.getElementById('cmp-run').options].map(o=>o.value)")
                    for run in runs:
                        await b.evaluate(f"document.getElementById('cmp-run').value={json.dumps(run)};document.getElementById('cmp-run').dispatchEvent(new Event('change'))")
                        await settled(b)
                        result = await b.evaluate(expression)
                        checks.append(dict(name=f"paired_fields_{wall}_{case}_{cells}_{run}", **result))
                        if await b.evaluate("!!studyReport.comparison.space().success&&!!studyReport.comparison.row()?.x_m?.length"):
                            completed = (wall, case, cells, run)
        if completed:
            wall, case, cells, run = completed
            await b.evaluate(f"document.getElementById('cmp-physics').value={json.dumps(wall)};document.getElementById('cmp-physics').dispatchEvent(new Event('change'));document.getElementById('case').value={json.dumps(case)};document.getElementById('case').dispatchEvent(new Event('change'))")
            await settled(b)
            await b.evaluate(f"document.getElementById('cmp-cells').value={json.dumps(cells)};document.getElementById('cmp-cells').dispatchEvent(new Event('change'));document.getElementById('cmp-run').value={json.dumps(run)};document.getElementById('cmp-run').dispatchEvent(new Event('change'))")
            await settled(b)
            baseline = await b.evaluate("document.getElementById('cmp-error-p').data[0].y")
            full_csv = await b.evaluate("studyReport.comparison.csv()")
            for mode in ("fvm", "space", "both"):
                await b.evaluate(f"document.getElementById('cmp-solvers').value='{mode}';document.getElementById('cmp-solvers').dispatchEvent(new Event('change'))")
                await settled(b)
                actual = await b.evaluate("document.getElementById('cmp-field-p').data.filter(t=>t.name.startsWith('FVM')||t.name.startsWith('NozzleX')).map(t=>t.mode)")
                expected = {"fvm": ["markers"], "space": ["lines"], "both": ["lines", "markers"]}[mode]
                unchanged = await b.evaluate("document.getElementById('cmp-error-p').data[0].y") == baseline
                checks.append(dict(name=f"solver_visibility_{mode}", passed=actual == expected and unchanged))
            for skip in (0, 1, 7, 999):
                await b.evaluate(f"document.getElementById('cmp-skip').value='{skip}';document.getElementById('cmp-skip').dispatchEvent(new Event('change'))")
                await settled(b)
                result = await b.evaluate(expression)
                unchanged = await b.evaluate("document.getElementById('cmp-error-p').data[0].y") == baseline
                checks.append(dict(name=f"skip_{skip}_preserves_all_error_and_csv_cells", passed=result["passed"] and unchanged and await b.evaluate("studyReport.comparison.csv()") == full_csv))
            synthetic = await b.evaluate("""(()=>{const a=studyReport.comparison,r={fields:{p:[5,7,1],quality:[1,.8,-1]},reference_fields:{p:[5,6,0],quality:[1,1,.5]}};const p=a.difference(r,'p'),q=a.difference(r,'liquid');return p[0]===0&&Math.abs(p[1]-100/6)<1e-12&&p[2]===null&&q[0]===0&&Math.abs(q[1]-20)<1e-12&&q[2]===null;})()""")
            checks.append(dict(name="zero_reference_and_liquid_quality_mask", passed=synthetic))
            await b.evaluate("(async()=>{await Plotly.relayout(document.getElementById('cmp-field-p'),{'xaxis.range':[0,25]});await new Promise(r=>setTimeout(r,80));})()")
            checks.append(dict(name="linked_profile_error_zoom", passed=await b.evaluate("document.getElementById('cmp-error-p')._fullLayout.xaxis.range") == [0, 25]))
            await b.evaluate("document.getElementById('cmp-skip').value='3';document.getElementById('cmp-skip').dispatchEvent(new Event('change'))")
            await settled(b)
            await b.evaluate("document.getElementById('cmp-field-p').scrollIntoView();document.getElementById('report-theme').value='dark';document.getElementById('report-theme').dispatchEvent(new Event('change'))")
            await settled(b)
            checks.append(dict(name="dark_theme_white_marker_faces", passed=await b.evaluate("document.documentElement.dataset.theme==='dark'&&document.getElementById('cmp-field-p').data.at(-1).marker.color==='#ffffff'")))
            await b.screenshot(OUT / "profiles_dark.png")
            mesh_ok=await b.evaluate("""(()=>{
              const data=studyReport.data.solver_comparison.variants[document.getElementById('cmp-physics').value].cases;
              const rows=[...document.querySelectorAll('#cmp-mesh-rows tr')];
              return rows.length===Object.keys(data).length&&rows.every(row=>{
                const cells=[...row.cells],item=data[cells[0].textContent];
                return cells.length===6&&[50,100,200].every((n,i)=>{
                  const saved=item.fvm.filter(r=>r.cells===n&&r.success).at(-1).errors.p.relative_percent_rms;
                  return Math.abs(Number(cells[i+1].textContent)-saved)<.005*Math.max(1,Math.abs(saved));
                });
              });})()""")
            checks.append(dict(name="grid_accuracy_table_matches_saved_errors",passed=mesh_ok))
            await b.evaluate("document.getElementById('cmp-mesh-heading').scrollIntoView()")
            await b.screenshot(OUT / "grid_accuracy.png")
        for page in ("overview", "explorer", "discussion", "protocol"):
            await b.evaluate(f"studyReport.showPage('{page}')")
            await settled(b)
            checks.append(dict(name=f"preserved_page_{page}", passed=await b.evaluate(f"!document.getElementById('page-{page}').hidden&&!document.body.classList.contains('comparison-mode')")))
        checks.append(dict(name="original_thirteen_attempts_retained", passed=await b.evaluate("studyReport.data.records.length===13")))
        await b.evaluate("studyReport.showPage('comparison')")
        await settled(b)
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=390, height=844, deviceScaleFactor=1, mobile=True))
        await b.evaluate("window.scrollTo(0,0);window.dispatchEvent(new Event('resize'));studyReport.render()")
        await settled(b)
        await asyncio.sleep(.2)
        checks.append(dict(name="mobile_no_horizontal_overflow", passed=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        await b.screenshot(OUT / "mobile_controls.png")
        await b.evaluate("document.querySelector('#page-comparison math').scrollIntoView()")
        await b.screenshot(OUT / "mobile_equations.png")
        await b.evaluate("document.getElementById('cmp-field-p').scrollIntoView()")
        await b.screenshot(OUT / "mobile_profiles.png")
        external = sorted({url for url in b.requests if url.startswith(("http:", "https:"))})
        checks.append(dict(name="offline_no_external_requests", passed=not external, detail=external))
        checks.append(dict(name="no_javascript_exceptions", passed=not b.exceptions, detail=b.exceptions))
        return dict(passed=all(c["passed"] for c in checks), checks=checks,
                    html_sha256=sha256((STUDY / "REPORT.html").read_bytes()).hexdigest(),
                    browser="Isolated headless Edge via repository CDP audit client",
                    screenshots=[p.name for p in sorted(OUT.glob("*.png"))])


def browser_audit():
    from development.real_gas.scripts.audit_report import EDGE
    OUT.mkdir(parents=True, exist_ok=True)
    standalone = OUT / "standalone.html"
    shutil.copyfile(STUDY / "REPORT.html", standalone)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run",
               "--disable-crash-reporter", "--disable-breakpad", "--disable-background-networking",
               "--disable-component-update", "--disable-sync", "--disable-extensions", "--no-default-browser-check",
               f"--remote-debugging-port={port}", f"--user-data-dir={OUT / 'browser_profile'}", "about:blank"]
    with (OUT / "edge.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            targets = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                        targets = json.load(response)
                    if any(t["type"] == "page" for t in targets):
                        break
                except (OSError, TimeoutError):
                    pass
                time.sleep(.15)
            target = next((t for t in targets if t["type"] == "page"), None)
            if target is None:
                raise RuntimeError(f"Headless Edge unavailable; inspect {OUT / 'edge.log'}")
            result = asyncio.run(exercise(target["webSocketDebuggerUrl"], standalone))
            (OUT / "browser.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            assert result["passed"], [c["name"] for c in result["checks"] if not c["passed"]]
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true", help="Also exercise the copied standalone HTML in Edge")
    args = parser.parse_args()
    result = structural()
    if args.browser and not result.get("skipped"):
        result = browser_audit()
    print(json.dumps(dict(passed=result["passed"], checks=len(result.get("checks", [])),
                         skipped=result.get("skipped")), indent=2))


if __name__ == "__main__":
    main()
