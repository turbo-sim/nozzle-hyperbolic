"""Exercise the standalone report in an isolated local headless Edge profile.

This audit reads a saved HTML report; it never imports or executes a flow
solver. All browser state and screenshots remain under the study outputs.
"""
from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import websockets

from development.real_gas.scripts.audit_report import Browser as BaseBrowser, EDGE
from .report import STUDY


class Browser(BaseBrowser):
    async def settled(self):
        return await self.evaluate("""(async()=>{
          const end=Date.now()+30000;
          while(Date.now()<end){
            if(window.backpressureReportLoadError)throw new Error(window.backpressureReportLoadError);
            if(window.backpressureReport){const s=backpressureReport.snapshot();
              if(s.renderNumber===s.rendered)return s;}
            await new Promise(r=>setTimeout(r,30));
          }
          throw new Error('Report rendering did not settle');
        })()""")


async def exercise(websocket_url, html, output):
    checks = []
    async with websockets.connect(websocket_url, max_size=80_000_000) as connection:
        b = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await b.command(f"{domain}.enable")
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=1440, height=1050, deviceScaleFactor=1, mobile=False))
        await b.command("Browser.setDownloadBehavior", dict(behavior="allow", downloadPath=str(output)))
        await b.command("Page.navigate", dict(url=html.resolve().as_uri()))
        initial = await b.settled()
        checks.append(dict(name="initial_render", passed=bool(initial), detail=initial))
        checks.append(dict(name="embedded_plotly", passed=await b.evaluate("typeof Plotly==='object'&&Plotly.version==='2.35.2'")))
        checks.append(dict(name="native_success_and_verified_statuses_distinct", passed=await b.evaluate("backpressureReport.campaign.results.every(r=>backpressureReport.verified(r)===(r.success===true&&r.audit?.passed===true))")))
        counts = await b.evaluate("""(()=>{
          const c=backpressureReport.campaign,rows=c.results,expected=c.cases.length*48;
          const verified=rows.filter(backpressureReport.verified).length;
          const final=rows.filter(r=>r.cells===200),target=final.filter(r=>r.target_correct===true).length;
          const joint=final.filter(r=>backpressureReport.verified(r)&&r.target_correct===true).length;
          const summary=document.getElementById('physical-target-summary');
          const audit=c.saved_audits?.['campaign_audit.json'],complete=rows.length===expected;
          const fresh=!complete||(audit?.passed===true&&audit.fresh_equations_evaluated===true&&audit.records?.length===rows.length&&audit.records.every(r=>r.passed&&r.checks?.fresh_audit_matches&&r.checks?.fresh_target_matches));
          return {rows:rows.length,expected,verified,shock_diagnostics_passed:target,joint_root_and_shock_passed:joint,
            complete,fresh_equations_evaluated:audit?.fresh_equations_evaluated??false,
            passed:document.getElementById('hero-verified').textContent===`${verified} / ${expected}`&&
              summary.textContent.includes(`${joint} / ${c.cases.length*16}`)&&
              Number(summary.dataset.jointPassed)===joint&&Number(summary.dataset.diagnosticsPassed)===target&&fresh};
        })()""")
        checks.append(dict(name="truthful_root_and_physical_target_counts", passed=counts["passed"], detail=counts))
        payload = await b.evaluate("""(()=>{
          const c=backpressureReport.campaign,d=document.getElementById('data');
          return {encoding:d.dataset.encoding,history_rows:c.results.reduce((n,r)=>n+(r.history?.length||0),0),
            passed:c.results.reduce((n,r)=>n+(r.history?.length||0),0)===Number(d.dataset.historyRows)};
        })()""")
        checks.append(dict(name="complete_embedded_histories", passed=payload["passed"], detail=payload))
        source = await b.evaluate("""(()=>{const c=backpressureReport.campaign,id=c.provenance?.source_sha256;
          return {id,passed:!id||c.results.every(r=>r.source_sha256===id)};})()""")
        checks.append(dict(name="result_source_identities", passed=source["passed"], detail=source))
        physical_inputs = await b.evaluate("""(async()=>{
          const c=backpressureReport.campaign,provenance=c.provenance||{},required=Boolean(provenance.snapshot);
          if(!required)return {passed:true,skipped_no_frozen_snapshot:true};
          const sourceMap=new Map();let complete=true;
          for(const physical of c.cases){const inputs=physical.physical_inputs,e=inputs?.effective;
            complete&&=inputs?.available===true&&Number.isFinite(e?.inlet?.p0)&&Number.isFinite(e?.inlet?.T0)&&Boolean(e?.geometry);
            if(physical.model==='euler_annular')complete&&=Number.isFinite(e?.inlet?.M_m)&&Number.isFinite(e?.inlet?.alpha_tangent??e?.inlet?.alpha_degrees);
            for(const source of inputs?.sources||[])sourceMap.set(source.path,source);
          }
          const checks=await Promise.all([...sourceMap.values()].map(async source=>{
            const bytes=new TextEncoder().encode(source.source_text);
            const digest=await crypto.subtle.digest('SHA-256',bytes);
            const actual=[...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('');
            return {path:source.path,passed:actual===source.sha256&&source.manifest_validated===true&&source.manifests.every(name=>provenance[name]?.[source.path]===actual)};
          }));
          return {passed:complete&&checks.length>0&&checks.every(x=>x.passed),cases:c.cases.length,sources:checks};
        })()""")
        checks.append(dict(name="embedded_physical_inputs_match_frozen_execution_bytes", passed=physical_inputs["passed"], detail=physical_inputs))
        independent_pairs = await b.evaluate("""(()=>{
          const r=backpressureReport,c=r.campaign,audit=c.saved_audits?.['sensitivity_audit.json'];
          if(!audit)return {passed:true,skipped_no_saved_numpy_audit:true};
          const byId=new Map(c.results.map(row=>[row.id,row])),close=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-12*Math.max(1,Math.abs(a),Math.abs(b));
          const matches=(audit.pairs||[]).map(pair=>{
            const actual=r.comparePair(byId.get(pair.a_id),byId.get(pair.b_id),{case_id:pair.case_id,cells:pair.cells,kind:pair.kind});
            return {a_id:pair.a_id,b_id:pair.b_id,kind:pair.kind,passed:actual.available===pair.available&&
              ['pressure_scale_Pa','max_delta_p_over_scale','max_delta_M_normal','relative_mean_face_flow_gap'].every(key=>actual[key]===null&&pair[key]===null||close(actual[key],pair[key]))};
          });
          return {passed:audit.passed===true&&audit.source_sha256===c.provenance.source_sha256&&matches.length===96&&matches.every(x=>x.passed),
            pairs:matches.length,initialization:matches.filter(p=>p.kind==='initialization').length,outlet:matches.filter(p=>p.kind==='outlet').length,failed:matches.filter(p=>!p.passed)};
        })()""")
        checks.append(dict(name="all_96_pairs_match_independent_numpy_audit", passed=independent_pairs["passed"], detail=independent_pairs))
        production_tests = await b.evaluate("""(()=>{
          const c=backpressureReport.campaign,e=c.saved_audits?.['test_evidence.json'];
          if(!e)return {passed:true,skipped_no_saved_test_evidence:true};
          return {passed:e.source_sha256===c.provenance.source_sha256&&e.total_passed===37&&e.runs.every(r=>r.exit_code===0)&&
            Object.entries(e.file_sha256).filter(([path])=>path.startsWith('src/')).every(([path,hash])=>c.provenance.source_files[path]===hash),
            total_passed:e.total_passed,source_sha256:e.source_sha256};
        })()""")
        checks.append(dict(name="focused_production_test_evidence_matches_execution_source", passed=production_tests["passed"], detail=production_tests))
        await b.evaluate("document.getElementById('theme').value='light';document.getElementById('theme').dispatchEvent(new Event('change'))")
        await b.settled()
        await b.screenshot(output / "overview_light.png")
        checks.append(dict(name="desktop_no_horizontal_overflow", passed=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        await b.evaluate("backpressureReport.switchPage('explorer')")
        await b.settled()
        case_ids = await b.evaluate("backpressureReport.campaign.cases.map(c=>c.id)")
        for case in case_ids:
            for grid in (50, 100, 200):
                await b.evaluate(f"document.getElementById('case').value={json.dumps(case)};document.getElementById('mesh').value='{grid}';backpressureReport.render()")
                snapshot = await b.settled()
                passed = await b.evaluate("backpressureReport.selectedRows().every(r=>r.case_id===document.getElementById('case').value&&r.cells===Number(document.getElementById('mesh').value))")
                checks.append(dict(name=f"case_mesh_{case}_{grid}", passed=bool(passed), detail=snapshot))
        for selector, values in dict(outlet=("acoustic", "finite_wave", "all"), start=("stagnant", "adapted", "all"),
                                     flux=("hll", "stadtke", "all"), reconstruction=("constant", "muscl_van_leer", "all")).items():
            for value in values:
                await b.evaluate(f"document.getElementById('{selector}').value={json.dumps(value)};backpressureReport.render()")
                await b.settled()
                passed = await b.evaluate(f"document.getElementById('{selector}').value==='all'||backpressureReport.selectedRows().every(r=>r.{selector}===document.getElementById('{selector}').value)")
                checks.append(dict(name=f"selector_{selector}_{value}", passed=bool(passed)))
        for case in case_ids:
            await b.evaluate(f"""(()=>{{
              document.getElementById('case').value={json.dumps(case)};document.getElementById('mesh').value='200';
              for(const id of ['outlet','start','flux','reconstruction'])document.getElementById(id).value='all';
              backpressureReport.switchView('fields');
            }})()""")
            await b.settled()
            actual = await b.evaluate("""(()=>{
              const c=backpressureReport.campaign,rows=backpressureReport.selectedRows(),hasData=rows.some(r=>r.fields?.x?.length);
              const pressure=document.getElementById('field-p').data||[],mach=document.getElementById('field-M').data||[];
              const numericalPressure=pressure.filter(t=>t.name!=='Continuous reference'),numericalMach=mach.filter(t=>t.name!=='Continuous reference');
              return {saved_rows:rows.length,pressure_curves:numericalPressure.length,normal_mach_curves:numericalMach.length,
                passed:!hasData||(numericalPressure.length===rows.length&&numericalMach.length===rows.length&&numericalMach.every(t=>t.y.every(Number.isFinite)))};
            })()""")
            checks.append(dict(name=f"all_selected_configurations_profiles_{case}", passed=actual["passed"], detail=actual))
            if case in ("supercritical_shock", "lettieri_equilibrium_shock"):
                await b.evaluate("window.scrollTo(0,document.getElementById('explorer').offsetTop)")
                await b.screenshot(output / f"all_profiles_{case}_light.png")
                await b.evaluate("window.scrollTo(0,0)")
        await b.evaluate("""(()=>{
          const rows=backpressureReport.campaign.results.filter(r=>r.fields?.x?.length);
          const row=rows.find(r=>r.case_id==='lettieri_equilibrium_shock'&&r.cells===200&&r.reconstruction==='muscl_van_leer')||rows.at(-1);
          if(row){document.getElementById('case').value=row.case_id;
            document.getElementById('mesh').value=String(row.cells);
            document.getElementById('reconstruction').value=row.reconstruction;}
          backpressureReport.render();
        })()""")
        await b.settled()
        visible_fields = await b.evaluate("""(()=>{
          const rows=backpressureReport.selectedRows(),available=rows.some(r=>r.fields?.x?.length);
          const pressure=document.getElementById('field-p').data||[],mach=document.getElementById('field-M').data||[];
          const actual=rows.find(r=>r.fields?.x?.length),series=actual?mach.find(t=>t.name.includes(actual.outlet==='finite_wave'?'Finite wave':'Acoustic')):null;
          return {available,pressure_traces:pressure.length,mach_traces:mach.length,
            passed:document.getElementById('comparison-key').textContent.includes('Analytical reference')&&
              (!available||(pressure.length>0&&mach.length>0&&series?.y?.every(Number.isFinite)))};
        })()""")
        checks.append(dict(name="actual_pressure_and_normal_mach_profiles", passed=visible_fields["passed"], detail=visible_fields))
        target_badges = await b.evaluate("""(()=>{
          const rows=backpressureReport.selectedRows(),text=document.getElementById('run-table').textContent;
          return {target_failed:rows.filter(r=>r.target_correct===false).length,
            root_verified:rows.filter(backpressureReport.verified).length,
            passed:rows.every(r=>r.target_correct!==false||text.includes(r.audit?.target?.resolution_limited?'Shock unresolved':'Target check failed'))};
        })()""")
        checks.append(dict(name="physical_target_badges_independent_of_root_verification", passed=target_badges["passed"], detail=target_badges))
        await b.evaluate("""(()=>{
          window.auditSavedSelection=Object.fromEntries(['case','mesh','outlet','start','flux','reconstruction'].map(k=>[k,document.getElementById(k).value]));
          const rows=backpressureReport.campaign.results.filter(r=>backpressureReport.verified(r)&&r.fields?.x?.length);
          const row=rows.find(a=>rows.some(b=>a.id!==b.id&&a.case_id===b.case_id&&a.cells===b.cells&&a.flux===b.flux&&a.reconstruction===b.reconstruction&&
            (a.outlet===b.outlet&&a.start!==b.start||a.start===b.start&&a.outlet!==b.outlet)));
          window.auditHasVerifiedPair=Boolean(row);
          if(row){document.getElementById('case').value=row.case_id;document.getElementById('mesh').value=String(row.cells);
            document.getElementById('outlet').value='all';document.getElementById('start').value='all';
            document.getElementById('flux').value='all';document.getElementById('reconstruction').value=row.reconstruction;}
          backpressureReport.render();
        })()""")
        await b.settled()
        pair_metrics = await b.evaluate("""(()=>{
          const r=backpressureReport,rows=r.selectedRows(),pairs=r.sensitivityPairs(),close=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-12*Math.max(1,Math.abs(a),Math.abs(b));
          const available=pairs.filter(p=>p.available);
          const correct=available.every(p=>{
            const a=rows.find(row=>row.id===p.a_id),b=rows.find(row=>row.id===p.b_id);
            if(!a||!b||!r.verified(a)||!r.verified(b))return false;
            const pa=a.fields.p,pb=b.fields.p,ma=a.fields.M||a.fields.M_euler,mb=b.fields.M||b.fields.M_euler;
            const dp=Math.max(...pa.map((v,i)=>Math.abs(v-pb[i])))/a.errors.p.scale;
            const dm=Math.max(...ma.map((v,i)=>Math.abs(v-mb[i])));
            const fa=a.audit.conservation.face_mass_flow_mean,fb=b.audit.conservation.face_mass_flow_mean;
            const gap=Math.abs(fa-fb)/((Math.abs(fa)+Math.abs(fb))/2);
            return a.case_id===b.case_id&&a.cells===b.cells&&a.flux===b.flux&&a.reconstruction===b.reconstruction&&
              (p.kind==='initialization'?a.outlet===b.outlet&&a.start==='stagnant'&&b.start==='adapted':a.start===b.start&&a.outlet==='acoustic'&&b.outlet==='finite_wave')&&
              close(p.max_delta_p_over_scale,dp)&&close(p.max_delta_M_normal,dm)&&close(p.relative_mean_face_flow_gap,gap);
          });
          const unavailable=pairs.filter(p=>!p.available);
          const noZero=unavailable.every(p=>p.max_delta_p_over_scale===null&&p.max_delta_M_normal===null&&p.relative_mean_face_flow_gap===null);
          const first=rows.find(row=>r.verified(row)),negative=first?r.comparePair(first,{...first,success:false}):r.comparePair();
          return {pairs:pairs.length,available:available.length,available_by_kind:{initialization:available.filter(p=>p.kind==='initialization').length,outlet:available.filter(p=>p.kind==='outlet').length},unavailable:unavailable.length,
            passed:correct&&noZero&&(!window.auditHasVerifiedPair||available.length>0)&&negative.available===false&&negative.max_delta_p_over_scale===null};
        })()""")
        checks.append(dict(name="paired_sensitivity_calculations_from_saved_verified_roots", passed=pair_metrics["passed"], detail=pair_metrics))
        await b.evaluate("window.scrollTo(0,document.getElementById('sensitivity-table').closest('.table-wrap').offsetTop-70)")
        await b.screenshot(output / "paired_sensitivity_light.png")
        await b.evaluate("for(const [key,value]of Object.entries(auditSavedSelection))document.getElementById(key).value=value;backpressureReport.render()")
        await b.settled()
        await b.evaluate("window.scrollTo(0,document.getElementById('explorer').offsetTop)")
        await b.screenshot(output / "profiles_light.png")
        await b.evaluate("window.scrollTo(0,0)")
        for field in ("h0", "h", "Q", "v_theta", "s"):
            await b.evaluate(f"document.getElementById('extra-field').value='{field}';backpressureReport.render()")
            await b.settled()
            checks.append(dict(name=f"additional_field_{field}", passed=True))
        for mode in ("absolute", "relative", "ratio"):
            await b.evaluate(f"document.getElementById('residual-mode').value='{mode}';backpressureReport.switchView('convergence')")
            await b.settled()
            checks.append(dict(name=f"residual_mode_{mode}", passed=True))
        for scope in ("final", "all"):
            await b.evaluate(f"document.getElementById('attempt-mode').value='{scope}';backpressureReport.render()")
            await b.settled()
            passed = await b.evaluate("""(()=>{
              const a=backpressureReport.selectedRows(),all=document.getElementById('attempt-mode').value==='all';
              return backpressureReport.selectedHistoryRows().length===a.length+(all?a.reduce((n,r)=>n+(r.attempts?.length||0),0):0);
            })()""")
            checks.append(dict(name=f"attempt_scope_{scope}", passed=bool(passed)))
        await b.evaluate("window.scrollTo(0,document.getElementById('explorer').offsetTop)")
        await b.screenshot(output / "convergence_light.png")
        await b.evaluate("backpressureReport.switchView('sequence')")
        await b.settled()
        await b.screenshot(output / "grid_sequence_light.png")
        for page in ("theory", "evidence", "overview"):
            await b.evaluate(f"backpressureReport.switchPage('{page}');window.scrollTo(0,0)")
            await b.settled()
            passed = await b.evaluate(f"document.getElementById('{page}').classList.contains('active')")
            checks.append(dict(name=f"page_{page}", passed=bool(passed)))
            if page == "theory":
                math = await b.evaluate("(async()=>{await MathJax.startup.promise;return document.querySelectorAll('#theory mjx-container[jax=SVG]').length;})()")
                checks.append(dict(name="offline_math_typeset", passed=math > 0, equations=math))
                await b.evaluate("window.scrollTo(0,document.getElementById('theory').offsetTop)")
                await b.screenshot(output / "theory_light.png")
        await b.evaluate("document.getElementById('theme').value='dark';document.getElementById('theme').dispatchEvent(new Event('change'));window.scrollTo(0,0)")
        await b.settled()
        checks.append(dict(name="dark_theme", passed=await b.evaluate("document.documentElement.dataset.theme==='dark'")))
        await b.screenshot(output / "overview_dark.png")
        await b.evaluate("""(()=>{window.auditBlobs=[];const original=URL.createObjectURL;
          URL.createObjectURL=function(blob){auditBlobs.push(blob);return original.call(this,blob)};
          document.getElementById('download-data').click();document.getElementById('download-fields').click();
          document.getElementById('download-history').click();document.getElementById('download-theory').click();
          document.getElementById('download-sensitivity').click();
        })()""")
        downloads = await b.evaluate("(async()=>await Promise.all(auditBlobs.map(async b=>({bytes:b.size,type:b.type,head:(await b.text()).slice(0,120)}))))()")
        checks.append(dict(name="embedded_json_csv_and_theory_downloads", passed=len(downloads) == 5 and all(d["bytes"] > 0 for d in downloads), detail=downloads))
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=390, height=844, deviceScaleFactor=1, mobile=True))
        for page in ("overview", "explorer", "theory", "evidence"):
            await b.evaluate(f"backpressureReport.switchPage('{page}');window.dispatchEvent(new Event('resize'));window.scrollTo(0,0)")
            await b.settled()
            await asyncio.sleep(.15)
            checks.append(dict(name=f"mobile_no_horizontal_overflow_{page}", passed=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
            if page in ("overview", "theory"):
                await b.screenshot(output / f"{page}_mobile.png")
        external = sorted({url for url in b.requests if url.startswith(("http:", "https:"))})
        checks.append(dict(name="offline_no_external_requests", passed=not external, detail=external))
        checks.append(dict(name="no_javascript_exceptions", passed=not b.exceptions, detail=b.exceptions))
        return dict(passed=all(c["passed"] for c in checks), checks=checks,
                    html_sha256=sha256(html.read_bytes()).hexdigest(), input=str(html),
                    complete_configuration_matrix=counts["complete"],
                    javascript_exceptions=b.exceptions, failed_requests=b.failed_requests,
                    screenshots=[p.name for p in sorted(output.glob("*.png"))])


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
    command = [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", "--no-sandbox",
               "--disable-crash-reporter", "--disable-breakpad", "--disable-background-networking",
               "--disable-component-update", "--disable-sync", "--disable-extensions",
               "--no-default-browser-check", "--remote-allow-origins=*", f"--remote-debugging-port={port}",
               f"--user-data-dir={output / 'browser_profile'}", "about:blank"]
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
                time.sleep(.15)
            target = next((t for t in targets if t["type"] == "page"), None)
            if target is None:
                raise RuntimeError(f"Headless Edge did not expose a page; inspect {output / 'edge.log'}")
            result = asyncio.run(exercise(target["webSocketDebuggerUrl"], args.input, output))
            (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]),
                                 failed=[c["name"] for c in result["checks"] if not c["passed"]],
                                 output=str(output)), indent=2))
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
