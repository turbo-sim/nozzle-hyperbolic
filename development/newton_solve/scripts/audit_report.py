"""Exercise a copied-alone offline report in an isolated headless Edge session."""
from __future__ import annotations

import argparse
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

from development.real_gas.scripts.audit_report import Browser as BaseBrowser, EDGE
from .report import STUDY


class Browser(BaseBrowser):
    async def settled(self):
        return await self.evaluate("""(async()=>{
          const end=Date.now()+30000;
          while(Date.now()<end){
            if(window.newtonReportLoadError)throw new Error(window.newtonReportLoadError);
            if(window.newtonReport){const s=newtonReport.snapshot();if(s.rendered===s.renderNumber)return s;}
            await new Promise(r=>setTimeout(r,30));
          }
          throw new Error('Report render did not settle');
        })()""")


async def exercise(websocket_url, html, output):
    checks = []
    async with websockets.connect(websocket_url, max_size=100_000_000) as connection:
        b = Browser(connection)
        for domain in ("Page", "Runtime", "Network"):
            await b.command(f"{domain}.enable")
        await b.command("Network.setBlockedURLs", dict(urls=["http://*", "https://*"]))
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=1440, height=1080, deviceScaleFactor=1, mobile=False))
        await b.command("Browser.setDownloadBehavior", dict(behavior="deny"))
        await b.command("Page.navigate", dict(url=html.resolve().as_uri()))
        initial = await b.settled()
        checks.append(dict(name="initial_render", passed=bool(initial), detail=initial))
        checks.append(dict(name="offline_plotly_runtime", passed=await b.evaluate("typeof Plotly==='object'&&Plotly.version==='2.35.2'")))
        counts = await b.evaluate("""(()=>{
          const r=newtonReport,c=r.campaign,rows=c.results,expected=c.summary?.expected_runs??96;
          const keys=rows.map(x=>[x.case_id,x.flux,x.reconstruction,x.outlet,x.start,x.strategy,x.cells].join('|'));
          return {expected,saved:rows.length,complete:rows.length===expected,unique:new Set(keys).size===keys.length,
            native:rows.filter(r.native).length,roots:rows.filter(r.verified).length,target:rows.filter(r.target).length,
            matched_initial: r.pairs(rows).filter(p=>p.identical_initial_state).length,
            passed:keys.length===new Set(keys).size&&document.getElementById('kpi-native').textContent===`${rows.filter(r.native).length} / ${expected}`&&
              document.getElementById('kpi-root').textContent===`${rows.filter(r.verified).length} / ${expected}`&&
              document.getElementById('kpi-target').textContent===`${rows.filter(r.target).length} / ${expected}`};
        })()""")
        checks.append(dict(name="honest_coverage_and_independent_outcomes", passed=counts["passed"], detail=counts))
        history = await b.evaluate("""(()=>{
          const c=newtonReport.campaign,actual=c.results.reduce((n,r)=>n+(r.history?.length||0),0);
          return {actual,registered:Number(document.getElementById('data').dataset.historyRows),passed:actual===Number(document.getElementById('data').dataset.historyRows)};
        })()""")
        checks.append(dict(name="complete_embedded_histories", passed=history["passed"], detail=history))
        scalar_work = await b.evaluate("""(()=>{
          const rows=newtonReport.campaign.results;
          const checks=rows.map(r=>{const counts=r.raw_record_evidence?.embedded_scalar_record_counts;
            return {id:r.id,trials:r.trials?.length||0,passed:!counts||counts.trials===(r.trials?.length||0)&&
              counts.native_history===(r.direct?.native_history?.length||0)&&counts.evaluations===(r.direct?.evaluations?.length||0)&&
              counts.domain_failures===(r.direct?.domain_failures?.length||0)};});
          return {passed:checks.every(c=>c.passed),trials:checks.reduce((n,c)=>n+c.trials,0),failed:checks.filter(c=>!c.passed)};
        })()""")
        checks.append(dict(name="all_scalar_native_trial_evaluation_and_domain_records_embedded", passed=scalar_work["passed"], detail=scalar_work))
        source = await b.evaluate("""(()=>{
          const c=newtonReport.campaign,s=c.provenance?.source_sha256;
          return {source_sha256:s,passed:!s||c.results.every(r=>r.source_sha256===s)};
        })()""")
        checks.append(dict(name="all_results_match_frozen_source", passed=source["passed"], detail=source))
        fresh = await b.evaluate("""(()=>{
          const r=newtonReport,c=r.campaign,a=c.saved_audits?.['verify_saved.json'],expected=c.summary?.expected_runs??96;
          if(!a)return {passed:c.results.length<expected,missing:true,partial_preview:c.results.length<expected};
          const byId=new Map((a.records||[]).map(x=>[x.id,x])),compared=c.results.map(row=>{
            const f=byId.get(row.id);return {id:row.id,passed:Boolean(f&&f.passed&&f.saved_verified===r.verified(row)&&
              f.fresh_verified===r.verified(row)&&f.saved_target_correct===row.target_correct&&f.fresh_target_correct===row.target_correct&&
              (!row.raw_record_evidence?.sha256||f.saved_file_sha256===row.raw_record_evidence.sha256))};
          });
          return {passed:a.passed===true&&a.expected_baseline_runs===expected&&!(a.missing_baseline_runs||[]).length&&
            a.source_sha256===c.provenance.source_sha256&&a.client_sha256===c.provenance.client_sha256&&
            a.settings_sha256===c.provenance.settings_sha256&&a.execution?.fresh_process===true&&
            a.execution?.physical_time_steps===0&&a.execution?.nonlinear_solves===0&&compared.every(x=>x.passed),
            audited_records:a.verified_saved_records,matched_report_records:compared.length,
            numerical_roots:c.results.filter(r.verified).length,failed:compared.filter(x=>!x.passed)};
        })()""")
        checks.append(dict(name="primary_fresh_saved_state_audit_matches_all_embedded_outcomes_and_raw_hashes", passed=fresh["passed"], detail=fresh))
        starts = await b.evaluate("""(()=>{
          const rows=newtonReport.campaign.results.filter(r=>r.direct?.initial_state_sha256);
          const checks=rows.map(r=>({id:r.id,passed:r.report_initial_state_bytes_sha256===r.direct.initial_state_sha256&&
            r.report_initial_state_bytes_sha256===r.direct.actual_initial_state_sha256}));
          return {checked:checks.length,passed:checks.every(c=>c.passed),failed:checks.filter(c=>!c.passed)};
        })()""")
        checks.append(dict(name="independent_saved_start_bytes_match_actual_direct_first_evaluation", passed=starts["passed"], detail=starts))
        inputs = await b.evaluate("""(async()=>{
          const a=newtonReport.campaign.frozen_physical_inputs,sources=a?.sources||[],rows=[];
          for(const s of sources){const bytes=new TextEncoder().encode(s.source_text),hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');rows.push({path:s.path,passed:hash===s.sha256});}
          return {available:a?.available,sources:rows,passed:!a?.available||rows.length>=2&&rows.every(r=>r.passed)};
        })()""")
        checks.append(dict(name="exact_frozen_case_yaml_bytes_and_hashes", passed=inputs["passed"], detail=inputs))
        await b.screenshot(output / "overview_light.png")
        if await b.evaluate("Boolean(document.querySelector('.measured-findings'))"):
            await b.evaluate("window.scrollTo(0,document.querySelector('.measured-findings').offsetTop-20)")
            await b.screenshot(output / "measured_findings_light.png")
            checks.append(dict(name="measured_findings_embedded_on_overview", passed=await b.evaluate("document.querySelector('.measured-findings').textContent.includes('96')")))
            await b.evaluate("window.scrollTo(0,0)")
        checks.append(dict(name="desktop_no_horizontal_overflow", passed=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
        await b.evaluate("newtonReport.switchPage('explorer')")
        await b.settled()
        for selector in ("case", "flux", "reconstruction", "outlet", "start", "strategy"):
            options = await b.evaluate(f"[...document.getElementById('{selector}').options].map(o=>o.value)")
            for value in options:
                await b.evaluate(f"document.getElementById('{selector}').value={json.dumps(value)};newtonReport.render()")
                await b.settled()
                valid = await b.evaluate(f"document.getElementById('{selector}').value==='all'||newtonReport.selectedRows().every(r=>r.{('case_id' if selector == 'case' else selector)}===document.getElementById('{selector}').value)")
                checks.append(dict(name=f"selector_{selector}_{value}", passed=bool(valid)))
        await b.evaluate("if(newtonReport.campaign.results.length)document.getElementById('case').value=newtonReport.campaign.results[0].case_id;for(const id of ['flux','reconstruction','outlet','start','strategy'])document.getElementById(id).value='all';newtonReport.render()")
        await b.settled()
        profiles = await b.evaluate("""(()=>{
          const rows=newtonReport.selectedRows(),traces=document.getElementById('field-p').data||[],available=rows.filter(r=>r.fields?.p?.length).length;
          return {selected:rows.length,available,numerical:traces.filter(t=>t.name!=='Analytical reference').length,
            passed:traces.filter(t=>t.name!=='Analytical reference').length===available};
        })()""")
        checks.append(dict(name="every_selected_final_iterate_in_profiles", passed=profiles["passed"], detail=profiles))
        checks.append(dict(name="analytical_reference_key", passed=await b.evaluate("document.querySelector('.comparison-key').textContent.includes('Analytical reference')")))
        comparison = await b.evaluate("""(()=>{
          const r=newtonReport,pairs=r.pairs(),good=pairs.filter(p=>p.available),bad=pairs.filter(p=>!p.available);
          const numeric=['max_delta_p_over_scale','max_delta_M','relative_mean_face_flow_gap'];
          const negative=r.comparison();
          return {pairs:pairs.length,available:good.length,unavailable:bad.length,
            passed:good.every(p=>p.identical_initial_state&&Number.isFinite(p.max_delta_p_over_scale)&&Number.isFinite(p.max_delta_M))&&
              bad.every(p=>numeric.every(k=>p[k]===null))&&negative.available===false&&numeric.every(k=>negative[k]===null)};
        })()""")
        checks.append(dict(name="paired_root_metrics_require_independent_roots_and_identical_starts", passed=comparison["passed"], detail=comparison))
        independence = await b.evaluate("""(()=>{
          const r=newtonReport;
          return r.verified({verified:true,native_success:false,audit:{passed:true}})&&
            !r.verified({verified:false,native_success:true,audit:{passed:false}})&&
            !r.target({verified:true,target_correct:false});
        })()""")
        checks.append(dict(name="native_root_and_target_are_independent", passed=bool(independence)))
        limit = await b.evaluate("""(()=>{
          const c=newtonReport.campaign,cap=c.settings?.direct?.max_accepted_steps;
          const a=c.results.filter(r=>r.strategy==='direct_newton').map(r=>({id:r.id,accepted:(r.history||[]).filter(h=>h.phase==='accepted').length}));
          return {declared_cap:cap,checked:a.length,passed:!Number.isFinite(cap)||a.every(r=>r.accepted<=cap),failed:a.filter(r=>Number.isFinite(cap)&&r.accepted>cap)};
        })()""")
        checks.append(dict(name="accepted_newton_update_budget_and_trial_work_are_separate", passed=limit["passed"], detail=limit))
        case_ids = await b.evaluate("newtonReport.campaign.cases.map(c=>c.id)")
        for case_id in case_ids:
            await b.evaluate(f"document.getElementById('case').value={json.dumps(case_id)};newtonReport.render()")
            await b.settled()
            actual = await b.evaluate("""(()=>{
              const rows=newtonReport.selectedRows(),pressure=document.getElementById('field-p').data||[],mach=document.getElementById('field-M').data||[];
              const available=rows.filter(r=>r.fields?.p?.length);
              const good=available.every(r=>{
                const p=pressure.find(t=>t.meta?.run_id===r.id),m=mach.find(t=>t.meta?.run_id===r.id),expectedM=r.fields.M||r.fields.M_euler||[];
                return p?.y?.length===r.fields.p.length&&p.y.every((v,i)=>v===(Number.isFinite(r.fields.p[i])?r.fields.p[i]*1e-3:null))&&
                  (!expectedM.length||m?.y?.length===expectedM.length&&m.y.every((v,i)=>v===expectedM[i]));
              });
              return {selected:rows.length,available:available.length,failed_final_iterates:available.filter(r=>!newtonReport.verified(r)).length,
                passed:good&&pressure.filter(t=>t.meta?.run_id).length===available.length};
            })()""")
            checks.append(dict(name=f"all_actual_pressure_mach_and_failed_final_iterates_{case_id}", passed=actual["passed"], detail=actual))
            await b.evaluate("document.getElementById('residual-mode').value='absolute';document.getElementById('residual-axis').value='linear';document.getElementById('equation').value='all';newtonReport.switchView('convergence')")
            await b.settled()
            convergence = await b.evaluate("""(()=>{
              const report=newtonReport,rows=report.selectedRows(),data=document.getElementById('residual-equations').data||[],names=['Mass','Momentum','Energy'];let samples=0,zeros=0;
              const good=rows.every(r=>names.every((name,i)=>{
                const expected=report.history(r).map(h=>(h.equation_norm||h.equation_l2||[])[i]).map(v=>Number.isFinite(v)?v:null);
                samples+=expected.length;zeros+=expected.filter(v=>v===0).length;
                const trace=data.find(t=>t.meta?.run_id===r.id&&t.name.endsWith(' · '+name));
                return trace?.customdata?.length===expected.length&&trace.customdata.every((v,i)=>v===expected[i])&&trace.y.every((v,i)=>v===expected[i]);
              }));
              return {runs:rows.length,samples,exact_zeros:zeros,passed:good};
            })()""")
            checks.append(dict(name=f"all_accepted_equation_samples_and_exact_zeros_{case_id}", passed=convergence["passed"], detail=convergence))
            await b.evaluate("newtonReport.switchView('fields')")
            await b.settled()
        await b.evaluate("""(()=>{
          const rows=newtonReport.campaign.results,r=rows.find(r=>r.flux==='stadtke'&&r.reconstruction==='muscl_van_leer')||rows[0];
          if(r){for(const [id,value]of Object.entries({case:r.case_id,flux:r.flux,reconstruction:r.reconstruction,outlet:r.outlet}))document.getElementById(id).value=value;}
          newtonReport.render();window.scrollTo(0,document.getElementById('explorer').offsetTop);
        })()""")
        await b.settled()
        await b.screenshot(output / "profiles_light.png")
        offered_fields = await b.evaluate("[...document.getElementById('extra-field').options].map(o=>o.value)")
        for field in offered_fields:
            await b.evaluate(f"document.getElementById('extra-field').value='{field}';newtonReport.render()")
            await b.settled()
            actual = await b.evaluate(r"""(()=>{
              const el=document.getElementById('extra-field'),key=el.value,rows=newtonReport.selectedRows(),g=document.getElementById('field-extra'),data=g.data||[];
              const offeredUnits=el.options[el.selectedIndex].textContent.match(/\[([^\]]+)\]/)?.[1],axis=g.layout?.yaxis?.title?.text||'';
              const available=rows.filter(r=>Array.isArray(r.fields?.[key])&&r.fields[key].length),numeric=data.filter(t=>t.meta?.run_id);
              return {key,selected:rows.length,available:available.length,numerical_curves:numeric.length,offered_units:offeredUnits,axis,
                passed:(!rows.length||available.length===rows.length&&numeric.length===available.length)&&axis.endsWith('['+offeredUnits+']')};
            })()""")
            checks.append(dict(name=f"additional_field_{field}_actual_numerical_curves_and_display_units", passed=actual["passed"], detail=actual))
        for mode in ("absolute", "relative", "ratio"):
            for axis in ("log", "linear"):
                await b.evaluate(f"document.getElementById('residual-mode').value='{mode}';document.getElementById('residual-axis').value='{axis}';newtonReport.switchView('convergence')")
                await b.settled()
                checks.append(dict(name=f"residual_{mode}_{axis}", passed=True))
        await b.evaluate("document.getElementById('residual-mode').value='absolute';document.getElementById('residual-axis').value='log';newtonReport.render()")
        await b.settled()
        await b.screenshot(output / "convergence_light.png")
        await b.evaluate("newtonReport.switchView('work')")
        await b.settled()
        focuses = await b.evaluate("[...document.getElementById('run-focus').options].map(o=>o.value)")
        for value in focuses:
            await b.evaluate(f"document.getElementById('run-focus').value={json.dumps(value)};newtonReport.render()")
            await b.settled()
            valid = await b.evaluate("document.getElementById('trial-json').textContent.includes(document.getElementById('run-focus').value)")
            checks.append(dict(name=f"actual_work_record_{value}", passed=bool(valid)))
        await b.screenshot(output / "work_light.png")
        direct_work = await b.evaluate("""(()=>{
          const r=newtonReport.selectedRows().find(r=>r.strategy==='direct_newton');
          if(!r)return null;document.getElementById('run-focus').value=r.id;newtonReport.render();return r.id;
        })()""")
        if direct_work:
            await b.settled()
            detail = await b.evaluate("""(()=>{
              const r=newtonReport.selectedRows().find(r=>r.id===document.getElementById('run-focus').value),data=document.getElementById('step-plot').data||[];
              const trials=newtonReport.trialRecords(r).filter(t=>Number.isFinite(t.step_fraction??t.fraction??t.alpha??t.step_length));
              const plotted=data.find(t=>t.name==='Actual line-search trial');
              return {id:r.id,saved_trials:trials.length,plotted_trials:plotted?.y?.length||0,
                passed:trials.length===(plotted?.y?.length||0)&&trials.every((t,i)=>(t.step_fraction??t.fraction??t.alpha??t.step_length)===plotted.y[i])};
            })()""")
            checks.append(dict(name="every_actual_newton_trial_fraction_plotted", passed=detail["passed"], detail=detail))
            await b.screenshot(output / "direct_work_light.png")
        failures = await b.evaluate("""(()=>{
          const rows=newtonReport.selectedRows().filter(r=>!newtonReport.verified(r)),text=document.getElementById('run-table').textContent;
          return {failures:rows.length,passed:rows.every(r=>text.includes(r.reason||r.native_status||'Unavailable'))};
        })()""")
        checks.append(dict(name="failed_terminations_remain_visible", passed=failures["passed"], detail=failures))
        for page in ("theory", "evidence", "overview"):
            await b.evaluate(f"newtonReport.switchPage('{page}');window.scrollTo(0,0)")
            await b.settled()
            checks.append(dict(name=f"page_{page}", passed=await b.evaluate(f"document.getElementById('{page}').classList.contains('active')")))
            if page == "theory":
                math = await b.evaluate("(async()=>{await MathJax.startup.promise;return document.querySelectorAll('#theory mjx-container[jax=SVG]').length})()")
                checks.append(dict(name="offline_equations_typeset", passed=math > 0, equations=math))
                await b.screenshot(output / "theory_light.png")
        await b.evaluate("document.getElementById('theme').value='dark';document.getElementById('theme').dispatchEvent(new Event('change'))")
        await b.settled()
        checks.append(dict(name="dark_theme", passed=await b.evaluate("document.documentElement.dataset.theme==='dark'")))
        await b.screenshot(output / "overview_dark.png")
        await b.evaluate("""(()=>{window.auditBlobs=[];const original=URL.createObjectURL;
          URL.createObjectURL=function(blob){auditBlobs.push(blob);return original.call(this,blob)};
          for(const id of ['download-data','download-fields','download-history','download-trials','download-theory'])document.getElementById(id).click();
        })()""")
        downloads = await b.evaluate("(async()=>await Promise.all(auditBlobs.map(async b=>({bytes:b.size,type:b.type,head:await b.slice(0,80).text(),magic:[...new Uint8Array(await b.slice(0,4).arrayBuffer())]}))))()")
        gzip_data = await b.evaluate("document.getElementById('data').dataset.encoding==='gzip-base64'")
        checks.append(dict(name="embedded_json_csv_theory_and_trial_downloads", passed=len(downloads) == 5 and all(d["bytes"] > 0 for d in downloads)
                           and (not gzip_data or downloads[0]["type"] == "application/gzip" and downloads[0]["magic"][:2] == [31, 139]), detail=downloads))
        await b.command("Emulation.setDeviceMetricsOverride", dict(width=390, height=844, deviceScaleFactor=1, mobile=True))
        for page in ("overview", "explorer", "theory", "evidence"):
            await b.evaluate(f"newtonReport.switchPage('{page}');window.dispatchEvent(new Event('resize'));window.scrollTo(0,0)")
            await b.settled()
            await asyncio.sleep(.15)
            checks.append(dict(name=f"mobile_no_horizontal_overflow_{page}", passed=await b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")))
            if page == "overview":
                await b.screenshot(output / "overview_mobile.png")
        external = sorted({u for u in b.requests if u.startswith(("http:", "https:"))})
        checks.append(dict(name="copied_alone_no_external_requests", passed=not external, detail=external))
        checks.append(dict(name="no_javascript_exceptions", passed=not b.exceptions, detail=b.exceptions))
        return dict(passed=all(c["passed"] for c in checks), checks=checks, html_sha256=sha256(html.read_bytes()).hexdigest(),
                    input=str(html), complete_configuration_matrix=counts["complete"], failed_requests=b.failed_requests,
                    javascript_exceptions=b.exceptions, screenshots=[p.name for p in sorted(output.glob("*.png"))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "REPORT.html")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/browser_audit")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    isolated = output / "isolated"
    isolated.mkdir(exist_ok=True)
    copied = isolated / "REPORT.html"
    shutil.copyfile(args.input, copied)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    profile = output / "browser_profile"
    command = [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", "--no-sandbox", "--disable-crash-reporter",
               "--disable-breakpad", "--disable-background-networking", "--disable-component-update", "--disable-sync",
               "--disable-extensions", "--no-default-browser-check", "--remote-allow-origins=*",
               f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "about:blank"]
    with (output / "edge.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            targets, deadline = [], time.monotonic()+15
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
                raise RuntimeError(f"Headless Edge did not expose a page; inspect {output / 'edge.log'}")
            result = asyncio.run(exercise(target["webSocketDebuggerUrl"], copied, output))
            (output / "audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), failed=[c["name"] for c in result["checks"] if not c["passed"]], output=str(output)), indent=2))
            if not result["passed"]:
                raise RuntimeError("Browser report audit failed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            # Only isolated artifacts created in this task's output folder.
            for target in (profile, isolated):
                resolved = target.resolve()
                if not resolved.is_relative_to(output) or resolved == output:
                    raise ValueError("Refusing cleanup outside browser audit output")
                if resolved.exists():
                    shutil.rmtree(resolved, ignore_errors=True)


if __name__ == "__main__":
    main()
