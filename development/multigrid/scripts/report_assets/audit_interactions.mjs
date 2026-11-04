/** Deterministic DOM/data audit; browser layout is a separate verification. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Blob, Buffer} from 'node:buffer';
import {DecompressionStream} from 'node:stream/web';
import {makeDOM, AuditEvent} from '../../../flux_methods/shared/report_dom.mjs';

const hash = value => createHash('sha256').update(value).digest('hex');

export async function auditReport(htmlPath, outputPath = null, {initialHash = ''} = {}) {
  const html = await fs.readFile(htmlPath, 'utf8');
  const payload = html.match(/<script[^>]*id="report-data"[^>]*>([\s\S]*?)<\/script>/)[1];
  const expected = JSON.parse(gunzipSync(Buffer.from(payload, 'base64')));
  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
  const app = scripts.at(-1)[1];
  new vm.Script(app);
  const document = makeDOM(html);
  const data = document.createElement('script');
  data.id = 'report-data'; data.textContent = payload; document.body.append(data);
  const plots = new Map(), errors = [], downloads = new Map(), storage = new Map();
  let readyResolve;
  const ready = new Promise(resolve => { readyResolve = resolve; });
  class AuditResponse {
    constructor(stream){this.stream=stream;}
    async text(){const reader=this.stream.getReader(),chunks=[];for(;;){const part=await reader.read();if(part.done)break;chunks.push(Buffer.from(part.value));}return Buffer.concat(chunks).toString('utf8');}
  }
  const window = {
    document, Blob, Response:AuditResponse, DecompressionStream, atob:value=>Buffer.from(value,'base64').toString('binary'), Event: AuditEvent,
    console: {log(){}, warn(){}, error: error => errors.push(String(error))},
    location: {hash: initialHash}, history: {replaceState(_state, _title, hashValue){window.location.hash=hashValue;}},
    setTimeout(){return 1;}, clearTimeout(){},
    localStorage: {getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)},
    matchMedia:()=>({matches:false,addEventListener(){}}),
    getComputedStyle:()=>({getPropertyValue:key=>({'--paper':'#fff','--ink':'#222','--muted':'#777','--line':'#ddd','--reference':'#888'}[key]||'')}),
    MathJax: {typesetPromise:()=>Promise.resolve()},
    URL: {createObjectURL(blob){const key='blob:audit-'+downloads.size;downloads.set(key,blob);return key;},revokeObjectURL(){}},
    dispatchEvent(event){if(event.type==='multigrid-report-ready')readyResolve();},
  };
  window.Plotly = {
    react(target,traces,layout,config){
      for(const trace of traces) if(Array.isArray(trace.x)&&Array.isArray(trace.y)) assert.equal(trace.x.length,trace.y.length,'paired arrays: '+target.id);
      plots.set(target.id,{traces,layout,config});target.classList.add('js-plotly-plot');return Promise.resolve();
    },
    purge(target){plots.delete(target.id);target.classList.remove('js-plotly-plot');},
  };
  window.window=window;window.globalThis=window;
  const context=vm.createContext(window);
  new vm.Script(app,{filename:'multigrid-report-app.js'}).runInContext(context,{timeout:10000});
  let timeout;
  await Promise.race([ready,new Promise((_,reject)=>{timeout=setTimeout(()=>reject(new Error('Report initialization timeout: '+errors.join('; '))),10000);})]);
  clearTimeout(timeout);
  const api=window.MultigridReport;
  assert.ok(api,'application initialized');
  const methods=expected.groups.flatMap(group=>group.methods.map(method=>({group,method})));
  const passed=methods.filter(({method})=>api.original(method.representative||method.warmup||{})===true).length;
  assert.equal(document.getElementById('hero-count').textContent,`${passed} / ${methods.length}`,'hero count initializes independently of active page');
  assert.ok(!document.getElementById('hero-context').textContent.includes('Loading'),'hero loading text is cleared on deep links');
  const initialPage=['overview','explorer','theory','protocol'].includes(initialHash.slice(1))?initialHash.slice(1):'overview';
  assert.ok(document.getElementById(initialPage).classList.contains('active'),'requested initial page is active');
  const complete=methods.filter(({method})=>method.samples?.length===method.requested_repeats&&method.samples.length>0);
  const eligible=complete.filter(({method})=>method.timing_eligible!==false&&method.samples.every(record=>record.success!==false&&api.original(record)===true));
  assert.equal(document.getElementById('kpi-eligible').textContent,`${eligible.length} / ${complete.length}`,'eligible complete sets are distinct from native convergence');
  const constant30=expected.groups.find(group=>group.case_id.endsWith('/shock_30')&&group.reconstruction==='constant');
  if(constant30?.methods.some(method=>method.samples?.length&&method.representative?.assessment?.analytical_passed===false))assert.ok(document.getElementById('findings').textContent.includes('30% shock: retained accuracy miss'),'constant shock30 accuracy limitation is explicit');
  const before=hash(JSON.stringify(api.state.data));
  const checked=['initial-page:'+initialPage,'initialized-summary'];
  for(const page of ['overview','explorer','theory','protocol']){
    api.navigate(page);assert.equal(document.querySelectorAll('.page.active').length,1);
    assert.ok(document.getElementById(page).classList.contains('active'));checked.push('page:'+page);
  }
  if(expected.tuning?.records?.length){
    const pilots=expected.tuning.records.filter(record=>record.kind!=='matched_cfl_diagnostic');
    const diagnostics=expected.tuning.records.filter(record=>record.kind==='matched_cfl_diagnostic');
    assert.equal(document.querySelectorAll('#tuning-table tr').length,pilots.length,'every tuning pilot is retained separately');
    assert.equal(document.querySelectorAll('#matched-table tr').length,diagnostics.length,'every matched-controller diagnostic is separate from pilot and primary samples');
    assert.equal(JSON.stringify(JSON.parse(document.getElementById('tuning-record').textContent)),JSON.stringify(expected.tuning),'raw tuning values remain unchanged');
    checked.push('separate-tuning-records');
  }
  const startup=expected.startup_probes?.records||[];
  if(startup.length){
    api.navigate('overview');
    assert.equal(document.querySelectorAll('#startup-table tr').length,startup.length,'all startup probes retained');
    for(const phase of ['empty_cache','populated_cache']){
      const records=startup.filter(record=>record.phase===phase);
      if(!records.length)continue;
      const label=phase==='empty_cache'?'Empty executable cache':'Populated executable cache';
      const process=plots.get('startup-process-chart').traces.find(trace=>trace.name===label);
      assert.ok(process,'startup process trace exists');
      assert.equal(JSON.stringify(process.y),JSON.stringify(records.map(record=>record.process_wall_s)),'startup process durations are unchanged');
      const compile=plots.get('startup-compile-chart').traces.find(trace=>trace.name===label);
      const durations=records.map(record=>{
        const contexts=[record.shared_compilation,record.setup_compilation,record.first_solve?.compilation].filter(value=>value&&typeof value==='object').map(value=>value.stats||value);
        const values=contexts.map(value=>value.compile_or_load_s).filter(value=>typeof value==='number'&&Number.isFinite(value));
        return values.length?values.reduce((sum,value)=>sum+value,0):null;
      });
      if(durations.some(value=>value!==null)){
        assert.ok(compile,'startup compile/load trace exists');
        assert.equal(JSON.stringify(compile.y),JSON.stringify(durations),'backend event contexts sum without adding process wall time');
      }
      checked.push('startup:'+phase);
    }
    if(startup.some(record=>record.first_solve?.success===false))assert.ok(document.getElementById('startup-table').textContent.includes('Failed'),'failed startup outcomes are visible');
  }
  api.navigate('explorer');
  for(const group of expected.groups){
    document.getElementById('physical-case').value=group.case_id;
    document.getElementById('reconstruction').value=group.reconstruction;
    api.renderExplorer();
    assert.equal(api.currentGroup().case_id,group.case_id);
    assert.equal(api.currentGroup().reconstruction,group.reconstruction);
    const outcomes=document.querySelectorAll('#case-table tr');
    group.methods.forEach((method,index)=>{
      const levels=(method.representative||method.warmup||{}).work?.levels;
      if(Array.isArray(levels)&&levels.length&&levels.every(row=>Number.isFinite(row.rejected_steps))){
        const expected=levels.reduce((sum,row)=>sum+row.rejected_steps,0);
        const cells=outcomes[index].querySelectorAll('td');
        assert.equal(Number(cells[cells.length-1].textContent.replaceAll(',','')),expected,'outcome ledger retains rejected implicit steps on all grids');
      }
    });
    const numeric=group.methods.find(method=>api.fields(api.state.data.groups.find(g=>g.case_id===group.case_id&&g.reconstruction===group.reconstruction).methods.find(m=>m.method_id===method.method_id).representative||method.warmup).p?.length);
    if(numeric){
      const actual=api.state.data.groups.find(g=>g.case_id===group.case_id&&g.reconstruction===group.reconstruction).methods.find(m=>m.method_id===numeric.method_id);
      const rep=actual.representative||actual.warmup;
      const trace=plots.get('profile-p')?.traces.find(trace=>trace.name.startsWith(actual.label||({sequence:'Grid sequencing',fas_fmg:'FAS with coarse-to-fine initialization'}[actual.method_id]||actual.method_id)));
      assert.ok(trace,'numerical pressure trace exists');
      assert.equal(JSON.stringify(trace.y),JSON.stringify(api.fields(rep).p),'saved pressure values are unchanged');
    }
    for(const mode of ['absolute','relative','boundary','global']){
      document.getElementById('residual-mode').value=mode;api.renderExplorer();
      const residual=plots.get('residual-chart');
      if(residual)for(const trace of residual.traces)assert.ok(trace.y.every(value=>value>0),'log traces contain positive recorded defects');
    }
    const sequence=api.state.data.groups.find(g=>g.case_id===group.case_id&&g.reconstruction===group.reconstruction).methods.find(method=>method.method_id==='sequence');
    if(sequence){
      const record=sequence.representative||sequence.warmup;
      const stages=api.sequenceStages(record);
      if(stages.length){
        document.getElementById('residual-mode').value='absolute';
        document.getElementById('history-scope').value='all';api.renderExplorer();
        const traces=plots.get('residual-chart').traces;
        let cumulative=0;
        for(const stage of stages){
          const source=record.stage_history?.length?record.stage_history.filter(row=>row.cells===stage.cells):(record.stages||[]).find(row=>row.cells===stage.cells)?.history||[];
          const count=source.filter(row=>row.phase==='accepted'||(row.accepted===true&&row.phase!=='initial'&&row.phase!=='rejected'&&row.step>0)).length;
          assert.equal(stage.accepted_steps,count,'initial observations are not counted as accepted steps');
          const trace=traces.find(trace=>trace.name.includes(`N=${stage.cells} (${count} steps)`));
          assert.ok(trace,'every sequencing grid has a labelled convergence trace');
          const positive=source.filter(row=>row.phase!=='rejected'&&row.accepted!==false&&Math.max(...row.equation_norm)>0);
          assert.equal(JSON.stringify(trace.y),JSON.stringify(positive.map(row=>Math.max(...row.equation_norm))),'all-grid residual values remain unchanged');
          assert.equal(trace.x[0],cumulative,'stage starts at previous cumulative accepted count');
          cumulative+=count;
          assert.equal(trace.x.at(-1),cumulative,'stage ends at cumulative accepted count');
        }
        assert.equal(document.querySelectorAll('#sequence-stage-table tr').length,stages.length,'every grid appears in stage count ledger');
        document.getElementById('history-scope').value='fine';api.renderExplorer();
        const finest=plots.get('residual-chart').traces.filter(trace=>trace.name.includes('N='));
        assert.equal(finest.length,1,'optional finest-only view is explicit');
        assert.equal(finest[0].x[0],0,'finest-only view uses local accepted steps');
        document.getElementById('history-scope').value='all';
        checked.push('all-grid-steps:'+group.case_id+'/'+group.reconstruction);
      }
    }
    document.getElementById('residual-axis').value='time';api.renderExplorer();
    if(sequence?.representative?.work?.stage_progress?.length){
      const progress=sequence.representative.work.stage_progress;
      const traces=plots.get('residual-chart')?.traces.filter(trace=>trace.name.includes('stage end'))||[];
      for(const trace of traces){
        assert.equal(trace.mode,'markers','sequence elapsed view does not imply per-iterate timing');
        assert.ok(progress.some(row=>row.elapsed_s===trace.x[0]),'wall-time point is a recorded stage completion');
      }
    }
    document.getElementById('residual-axis').value='work';
    const initialization=group.methods.flatMap(method=>api.initializationSummary(method.representative||method.warmup||{}));
    if(initialization.length){assert.equal(document.querySelectorAll('#initialization-table tr').length,initialization.length,'all recorded initialization stages remain visible');assert.ok(document.getElementById('history-caption').textContent.includes('before fine cycle 1'),'initialization work is explicit before fine cycles');}
    for(const key of api.workMetrics(group.methods.flatMap(method=>api.workRows(method.representative||method.warmup||{})))){
      document.getElementById('work-metric').value=key;api.renderExplorer();
      for(const trace of plots.get('work-chart')?.traces||[])assert.ok(trace.y.every(value=>value===null||Number.isFinite(value)),'counter plots preserve missing observations');
    }
    checked.push('group:'+group.case_id+'/'+group.reconstruction);
  }
  const input=document.querySelector('#methods input');
  const firstId=input.value;
  input.click();assert.ok(!api.state.enabled.has(firstId));
  input.click();assert.ok(api.state.enabled.has(firstId));
  checked.push('method-toggle');
  for(const theme of ['light','dark','auto']){api.setTheme(theme);assert.equal(api.state.theme,theme);checked.push('theme:'+theme);}
  for(const id of ['download-data','download-csv','download-fields','download-history'])document.getElementById(id).click();
  assert.equal(downloads.size,4,'four embedded downloads work');
  const downloaded=JSON.parse(await [...downloads.values()][0].text());
  assert.equal(JSON.stringify(downloaded),JSON.stringify(api.state.data));
  const csv=await [...downloads.values()][1].text();
  assert.equal(csv.split('\r\n').length,1+expected.groups.flatMap(g=>g.methods).reduce((sum,m)=>sum+(m.samples||[]).length,0));
  const histories=JSON.parse(await [...downloads.values()][3].text());
  for(const method of histories.methods){
    const saved=api.currentGroup().methods.find(row=>row.method_id===method.method_id);
    assert.equal(JSON.stringify(method.stage_history),JSON.stringify((saved.representative||saved.warmup).stage_history||[]),'downloads include every saved grid history');
  }
  assert.equal(before,hash(JSON.stringify(api.state.data)),'interactions do not mutate campaign values');
  assert.equal(errors.length,0,'no application console errors');
  const result={passed:true,html_sha256:hash(html),initial_hash:initialHash,groups:expected.groups.length,startup_probes:startup.length,eligible_complete_sets:eligible.length,total_complete_sets:complete.length,interactions:checked.length,checks:checked,downloads:downloads.size,paired_plot_arrays:true,saved_pressure_values_unchanged:true,startup_durations_unchanged:true,embedded_data_unchanged:true,browser_verified:false,note:'Instrumented DOM and Plotly interface audit; this does not verify browser layout or real Plotly rendering.'};
  if(outputPath){await fs.mkdir(path.dirname(outputPath),{recursive:true});await fs.writeFile(outputPath,JSON.stringify(result,null,2)+'\n');}
  return result;
}

if(globalThis.process?.argv[1]&&path.resolve(globalThis.process.argv[1])===fileURLToPath(import.meta.url)){
  const html=process.argv[2];
  if(!html)throw new Error('Usage: node audit_interactions.mjs REPORT.html [audit.json] [initialHash]');
  console.log(JSON.stringify(await auditReport(html,process.argv[3]||null,{initialHash:process.argv[4]||''}),null,2));
}
