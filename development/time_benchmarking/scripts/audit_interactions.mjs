/* Exercise actual embedded application code with an instrumented Plotly/DOM.
 * This checks selectors and exact data wiring, separately from browser layout.
 */
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {makeDOM,AuditEvent} from '../../flux_methods/shared/report_dom.mjs';

export async function auditInteractions(html) {
  const document=makeDOM(html),plots=new Map(),stored=new Map();
  const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].map(m=>({id:m[1].match(/\bid="([^"]+)"/)?.[1],source:m[2]}));
  for(const s of scripts){if(!s.id)continue;const e=document.createElement('script');e.id=s.id;e.textContent=s.source;document.body.append(e);}
  const window={document,console,Promise,Event:AuditEvent,localStorage:{getItem:k=>stored.get(k)??null,setItem:(k,v)=>stored.set(k,String(v))},
    matchMedia:()=>({matches:false,addEventListener(){}}),addEventListener(){},dispatchEvent(){},setTimeout:fn=>{fn();return 1;},clearTimeout(){},
    requestAnimationFrame:fn=>{fn();return 1;},getComputedStyle:()=>({getPropertyValue:()=>''})};
  window.window=window;window.globalThis=window;
  window.Plotly={react(target,data,layout,config){
    for(const t of data){
      if(t.type==='heatmap'){assert.equal(t.z.length,t.y.length);t.z.forEach(row=>assert.equal(row.length,t.x.length));}
      else if(t.x && t.y)assert.equal(t.x.length,t.y.length,'paired trace values');
    }
    plots.set(target.id,{data,layout,config});target.data=data;target.layout=layout;target.classList.add('js-plotly-plot');
    target._plotEvents=target._plotEvents||{};target.on=(name,fn)=>{target._plotEvents[name]=fn;};return Promise.resolve();
  },Plots:{resize(){}}};
  const context=vm.createContext(window);
  for(const id of ['report-theme-runtime','report-app'])new vm.Script(scripts.find(s=>s.id===id).source,{filename:id+'.js'}).runInContext(context,{timeout:30000});
  const app=window.productionReport,D=app.data,$=id=>document.getElementById(id),checks=[];
  const change=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new AuditEvent('change'));};
  const verify=(condition,name)=>{assert.ok(condition,name);checks.push(name);};
  verify(D.rows.length===D.methods.length*D.protocol.reconstructions.length*D.cases.length && D.cases.length===6,'all configurations and six physical cases available');
  verify($('case-cards').querySelectorAll('button').length===12,'all six cases expose both reconstructions');
  change('model','euler_annular');verify(app.snapshot().model==='euler_annular' && $('case').options.length===3,'model selector updates available operating cases');
  change('case','euler_annular/supersonic_expansion');verify(app.snapshot().caseId==='euler_annular/supersonic_expansion','case dropdown updates the comparison');
  change('reconstruction','muscl_van_leer');verify(app.snapshot().reconstruction==='muscl_van_leer','reconstruction selector updates the comparison');
  const metrics={correction:['fixed20',50],residual:['residual',1000],jacobian_blocks:['jacobian_blocks',1000],linear_solve:['linear_solve',1000],direction:['direction',1000]};
  let selections=0,operations=0;
  for(const c of D.cases)for(const recon of D.protocol.reconstructions){
    app.selectCase(c.id,recon,'timing');
    const selected=D.rows.filter(r=>r.case_id===c.id && r.reconstruction===recon);
    verify(app.snapshot().rows.length===D.methods.length && app.snapshot().caseId===c.id,'selection '+c.id+' '+recon);
    assert.ok($('selection-summary').textContent.includes(c.id));
    for(const [id,[key,factor]] of Object.entries(metrics)){
      change('metric',id);const bars=plots.get('operation-plot').data;
      assert.equal(bars.length,D.methods.length);
      bars.forEach((bar,i)=>assert.equal(bar.x[0],selected[i].kernel_timing[key].mean_s*factor));
      const samples=plots.get('samples-plot').data;
      samples.forEach((trace,i)=>assert.deepEqual(Array.from(trace.y),Array.from(selected[i].kernel_timing[key].samples_s,v=>v*factor)));
      operations++;
    }
    const totals=plots.get('total-plot').data;
    totals.forEach((trace,i)=>assert.equal(trace.x[0],selected[i].timing.mean_s));
    const stage=plots.get('stages-plot').data;
    selected.forEach((r,i)=>assert.ok(Math.abs(stage.reduce((s,t)=>s+t.x[i],0)-r.timing.mean_s)<1e-10));
    app.showPage('fields');
    const pressure=plots.get('pressure-plot').data;
    pressure.slice(0,D.methods.length).forEach((trace,i)=>assert.deepEqual(Array.from(trace.y),Array.from(selected[i].fields.values.p)));
    for(const mode of ['equation','boundary','global']){
      change('residual-mode',mode);
      assert.equal(plots.get('residual-plot').data.length,D.methods.length);
      assert.ok(plots.get('residual-plot').data.every(t=>t.x.length>0));
    }
    if(c.id.includes('shock')){change('profile-range','shock');assert.equal(plots.get('pressure-plot').layout.xaxis.range.length,2);}
    selections++;
  }
  checks.push(`${selections} case/reconstruction selections; ${operations} exact operation/sample mappings; all stage totals and saved profiles match`);
  app.selectCase('euler_nozzle/shock_35','constant','fields');
  verify($('accuracy-notice').textContent.includes('Rusanov'),'constant nozzle Rusanov accuracy miss stays visible');
  for(const c of D.comparisons){
    app.selectCase(c.case,c.reconstruction,'timing');
    verify($('case-interpretation').textContent.includes(c.variant),'same-flux interpretation '+c.variant+' '+c.case+' '+c.reconstruction);
    if(!c.comparison_field_agreement_passed)verify($('case-interpretation').textContent.includes('not a verified time-to-identical-state'),'state disagreement disclosed '+c.variant+' '+c.case);
  }
  const modified=D.methods.find(m=>m.id==='single_stadtke_modified');
  verify(modified && modified.grids.length===1 && modified.grids[0]===400 && modified.custom_jump_correction && modified.custom_endpoint_envelope,'modified Stadtke single-grid method retains both corrections');
  verify(D.comparisons.filter(c=>c.single_method===modified.id).length===D.cases.length*D.protocol.reconstructions.length,'modified Stadtke has every single-grid/sequence comparison');
  $('methods-none').click();
  verify(app.snapshot().rows.length===0 && !$('empty-selection').hidden,'empty method selection never substitutes a method');
  assert.equal(plots.get('operation-plot').data.length,0);
  $('methods-all').click();verify(app.snapshot().rows.length===D.methods.length,'all methods restored');
  const first=$('methods').querySelector('input');first.click();
  verify(app.snapshot().rows.length===D.methods.length-1,'individual method checkbox filters all views');first.click();
  for(const page of ['overview','timing','fields','protocol']){
    app.showPage(page);assert.equal(document.querySelectorAll('.report-page').filter(e=>!e.hidden).length,1);
    assert.equal($('page-'+page).hidden,false);
  }
  checks.push('all four pages remain mutually exclusive and accessible');
  const target=$('tab-overview');target.dispatchEvent(new AuditEvent('keydown',{key:'ArrowRight'}));
  verify($('tab-timing').getAttribute('aria-selected')==='true','keyboard arrow navigation works');
  window.FluxReportUI.setTheme('dark');verify(document.documentElement.dataset.theme==='dark','dark theme applied');
  assert.equal(plots.get('operation-plot').layout.font.color,'#e5eff8');
  window.FluxReportUI.setTheme('light');verify(document.documentElement.dataset.theme==='light','light theme applied');
  $('case-cards').querySelector('button').click();verify(app.snapshot().page==='timing','case cards open the timing explorer');
  await Promise.resolve();
  $('coverage-plot')._plotEvents.plotly_click({points:[{pointNumber:[7,0]}]});
  verify(app.snapshot().caseId==='euler_annular/shock35' && app.snapshot().reconstruction==='muscl_van_leer','heatmap cells select their physical case and reconstruction');
  return {passed:true,checks,cases:D.cases.length,configurations:D.rows.length,case_reconstruction_selections:selections,operation_comparisons:operations,
    plotly:'instrumented API; actual browser rendering is a separate check',numerical_timings_rerun:false};
}
