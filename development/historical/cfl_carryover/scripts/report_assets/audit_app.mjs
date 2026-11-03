/* Execute the actual application against the same deterministic DOM as flux reports. */
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {makeDOM,AuditEvent} from '../../../../flux_methods/shared/report_dom.mjs';

export async function auditApplication(html){
  const document=makeDOM(html),plots=new Map(),storage=new Map();
  const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].map(m=>({id:m[1].match(/\bid="([^"]+)"/)?.[1],source:m[2]}));
  for(const s of scripts)if(s.id){const e=document.createElement('script');e.id=s.id;e.textContent=s.source;document.body.append(e);}
  const window={document,console,Promise,Event:AuditEvent,KeyboardEvent:AuditEvent,
    localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,String(v))},
    matchMedia:()=>({matches:false,addEventListener(){}}),addEventListener(){},dispatchEvent(){},
    setTimeout:fn=>{fn();return 1;},clearTimeout(){},requestAnimationFrame:fn=>{fn();return 1;},
    getComputedStyle:()=>({getPropertyValue:()=>''})};
  window.window=window;window.globalThis=window;
  window.Plotly={react(target,data,layout){for(const t of data)if(t.x&&t.y)assert.equal(t.x.length,t.y.length,'paired arrays: '+target.id);
    plots.set(target.id,{data,layout});target.data=data;target.layout=layout;target.classList.add('js-plotly-plot');return Promise.resolve();},Plots:{resize(){}}};
  const context=vm.createContext(window);
  for(const id of ['report-theme-runtime','study-runtime','report-app'])new vm.Script(scripts.find(s=>s.id===id).source,{filename:id+'.js'}).runInContext(context,{timeout:30000});
  const app=window.studyReport,D=app.data,$=id=>document.getElementById(id),checks=[];
  const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new AuditEvent('change'));};
  const same=(a,b)=>assert.equal(JSON.stringify(a),JSON.stringify(b));
  const verify=(condition,name)=>{assert.ok(condition,name);checks.push(name);};
  verify(D.records.length===32&&D.comparisons.length===4,'32 actual hierarchies / 4 comparisons');
  let selections=0;
  for(const caseId of ['euler_nozzle/shock_35','euler_nozzle/shock_70'])for(const recon of ['constant','muscl_van_leer']){
    change('case',caseId);change('reconstruction',recon);app.showPage('explorer');
    const meshes=[50,100,200,400];
    assert.equal($('cells').options.length,meshes.length);
    for(const cells of meshes)for(const sample of ['median','1','2','3']){
      change('cells',cells);change('sample',sample);
      const rows=app.snapshot().rows.map(id=>D.records.find(r=>r.id===id));
      assert.equal(rows.length,2);
      for(let i=0;i<2;i++){
        const r=rows[i],s=r.stages.find(s=>s.cells===cells),q=D.comparisons.find(g=>g.case_id===caseId&&g.reconstruction===recon);
        assert.equal(r.phase,'measured');assert.equal(r.policy,D.methods[i]);
        if(sample!=='median')assert.equal(r.repeat,Number(sample));else assert.equal(r.id,q.timing[r.policy].representative_id);
        same(plots.get('pressure-plot').data[i].y,s.state.slice(1,-1).map(w=>w[0]/1e5));
        same(plots.get('velocity-plot').data[i].y,s.state.slice(1,-1).map(w=>w[1]));
        same(plots.get('entry-cfl-plot').data[2*i].y,r.stages.map(s=>s.initial_cfl));
        same(plots.get('entry-cfl-plot').data[2*i+1].y,r.stages.map(s=>s.last_accepted_cfl));
        same(plots.get('work-plot').data[i].y,['accepted','rejected','corrections','backtracks'].map(k=>s.work[k]));
        same(plots.get('selected-timing').data[2*i].y,q.timing[r.policy].samples_s);
        const rejected=s.history.filter(h=>h.phase==='rejected');same(plots.get('cfl-plot').data[2*i+1].y,rejected.map(h=>h.cfl));
      }
      assert.equal(plots.get('pressure-plot').data.length,cells===400?3:2);
      if(cells===400)same(plots.get('pressure-plot').data.at(-1).y,rows[0].reference_pressure.map(v=>v/1e5));
      for(const view of ['ratio','equations','global','boundary']){
        change('residual-mode',view);assert.equal(plots.get('residual-plot').data.length,view==='equations'?6:2);
        assert.ok(plots.get('residual-plot').data.every(t=>t.x[0]===0));
      }
      selections++;
    }
  }
  checks.push(`${selections} case / reconstruction / mesh / run selections preserve saved arrays and all residual modes`);
  change('cells',400);change('profile-range','shock');verify(plots.get('pressure-plot').layout.xaxis.range.length===2,'shock zoom updates axes');
  $('methods-none').click();verify(app.snapshot().rows.length===0&&plots.get('pressure-plot').data.length===0&&!$('empty-selection').hidden,'empty selection remains empty');
  $('methods-all').click();const input=$('methods').querySelector('input');input.click();verify(app.snapshot().rows.length===1,'individual method selection');input.click();
  for(const page of ['overview','explorer','discussion','protocol']){app.showPage(page);verify(document.querySelectorAll('.report-page').filter(p=>!p.hidden).length===1,'exclusive navigation '+page);}
  $('tab-overview').dispatchEvent(new AuditEvent('keydown',{key:'ArrowRight'}));verify($('tab-explorer').getAttribute('aria-selected')==='true','keyboard tabs');
  window.FluxReportUI.setTheme('dark');verify(document.documentElement.dataset.theme==='dark'&&plots.get('pressure-plot').layout.font.color==='#e5eff8','dark theme recolors actual plots');
  window.FluxReportUI.setTheme('light');verify(document.documentElement.dataset.theme==='light','light theme');
  app.showPage('overview');$('all-rows').querySelectorAll('button')[3].click();verify(app.snapshot().page==='explorer'&&app.snapshot().caseId==='euler_nozzle/shock_70','table opens selected configuration');
  app.showPage('overview');$('case-cards').querySelectorAll('button')[0].click();verify(app.snapshot().caseId==='euler_nozzle/shock_35'&&app.snapshot().page==='explorer','case card navigation');
  await Promise.resolve();
  return {passed:true,selections,checks,numerical_cases_rerun:false,plotly:'Actual application with instrumented data API; browser rendering checked separately'};
}
