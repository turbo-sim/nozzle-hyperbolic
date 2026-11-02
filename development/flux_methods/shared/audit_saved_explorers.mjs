/* Execute actual report application code against a small DOM/Plotly adapter.
 * Browser rendering and layout are a separate check.
 */
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {makeDOM,AuditEvent} from './report_dom.mjs';

export async function auditExplorer(html) {
  const document=makeDOM(html),plots=new Map(),storage=new Map(),checks=[];
  const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].map(m=>({id:m[1].match(/\bid="([^"]+)"/)?.[1],source:m[2]}));
  for(const s of scripts)if(s.id){const e=document.createElement('script');e.id=s.id;e.textContent=s.source;document.body.append(e);}
  const window={document,console,Promise,Event:AuditEvent,localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,String(v))},matchMedia:()=>({matches:false,addEventListener(){}}),dispatchEvent(){},addEventListener(){},setTimeout:fn=>{fn();return 1;},clearTimeout(){},requestAnimationFrame:fn=>{fn();return 1;},getComputedStyle:()=>({getPropertyValue:()=>''})};
  window.window=window;window.globalThis=window;
  window.Plotly={react(target,data,layout,config){for(const t of data)if(t.x&&t.y)assert.equal(t.x.length,t.y.length,'paired values in '+target.id);plots.set(target.id,{data,layout,config});target.data=data;target.layout=layout;target.classList.add('js-plotly-plot');return Promise.resolve();},Plots:{resize(){}}};
  const context=vm.createContext(window);
  for(const id of ['report-theme-runtime','study-runtime','report-app'])new vm.Script(scripts.find(s=>s.id===id).source,{filename:id+'.js'}).runInContext(context,{timeout:30000});
  const app=window.studyReport,D=app.data,$=id=>document.getElementById(id);
  const verify=(value,name)=>{assert.ok(value,name);checks.push(name);};
  const same=(a,b)=>assert.equal(JSON.stringify(a),JSON.stringify(b));
  const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new AuditEvent('change'));};
  let states=0;
  if(D.study==='limiters'){
    verify(D.records.length===8 && D.cells.length===2 && D.methods.length===4,'one physical case, four limiters, two meshes');
    verify($('case').options.length===1,'case selector does not invent additional problems');
    for(const n of D.cells){change('cells',n);app.showPage('explorer');const rows=D.methods.map(m=>D.records.find(r=>r.limiter===m&&r.cells===n));
      verify(app.snapshot().rows.length===4,'all four limiters selected at '+n+' cells');
      const pressure=plots.get('pressure-plot').data;
      pressure.slice(0,4).forEach((t,i)=>same(t.y,rows[i].pressure.map(v=>v/1e5)));
      same(pressure.at(-2).y,rows[0].analytical_pressure.map(v=>v/1e5));same(pressure.at(-1).y,rows[0].reference_pressure.map(v=>v/1e5));
      verify(plots.get('mach-plot').data.length===4,'Mach view uses saved numerical profiles only');
      for(const mode of ['ratio','equations','global','boundary']){change('residual-mode',mode);const data=plots.get('residual-plot').data;assert.equal(data.length,mode==='equations'?12:4);assert.ok(data.every(t=>t.x[0]===0));}
      const work=plots.get('work-plot').data;work.forEach((t,i)=>assert.equal(t.y[2],rows[i].history.reduce((sum,h)=>sum+(h.inner_iterations||0),0)));
      change('profile-range','shock');assert.ok(plots.get('pressure-plot').layout.xaxis.range);states++;
    }
    $('methods-none').click();verify(app.snapshot().rows.length===0 && plots.get('pressure-plot').data.length===0,'empty limiter selection remains empty');
    $('methods-all').click();const input=$('methods').querySelector('input');input.click();verify(app.snapshot().rows.length===3,'individual limiter filter updates plots');input.click();
    app.showPage('overview');$('case-cards').querySelector('button').click();verify(app.snapshot().page==='explorer','mesh cards navigate to case explorer');
  }else{
    verify(D.cases.length===5 && D.records.length===13,'five physical cases and thirteen retained attempts');
    const gallery=plots.get('gallery-plot');assert.equal(gallery.layout.grid.rows,5);verify(new Set(gallery.data.map(t=>t.yaxis)).size===5,'campaign gallery has five distinct operating-point panels');
    for(const c of D.cases){change('case',c.id);change('cells',96);app.showPage('explorer');const r=D.records.find(r=>r.id===app.snapshot().rows[0]);
      verify(app.snapshot().rows.length===1 && r.property_domain_audited,'finest audited default for '+c.id);
      const p=plots.get('pressure-plot').data;same(p[0].y,r.fields.p.map(v=>v*1e-5));same(p[1].y,c.experiment.pressure_pa.map(v=>v/1e5));
      assert.equal(p.length,c.upstream_cfd?3:2);
      const q=plots.get('quality-plot').data[0];same(q.y,r.displayed_liquid_fraction);verify(q.y.filter(v=>v==null).length===r.quality_diagnostic.masked_samples,'raw-quality gaps preserved for '+c.id);
      assert.equal(plots.get('quality-plot').data.length,c.upstream_cfd?2:1);
      assert.equal(plots.get('residual-plot').data.length,3);
      if(!c.upstream_cfd){assert.equal($('show-cfd').disabled,true);assert.ok($('reference-notice').textContent.includes('No reference CFD'));}
      states++;
    }
    change('case','L1');change('cells','all');change('attempt-scope','all');verify(app.snapshot().rows.length===5,'all L1 attempts include the retained preflight and stretched grid');
    change('cells',24);verify(app.snapshot().rows.length===2,'24-cell preflight and audited retry remain distinguishable');
    change('attempt-scope','audited');verify(app.snapshot().rows.length===1,'property-domain filter excludes only the preflight');
    change('cells',48);change('grid-kind','stretched');verify(app.snapshot().rows.length===1 && app.snapshot().rows[0].includes('1.6'),'stretched grid is selectable');
    change('case','L2');verify(app.snapshot().rows.length===0 && !$('empty-selection').hidden,'unavailable stretched case remains empty');
    change('grid-kind','all');change('cells',96);change('case','L1');
    $('show-cfd').click();verify(plots.get('pressure-plot').data.length===2,'CFD reference can be disabled');$('show-cfd').click();
    $('show-onset').click();verify(!plots.get('pressure-plot').layout.shapes,'measured-onset marker can be disabled');$('show-onset').click();
    $('runs-none').click();verify(app.snapshot().rows.length===0 && plots.get('pressure-plot').data.length===0,'empty solution selection is not substituted');$('runs-all').click();
    const input=$('runs').querySelector('input');input.click();assert.equal(app.snapshot().rows.length,0);input.click();
    app.showPage('overview');$('case-cards').querySelectorAll('button')[4].click();verify(app.snapshot().caseId==='L5' && app.snapshot().page==='explorer','physical case cards navigate to the correct solution');
  }
  for(const page of ['overview','explorer','discussion','protocol']){app.showPage(page);verify(document.querySelectorAll('.report-page').filter(e=>!e.hidden).length===1 && !$('page-'+page).hidden,'exclusive navigation '+page);}
  $('tab-overview').dispatchEvent(new AuditEvent('keydown',{key:'ArrowRight'}));verify($('tab-explorer').getAttribute('aria-selected')==='true','keyboard page navigation works');
  window.FluxReportUI.setTheme('dark');verify(document.documentElement.dataset.theme==='dark','dark theme applied');assert.equal(plots.get('pressure-plot').layout.font.color,'#e5eff8');window.FluxReportUI.setTheme('light');verify(document.documentElement.dataset.theme==='light','light theme applied');
  await Promise.resolve();
  return {passed:true,study:D.study,records:D.records.length,case_mesh_states:states,checks,numerical_cases_rerun:false,plotly:'instrumented data API; browser layout is a separate check'};
}
