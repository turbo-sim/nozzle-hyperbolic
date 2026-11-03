/* Saved-data explorer. No flow calculation, EOS evaluation or timing occurs here. */
(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('payload').textContent);
  const U=StudyReportUI, {$,escape:E,fmt:F,badge:B,stat:S}=U;
  const names={reset:'Reset CFL',carry:'Carry accepted CFL'};
  const recons={constant:'Constant',muscl_van_leer:'MUSCL · Van Leer'};
  const colors={reset:'#6a00a8',carry:'#fca636'}, dash={reset:'solid',carry:'dash'};
  const chosen=new Set(D.methods);
  let controller;
  const caseId=()=>$('case').value, recon=()=>$('reconstruction').value;
  const cells=()=>Number($('cells').value);
  const group=()=>D.comparisons.find(g=>g.case_id===caseId()&&g.reconstruction===recon());
  const record=id=>D.records.find(r=>r.id===id);
  function rows(){
    const g=group();
    return D.methods.filter(m=>chosen.has(m)).map(m=>$('sample').value==='median'
      ?record(g.timing[m].representative_id)
      :D.records.find(r=>r.case_id===caseId()&&r.reconstruction===recon()&&r.phase==='measured'&&r.repeat===Number($('sample').value)&&r.policy===m)).filter(Boolean);
  }
  const stage=r=>r.stages.find(s=>s.cells===cells());
  const accepted=s=>s.history.filter(h=>['initial','accepted'].includes(h.phase));
  const label=r=>`${names[r.policy]} · run ${r.repeat}`;
  function coordinates(r,s){
    if(s.cells===400)return r.coordinate;
    const dx=r.coordinate[1]-r.coordinate[0],left=r.coordinate[0]-.5*dx,length=400*dx;
    return Array.from({length:s.cells},(_,i)=>left+(i+.5)*length/s.cells);
  }
  function syncCells(){
    const old=cells(),values=D.cells;
    $('cells').innerHTML=values.map(n=>`<option value="${n}">${n} physical cells</option>`).join('');
    $('cells').value=String(values.includes(old)?old:400);
  }
  function layout(title,y,x){const l=U.layout(title,y);l.xaxis.title={text:x};return l;}
  function points(g,method,position){
    const t=g.timing[method];
    return {type:'scatter',mode:'markers',name:names[method],legendgroup:method,
      x:t.samples_s.map((_,i)=>position+[-.04,0,.04][i]),y:t.samples_s,
      customdata:t.sample_ids.map(id=>record(id).repeat),marker:{color:colors[method],size:9},
      hovertemplate:`${E(names[method])}<br>Run %{customdata}<br>%{y:.6f} s<extra></extra>`};
  }
  function mean(g,method,position){return {type:'scatter',mode:'markers',name:names[method]+' mean',
    x:[position],y:[g.timing[method].mean_s],marker:{symbol:'line-ew',color:colors[method],size:24,line:{width:3,color:colors[method]}},
    showlegend:false,hovertemplate:'Mean: %{y:.6f} s<extra></extra>'};}
  function overview(){
    const g=group(),rr=rows(),selectedMethods=D.methods.filter(m=>chosen.has(m));
    const measured=D.records.filter(r=>r.phase==='measured');
    $('kpis').innerHTML=S('Timed complete hierarchies',`${measured.filter(r=>r.success).length} / ${measured.length}`,'Three per policy / configuration')
      +S('Saved stage audits','128 / 128','Includes eight excluded complete warmups')
      +S('Selected reset / carry',selectedMethods.length===2?F(g.reset_over_carry)+'×':'—','Above one favors carryover; three samples')
      +S('Largest final-state spread',F(Math.max(...D.comparisons.map(r=>r.normalized_state_spread))),'Fixed variable scales; same-root limit 1e−6');
    $('case-cards').innerHTML=D.cases.map(c=>{
      const a=c.definition,percent=c.id.endsWith('35')?'35%':'70%';
      return `<article class="case-card ${c.id===caseId()?'selected':''}"><p class="eyebrow">PERFECT-GAS EULER NOZZLE</p><h3>Shock at ${percent} of divergence</h3><p>Air · p₀ = ${F(a.inlet.p0/1e5)} bar · T₀ = ${F(a.inlet.T0)} K. Constant and MUSCL; four sequenced meshes.</p><button data-case="${E(c.id)}">Explore this case</button></article>`;
    }).join('');
    $('case-cards').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('case').value=b.dataset.case;controller.showPage('explorer');});
    const ratios=D.comparisons.map(g=>g.reset_over_carry);
    $('findings').innerHTML=`<p>The reset / carry mean-time ratio is <strong>${F(Math.min(...ratios))}–${F(Math.max(...ratios))}×</strong> in these four configurations.</p><p>Every original stage audit passes. Final states agree within ${F(Math.max(...D.comparisons.map(g=>g.normalized_state_spread)))} in fixed variable scales.</p><p>The evidence supports the adopted carryover default here. Three samples on one machine/session do not establish a general speedup guarantee.</p>`;
    const groups=D.comparisons,traces=[];
    for(const m of selectedMethods)groups.forEach((q,i)=>{const x=i+(m==='reset'?-.13:.13),p=points(q,m,x);p.showlegend=i===0;traces.push(p,mean(q,m,x));});
    const t=layout('Complete hierarchy timings','Complete solve time [s]','Shock / reconstruction');
    t.xaxis.tickvals=groups.map((_,i)=>i);t.xaxis.ticktext=groups.map(q=>(q.case_id.endsWith('35')?'35%':'70%')+' · '+(q.reconstruction==='constant'?'Constant':'MUSCL'));
    U.plot('timing-plot',traces,t);
    const savings=layout('Benefit of carrying accepted CFL','Reset / carry mean time [−]','Shock / reconstruction');
    U.plot('savings-plot',selectedMethods.length===2?[{type:'bar',name:'Reset / carry',
      x:groups.map(q=>(q.case_id.endsWith('35')?'35%':'70%')+' · '+(q.reconstruction==='constant'?'Constant':'MUSCL')),
      y:groups.map(q=>q.timing_eligible?q.reset_over_carry:null),marker:{color:colors.carry},hovertemplate:'%{x}<br>%{y:.3f}×<extra></extra>'}]:[],savings);
    $('all-rows').innerHTML=groups.map((q,i)=>`<tr><td>${q.case_id.endsWith('35')?'35%':'70%'} · ${E(recons[q.reconstruction])}</td>${['reset','carry'].map(m=>`<td>${F(q.timing[m].mean_s)} [${F(q.timing[m].min_s)}–${F(q.timing[m].max_s)}]</td>`).join('')}<td>${q.timing_eligible?F(q.reset_over_carry)+'×':'Ineligible'}</td><td>${B(q.timing_eligible,'Pass','Ineligible')}</td><td><button class="small-button" data-comparison="${i}">Inspect</button></td></tr>`).join('');
    $('all-rows').querySelectorAll('button').forEach(b=>b.onclick=()=>{const q=D.comparisons[Number(b.dataset.comparison)];$('case').value=q.case_id;$('reconstruction').value=q.reconstruction;syncCells();controller.showPage('explorer');});
  }
  function explorer(rr){
    const g=group(),methods=D.methods.filter(m=>chosen.has(m)),timing=[];
    methods.forEach((m,i)=>timing.push(points(g,m,i),mean(g,m,i)));
    const t=layout('All measured complete-solve timings','Complete solve time [s]','CFL policy');
    t.xaxis.tickvals=methods.map((_,i)=>i);t.xaxis.ticktext=methods.map(m=>names[m]);U.plot('selected-timing',timing,t);
    const stages=layout('Mean stage wall time','Mean stage time [s]','Physical cells');
    stages.xaxis.tickvals=D.cells;
    U.plot('stage-time',methods.map(m=>{const samples=g.timing[m].sample_ids.map(record);return {type:'scatter',mode:'lines+markers',name:names[m],
      x:stages.xaxis.tickvals,y:stages.xaxis.tickvals.map(n=>samples.reduce((sum,r)=>sum+r.stages.find(s=>s.cells===n).stage_elapsed_s,0)/samples.length),
      line:{color:colors[m],dash:dash[m]},marker:{size:7}};}),stages);
    const curve=(r,values)=>({type:'scatter',mode:'lines',name:label(r),x:coordinates(r,stage(r)),y:values,line:{color:colors[r.policy],dash:dash[r.policy],width:2}});
    const p=layout(`Saved pressure · ${cells()} cells`,'Pressure [bar]','Axial coordinate [m]');
    if($('profile-range').value==='shock'&&rr.length){const x=rr[0].reference_shock_m;p.xaxis.range=[x-.015,x+.015];}
    const pressure=rr.map(r=>curve(r,stage(r).state.slice(1,-1).map(w=>w[0]/1e5)));
    if(rr.length&&cells()===400)pressure.push({type:'scatter',mode:'lines',name:'Analytical cell projection',x:rr[0].coordinate,y:rr[0].reference_pressure.map(v=>v/1e5),line:{color:FluxReportUI.palette().ink,dash:'dot',width:1.4}});
    U.plot('pressure-plot',pressure,p);
    const v=layout(`Saved axial velocity · ${cells()} cells`,'Velocity [m/s]','Axial coordinate [m]');v.xaxis={...p.xaxis};
    U.plot('velocity-plot',rr.map(r=>curve(r,stage(r).state.slice(1,-1).map(w=>w[1]))),v);
    const mode=$('residual-mode').value,policy=D.metadata.settings.residual,history=[];
    for(const r of rr){const h=accepted(stage(r));if(mode==='equations')for(let k=0;k<3;k++)history.push({type:'scatter',mode:'lines',name:`${label(r)} · ${['mass','momentum','energy'][k]}`,
      x:h.map(v=>v.step),y:h.map(v=>Math.max(v.equation_l2[k],1e-30)),line:{color:colors[r.policy],dash:['solid','dash','dot'][k],width:1.6}});
      else history.push({type:'scatter',mode:'lines',name:label(r),x:h.map(v=>v.step),y:h.map(v=>Math.max(mode==='ratio'?v.equation_max_ratio:mode==='global'?Math.max(...v.global_defect)/policy.global_tolerance:v.boundary_max/policy.boundary_tolerance,1e-30)),line:{color:colors[r.policy],dash:dash[r.policy]}});
    }
    const residual=layout('Original steady residual checks',mode==='equations'?'Per-equation fixed-scale L2 [−]':'Residual / allowed threshold [−]','Accepted update (initial state at zero)');residual.yaxis.type='log';
    if(mode!=='equations')residual.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:FluxReportUI.palette().muted,dash:'dot'}}];
    U.plot('residual-plot',history,residual);
    const cfl=layout('CFL and rejected timestep attempts','CFL [−]','Accepted-update counter');cfl.yaxis.type='log';
    const cf=[];for(const r of rr){const h=accepted(stage(r)),rejected=stage(r).history.filter(h=>h.phase==='rejected');
      cf.push({type:'scatter',mode:'lines',name:label(r),x:h.map(v=>v.step),y:h.map(v=>v.cfl),line:{color:colors[r.policy],dash:dash[r.policy]}});
      cf.push({type:'scatter',mode:'markers',name:names[r.policy]+' rejected',x:rejected.map(v=>v.step),y:rejected.map(v=>v.cfl),marker:{color:colors[r.policy],symbol:'x',size:7}});
    }U.plot('cfl-plot',cf,cfl);
    const w=layout('Selected-stage nonlinear work','Recorded count [−]','Work measure');w.barmode='group';
    U.plot('work-plot',rr.map(r=>({type:'bar',name:label(r),x:['Accepted','Rejected','Newton corrections','Newton trial rejections'],y:['accepted','rejected','corrections','backtracks'].map(k=>stage(r).work[k]),marker:{color:colors[r.policy]}})),w);
    const l=layout('CFL entering and leaving each mesh','CFL [−]','Physical cells');l.yaxis.type='log';l.xaxis.tickvals=D.cells;
    U.plot('entry-cfl-plot',rr.flatMap(r=>['initial_cfl','last_accepted_cfl'].map((key,i)=>({type:'scatter',mode:'lines+markers',name:label(r)+(i?' · last accepted':' · initial'),
      x:r.stages.map(s=>s.cells),y:r.stages.map(s=>s[key]),line:{color:colors[r.policy],dash:i?'dot':dash[r.policy]},marker:{symbol:i?'diamond-open':'circle',size:7}}))),l);
    $('selected-rows').innerHTML=rr.map(r=>{const s=stage(r),t=g.timing[r.policy];return `<tr><td>${E(label(r))}</td><td>${B(s.success,'Root','Failed')} ${B(s.audit.passed,'Audit','Audit miss')}</td><td>${F(s.initial_cfl)} / ${F(s.last_accepted_cfl)}</td><td>${s.work.accepted} / ${s.work.rejected}</td><td>${s.work.corrections}</td><td>${F(Math.max(...s.audit.equation_l2))}</td><td>${F(s.audit.boundary_max)} / ${F(Math.max(...s.audit.global_defect))}</td><td>${F(t.mean_s)} [${F(t.min_s)}–${F(t.max_s)}]</td></tr>`;}).join('');
  }
  function render(page){
    const rr=rows(),g=group();$('empty-selection').hidden=!!rr.length;
    $('selection-summary').innerHTML=`<strong>${caseId().endsWith('35')?'35%':'70%'} nozzle shock</strong> · ${E(recons[recon()])} · 50 → 100 → 200 → 400 · ${cells()}-cell stage · ${rr.length} selected policies <code>${E(caseId())}</code>`;
    $('case-heading').textContent=`${caseId().endsWith('35')?'35%':'70%'} shock · ${recons[recon()]} · ${cells()} cells`;
    $('selection-interpretation').innerHTML=`<h3>Interpret this selection</h3><p>Recorded mean times: reset ${F(g.timing.reset.mean_s)} s, carry ${F(g.timing.carry.mean_s)} s; ratio ${F(g.reset_over_carry)}×. ${g.timing_eligible?'The saved original audits, shock check and same-root comparison pass.':'This comparison is ineligible for a speedup claim.'}</p><p>${$('sample').value==='median'?'Each policy shows its own actual median-duration complete solve.':'The same numbered measured repetition is selected for both policies.'} ${rr.length?rr.map(r=>`${E(names[r.policy])}: run ${r.repeat}`).join(' · '):'No policy is selected; no replacement is substituted.'} Profiles and histories below use the selected mesh stage; timing samples retain the complete solve boundary.</p>`;
    $('selected-details').textContent=JSON.stringify(rr.map(r=>({id:r.id,phase:r.phase,repeat:r.repeat,input_file:r.input_file,scheme:r.scheme,elapsed_s:r.elapsed_s,total_work:r.total_work,stage:{...stage(r),history:undefined,state:undefined},source_head:D.metadata.source_head})),null,2);
    if(page==='overview')overview();
    if(page==='explorer')explorer(rr);
  }
  $('methods').innerHTML=D.methods.map(m=>`<label><input type="checkbox" value="${m}" checked><span class="dot" style="background:${colors[m]}"></span>${names[m]}</label>`).join('');
  $('methods').querySelectorAll('input').forEach(i=>i.onchange=()=>{i.checked?chosen.add(i.value):chosen.delete(i.value);controller.render();});
  for(const action of ['all','none'])$('methods-'+action).onclick=()=>{$('methods').querySelectorAll('input').forEach(i=>{i.checked=action==='all';i.checked?chosen.add(i.value):chosen.delete(i.value);});controller.render();};
  $('cells').innerHTML='<option value="400">400 physical cells</option>';syncCells();
  for(const id of ['case','reconstruction','cells','sample','profile-range','residual-mode'])$(id).onchange=()=>{controller.render();};
  $('provenance-details').textContent=JSON.stringify({metadata:D.metadata,presentation:D.presentation,production_check:D.production_check},null,2);
  controller=U.mount(render);
  window.studyReport={data:D,...controller,snapshot:()=>({...controller.snapshot(),caseId:caseId(),reconstruction:recon(),cells:cells(),sample:$('sample').value,methods:[...chosen],rows:rows().map(r=>r.id)})};
  const hash=window.location?.hash?.slice(1);if(['overview','explorer','discussion','protocol'].includes(hash))controller.showPage(hash);
  if(window.location?.search?.includes('audit=1'))window.addEventListener('load',async()=>{
    const checks=[];const wait=async()=>{for(let i=0;i<600;i++){if(controller.snapshot().rendered===controller.snapshot().renderNumber)return;await new Promise(r=>setTimeout(r,20));}throw Error('Plot rendering timeout');};
    const change=async(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new Event('change'));await wait();};
    try{
      await wait();
      for(const page of ['overview','explorer','discussion','protocol']){controller.showPage(page);await wait();checks.push(!document.getElementById('page-'+page).hidden&&document.querySelectorAll('.report-page:not([hidden])').length===1);}
      controller.showPage('explorer');await wait();
      await change('case','euler_nozzle/shock_70');await change('sample','2');
      checks.push(rows().every(r=>r.repeat===2)&&$('pressure-plot').data.length===3);
      await change('cells',100);checks.push($('pressure-plot').data.length===2&&$('pressure-plot').data.every(t=>t.y.length===100));
      await change('residual-mode','equations');checks.push($('residual-plot').data.length===6);
      $('methods-none').click();await wait();checks.push(rows().length===0&&$('pressure-plot').data.length===0&&!$('empty-selection').hidden);$('methods-all').click();await wait();
      for(const theme of ['dark','light']){FluxReportUI.setTheme(theme);await wait();checks.push(document.documentElement.dataset.theme===theme&&$('pressure-plot').layout.font.color===FluxReportUI.palette().ink);}
      await change('case','euler_nozzle/shock_35');await change('reconstruction','muscl_van_leer');await change('cells',400);await change('sample','median');await change('residual-mode','ratio');await change('profile-range','shock');
      checks.push($('pressure-plot').layout.xaxis.range.length===2);
      await Plotly.relayout($('pressure-plot'),{'xaxis.range':[.125,.145]});checks.push($('pressure-plot').layout.xaxis.range[0]===.125);
      await change('profile-range','domain');
      $('tab-overview').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));await wait();checks.push($('tab-explorer').getAttribute('aria-selected')==='true');
      const a=[...document.querySelectorAll('a[download]')].find(a=>a.download==='plotted_data.json');checks.push(JSON.parse(atob(a.href.split(',')[1])).records.length===32);
      controller.showPage(hash==='explorer'?'explorer':'overview');await wait();
      const result={passed:checks.every(Boolean),checks,numerical_cases_rerun:false,viewport:{width:window.innerWidth,documentWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,deviceScale:window.devicePixelRatio}};const p=document.createElement('pre');p.id='browser-audit';p.textContent=JSON.stringify(result);document.body.append(p);
    }catch(error){const p=document.createElement('pre');p.id='browser-audit';p.textContent=JSON.stringify({passed:false,error:String(error),checks});document.body.append(p);}
  });
})();
