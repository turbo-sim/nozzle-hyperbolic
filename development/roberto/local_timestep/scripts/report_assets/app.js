/* Saved-data explorer. No flow calculation, EOS evaluation or timing occurs here. */
(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('payload').textContent);
  const U=StudyReportUI, {$,escape:E,fmt:F,badge:B,stat:S}=U;
  const names={global:'Global timestep',local:'Local timestep'};
  const modes={sequence:'50 → 100 → 200 → 400',single_400:'Cold 400-cell grid'};
  const recons={constant:'Constant',muscl_van_leer:'MUSCL · Van Leer'};
  const colors={global:'#6a00a8',local:'#fca636'}, dash={global:'solid',local:'dash'};
  const chosen=new Set(D.methods);
  let controller;
  const caseId=()=>$('case').value, strategy=()=>$('strategy').value, recon=()=>$('reconstruction').value;
  const cells=()=>Number($('cells').value);
  const group=()=>D.comparisons.find(g=>g.case_id===caseId()&&g.mode===strategy()&&g.reconstruction===recon());
  const record=id=>D.records.find(r=>r.id===id);
  function rows(){
    const g=group();
    return D.methods.filter(m=>chosen.has(m)).map(m=>$('sample').value==='median'
      ?record(g.timing[m].representative_id)
      :D.records.find(r=>r.case_id===caseId()&&r.mode===strategy()&&r.reconstruction===recon()&&r.phase==='measured'&&r.repeat===Number($('sample').value)&&r.method===m)).filter(Boolean);
  }
  const stage=r=>r.stages.find(s=>s.cells===cells());
  const accepted=s=>s.history.filter(h=>['initial','accepted'].includes(h.phase));
  const label=r=>`${names[r.method]} · run ${r.repeat}`;
  function coordinates(r,s){
    if(s.cells===400)return r.coordinate;
    const dx=r.coordinate[1]-r.coordinate[0],left=r.coordinate[0]-.5*dx,length=400*dx;
    return Array.from({length:s.cells},(_,i)=>left+(i+.5)*length/s.cells);
  }
  function syncCells(){
    const old=cells(),values=strategy()==='sequence'?D.cells:[400];
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
    $('kpis').innerHTML=S('Timed complete solves',`${measured.filter(r=>r.success).length} / ${measured.length}`,'Three per method / strategy / configuration')
      +S('Saved stage audits','160 / 160','Includes 16 excluded complete warmups')
      +S('Selected global / local',selectedMethods.length===2?F(g.global_over_local)+'×':'—','Above one favors local; three samples')
      +S('Largest cross-strategy spread',F(Math.max(...D.cross_strategy.map(r=>r.all_methods_and_strategies_state_spread))),'Fixed variable scales; same-root limit 1e−6');
    $('case-cards').innerHTML=D.cases.map(c=>{
      const a=c.definition,percent=c.id.endsWith('35')?'35%':'70%';
      return `<article class="case-card ${c.id===caseId()?'selected':''}"><p class="eyebrow">PERFECT-GAS EULER NOZZLE</p><h3>Shock at ${percent} of divergence</h3><p>Air · p₀ = ${F(a.inlet.p0/1e5)} bar · T₀ = ${F(a.inlet.T0)} K. Constant and MUSCL, sequenced and cold-grid solutions.</p><button data-case="${E(c.id)}">Explore this case</button></article>`;
    }).join('');
    $('case-cards').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('case').value=b.dataset.case;controller.showPage('explorer');});
    const ranges=(mode,rec)=>D.comparisons.filter(g=>g.mode===mode&&g.reconstruction===rec).map(g=>g.global_over_local);
    const seq=ranges('sequence','muscl_van_leer'),single=ranges('single_400','muscl_van_leer');
    $('findings').innerHTML=`<p>MUSCL mean-time ratios are <strong>${F(Math.min(...seq))}–${F(Math.max(...seq))}×</strong> with sequencing and <strong>${F(Math.min(...single))}–${F(Math.max(...single))}×</strong> on cold 400-cell grids.</p><p>Constant reconstruction remains close to parity. Grid sequencing is the larger acceleration in these saved data: complete cold/sequence ratios range from ${F(Math.min(...D.cross_strategy.flatMap(r=>Object.values(r.single_over_sequence))))} to ${F(Math.max(...D.cross_strategy.flatMap(r=>Object.values(r.single_over_sequence))))}×.</p><p>The development adapter changes no physical equations or production defaults. Every timing sample is retained; small differences and overlapping ranges deserve caution.</p>`;
    const groups=D.comparisons.filter(g=>g.mode===strategy()),traces=[];
    for(const m of selectedMethods)groups.forEach((q,i)=>{const x=i+(m==='global'?-.13:.13),p=points(q,m,x);p.showlegend=i===0;traces.push(p,mean(q,m,x));});
    const t=layout('Complete timings · '+modes[strategy()],'Complete solve time [s]','Shock / reconstruction');
    t.xaxis.tickvals=groups.map((_,i)=>i);t.xaxis.ticktext=groups.map(q=>(q.case_id.endsWith('35')?'35%':'70%')+' · '+(q.reconstruction==='constant'?'Constant':'MUSCL'));
    U.plot('timing-plot',traces,t);
    const s=layout('What grid sequencing adds','Cold 400 / grid-sequence mean time [−]','Shock / reconstruction');s.barmode='group';
    U.plot('strategy-plot',selectedMethods.map(m=>({type:'bar',name:names[m],
      x:D.cross_strategy.map(r=>(r.case_id.endsWith('35')?'35%':'70%')+' · '+(r.reconstruction==='constant'?'Constant':'MUSCL')),
      y:D.cross_strategy.map(r=>r.eligible?r.single_over_sequence[m]:null),marker:{color:colors[m]},hovertemplate:'%{x}<br>%{y:.3f}×<extra></extra>'})),s);
    $('all-rows').innerHTML=D.comparisons.map((q,i)=>`<tr><td>${q.case_id.endsWith('35')?'35%':'70%'} · ${E(recons[q.reconstruction])}</td><td>${E(modes[q.mode])}</td><td>${F(q.timing.global.mean_s)}</td><td>${F(q.timing.local.mean_s)}</td><td>${q.eligible?F(q.global_over_local)+'×':'Ineligible'}</td><td>${B(q.eligible,'Pass','Ineligible')}</td><td><button class="small-button" data-comparison="${i}">Inspect</button></td></tr>`).join('');
    $('all-rows').querySelectorAll('button').forEach(b=>b.onclick=()=>{const q=D.comparisons[Number(b.dataset.comparison)];$('case').value=q.case_id;$('strategy').value=q.mode;$('reconstruction').value=q.reconstruction;syncCells();controller.showPage('explorer');});
  }
  function explorer(rr){
    const g=group(),methods=D.methods.filter(m=>chosen.has(m)),timing=[];
    methods.forEach((m,i)=>timing.push(points(g,m,i),mean(g,m,i)));
    const t=layout('All measured complete-solve timings','Complete solve time [s]','Pseudo-time method');
    t.xaxis.tickvals=methods.map((_,i)=>i);t.xaxis.ticktext=methods.map(m=>names[m]);U.plot('selected-timing',timing,t);
    const stages=layout('Mean stage wall time','Mean stage time [s]','Physical cells');
    stages.xaxis.tickvals=strategy()==='sequence'?D.cells:[400];
    U.plot('stage-time',methods.map(m=>{const samples=g.timing[m].sample_ids.map(record);return {type:'scatter',mode:'lines+markers',name:names[m],
      x:stages.xaxis.tickvals,y:stages.xaxis.tickvals.map(n=>samples.reduce((sum,r)=>sum+r.stages.find(s=>s.cells===n).stage_elapsed_s,0)/samples.length),
      line:{color:colors[m],dash:dash[m]},marker:{size:7}};}),stages);
    const curve=(r,values)=>({type:'scatter',mode:'lines',name:label(r),x:coordinates(r,stage(r)),y:values,line:{color:colors[r.method],dash:dash[r.method],width:2}});
    const p=layout(`Saved pressure · ${cells()} cells`,'Pressure [bar]','Axial coordinate [m]');
    if($('profile-range').value==='shock'&&rr.length){const x=rr[0].reference_shock_m;p.xaxis.range=[x-.015,x+.015];}
    const pressure=rr.map(r=>curve(r,stage(r).state.slice(1,-1).map(w=>w[0]/1e5)));
    if(rr.length&&cells()===400)pressure.push({type:'scatter',mode:'lines',name:'Analytical cell projection',x:rr[0].coordinate,y:rr[0].reference_pressure.map(v=>v/1e5),line:{color:FluxReportUI.palette().ink,dash:'dot',width:1.4}});
    U.plot('pressure-plot',pressure,p);
    const v=layout(`Saved axial velocity · ${cells()} cells`,'Velocity [m/s]','Axial coordinate [m]');v.xaxis={...p.xaxis};
    U.plot('velocity-plot',rr.map(r=>curve(r,stage(r).state.slice(1,-1).map(w=>w[1]))),v);
    const mode=$('residual-mode').value,policy=D.metadata.settings.residual,history=[];
    for(const r of rr){const h=accepted(stage(r));if(mode==='equations')for(let k=0;k<3;k++)history.push({type:'scatter',mode:'lines',name:`${label(r)} · ${['mass','momentum','energy'][k]}`,
      x:h.map(v=>v.step),y:h.map(v=>Math.max(v.equation_l2[k],1e-30)),line:{color:colors[r.method],dash:['solid','dash','dot'][k],width:1.6}});
      else history.push({type:'scatter',mode:'lines',name:label(r),x:h.map(v=>v.step),y:h.map(v=>Math.max(mode==='ratio'?v.equation_max_ratio:mode==='global'?Math.max(...v.global_defect)/policy.global_tolerance:v.boundary_max/policy.boundary_tolerance,1e-30)),line:{color:colors[r.method],dash:dash[r.method]}});
    }
    const residual=layout('Original steady residual checks',mode==='equations'?'Per-equation fixed-scale L2 [−]':'Residual / allowed threshold [−]','Accepted update (initial state at zero)');residual.yaxis.type='log';
    if(mode!=='equations')residual.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:FluxReportUI.palette().muted,dash:'dot'}}];
    U.plot('residual-plot',history,residual);
    const cfl=layout('CFL and rejected timestep attempts','CFL [−]','Accepted-update counter');cfl.yaxis.type='log';
    const cf=[];for(const r of rr){const h=accepted(stage(r)),rejected=stage(r).history.filter(h=>h.phase==='rejected');
      cf.push({type:'scatter',mode:'lines',name:label(r),x:h.map(v=>v.step),y:h.map(v=>v.cfl),line:{color:colors[r.method],dash:dash[r.method]}});
      cf.push({type:'scatter',mode:'markers',name:names[r.method]+' rejected',x:rejected.map(v=>v.step),y:rejected.map(v=>v.cfl),marker:{color:colors[r.method],symbol:'x',size:7}});
    }U.plot('cfl-plot',cf,cfl);
    const w=layout('Selected-stage nonlinear work','Recorded count [−]','Work measure');w.barmode='group';
    U.plot('work-plot',rr.map(r=>({type:'bar',name:label(r),x:['Accepted','Rejected','Newton corrections','Newton trial rejections'],y:['accepted','rejected','corrections','backtracks'].map(k=>stage(r).work[k]),marker:{color:colors[r.method]}})),w);
    const l=layout('Final cell-step multipliers','Cell step / minimum step [−]','Physical cell index');
    U.plot('local-step-plot',rr.map(r=>({type:'scatter',mode:'lines',name:label(r),x:Array.from({length:cells()},(_,i)=>i+1),y:stage(r).final_cell_step_multipliers,line:{color:colors[r.method],dash:dash[r.method]}})),l);
    $('selected-rows').innerHTML=rr.map(r=>{const s=stage(r),t=g.timing[r.method];return `<tr><td>${E(label(r))}</td><td>${B(s.success,'Root','Failed')} ${B(s.audit.passed,'Audit','Audit miss')}</td><td>${F(s.initial_cfl)} / ${F(s.last_accepted_cfl)}</td><td>${s.work.accepted} / ${s.work.rejected}</td><td>${s.work.corrections}</td><td>${F(Math.max(...s.audit.equation_l2))}</td><td>${F(s.audit.boundary_max)} / ${F(Math.max(...s.audit.global_defect))}</td><td>${F(t.mean_s)} [${F(t.min_s)}–${F(t.max_s)}]</td></tr>`;}).join('');
  }
  function render(page){
    const rr=rows(),g=group();$('empty-selection').hidden=!!rr.length;
    $('selection-summary').innerHTML=`<strong>${caseId().endsWith('35')?'35%':'70%'} nozzle shock</strong> · ${E(recons[recon()])} · ${E(modes[strategy()])} · ${cells()}-cell stage · ${rr.length} selected methods <code>${E(caseId())}</code>`;
    $('case-heading').textContent=`${caseId().endsWith('35')?'35%':'70%'} shock · ${recons[recon()]} · ${cells()} cells`;
    $('selection-interpretation').innerHTML=`<h3>Interpret this selection</h3><p>Recorded mean times: global ${F(g.timing.global.mean_s)} s, local ${F(g.timing.local.mean_s)} s; ratio ${F(g.global_over_local)}×. ${g.eligible?'The saved original audits, shock check and same-root comparison pass.':'This comparison is ineligible for a speedup claim.'}</p><p>${$('sample').value==='median'?'Each method shows its own actual median-duration complete solve.':'The same numbered measured repetition is selected for both methods.'} ${rr.length?rr.map(r=>`${E(names[r.method])}: run ${r.repeat}`).join(' · '):'No method is selected; no replacement is substituted.'} Profiles and histories below use the selected mesh stage; timing samples retain the complete solve boundary.</p>`;
    $('selected-details').textContent=JSON.stringify(rr.map(r=>({id:r.id,phase:r.phase,repeat:r.repeat,input_file:r.input_file,scheme:r.scheme,elapsed_s:r.elapsed_s,total_work:r.total_work,stage:{...stage(r),history:undefined,state:undefined},source_head:D.metadata.source_head})),null,2);
    if(page==='overview')overview();
    if(page==='explorer')explorer(rr);
  }
  $('methods').innerHTML=D.methods.map(m=>`<label><input type="checkbox" value="${m}" checked><span class="dot" style="background:${colors[m]}"></span>${names[m]}</label>`).join('');
  $('methods').querySelectorAll('input').forEach(i=>i.onchange=()=>{i.checked?chosen.add(i.value):chosen.delete(i.value);controller.render();});
  for(const action of ['all','none'])$('methods-'+action).onclick=()=>{$('methods').querySelectorAll('input').forEach(i=>{i.checked=action==='all';i.checked?chosen.add(i.value):chosen.delete(i.value);});controller.render();};
  $('cells').innerHTML='<option value="400">400 physical cells</option>';syncCells();
  for(const id of ['case','strategy','reconstruction','cells','sample','profile-range','residual-mode'])$(id).onchange=()=>{if(id==='strategy')syncCells();controller.render();};
  $('provenance-details').textContent=JSON.stringify({metadata:D.metadata,presentation:D.presentation,cross_strategy:D.cross_strategy},null,2);
  controller=U.mount(render);
  window.studyReport={data:D,...controller,snapshot:()=>({...controller.snapshot(),caseId:caseId(),strategy:strategy(),reconstruction:recon(),cells:cells(),sample:$('sample').value,methods:[...chosen],rows:rows().map(r=>r.id)})};
  const hash=window.location?.hash?.slice(1);if(['overview','explorer','discussion','protocol'].includes(hash))controller.showPage(hash);
  if(window.location?.search?.includes('audit=1'))window.addEventListener('load',async()=>{
    const checks=[];const wait=async()=>{for(let i=0;i<600;i++){if(controller.snapshot().rendered===controller.snapshot().renderNumber)return;await new Promise(r=>setTimeout(r,20));}throw Error('Plot rendering timeout');};
    const change=async(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new Event('change'));await wait();};
    try{
      await wait();
      for(const page of ['overview','explorer','discussion','protocol']){controller.showPage(page);await wait();checks.push(!document.getElementById('page-'+page).hidden&&document.querySelectorAll('.report-page:not([hidden])').length===1);}
      controller.showPage('explorer');await wait();
      await change('strategy','single_400');checks.push($('cells').options.length===1&&cells()===400);
      await change('case','euler_nozzle/shock_70');await change('sample','2');
      checks.push(rows().every(r=>r.repeat===2)&&$('pressure-plot').data.length===3);
      await change('strategy','sequence');await change('cells',100);checks.push($('pressure-plot').data.length===2&&$('pressure-plot').data.every(t=>t.y.length===100));
      await change('residual-mode','equations');checks.push($('residual-plot').data.length===6);
      $('methods-none').click();await wait();checks.push(rows().length===0&&$('pressure-plot').data.length===0&&!$('empty-selection').hidden);$('methods-all').click();await wait();
      for(const theme of ['dark','light']){FluxReportUI.setTheme(theme);await wait();checks.push(document.documentElement.dataset.theme===theme&&$('pressure-plot').layout.font.color===FluxReportUI.palette().ink);}
      await change('case','euler_nozzle/shock_35');await change('reconstruction','muscl_van_leer');await change('cells',400);await change('sample','median');await change('residual-mode','ratio');await change('profile-range','shock');
      checks.push($('pressure-plot').layout.xaxis.range.length===2);
      await Plotly.relayout($('pressure-plot'),{'xaxis.range':[.125,.145]});checks.push($('pressure-plot').layout.xaxis.range[0]===.125);
      await change('profile-range','domain');
      $('tab-overview').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));await wait();checks.push($('tab-explorer').getAttribute('aria-selected')==='true');
      const a=[...document.querySelectorAll('a[download]')].find(a=>a.download==='plotted_data.json');checks.push(JSON.parse(atob(a.href.split(',')[1])).records.length===64);
      controller.showPage(hash==='explorer'?'explorer':'overview');await wait();
      const result={passed:checks.every(Boolean),checks,numerical_cases_rerun:false,viewport:{width:window.innerWidth,documentWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,deviceScale:window.devicePixelRatio}};const p=document.createElement('pre');p.id='browser-audit';p.textContent=JSON.stringify(result);document.body.append(p);
    }catch(error){const p=document.createElement('pre');p.id='browser-audit';p.textContent=JSON.stringify({passed:false,error:String(error),checks});document.body.append(p);}
  });
})();
