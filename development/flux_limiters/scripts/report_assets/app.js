(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('payload').textContent),U=StudyReportUI,{$,escape:E,fmt:F,badge:B,stat:S}=U;
  const names={van_leer:'Van Leer',minmod:'Minmod',superbee:'Superbee',van_albada:'Van Albada'};
  const color=id=>['#6a00a8','#b12a90','#e16462','#fca636'][D.methods.indexOf(id)];
  const dash=id=>['solid','dash','dashdot','dot'][D.methods.indexOf(id)];
  let cells=D.cells.at(-1),selected=new Set(D.methods),controller;
  const rows=()=>D.methods.filter(m=>selected.has(m)).map(m=>D.records.find(r=>r.limiter===m&&r.cells===cells)).filter(Boolean);
  const hist=r=>(r.history||[]).filter(h=>['initial','accepted'].includes(h.phase));
  const work=r=>(r.history||[]).reduce((s,h)=>s+(h.inner_iterations||0),0);
  function overview(){
    $('kpis').innerHTML=S('Physical cases',1,'Euler nozzle · shock at 35%')+S('Saved configurations',D.records.length,`${D.methods.length} limiters × ${D.cells.length} meshes`)+S('Recorded convergence',`${D.records.filter(r=>r.success).length} / ${D.records.length}`,'Original saved driver outcomes')+S('Cells',D.cells.join(' / '),'Coarse-grid sensitivity; not a timing study');
    $('case-cards').innerHTML=D.cells.map(n=>`<article class="case-card ${n===cells?'selected':''}"><p class="eyebrow">EULER NOZZLE / SHOCK 35%</p><h3>${n} physical cells</h3><p>Air, perfect gas · p0 = ${F(D.metadata.case.inlet.p0/1000)} kPa, T0 = ${F(D.metadata.case.inlet.T0)} K. ${D.records.filter(r=>r.cells===n&&r.success).length}/${D.methods.length} recorded roots.</p><button data-cells="${n}">Compare limiters</button></article>`).join('');
    $('case-cards').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{$('cells').value=b.dataset.cells;cells=Number(b.dataset.cells);controller.showPage('explorer');}));
    const finest=D.records.filter(r=>r.cells===D.cells.at(-1)&&r.success),best=finest.reduce((a,b)=>a.pressure_normalized_l1<b.pressure_normalized_l1?a:b),bee=finest.find(r=>r.limiter==='superbee');
    $('findings').innerHTML=`<p>All ${D.records.filter(r=>r.success).length} saved configurations report convergence. At ${D.cells.at(-1)} cells, ${E(names[best.limiter])} has the smallest recorded pressure L1/p0: ${F(best.pressure_normalized_l1)}.</p><p>Superbee at the finest saved mesh uses ${bee.accepted_steps} accepted and ${bee.rejected_steps} rejected steps, with pressure L1/p0 ${F(bee.pressure_normalized_l1)}. More compressive limiting did not give the lowest error or least nonlinear work here.</p><p>Pressure error decreases from ${D.cells[0]} to ${D.cells.at(-1)} cells for each limiter in these saved data. This two-level observation is not a formal convergence-order estimate.</p>`;
    const traces=D.methods.filter(m=>selected.has(m)).map(m=>{const rr=D.records.filter(r=>r.limiter===m).sort((a,b)=>a.cells-b.cells);return {type:'scatter',mode:'lines+markers',name:names[m],x:rr.map(r=>r.cells),y:rr.map(r=>r.pressure_normalized_l1),line:{color:color(m),dash:dash(m)},marker:{size:9,symbol:rr.map(r=>r.success?'circle':'circle-open')}};});
    const l=U.layout('Pressure error across the two saved meshes','Volume-weighted pressure L1 / p0');l.xaxis.title={text:'Physical cells'};l.xaxis.tickvals=D.cells;l.yaxis.type='log';U.plot('overview-errors',traces,l);
    $('all-rows').innerHTML=D.records.slice().sort((a,b)=>a.cells-b.cells||D.methods.indexOf(a.limiter)-D.methods.indexOf(b.limiter)).map(r=>`<tr><td>${E(names[r.limiter])}</td><td>${r.cells}</td><td>${B(r.success,'Converged',r.reason)}</td><td>${F(r.pressure_normalized_l1)}</td><td>${F(Math.max(...(r.conservative_normalized_l1||[])))}</td><td>${r.accepted_steps||0} / ${r.rejected_steps||0}</td></tr>`).join('');
  }
  function explorer(rr){
    const base=D.records.find(r=>r.cells===cells&&r.pressure),ink=FluxReportUI.palette().ink;
    const curve=(r,y)=>({type:'scatter',mode:'lines+markers',name:names[r.limiter]+(r.success?'':' · unfinished'),x:r.coordinate,y,line:{color:color(r.limiter),dash:dash(r.limiter),width:2},marker:{size:4}});
    const pressure=rr.filter(r=>r.pressure).map(r=>curve(r,r.pressure.map(v=>v/1e5)));
    if(rr.length){pressure.push({type:'scatter',mode:'lines',name:'Analytical point curve',x:base.analytical_coordinate,y:base.analytical_pressure.map(v=>v/1e5),line:{color:ink,width:1.4}});pressure.push({type:'scatter',mode:'markers',name:'Projected cell reference',x:base.coordinate,y:base.reference_pressure.map(v=>v/1e5),marker:{color:ink,symbol:'x',size:5}});}
    const p=U.layout('Pressure profiles','Pressure [bar]');p.xaxis.title={text:'Axial coordinate [m]'};
    if($('profile-range').value==='shock'&&base){const x=base.reference_shock_location_m;p.xaxis.range=[x-.025,x+.025];}
    U.plot('pressure-plot',pressure,p);
    const mach=U.layout('Numerical Mach profiles','Mach [−]');mach.xaxis={...p.xaxis};U.plot('mach-plot',rr.filter(r=>r.mach).map(r=>curve(r,r.mach)),mach);
    const mode=$('residual-mode').value,policy=D.metadata.settings.residual,traces=[];
    for(const r of rr){const h=hist(r);if(mode==='equations')for(let i=0;i<3;i++)traces.push({type:'scatter',mode:'lines',name:`${names[r.limiter]} · ${['mass','momentum','energy'][i]}`,x:h.map(v=>v.step),y:h.map(v=>Math.max(v.equation_l2?.[i]??v.equation_norm?.[i]??1e-30,1e-30)),line:{color:color(r.limiter),dash:['solid','dash','dot'][i],width:1.6}});
      else traces.push({type:'scatter',mode:'lines',name:names[r.limiter],x:h.map(v=>v.step),y:h.map(v=>Math.max(mode==='global'?Math.max(...v.global_defect)/policy.global_tolerance:mode==='boundary'?v.boundary_max/policy.boundary_tolerance:v.equation_max_ratio,1e-30)),line:{color:color(r.limiter),dash:dash(r.limiter),width:1.7}});}
    const residual=U.layout('Original residual checks',mode==='equations'?'Per-equation fixed-scale L2 [−]':'Residual / allowed threshold');residual.xaxis.title={text:'Accepted update (initial state at zero)'};residual.yaxis.type='log';if(mode!=='equations')residual.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:ink,dash:'dot'}}];U.plot('residual-plot',traces,residual);
    const cfl=U.layout('Accepted CFL history','CFL [−]');cfl.xaxis.title={text:'Accepted update'};cfl.yaxis.type='log';U.plot('cfl-plot',rr.map(r=>({type:'scatter',mode:'lines',name:names[r.limiter],x:hist(r).map(h=>h.step),y:hist(r).map(h=>h.cfl),line:{color:color(r.limiter),dash:dash(r.limiter)}})),cfl);
    const w=U.layout('Recorded nonlinear work','Count [−]');w.xaxis.title={text:'Work measure'};w.barmode='group';U.plot('work-plot',rr.map(r=>({type:'bar',name:names[r.limiter],x:['Accepted steps','Rejected steps','Newton corrections'],y:[r.accepted_steps||0,r.rejected_steps||0,work(r)],marker:{color:color(r.limiter)}})),w);
    const sh=U.layout('Saved shock-location error','Absolute position error [mm]');sh.xaxis.title={text:'Limiter'};U.plot('shock-plot',rr.length?[{type:'bar',showlegend:false,x:rr.map(r=>names[r.limiter]),y:rr.map(r=>r.shock_error_m*1000),marker:{color:rr.map(r=>color(r.limiter))},hovertemplate:'%{x}: %{y:.5g} mm<extra></extra>'}]:[],sh);
  }
  function render(page){
    const rr=rows();$('empty-selection').hidden=!!rr.length;$('case-heading').textContent=`Nozzle shock 35% · ${cells} cells`;
    $('selection-summary').innerHTML=`<strong>Euler nozzle / shock at 35%</strong> · ${cells} cells · ${rr.length} selected limiters <code>euler_nozzle/shock_35</code>`;
    const best=rr.filter(r=>r.success).sort((a,b)=>a.pressure_normalized_l1-b.pressure_normalized_l1)[0];
    $('selection-interpretation').innerHTML=`<h3>Interpret this selection</h3><p>${best?`${E(names[best.limiter])} has the smallest saved pressure L1/p0 among the selected converged runs: ${F(best.pressure_normalized_l1)}. The work plots retain accepted steps, rejected attempts and all recorded inner corrections separately.`:'No converged selected configuration is available.'}</p><p>All profiles are the actual saved states. Only limiter and mesh differ; no speedup or universal limiter ranking is inferred.</p>`;
    $('selected-details').textContent=JSON.stringify(rr.map(r=>({id:r.id,scheme:r.scheme,metrics:r.metrics,reason:r.reason,errors:{pressure:r.pressure_normalized_l1,conservative:r.conservative_normalized_l1},input_file:r.input_file})),null,2);
    if(page==='overview')overview();if(page==='explorer')explorer(rr);
  }
  $('cells').innerHTML=D.cells.map(n=>`<option value="${n}">${n} cells</option>`).join('');$('cells').value=String(cells);
  $('methods').innerHTML=D.methods.map(m=>`<label><input type="checkbox" value="${m}" checked><span class="dot" style="background:${color(m)}"></span>${names[m]}</label>`).join('');
  $('methods').querySelectorAll('input').forEach(i=>i.addEventListener('change',()=>{i.checked?selected.add(i.value):selected.delete(i.value);controller.render();}));
  for(const type of ['all','none'])$('methods-'+type).addEventListener('click',()=>{$('methods').querySelectorAll('input').forEach(i=>{i.checked=type==='all';i.checked?selected.add(i.value):selected.delete(i.value);});controller.render();});
  for(const id of ['case','cells','profile-range','residual-mode'])$(id).addEventListener('change',()=>{cells=Number($('cells').value);controller.render();});
  $('provenance-details').textContent=JSON.stringify({metadata:D.metadata,presentation:D.presentation},null,2);
  controller=U.mount(render);window.studyReport={data:D,...controller,snapshot:()=>({...controller.snapshot(),cells,methods:[...selected],rows:rows().map(r=>r.id)})};
})();
