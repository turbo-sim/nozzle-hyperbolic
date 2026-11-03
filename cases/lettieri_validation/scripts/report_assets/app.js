(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('payload').textContent),U=StudyReportUI,{$,escape:E,fmt:F,badge:B,stat:S}=U;
  let caseId=D.cases[0].id,cells='96',selected=new Set(),controller,comparison;
  const info=()=>D.cases.find(c=>c.id===caseId);
  const available=()=>D.records.filter(r=>r.case_id===caseId&&(cells==='all'||r.cells===Number(cells))&&($('attempt-scope').value==='all'||r.property_domain_audited)&&($('grid-kind').value==='all'||($('grid-kind').value==='uniform'?r.stretch===1:r.stretch!==1)));
  const rows=()=>available().filter(r=>selected.has(r.id));
  const title=r=>`${r.case_id} · N=${r.cells}${r.stretch===1?'':` · stretch ${r.stretch}`}${r.property_domain_audited?'':' · preflight'}`;
  const color=r=>r.cells<=24?'#6a00a8':r.cells<=48?'#cc4778':'#fca636';
  const dash=r=>!r.property_domain_audited?'dot':r.stretch!==1?'dashdot':'solid';
  const finest=id=>D.records.filter(r=>r.case_id===id&&r.property_domain_audited&&r.stretch===1).sort((a,b)=>b.cells-a.cells)[0];
  const accepted=r=>(r.history||[]).filter(h=>['initial','accepted'].includes(h.phase));
  function configure(reset=true){
    const levels=[...new Set(D.records.filter(r=>r.case_id===caseId).map(r=>r.cells))].sort((a,b)=>a-b);
    if(cells!=='all'&&!levels.includes(Number(cells)))cells=String(levels.at(-1));
    $('cells').innerHTML='<option value="all">All saved meshes</option>'+levels.map(n=>`<option value="${n}">${n} cells</option>`).join('');$('cells').value=cells;
    if(reset)selected=new Set(available().map(r=>r.id));
    $('runs').innerHTML=available().map(r=>`<label><input type="checkbox" value="${E(r.id)}" ${selected.has(r.id)?'checked':''}><span class="dot" style="background:${color(r)}"></span>${E(title(r))}</label>`).join('');
    $('runs').querySelectorAll('input').forEach(i=>i.addEventListener('change',()=>{i.checked?selected.add(i.value):selected.delete(i.value);controller.render();}));
    $('show-cfd').disabled=!info().upstream_cfd;
  }
  function experimentTrace(c,showlegend=true){return {type:'scatter',mode:'markers',name:c.experiment.origin.includes('approximate')?'Experiment · approximate tap x':'Experiment · digitized x',x:c.experiment.x_m.map(v=>v*1000),y:c.experiment.pressure_pa.map(v=>v/1e5),marker:{color:FluxReportUI.palette().ink,size:6,symbol:'square'},showlegend};}
  function referenceTrace(c,field,showlegend=true){
    const points=c.upstream_cfd?.[field];if(!points)return null;
    return {type:'scatter',mode:'lines',name:'Supplied reference CFD',x:points.map(p=>p[0]*1000),y:points.map(p=>field==='pressure'?p[1]/1e5:p[1]),line:{color:'#8794a3',dash:'dash',width:1.7},showlegend};
  }
  function fieldTrace(r,key,scale=1){return {type:'scatter',mode:'lines',name:title(r),x:(r.fields.x_m||[]).map(v=>v*1000),y:(key==='displayed_liquid_fraction'?r.displayed_liquid_fraction:r.fields[key]||[]).map(v=>v==null?null:v*scale),connectgaps:false,line:{color:color(r),dash:dash(r),width:2}};}
  function profileLayout(name,y){const l=U.layout(name,y);l.xaxis.title={text:'Position relative to throat [mm]'};l.xaxis.range=[-32.013,66.355];if($('show-onset').checked)l.shapes=[{type:'line',x0:info().definition.onset_mm,x1:info().definition.onset_mm,yref:'paper',y0:0,y1:1,line:{color:FluxReportUI.palette().ink,dash:'dot',width:1}}];return l;}
  function overview(){
    const finals=D.cases.map(c=>finest(c.id)),verified=finals.filter(r=>r.saved_verification?.verified).length,errors=finals.map(r=>r.comparison.pressure_rmse_bar);
    $('kpis').innerHTML=S('Operating cases',D.cases.length,'13 measured taps per case')+S('Retained attempts',D.records.length,`${D.records.filter(r=>!r.property_domain_audited).length} domain-unaudited preflight`)+S('Finest saved audits',`${verified} / ${finals.length}`,'Original equations and physical checks')+S('Finest pressure RMSE',`${F(Math.min(...errors))}–${F(Math.max(...errors))}`,'bar · agreement with experiment');
    $('case-cards').innerHTML=D.cases.map(c=>{const r=finest(c.id);return `<article class="case-card ${caseId===c.id?'selected':''}"><p class="eyebrow">LETTIERI / CASE ${E(c.id)}</p><h3>${E(c.id)} · ${F(c.definition.p0_bar)} bar</h3><p>T0 = ${F(c.definition.T0_K)} K<br>Measured onset = ${F(c.definition.onset_mm)} mm<br>Finest saved grid: ${r.cells} cells<br>Pressure RMSE: ${F(r.comparison.pressure_rmse_bar)} bar</p><p>${c.upstream_cfd?'Reference CFD supplied':'No reference CFD supplied'}</p><button data-case="${c.id}">Explore ${c.id}</button></article>`;}).join('');
    $('case-cards').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectCase(b.dataset.case,'explorer')));
    const l3=D.records.filter(r=>r.case_id==='L3'&&r.property_domain_audited&&r.stretch===1).sort((a,b)=>a.cells-b.cells);
    $('findings').innerHTML=`<p>${verified}/${finals.length} finest property-audited uniform solutions pass their saved residual and physical-budget audits. Their pressure RMSE spans ${F(Math.min(...errors))}–${F(Math.max(...errors))} bar.</p><p>Refinement is not uniformly beneficial for the experimental score: L3 changes from ${F(l3[0].comparison.pressure_rmse_bar)} bar at ${l3[0].cells} cells to ${F(l3.at(-1).comparison.pressure_rmse_bar)} bar at ${l3.at(-1).cells} cells. A smaller equation residual is not an experimental validation score.</p><p>L2–L4 use approximate nominal pressure-tap positions. CFD pressure/liquid-fraction references exist only for L1 and L5; unavailable series are never substituted.</p>`;
    const gallery=[],layout=U.layout('Five operating points · finest audited uniform HEM','Pressure [bar]');layout.height=1300;layout.margin={l:80,r:35,t:75,b:110};layout.grid={rows:5,columns:1,pattern:'independent',ygap:.16};layout.annotations=[];layout.shapes=[];
    D.cases.forEach((c,i)=>{const suffix=i?String(i+1):'',r=finest(c.id),traces=[fieldTrace(r,'p',1e-5),experimentTrace(c,i===0)];traces[0].name='HEM · finest uniform';traces[0].showlegend=i===0;const cf=referenceTrace(c,'pressure',i===0);if(cf)traces.push(cf);for(const t of traces){t.xaxis='x'+suffix;t.yaxis='y'+suffix;gallery.push(t);}layout['xaxis'+suffix]={title:{text:i===4?'Position relative to throat [mm]':''},range:[-32.013,66.355],gridcolor:FluxReportUI.palette().grid};layout['yaxis'+suffix]={title:{text:'Pressure [bar]'},gridcolor:FluxReportUI.palette().grid,automargin:true};layout.annotations.push({xref:'paper',yref:'paper',x:0,y:1-i*.2,text:`<b>${c.id}</b> · ${F(c.definition.p0_bar)} bar, ${F(c.definition.T0_K)} K`,showarrow:false,xanchor:'left',yanchor:'bottom',font:{size:12}});if($('show-onset').checked)layout.shapes.push({type:'line',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:c.definition.onset_mm,x1:c.definition.onset_mm,y0:0,y1:1,line:{color:FluxReportUI.palette().ink,dash:'dot',width:1}});});
    U.plot('gallery-plot',gallery,layout);
    $('all-rows').innerHTML=D.records.map(r=>`<tr><td>${E(r.id)}</td><td>${r.cells}</td><td>${r.stretch}</td><td>${B(r.success,'Converged',r.reason)}</td><td>${B(r.property_domain_audited,'Audited','Preflight')}</td><td>${r.saved_verification?B(r.saved_verification.verified):'Not supplied'}</td><td>${F(r.comparison?.pressure_rmse_bar)}</td></tr>`).join('');
  }
  function explorer(rr){
    const c=info();$('reference-notice').textContent=c.experiment.origin+'. '+(c.upstream_cfd?'Supplied reference CFD is available for this case.':'No reference CFD or liquid-fraction reference is available for this case.');
    const pressure=rr.filter(r=>r.fields.p).map(r=>fieldTrace(r,'p',1e-5));if(rr.length){pressure.push(experimentTrace(c));if($('show-cfd').checked){const cf=referenceTrace(c,'pressure');if(cf)pressure.push(cf);}}
    U.plot('pressure-plot',pressure,profileLayout('Pressure versus experiment','Pressure [bar]'));
    const quality=rr.filter(r=>r.fields.quality).map(r=>fieldTrace(r,'displayed_liquid_fraction'));if(rr.length&&$('show-cfd').checked){const cf=referenceTrace(c,'liquid_fraction');if(cf)quality.push(cf);}
    U.plot('quality-plot',quality,profileLayout('Equilibrium liquid mass fraction','Liquid mass fraction [−]'));
    const ink=FluxReportUI.palette().ink,residual=[];
    for(const r of rr)for(let i=0;i<3;i++){const h=accepted(r);residual.push({type:'scatter',mode:'lines',name:`N=${r.cells}${r.stretch===1?'':` stretch ${r.stretch}`} · ${['mass','momentum','energy'][i]}`,x:h.map(v=>v.step),y:h.map(v=>Math.max(v.equation_l2?.[i]??v.equation_norm?.[i]??1e-30,1e-30)),line:{color:color(r),dash:['solid','dash','dot'][i],width:1.6}});}
    const l=U.layout('Per-equation convergence','Fixed-scale L2 [−]');l.xaxis.title={text:'Accepted update (initial state at zero)'};l.yaxis.type='log';l.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1e-9,y1:1e-9,line:{color:ink,dash:'dash'}}];U.plot('residual-plot',residual,l);
    const cfl=U.layout('Accepted CFL history','CFL [−]');cfl.xaxis.title={text:'Accepted update'};cfl.yaxis.type='log';U.plot('cfl-plot',rr.map(r=>({type:'scatter',mode:'lines',name:title(r),x:accepted(r).map(v=>v.step),y:accepted(r).map(v=>v.cfl),line:{color:color(r),dash:dash(r)}})),cfl);
    const flow=rr.filter(r=>r.fields.mdot).map(r=>fieldTrace(r,'mdot'));
    for(const r of rr){const v=r.saved_verification?.mass_flow?.numerical_inlet_kg_s;if(v!=null)flow.push({type:'scatter',mode:'lines',name:`Numerical face flow · N=${r.cells}`,x:[-32.013,66.355],y:[v,v],line:{color:color(r),dash:'dash',width:1}});}
    U.plot('massflow-plot',flow,profileLayout('Cell diagnostic versus numerical face flow','Full-channel mass flow [kg/s]'));
    const refinement=[],all=D.records.filter(r=>r.case_id===caseId&&r.property_domain_audited);
    for(const stretch of [...new Set(all.map(r=>r.stretch))]){const group=all.filter(r=>r.stretch===stretch).sort((a,b)=>a.cells-b.cells);refinement.push({type:'scatter',mode:'lines+markers',name:stretch===1?'Uniform meshes':`Stretch ${stretch}`,x:group.map(r=>r.cells),y:group.map(r=>r.comparison.pressure_rmse_bar),line:{color:stretch===1?'#6a00a8':'#ed7953'},marker:{size:9,symbol:group.map(r=>r.success?'circle':'circle-open')}});}
    const mesh=U.layout('Mesh sensitivity · this operating point','Pressure RMSE versus experiment [bar]');mesh.xaxis.title={text:'Physical cells'};mesh.xaxis.tickvals=[24,48,96];U.plot('mesh-plot',refinement,mesh);
    $('selected-rows').innerHTML=rr.map(r=>{const v=r.saved_verification,m=v||r.metrics,eq=v?.equation_l2||r.metrics?.equation_norm||[];return `<tr><td>${E(title(r))}</td><td>${eq.map(F).join(' / ')}</td><td>${F(m?.boundary_max)} / ${F(Math.max(...(m?.global_defect||[])))}</td><td>${v?B(v.verified):'Not supplied'}</td><td>${F(r.comparison?.pressure_rmse_bar)}</td><td>${r.quality_diagnostic.masked_samples}/${r.quality_diagnostic.samples}</td><td>${F(v?.mass_flow?.numerical_inlet_kg_s)}</td></tr>`;}).join('');
  }
  function render(page){const rr=rows(),c=info();comparison?.activate(page);$('empty-selection').hidden=!!rr.length;$('case-heading').textContent=`${caseId} · ${F(c.definition.p0_bar)} bar, ${F(c.definition.T0_K)} K`;
    $('selection-summary').innerHTML=`<strong>${E(caseId)}</strong> · HLL / constant · ${cells==='all'?'all saved meshes':cells+' cells'} · ${E($('grid-kind').value)} grids · ${rr.length} selected attempts <code>${$('attempt-scope').value==='all'?'preflight included':'property-domain audited scope'}</code>`;
    const statements=rr.map(r=>`${title(r)}: ${r.success?'recorded convergence':'unfinished '+r.reason}; saved audit ${r.saved_verification?.verified?'passes':'not established'}; pressure RMSE ${F(r.comparison?.pressure_rmse_bar)} bar; ${r.quality_diagnostic.masked_samples} raw-quality sample(s) masked for the liquid-fraction plot.`);
    $('selection-interpretation').innerHTML='<h3>Interpret this selection</h3>'+statements.map(s=>`<p>${E(s)}</p>`).join('')+'<p>Residual and physical-budget checks characterize the numerical equations. Pressure RMSE measures agreement with the experiment. Their meanings are distinct; the measured onset marker is not a computed equilibrium onset.</p>';
    $('selected-details').textContent=JSON.stringify(rr.map(r=>({id:r.id,success:r.success,reason:r.reason,property_domain_audited:!!r.property_domain_audited,comparison:r.comparison,quality_diagnostic:r.quality_diagnostic,scheme:r.scheme,steady:r.steady,initialization:r.initialization,source:r.source,saved_verification:r.saved_verification})),null,2);
    if(page==='overview')overview();if(page==='explorer')explorer(rr);if(page==='comparison')comparison?.render(caseId);
  }
  function selectCase(id,page){caseId=id;$('case').value=id;configure(true);controller.showPage(page||'explorer');}
  $('case').innerHTML=D.cases.map(c=>`<option value="${c.id}">${c.id} · ${F(c.definition.p0_bar)} bar / ${F(c.definition.T0_K)} K</option>`).join('');
  configure(true);
  $('case').addEventListener('change',()=>{caseId=$('case').value;configure(true);controller.render();});
  $('cells').addEventListener('change',()=>{cells=$('cells').value;configure(true);controller.render();});
  for(const id of ['grid-kind','attempt-scope'])$(id).addEventListener('change',()=>{configure(true);controller.render();});
  for(const id of ['show-cfd','show-onset'])$(id).addEventListener('change',()=>controller.render());
  for(const op of ['all','none'])$('runs-'+op).addEventListener('click',()=>{selected=new Set(op==='all'?available().map(r=>r.id):[]);configure(false);controller.render();});
  $('provenance-details').textContent=JSON.stringify({publication:D.publication,settings:D.settings,presentation:D.presentation},null,2);
  comparison=window.LettieriComparison?.install(D,U,{info,experimentTrace,referenceTrace,redraw:()=>controller.render()});
  controller=U.mount(render);window.studyReport={data:D,...controller,selectCase,comparison,
    snapshot:()=>({...controller.snapshot(),caseId,cells,rows:rows().map(r=>r.id),comparison:comparison?.snapshot()||null})};
  if(comparison?.available())controller.showPage('comparison');
})();
