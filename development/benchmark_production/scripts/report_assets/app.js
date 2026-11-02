/* Offline saved-data explorer. No flow solver, network or measurement calls. */
(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('payload').textContent);
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(Number(value).toPrecision(4)).toString();
  const mean = a => a.length ? a.reduce((s,v) => s+v,0)/a.length : 0;
  const method = id => DATA.methods.find(m => m.id === id);
  const modelName = id => id === 'euler_nozzle' ? 'Euler nozzle' : 'Euler annular';
  const reconName = id => id === 'constant' ? 'Constant' : 'MUSCL · Van Leer';
  const shortCase = c => c.id.includes('shock') ? `Shock ${c.id.includes('35') ? '35%' : '70%'}` : c.model === 'euler_nozzle' ? 'Adapted expansion' : 'Supersonic expansion';
  const badge = (value, yes='Pass', no='Miss') => `<span class="badge ${value ? 'pass' : 'warn'}">${escape(value ? yes : no)}</span>`;
  const label = id => ({single_stadtke:'Vanilla Stadtke · single',multigrid_stadtke:'Vanilla Stadtke · sequence',multigrid_stadtke_modified:'Modified Stadtke · sequence',multigrid_hll:'HLL · sequence',multigrid_rusanov:'Rusanov · sequence'}[id] || id);
  const rowFor = (caseId,recon,id) => DATA.rows.find(r => r.case_id === caseId && r.reconstruction === recon && r.method === id);
  const state = {model:DATA.cases[0].model,caseId:DATA.cases[0].id,reconstruction:'constant',metric:'correction',
    methods:new Set(DATA.methods.map(m => m.id)),page:'overview',renderNumber:0,rendered:0};
  const metricInfo = () => ({key:state.metric === 'correction' ? 'fixed20' : state.metric,
    factor:state.metric === 'correction' ? 50 : 1000,
    title:({correction:'Checked Newton correction',residual:'Residual evaluation',jacobian_blocks:'Jacobian assembly',linear_solve:'Preassembled linear solve',direction:'Complete Newton direction'})[state.metric]});
  const currentCase = () => DATA.cases.find(c => c.id === state.caseId);
  const selectedRows = () => DATA.methods.filter(m => state.methods.has(m.id)).map(m => rowFor(state.caseId,state.reconstruction,m.id)).filter(Boolean);
  const tasks = [];
  let tabs;

  function layout(title,axis='Time [ms]') {
    const base = FluxReportUI.layout(title,axis);
    base.margin = {l:65,r:25,t:65,b:110};
    base.legend = {orientation:'h',x:0,y:-.22,font:{size:10}};
    return base;
  }
  function plot(id,traces,options={}) {
    const spec = Object.assign(layout(''), options);
    if (!traces.length) spec.annotations=[{text:'Select at least one method',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
    const work = Plotly.react($(id),traces,spec,{responsive:true,displaylogo:false,scrollZoom:false,
      modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'svg',filename:id}});
    tasks.push(Promise.resolve(work));
    return work;
  }
  function fillCases() {
    const choices = DATA.cases.filter(c => c.model === state.model);
    if (!choices.some(c => c.id === state.caseId)) state.caseId=choices[0].id;
    $('case').innerHTML=choices.map(c => `<option value="${escape(c.id)}">${escape(shortCase(c))}</option>`).join('');
    $('case').value=state.caseId;
  }
  function stat(title,value,note) {return `<article class="stat"><span>${escape(title)}</span><strong>${escape(value)}</strong><small>${escape(note)}</small></article>`;}
  function common(rows) {
    const c=currentCase();
    $('selection-summary').innerHTML=`<strong>${escape(modelName(c.model))} / ${escape(shortCase(c))}</strong><span>· ${escape(reconName(state.reconstruction))} · 400 cells</span><code>${escape(c.id)}</code><span class="scope">${rows.length} methods selected · explorer scope</span>`;
    $('empty-selection').hidden=rows.length>0;
    $('timing-heading').textContent=`${modelName(c.model)} · ${shortCase(c)} · ${reconName(state.reconstruction)}`;
    $('fields-heading').textContent=$('timing-heading').textContent;
    $('physical-case-details').textContent=JSON.stringify(c.physical,null,2);
    $('case-kpis').innerHTML=stat('Methods in view',rows.length,'Same physical case and reconstruction')+
      stat('Measured complete solves',rows.reduce((s,r)=>s+r.samples.length,0),c.model==='euler_nozzle'?'10 per selected method':'3 per selected method')+
      stat('Original analytical targets',`${rows.filter(r=>r.assessment.analytical_passed).length} / ${rows.length}`,'Convergence is a separate check')+
      stat('Kernel calls per operation',DATA.protocol.kernel_repeats,'Each correction call contains 20 corrections');
    $('timing-rows').innerHTML=rows.map(r=>{
      const a=r.assessment,w=r.representative_stages.at(-1).work;
      return `<tr><td><span class="dot" style="background:${method(r.method).color}"></span>${escape(label(r.method))}</td><td>${badge(r.success)}</td><td>${badge(a.branch_passed)}</td><td>${badge(a.analytical_passed)}</td><td>${badge(r.all_stages_success)}</td><td>${r.samples.length}</td><td>${fmt(r.timing.mean_s)}</td><td>${w.accepted_steps} / ${w.rejected_steps}</td><td>${w.newton_corrections}</td></tr>`;
    }).join('');
    $('selected-details').textContent=JSON.stringify(rows.map(r=>({id:r.id,method:method(r.method),assessment:r.assessment,
      kernel_metadata:r.kernel_metadata,representative_file:r.representative_file,stages:r.representative_stages,repeat_consistency:r.repeat_consistency})),null,2);
    const notices=rows.filter(r=>!r.assessment.analytical_passed).map(r=>{
      const a=r.assessment,failed=Object.keys(a.accuracy_checks).filter(k=>!a.accuracy_checks[k]);
      return `${label(r.method)} misses: ${failed.join(', ')}.`;
    });
    const comparison=DATA.comparisons.find(c=>c.case===state.caseId && c.reconstruction===state.reconstruction);
    if (comparison && !comparison.comparison_field_agreement_passed && state.methods.has('single_stadtke') && state.methods.has('multigrid_stadtke'))
      notices.push(`Both vanilla initializations pass original checks, but their fine states differ locally by up to ${fmt(comparison.maximum_field_scaled_state_deviation)} of a fixed field scale. Treat the total-time ratio as descriptive.`);
    $('accuracy-notice').textContent=notices.join(' ');
    $('accuracy-notice').hidden=!notices.length;
    const aRows=rows.map(r=>{
      const a=r.assessment,e=a.errors;
      const field=Math.max(...['p','rho','velocity','M','T'].filter(k=>e[k]).map(k=>e[k].normalized_l1));
      const bound=(v,t)=>`${fmt(v)}${t==null?'':` / target ${fmt(t)}`}`;
      const checks=Object.entries(a.accuracy_checks).map(([k,v])=>`${escape(k)}: ${badge(v)}`).join('<br>');
      return `<tr><td>${escape(label(r.method))}</td><td>${bound(field,a.accuracy_targets.normalized_field_l1)}</td><td>${bound(a.metrics.max_conservative_normalized_l1,a.accuracy_targets.conservative_normalized_l1)}</td><td>${bound(e.Cp?.l1,a.accuracy_targets.Cp_length_mean_absolute)}</td><td>${bound(a.shock.error_cells,a.shock.expected==null?null:a.accuracy_targets.shock_position_cells)}</td><td>${checks}</td></tr>`;
    });
    $('accuracy-rows').innerHTML=aRows.join('');
  }
  function caseInterpretation(rows) {
    const p=[];
    const c=DATA.comparisons.find(c=>c.case===state.caseId && c.reconstruction===state.reconstruction);
    if (c && state.methods.has('single_stadtke') && state.methods.has('multigrid_stadtke')) {
      const direction=c.sequence_over_single_total<1?'lower':'higher';
      p.push(`With vanilla Stadtke, sequencing gives ${fmt(c.sequence_mean_s)} s versus ${fmt(c.single_mean_s)} s on the single grid: a ${direction} total (${fmt(c.sequence_over_single_total)}×). Its fine-grid cost per correction is ${fmt(c.sequence_over_single_correction)}× the single-grid cost. The large total-time differences come from the convergence path, not a smaller final linear system.`);
      if(!c.comparison_field_agreement_passed)p.push(`State-agreement caveat: the largest local difference is ${fmt(c.maximum_field_scaled_state_deviation)} of a fixed field scale. This total-time ratio is not a verified time-to-identical-state speedup.`);
      else p.push(`The vanilla fine states agree within ${fmt(c.maximum_field_scaled_state_deviation)} of fixed field scales. Original analytical checks pass for both: ${c.original_checks_passed?'yes':'no'}.`);
    }
    const base=rowFor(state.caseId,state.reconstruction,'multigrid_stadtke');
    if (base) for(const r of rows.filter(r=>['multigrid_hll','multigrid_rusanov','multigrid_stadtke_modified'].includes(r.method))) {
      const ratio=r.kernel_timing.fixed20.mean_s/base.kernel_timing.fixed20.mean_s;
      p.push(`${label(r.method)} costs ${fmt(ratio)}× vanilla sequenced Stadtke per checked correction. Its original analytical targets ${r.assessment.analytical_passed?'pass':'are missed'}; kernel cost alone is not an accuracy recommendation.`);
    }
    if (!p.length)p.push('Select the vanilla single-grid and sequenced methods to compare initialization cost, or add flux variants to inspect their operation costs.');
    $('case-interpretation').innerHTML='<h3>Interpret this selection</h3>'+p.map(t=>`<p>${escape(t)}</p>`).join('');
  }
  function renderOverview() {
    const c=DATA.counts;
    $('campaign-kpis').innerHTML=stat('Physical operating cases',DATA.cases.length,'3 nozzle + 3 annular')+stat('Saved configurations',c.configurations,'5 methods × 2 reconstructions × 6 cases')+
      stat('Measured complete solves',c.measured_solves,`${c.excluded_solves} excluded complete solves`)+stat('Analytical targets passed',`${c.accurate} / ${c.configurations}`,`${c.converged} converged fine-grid solves`);
    $('case-cards').innerHTML=DATA.cases.map(c=>{
      const groups=['constant','muscl_van_leer'].map(recon=>{
        const rows=DATA.rows.filter(r=>r.case_id===c.id && r.reconstruction===recon);
        return `<button type="button" data-case="${escape(c.id)}" data-recon="${recon}">${escape(reconName(recon))} · accuracy ${rows.filter(r=>r.assessment.analytical_passed).length}/${rows.length}</button>`;
      }).join('');
      return `<article class="case-card ${c.id===state.caseId?'selected':''}"><p class="eyebrow">${escape(modelName(c.model))}</p><h3>${escape(shortCase(c))}</h3><p>${escape(c.label)}. ${escape(c.physical.fluid.name)} · ${escape(c.physical.fluid.backend.replaceAll('_',' '))}. Reservoir ${fmt(c.physical.inlet.p0/1000)} kPa, ${fmt(c.physical.inlet.T0)} K.</p><code>${escape(c.id)}</code><div class="case-actions">${groups}</div></article>`;
    }).join('');
    $('case-cards').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectCase(b.dataset.case,b.dataset.recon,'timing')));
    $('global-findings').innerHTML=DATA.findings.map(t=>`<p>${escape(t)}</p>`).join('');
    const m=metricInfo(),methods=DATA.methods.filter(v=>state.methods.has(v.id)),pairs=DATA.cases.flatMap(c=>['constant','muscl_van_leer'].map(recon=>({c,recon})));
    const z=pairs.map(({c,recon})=>methods.map(method=>rowFor(c.id,recon,method.id)?.kernel_timing[m.key]?.mean_s*m.factor));
    const hover=pairs.map(({c,recon})=>methods.map(method=>{
      const r=rowFor(c.id,recon,method.id),t=r?.kernel_timing[m.key];
      return `${modelName(c.model)} / ${shortCase(c)}<br>${reconName(recon)}<br>${label(method.id)}<br>Mean ${fmt(t?.mean_s*m.factor)} ms; min ${fmt(t?.min_s*m.factor)}; max ${fmt(t?.max_s*m.factor)}<br>n=${t?.samples_s.length}; analytical ${r?.assessment.analytical_passed?'pass':'miss'}`;
    }));
    const trace={type:'heatmap',x:methods.map(m=>label(m.id)),y:pairs.map(({c,recon})=>`${modelName(c.model)} · ${shortCase(c)} · ${reconName(recon)}`),z,
      text:z.map(a=>a.map(v=>fmt(v))),texttemplate:'%{text}',hovertext:hover,hoverinfo:'text',colorscale:[[0,'#f7f1fb'],[.45,'#cc4778'],[1,'#6a00a8']],colorbar:{title:{text:'ms'},thickness:12},xgap:3,ygap:3};
    const drawn=plot('coverage-plot',methods.length?[trace]:[],Object.assign(layout(m.title+' · all cases'),{margin:{l:270,r:65,t:65,b:90},xaxis:{side:'bottom',tickfont:{size:10},automargin:true},yaxis:{autorange:'reversed',tickfont:{size:11},automargin:true}}));
    const el=$('coverage-plot');
    Promise.resolve(drawn).then(()=>{if(typeof el.on==='function' && !el._caseClickBound){el.on('plotly_click',e=>{const index=e.points[0].pointIndex?.[0] ?? e.points[0].pointNumber?.[0];if(Number.isInteger(index))selectCase(pairs[index].c.id,pairs[index].recon,'timing');});el._caseClickBound=true;}});
  }
  function renderTiming(rows) {
    const m=metricInfo(),names=rows.map(r=>label(r.method)),palette=FluxReportUI.palette();
    const horizontal=title=>Object.assign(layout(title),{margin:{l:185,r:55,t:65,b:65},xaxis:{title:{text:'Time [ms]'},gridcolor:palette.grid,zeroline:false},yaxis:{autorange:'reversed',automargin:true,tickfont:{size:11}}});
    const operations=rows.map(r=>{
      const t=r.kernel_timing[m.key],scale=m.factor;
      return {type:'bar',orientation:'h',name:label(r.method),x:[t.mean_s*scale],y:[label(r.method)],marker:{color:method(r.method).color},showlegend:false,
        error_x:{type:'data',symmetric:false,array:[(t.max_s-t.mean_s)*scale],arrayminus:[(t.mean_s-t.min_s)*scale],color:palette.ink,thickness:1.5},
        customdata:[[t.min_s*scale,t.max_s*scale,t.samples_s.length,r.assessment.analytical_passed?'pass':'miss']],
        hovertemplate:'%{y}<br>Mean %{x:.5g} ms<br>Min %{customdata[0]:.5g}; max %{customdata[1]:.5g} ms<br>n=%{customdata[2]}; analytical %{customdata[3]}<extra></extra>'};
    });
    plot('operation-plot',operations,horizontal(m.title));
    const spread=rows.length?Math.max(...rows.map(r=>r.timing.mean_s))/Math.min(...rows.map(r=>r.timing.mean_s)):1;
    const logarithmic=spread>15;
    const totalSpec=horizontal('Complete solve'+(logarithmic?' · logarithmic axis':''));
    totalSpec.xaxis={title:{text:'Full hierarchy / single-grid time [s]'},type:logarithmic?'log':'linear',gridcolor:palette.grid,zeroline:false};
    plot('total-plot',rows.map(r=>({type:'scatter',mode:'markers',name:label(r.method),y:[label(r.method)],x:[r.timing.mean_s],showlegend:false,
      marker:{color:method(r.method).color,size:13,symbol:r.assessment.analytical_passed?'circle':'diamond-open',line:{width:2}},
      error_x:{type:'data',symmetric:false,array:[r.timing.max_s-r.timing.mean_s],arrayminus:[r.timing.mean_s-r.timing.min_s],color:method(r.method).color},
      customdata:[[r.timing.min_s,r.timing.max_s,r.samples.length,r.assessment.analytical_passed?'pass':'miss']],
      hovertemplate:'%{y}<br>Mean %{x:.5g} s<br>Min %{customdata[0]:.5g}; max %{customdata[1]:.5g} s<br>n=%{customdata[2]}; analytical %{customdata[3]}<extra></extra>'})),totalSpec);
    const sampleSpec=layout(m.title+' · measured samples','Time [ms]');sampleSpec.xaxis.title={text:'Recorded sample within each method'};
    plot('samples-plot',rows.map(r=>({type:'scatter',mode:'lines+markers',name:label(r.method),x:r.kernel_timing[m.key].samples_s.map((_,i)=>i+1),
      y:r.kernel_timing[m.key].samples_s.map(v=>v*m.factor),line:{color:method(r.method).color,width:1.5},marker:{size:5},hovertemplate:'Sample %{x}: %{y:.5g} ms<extra>%{fullData.name}</extra>'})),sampleSpec);
    const stageColors=['#d8bfdc','#b780bd','#8e4a9a','#6a237f'];
    const stages=[50,100,200,400].map((n,i)=>({type:'bar',orientation:'h',name:`${n} cells`,y:names,
      x:rows.map(r=>mean(r.stage_timings.find(s=>s.cells===n)?.samples_s||[])),marker:{color:stageColors[i]},hovertemplate:'%{y}<br>'+n+' cells: %{x:.5g} s<extra></extra>'}));
    stages.push({type:'bar',orientation:'h',name:'Other hierarchy overhead',y:names,x:rows.map(r=>Math.max(0,r.timing.mean_s-r.stage_timings.reduce((s,t)=>s+mean(t.samples_s),0))),marker:{color:'#929fae'}});
    const stageSpec=horizontal('Where hierarchy time is spent');stageSpec.margin.b=100;stageSpec.barmode='stack';stageSpec.xaxis.title={text:'Mean time [s] · segment colors identify grids'};stageSpec.legend={orientation:'h',x:0,y:-.22,font:{size:10}};
    plot('stages-plot',rows.length?stages:[],stageSpec);
  }
  function renderFields(rows) {
    const ref=DATA.references[state.caseId],p=FluxReportUI.palette(),shock=rows[0]?.assessment.shock.expected;
    if(shock==null)$('profile-range').value='domain';
    const curve=(r,key)=>({type:'scatter',mode:'lines',name:label(r.method),x:r.fields.x,y:r.fields.values[key]||r.fields.values.M,
      line:{color:method(r.method).color,width:2,dash:({single_stadtke:'solid',multigrid_stadtke:'dash',multigrid_stadtke_modified:'dashdot',multigrid_hll:'dot',multigrid_rusanov:'longdash'})[r.method]}});
    function profile(id,key,title,units) {
      const traces=rows.map(r=>curve(r,key));
      if(rows.length)traces.push({type:'scatter',mode:'lines',name:'Analytical point reference',x:ref.x,y:ref.fields[key]||ref.fields.M,line:{color:p.ink,width:1.5,dash:'dot'}});
      const spec=layout(title,units);
      if($('profile-range').value==='shock' && shock!=null){const width=(ref.x.at(-1)-ref.x[0])*.08;spec.xaxis.range=[shock-width,shock+width];}
      plot(id,traces,spec);
    }
    profile('pressure-plot','p','Pressure','Pressure [Pa]');profile('mach-plot',state.model==='euler_annular'?'M_m':'M','Mach number',state.model==='euler_annular'?'Meridional Mach [−]':'Axial Mach [−]');
    $('profile-range').querySelector('option[value="shock"]').disabled=shock==null;
    const mode=$('residual-mode').value;
    const values=h=>mode==='boundary'?h.boundary_max/DATA.protocol.residual.boundary_tolerance:mode==='global'?Math.max(...h.global_defect)/DATA.protocol.residual.global_tolerance:h.equation_max_ratio;
    const residual=rows.map(r=>{const hist=r.history.filter(h=>['initial','accepted'].includes(h.phase));return {type:'scatter',mode:'lines',name:label(r.method),x:hist.map(h=>h.step),y:hist.map(h=>Math.max(values(h)||0,1e-16)),line:{color:method(r.method).color,width:1.7}};});
    const spec=layout('Original checks · actual fine-grid trajectory','Residual / allowed threshold');spec.xaxis.title={text:'Accepted fine-grid step'};spec.yaxis.type='log';
    spec.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:p.ink,dash:'dot'}}];
    plot('residual-plot',residual,spec);
    const cfl=layout('Accepted CFL history','CFL [−]');cfl.xaxis.title={text:'Accepted fine-grid step'};cfl.yaxis.type='log';
    plot('cfl-plot',rows.map(r=>{const h=r.history.filter(v=>['initial','accepted'].includes(v.phase));return {type:'scatter',mode:'lines',name:label(r.method),x:h.map(v=>v.step),y:h.map(v=>v.cfl),line:{color:method(r.method).color,width:1.5}};}),cfl);
  }
  function render() {
    const number=++state.renderNumber;tasks.length=0;
    const rows=selectedRows();common(rows);caseInterpretation(rows);
    if(state.page==='overview')renderOverview();
    if(state.page==='timing')renderTiming(rows);
    if(state.page==='fields')renderFields(rows);
    Promise.all(tasks).then(()=>{if(number===state.renderNumber)state.rendered=number;});
    $('announcement').textContent=`${modelName(state.model)}, ${shortCase(currentCase())}, ${reconName(state.reconstruction)}, ${rows.length} methods.`;
  }
  function selectCase(caseId,recon,page=state.page) {
    const c=DATA.cases.find(c=>c.id===caseId);if(!c)throw new Error('Unknown physical case');
    state.model=c.model;state.caseId=caseId;state.reconstruction=recon;
    $('model').value=state.model;fillCases();$('reconstruction').value=recon;
    tabs.activate('page-'+page);
  }
  $('methods').innerHTML=DATA.methods.map(m=>`<label class="method-choice"><input type="checkbox" value="${escape(m.id)}" checked><span class="dot" style="background:${m.color}"></span>${escape(label(m.id))}</label>`).join('');
  $('methods').querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{input.checked?state.methods.add(input.value):state.methods.delete(input.value);render();}));
  for(const action of ['all','none'])$('methods-'+action).addEventListener('click',()=>{$('methods').querySelectorAll('input').forEach(input=>{input.checked=action==='all';input.checked?state.methods.add(input.value):state.methods.delete(input.value);});render();});
  $('model').value=state.model;fillCases();
  $('model').addEventListener('change',()=>{state.model=$('model').value;fillCases();render();});
  $('case').addEventListener('change',()=>{state.caseId=$('case').value;render();});
  $('reconstruction').addEventListener('change',()=>{state.reconstruction=$('reconstruction').value;render();});
  $('metric').addEventListener('change',()=>{state.metric=$('metric').value;render();});
  for(const id of ['profile-range','residual-mode'])$(id).addEventListener('change',render);
  $('protocol-details').textContent=JSON.stringify(DATA.protocol,null,2);
  $('provenance-details').textContent=JSON.stringify({measurement_summary_sha256:DATA.measurement_summary_sha256,provenance:DATA.provenance,presentation:DATA.presentation},null,2);
  $('method-rows').innerHTML=DATA.methods.map(m=>`<tr><td><span class="dot" style="background:${m.color}"></span>${escape(label(m.id))}</td><td>${escape(m.flux)}</td><td>${m.grids.join(' → ')}</td><td>${m.flux==='stadtke'?String(m.custom_jump_correction):'Not used'}</td><td>${m.flux==='stadtke'?String(m.custom_endpoint_envelope):'Not used'}</td></tr>`).join('');
  FluxReportUI.init({onThemeChange:render});
  tabs=FluxReportUI.bindTabs({nav:'#page-tabs',panels:'.report-page',onChange:id=>{state.page=id.replace('page-','');render();}});
  window.productionReport={data:DATA,render,selectCase,showPage:page=>tabs.activate('page-'+page),
    snapshot:()=>({model:state.model,caseId:state.caseId,reconstruction:state.reconstruction,metric:state.metric,methods:[...state.methods],page:state.page,rows:selectedRows().map(r=>r.id),renderNumber:state.renderNumber,rendered:state.rendered})};
})();
