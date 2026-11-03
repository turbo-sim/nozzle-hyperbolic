/* Saved-data viewer. No network, EOS or numerical solve is used by this report. */
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const definitions = data.definition;
  const cases = definitions.cases;
  const runs = data.runs;
  const colors = ['#6a00a8', '#b12a90', '#e16462', '#fca636', '#cc4778'];
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const pretty = value => JSON.stringify(value, null, 2);
  const fmt = (value, digits=3) => typeof value === 'number' && Number.isFinite(value) ? value.toPrecision(digits) : '—';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const maximum = value => Array.isArray(value) ? Math.max(...value.filter(finite)) : value;
  const verified = row => Boolean(row?.verification?.verified ?? row?.saved_verification?.verified);
  const runSuffix = row => verified(row)?'':row?.verification?.absolute_targets_pass===true&&row?.verification?.intended_regime_pass===false?' · wrong regime':row?.success?' · unverified':' · unfinished';
  const label = item => item.label || item.id;
  const targetCells = item => Array.isArray(item.cells) ? Math.max(...item.cells) : item.cells;
  const similarity = item => ['perfect_gas','spinelli'].includes(item.family);
  const positionScale = item => similarity(item)?1:1000;
  const fluxes = item => item.fluxes || ['rusanov', 'hll'];
  const caseRuns = id => runs.filter(row => row.case_id === id);
  const latest = rows => [...rows].sort((a,b) => a.id.localeCompare(b.id, undefined, {numeric:true})).at(-1);
  const selected = () => runs.find(row => row.id === $('attempt').value);
  const selectedCase = () => cases.find(item => item.id === $('case').value) || cases[0];
  const peerRuns = () => {
    const item = selectedCase(), n = Number($('grid').value), chosen = selected();
    const sameSetup=row=>!chosen || ['p0_Pa','T0_K','p_out_Pa','friction','roughness_m','geometry','outlet_regime'].every(key=>row.case?.[key]===chosen.case?.[key]);
    const matching=caseRuns(item.id).filter(row=>row.cells===n&&sameSetup(row));
    return [...new Set(matching.map(row => row.flux))].map(flux => {
      if (chosen?.flux === flux) return chosen;
      return latest(matching.filter(row => row.flux === flux));
    });
  };
  function table(headers, rows) {
    return '<div class="table-wrap"><table><thead><tr>' + headers.map(h => '<th>'+esc(h)+'</th>').join('') + '</tr></thead><tbody>' + rows.map(row => '<tr>'+row.map(v => '<td class="wrap">'+v+'</td>').join('')+'</tr>').join('') + '</tbody></table></div>';
  }
  function options(id, values, selectedValue) {
    const previous = selectedValue ?? $(id).value;
    $(id).innerHTML = values.map(item => '<option value="'+esc(item.value)+'">'+esc(item.label)+'</option>').join('');
    if (values.some(item => String(item.value) === previous)) $(id).value = previous;
  }
  function updateCases() {
    const family = $('family').value;
    options('case', cases.filter(item => family === 'all' || item.family === family).map(item => ({value:item.id,label:label(item)})));
    updateGrids();
  }
  function updateGrids() {
    const item = selectedCase();
    const counts = [...new Set([...caseRuns(item.id).map(row => row.cells), targetCells(item)].filter(finite))].sort((a,b) => a-b);
    options('grid', counts.map(n => ({value:String(n),label:n+' cells'+(n === targetCells(item)?' · paper mesh':'')})), String(counts.at(-1)));
    updateAttempts();
  }
  function updateAttempts() {
    const rows = caseRuns($('case').value).filter(row => row.cells === Number($('grid').value));
    options('attempt', rows.map(row => ({value:row.id,label:row.flux.toUpperCase()+' · '+row.id.split('/').at(-1)+' · '+(verified(row)?'verified':row.reason || 'unfinished')})), latest(rows)?.id);
  }
  function chooseCase(id) {
    const item = cases.find(candidate => candidate.id === id);
    $('family').value = item.family;
    updateCases();
    $('case').value = id;
    updateGrids();
    showPage('profiles');
  }
  function status(row) {
    if (!row) return '<span class="badge">Not run</span>';
    const prior=latest(caseRuns(row.case_id).filter(candidate=>candidate.flux===row.flux&&candidate.cells===row.cells))?.id!==row.id;
    const v=row.verification||{},wrongRegime=v.absolute_targets_pass===true&&v.intended_regime_pass===false;
    return '<span class="badge '+(verified(row)?'good':'')+'">'+esc((prior?'Prior attempt · ':'')+(verified(row)?'Verified steady root':wrongRegime?'Converged root; shock regime absent':row.reason || (row.success?'Audit unavailable':'Unfinished')))+'</span>';
  }
  function overview() {
    const requested = cases.flatMap(item => fluxes(item).map(flux => ({item,flux,row:latest(caseRuns(item.id).filter(row => row.flux === flux && row.cells === targetCells(item)))})));
    const passed = requested.filter(entry => verified(entry.row)).length;
    const attempted = requested.filter(entry => entry.row).length;
    const meshRequested=(definitions.mesh_studies || []).flatMap(study=>study.cells.map(cells=>({case_id:study.case_id,flux:study.flux,cells})));
    const meshSaved=meshRequested.map(point=>latest(caseRuns(point.case_id).filter(row=>row.flux===point.flux&&row.cells===point.cells)));
    const meshPassed=meshSaved.filter(verified).length,meshAttempted=meshSaved.filter(Boolean).length;
    const all = passed === requested.length && requested.length > 0 && meshPassed === meshRequested.length;
    $('verdict').classList.toggle('good', all);
    $('verdict').innerHTML = '<h2>'+(all?'All primary solutions and Appendix mesh points pass the saved numerical checks':'Reproduction evidence: '+passed+' of '+requested.length+' primary solutions verified')+'</h2><p>Appendix mesh points: '+meshPassed+' of '+meshRequested.length+' verified, '+meshAttempted+' attempted. '+(attempted < requested.length ? (requested.length-attempted)+' requested primary solutions have no saved attempt. ' : '')+'Published agreement is evaluated independently from convergence. All retained attempts and available comparison data are accessible below.</p>';
    const families = new Set(cases.map(item => item.family)).size;
    $('metrics').innerHTML = [
      [cases.length, 'Operating points', families+' physical families'],
      [passed+' / '+requested.length, 'Verified primary solutions', 'Requested paper mesh, case and method'],
      [runs.length, 'Retained attempts', runs.filter(row=>!verified(row)).length+' without verification; all attempts retained'],
      [meshPassed+' / '+meshRequested.length, 'Verified Appendix mesh points', meshAttempted+' requested mesh points attempted']
    ].map(([value,title,note]) => '<div class="metric"><span>'+esc(title)+'</span><strong>'+esc(value)+'</strong><small>'+esc(note)+'</small></div>').join('');
    $('coverage-table').innerHTML = table(['Case / operating point','Family','Method','Paper cells','Saved numerical status','Pressure comparison'], requested.map(({item,flux,row}) => [
      '<button class="link-button" data-case="'+esc(item.id)+'">'+esc(label(item))+'</button><br><small>'+esc(item.id)+'</small>',
      esc(item.family),esc(flux.toUpperCase()),esc(targetCells(item)),status(row),
      row?.comparison ? comparisonSummary(row.comparison) : 'Reference data and limitations in selected details'
    ]));
    document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => chooseCase(button.dataset.case)));
    const mapping = definitions.figure_map || {};
    const mappingRows = Array.isArray(mapping) ? mapping.map((value,i) => [esc(value.figure || value.id || i),esc(value.description || value.title || ''),esc(value.cases?.join(', ') || value.case_ids?.join(', ') || ''),esc(value.notes || value.status || '')]) : Object.entries(mapping).map(([key,value]) => [esc(key),esc(typeof value === 'string'?value:value.description || value.title || ''),esc(value.cases?.join(', ') || value.case_ids?.join(', ') || ''),esc(value.notes || value.status || '')]);
    $('figure-table').innerHTML = mappingRows.length ? table(['Paper result','Purpose','Cases','Reference / scope'],mappingRows) : '<p class="muted">The full requested matrix is available under Method &amp; provenance.</p>';
    $('paper-tables').innerHTML = paperTables(definitions.paper_tables);
    $('known-limits').innerHTML = '<ul>'+(definitions.known_limits || []).map(note=>'<li>'+esc(note)+'</li>').join('')+'</ul>';
    scalarComparisons();
    $('attempts-table').innerHTML = table(['Case','Method','Cells','Numerical status','Accepted / rejected','Solve time [s]','Saved result'],runs.map(row => {
      const history = row.history || [], accepted = row.accepted_steps ?? history.filter(item => item.phase === 'accepted').length, rejected = row.rejected_attempts ?? history.filter(item => item.accepted === false).length;
      return [esc(row.case_id),esc(row.flux.toUpperCase()),esc(row.cells),status(row),accepted+' / '+rejected,fmt(row.elapsed_s),esc(row.result_file)];
    }));
  }
  function scalarComparisons() {
    let html='';
    for(const [key,title] of [['table1','Table 1 · perfect-gas mass flow and entropy'],['table3','Table 3 · Lettieri mass-flow comparison']]) {
      const value=data.summary?.[key];
      if(!value)continue;
      const rows=Array.isArray(value)?value:value.rows || [];
      if(rows.length && key==='table1')html+='<h3>'+esc(title)+'</h3><p class="muted">The selected contour matches the paper\'s area ratio and critical mass flow. Its exact shock entropy differs from the paper: error against that independently solved contour and disagreement with the historical scalar are reported separately.</p>'+table(['Case','Method','Verified','Current mass flow [kg/s]','Paper mass flow [kg/s]','Mass-flow difference [%]','Current Δs [J/(kg K)]','Contour exact Δs [J/(kg K)]','Difference from contour [%]','Paper Δs [J/(kg K)]','Difference from paper [%]'],rows.map(row=>[esc(row.case_id),esc(row.flux.toUpperCase()),esc(row.verified),fmt(row.mass_flow_kg_s,6),fmt(row.paper_exact_mass_flow_kg_s,6),fmt(row.mass_flow_difference_from_paper_percent),fmt(row.entropy_rise_J_kgK,6),fmt(row.selected_contour_exact_entropy_J_kgK,6),fmt(row.entropy_difference_from_selected_contour_percent),fmt(row.paper_entropy_J_kgK,6),fmt(row.entropy_difference_from_paper_percent)]));
      else if(rows.length && key==='table3')html+='<h3>'+esc(title)+'</h3><p class="muted">The publication prints kg/s, while the archived values use half-channel flow per unit depth. The matching current quantity is full-channel mass flow divided by twice the 6.48 mm depth.</p>'+table(['Case','Method','Verified','Full-channel [kg/s]','Half-channel per depth [kg/(s m)]','Paper HEM per depth','Difference from HEM [%]','Paper CFD per depth','Difference from CFD [%]'],rows.map(row=>[esc(row.case_id),esc(row.flux.toUpperCase()),esc(row.verified),fmt(row.full_channel_mass_flow_kg_s,6),fmt(row.half_channel_mass_flow_kg_s_m,6),fmt(row.paper_HEM_reported,6),fmt(row.difference_from_paper_HEM_percent),fmt(row.paper_CFD_mixture_reported,6),fmt(row.difference_from_paper_CFD_percent)]));
      else html+='<details><summary>'+esc(title)+'</summary><pre>'+esc(pretty(value))+'</pre></details>';
    }
    const digitized=data.summary?.digitized_pressure_comparisons || [];
    if(digitized.length)html+='<h3>Figure 9 · approximate historical pressure samples</h3><p class="muted">Current predictions are sampled at observed, calibrated figure points. Figure 9 uses the historical Rusanov HEM; comparison of current HLL with that curve is a cross-method comparison, not historical HLL regression. Original HEM/CFD arrays are unavailable. Graph uncertainty is a stroke/calibration bound; it is not experimental uncertainty. Regions with overlapping strokes and omitted shock pixels are excluded.</p>'+table(['Case','Current method','Historical figure series','Observed points','RMSE [bar]','RMSE [% p₀]','Maximum error [bar]','Maximum graph pressure bound [bar]','Graph position bound [mm]'],digitized.map(row=>[esc(row.case_id),esc(row.flux.toUpperCase()),esc(row.reference_label),esc(row.samples),fmt(row.pressure_rmse_bar),fmt(row.pressure_rmse_percent_p0),fmt(row.pressure_max_error_bar),fmt(row.digitization_pressure_uncertainty_bar_max),fmt(row.x_uncertainty_mm_max)]));
    $('scalar-comparisons').innerHTML=html || '<p class="muted">Current scalar comparisons are added when campaign summary measurements are available.</p>';
  }
  function meshStudyRows() {
    const value=data.summary?.mesh_studies || [];
    if(Array.isArray(value))return value;
    return Object.entries(value).flatMap(([key,item])=>Array.isArray(item)?item.map(row=>({...row,case_id:row.case_id || key})):({...item,case_id:item.case_id || key}));
  }
  let meshCase='';
  async function meshPlots() {
    const studies=meshStudyRows(),ids=[...new Set(studies.map(item=>item.case_id || item.id).filter(Boolean))];
    if(!$('mesh-case')) {
      $('mesh-select').innerHTML='<label>Mesh-study operating point<select id="mesh-case"></select></label>';
      $('mesh-case').addEventListener('change',()=>{meshCase=$('mesh-case').value;meshPlots();});
    }
    options('mesh-case',ids.map(value=>({value,label:value})),meshCase || ids[0]);
    meshCase=$('mesh-case').value;
    const active=studies.filter(item=>(item.case_id || item.id)===meshCase),pressure=[],mass=[],rows=[];
    active.forEach((study,index)=>{
      const points=study.points || study.rows || [],valid=points.filter(point=>point.verified),finest=[...valid].sort((a,b)=>a.cells-b.cells).at(-1),color=colors[index%colors.length];
      for(const [key,name,dash] of [['p_throat_Pa','Throat pressure','solid'],['p_out_Pa','Outlet pressure','dash'],['p_min_Pa','Minimum pressure','dot']]) {
        if(!finite(finest?.[key])||finest[key]===0)continue;
        pressure.push({x:points.map(point=>point.cells),y:points.map(point=>point[key+'_difference_percent'] === undefined?(finite(point[key])?100*Math.abs(point[key]/finest[key]-1):null):point[key+'_difference_percent']),name:(study.flux || '')+' · '+name,mode:'lines+markers',line:{color,width:1.8,dash},marker:{size:points.map(point=>point.verified?5:8),symbol:points.map(point=>point.verified?'circle':'x')}});
      }
      const mk=finite(finest?.mass_flow_kg_s)?'mass_flow_kg_s':'mdot_kg_s';
      if(finite(finest?.[mk])&&finest[mk]!==0)mass.push({x:points.map(point=>point.cells),y:points.map(point=>point[mk+'_difference_percent'] === undefined?(finite(point[mk])?100*Math.abs(point[mk]/finest[mk]-1):null):point[mk+'_difference_percent']),name:(study.flux || '').toUpperCase(),mode:'lines+markers',line:{color,width:2},marker:{size:points.map(point=>point.verified?5:8),symbol:points.map(point=>point.verified?'circle':'x')}});
      points.forEach(point=>rows.push([esc(study.flux),esc(point.cells),esc(point.verified),fmt(point.p_throat_Pa*1e-5),fmt(point.p_out_Pa*1e-5),fmt(point.p_min_Pa*1e-5),fmt(point[mk],6)]));
    });
    await Promise.all([plot('mesh-pressure',pressure,'Pressure sensitivity relative to finest verified mesh','Number of cells [–]','Absolute relative difference [%]'),plot('mesh-mass',mass,'Mass-flow sensitivity relative to finest verified mesh','Number of cells [–]','Absolute relative difference [%]')]);
    const wrongBranch=active.flatMap(study=>(study.points || []).filter(point=>{const row=latest(caseRuns(study.case_id).filter(run=>run.flux===study.flux&&run.cells===point.cells));return row?.verification?.absolute_targets_pass===true&&row?.verification?.intended_regime_pass===false;}).map(point=>point.cells));
    const branchNote=wrongBranch.length?'<p class="note">At '+esc([...new Set(wrongBranch)].sort((a,b)=>a-b).join(' and '))+' cells the steady discrete equations converge, but the solution remains entirely subsonic and the intended shock is absent. These are converged coarse-grid roots on the wrong physical branch. Cross markers retain their mesh diagnostics without claiming the requested regime.</p>':'';
    $('mesh-table').innerHTML=rows.length?branchNote+table(['Method','Cells','Verified','p throat [bar]','p outlet [bar]','p min [bar]','Numerical mass flow [kg/s]'],rows):'<p class="muted">No saved mesh-study summary yet. Individual retained grids remain accessible under Flow profiles.</p>';
  }
  function paperTables(tables) {
    if (!tables) return '';
    const entries = Array.isArray(tables) ? tables.map((value,i) => [value.id || value.table || i,value]) : Object.entries(tables);
    return entries.map(([key,value]) => '<details><summary>'+esc(key)+(value.title?' · '+esc(value.title):'')+'</summary><p class="muted">Published scalar values are retained at their reported precision. Their units and mass-flow convention follow the source definition.</p><pre>'+esc(pretty(value))+'</pre></details>').join('');
  }
  function comparisonSummary(comparison) {
    if (comparison.errors?.length) return comparison.errors.map(item => esc(item.label)+': RMSE '+fmt(item.pressure_rmse_bar)+' bar').join('<br>');
    const errors = comparison.pressure_rmse_bar ?? comparison.pressure?.rmse_bar ?? comparison.pressure_rms_bar;
    if (finite(errors)) return 'RMSE '+fmt(errors)+' bar';
    if (comparison.pressure_relative_rms_percent !== undefined) return 'RMS '+fmt(comparison.pressure_relative_rms_percent)+'%';
    if (comparison.available === false) return esc(comparison.reason || 'Published numerical profile unavailable');
    return 'Saved comparison details';
  }
  function theme() {
    const style = getComputedStyle(document.documentElement);
    const get = key => style.getPropertyValue(key).trim();
    return {paper:get('--paper'),ink:get('--ink'),muted:get('--muted'),grid:get('--grid')};
  }
  function plot(id, traces, title, xTitle, yTitle, extras={}) {
    const t = theme();
    return Plotly.react(id,traces,{title:{text:title,x:0.05,y:0.98,yanchor:'top',font:{size:14}},margin:{l:70,r:22,t:85,b:57},paper_bgcolor:t.paper,plot_bgcolor:t.paper,font:{family:'Segoe UI,Arial,sans-serif',size:11,color:t.ink},xaxis:{title:{text:xTitle,standoff:12},gridcolor:t.grid,zeroline:false},yaxis:{title:{text:yTitle,standoff:8},gridcolor:t.grid,zeroline:false},legend:{orientation:'h',x:0,y:1.01,xanchor:'left',yanchor:'bottom',font:{size:10}},hovermode:'x unified',...extras},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}});
  }
  function fields(row) {
    const f = row?.fields || {};
    const quality=f.quality || f.Q;
    const s0=row?.reservoir?.s0_J_kgK ?? row?.reservoir?.s0 ?? row?.reservoir?.s ?? row?.s0;
    const entropyRise=f.s?.map(value=>finite(value)?value-(finite(s0)?s0:f.s[0]):null);
    const liquid=f.liquid_mass_fraction || quality?.map(value=>finite(value)&&value>=0&&value<=1?1-value:null);
    const h0=f.h0 || f.h?.map((value,index)=>value+0.5*f.u[index]**2);
    return {...f,x_m:f.x_m || f.x || row?.x_m,p:f.p || f.pressure_Pa,M:f.M || f.mach,quality,Z:f.Z || f.compressibility_factor,'entropy-rise':entropyRise,liquid,h0};
  }
  function allReferenceSeries(item,row) {
    const result=[];
    function add(value) {
      if (!value) return;
      if (Array.isArray(value)) {value.forEach(add);return;}
      if (typeof value !== 'object') return;
      if ((value.x_m || value.x) && (value.p_Pa || value.p || value.pressure_Pa || value.M || value.quality || value.compressibility_factor || value.liquid_mass_fraction)) {result.push(value);return;}
      if (value.series) add(value.series);
      if (value.references) add(value.references);
      if (value[item.id]) add(value[item.id]);
      if (value.cases?.[item.id]) add(value.cases[item.id]);
      if (value.case_id === item.id) add(value.reference);
    }
    add(item.reference);
    add(row?.reference);
    add(row?.comparison?.series);
    Object.values(data.references).forEach(add);
    return result.filter((value,index,array) => array.findIndex(candidate => JSON.stringify(candidate)===JSON.stringify(value))===index);
  }
  function fieldTraces(item,rows,key,scale) {
    const traces=[];
    allReferenceSeries(item,selected()).forEach((ref,index) => {
      let values=key==='p'?(ref.p_Pa || ref.pressure_Pa || ref.p):key==='Z'?(ref.Z || ref.compressibility_factor):key==='liquid'?(ref.liquid_mass_fraction || ref.liquid):ref[key];
      if(key==='entropy-rise'&&ref.kind==='analytical'&&ref.T&&ref.p_Pa){const gamma=item.gamma||1.4,R=item.R_J_kgK||287;values=ref.T.map((temperature,i)=>gamma*R/(gamma-1)*Math.log(temperature/item.T0_K)-R*Math.log(ref.p_Pa[i]/item.p0_Pa));}
      if(key==='h0'&&ref.h&&ref.u)values=ref.h.map((enthalpy,i)=>enthalpy+0.5*ref.u[i]**2);
      if (!values) return;
      const uncertainty=key==='p'&&ref.pressure_uncertainty_Pa?ref.pressure_uncertainty_Pa.map(value=>value*scale):key==='p'&&ref.p_over_p0_uncertainty?ref.p_over_p0_uncertainty.map(value=>value*item.p0_Pa*scale):key==='Z'?ref.uncertainty:null;
      const digitized=ref.kind?.startsWith('digitized_paper');
      traces.push({x:(ref.x_m || ref.x).map(v=>v*positionScale(item)),y:values.map(v=>finite(v)?v*scale:null),name:ref.label || ref.kind || 'Published reference',mode:ref.kind==='analytical'?'lines':'markers',line:{color:'#7b8595',dash:'dash',width:1.5},marker:{color:ref.kind==='coordinate_diagnostic'?'#fca636':'#fff',line:{color:'#34465d',width:digitized?1:1.3},size:digitized?4:6,symbol:ref.kind==='coordinate_diagnostic'?'x':ref.kind==='digitized_paper_hem'?'diamond':ref.kind==='digitized_paper_cfd'?'square':index?'square':'circle'},error_y:uncertainty?{type:'data',array:uncertainty,visible:true,color:'#8b95a3',thickness:0.7,width:2}:undefined,error_x:digitized&&finite(ref.x_uncertainty_m)?{type:'constant',value:ref.x_uncertainty_m*positionScale(item),visible:true,color:'#8b95a3',thickness:0.7,width:2}:undefined,connectgaps:false});
    });
    rows.forEach((row,index) => {
      const f=fields(row), values=f[key];
      if (!f.x_m || !values) return;
      const y=values.map(v => finite(v) && (key!=='quality' || (v>=0 && v<=1)) ? v*scale : null);
      traces.push({x:f.x_m.map(v=>v*positionScale(item)),y,name:row.flux.toUpperCase()+runSuffix(row),mode:'lines',line:{color:colors[index%colors.length],width:2,dash:verified(row)?'solid':'dot'},connectgaps:false});
    });
    return traces;
  }
  async function profiles() {
    const item=selectedCase(),row=selected(),peers=peerRuns();
    $('selected-status').innerHTML='<strong>'+esc(label(item))+'</strong> · '+status(row)+'<br>'+esc(row?.initialization || 'No saved attempt at the selected resolution.')+'<br><span class="muted">All available methods on this mesh are overlaid. Dotted lines show unfinished or unverified fields. Quality outside [0,1] is omitted as a single-phase sentinel or invalid diagnostic.</span>';
    const f=fields(row), area=f.area_m2 || f.area_centers;
    const areaScale=similarity(item)&&area?1/Math.min(...area):1e6;
    const geometry=area && f.x_m ? [{x:f.x_m.map(v=>v*positionScale(item)),y:area.map(v=>v*areaScale),mode:'lines',name:similarity(item)?'Area / minimum sampled area':'Cell-center area',line:{color:'#9c179e',width:2},fill:'tozeroy',fillcolor:'rgba(156,23,158,0.08)'}] : [];
    const ts=peers.filter(run=>fields(run).s&&fields(run).T).map((run,index)=>({x:fields(run).s.map(value=>value/1000),y:fields(run).T,name:run.flux.toUpperCase()+runSuffix(run),mode:'lines',line:{color:colors[index%colors.length],width:2,dash:verified(run)?'solid':'dot'}}));
    const phase=data.phase_diagrams.fluids?.find(fluid=>fluid.fluid===item.fluid);
    if(phase){
      for(const key of ['saturation_liquid','saturation_vapor'])if(phase[key]?.s)ts.unshift({x:phase[key].s.map(value=>value/1000),y:phase[key].T,name:key.replace('_',' '),mode:'lines',line:{color:'#7b8595',width:1.2},showlegend:key==='saturation_liquid'});
      if(phase.reservoirs?.length)ts.push({x:phase.reservoirs.map(value=>(value.s0 ?? value.s0_J_kgK)/1000),y:phase.reservoirs.map(value=>value.T0 ?? value.T0_K),text:phase.reservoirs.map(value=>value.case_id),name:'Reservoir states',mode:'markers',marker:{color:'#fff',size:6,line:{color:'#34465d',width:1.3}},hovertemplate:'%{text}<br>s=%{x:.3f} kJ/(kg K)<br>T=%{y:.2f} K<extra></extra>'});
    }
    const normalized=similarity(item);
    const xTitle=normalized?'Axial coordinate x/L [–]':'Axial coordinate x [mm]';
    const entropyTitle=finite(row?.reservoir?.s0_J_kgK ?? row?.reservoir?.s0 ?? row?.reservoir?.s ?? row?.s0)?'s − s₀ [J/(kg K)]':'s − s(first cell) [J/(kg K)]';
    const phaseAvailable=peers.some(run=>fields(run).quality?.some(value=>finite(value)&&value>=0&&value<=1));
    const phaseAxes=title=>({yaxis:{title,range:[0,1],gridcolor:theme().grid,zeroline:false},annotations:phaseAvailable?[]:[{text:'Current state: no finite two-phase-quality samples',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,font:{size:12,color:theme().muted}}]});
    await Promise.all([
      plot('geometry',geometry,normalized?'Nozzle area / minimum sampled cell area':'Nozzle geometry',xTitle,normalized?'A / min(A cells) [–]':'Area [mm²]',{margin:{l:70,r:22,t:48,b:53}}),
      plot('field-p',fieldTraces(item,peers,'p',1e-5),'Static pressure',xTitle,'p [bar]'),
      plot('field-M',fieldTraces(item,peers,'M',1),'Mach number',xTitle,'M [–]'),
      plot('field-quality',fieldTraces(item,peers,'quality',1),'Equilibrium vapor mass fraction',xTitle,'Q [kg/kg]',phaseAxes('Q [kg/kg]')),
      plot('field-u',fieldTraces(item,peers,'u',1),'Axial velocity',xTitle,'u [m/s]'),
      plot('field-rho',fieldTraces(item,peers,'rho',1),'Density',xTitle,'ρ [kg/m³]'),
      plot('field-entropy-rise',fieldTraces(item,peers,'entropy-rise',1),'Entropy change',xTitle,entropyTitle),
      plot('field-Z',fieldTraces(item,peers,'Z',1),'Compressibility factor',xTitle,'Z [–]'),
      plot('field-T',fieldTraces(item,peers,'T',1),'Temperature',xTitle,'T [K]'),
      plot('field-liquid',fieldTraces(item,peers,'liquid',1),'Equilibrium liquid mass fraction',xTitle,'1 − Q [kg/kg]',phaseAxes('1 − Q [kg/kg]')),
      plot('field-mdot',fieldTraces(item,peers,'mdot',1),'Cell mass-flow diagnostic',xTitle,item.family==='spinelli'?'Computational ρuA [–]':'ρuA [kg/s]'),
      plot('field-Ts',ts,'Temperature–entropy expansion path','s [kJ/(kg K)]','T [K]'),
      plot('field-h0',fieldTraces(item,peers,'h0',1e-3),'Stagnation enthalpy',xTitle,'h + u²/2 [kJ/kg]')
    ]);
    const matching=['lettieri','berana','berana_sensitivity'].includes(item.family)?cases.filter(candidate=>candidate.geometry===item.geometry&&candidate.p0_Pa===item.p0_Pa&&candidate.T0_K===item.T0_K&&candidate.p_out_Pa===item.p_out_Pa):[];
    $('variant-panel').hidden=matching.length<2;
    if(matching.length>1){
      const variantRuns=matching.flatMap(candidate=>fluxes(candidate).map(flux=>latest(caseRuns(candidate.id).filter(run=>run.flux===flux&&run.cells===Number($('grid').value))))).filter(Boolean);
      const variants=key=>variantRuns.filter(run=>fields(run).x_m&&fields(run)[key]).map((run,index)=>({x:fields(run).x_m.map(value=>value*positionScale(item)),y:fields(run)[key].map(value=>finite(value)?value*(key==='p'?1e-5:1):null),name:run.case_id+' · '+run.flux.toUpperCase()+runSuffix(run),mode:'lines',line:{color:colors[index%colors.length],width:1.8,dash:index>=colors.length?'dash':'solid'}}));
      await Promise.all([plot('variant-pressure',variants('p'),'Pressure across wall variants',xTitle,'p [bar]',{margin:{l:70,r:22,t:90,b:57}}),plot('variant-liquid',variants('liquid'),'Liquid fraction across wall variants',xTitle,'1 − Q [kg/kg]',{margin:{l:70,r:22,t:90,b:57}})]);
    }
    $('settings').textContent=pretty({requested_definition:item,effective_saved_case:row?.case,scheme:row?.scheme,steady:row?.steady,initialization:row?.initialization,properties:row?.properties});
    const digitized=allReferenceSeries(item,row).filter(ref=>ref.kind?.startsWith('digitized_paper'));
    $('comparison').innerHTML = '<p>'+esc(item.notes?.join?.(' ') || item.notes || '')+'</p>'+(digitized.length?'<p class="note">Figure 9 historical Rusanov HEM/CFD comparisons use approximate, calibrated samples of the original high-resolution raster plot. Current HLL against the historical Rusanov curve is a cross-method comparison. Original numerical arrays remain unavailable. Only visible, identifiable strokes are retained; overlapping curves, merged tails and steep shock pixels are omitted. Error bars bound stroke thickness and tick calibration, and are not statistical confidence intervals. No historical curve is interpolated through gaps.</p>':'')+'<details open><summary>Saved comparison metrics and reference metadata</summary><pre>'+esc(pretty(row?.comparison || {note:'No quantitative comparison saved for this selected attempt.'}))+'</pre></details>'+(digitized.length?'<details><summary>Digitized historical samples and uncertainty</summary><pre>'+esc(pretty(digitized))+'</pre></details>':'');
  }
  function historyRows(row) {
    return (row?.history || []).filter(item => item.accepted !== false);
  }
  async function convergence() {
    const row=selected(),peers=peerRuns(),residual=[],cfl=[];
    peers.forEach((run,index) => {
      const h=historyRows(run);
      for(let k=0;k<3;k++) {
        const y=h.map(item => Array.isArray(item.equation_norm)?item.equation_norm[k]:Array.isArray(item.equation_l2)?item.equation_l2[k]:null);
        residual.push({x:h.map((item,i)=>item.step ?? i),y:y.map(value=>finite(value)&&value>0?value:null),name:run.flux.toUpperCase()+' · '+['mass','momentum','energy'][k],mode:'lines',line:{color:colors[index%colors.length],width:1.6,dash:['solid','dash','dot'][k]}});
      }
      cfl.push({x:h.map((item,i)=>item.step ?? i),y:h.map(item=>item.cfl ?? item.requested_cfl ?? null),name:run.flux.toUpperCase(),mode:'lines',line:{color:colors[index%colors.length],width:2}});
    });
    const t=theme();
    await Promise.all([
      plot('history-residual',residual,'Original spatial residual','Accepted pseudo-time step','Fixed-scale spatial RMS [–]',{yaxis:{title:'Fixed-scale spatial RMS [–]',type:'log',gridcolor:t.grid,zeroline:false},margin:{l:75,r:20,t:85,b:57}}),
      plot('history-cfl',cfl,'CFL controller','Accepted pseudo-time step','Requested CFL [–]',{yaxis:{title:'Requested CFL [–]',type:'log',gridcolor:t.grid,zeroline:false}})
    ]);
    const v=row?.verification || row?.saved_verification || {}, metrics=row?.metrics || {};
    $('verification').innerHTML=table(['Check','Saved measurement','Pass condition'],[
      ['Independent audit',status(row),'Original equations, boundaries, balances and physical branch'],
      ['Spatial RMS, mass / momentum / energy',esc((v.equation_l2 || metrics.equation_norm || []).map(value=>fmt(value)).join(' / ')),'Recorded original spatial targets'],
      ['Boundary maximum',fmt(v.boundary_max ?? metrics.boundary_max),'Recorded boundary target'],
      ['Global mass / momentum / energy defect',esc((v.global_defect || metrics.global_defect || []).map(value=>fmt(value)).join(' / ')),'Numerical boundary-flux/source balance'],
      ['Admissible thermodynamic states',esc(v.admissible ?? 'Unavailable'),'All model checks satisfied'],
      ['Intended physical branch',esc(v.physical_branch_pass ?? v.branch ?? 'Unavailable'),'Requested inlet/outlet regime'],
      ['Boundary characteristic regime',esc(v.boundary_characteristic_regime_pass ?? 'Unavailable'),'Physical inlet and outlet traces'],
      ['Intended internal regime',esc(v.intended_regime_pass ?? 'Unavailable'),esc(v.intended_regime || 'Requested subsonic or shocked interior')],
      ['Interior Mach range',esc((v.interior_mach_range || []).map(value=>fmt(value)).join(' to ')),'A shocked case must include a supersonic interior'],
      ['Solve time [s]',fmt(row?.elapsed_s),'One observation; no performance distribution']
    ]);
    $('residual-details').textContent=pretty({verification:v,metrics,reason:row?.reason,error:row?.error});
  }
  let page='overview',renderNumber=0,rendered=0;
  async function render() {
    const request=++renderNumber;
    if(page==='overview') await meshPlots();
    if(page==='profiles') await profiles();
    if(page==='convergence') await convergence();
    if (request===renderNumber) rendered=request;
  }
  function showPage(name) {
    page=name;
    document.querySelectorAll('.page').forEach(element=>element.hidden=element.id!==name);
    document.querySelectorAll('[data-page]').forEach(element=>element.setAttribute('aria-selected',element.dataset.page===name));
    $('shared-controls').hidden=!['profiles','convergence'].includes(name);
    render();
  }
  function download(name,bytes,type) {
    const link=document.createElement('a'),url=URL.createObjectURL(new Blob([bytes],{type}));
    link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function downloadEvidence() {
    const raw=atob($('evidence-data').textContent.trim()),bytes=new Uint8Array(raw.length);
    for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
    download('TurboExpo2025-reproduction-evidence.zip',bytes,'application/zip');
  }
  $('download-evidence').addEventListener('click',downloadEvidence);
  $('theme').addEventListener('click',()=>{
    const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';
    $('theme').textContent=dark?'Light theme':'Dark theme';render();
  });
  options('family',[{value:'all',label:'All physical families'},...[...new Set(cases.map(item=>item.family))].map(value=>({value,label:value}))]);
  updateCases();
  $('family').addEventListener('change',()=>{updateCases();render();});
  $('case').addEventListener('change',()=>{updateGrids();render();});
  $('grid').addEventListener('change',()=>{updateAttempts();render();});
  $('attempt').addEventListener('change',render);
  document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>showPage(button.dataset.page)));
  overview();
  $('policy').innerHTML='<details><summary>Declared numerical policy</summary><pre>'+esc(pretty(definitions.numerical_policy || {}))+'</pre></details>';
  $('provenance').textContent=pretty({paper:definitions.paper,data:data.provenance,paper_digitization:data.references.paper_digitization?{source:data.references.paper_digitization.source,method:data.references.paper_digitization.method,panels:data.references.paper_digitization.panels,visual_review:data.references.paper_digitization.visual_review}:undefined});
  if(data.digitization_qa?.length){const qa=document.createElement('details');qa.innerHTML='<summary>Figure 9 digitization · original-raster QA overlays</summary><p>Original embedded plot panels with accepted sample locations: cyan circles show historical HEM samples; blue crosses show separately identifiable CFD dash centers. Only these visible sections are used. The calibrated image coordinates, source hash, exclusions and pressure/position bounds are retained in the evidence ZIP.</p><div class="qa-grid">'+data.digitization_qa.map(image=>'<figure><img class="qa-image" alt="Figure 9 '+esc(image.name.includes('9a')?'a':'b')+' digitization overlay with accepted HEM and CFD sample positions" src="'+image.data_url+'"><figcaption>'+esc(image.name.includes('9a')?'Figure 9a · cases P1 and P3':'Figure 9b · cases P2 and P4')+'</figcaption></figure>').join('')+'</div>';$('geometry-details').closest('details').before(qa);}
  $('geometry-details').textContent=pretty(definitions.geometries);
  $('definition').textContent=pretty(definitions);
  const heads=[...new Set(runs.map(row=>row.source?.git_head || row.source?.head).filter(Boolean))];
  $('source-summary').innerHTML=table(['Saved source epochs','Generated report','Backend / precision'],[[esc(heads.join(', ') || 'See per-attempt source metadata'),esc(data.generated_at),esc([...new Set(runs.map(row=>[row.source?.backend,row.source?.x64?'float64':''].filter(Boolean).join(' / ')).filter(Boolean))].join('; ') || 'See per-attempt metadata')]]);
  $('footer').textContent='Generated '+data.generated_at+'. Saved results only. Offline Plotly 2.35.2 and MathJax 3.2.2; license texts and raw evidence are embedded in the ZIP.';
  window.turboReport={data,render,showPage,chooseCase,selected,peerRuns,allReferenceSeries,downloadEvidence,snapshot:()=>({page,renderNumber,rendered,case_id:$('case').value,cells:Number($('grid').value),selected:selected()?.id,plots:document.querySelectorAll('.js-plotly-plot').length})};
  showPage('overview');
})();
