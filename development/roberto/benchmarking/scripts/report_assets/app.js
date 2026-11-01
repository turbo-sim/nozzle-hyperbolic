(() => {
  'use strict';
  const payload=JSON.parse(document.getElementById('benchmark-data').textContent);
  const campaign=payload.campaign, baseline=payload.baseline, report=payload.report;
  const $=id=>document.getElementById(id);
  const finite=value=>typeof value==='number'&&Number.isFinite(value);
  const number=value=>value===null||value===undefined||typeof value==='boolean'?null:(Number.isFinite(Number(value))?Number(value):null);
  const fmt=(value,digits=3)=>finite(number(value))?Number(value).toLocaleString(undefined,{maximumFractionDigits:digits}):'—';
  const sci=value=>finite(number(value))?Number(value).toExponential(2):'—';
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const array=value=>Array.isArray(value)?value:[];
  const timingEnvironmentStatus=value=>typeof value==='object'&&value!==null?value.status:value;
  const unchanged=(data,key)=>{const values=[data?.[key],data?.provenance?.[key]];return values.includes(false)?false:values.includes(true)?true:null;};
  const campaignTimingValid=data=>(data.completed??data.complete)===true&&unchanged(data,'source_unchanged')===true&&unchanged(data,'harness_unchanged')===true&&timingEnvironmentStatus(data.timing_environment)==='clean';
  const max=values=>{const usable=values.map(number).filter(finite);return usable.length?Math.max(...usable):null;};
  const mean=values=>{const usable=values.map(number).filter(finite);return usable.length?usable.reduce((a,b)=>a+b,0)/usable.length:null;};
  const reconLabel=name=>name==='constant'?'Constant':name==='muscl_van_leer'?'MUSCL':name==='muscl_van_leer_smooth'?'Smooth MUSCL':name||'Unspecified';
  const modelLabel=name=>String(name||'').includes('annular')?'Annular Euler':String(name||'').includes('nozzle')?'Euler nozzle':name||'Unspecified';
  const backendKind=name=>/legacy|native|old/i.test(name)?'legacy':/current|production|new/i.test(name)?'current':name;
  const fluxLabel=name=>({roe:'Roe',rusanov:'Rusanov',hll:'HLL',stadtke:'Städtke FVS'}[name]||name||'Unspecified');
  const backendLabel=name=>backendKind(name)==='legacy'?'Old native':backendKind(name)==='current'?'Current production':name||'Unspecified';
  const unique=items=>[...new Set(items.filter(x=>x!==null&&x!==undefined&&x!==''))];
  const sortedNumbers=values=>unique(values).sort((a,b)=>Number(a)-Number(b));
  const stable=value=>JSON.stringify(value&&typeof value==='object'&&!Array.isArray(value)?Object.fromEntries(Object.keys(value).sort().map(k=>[k,JSON.parse(stable(value[k]))])):Array.isArray(value)?value.map(x=>JSON.parse(stable(x))):value??null);
  const knownFields={p:['Pressure','Pa'],pressure:['Pressure','Pa'],rho:['Density','kg/m³'],density:['Density','kg/m³'],u:['Axial velocity','m/s'],v_m:['Meridional velocity','m/s'],v_theta:['Tangential velocity','m/s'],T:['Temperature','K'],M:['Mach number',''],M_m:['Meridional Mach number',''],M_theta:['Tangential Mach number',''],h:['Static enthalpy','J/kg'],h0:['Total enthalpy','J/kg'],ell:['Angular momentum','m²/s'],mdot:['Mass flow','kg/s'],s:['Entropy','J/(kg K)'],ds:['Entropy increase','J/(kg K)']};
  function fields(value){
    if(!value||Array.isArray(value)||typeof value!=='object')return {};
    const values=value.fields||value.values||value;
    return Object.fromEntries(Object.entries(values).filter(([key,v])=>!['x','m','coordinate','coordinates','W','q','state','conservative','faces'].includes(key)&&Array.isArray(v)&&v.every(item=>item===null||typeof item==='number')));
  }
  function curve(value,fallback){
    value=value||{};const profile=value.dense||value.profile||value;
    return {x:array(profile.x||profile.m||profile.coordinate||profile.coordinates||fallback),fields:fields(profile)};
  }
  function normalize(data){
    const list=data?.results||data?.records||data?.scenarios||[];
    return array(list).map((source,index)=>{
      const representative=source.representative||source.representative_sample||source.solution||{};
      const raw={...representative,...source};
      const timing=raw.timing||raw.timings||{};
      const repeatRecords=array(raw.samples||raw.repetitions||timing.repetitions);
      const times=array(timing.samples_s||raw.samples_s).length?array(timing.samples_s||raw.samples_s):repeatRecords.map(item=>typeof item==='number'?item:item.elapsed_s??item.time_s??item.timing_s);
      const usable=times.map(number).filter(finite);
      const explicitSuccess=raw.success??raw.converged??raw.metrics?.converged;
      let status=raw.status||(explicitSuccess===true?'passed':explicitSuccess===false?'failed':'unclassified');
      if(raw.presentation_missing_data)status='missing data';
      const numerical=curve(raw.state||raw.numerical||{x:raw.x,fields:raw.fields},raw.x);
      let reference=raw.reference||raw.analytical||raw.reference_dense;
      if(reference?.shared_reference_sha256)reference=data.presentation_references?.[reference.shared_reference_sha256];
      const analytical=curve(reference,raw.reference_x||numerical.x);
      const finalMetrics=raw.metrics||raw.audit||{};
      const accuracyAssessed=raw.accuracy_assessed??raw.validation?.accuracy_assessed;
      const accuracy=accuracyAssessed===false?null:(raw.accuracy_passed??raw.analytical_passed??finalMetrics.accuracy_passed??finalMetrics.analytical_passed??raw.validation?.passed);
      // Historical recorded identifiers remain in raw evidence; only the view name changes.
      const recordedFlux=raw.flux||raw.scheme?.flux||raw.settings?.scheme?.flux||data.protocol?.scheme?.flux||'stadtke';
      const displayedFlux=recordedFlux==='characteristic'?'stadtke':recordedFlux;
      return {raw,index,recordedFlux,caseId:String(raw.case_id||raw.case?.id||raw.id||'unknown'),backend:String(raw.backend||raw.method||'current'),
        flux:displayedFlux,
        reconstruction:raw.reconstruction||raw.scheme?.reconstruction||raw.settings?.scheme?.reconstruction||'constant',
        cells:number(raw.cells??raw.n_cells??raw.mesh?.cells??raw.settings?.cells),cap:number(raw.cap??raw.newton_cap??raw.scheme?.newton_iterations??raw.settings?.scheme?.newton_iterations),
        model:raw.model||raw.case?.model||'',status:String(status),success:explicitSuccess,accuracy,accuracyAssessed,
        verificationOnly:data.verification_only===true||data.protocol?.verification_only===true,
        timingValid:raw.timing_valid===true&&campaignTimingValid(data)&&timingEnvironmentStatus(raw.timing_environment)==='clean',
        timing:{...timing,samples_s:usable,mean_s:number(timing.mean_s??raw.mean_s)??mean(usable),min_s:number(timing.min_s)??(usable.length?Math.min(...usable):null),max_s:number(timing.max_s)??(usable.length?Math.max(...usable):null),warmup_s:number(timing.warmup_s??raw.warmup_s)},
        repeatRecords,allRepeatsPassed:repeatRecords.every(item=>typeof item!=='object'||item===null||(item.success??item.passed??true)!==false),
        state:numerical,reference:analytical,history:array(raw.history||representative.history),errors:raw.errors||raw.analytical_errors||finalMetrics.errors||{},metrics:finalMetrics};
    });
  }
  const results=normalize(campaign), baselineResults=baseline?normalize(baseline):[];
  const multiFlux=unique(results.map(r=>r.flux)).length>1;
  const verificationOnly=campaign.verification_only===true||campaign.protocol?.verification_only===true;
  const referenceFlux='stadtke';
  const caseMap=new Map();
  const caseValues=Array.isArray(campaign.cases)?campaign.cases:Object.entries(campaign.cases||{}).map(([id,value])=>({id,...value}));
  caseValues.forEach(item=>{const value=typeof item==='string'?{id:item}:item;caseMap.set(String(value.id||value.case_id),value);});
  results.forEach(result=>{if(!caseMap.has(result.caseId))caseMap.set(result.caseId,{id:result.caseId,label:result.raw.case_label||result.raw.label||result.caseId,model:result.model});if(!result.model)result.model=caseMap.get(result.caseId)?.model||'';});
  const cases=[...caseMap.values()].sort((a,b)=>String(a.model||'').localeCompare(String(b.model||''))||String(a.label||a.id).localeCompare(String(b.label||b.id)));
  const caseLabel=id=>caseMap.get(id)?.label||caseMap.get(id)?.name||id;
  const outcome=result=>{
    if(/unsupported|skipped|pending|missing|unclassified/i.test(result.status))return {label:result.status,kind:'warn'};
    if(/timeout/i.test(result.status))return {label:'Timed out',kind:'fail'};
    if(result.status==='branch_failed')return {label:'Physical branch failed',kind:'fail'};
    if(result.status==='assessment_failed')return {label:'Independent assessment failed',kind:'fail'};
    if(result.status==='accuracy_failed')return {label:'Accuracy target failed',kind:'warn'};
    if(/fail|error|invalid/i.test(result.status)||result.success===false)return {label:'Failed',kind:'fail'};
    if(result.verificationOnly){
      if(result.accuracyAssessed!==false&&result.accuracy===false)return {label:'Accuracy target failed',kind:'warn'};
      if(result.accuracyAssessed===false)return {label:'Numerically converged; accuracy unassessed',kind:'pass'};
      return {label:result.accuracy===true?'Numerically verified + accuracy':'Numerically converged',kind:'pass'};
    }
    if(!result.timingValid)return {label:'Converged; timing invalid',kind:'warn'};
    if(result.accuracyAssessed===false)return {label:'Converged; accuracy unassessed',kind:'pass'};
    if(result.accuracy===false)return {label:'Converged · accuracy limit',kind:'warn'};
    return {label:result.accuracy===true?'Passed + accuracy':'Passed',kind:'pass'};
  };
  const good=result=>result.success===true&&result.timingValid&&outcome(result).kind==='pass'&&result.allRepeatsPassed&&result.timing.samples_s.length>=2&&finite(result.timing.mean_s)&&result.timing.mean_s>0;
  const errorItem=(result,field)=>typeof result.errors[field]==='object'&&result.errors[field]!==null?result.errors[field]:{};
  const normalizedL1=(result,field)=>{
    const e=errorItem(result,field);
    return number(e.normalized_l1??e.normalized_L1??e.l1_normalized??e.l1_scaled??e.scaled_l1);
  };
  const largestError=result=>max(Object.keys(result.errors).map(field=>normalizedL1(result,field)));
  const stat=(label,value,description,kind='')=>`<article class="stat panel"><p class="stat-label">${esc(label)}</p><p class="stat-value ${kind}">${esc(value)}</p><p class="stat-description">${esc(description)}</p></article>`;
  function renderRoeValidation(){
    const audit=payload.roe_validation;
    if(!audit)return;
    const summary=audit.summary||{},settings=audit.settings||{},source=audit.campaign_source_comparison||{};
    $('roe-validation').hidden=false;
    const passed=audit.algebra_checks_passed===true;
    $('roe-validation-status').className='status-badge '+(passed&&source.verified?'pass':'warn');
    $('roe-validation-status').textContent=passed?'Recorded algebra checks passed':'Inspect retained audit failures';
    $('roe-validation-description').textContent=`${settings.fluid||'Recorded fluid'} / ${settings.backend||'recorded backend'}; ${settings.numerical_input_pair||'PH property calls'}. Algebra and host derivative checks are separate from the perfect-gas analytical flow suite.`;
    $('roe-validation-cards').innerHTML=stat('State-pair checks',fmt(summary.pair_checks_passed,0),`${fmt(summary.pair_checks_failed,0)} failed; ${fmt(summary.invalid_roe_pairs,0)} invalid`)+stat('Tiny jumps',`${fmt(summary.tiny_jump_valid,0)} / ${fmt(summary.tiny_jump_total,0)}`,'Recorded near-critical PH and roundoff checks')+stat('Property failures',fmt(summary.property_failure_count,0),'Saved endpoint/property evaluation failures')+stat('Host JVP mismatch',sci(summary.host_jvp_relative_mismatch_max),'Maximum relative matrix Frobenius mismatch; see finite-difference caveat');
    const notices=[];
    if(!passed)notices.push(notice('The supplied Roe audit retains failed or invalid checks. Inspect its complete evidence before accepting the implementation.','danger'));
    if(!source.verified)notices.push(notice('The audit source fingerprints are not fully matched to this campaign. The thermodynamic evidence is shown with its own recorded provenance; it does not certify the campaign revision.','warning'));
    array(audit.limitations).forEach(message=>notices.push(notice(message)));
    $('roe-validation-notices').innerHTML=notices.join('');
    $('roe-validation-table').innerHTML=array(audit.pair_audits).map(item=>`<tr><td>${esc(item.kind)}</td><td class="num">${fmt(item.pair_count,0)}</td><td class="num">${fmt(item.passed,0)}</td><td class="num">${fmt(item.invalid,0)}</td><td class="num">${sci(item.mixed_matrix_jump_defect?.max)}</td><td class="num">${sci(item.eigendecomposition_defect?.max)}</td><td class="num">${sci(item.energy_shift_matrix_defect?.max)}</td><td class="num">${sci(item.energy_shift_projector_defect?.max)}</td><td class="num">${sci(item.energy_shift_speed_defect?.max)}</td></tr>`).join('');
    $('roe-validation-interpretation').textContent=`The recorded audit checks the pressure secant, conservative matrix jump, projectors and eigenstructure for nozzle and annular Euler pairs. It repeats the matrix/projector/speed checks after an enthalpy reference shift of ${fmt(settings.energy_reference_shift,0)} J/kg. These are algebra/property checks; they do not establish real-gas flow convergence, positivity or a nonlinear entropy theorem.`;
    $('roe-validation-jvp').textContent=`The host JVP comparison is a separate backend diagnostic: the configured JAXProp host bridge uses one-sided property finite differences with step 1e-6*(abs(input)+1). Its maximum relative matrix mismatch is ${sci(summary.host_jvp_relative_mismatch_max)}. This is finite-difference derivative agreement, not a claim of exact automatic differentiation through the property library.`;
    $('roe-validation-archive').textContent=`The complete original JSON, including every state-pair record, is embedded byte for byte at ${audit.evidence?.archive_entry||'the declared validation entry'} in the Downloads evidence ZIP. SHA256: ${audit.evidence?.sha256||'unavailable'}.`;
    $('roe-validation-json').textContent=JSON.stringify(audit,null,2);
    $('roe-audit-download').addEventListener('click',()=>document.querySelector('a[download="benchmark_evidence.zip"]').click());
  }
  function renderFlowValidation(){
    const data=payload.roe_flow_validation;if(!data)return;
    const sod=data.sod,gas=array(data.real_gas),assessment=sod?.assessment||{},sodRows=array(sod?.records);
    const numerical=assessment.numerical_checks_passed??sodRows.filter(r=>r.classification?.numerical_passed===true).length;
    const strict=assessment.analytic_boundary_transfer_checks_passed??sodRows.filter(r=>r.passed===true).length;
    const pass=gas.filter(r=>r.steady_verified===true).length,budgets=gas.filter(r=>r.budget_incomplete===true).length;
    const flag=value=>value===true?'yes':value===false?'no':'unavailable';
    const vector=values=>array(values).map(v=>sci(v)).join(' / ')||'unavailable';
    $('roe-flow-validation').hidden=false;
    $('roe-flow-cards').innerHTML=stat('Sod numerical checks',sod?`${numerical} / ${sodRows.length}`:'Not supplied','Accuracy, admissibility, BC and actual discrete budget')+stat('Sod original strict checks',sod?`${strict} / ${sodRows.length}`:'Not supplied','Undisturbed analytic endpoint-transfer criterion; failed cases retained')+stat('Actual-EOS steady roots',`${pass} / ${gas.length}`,'Completed original spatial/BC/global steady checks')+stat('Retained budget stops',budgets,'Incomplete attempts; not final steady roots');
    if(sod){
      $('roe-sod-section').hidden=false;
      $('roe-sod-summary').textContent=`${sodRows.filter(r=>r.success===true).length}/${sodRows.length} completed physical-time solves; ${numerical}/${sodRows.length} pass the numerical checks and ${strict}/${sodRows.length} pass the original strict analytic boundary-transfer criterion. Original aggregate passed flag: ${String(sod.passed)}. Grids: ${array(sod.settings?.grids).join(', ')}; CFL 0.5; constant reconstruction.`;
      const orders=array(sod.refinement).flatMap(r=>array(r.observed_order)).filter(finite),differences=array(sod.implicit_boundary_comparisons).flatMap(r=>array(r.normalized_max_primitive_difference)).filter(finite);
      $('roe-sod-refinement').textContent=`All saved conservative L1 errors decrease on refinement: ${flag(assessment.all_errors_reduce_on_refinement)}. Recorded orders ${orders.length?fmt(Math.min(...orders),2)+' to '+fmt(Math.max(...orders),2):'unavailable'}. Coupled/segregated backward-Euler agreement passed: ${flag(assessment.implicit_boundary_agreement_passed)}; maximum normalized primitive difference ${sci(max(differences))}.`;
      $('roe-sod-caveat').textContent=assessment.interpretation||'The original strict analytic endpoint-transfer criterion is distinct from discrete conservation, which uses the accepted numerical endpoint transfers. Failed strict checks remain in the record. Exact cell-reference quadrature has the saved 64-vs128 sensitivity; these are perfect-gas transient checks, not real-gas flow accuracy.';
      const labels={euler_explicit:'Explicit Euler / segregated',rk3:'SSPRK3 / segregated',implicit_coupled:'Backward Euler / coupled',implicit_segregated:'Backward Euler / segregated'};
      $('roe-sod-table').innerHTML=sodRows.map(r=>`<tr><td>${fmt(r.cells,0)}</td><td>${esc(labels[r.method?.id]||r.method?.id)}</td><td>${fmt(r.accepted_steps,0)} / ${fmt(r.rejected_steps,0)}</td><td>${r.classification?.numerical_passed===true?'passed':'inspect saved checks'}</td><td>${r.passed===true?'passed':'failed; retained'}</td><td>${esc(vector(r.conservative_errors?.L1))}</td><td>${sci(max(array(r.independent_summed_transfer_budget).map(Math.abs)))}</td></tr>`).join('');
      if(data.sod_figure){$('roe-sod-figure').hidden=false;$('roe-sod-image').src=data.sod_figure;}
    }
    if(gas.length){
      $('roe-real-gas-section').hidden=false;
      $('roe-real-gas-summary').textContent=`${pass} completed coarse steady roots and ${budgets} retained incomplete budget stops across ${gas.length} attempts. Both models use CO2/HEOS at reservoir 8 MPa / 320 K and backpressure 7.98 MPa, acoustic outlets, constant Roe reconstruction and coupled backward Euler. The follow-up changes cell count only.`;
      $('roe-real-gas-limits').textContent='A converged accepted-step Newton root is distinct from convergence of the original steady equations. Budget rows show their last sampled norms and do not claim a final model/positive-property audit. A five-cell pass does not replace the 25-cell verification; no analytical real-gas flow-accuracy or runtime ranking is established. Actual host property derivatives use the existing finite-difference JVP bridge.';
      const manifests=array(data.campaign_source_comparisons);if(manifests.length)$('roe-real-gas-limits').textContent+=` Saved production-source manifests match this campaign: ${flag(manifests.every(item=>item.matched))}; ${manifests.reduce((sum,item)=>sum+item.matched_files,0)} matched, ${manifests.reduce((sum,item)=>sum+item.mismatch_count,0)} mismatched and ${manifests.reduce((sum,item)=>sum+item.unavailable_count,0)} unavailable recorded files. Aggregate hashes from different manifest formats are not compared.`;
      $('roe-real-gas-table').innerHTML=gas.map(r=>`<tr><td>${esc(r.case)} / ${fmt(r.cells,0)}</td><td>${r.steady_verified?'steady verified':r.budget_incomplete?'budget stop; incomplete':'failed; retained'}</td><td>${fmt(r.accepted_steps,0)} / ${fmt(r.rejected_steps,0)}</td><td>${flag(r.accepted_newton_status_residual_audit)}</td><td>${flag(r.all_equations_passed)}</td><td>${esc(vector(r.equation_l2))}</td><td>${sci(r.boundary_max)} (${flag(r.boundary_passed)})</td><td>${sci(r.global_max)} (${flag(r.global_passed)})</td><td>${flag(r.admissible)}</td><td>${flag(r.model_audit)} / ${flag(r.positive_properties)}</td><td>${flag(r.numerical_ph_only)}; ${fmt(r.numerical_ph_calls,0)} calls</td><td>${flag(r.source_unchanged)}</td></tr>`).join('');
    }
    $('roe-flow-archive').textContent=`${array(data.evidence).length} original flow artifacts are embedded byte for byte in the evidence ZIP, including the complete JSONs and declared actual-EOS raw histories/states. Their SHA256 values are checked during report generation. No flow solver is imported by this report generator.`;
    $('roe-flow-json').textContent=JSON.stringify({...data,sod_figure:data.sod_figure?'Inline PNG; exact original in evidence ZIP':null},null,2);
    $('roe-flow-download').addEventListener('click',()=>document.querySelector('a[download="benchmark_evidence.zip"]').click());
  }
  function notice(text,kind=''){return `<div class="notice ${kind}">${esc(text)}</div>`;}
  function option(select,value,label){const item=document.createElement('option');item.value=String(value);item.textContent=label;select.append(item);}
  function setOptions(select,values,current,label){select.replaceChildren();values.forEach(value=>option(select,value,label(value)));select.value=values.includes(current)?String(current):String(values[values.length-1]??'');}
  let activePage='summary', selectedCase=cases[0]?.id||'', theme='light';
  try{theme=localStorage.getItem('nh-benchmark-theme')||'light';}catch(_error){}
  document.documentElement.dataset.theme=theme;
  $('theme-toggle').textContent=theme==='dark'?'Light theme':'Dark theme';
  const colors={legacy:{constant:'#6a00a8',muscl_van_leer:'#9c179e',muscl_van_leer_smooth:'#9c179e'},current:{constant:'#dc5e66',muscl_van_leer:'#fca636',muscl_van_leer_smooth:'#fca636'}};
  function style(result){const kind=backendKind(result.backend),fluxColors={roe:'#6a00a8',rusanov:'#b12a90',hll:'#e16462',stadtke:'#fca636'};return {color:multiFlux?fluxColors[result.flux]||'#b12a90':(colors[kind]||colors.current)[result.reconstruction]||'#8c6bb1',dash:kind==='legacy'?'dash':multiFlux&&result.reconstruction!=='constant'?'dash':'solid',width:result.reconstruction==='constant'?2:2.6};}
  function rowLabel(result,extra=false){return `${backendLabel(result.backend)} · ${fluxLabel(result.flux)} · ${reconLabel(result.reconstruction)}${extra?` · ${result.cells} cells · cap ${result.cap??'—'}`:''}`;}
  function fieldLabel(field){const c=caseMap.get(selectedCase)||{};const info=c.field_metadata?.[field]||c.fields?.[field];const label=typeof info==='object'?info.label:null;const units=typeof info==='object'?info.units:null;const known=knownFields[field]||[field,''];return {label:label||known[0],units:units??known[1]};}
  const plotIds=['speed-chart','error-chart','regression-chart','field-chart','residual-chart','cfl-chart','refinement-chart','mesh-timing-chart','samples-chart'];
  let plotGeneration=0;
  const pendingPlots=new Map();
  function palette(){const style=getComputedStyle(document.documentElement);return {paper:style.getPropertyValue('--paper').trim(),ink:style.getPropertyValue('--ink').trim(),muted:style.getPropertyValue('--muted').trim(),grid:style.getPropertyValue('--grid').trim()};}
  function draw(id,traces,layout={},empty='No recorded data match this selection.'){
    const element=$(id), p=palette();
    if(!traces.length){if(element.data)Plotly.purge(element);element.innerHTML=`<div class="plot-empty"><strong>No plotted series</strong><span>${esc(empty)}</span></div>`;return;}
    if(!element.data)element.replaceChildren();
    const base={autosize:true,paper_bgcolor:p.paper,plot_bgcolor:p.paper,font:{family:'Segoe UI, Arial, sans-serif',size:11,color:p.ink},margin:{l:67,r:25,t:26,b:80},showlegend:true,legend:{orientation:'h',y:-.25,x:0,font:{size:10}},hovermode:'closest',xaxis:{gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,title:{font:{size:11}}},yaxis:{gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,title:{font:{size:11}}},...layout};
    base.xaxis={gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,...layout.xaxis};base.yaxis={gridcolor:p.grid,zerolinecolor:p.grid,automargin:true,...layout.yaxis};
    const generation=plotGeneration;pendingPlots.set(generation,(pendingPlots.get(generation)||0)+1);
    const promise=Plotly.react(element,traces,base,{responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'svg',filename:'benchmark_'+id,width:1200,height:650},modeBarButtonsToRemove:['lasso2d','select2d']});
    Promise.resolve(promise).catch(error=>{console.error(error);element.innerHTML=`<div class="plot-empty"><strong>Plot unavailable</strong><span>${esc(error.message||error)}</span></div>`;}).finally(()=>{pendingPlots.set(generation,Math.max(0,(pendingPlots.get(generation)||0)-1));if(pendingPlots.get(generation)===0&&generation===plotGeneration)window.benchmarkPlotsReady=true;});
  }
  function beginPlots(){plotGeneration++;pendingPlots.set(plotGeneration,0);window.benchmarkPlotsReady=false;}
  function finishPlots(){if(pendingPlots.get(plotGeneration)===0)window.benchmarkPlotsReady=true;}
  function modelName(result){return modelLabel(result.model||caseMap.get(result.caseId)?.model);}
  function identity(result){return [result.caseId,result.reconstruction,result.cells,result.cap,result.flux].join('|');}
  function performanceIdentity(result){return [identity(result),backendKind(result.backend)].join('|');}
  function context(data){
    const p=data?.provenance||{},protocol=data?.protocol||{},machine=p.machine||{};
    return {machine_sha256:p.machine_sha256||data?.machine_sha256,
      platform:p.platform||p.environment?.platform||machine.platform||machine.system,
      release:machine.release,machine:machine.machine,
      processor:p.processor||p.environment?.processor||machine.processor,
      python:p.python||p.environment?.python||machine.python,
      packages:p.versions||p.packages||p.environment?.versions||machine.packages,
      precision:p.precision||p.dtype||p.environment?.precision||protocol.precision||protocol.dtype,
      threads:p.threads||p.environment?.threads||machine.threads||protocol.threads,
      device:p.device||p.environment?.device||protocol.device};
  }
  function numericalIdentity(result){
    const raw=result.raw;
    return {settings:raw.settings||null,scheme:raw.scheme||raw.settings?.scheme||null,
      steady:raw.steady||raw.settings?.steady||null,residual:raw.residual||raw.residual_policy||raw.settings?.residual||null,
      norm:raw.norm||raw.settings?.norm||null,
      profile_sha256:raw.numerical_profile_sha256||raw.profile_sha256||raw.provenance?.numerical_profile_sha256||null};
  }
  function baselineMatch(result){
    const match=baselineResults.find(item=>performanceIdentity(item)===performanceIdentity(result));
    if(!match)return {matched:false,reason:'No matching baseline configuration'};
    const beforeHash=match.raw.case_sha256||match.raw.case_hash||match.raw.provenance?.case_sha256;
    const afterHash=result.raw.case_sha256||result.raw.case_hash||result.raw.provenance?.case_sha256;
    if(beforeHash&&afterHash&&beforeHash!==afterHash)return {matched:false,reason:'Physical case definition changed'};
    const beforeSettings=numericalIdentity(match),afterSettings=numericalIdentity(result);
    const settingsKnown=!!(match.raw.settings&&result.raw.settings);
    if(stable(beforeSettings)!==stable(afterSettings))return {matched:false,reason:'Numerical settings, scheme, residual policy, or profile identity changed'};
    const a=context(baseline),b=context(campaign);const environmentKnown=!!(((a.machine_sha256&&b.machine_sha256)||(a.platform&&b.platform&&a.processor&&b.processor))&&a.python&&b.python&&a.packages&&b.packages&&a.precision&&b.precision&&a.threads&&b.threads);
    const environmentSame=stable(a)===stable(b);
    if(!environmentSame)return {matched:false,reason:'Machine, environment, precision, or thread configuration changed',match};
    if(!good(result)||!good(match))return {matched:false,reason:'A compared configuration did not fully pass',match};
    return {matched:true,match,ratio:result.timing.mean_s/match.timing.mean_s,verified:environmentKnown&&environmentSame&&settingsKnown&&!!(beforeHash&&afterHash)};
  }
  function filtered(){
    let rows=results.filter(r=>($('summary-model').value==='all'||String(r.model)===$('summary-model').value)&&($('summary-reconstruction').value==='all'||r.reconstruction===$('summary-reconstruction').value)&&($('summary-cap').value==='all'||String(r.cap)===$('summary-cap').value)&&($('summary-flux').value==='all'||r.flux===$('summary-flux').value));
    if($('summary-grid').value==='finest'){
      const largest=new Map();rows.forEach(r=>largest.set(r.caseId,Math.max(largest.get(r.caseId)||0,r.cells||0)));rows=rows.filter(r=>r.cells===largest.get(r.caseId));
    }else if($('summary-grid').value!=='all')rows=rows.filter(r=>String(r.cells)===$('summary-grid').value);
    return rows;
  }
  function speedPairs(rows){const groups=new Map();rows.forEach(r=>{const key=identity(r), group=groups.get(key)||{};group[backendKind(r.backend)]=r;groups.set(key,group);});return [...groups.values()].filter(g=>g.current&&g.legacy&&good(g.current)&&good(g.legacy)).map(g=>({...g,ratio:g.current.timing.mean_s/g.legacy.timing.mean_s}));}
  function comparisonPairs(rows){
    if(!multiFlux)return speedPairs(rows);
    const groups=new Map();
    rows.filter(r=>backendKind(r.backend)==='current').forEach(r=>{const key=[r.caseId,r.reconstruction,r.cells,r.cap].join('|'),group=groups.get(key)||{};group[r.flux]=r;groups.set(key,group);});
    return [...groups.values()].flatMap(group=>{const first=Object.values(group)[0],reference=group[referenceFlux]||results.find(r=>r.caseId===first.caseId&&r.reconstruction===first.reconstruction&&r.cells===first.cells&&r.cap===first.cap&&backendKind(r.backend)==='current'&&r.flux===referenceFlux);return reference&&good(reference)?Object.values(group).filter(r=>r.flux!==referenceFlux&&good(r)).map(r=>({current:r,legacy:reference,ratio:r.timing.mean_s/reference.timing.mean_s})):[];});
  }
  function renderSummary(){
    beginPlots();const rows=filtered(),goodRows=results.filter(good),failed=results.filter(r=>outcome(r).kind==='fail'),warnings=results.filter(r=>outcome(r).kind==='warn'),pairs=comparisonPairs(rows);
    const ratio=pairs.length?pairs.reduce((sum,p)=>sum+p.current.timing.mean_s,0)/pairs.reduce((sum,p)=>sum+p.legacy.timing.mean_s,0):null;
    $('summary-cards').innerHTML=stat('Physical cases',cases.length,`${unique(cases.map(c=>c.model)).length} model families · versioned definitions`)+stat('Recorded configurations',results.length,`${goodRows.length} eligible repeated timings · ${failed.length} failed`,failed.length?'bad':'')+stat('Measured solve samples',results.reduce((sum,r)=>sum+r.timing.samples_s.length,0),'Complete solves; warmups excluded')+stat(multiFlux?'Selected flux / FVS time':'Current / native time',ratio===null?'—':fmt(ratio,2)+'×',`${pairs.length} successful matched pairs in this view`);
    if(verificationOnly){$('summary-cards').children[1].querySelector('.stat-description').textContent=`${results.filter(r=>r.success===true).length} solve successes; ${failed.length} failed numerical/branch records`;$('summary-cards').children[2].innerHTML=`<p class="stat-label">Complete verification solves</p><p class="stat-value">${results.filter(r=>r.raw.raw_result_file).length}</p><p class="stat-description">One cold solve per configuration; no timing warmups or repetitions</p>`;$('summary-cards').children[3].innerHTML='<p class="stat-label">Performance timing</p><p class="stat-value">Not collected</p><p class="stat-description">Numerical-only verification; load-dependent elapsed diagnostics excluded</p>';}
    const complete=campaign.completed??campaign.complete;
    $('campaign-status').className='status-badge '+(failed.length?'fail':warnings.length||complete===false?'warn':'pass');
    $('campaign-status').textContent=complete===false?'Campaign incomplete':failed.length?`${failed.length} failed configurations`:warnings.length?`${warnings.length} limited / unavailable`:'Recorded campaign complete';
    const notes=[];
    if(verificationOnly)notes.push(notice('This is numerical-only verification. Each configuration receives one cold solve; performance samples, excluded timing warmups and speed rankings were not collected. Concurrent workloads are permitted.'));
    if(complete===false)notes.push(notice('This campaign is incomplete. The report shows the retained attempts and missing configurations rather than presenting a partial run as the complete suite.','warning'));
    if(unchanged(campaign,'source_unchanged')===false||unchanged(campaign,'harness_unchanged')===false)notes.push(notice('Measured source or benchmark code changed during this campaign. All retained timings are invalid for comparison; no speed ratio is reported. Inspect the Protocol page for the recorded changes.','danger'));
    if(!verificationOnly&&timingEnvironmentStatus(campaign.timing_environment)!=='clean')notes.push(notice('The campaign timing environment is not verified clean. Retained timings are shown as evidence and excluded from comparative speed claims.','warning'));
    if(failed.length)notes.push(notice(`${failed.length} configuration(s) failed, timed out, or returned invalid states. Their measured work remains in the table and data downloads; they are excluded from success-based speed ratios.`,'danger'));
    if(warnings.length)notes.push(notice(verificationOnly&&warnings.every(r=>r.status==='accuracy_failed')?`${warnings.length} converged configurations miss their stated analytical accuracy targets. The original targets remain unchanged; inspect their refinement and reconstruction results.`:`${warnings.length} configuration(s) have accuracy limits, invalid or unverified timings, missing detail data, or unsupported status. Inspect their case evidence before interpreting timing.`,'warning'));
    if(rows.some(r=>r.accuracyAssessed===false))notes.push(notice('Coarse meshes report analytical errors for refinement only. Accuracy acceptance targets are assessed at the recommended finest mesh; a valid coarse-grid solve or timing is not an accuracy certificate.'));
    if(campaign.notes)notes.push(notice(typeof campaign.notes==='string'?campaign.notes:JSON.stringify(campaign.notes)));
    array(campaign.presentation_warnings).forEach(message=>notes.push(notice(message,'warning')));
    array(baseline?.presentation_warnings).forEach(message=>notes.push(notice('Saved baseline: '+message,'warning')));
    $('campaign-notices').innerHTML=notes.join('');$('summary-count').textContent=`${rows.length} configurations shown`;
    const speedTraces=[];
    unique(pairs.map(p=>p.current.flux+'|'+p.current.reconstruction+'|'+p.current.cap)).forEach(group=>{const [flux,recon,cap]=group.split('|');const selected=pairs.filter(p=>p.current.flux===flux&&p.current.reconstruction===recon&&String(p.current.cap)===cap);speedTraces.push({type:'bar',orientation:'h',name:`${multiFlux?fluxLabel(flux)+" · ":""}${reconLabel(recon)} · cap ${cap}`,y:selected.map(p=>caseLabel(p.current.caseId)),x:selected.map(p=>p.ratio),marker:{color:style(selected[0].current).color},customdata:selected.map(p=>[p.current.timing.mean_s,p.legacy.timing.mean_s,p.current.cells]),hovertemplate:'%{y}<br>Method/reference: %{x:.3f}×<br>Method: %{customdata[0]:.5f} s<br>Reference: %{customdata[1]:.5f} s<br>Cells: %{customdata[2]}<extra>%{fullData.name}</extra>'});});
    const caseCount=unique(pairs.map(p=>p.current.caseId)).length;
    $('speed-chart').style.height=Math.max(355,caseCount*33+130)+'px';
    draw('speed-chart',speedTraces,{barmode:'group',margin:{l:160,r:20,t:25,b:90},xaxis:{title:multiFlux?'Flux / Städtke-type FVS mean time':'Current / native mean time'},yaxis:{autorange:'reversed'},shapes:[{type:'line',xref:'x',yref:'paper',x0:1,x1:1,y0:0,y1:1,line:{color:palette().muted,dash:'dot',width:1}}]},'No repeated method/reference pair passed with the same case, mesh, reconstruction and Newton cap in a complete, stable, clean campaign.');
    $('speed-chart-title').textContent=multiFlux?'Flux runtime relative to current FVS':'Current versus native runtime';
    $('speed-chart-caption').textContent=multiFlux?'Matched case, mesh, reconstruction and Newton cap; successful production solves with at least two measured samples in a complete, stable, clean campaign. Above 1 means the selected flux took longer than original Stadtke FVS.':'Mean time ratio for comparable successful configurations with at least two measured samples in a complete, stable, clean campaign. Above 1 means the current solver took longer.';
    if(verificationOnly){
      $('speed-chart-title').textContent='Declared accuracy checks';
      $('speed-chart-caption').textContent='Acceptance is assessed at each case\'s recommended resolution. A converged root can miss an accuracy target; coarse meshes retain refinement errors without an accuracy pass/fail claim.';
      const assessed=rows.filter(r=>r.accuracyAssessed===true),reconstructions=unique(assessed.map(r=>r.reconstruction));
      const passed=reconstructions.map(name=>assessed.filter(r=>r.reconstruction===name&&r.accuracy===true).length),missed=reconstructions.map(name=>assessed.filter(r=>r.reconstruction===name&&r.accuracy!==true).length);
      draw('speed-chart',assessed.length?[{type:'bar',name:'Target met',x:reconstructions.map(reconLabel),y:passed,marker:{color:'#7e03a8'},text:passed,textposition:'auto'},{type:'bar',name:'Target missed; retained',x:reconstructions.map(reconLabel),y:missed,marker:{color:'#e16462'},text:missed,textposition:'auto'}]:[],{barmode:'stack',xaxis:{title:'Reconstruction'},yaxis:{title:'Assessed configurations',dtick:1},margin:{l:65,r:25,t:25,b:80}},'No declared accuracy acceptance is assessed at the selected coarse resolutions.');
    }
    const errorTraces=[];
    unique(rows.map(r=>r.backend+'|'+r.reconstruction+'|'+r.flux)).forEach(group=>{const [backend,recon,flux]=group.split('|');const selected=rows.filter(r=>r.backend===backend&&r.reconstruction===recon&&r.flux===flux&&finite(largestError(r)));if(selected.length)errorTraces.push({type:'scatter',mode:'markers',name:`${backendLabel(backend)} · ${fluxLabel(flux)} · ${reconLabel(recon)}`,x:selected.map(r=>caseLabel(r.caseId)),y:selected.map(largestError),marker:{color:style(selected[0]).color,size:8,symbol:backendKind(backend)==='legacy'?'diamond-open':'circle'},text:selected.map(r=>`${r.cells} cells · cap ${r.cap} · ${outcome(r).label}`),hovertemplate:'%{x}<br>Normalized L1: %{y:.3e}<br>%{text}<extra>%{fullData.name}</extra>'});});
    $('error-chart').style.height=Math.max(355,caseCount*33+130)+'px';
    draw('error-chart',errorTraces,{xaxis:{tickangle:-40},yaxis:{type:'log',title:'Max available normalized L1 field error'},margin:{l:75,r:25,t:25,b:180}},'Normalized analytical errors have not been recorded for the selected configurations.');
    renderRegression(rows);
    $('summary-table').innerHTML=rows.length?rows.sort((a,b)=>caseLabel(a.caseId).localeCompare(caseLabel(b.caseId))||(a.cells-b.cells)||a.reconstruction.localeCompare(b.reconstruction)||a.backend.localeCompare(b.backend)).map(r=>{const state=outcome(r);return `<tr><td><button class="case-open" data-case="${esc(r.caseId)}">${esc(caseLabel(r.caseId))}</button></td><td>${esc(modelName(r))}</td><td>${esc(backendLabel(r.backend))}</td><td>${esc(fluxLabel(r.flux))}</td><td>${esc(reconLabel(r.reconstruction))}</td><td class="num">${fmt(r.cells,0)}</td><td class="num">${fmt(r.cap,0)}</td><td><span class="badge ${state.kind}">${esc(state.label)}</span></td><td class="num">${sci(largestError(r))}</td><td class="num">${fmt(r.timing.mean_s,5)}</td><td class="num">${fmt(r.timing.min_s,5)}–${fmt(r.timing.max_s,5)}</td><td class="num">${r.timing.samples_s.length}</td></tr>`;}).join(''):'<tr><td colspan="12" class="table-empty">No recorded configurations match the filters.</td></tr>';
    $('summary-table').querySelectorAll('[data-case]').forEach(button=>button.addEventListener('click',()=>openCase(button.dataset.case)));
    finishPlots();
  }
  function renderRegression(rows){
    if(verificationOnly){$('baseline-caption').textContent='No performance timing was collected in this numerical-only verification.';draw('regression-chart',[],{},'Performance regression requires a separate clean repeated-timing campaign.');return;}
    if(!baseline){$('baseline-caption').textContent='No prior campaign was supplied. This report establishes a baseline; use --baseline <prior summary.json> in a later report to monitor changes.';draw('regression-chart',[],{},'No historical performance baseline supplied.');return;}
    const matches=rows.map(r=>({result:r,...baselineMatch(r)})),usable=matches.filter(m=>m.matched),verified=usable.filter(m=>m.verified),unmatched=matches.filter(m=>!m.matched);
    const threshold=report.regression_threshold;
    $('baseline-caption').textContent=`${usable.length} matched configurations, including ${verified.length} with verified case/settings/environment identity. ${unmatched.length} were not comparable. Alert threshold: ${fmt(threshold,2)}×. ${usable.some(m=>!m.verified)?'Unverified ratios are exploratory and do not establish a code regression.':''}`;
    const traces=usable.length?[{type:'bar',x:usable.map(m=>`${caseLabel(m.result.caseId)}<br>${backendLabel(m.result.backend)} · ${reconLabel(m.result.reconstruction)}`),y:usable.map(m=>m.ratio),marker:{color:usable.map(m=>m.ratio>threshold?'#d15a65':m.verified?'#7e03a8':'#c79b52')},text:usable.map(m=>m.verified?'Verified identity':'Identity incomplete'),hovertemplate:'%{x}<br>Current/baseline: %{y:.3f}×<br>%{text}<extra></extra>'}]:[];
    draw('regression-chart',traces,{showlegend:false,yaxis:{title:'Current / saved-baseline mean time'},margin:{l:65,r:25,t:25,b:170},shapes:[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:threshold,y1:threshold,line:{color:'#c79b52',dash:'dot',width:1}}]},unmatched.length?unique(unmatched.map(m=>m.reason)).join('; '):'No matching fully successful baseline configurations.');
  }
  function updateCaseOptions(reset=true){
    const rows=results.filter(r=>r.caseId===selectedCase);
    const oldGrid=number($('case-grid').value),oldCap=number($('case-cap').value),oldField=$('field-select').value;
    setOptions($('case-grid'),sortedNumbers(rows.map(r=>r.cells)),reset?null:oldGrid,v=>`${v} physical cells`);
    setOptions($('case-cap'),sortedNumbers(rows.map(r=>r.cap)),reset?null:oldCap,v=>`Cap ${v}`);
    const fieldNames=unique(rows.flatMap(r=>[...Object.keys(r.state.fields),...Object.keys(r.reference.fields),...Object.keys(r.errors)]));
    const preferred=fieldNames.includes('p')?'p':fieldNames[0];
    setOptions($('field-select'),fieldNames,reset?preferred:oldField,v=>fieldLabel(v).label);
  }
  function caseFilter(r){return r.caseId===selectedCase&&String(r.cap)===$('case-cap').value&&($('case-flux').value==='all'||r.flux===$('case-flux').value)&&($('case-reconstruction').value==='all'||r.reconstruction===$('case-reconstruction').value);}
  function selectedRows(){return results.filter(r=>caseFilter(r)&&String(r.cells)===$('case-grid').value);}
  function historyX(row,index){return $('history-axis').value==='time'?number(row.elapsed_s??row.time_s??row.wall_s):number(row.step??row.iteration??row.update??row.accepted_step)??index;}
  function residualValue(row){
    const direct=number(row.residual??row.spatial_rms??row.residual_rms??row.max_residual??row.spatial_norm);
    if(direct!==null)return direct;
    const values=row.equation_norm||row.equation_norms||row.equation_residual||row.residuals||row.norms;
    return max(Array.isArray(values)?values:Object.values(values||{}));
  }
  function metricText(result){const m=result.metrics;const fields=[['BC',m.boundary_max??m.bc_max??m.boundary],['global',max(array(m.global_defect||m.global_balance))??m.global_max],['spatial',max(array(m.equation_norm||m.equation_norms))??m.spatial_rms]];return fields.filter(([,v])=>number(v)!==null).map(([name,value])=>name+' '+sci(value)).join(' · ')||result.raw.reason||result.status;}
  function renderCase(){
    beginPlots();const definition=caseMap.get(selectedCase)||{id:selectedCase},rows=selectedRows(),allRows=results.filter(caseFilter),field=$('field-select').value,info=fieldLabel(field);
    $('case-title').textContent=caseLabel(selectedCase);$('case-model').textContent=modelLabel(definition.model||rows[0]?.model);$('case-description').textContent=definition.description||definition.summary||'';
    const notes=[];if(verificationOnly)notes.push(notice('Numerical-only verification: no timing samples or rankings. Root convergence, physical branch and analytical accuracy remain separate checks.'));const validation=definition.validation||definition.expected||{};
    const limitation=definition.known_limit||definition.known_limitation||validation.known_limit||validation.known_limitation;
    if(limitation)notes.push(notice('Known case limitation: '+(typeof limitation==='string'?limitation:JSON.stringify(limitation)),'warning'));
    if(!rows.length)notes.push(notice('No retained result matches this case, mesh, and cap.','warning'));
    if(rows.some(r=>r.accuracyAssessed===false))notes.push(notice('Analytical acceptance targets are not assessed at this coarse mesh. The errors below describe refinement behavior; accuracy pass/fail applies at the case\'s recommended finest mesh.'));
    rows.filter(r=>outcome(r).kind!=='pass').forEach(r=>notes.push(notice(`${rowLabel(r)}: ${outcome(r).label}. ${r.raw.reason||r.raw.error||''}`,outcome(r).kind==='fail'?'danger':'warning')));
    if(!verificationOnly&&rows.some(r=>r.timing.samples_s.length<2))notes.push(notice('At least one selected configuration has fewer than two measured solves. Its timing gives cost context; repeated timing variability is not established.'));
    $('case-notices').innerHTML=notes.join('');
    const pairs=comparisonPairs(rows),ratio=pairs.length?pairs.reduce((sum,p)=>sum+p.current.timing.mean_s,0)/pairs.reduce((sum,p)=>sum+p.legacy.timing.mean_s,0):null;
    $('case-cards').innerHTML=stat('Selected mesh',fmt(number($('case-grid').value),0),'Physical cells; boundary traces are separate')+stat('Passed configurations',`${rows.filter(r=>outcome(r).kind==='pass').length} / ${rows.length}`,'Analytical accuracy remains separately reported')+stat('Largest normalized L1',sci(max(rows.map(largestError))),`${info.label} and other available fields`)+stat(multiFlux?'Selected flux / FVS time':'Current / native time',ratio===null?'—':fmt(ratio,2)+'×',`${pairs.length} successful reconstruction pair(s)`);
    if(verificationOnly)$('case-cards').children[3].innerHTML='<p class="stat-label">Performance timing</p><p class="stat-value">Not collected</p><p class="stat-description">Fields and convergence only; no speed ratio</p>';
    const fieldTraces=[],seenReference=new Set();
    rows.forEach(r=>{
      const ref=r.reference.fields[field],x=r.reference.x;
      if(ref&&x.length===ref.length){const key=JSON.stringify([x,ref]);if(!seenReference.has(key)){seenReference.add(key);fieldTraces.push({type:'scatter',mode:'lines',name:seenReference.size===1?'Analytical reference':`Analytical · ${backendLabel(r.backend)}`,x,y:ref,line:{color:palette().muted,dash:'dot',width:2.5},hovertemplate:'x = %{x:.5g} m<br>%{y:.6g}<extra>Analytical reference</extra>'});}}
      const values=r.state.fields[field];if(values&&r.state.x.length===values.length)fieldTraces.push({type:'scatter',mode:'lines+markers',name:rowLabel(r),x:r.state.x,y:values,line:style(r),marker:{size:3,color:style(r).color},connectgaps:false,hovertemplate:'x = %{x:.5g} m<br>%{y:.6g}<extra>%{fullData.name}</extra>'});
    });
    $('field-chart-title').textContent=`${info.label}: numerical and analytical`;$('field-chart-caption').textContent=`Selected fluxes, backends and reconstructions are overlaid at the selected mesh. ${seenReference.size>1?'Backend-specific analytical sampling is retained. ':''}Legend entries toggle individual curves; every graph can be exported as SVG.`;
    draw('field-chart',fieldTraces,{xaxis:{title:definition.model?.includes('annular')?'Meridional coordinate [m]':'Axial coordinate [m]'},yaxis:{title:info.label+(info.units?' ['+info.units+']':'')},margin:{l:85,r:30,t:25,b:85}},'Numerical and/or analytical field arrays are unavailable for this selection.');
    const axisTitle=$('history-axis').value==='time'?'Elapsed solve time [s]':'Recorded step / attempt index';
    const historyTraces=(selector,log)=>rows.flatMap(r=>{const points=r.history.map((h,index)=>({x:historyX(h,index),y:selector(h),accepted:h.accepted??(h.phase!=='rejected')})).filter(p=>finite(p.x)&&finite(p.y)&&(!log||p.y>0));return points.length?[{type:'scatter',mode:'lines',name:rowLabel(r),x:points.map(p=>p.x),y:points.map(p=>p.y),line:style(r),hovertemplate:'%{x:.6g}<br>%{y:.4e}<extra>%{fullData.name}</extra>'}]:[];});
    draw('residual-chart',historyTraces(residualValue,true),{xaxis:{title:axisTitle},yaxis:{type:'log',title:'Max recorded fixed-scale equation residual'}},'Residual histories have not been recorded for this selection.');
    draw('cfl-chart',historyTraces(h=>number(h.cfl??h.CFL??h.cfl_requested??h.requested_cfl),true),{xaxis:{title:axisTitle},yaxis:{type:'log',title:'CFL'}},'CFL histories have not been recorded for this selection.');
    const refinement=[],timing=[];
    unique(allRows.map(r=>r.backend+'|'+r.reconstruction+'|'+r.flux)).forEach(group=>{
      const [backend,recon,flux]=group.split('|');const ordered=allRows.filter(r=>r.backend===backend&&r.reconstruction===recon&&r.flux===flux).sort((a,b)=>a.cells-b.cells);const errorRows=ordered.filter(r=>finite(normalizedL1(r,field))&&normalizedL1(r,field)>0);const timed=ordered.filter(r=>finite(r.timing.mean_s));
      if(errorRows.length)refinement.push({type:'scatter',mode:'lines+markers',name:`${backendLabel(backend)} · ${fluxLabel(flux)} · ${reconLabel(recon)}`,x:errorRows.map(r=>r.cells),y:errorRows.map(r=>normalizedL1(r,field)),line:style(ordered[0]),marker:{size:7},text:errorRows.map(r=>outcome(r).label),hovertemplate:'%{x} cells<br>Normalized L1: %{y:.3e}<br>%{text}<extra>%{fullData.name}</extra>'});
      if(timed.length)timing.push({type:'scatter',mode:'lines+markers',name:`${backendLabel(backend)} · ${fluxLabel(flux)} · ${reconLabel(recon)}`,x:timed.map(r=>r.cells),y:timed.map(r=>r.timing.mean_s),line:style(ordered[0]),marker:{size:7},error_y:{type:'data',symmetric:false,array:timed.map(r=>Math.max(0,(r.timing.max_s??r.timing.mean_s)-r.timing.mean_s)),arrayminus:timed.map(r=>Math.max(0,r.timing.mean_s-(r.timing.min_s??r.timing.mean_s))),thickness:1,width:3},text:timed.map(r=>`${r.timing.samples_s.length} samples · ${outcome(r).label}`),hovertemplate:'%{x} cells<br>Mean: %{y:.5f} s<br>%{text}<extra>%{fullData.name}</extra>'});
    });
    draw('refinement-chart',refinement,{xaxis:{type:'log',title:'Physical cells'},yaxis:{type:'log',title:info.label+' normalized L1 error'}},'At least one positive normalized field error is needed to plot refinement.');
    draw('mesh-timing-chart',timing,{xaxis:{type:'log',title:'Physical cells'},yaxis:{type:'log',title:'Complete solve time [s]'}},'No finite solve timings have been retained.');
    const samples=rows.filter(r=>r.timing.samples_s.length).map(r=>({type:'scatter',mode:'lines+markers',name:rowLabel(r),x:r.timing.samples_s.map((_,i)=>i+1),y:r.timing.samples_s,line:{...style(r),width:1},marker:{size:7},hovertemplate:'Measured repetition %{x}<br>%{y:.6f} s<extra>%{fullData.name}</extra>'}));
    draw('samples-chart',samples,{xaxis:{title:'Measured complete-solve repetition',dtick:1},yaxis:{title:'Solve time [s]'}},'Individual measured solve times were not recorded.');
    const errorRows=[];rows.forEach(r=>{const names=Object.keys(r.errors);if(!names.length)errorRows.push(`<tr><td>${esc(rowLabel(r))}</td><td colspan="5" class="muted">No field-error record</td><td>${esc(metricText(r))}</td></tr>`);names.forEach(name=>{const e=errorItem(r,name);errorRows.push(`<tr><td>${esc(rowLabel(r))}</td><td>${esc(fieldLabel(name).label)}</td><td class="num">${sci(e.l1??e.L1)}</td><td class="num">${sci(e.l2??e.L2)}</td><td class="num">${sci(e.linf??e.Linf??e.L_inf)}</td><td class="num">${sci(normalizedL1(r,name))}</td><td>${esc(metricText(r))}</td></tr>`);});});
    $('case-errors-table').innerHTML=errorRows.join('')||'<tr><td colspan="7" class="table-empty">No recorded result for this selection.</td></tr>';
    $('case-evidence').textContent=JSON.stringify({case:definition,selected_results:rows.map(r=>({backend:r.backend,flux:r.flux,reconstruction:r.reconstruction,cells:r.cells,cap:r.cap,status:r.status,reason:r.raw.reason,accuracy_assessed:r.accuracyAssessed,analytical_passed:r.raw.analytical_passed,accuracy_targets:r.raw.accuracy_targets,accuracy_checks:r.raw.accuracy_checks,branch_passed:r.raw.branch_passed,branch_checks:r.raw.branch_checks,shock:r.raw.shock,timing_valid:r.timingValid,timing_environment:r.raw.timing_environment,settings:r.raw.settings,scheme:r.raw.scheme,metadata:r.raw.metadata,work:r.raw.work,roe_diagnostics:r.raw.roe_diagnostics,provenance:r.raw.provenance,case_sha256:r.raw.case_sha256,metrics:r.metrics,errors:r.errors,timing:r.timing}))},null,2);
    finishPlots();
  }
  function openCase(id){selectedCase=id;$('case-select').value=id;updateCaseOptions(true);showPage('case');}
  const pages=['summary','case','protocol','theory'];
  function showPage(name,focus=false){if(!pages.includes(name))name='summary';activePage=name;for(const page of pages){$('page-'+page).hidden=page!==name;const tab=$('tab-'+page);tab.setAttribute('aria-selected',String(page===name));tab.tabIndex=page===name?0:-1;}if(name==='summary')renderSummary();else if(name==='case')renderCase();if(focus)$('tab-'+name).focus();window.scrollTo({top:0,behavior:'instant'});$('announcement').textContent=name==='case'?caseLabel(selectedCase)+' selected.':name+' page selected.';}
  function download(name,value,mime){const blob=new Blob([value],{type:mime});const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function csv(rows){return rows.map(row=>row.map(value=>'"'+String(value??'').replace(/"/g,'""')+'"').join(',')).join('\r\n')+'\r\n';}
  document.querySelectorAll('[data-download]').forEach(button=>button.addEventListener('click',()=>{
    if(button.dataset.download==='campaign')download('benchmark_campaign.json',JSON.stringify(campaign,null,2)+'\n','application/json');
    else if(button.dataset.download==='timings'){const rows=[['case_id','model','backend','flux','reconstruction','cells','newton_cap','status','timing_valid','timing_environment','campaign_completed','source_unchanged','harness_unchanged','repetition','time_s']];results.forEach(r=>r.timing.samples_s.forEach((time,i)=>rows.push([r.caseId,r.model,r.backend,r.flux,r.reconstruction,r.cells,r.cap,r.status,r.timingValid,timingEnvironmentStatus(r.raw.timing_environment),campaign.completed??campaign.complete,unchanged(campaign,'source_unchanged'),unchanged(campaign,'harness_unchanged'),i+1,time])));download('benchmark_timing_samples.csv',csv(rows),'text/csv;charset=utf-8');}
    else{const rows=[['case_id','backend','flux','reconstruction','cells','newton_cap','status','field','L1','L2','Linf','normalized_L1']];results.forEach(r=>Object.keys(r.errors).forEach(field=>{const e=errorItem(r,field);rows.push([r.caseId,r.backend,r.flux,r.reconstruction,r.cells,r.cap,r.status,field,e.l1??e.L1,e.l2??e.L2,e.linf??e.Linf,normalizedL1(r,field)]);}));download('benchmark_analytical_errors.csv',csv(rows),'text/csv;charset=utf-8');}
    $('download-menu').hidden=true;$('download-toggle').setAttribute('aria-expanded','false');
  }));
  $('download-toggle').addEventListener('click',()=>{const menu=$('download-menu');menu.hidden=!menu.hidden;$('download-toggle').setAttribute('aria-expanded',String(!menu.hidden));});
  document.addEventListener('click',event=>{if(!event.target.closest('.header-tools')){$('download-menu').hidden=true;$('download-toggle').setAttribute('aria-expanded','false');}});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'){$('download-menu').hidden=true;$('download-toggle').setAttribute('aria-expanded','false');}});
  document.querySelectorAll('[data-page]').forEach(button=>{button.addEventListener('click',()=>showPage(button.dataset.page));button.addEventListener('keydown',event=>{let index=pages.indexOf(button.dataset.page);if(event.key==='ArrowRight')index=(index+1)%pages.length;else if(event.key==='ArrowLeft')index=(index+pages.length-1)%pages.length;else if(event.key==='Home')index=0;else if(event.key==='End')index=pages.length-1;else return;event.preventDefault();showPage(pages[index],true);});});
  $('theme-toggle').addEventListener('click',()=>{theme=theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;$('theme-toggle').textContent=theme==='dark'?'Light theme':'Dark theme';try{localStorage.setItem('nh-benchmark-theme',theme);}catch(_error){}if(activePage==='summary')renderSummary();if(activePage==='case')renderCase();});
  ['summary-model','summary-reconstruction','summary-cap','summary-grid','summary-flux'].forEach(id=>$(id).addEventListener('change',renderSummary));
  $('case-select').addEventListener('change',()=>{selectedCase=$('case-select').value;updateCaseOptions(true);renderCase();});
  ['case-grid','case-cap','field-select','history-axis','case-flux','case-reconstruction'].forEach(id=>$(id).addEventListener('change',renderCase));
  unique(results.map(r=>r.flux)).forEach(flux=>{option($('summary-flux'),flux,fluxLabel(flux));option($('case-flux'),flux,fluxLabel(flux));});
  unique(results.map(r=>r.model)).forEach(model=>option($('summary-model'),model,modelLabel(model)));
  sortedNumbers(results.map(r=>r.cap)).forEach(cap=>option($('summary-cap'),cap,'Cap '+cap));
  sortedNumbers(results.map(r=>r.cells)).forEach(cells=>option($('summary-grid'),cells,cells+' cells'));
  cases.forEach(value=>option($('case-select'),value.id,`${modelLabel(value.model)} / ${value.label||value.id}`));
  if(selectedCase)$('case-select').value=selectedCase;
  updateCaseOptions(true);
  $('campaign-subtitle').textContent=verificationOnly?`${campaign.protocol?.cases?.length||cases.length} physical cases; ${campaign.protocol?.fluxes?.join(', ')||'recorded fluxes'}; one cold solve per configuration. Performance timing was not collected.`:campaign.title||campaign.description||'Numerical accuracy, convergence, and performance in one reproducible record.';
  if(verificationOnly){document.querySelector('.identity h1').textContent=campaign.protocol?.fluxes?.length===1&&campaign.protocol.fluxes[0]==='roe'?'Roe numerical verification':'Steady numerical verification';$('page-summary').querySelector('.section-heading h2').textContent='Convergence and accuracy, case by case.';}
  $('campaign-stamp').textContent=[campaign.campaign_id||campaign.id||report.input_name,campaign.created_at||campaign.provenance?.created||''].filter(Boolean).join(' · ');
  $('protocol-json').textContent=JSON.stringify(campaign.protocol||campaign.settings||{},null,2);
  $('artifact-json').textContent=JSON.stringify({campaign:campaign.presentation_artifacts||campaign.artifacts||[],baseline:baseline?(baseline.presentation_artifacts||baseline.artifacts||[]):null,archive_prefixes:{campaign:'campaign/',baseline:'baseline/'}},null,2);
  $('campaign-integrity').innerHTML=[unchanged(campaign,'source_unchanged')===false?'Production source or physical cases changed during this campaign. All timing comparisons are invalid; retained results remain available as evidence.':null,unchanged(campaign,'harness_unchanged')===false?'Benchmark harness or effective configuration changed during this campaign. All timing comparisons are invalid; rerun the campaign with a frozen harness.':null,!verificationOnly&&timingEnvironmentStatus(campaign.timing_environment)!=='clean'?'The timing environment is not verified clean. All comparative speed claims are disabled.':null].filter(Boolean).map(message=>notice(message,'danger')).join('');
  $('provenance-json').textContent=JSON.stringify({campaign:campaign.campaign_id||campaign.id,completed:campaign.completed??campaign.complete,interrupted_reason:campaign.interrupted_reason,source_unchanged:unchanged(campaign,'source_unchanged'),harness_unchanged:unchanged(campaign,'harness_unchanged'),measured_code_changes:campaign.measured_code_changes,timing_environment:campaign.timing_environment,provenance:campaign.provenance,case_definitions:cases.map(c=>({id:c.id,model:c.model,sha256:c.sha256||c.case_sha256,source:c.source||c.path})),baseline:baseline?{campaign_id:baseline.campaign_id||baseline.id,source_unchanged:unchanged(baseline,'source_unchanged'),harness_unchanged:unchanged(baseline,'harness_unchanged'),timing_environment:baseline.timing_environment,provenance:baseline.provenance}:null,report},null,2);
  $('regression-threshold').textContent=fmt(report.regression_threshold,2)+'×';
  renderRoeValidation();
  renderFlowValidation();
  window.benchmarkReport={schema_version:2,fluxes:unique(results.map(r=>r.flux)),results:results.length,cases:cases.length,openCase,showPage,selectedRows,baselineComparisons:()=>results.map(r=>({case_id:r.caseId,backend:r.backend,...baselineMatch(r)}))};
  window.benchmarkReportReady=true;
  renderSummary();
})();
