/* Saved-data presentation only. No model or solver implementation is embedded. */
(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('payload').textContent), R=D.records;
  const $=id=>document.getElementById(id), uniq=values=>[...new Set(values)];
  const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const fmt=value=>value===null||value===undefined?'\u2014':typeof value==='number'?Number(value).toPrecision(5):String(value);
  const keys=['p','velocity','T','h0'];
  const config={responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'svg',filename:'flux_method_plot',width:1200,height:700},modeBarButtonsToRemove:['lasso2d','select2d']};
  const topAudit=D.audits['fresh_audit.json']||D.audits['audit.json']||{};
  let pending=0, pageTabs;
  function caseLabel(value){const model=value.includes('annular')?'Annular Euler':'Euler nozzle',match=value.match(/shock_?(\d+)/);return model+(match?' \u00b7 shock at '+match[1]+'%':' \u00b7 '+value.split('/').at(-1));}
  const reconstructionLabel=value=>({constant:'Constant',muscl_van_leer:'MUSCL \u00b7 Van Leer',muscl_van_leer_smooth:'MUSCL \u00b7 smooth Van Leer'}[value]||value);
  function metricParts(key){
    const parts=key.split('.'),name=parts.at(-1),field=parts[0]==='errors'?parts[2]:parts.slice(1,-1).join('.');
    const labels={p:'Pressure',velocity:'Normal velocity',T:'Temperature',h0:'Total enthalpy',rho:'Density',s:'Entropy'};
    const fieldLabel=labels[field]||field,norm=(name.startsWith('normalized_')?'normalized ':'')+(name.endsWith('linf')?'L\u221e':'L1');
    const reference=parts[0]==='errors'?(parts[1]==='cell'?'conservative cell projection':'point reference'):'conserved state';
    const unit=name.startsWith('normalized_')?'1':parts[0]==='errors'?(D.fields[field]?.[1]||''):'';
    return {field:fieldLabel,norm,reference,unit};
  }
  function metricLabel(key){const value=metricParts(key);return value.field+' \u00b7 '+value.reference+' \u00b7 '+value.norm;}

  function menu(id,values,wanted){
    const old=$(id).value;
    $(id).innerHTML=values.map(value=>{const label=id==='variant'?(D.labels[value]||value):id==='case'?caseLabel(value):id==='reconstruction'?reconstructionLabel(value):id==='errorMetric'?metricLabel(value):value;return `<option value="${esc(value)}">${esc(label)}</option>`;}).join('');
    const preferred=wanted??old;
    if(values.map(String).includes(String(preferred)))$(id).value=String(preferred);
  }
  function variants(){return Array.from($('variant').selectedOptions).map(option=>option.value);}
  function accepts(row){return variants().includes(row.variant);}
  function auditStatus(row){
    const audit=row.fresh_audit;
    if(row.stale_audit)return 'STALE';
    if(topAudit.snapshot_source_match===false||topAudit.snapshot_client_match===false||topAudit.snapshot_case_match===false)return 'snapshot mismatch';
    if(topAudit.evaluation_epoch_stable===false||topAudit.audit_valid===false)return 'failed aggregate audit';
    if(!audit)return 'not yet retained';
    const identity=row.audit_required_true||D.audit_required_true;
    if(identity.some(key=>audit[key]===false))return 'identity mismatch';
    if(audit.error||audit.all_checks_passed===false)return 'failed checks';
    return audit.all_checks_passed===true&&identity.every(key=>audit[key]===true)?'validated':'incomplete audit';
  }
  function pointSymbol(row){const status=auditStatus(row);return !row.success||/failed|mismatch|STALE/.test(status)?'x':status==='validated'?'circle':'diamond-open';}
  function statusKind(row){return !row.success||/failed|mismatch|STALE/.test(auditStatus(row))?'fail':auditStatus(row)==='validated'?'pass':'warn';}
  function caseRows(){return R.filter(row=>row.case===$('case').value);}
  function selected(){return caseRows().filter(row=>row.cells===Number($('cells').value)&&row.reconstruction===$('reconstruction').value&&String(row.alignment)===$('alignment').value&&accepts(row));}
  function label(row){return (D.labels[row.variant]||row.variant)+(row.success?'':' [unsuccessful]');}
  function line(row){return {color:D.colors[row.variant],width:2.2,dash:['solid','dash','dot','dashdot'][D.variants.indexOf(row.variant)%4]};}
  function layout(title,y){return window.FluxReportUI.layout(title,y);}
  function plot(id,traces,settings){
    if(!traces.length)settings.annotations=[{text:'No saved data match this selection.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,font:{size:14}}];
    pending++;window.reportPlotsReady=false;
    Promise.resolve(Plotly.react(id,traces,settings,config)).catch(error=>{console.error(error);}).finally(()=>{pending--;if(pending===0)window.reportPlotsReady=true;});
  }
  function configurationControls(){
    $('configuration-controls').innerHTML=Array.from($('variant').options).map(option=>`<label class="configuration-choice"><input type="checkbox" data-configuration="${esc(option.value)}" value="${esc(option.value)}" checked><span>${esc(D.labels[option.value]||option.value)}</span></label>`).join('');
  }
  function syncChoices(){const chosen=variants();document.querySelectorAll('#configuration-controls input').forEach(input=>{input.checked=chosen.includes(input.value);});}
  function resetCase(){
    const rows=caseRows();
    menu('cells',uniq(rows.map(row=>row.cells)).sort((a,b)=>a-b));
    menu('reconstruction',uniq(rows.map(row=>row.reconstruction)).sort(),'constant');
    menu('alignment',uniq(rows.map(row=>String(row.alignment))).sort(),'0');
    menu('variant',D.variants.filter(value=>rows.some(row=>row.variant===value)));
    for(const option of $('variant').options)option.selected=true;
    configurationControls();
    const metrics=uniq(rows.flatMap(row=>Object.keys(row.errors))).sort();
    menu('errorMetric',metrics,metrics.includes('errors.cell.p.normalized_l1')?'errors.cell.p.normalized_l1':metrics.find(value=>value.includes('cell')&&value.includes('normalized_l1'))||metrics[0]);
    draw();
  }
  function profiles(rows){
    const base=rows.find(row=>row.profile.x.length)||rows[0];
    keys.forEach((key,index)=>{
      const traces=[];
      rows.forEach(row=>{if(row.profile.fields[key])traces.push({x:row.profile.x,y:row.profile.fields[key],name:label(row),mode:'lines+markers',marker:{size:4},line:line(row)});});
      if(base){
        const reference=base.reference_point;
        if(reference.fields[key]){
          const shock=base.shock_position,masks=shock==null?[reference.x.map(()=>true)]:[reference.x.map(x=>x<=shock),reference.x.map(x=>x>shock)];
          masks.forEach((mask,part)=>{if(mask.some(Boolean))traces.push({x:reference.x.filter((_,n)=>mask[n]),y:reference.fields[key].filter((_,n)=>mask[n]),name:'Analytical point branches',legendgroup:'reference',showlegend:part===0,mode:'lines',line:{color:'#8795a5',dash:'dot'}});});
        }
        if(base.reference_cell.fields[key])traces.push({x:base.reference_cell.x,y:base.reference_cell.fields[key],name:'Conservative cell projection',mode:'markers',marker:{color:'#7e8a99',symbol:'x',size:5}});
      }
      const settings=layout(D.fields[key][0],D.fields[key][1]);
      if(base?.shock_position!=null&&base.profile.x.length>1){
        const shock=base.shock_position;
        settings.shapes=[{type:'line',x0:shock,x1:shock,y0:0,y1:1,yref:'paper',line:{color:'#8795a5',dash:'dot'}}];
        if($('range').value==='shock'){
          const length=Math.max(...base.profile.x)-Math.min(...base.profile.x),width=Math.max(6*length/base.cells,.065*length);
          settings.xaxis.range=[shock-width,shock+width];
        }
      }
      plot('field'+index,traces,settings);
    });
  }
  function histories(rows){
    for(const [id,key,title] of [['residual','overall_relative','Original spatial residual'],['cfl','cfl','Actual accepted CFL']]){
      const traces=rows.map(row=>({x:row.history.map(item=>item.step),y:row.history.map(item=>item[key]),name:label(row),mode:'lines',line:line(row)}));
      const settings=layout(title,id==='cfl'?'CFL':'Maximum relative equation norm');
      settings.xaxis.title={text:'Accepted outer step',standoff:12};settings.yaxis.type='log';
      plot(id,traces,settings);
    }
    const traces=[['accepted_steps','Accepted steps'],['rejected_attempts','Rejected attempts'],['newton_corrections','Newton corrections including rejected work']].map(([key,name])=>({x:rows.map(label),y:rows.map(row=>row.work[key]??null),name,type:'bar'}));
    const settings=layout('Complete saved nonlinear work','Count');settings.xaxis.title={text:'Configuration',standoff:12};settings.barmode='group';
    plot('work',traces,settings);
  }
  function refinement(){
    const key=$('errorMetric').value,rows=caseRows().filter(row=>row.reconstruction===$('reconstruction').value&&String(row.alignment)===$('alignment').value&&accepts(row)),traces=[];
    for(const variant of D.variants){
      const values=rows.filter(row=>row.variant===variant).sort((a,b)=>a.cells-b.cells);
      if(values.length)traces.push({x:values.map(row=>row.cells),y:values.map(row=>row.errors[key]??null),name:D.labels[variant]||variant,meta:{configuration_id:variant},mode:'lines+markers',marker:{symbol:values.map(pointSymbol),size:8},line:line(values[0])});
    }
    const metric=key?metricParts(key):null;
    const settings=layout('Grid refinement \u00b7 '+caseLabel($('case').value),metric?metric.field+' \u00b7 '+metric.norm+(metric.unit?' ['+metric.unit+']':''):'No saved error metric');
    settings.xaxis.title={text:'Physical cells',standoff:12};settings.xaxis.type='log';settings.yaxis.type='log';
    plot('refinementPlot',traces,settings);
  }
  function table(rows){
    $('rows').innerHTML=rows.map(row=>{
      const metrics=row.profile_metrics||{},parameters=Object.entries(row.parameters||{}).map(([key,value])=>key+'='+String(value)).join('; '),audit=auditStatus(row);
      const first=[D.labels[row.variant]||row.variant,parameters,row.reason,row.work.accepted_steps,row.work.rejected_attempts,row.work.newton_corrections];
      return `<tr class="${row.success?'':'failed'}">${first.map(value=>`<td>${esc(fmt(value))}</td>`).join('')}<td><span class="badge ${statusKind(row)}">${esc(audit)}</span></td>${[metrics.pressure_local_peak_pct,metrics.velocity_local_dip_pct,metrics.h0_relative_max].map(value=>`<td class="num">${esc(fmt(value))}</td>`).join('')}</tr>`;
    }).join('');
    $('details').textContent=JSON.stringify(rows,null,2);
  }
  function overview(){
    const native=R.filter(row=>row.success).length,validated=R.filter(row=>row.success&&auditStatus(row)==='validated').length;
    const fullyValidated=D.complete&&topAudit.complete===true&&R.length>0&&validated===R.length;
    const bad=R.some(row=>statusKind(row)==='fail');
    $('overview-status').className='status-badge '+(bad?'fail':fullyValidated?'pass':'warn');
    $('overview-status').textContent=bad?'Retained checks need attention':fullyValidated?'Recorded population validated':'Coverage or checks incomplete';
    const cards=[['Physical cases',D.cases.length,'Independent packaged nozzle and annular reference problems.'],['Saved states',R.length,D.complete?'Requested saved population is complete.':'Requested population is incomplete.'],['Recorded successes',native,'Native convergence is separate from analytical accuracy.'],['Validated states',validated,'Passing hash-joined numerical and identity checks.']];
    $('overview-cards').innerHTML=cards.map(([name,value,text])=>`<article class="panel stat"><p class="stat-label">${esc(name)}</p><p class="stat-value">${esc(value)}</p><p class="stat-description">${esc(text)}</p></article>`).join('');
    $('study-scope').textContent=D.report_profile.evidence_scope||'Saved numerical evidence with independently recorded checks and source identity.';
    const byConfig=D.variants.map(variant=>{const rows=R.filter(row=>row.variant===variant);return {variant,rows,native:rows.filter(row=>row.success).length,valid:rows.filter(row=>row.success&&auditStatus(row)==='validated').length,failed:rows.filter(row=>statusKind(row)==='fail').length,incomplete:rows.filter(row=>statusKind(row)==='warn').length};});
    $('overview-rows').innerHTML=byConfig.map(item=>{const parameters=D.report_profile.configurations.find(value=>value.id===item.variant)?.parameters||{};return `<tr>${[D.labels[item.variant]||item.variant,Object.entries(parameters).map(([key,value])=>key+'='+value).join('; '),item.rows.length,item.native,item.valid,item.failed,item.incomplete].map(value=>`<td>${esc(value)}</td>`).join('')}</tr>`;}).join('');
    const traces=[['rows','Recorded states'],['native','Native convergence'],['valid','Validated saved states']].map(([key,name])=>({x:byConfig.map(item=>D.labels[item.variant]||item.variant),y:byConfig.map(item=>key==='rows'?item.rows.length:item[key]),name,type:'bar'}));
    const settings=layout('Saved population coverage','Count');settings.xaxis.title={text:'Configuration',standoff:12};settings.barmode='group';
    plot('coverage-plot',traces,settings);
  }
  function draw(){
    const rows=selected(),groups=uniq(rows.map(row=>row.comparison_identity));
    $('selection').textContent=`${caseLabel($('case').value)} \u00b7 ${rows.length} saved configuration(s) \u00b7 ${rows.filter(row=>row.fresh_audit).length} hash-joined fresh audit(s). ${groups.length>1?'Multiple source/control identities are retained; inspect provenance before interpreting a matched comparison.':'Shared source, physical and control identity for the selected comparison.'}`;
    $('case-status').textContent=caseLabel($('case').value);
    $('empty-selection').hidden=rows.length!==0;
    syncChoices();profiles(rows);histories(rows);table(rows);refinement();
    $('announcement').textContent=rows.length+' saved configurations selected for '+caseLabel($('case').value);
  }
  function save(name,text,type){const anchor=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));anchor.href=url;anchor.download=name;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),0);}
  function downloadCsv(){
    const headers=['case','cells','reconstruction','configuration','outcome','accepted_steps','rejected_attempts','newton_corrections','fresh_audit','pressure_local_peak_pct','velocity_local_dip_pct','h0_relative_error'];
    const quote=value=>'"'+String(value??'').replace(/"/g,'""')+'"';
    const rows=selected().map(row=>[row.case,row.cells,row.reconstruction,row.variant,row.reason,row.work.accepted_steps,row.work.rejected_attempts,row.work.newton_corrections,auditStatus(row),row.profile_metrics?.pressure_local_peak_pct,row.profile_metrics?.velocity_local_dip_pct,row.profile_metrics?.h0_relative_max]);
    save(D.study+'_selected_states.csv',[headers,...rows].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8');
  }
  const findingsValid=D.findings?.verified&&topAudit.complete===true&&['evaluation_epoch_stable','audit_valid','snapshot_source_match','snapshot_client_match','snapshot_case_match'].every(key=>topAudit[key]!==false);
  $('findings').hidden=!findingsValid;
  $('findings').innerHTML=findingsValid?'<h3>Principal findings from the complete matrix</h3>'+D.findings.statements.map(text=>'<p>'+esc(text)+'</p>').join('')+((D.findings.accuracy_details||[]).length?'<details><summary>Accuracy magnitudes and absolute scale</summary>'+D.findings.accuracy_details.map(text=>'<p>'+esc(text)+'</p>').join('')+'</details>':''):'';
  $('title').textContent=D.title;$('lead').textContent=D.report_profile.lead;
  $('theoryHeading').textContent=D.report_profile.theory_heading;$('theoryLead').textContent=D.report_profile.theory_lead;
  if(D.report_profile.equation_latex)$('equation').textContent='$$'+D.report_profile.equation_latex+'$$';else $('equation').innerHTML=D.report_profile.equation_html;$('parameterHeader').textContent=D.report_profile.parameter_heading;
  $('theory-source').textContent=D.documents['THEORY.md']||'';
  $('epochRecovery').textContent=D.epoch_recovery?.statement||'';$('focusedTests').textContent=D.current_focused_status||'';
  $('auditSummary').textContent=JSON.stringify({summary:D.summary,audits:D.audits},null,2);
  $('manifest').textContent=JSON.stringify(D.evidence_manifest,null,2);$('supplemental').textContent=JSON.stringify(D.supplemental_evidence||{},null,2);
  $('progress').textContent=D.complete?(D.report_profile.completion_note||'Saved population coverage is complete; evaluation checks and analytical accuracy remain separate.'):'Population incomplete or in progress. Missing records are not counted as passes.';
  $('campaign-stamp').textContent=R.length+' saved states \u00b7 '+D.cases.length+' physical cases';
  window.FluxReportUI.init({onThemeChange:()=>{overview();draw();}});
  menu('case',D.cases,R.some(row=>row.case==='euler_annular/shock70')?'euler_annular/shock70':R[0]?.case||D.cases[0]);
  $('range').value='domain';$('case').onchange=resetCase;
  for(const id of ['cells','reconstruction','variant','alignment','range'])$(id).onchange=draw;
  $('errorMetric').onchange=refinement;
  $('configuration-controls').addEventListener('change',event=>{const input=event.target;if(!input.dataset.configuration)return;const option=Array.from($('variant').options).find(item=>item.value===input.value);if(option)option.selected=input.checked;draw();});
  $('config-all').addEventListener('click',()=>{for(const option of $('variant').options)option.selected=true;draw();});
  $('config-none').addEventListener('click',()=>{for(const option of $('variant').options)option.selected=false;draw();});
  $('download-toggle').addEventListener('click',()=>{const opened=$('download-menu').hidden;$('download-menu').hidden=!opened;$('download-toggle').setAttribute('aria-expanded',String(opened));});
  $('download-json').addEventListener('click',()=>save(D.study+'_plot_data.json',JSON.stringify(D,null,2),'application/json'));
  $('download-csv').addEventListener('click',downloadCsv);
  $('download-evidence').addEventListener('click',()=>$('evidence-download').click());
  document.addEventListener('keydown',event=>{if(event.key==='Escape'){$('download-menu').hidden=true;$('download-toggle').setAttribute('aria-expanded','false');}});
  document.addEventListener('click',event=>{if(!event.target.closest?.('.header-tools')){$('download-menu').hidden=true;$('download-toggle').setAttribute('aria-expanded','false');}});
  pageTabs=window.FluxReportUI.bindTabs({nav:$('report-tabs'),panels:document.querySelectorAll('.report-page'),onChange:()=>{}});
  window.FluxReportUI.bindTabs({nav:$('case-tabs'),panels:document.querySelectorAll('.case-page'),onChange:()=>{}});
  $('open-case').addEventListener('click',()=>pageTabs.activate('page-case'));
  window.REPORT={selected,caseRows,draw,resetCase,auditStatus,activatePage:id=>pageTabs.activate(id)};
  overview();resetCase();
})();
