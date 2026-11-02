(async()=>{
"use strict";
const element=document.getElementById('data');
let text=element.textContent.trim();
if(element.dataset.encoding==='gzip-base64'){
 const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));
 text=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
}
if(new TextEncoder().encode(text).length!==Number(element.dataset.jsonBytes))throw Error('Embedded data length mismatch');
const data=JSON.parse(text);text='';
const $=id=>document.getElementById(id);
const cases=data.cases,byId=new Map(cases.map(c=>[c.id,c]));
const shortName={subsonic:'Subsonic',supersonic_shock:'Shock',supersonic_adapted:'Adapted'};
const rows=data.results,colors={subsonic:'#6a00a8',supersonic_shock:'#bd3786',supersonic_adapted:'#fca636'};
const meshDash={50:'longdashdot',100:'dash',200:'dot',400:'solid'};
const fields={p:{label:'Pressure',unit:'bar',scale:1e-5},u_g:{label:'Gas velocity',unit:'m/s',scale:1},
 u_l:{label:'Liquid velocity',unit:'m/s',scale:1},Ma_mix:{label:'Mixture Mach number',unit:'—',scale:1},
 alpha_g:{label:'Gas void fraction',unit:'—',scale:1},quality_flow:{label:'Gas mass-flow fraction',unit:'—',scale:1},
 T_g:{label:'Gas temperature',unit:'K',scale:1},T_l:{label:'Liquid temperature',unit:'K',scale:1},
 h_g:{label:'Gas enthalpy',unit:'kJ/kg',scale:1e-3},h_l:{label:'Liquid enthalpy',unit:'kJ/kg',scale:1e-3},
 rho_g:{label:'Gas density',unit:'kg/m³',scale:1},rho_l:{label:'Liquid density',unit:'kg/m³',scale:1}};
const groups={flow:['p','u_g','u_l','Ma_mix','alpha_g','quality_flow'],thermal:['T_g','T_l','h_g','h_l','rho_g','rho_l']};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const max=a=>{const v=a.flat(Infinity).filter(finite);return v.length?Math.max(...v):null;};
const fmt=(x,n=3)=>!finite(x)?'—':Math.abs(x)>0&&(Math.abs(x)<1e-3||Math.abs(x)>=1e5)?x.toExponential(n):x.toLocaleString('en-GB',{maximumFractionDigits:n});
const escape=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const accepted=row=>(row.history||[]).filter(h=>h.phase!=='rejected');
const steps=row=>(row.history||[]).filter(h=>h.phase==='accepted').length;
const rejects=row=>(row.history||[]).filter(h=>h.phase==='rejected').length;
const selectedCases=()=>cases.filter(c=>$('regime').value==='all'||c.id===$('regime').value);
function selectedRows(allMeshes=false){return rows.filter(r=>($('regime').value==='all'||r.regime===$('regime').value)&&(allMeshes||$('mesh').value==='all'||r.cells===Number($('mesh').value)));}
function status(row){
 if(row.reason==='not_run')return {label:'Not run',css:'partial'};
 if(row.reason==='preparing')return {label:'Running',css:'partial'};
 if(row.reason==='case_regime_mismatch')return {label:'Root converged · regime mismatch',css:'fail'};
 if(row.verified)return {label:'Verified',css:'pass'};
 if(row.success)return {label:'Converged · audit pending',css:'partial'};
 return {label:row.reason.replaceAll('_',' '),css:'fail'};
}
const badge=row=>{const s=status(row);return `<span class="badge ${s.css}">${escape(s.label)}</span>`;};
const name=row=>`${shortName[row.regime]} · N${row.cells}`;
const style=row=>({color:colors[row.regime],width:1.9,dash:$('mesh').value==='all'?meshDash[row.cells]:'solid'});
function theme(){const css=getComputedStyle(document.documentElement);return Object.fromEntries(['paper','ink','muted','grid'].map(k=>[k,css.getPropertyValue('--'+k).trim()]));}
function layout(title,xTitle,yTitle,extra={}){
 const t=theme();return {title:{text:title,x:.06,xanchor:'left',font:{size:14,color:t.ink}},
  paper_bgcolor:t.paper,plot_bgcolor:t.paper,font:{family:'Segoe UI,Arial,sans-serif',color:t.muted,size:11},
  margin:{l:74,r:20,t:52,b:88},hovermode:'closest',
  xaxis:{title:{text:xTitle,standoff:10},gridcolor:t.grid,zeroline:false,ticks:'outside'},
  yaxis:{title:{text:yTitle,standoff:10},gridcolor:t.grid,zeroline:false,ticks:'outside'},
  legend:{orientation:'h',y:-.3,x:0,font:{size:9},groupclick:'toggleitem',entrywidth:.32,entrywidthmode:'fraction'},
  uirevision:`${$('regime').value}-${$('mesh').value}-${title}`,...extra};
}
let page='solution',renderNumber=0,rendered=0,lastPlots=[];
function graph(id,traces,settings){
 lastPlots.push(id);const target=$(id);target.dataset.traceCount=traces.length;
 if(settings.yaxis.type==='log')Object.assign(settings.yaxis,{tickformat:'.0e',exponentformat:'e',showexponent:'all'});
 if(!traces.length){if(target._fullLayout)Plotly.purge(target);target.innerHTML='<div class="empty-chart">No saved data for this selection.</div>';return Promise.resolve();}
 if(!target._fullLayout)target.innerHTML='';
 settings.showlegend=traces.length<=8;
 return Plotly.react(target,traces,settings,{responsive:true,displaylogo:false,
  modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'svg',filename:`nicfd_${id}_${$('regime').value}_N${$('mesh').value}`}});
}
function profileTrace(dataset,regime,mesh,profile,key,row){
 const info=fields[key],published=dataset==='published';
 return {type:'scatter',mode:!published&&mesh<=100?'lines+markers':'lines',
  name:`${shortName[regime]} · ${published?'published':`N${mesh}`}`,
  x:profile.x.map(x=>x*1e3),y:profile[key].map(x=>finite(x)?x*info.scale:null),
  line:published?{color:colors[regime],width:1.25,dash:'dashdot'}:style(row),
  marker:{size:3},opacity:published ? .8 : row.success ? 1 : .55,
  meta:{dataset,regime,cells:mesh,field:key},
  hovertemplate:`${escape(byId.get(regime).label)} · ${published?'published data':`new N=${mesh}, ${escape(status(row).label)}`}<br>x=%{x:.4f} mm<br>${info.label}=%{y:.7g} ${info.unit}<extra></extra>`};
}
function renderProfiles(){
 const selected=selectedRows(),source=$('data-source').value;
 $('solution-context').textContent=`${selectedCases().map(c=>c.label).join(' / ')} · ${source==='both'?'Published and new':source==='published'?'Published only':'New only'} · published: 498 interior stations · new meshes: ${$('mesh').value==='all'?'50 / 100 / 200 / 400':$('mesh').value} physical cells`;
 return Promise.all(groups[$('field-group').value].map((key,i)=>{
  const traces=[];
  if(source!=='new')for(const c of selectedCases())traces.push(profileTrace('published',c.id,498,c.reference.fields,key));
  if(source!=='published')for(const row of selected)if(row.fields.x?.length)traces.push(profileTrace('new',row.regime,row.cells,row.fields,key,row));
  const info=fields[key],l=layout(info.label,'Axial position x [mm]',`${info.label} [${info.unit}]`);
  if(key==='Ma_mix')l.shapes=[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:1,y1:1,line:{color:theme().muted,width:.8,dash:'dot'}}];
  if(key==='alpha_g'||key==='quality_flow'){
   const values=traces.flatMap(t=>t.y).filter(finite);
   l.yaxis.range=key==='alpha_g'?[Math.min(.9,...values),Math.max(1,...values)]:[Math.min(0,...values),Math.max(.6,...values)];
  }
  return graph(`field-${i}`,traces,l);
 }));
}
function interpolate(x,xs,ys){
 if(x<=xs[0])return ys[0];if(x>=xs.at(-1))return ys.at(-1);
 let lo=0,hi=xs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(xs[mid]>x)hi=mid;else lo=mid;}
 return ys[lo]+(ys[hi]-ys[lo])*(x-xs[lo])/(xs[hi]-xs[lo]);
}
function difference(row,key){const ref=byId.get(row.regime).reference.fields;return row.fields.x.map((x,i)=>(row.fields[key][i]-interpolate(x,ref.x,ref[key]))*fields[key].scale);}
function renderDifferences(){
 const key=$('error-field').value,info=fields[key],selected=selectedRows(),all=selectedRows(true),traces=[];
 for(const row of selected)if(row.fields.x?.length)traces.push({type:'scatter',mode:'lines',name:name(row),
  x:row.fields.x.map(x=>x*1e3),y:difference(row,key),line:style(row),opacity:row.success?1:.55,
  meta:{dataset:'difference',regime:row.regime,cells:row.cells},
  hovertemplate:`${escape(name(row))}<br>x=%{x:.4f} mm<br>New − published=%{y:.6g} ${info.unit}<extra></extra>`});
 const dl=layout(`Difference in ${info.label.toLowerCase()}`,'Axial position x [mm]',`New − published [${info.unit}]`);
 dl.shapes=[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:0,y1:0,line:{color:theme().muted,width:.8,dash:'dot'}}];
 const curves=value=>selectedCases().map(c=>{const list=all.filter(r=>r.regime===c.id&&r.errors[key]);return {type:'scatter',mode:'lines+markers',name:c.label,
  x:list.map(r=>r.cells),y:list.map(value),line:{color:colors[c.id],width:1.8},
  marker:{size:7,symbol:list.map(r=>r.verified?'circle':'circle-open')},
  customdata:list.map(r=>status(r).label),hovertemplate:`${c.label}<br>N=%{x}<br>%{y:.6g}<br>%{customdata}<extra></extra>`};});
 const er=layout(`${info.label} RMSE`,'Physical cells',`RMSE [${info.unit}]`);er.xaxis.type='log';er.yaxis.type='log';
 const count=layout('Accepted updates','Physical cells','Accepted updates');count.xaxis.type='log';
 const residual=layout('Final spatial residual','Physical cells','Maximum fixed-scale RMS');residual.xaxis.type='log';residual.yaxis.type='log';
 for(const item of [er,count,residual])Object.assign(item.xaxis,{tickmode:'array',tickvals:[50,100,200,400],ticktext:['50','100','200','400']});
 $('error-heading').textContent=`RMSE [${info.unit}]`;
 $('grid-table').innerHTML=all.map(r=>{const e=r.errors[key]||{};return `<tr><td>${escape(byId.get(r.regime).label)}</td><td>${r.cells}</td><td>${badge(r)}</td><td>${fmt(e.rmse*info.scale,5)}</td><td>${fmt(e.max_abs*info.scale,5)}</td><td>${fmt(e.normalized_rmse,5)}</td><td>${steps(r)} / ${rejects(r)}</td></tr>`;}).join('');
 return Promise.all([graph('difference-profile',traces,dl),graph('grid-error',curves(r=>r.errors[key].rmse*info.scale),er),
  graph('grid-iterations',curves(r=>steps(r)),count),graph('grid-residual',curves(r=>max(r.metrics.equation_norm||[])),residual)]);
}
function historyTrace(row,key,label,dash='solid'){
 const history=accepted(row).filter(h=>finite(h[key]));return {type:'scatter',mode:'lines',name:`${name(row)} · ${label}`,
  x:history.map(h=>h.step),y:history.map(h=>h[key]>0?h[key]:null),line:{color:colors[row.regime],width:1.6,dash},meta:{dataset:'new',regime:row.regime,cells:row.cells}};
}
function renderHistory(){
 const residual=[],constraints=[],cfl=[],inner=[],relative=$('residual-mode').value==='relative';
 for(const row of selectedRows()){
  const hs=accepted(row).filter(h=>h.equation_norm),initial=hs[0]?.equation_norm;
  for(let k=0;k<6;k++){
   residual.push({type:'scatter',mode:'lines',name:`${name(row)} · eq${k+1}`,
    x:hs.map(h=>h.step),y:hs.map(h=>{const n=h.equation_norm[k];return relative?n>0&&initial[k]>0?n/initial[k]:null:n>0?n:null;}),
    line:{color:colors[row.regime],width:1.3,dash:['solid','dash','dot','dashdot','longdash','longdashdot'][k]},meta:{dataset:'new',regime:row.regime,cells:row.cells}});
  }
  for(const [key,label,dash]of [['boundary_max','BC','solid'],['constraint_max','EOS','dash']])constraints.push(historyTrace(row,key,label,dash));
  const gh=hs.filter(h=>h.global_defect);
  constraints.push({type:'scatter',mode:'lines',name:`${name(row)} · global`,x:gh.map(h=>h.step),y:gh.map(h=>max(h.global_defect)),line:{color:colors[row.regime],width:1.2,dash:'dot'}});
  cfl.push(historyTrace(row,'cfl','CFL'));
  const rejected=row.history.filter(h=>h.phase==='rejected');
  if(rejected.length)cfl.push({type:'scatter',mode:'markers',name:`${name(row)} · rejected`,x:rejected.map(h=>h.step),y:rejected.map(h=>h.cfl),marker:{symbol:'x',size:6,color:colors[row.regime]}});
  const updates=hs.filter(h=>finite(h.inner_iterations));
  inner.push({type:'scatter',mode:'lines+markers',name:name(row),x:updates.map(h=>h.step),y:updates.map(h=>h.inner_iterations),line:style(row),marker:{size:3}});
 }
 const a=layout('Six transported equations','Accepted update',relative?'Initial-relative RMS':'Fixed-scale RMS');a.yaxis.type='log';
 const b=layout('EOS, BC and global defects','Accepted update','Fixed-scale defect');b.yaxis.type='log';
 b.shapes=[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:1e-9,y1:1e-9,line:{color:theme().muted,width:.7,dash:'dot'}}];
 const c=layout('CFL and rejected trials','Accepted update','Actual wave CFL');c.yaxis.type='log';
 return Promise.all([graph('history-residual',residual,a),graph('history-constraints',constraints,b),graph('history-cfl',cfl,c),graph('history-inner',inner,layout('Newton corrections','Accepted update','Inner corrections'))]);
}
function updateEvidence(){
 const selected=selectedRows(),verified=selected.filter(r=>r.verified).length;
 $('kpi-success').textContent=`${verified} / ${selected.length}`;
 const errors=selected.map(r=>r.errors.p?.rmse).filter(finite).map(x=>x/1e5);
 $('kpi-error').textContent=errors.length?`${fmt(Math.min(...errors),4)}${errors.length>1?'–'+fmt(Math.max(...errors),4):''} bar`:'—';
 $('kpi-residual').textContent=fmt(max(selected.map(r=>r.metrics.equation_norm||[])),2);
 const bad=selected.filter(r=>!r.verified&&r.reason!=='not_run');
 $('selection-notice').style.display=bad.length?'block':'none';
 $('selection-notice').textContent=bad.length?`${bad.length} selected case(s) are unfinished, unaudited or on an unintended regime. Their actual profiles and errors remain visible; they are not counted as verified reproductions.`:'';
 $('run-table').innerHTML=selected.map(r=>`<tr><td><span class="dot" style="background:${colors[r.regime]}"></span>${escape(byId.get(r.regime).label)}</td><td>${r.cells}</td><td>${badge(r)}</td><td>${fmt(r.errors.p?.rmse/1e5,5)}</td><td>${fmt(r.errors.u_g?.rmse,3)} / ${fmt(r.errors.u_l?.rmse,3)}</td><td>${fmt(max(r.metrics.equation_norm||[]),2)}</td><td>${fmt(r.metrics.constraint_max,2)} / ${fmt(r.metrics.boundary_max,2)}</td><td>${steps(r)} / ${rejects(r)}</td><td>${fmt(r.solve_s,2)} s</td></tr>`).join('');
}
async function render(){
 const token=++renderNumber;lastPlots=[];updateEvidence();
 if(page==='solution')await renderProfiles();else if(page==='differences')await renderDifferences();else if(page==='convergence')await renderHistory();
 if(token===renderNumber)rendered=token;
}
async function switchPage(next){page=next;document.querySelectorAll('.page').forEach(e=>e.classList.toggle('active',e.id===page));document.querySelectorAll('.tabs button').forEach(e=>e.classList.toggle('active',e.dataset.page===page));await render();}
function download(text,filename,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const csvValue=x=>String(x??'').includes(',')?'"'+String(x).replaceAll('"','""')+'"':String(x??'');
function downloadFields(){
 const keys=['x','A','p','alpha_g','u_g','u_l','T_g','T_l','h_g','h_l','rho_g','rho_l','Ma_mix','quality_flow','mdot'];
 const lines=[['dataset','regime','physical_cells',...keys].join(',')];
 const append=(source,regime,n,f)=>{for(let i=0;i<f.x.length;i++)lines.push([source,regime,n,...keys.map(k=>f[k]?.[i])].map(csvValue).join(','));};
 if($('data-source').value!=='new')for(const c of selectedCases())append('published',c.id,498,c.reference.fields);
 if($('data-source').value!=='published')for(const r of selectedRows())if(r.fields.x?.length)append('new',r.regime,r.cells,r.fields);
 download(lines.join('\n'),'nicfd_selected_fields_SI.csv','text/csv');
}
function downloadHistory(){
 const keys=['step','phase','retry','cfl','dt','inner_iterations','inner_residual','rejected_newton_trials','constraint_max','boundary_max','elapsed_s'];
 const lines=[['regime','cells',...keys,...Array.from({length:6},(_,i)=>'equation_'+(i+1))].join(',')];
 for(const r of selectedRows())for(const h of r.history)lines.push([r.regime,r.cells,...keys.map(k=>h[k]),...Array.from({length:6},(_,i)=>h.equation_norm?.[i])].map(csvValue).join(','));
 download(lines.join('\n'),'nicfd_selected_history.csv','text/csv');
}
$('campaign-count').textContent=rows.length;
$('regime-counts').innerHTML=cases.map(c=>{const rs=rows.filter(r=>r.regime===c.id);return `<span class="hero-flux-count"><i class="dot" style="background:${colors[c.id]}"></i>${escape(c.label)}<strong>${rs.filter(r=>r.verified).length}/${rs.length}</strong></span>`;}).join('');
$('regime-key').innerHTML=cases.map(c=>`<span class="method-chip"><i class="dot" style="background:${colors[c.id]}"></i>${escape(c.label)}</span>`).join('');
$('provenance').textContent=JSON.stringify({settings:data.settings,environment:data.environment,provenance:data.provenance,effective_by_case:rows.filter(r=>r.cells===400).map(r=>({regime:r.regime,effective:r.effective,audit:r.audit}))},null,2);
$('plotly-license').textContent=data.plotly_license;
$('archive-table').innerHTML=data.archive.map(r=>`<tr><td>${escape(byId.get(r.regime).label)}</td><td>${r.cells}</td><td>${r.area_law}</td><td>${r.reason}</td><td>${r.intended_regime===null?'—':r.intended_regime?'Passed':'Mismatch'}</td><td>${fmt(r.pressure_rmse_bar,5)}</td></tr>`).join('');
for(const id of ['regime','mesh','data-source','field-group','error-field','residual-mode'])$(id).addEventListener('change',()=>render().catch(fail));
document.querySelectorAll('.tabs button').forEach(button=>button.addEventListener('click',()=>switchPage(button.dataset.page).catch(fail)));
$('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='light'?'dark':'light';render().catch(fail);});
$('download-data').addEventListener('click',()=>download(JSON.stringify(data,null,2),'nicfd_complete_report_data.json','application/json'));
$('download-fields').addEventListener('click',downloadFields);$('download-history').addEventListener('click',downloadHistory);
window.nicfdReport={data,render,switchPage,selectedRows,selectedCases,difference,downloadFields,downloadHistory,
 snapshot:()=>({page,renderNumber,rendered,plots:lastPlots,regime:$('regime').value,mesh:$('mesh').value,source:$('data-source').value,selected:selectedRows().map(r=>r.id)})};
await render();
function fail(error){window.nicfdReportLoadError=String(error);$('selection-notice').style.display='block';$('selection-notice').textContent='Report error: '+error;throw error;}
})().catch(error=>{window.nicfdReportLoadError=String(error);const box=document.getElementById('selection-notice');box.style.display='block';box.textContent='Report could not load: '+error;console.error(error);});
