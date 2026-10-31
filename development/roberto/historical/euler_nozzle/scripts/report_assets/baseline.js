/* Presentation only: every plotted solver sample is embedded in the document. */
const DATA=JSON.parse(document.getElementById('baseline-data').textContent);
const steady=DATA.kind==='steady', methods=['constant','muscl_van_leer'];
const labels={constant:'Constant',muscl_van_leer:'MUSCL'};
const methodColor={constant:'#b12a90',muscl_van_leer:'#fca636'};
const eqColor=['#b12a90','#d8576b','#fca636'];
const names=['Mass','Momentum','Energy'];
const $=id=>document.getElementById(id), finite=v=>v!==null&&Number.isFinite(v);
const fmt=(x,n=3)=>finite(x)?Number(x).toPrecision(n):'—';
const positive=values=>values.map(v=>finite(v)&&v>0?v:null);
let generation=0;
function theme(){return document.documentElement.dataset.theme||'light';}
function palette(){let s=getComputedStyle(document.documentElement);return Object.fromEntries(['ink','muted','panel','grid','line'].map(k=>[k,s.getPropertyValue('--'+k).trim()]));}
function trace(x,y,name,color,dash='solid'){return {x,y,name,type:'scatter',mode:'lines',line:{color,width:2,dash},hovertemplate:'%{x:.6g}<br>%{y:.6g}<extra>%{fullData.name}</extra>'};}
function plot(id,traces,xlabel,ylabel,{log=false,logx=false,target=null,height=365}={}){
  const c=palette();
  let layout={height,paper_bgcolor:c.panel,plot_bgcolor:c.panel,font:{family:'Segoe UI, sans-serif',size:11,color:c.ink},
    margin:{l:68,r:20,t:70,b:60},xaxis:{title:xlabel,gridcolor:c.grid,zeroline:false,type:logx?'log':'linear'},
    yaxis:{title:ylabel,gridcolor:c.grid,zeroline:false,type:log?'log':'linear',...(log?{dtick:1,exponentformat:'power',showexponent:'all'}:{})},
    legend:{orientation:'h',y:1.24,x:0,entrywidth:.48,entrywidthmode:'fraction',font:{size:10}},
    hovermode:'closest',uirevision:id+theme()+$('case-select').value+$('mesh-select').value};
  if(target!==null) layout.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:target,y1:target,line:{color:c.muted,dash:'dot'}}];
  return Plotly.react(id,traces,layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'euler_nozzle_'+id}});
}
function selected(){
  const grid=Number($('mesh-select').value);
  if(steady)return DATA.runs.filter(r=>r.case.id===$('case-select').value&&r.points===grid).sort((a,b)=>methods.indexOf(a.reconstruction)-methods.indexOf(b.reconstruction));
  return DATA.runs.find(r=>r.physical_cells===grid);
}
function field(r,key){
  if(key==='mdot')return r.numerical.rho.map((rho,i)=>rho*r.numerical.u[i]*r.area[i]);
  if(key==='e')return r.numerical.p.map((p,i)=>p/((r.gamma-1)*r.numerical.rho[i]));
  return r.numerical[key];
}
function section(id,title,content){return `<details class="block" id="${id}" open><summary>${title}</summary><div class="block-content">${content}</div></details>`;}
function panel(id,title,note='',cls=''){return `<div class="panel"><div class="panel-head"><h3>${title}</h3></div><div class="plot ${cls}" id="${id}"></div>${note?`<div class="panel-note">${note}</div>`:''}</div>`;}
function table(headers,rows){return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function status(pass){return `<span class="${pass?'pass':'fail'}">${pass?'PASS':'FAIL'}</span>`;}
function tableRows(){
  const all=$('all-grids').checked;
  const finest=Math.max(...DATA.runs.map(r=>steady?r.points:r.physical_cells));
  if(steady){
    const rows=DATA.runs.filter(r=>all||r.points===finest).map(r=>[r.case.label,r.physical_cells,labels[r.reconstruction],fmt(r.case.back_pressure_ratio,6),r.history.length-1,status(r.converged),
      fmt(r.normalized_L1.p),fmt(r.normalized_L1.rho),fmt(r.normalized_L1.M),fmt(r.relative_boundary_mass_imbalance),fmt(r.throat_M_numerical,5),status(r.accuracy_passed)]);
    $('table-content').innerHTML=table(['Case','Physical points','Method','pᵦ/p₀','Updates','Root','L1 p/p₀','L1 ρ/ρ₀','L1 Mach','Mass imbalance','Throat Mach','Field accuracy'],rows);
  }else{
    const rows=DATA.runs.filter(r=>all||r.physical_cells===finest).flatMap(r=>r.snapshots.map(s=>[r.physical_cells,s.tau,fmt(s.time_s*1e3,5),s.steps,fmt(s.normalized_L1.rho),fmt(s.normalized_L1.u),fmt(s.normalized_L1.p),fmt(Math.max(...s.normalized_conservation_error.map(Math.abs))),status(s.passed)]));
    $('table-content').innerHTML=table(['Physical cells','τ','Time [ms]','Updates','L1 ρ/ρL','L1 u/aL','L1 p/pL','Max budget defect','Accuracy + budget'],rows);
  }
}
async function render(){
  const token=++generation,c=palette(); let tasks=[];
  if(steady){
    const runs=selected(),base=runs[0],ref=base.dense_reference;
    $('selection-note').innerHTML=`<strong>${base.case.label}</strong> · pᵦ/p₀ = ${fmt(base.case.back_pressure_ratio,7)} · analytical branch: ${base.analytical_regime.replaceAll('_',' ')}.<br>`+
      runs.map(r=>`${labels[r.reconstruction]}: ${r.history.length-1} updates; native SSR ${fmt(r.history.at(-1).source_ssr)}; throat Mach ${fmt(r.throat_M_numerical,6)} (reference ${fmt(r.throat_M_reference,6)}).`).join('<br>');
    const radius=base.area.map(a=>Math.sqrt(a/Math.PI)*1000);
    tasks.push(plot('geometry-profile',[trace(base.x,radius,'Upper wall',c.ink),trace(base.x,radius.map(r=>-r),'Lower wall',c.ink)],'x [m]','Equivalent radius [mm]',{height:275}));
    tasks.push(plot('geometry-area',[trace(base.x,base.area.map(a=>a*1e6),'Area','#8f0da4')],'x [m]','A [mm²]',{height:275}));
    const metadata=[['p','Static pressure [kPa]',1e3],['rho','Density [kg/m³]',1],['u','Axial velocity [m/s]',1],['M','Mach number',1],['T','Temperature [K]',1],['mdot','Cell-center mass flow [kg/s]',1]];
    for(let i=0;i<metadata.length;i++){
      const [key,title,scale]=metadata[i];
      const exact=key==='mdot'?ref.rho.map((rho,j)=>rho*ref.u[j]*DATA.geometry_area_dense[j]):ref[key];
      const traces=[trace(ref.x,exact.map(v=>v/scale),'Analytical',c.muted,'dot'),...runs.map(r=>trace(r.x,field(r,key).map(v=>v/scale),labels[r.reconstruction],methodColor[r.reconstruction],r.reconstruction==='constant'?'dash':'solid'))];
      tasks.push(plot('field-'+i,traces,'x [m]',title));
    }
    tasks.push(plot('overall',runs.map(r=>trace(r.history.map(h=>h.update),positive(r.history.map(h=>h.spatial_rms)),labels[r.reconstruction],methodColor[r.reconstruction],r.reconstruction==='constant'?'dash':'solid')),'Accepted updates','Fixed study spatial RMS',{log:true,target:1e-8}));
    tasks.push(plot('cfl',runs.map(r=>trace(r.trials.map(h=>h.update),positive(r.trials.map(h=>h.cfl)),labels[r.reconstruction],methodColor[r.reconstruction],r.reconstruction==='constant'?'dash':'solid')),'Attempted update (retries share x)','Actual trial CFL',{log:true}));
    for(const [id,key,ylabel] of [['equations','equation_relative','Rⱼ / Rⱼ,initial'],['balances','global_defect','Absolute scaled global defect']]){
      const traces=runs.flatMap(r=>names.map((name,j)=>trace(r.history.map(h=>h.update),positive(r.history.map(h=>h[key][j])),name+' / '+labels[r.reconstruction],eqColor[j],r.reconstruction==='constant'?'dash':'solid')));
      tasks.push(plot(id,traces,'Accepted updates',ylabel,{log:true}));
    }
    const all=DATA.runs.filter(r=>r.case.id===base.case.id);
    const keys=['p','rho','M'];
    const error=methods.flatMap(method=>keys.map((key,j)=>{let rows=all.filter(r=>r.reconstruction===method).sort((a,b)=>a.physical_cells-b.physical_cells);return trace(rows.map(r=>r.physical_cells),rows.map(r=>r.normalized_L1[key]),key+' / '+labels[method],eqColor[j],method==='constant'?'dash':'solid');}));
    const conservative=methods.flatMap(method=>names.map((name,j)=>{let rows=all.filter(r=>r.reconstruction===method).sort((a,b)=>a.physical_cells-b.physical_cells);return trace(rows.map(r=>r.physical_cells),rows.map(r=>r.conservative_L1[j]),name+' / '+labels[method],eqColor[j],method==='constant'?'dash':'solid');}));
    tasks.push(plot('errors',error,'Physical points','Normalized mean absolute error',{log:true,logx:true,target:.03}));
    tasks.push(plot('state-errors',conservative,'Physical points','Normalized conservative-state L1',{log:true,logx:true}));
  }else{
    const run=selected(),snapshot=run.snapshots.find(s=>String(s.tau)===$('case-select').value),ref=snapshot.dense_reference;
    $('selection-note').innerHTML=`<strong>Standard Sod problem · τ = ${snapshot.tau}</strong> · physical time ${fmt(snapshot.time_s*1e3,6)} ms · ${run.physical_cells} cells · ${snapshot.steps} updates.<br>HLL / constant reconstruction / forward Euler. All output times are reached by clipping the timestep before the update.`;
    const setup=DATA.physical_setup;
    tasks.push(plot('geometry-profile',[trace([0,setup.diaphragm_m,setup.diaphragm_m,setup.length_m],[setup.left.rho,setup.left.rho,setup.right.rho,setup.right.rho],'Initial density','#8f0da4')],'x [m]','Initial density [kg/m³]',{height:275}));
    tasks.push(plot('geometry-area',[trace([0,setup.diaphragm_m,setup.diaphragm_m,setup.length_m],[setup.left.p/1e3,setup.left.p/1e3,setup.right.p/1e3,setup.right.p/1e3],'Initial pressure','#d8576b')],'x [m]','Initial pressure [kPa]',{height:275}));
    const metadata=[['rho','Density [kg/m³]',1],['p','Static pressure [kPa]',1e3],['u','Velocity [m/s]',1],['M','Mach number',1],['T','Temperature [K]',1],['e','Specific internal energy [kJ/kg]',1e3]];
    for(let i=0;i<metadata.length;i++){
      const [key,title,scale]=metadata[i];
      const exact=key==='e'?ref.p.map((p,j)=>p/((run.gamma-1)*ref.rho[j])):ref[key];
      const numerical=key==='e'?snapshot.numerical.p.map((p,j)=>p/((run.gamma-1)*snapshot.numerical.rho[j])):snapshot.numerical[key];
      tasks.push(plot('field-'+i,[trace(ref.x,exact.map(v=>v/scale),'Exact Riemann solution',c.muted,'dot'),trace(snapshot.x,numerical.map(v=>v/scale),'HLL / constant','#d8576b')],'x [m]',title));
    }
    const history=run.history;
    tasks.push(plot('overall',[trace(history.map(h=>h.iteration),history.map(h=>h.tau),'Physical clock','#8f0da4')],'Physical updates','τ = t aL / L'));
    tasks.push(plot('cfl',[trace(history.map(h=>h.iteration),history.map(h=>h.cfl),'Actual physical CFL','#fca636')],'Physical updates','CFL',{target:run.CFL}));
    tasks.push(plot('equations',names.map((name,j)=>trace(history.map(h=>h.tau),positive(history.map(h=>Math.abs(h.conservation_error[j]))),name,eqColor[j])),'τ = t aL / L','Absolute boundary-flux budget defect',{log:true,target:1e-10}));
    tasks.push(plot('balances',['rho','u','p'].map((key,j)=>trace(run.snapshots.map(s=>s.tau),run.snapshots.map(s=>s.normalized_L1[key]),key,eqColor[j])),'τ = t aL / L','Normalized mean absolute error',{log:true,target:.04}));
    const rows=DATA.runs.slice().sort((a,b)=>a.physical_cells-b.physical_cells);
    tasks.push(plot('errors',['rho','u','p'].map((key,j)=>trace(rows.map(r=>r.physical_cells),rows.map(r=>r.snapshots.find(s=>s.tau===snapshot.tau).normalized_L1[key]),key,eqColor[j])),'Physical cells','Normalized mean absolute error',{log:true,logx:true,target:.04}));
    tasks.push(plot('state-errors',names.map((name,j)=>trace(rows.map(r=>r.physical_cells),rows.map(r=>r.snapshots.find(s=>s.tau===snapshot.tau).conservative_L1[j]),name,eqColor[j])),'Physical cells','Normalized conservative-state L1',{log:true,logx:true}));
  }
  await Promise.all(tasks);
  if(token===generation){document.body.dataset.ready='true';tableRows();}
}
function initialize(){
  const kind=steady?'Steady nozzle':'Sod shock tube';
  $('title').textContent=kind+' · pre-refactor baseline';
  if(steady){const cases=[...new Map(DATA.runs.map(r=>[r.case.id,r.case])).values()];$('case-select').innerHTML=cases.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');}
  else $('case-select').innerHTML=DATA.runs[0].snapshots.map(s=>`<option value="${s.tau}">τ = ${s.tau}</option>`).join('');
  const grids=[...new Set(DATA.runs.map(r=>steady?r.points:r.physical_cells))].sort((a,b)=>a-b);
  $('mesh-select').innerHTML=grids.map(n=>`<option value="${n}">${steady?n-2:n} physical ${steady?'points ('+n+' nodes)':'cells'}</option>`).join('');
  $('mesh-select').value=grids.at(-1);
  if(!steady)$('case-select').value=DATA.runs[0].snapshots.at(-1).tau;
  $('plots-fields').innerHTML=Array.from({length:6},(_,i)=>panel('field-'+i,(steady?['Static pressure','Density','Axial velocity','Mach number','Static temperature','Cell-center mass flow']:['Density','Static pressure','Velocity','Mach number','Static temperature','Specific internal energy'])[i])).join('');
  $('plots-history').innerHTML=steady?
    panel('overall','Overall spatial residual','The fixed study RMS is used with native SSR for stopping. This is the current baseline, not annular’s future nine-order criterion.')+
    panel('cfl','CFL per attempted update','Actual dt × pre-update max(|u|+a) / dx. Repeated x coordinates are rejected retries.')+
    panel('equations','Relative residuals R1–R3','Volume-weighted RMS of each local signed equation, divided by its own initial norm. Observational diagnostics only in this baseline.')+
    panel('balances','Three global conservation defects','Absolute sum of signed integrated spatial rows, including geometry sources, divided by fixed flux scales. Global cancellation can hide large local errors.'):
    panel('overall','Physical time reached','This is a transient calculation. A decaying steady residual is not a convergence requirement.')+
    panel('cfl','CFL per physical update','The smaller CFL values are output-time clipping, not rejected pseudo-time trials.')+
    panel('equations','Three time-integrated conservation budgets','Change in domain contents minus integrated numerical boundary influx. Exact zeros are omitted on the logarithmic axis.')+
    panel('balances','Accuracy at the three saved times','These are exact-solution errors, not solver residuals. The dotted line is the 0.04 accuracy threshold.');
  for(const id of ['case-select','mesh-select'])$(id).addEventListener('change',()=>{document.body.dataset.ready='false';render();});
  $('all-grids').addEventListener('change',tableRows);
  document.querySelectorAll('[data-page-choice]').forEach(button=>button.addEventListener('click',()=>showPage(button.dataset.pageChoice)));
  document.querySelectorAll('[data-theme-choice]').forEach(button=>button.addEventListener('click',()=>{document.documentElement.dataset.theme=button.dataset.themeChoice;document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',b===button));render();}));
  document.querySelectorAll('details.block').forEach(detail=>detail.addEventListener('toggle',()=>{if(detail.open)detail.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));}));
  $('download-data').addEventListener('click',()=>download(DATA,'euler_nozzle_'+DATA.kind+'_baseline.json','application/json'));
  $('download-selected').addEventListener('click',()=>download(selected(),'euler_nozzle_'+DATA.kind+'_selected.json','application/json'));
  $('print').addEventListener('click',()=>window.print());
  document.querySelectorAll('.sidebar a').forEach(a=>a.addEventListener('click',()=>{showPage('results');const section=document.querySelector(a.getAttribute('href'));if(section)section.open=true;}));
  showPage(location.hash==='#theory'?'theory':location.hash==='#reproduce'?'reproduce':'results');
  render();
}
function showPage(page){document.querySelectorAll('[data-report-page]').forEach(el=>el.hidden=el.dataset.reportPage!==page);document.querySelectorAll('[data-page-choice]').forEach(el=>el.setAttribute('aria-current',el.dataset.pageChoice===page?'page':'false'));document.body.dataset.page=page;if(page==='results')setTimeout(()=>document.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el)),0);}
function download(value,name,type){const blob=new Blob([JSON.stringify(value,null,2)],{type});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
initialize();
