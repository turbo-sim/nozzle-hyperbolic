(async function boot(){
"use strict";
const $=id=>document.getElementById(id),dataElement=$("data");
let raw=dataElement.textContent.trim();
if(dataElement.dataset.encoding==="gzip-base64"){
 if(typeof DecompressionStream!=="function")throw new Error("This report needs a modern browser, or rebuild with --data-encoding plain.");
 const binary=atob(raw),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
 raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
 if(new TextEncoder().encode(raw).length!==Number(dataElement.dataset.jsonBytes))throw new Error("Embedded evidence length check failed");
}
const campaign=JSON.parse(raw);raw="";
const rows=campaign.results||[],cases=campaign.cases||[],settings=campaign.settings||{};
const arr=x=>Array.isArray(x)?x:[],finite=x=>typeof x==="number"&&Number.isFinite(x),unique=x=>[...new Set(x)];
const max=x=>{const a=arr(x).flat(Infinity).filter(finite);return a.length?Math.max(...a):null;};
const n=(x,d=3)=>!finite(x)?"—":Math.abs(x)>0&&(Math.abs(x)<1e-3||Math.abs(x)>=1e5)?x.toExponential(d):x.toLocaleString("en-GB",{maximumFractionDigits:d});
const pct=x=>finite(x)?n(100*x)+"%":"—";
const esc=x=>String(x??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;");
const direct=r=>/direct|newton/.test(r.strategy||"");
const strategyName=x=>/direct|newton/.test(x||"")?"Direct Newton":"Implicit Euler";
const native=r=>r.native_success??r.success??false;
const verified=r=>r.verified===true||r.verified===undefined&&r.audit?.passed===true;
const target=r=>verified(r)&&r.target_correct===true;
const nameFlux=x=>({rusanov:"Rusanov",hll:"HLL",stadtke:"Stadtke"}[x]||x);
const nameRecon=x=>x==="constant"?"Constant":"MUSCL";
const nameBC=x=>x==="finite_wave"?"Finite wave":"Acoustic";
const nameStart=x=>x==="adapted"?"Adapted":"Stagnant";
const label=r=>`${strategyName(r.strategy)} · ${nameFlux(r.flux)} · ${nameRecon(r.reconstruction)} · ${nameBC(r.outlet)} · ${nameStart(r.start)}`;
const colors=["#6a00a8","#7c02a7","#8f0da4","#a21d9a","#b52f8c","#c6417d","#d5546e","#e3685f","#ed7953","#f58c46","#fba139","#fca636"];
const color=r=>colors[((['rusanov','hll','stadtke'].indexOf(r.flux)+3)%3)*4+(r.start==="adapted"?2:0)+(r.outlet==="finite_wave"?1:0)];
const line=r=>({color:color(r),width:1.8,dash:direct(r)?r.reconstruction==="constant"?"dashdot":"dash":r.reconstruction==="constant"?"dot":"solid"});
const history=r=>arr(r.history).filter(h=>h.phase!=="rejected"&&h.accepted!==false);
const initialHash=r=>r.report_initial_state_bytes_sha256||r.direct?.initial_state_sha256||r.initial_state_sha256||null;
const metrics=r=>r.audit?.residual||r.metrics||{};
const equation=h=>arr(h?.equation_norm||h?.equation_l2||h?.residual_l2);
const norm=r=>max(equation(metrics(r)))??max(equation(history(r).at(-1)))??metrics(r).residual??null;
const boundary=r=>metrics(r).boundary_max??history(r).at(-1)?.boundary_max??null;
const global=r=>max(metrics(r).global_defect)??metrics(r).global_max??max(history(r).at(-1)?.global_defect);
const accepted=r=>r.work?.accepted_steps??r.work?.accepted??r.counters?.accepted_steps??Math.max(0,history(r).length-1);
const rejected=r=>direct(r)?r.counters?.nrejected??r.work?.nrejected??arr(r.trials).filter(t=>t.accepted===false).length:r.work?.rejected_steps??r.work?.rejected??0;
const status=r=>verified(r)?"Verified root":native(r)?"Native success · unverified":"Not converged";
const badge=(text,pass)=>`<span class="badge ${pass===true?'good':pass===false?'bad':'warn'}">${esc(text)}</span>`;
const table=(id,headers,records)=>$(id).innerHTML='<thead><tr>'+headers.map(h=>`<th>${esc(h)}</th>`).join('')+'</tr></thead><tbody>'+records.map(c=>'<tr>'+c.map(v=>`<td>${v??'—'}</td>`).join('')+'</tr>').join('')+'</tbody>';
const physicalBadge=r=>{if(!r)return badge('Not assessed',null);const failed=Object.entries(r.audit?.target?.checks||{}).filter(([,v])=>!v).map(([k])=>k),positionOnly=failed.length===1&&failed[0]==='declared_shock_position_accuracy';return `<span title="${esc(failed.join(', ')||'Canonical shock branch and position checks')}">${badge(r.target_correct===true?'Shock/accuracy passes':positionOnly?'Position tolerance missed':r.target_correct===false?'Shock/accuracy fails':'Not assessed',positionOnly?null:r.target_correct)}</span>`;};
const nativeBadge=r=>badge(native(r)?'True':r.native_status===5?'False · callback stop (5)':'False',native(r)?true:r.native_status===5&&verified(r)?null:false);
const fieldInfo={p:["Pressure","kPa",1e-3],M:["Mach number","—",1],u:["Axial velocity","m/s",1],rho:["Density","kg/m³",1],T:["Temperature","K",1],h0:["Stagnation enthalpy","kJ/kg",1e-3],h:["Static enthalpy","kJ/kg",1e-3],s:["Entropy","J/(kg K)",1]};
let page="overview",view="fields",renderNumber=0,rendered=0,queue=Promise.resolve();
const strategies=unique(rows.map(r=>r.strategy));
$("strategy").innerHTML='<option value="all">Both strategies</option>'+strategies.map(s=>`<option value="${esc(s)}">${esc(strategyName(s))}</option>`).join('');
$("case").innerHTML=cases.map(c=>`<option value="${esc(c.id)}">${esc(c.label||c.id)}</option>`).join('');
if(rows.length&&!rows.some(r=>r.case_id===$("case").value&&r.flux===$("flux").value&&r.reconstruction===$("reconstruction").value&&r.outlet===$("outlet").value)){
 const first=rows.find(r=>r.flux==='stadtke'&&r.reconstruction==='muscl_van_leer')||rows[0];
 for(const [id,value]of Object.entries({case:first.case_id,flux:first.flux,reconstruction:first.reconstruction,outlet:first.outlet}))$(id).value=value;
}
function selectedCase(){return cases.find(c=>c.id===$("case").value);}
function selectedRows(){return rows.filter(r=>r.case_id===$("case").value&&['flux','reconstruction','outlet','start','strategy'].every(k=>$(k).value==="all"||r[k]===$(k).value)).sort((a,b)=>label(a).localeCompare(label(b)));}
function field(f,key){if(Array.isArray(f?.[key]))return f[key];if(key==="M")return arr(f?.M_euler);if(key==="u")return arr(f?.velocity);if(key==="h0"&&f?.h&&f?.u)return f.h.map((h,i)=>h+.5*f.u[i]**2);return [];}
function reference(c){return c?.reference?.fields||c?.reference||{};}
function theme(){const c=getComputedStyle(document.documentElement);return Object.fromEntries(['paper','ink','muted','line'].map(k=>[k,c.getPropertyValue('--'+k).trim()]));}
function layout(x,y,extra={}){const t=theme();return {paper_bgcolor:t.paper,plot_bgcolor:t.paper,font:{family:'Segoe UI,Arial,sans-serif',size:10,color:t.muted},margin:{l:72,r:22,t:32,b:70},xaxis:{title:{text:x,standoff:12},gridcolor:t.line,zeroline:false},yaxis:{title:{text:y,standoff:12},gridcolor:t.line,zeroline:false},showlegend:false,legend:{orientation:'h',y:-.24,font:{size:9}},hovermode:'closest',uirevision:`${$("case").value}-${document.documentElement.dataset.theme}`,...extra};}
function graph(id,traces,l){return Plotly.react($(id),traces,l,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'svg',filename:'newton_'+id}});}
function trace(r,x,y,extra={}){return {type:'scatter',mode:'lines',name:label(r)+(verified(r)?'':' · '+status(r)),meta:{run_id:r.id},x,y,line:line(r),hovertemplate:'%{x:.5g}<br>%{y:.6g}<extra>%{fullData.name}</extra>',...extra};}
function comparison(a,b){
 const result={implicit_id:a?.id??null,direct_id:b?.id??null,available:false,reason:'',identical_initial_state:Boolean(a&&b&&initialHash(a)&&initialHash(a)===initialHash(b)),implicit_target_correct:a?.target_correct??null,direct_target_correct:b?.target_correct??null,both_target_correct:Boolean(a&&b&&target(a)&&target(b)),pressure_scale_Pa:null,max_delta_p_over_scale:null,max_delta_M:null,relative_mean_face_flow_gap:null};
 if(!a||!b)return {...result,reason:'Missing or excluded by selection'};
 if(!result.identical_initial_state)return {...result,reason:'Initial state byte hashes differ or are unavailable'};
 if(!verified(a)||!verified(b))return {...result,reason:'Both original-equation roots must pass the independent audit'};
 const x=arr(a.fields?.x),y=arr(b.fields?.x),pa=field(a.fields,'p'),pb=field(b.fields,'p'),ma=field(a.fields,'M'),mb=field(b.fields,'M');
 if(!x.length||x.length!==y.length||x.some((v,i)=>!finite(v)||v!==y[i])||[pa,pb,ma,mb].some(v=>v.length!==x.length||v.some(x=>!finite(x))))return {...result,reason:'Complete fields on identical cell centers are unavailable'};
 const scale=a.errors?.p?.scale??selectedCase()?.scales?.p,otherScale=b.errors?.p?.scale??selectedCase()?.scales?.p;
 if(!finite(scale)||scale<=0||!finite(otherScale)||Math.abs(scale-otherScale)>1e-12*scale)return {...result,reason:'A shared fixed pressure normalization is unavailable'};
 const fa=a.audit?.conservation?.face_mass_flow_mean,fb=b.audit?.conservation?.face_mass_flow_mean,den=(Math.abs(fa)+Math.abs(fb))/2;
 return {...result,available:true,reason:'Same initial bytes; both original-equation roots independently verified; canonical shock accuracy is assessed separately',pressure_scale_Pa:scale,max_delta_p_over_scale:max(pa.map((v,i)=>Math.abs(v-pb[i])))/scale,max_delta_M:max(ma.map((v,i)=>Math.abs(v-mb[i]))),relative_mean_face_flow_gap:finite(fa)&&finite(fb)&&den>0?Math.abs(fa-fb)/den:null};
}
function pairs(selection=selectedRows()){
 const groups=new Map();for(const r of selection){const k=[r.case_id,r.flux,r.reconstruction,r.outlet,r.start,r.cells].join('|');if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r);}
 return [...groups.values()].map(g=>({held_fixed:[nameFlux(g[0].flux),nameRecon(g[0].reconstruction),nameBC(g[0].outlet),nameStart(g[0].start)].join(' · '),...comparison(g.find(r=>!direct(r)),g.find(direct))}));
}
function overview(){
 const expected=campaign.summary?.expected_runs??96,root=rows.filter(verified).length,joint=rows.filter(target).length,pp=pairs(rows),same=pp.filter(p=>p.identical_initial_state).length;
 $("hero-count").textContent=`${root} / ${expected}`;$("hero-status").textContent=`${rows.length} saved targets · ${Math.max(0,expected-rows.length)} not yet recorded · ${rows.filter(r=>!verified(r)).length} unverified`;
 $("kpi-native").textContent=`${rows.filter(native).length} / ${expected}`;$("kpi-root").textContent=`${root} / ${expected}`;$("kpi-target").textContent=`${joint} / ${expected}`;$("kpi-pairs").textContent=`${same} / ${expected/2}`;
 $("outcome-summary").innerHTML=`<strong>${rows.length} of ${expected} requested target results are saved.</strong> ${rows.filter(native).length} report native convergence; ${root} pass the independent original-equation audit; ${joint} also pass the intended shock checks. ${same} direct/implicit pairs have identical initial-state hashes. These observations concern the recorded starts, fluxes, reconstructions and outlet laws. Direct failures remain part of the evidence. Timings are single trajectories with different linear algebra and compilation costs.`;
 table('coverage-table',['Case','Strategy','Flux','Native','Independent roots','Roots + shock','Recorded / expected'],cases.flatMap(c=>strategies.flatMap(s=>['rusanov','hll','stadtke'].map(f=>{const a=rows.filter(r=>r.case_id===c.id&&r.strategy===s&&r.flux===f);return [esc(c.label||c.id),strategyName(s),nameFlux(f),a.filter(native).length,a.filter(verified).length,a.filter(target).length,`${a.length} / 8`];}))));
 const cats=strategies.map(strategyName),out=[['Native success',native,'#b52f8c'],['Independent root',verified,'#d5546e'],['Root + intended shock',target,'#fca636']].map(([name,pred,c])=>({type:'bar',name,x:cats,y:strategies.map(s=>rows.filter(r=>r.strategy===s&&pred(r)).length),marker:{color:c}}));
 const ft=strategies.map((s,i)=>({type:'bar',name:strategyName(s),x:['Rusanov','HLL','Stadtke'],y:['rusanov','hll','stadtke'].map(f=>rows.filter(r=>r.strategy===s&&r.flux===f&&target(r)).length),marker:{color:colors[i?4:1]}}));
 return Promise.all([graph('coverage-plot',out,layout('','Number of targets',{barmode:'group',showlegend:true})),graph('flux-plot',ft,layout('','Verified roots with intended shock',{barmode:'group',showlegend:true}))]);
}
function tables(){
 const selected=selectedRows(),c=selectedCase();$("case-title").textContent=c?.label||'Physical case';$("case-description").textContent=`200 cells · backpressure ${n((c?.back_pressure??0)*1e-3)} kPa · reference shock x = ${n(c?.shock_position)} m · ${selected.length} selected records`;
 const notice=$("selection-notice"),bad=selected.filter(r=>!verified(r));notice.style.display=!selected.length||bad.length?'block':'none';notice.textContent=!selected.length?'No saved records match the selection. Missing results are not converged solutions.':`${bad.length} selected result${bad.length===1?' is':'s are'} not an independently verified root. Their actual saved final iterates remain visible.`;
 table('run-table',['Configuration','Native flag / status','Independent root','Shock target','Shock x [m]','Shock error [cells]','p error L₁ / scale','Max equation RMS','Boundary max','Global max','Accepted / rejected','Compile + solve [s]','Termination'],selected.map(r=>[esc(label(r)),nativeBadge(r),badge(verified(r)?'Verified':'Unverified',verified(r)),physicalBadge(r),n(r.audit?.shock?.position_m),n(r.audit?.shock?.position_error_cells),pct(r.errors?.p?.normalized_l1),n(norm(r)),n(boundary(r)),n(global(r)),`${accepted(r)} / ${rejected(r)}`,n(r.elapsed_s??r.elapsed_seconds),`<span title="${esc(r.native_message||r.reason)}">${esc(r.reason||r.native_status||'Unavailable')}</span>`]));
 table('pair-table',['Held fixed','Identical initial bytes','Root comparison','Implicit shock/accuracy','Direct shock/accuracy','Max |Δp| / pressure scale','Max |ΔM|','Relative mean face-flow gap'],pairs().map(p=>[esc(p.held_fixed),badge(p.identical_initial_state?'Yes':'No',p.identical_initial_state),`<span title="${esc(p.reason)}">${p.available?'Both equation roots verified':esc(p.reason)}</span>`,physicalBadge(rows.find(r=>r.id===p.implicit_id)),physicalBadge(rows.find(r=>r.id===p.direct_id)),pct(p.max_delta_p_over_scale),n(p.max_delta_M),pct(p.relative_mean_face_flow_gap)]));
 const focus=$("run-focus").value;$("run-focus").innerHTML=selected.map(r=>`<option value="${esc(r.id)}">${esc(label(r))}</option>`).join('');if(selected.some(r=>r.id===focus))$("run-focus").value=focus;
 $("line-key").innerHTML=selected.map(r=>`<span class="legend-item"><i style="color:${color(r)};border-top-style:${direct(r)?'dashed':r.reconstruction==='constant'?'dotted':'solid'}"></i>${esc(label(r))}${verified(r)?'':' · '+esc(status(r))}</span>`).join('');
}
function fields(){
 const selected=selectedRows(),ref=reference(selectedCase()),keys=['p','M','u','h0',$("extra-field").value];
 return Promise.all(keys.map((key,i)=>{const info=fieldInfo[key],traces=[];if(field(ref,key).length)traces.push({type:'scatter',mode:'lines',name:'Analytical reference',x:ref.x,y:field(ref,key).map(v=>v*info[2]),line:{color:theme().ink,width:2.2,dash:'longdash'},hovertemplate:'x = %{x:.5g} m<br>%{y:.6g}<extra>Analytical reference</extra>'});
  for(const r of selected){const f=r.fields||{},values=field(f,key);if(values.length)traces.push(trace(r,f.x,values.map(v=>finite(v)?v*info[2]:null),{opacity:verified(r)?1:.7}));}
  return graph(i===4?'field-extra':'field-'+key,traces,layout('Axial coordinate x [m]',`${info[0]} [${info[1]}]`));
 }));
}
function residualValue(h,index,mode,first){
 const v=equation(h)[index];if(!finite(v))return null;if(mode==='relative'){const z=equation(first)[index];return z>0?v/z:v===0&&z===0?0:null;}if(mode==='ratio')return h.equation_max_ratio??max(h.equation_ratio);return v;
}
function axisLayout(x,y){const l=layout(x,y);if($("residual-axis").value==='log')l.yaxis={...l.yaxis,type:'log',exponentformat:'e',tickformat:'.0e'};return l;}
function residualTrace(r,h,values,name,x=null,extra={}){
 const log=$("residual-axis").value==='log';return trace(r,x||h.map((v,i)=>v.step??v.iteration??i),values.map(v=>finite(v)?log&&v===0?1e-18:v:null),{name:label(r)+' · '+name,customdata:values,hovertemplate:'%{x:.6g}<br>Actual residual = %{customdata:.6g}<extra>%{fullData.name}</extra>',...extra});
}
function convergence(){
 const selected=selectedRows(),mode=$("residual-mode").value,chosen=$("equation").value,eq=[],bc=[],gc=[],time=[];let zeros=0,undefinedRelative=0;
 for(const r of selected){const h=history(r),first=h[0];for(const i of chosen==='all'?[0,1,2]:[Number(chosen)]){if(mode==='ratio'&&i!==0&&chosen==='all')continue;const v=h.map(v=>residualValue(v,i,mode,first));zeros+=v.filter(v=>v===0).length;undefinedRelative+=v.filter(v=>v===null).length;eq.push(residualTrace(r,h,v,mode==='ratio'?'Maximum equation stopping ratio':['Mass','Momentum','Energy'][i],null,{line:{...line(r),width:i===0?2:1.3},marker:{symbol:i===0?'circle':i===1?'diamond':'square',size:4}}));}
  const bv=h.map(v=>finite(v.boundary_max)?v.boundary_max:null),gv=h.map(v=>max(v.global_defect)??v.global_max??null);bc.push(residualTrace(r,h,bv,'Boundary'));gc.push(residualTrace(r,h,gv,'Global conservation'));
  const tv=h.map(v=>max(equation(v)));if(h.some(v=>finite(v.elapsed_seconds??v.elapsed_s)))time.push(residualTrace(r,h,tv,'Maximum equation RMS',h.map(v=>v.elapsed_seconds??v.elapsed_s??null)));
 }
 $("residual-note").textContent=`Only initial and accepted states form trajectories; rejected work is retained in the work view. Equation norms are fixed-scale volume-weighted RMS values. Common root checks use per-equation absolute OR relative tolerance plus boundary/global checks. ${zeros} selected equation samples are exactly zero; log plots mark them at 10⁻¹⁸ while hover and downloads retain zero. ${undefinedRelative} equation samples have no defined plotted value in this mode. Time curves include only recorded timestamps; implicit Euler does not record per-update wall times in this campaign, and direct timestamps exclude its separately recorded compilation. Timings do not establish a speed ranking.`;
 const threshold=(v)=>({type:'line',xref:'paper',x0:0,x1:1,y0:v,y1:v,line:{color:theme().muted,width:1,dash:'dot'}});
 const eLayout=axisLayout('Accepted outer update',mode==='absolute'?'Fixed-scale spatial RMS':mode==='relative'?'Spatial RMS / initial RMS':'Maximum common stopping ratio');if(mode==='ratio')eLayout.shapes=[threshold(1)];
 const bLayout=axisLayout('Accepted outer update','Boundary residual maximum');bLayout.shapes=[threshold(1e-9)];const gLayout=axisLayout('Accepted outer update','Global conservation defect');gLayout.shapes=[threshold(1e-9)];
 return Promise.all([graph('residual-equations',eq,eLayout),graph('residual-boundary',bc,bLayout),graph('residual-global',gc,gLayout),graph('residual-time',time,axisLayout('Elapsed solve time [s]','Maximum fixed-scale equation RMS'))]);
}
function focused(){return selectedRows().find(r=>r.id===$("run-focus").value);}
function trialRecords(r){return arr(r?.trials||r?.direct?.trials||r?.solver_result?.trials);}
function workView(){
 const r=focused(),h=history(r||{}),trials=trialRecords(r),acceptedPoints=h.filter(h=>h.phase!=='initial'&&finite(h.fraction??h.step_fraction??h.alpha));
 const t=trials.filter(t=>finite(t.fraction??t.step_fraction??t.alpha??t.step_length));
 const x=t.map((v,i)=>v.iteration??v.step??i),y=t.map(v=>v.fraction??v.step_fraction??v.alpha??v.step_length);
 const fraction=acceptedPoints.length?[{type:'scatter',mode:'lines+markers',name:'Accepted Newton fraction',x:acceptedPoints.map((v,i)=>v.step??v.iteration??i),y:acceptedPoints.map(v=>v.fraction??v.step_fraction??v.alpha),line:{color:'#b52f8c'},marker:{size:6}}]:[];
 if(t.length)fraction.push({type:'scatter',mode:'markers',name:'Actual line-search trial',x,y,customdata:t.map(v=>v.reason||v.message||v.status||''),marker:{size:7,symbol:t.map(v=>v.accepted===true?'circle':v.valid===false?'triangle-up-open':'x'),color:t.map(v=>v.accepted===true?'#ed7953':v.valid===false?'#c63e52':'#6a00a8')},hovertemplate:'Iteration %{x}<br>Trial fraction %{y:.6g}<br>%{customdata}<extra>Trial evidence</extra>'});
 const cfl=h.filter(v=>finite(v.cfl)),cflTrace=cfl.length?[{type:'scatter',mode:'lines+markers',name:'Actual implicit CFL',x:cfl.map((v,i)=>v.step??i),y:cfl.map(v=>v.cfl),line:{color:'#b52f8c'},marker:{size:4}}]:[];
 $("work-note").textContent=r?`${label(r)}. ${accepted(r)} accepted updates; ${rejected(r)} rejected ${direct(r)?'line-search trials':'outer steps'}; ${direct(r)?r.counters?.ninvalid??0:r.work?.rejected_newton_trials??0} ${direct(r)?'invalid domain trials':'rejected inner Newton trials'} in saved counters. Native status: ${r.native_status??r.reason??'unavailable'}. ${direct(r)?`Initial Newton fraction ${r.options?.first_step_fraction??.25}, growth ${r.options?.step_growth_factor??1.5}, cap 1; actual accepted fractions and rejected trial details below. A singular initial Jacobian may produce zero accepted steps.`:'Implicit Euler regularizes each spatial Newton system by physical storage / pseudo-timestep; algebraic boundary traces have zero storage.'}`:'No saved run selected.';
 if(r&&direct(r))$("work-note").textContent+=' Trial markers: orange circles accepted, purple crosses rejected by decrease tests, red open triangles invalid in the model domain. The initial state is retained in the table; it is not a Newton update.';
 table('work-table',['Accepted update','Equation RMS max','Boundary max','Global max','Fraction','Backtracks','Domain rejects','CFL','Residual evals','Jacobian evals'],h.map((v,i)=>[v.step??v.iteration??i,n(max(equation(v))),n(v.boundary_max),n(max(v.global_defect)),n(v.fraction??v.step_fraction??v.alpha),v.backtracks??'—',v.domain_rejections??'—',n(v.cfl),v.nfev??'—',v.njev??'—']));
 $("trial-json").textContent=JSON.stringify(r?{id:r.id,native_success:native(r),native_status:r.native_status,native_message:r.native_message,reason:r.reason,options:r.options||r.direct_options,counters:r.counters,work:r.work,exception:r.exception,trials,direct:r.direct,initial_hash:initialHash(r)}:null,null,2);
 const stepLayout=layout('Newton iteration / accepted outer update','Step fraction α',{annotations:fraction.length?[]:[{text:'No Newton step fractions saved for this run',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,font:{color:theme().muted}}]});if($("step-axis").value==='log')stepLayout.yaxis={...stepLayout.yaxis,type:'log',tickformat:'.0e'};
 const cflLayout=layout('Accepted outer update','CFL',{annotations:cflTrace.length?[]:[{text:direct(r||{})?'Direct Newton has no pseudo-timestep':'No accepted implicit CFL trajectory saved',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,font:{color:theme().muted}}]});cflLayout.yaxis.type='log';
 return Promise.all([graph('step-plot',fraction,stepLayout),graph('cfl-plot',cflTrace,cflLayout)]);
}
function evidence(){
 const cap=settings.direct?.max_accepted_steps,total=settings.direct?.max_iterations;
 $("direct-budget").innerHTML='<strong>Direct Newton stopping budget.</strong> '+(finite(cap)?`This campaign allows at most ${cap} accepted Newton updates. A run that has not passed the independent common equation checks by that point is recorded as unsuccessful. `:'This saved campaign does not declare a separate accepted-update cap. ')+(finite(total)?`The backend also limits total line-search trials to ${total}. `:'')+'Rejected line-search trials count as work and do not consume accepted-update slots. Every unsuccessful or unfinished run retains its last flow profiles and complete accepted convergence history in this report.';
 for(const [id,value]of Object.entries({'settings-json':settings,'provenance-json':campaign.provenance,'physical-json':campaign.frozen_physical_inputs,'seeds-json':campaign.seeds,'audit-json':campaign.saved_audits,'report-json':{provenance:campaign.report_provenance,dataset_scope:campaign.report_dataset_scope,raw_records:rows.map(r=>({id:r.id,...r.raw_record_evidence}))}}))$(id).textContent=JSON.stringify(value,null,2);
}
function render(){const number=++renderNumber;queue=queue.catch(()=>{}).then(async()=>{if(number!==renderNumber)return;tables();if(page==='overview')await overview();if(page==='explorer'){if(view==='fields')await fields();if(view==='convergence')await convergence();if(view==='work')await workView();}if(number===renderNumber)rendered=number;});return queue;}
function switchPage(value){page=value;for(const e of document.querySelectorAll('.page'))e.classList.toggle('active',e.id===value);for(const e of document.querySelectorAll('[data-page]'))e.classList.toggle('active',e.dataset.page===value);return render();}
function switchView(value){view=value;for(const e of document.querySelectorAll('.case-view'))e.classList.toggle('active',e.id==='view-'+value);for(const e of document.querySelectorAll('[data-view]'))e.classList.toggle('active',e.dataset.view===value);return render();}
function download(name,text,type='application/json'){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const csvCell=x=>{const v=x==null?'':typeof x==='object'?JSON.stringify(x):String(x);return '"'+v.replaceAll('"','""')+'"';};
function csv(records){const keys=unique(records.flatMap(Object.keys));return '\ufeff'+keys.map(csvCell).join(',')+'\r\n'+records.map(r=>keys.map(k=>csvCell(r[k])).join(',')).join('\r\n');}
for(const e of document.querySelectorAll('[data-page]'))e.addEventListener('click',()=>switchPage(e.dataset.page));for(const e of document.querySelectorAll('[data-view]'))e.addEventListener('click',()=>switchView(e.dataset.view));
for(const id of ['case','flux','reconstruction','outlet','start','strategy','extra-field','residual-mode','residual-axis','equation','run-focus','step-axis'])$(id).addEventListener('change',render);
$("theme").addEventListener('change',()=>{document.documentElement.dataset.theme=$("theme").value;render();});
$("download-data").textContent=dataElement.dataset.encoding==='gzip-base64'?'Download report dataset · JSON.gz':'Download report dataset · JSON';
$("download-data").addEventListener('click',()=>{
 if(dataElement.dataset.encoding==='gzip-base64'){
  const binary=atob(dataElement.textContent.trim()),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
  download('newton-report-dataset.json.gz',bytes,'application/gzip');
 }else download('newton-report-dataset.json',dataElement.textContent.trim());
});
$("download-fields").addEventListener('click',()=>download('newton-selected-fields.csv',csv(selectedRows().flatMap(r=>arr(r.fields?.x).map((_,i)=>({id:r.id,case_id:r.case_id,strategy:r.strategy,native_success:native(r),verified:verified(r),target_correct:r.target_correct,...Object.fromEntries(Object.entries(r.fields).filter(([,v])=>Array.isArray(v)&&v.length===r.fields.x.length).map(([k,v])=>[k,v[i]]))})))),'text/csv;charset=utf-8'));
$("download-history").addEventListener('click',()=>download('newton-selected-history.csv',csv(selectedRows().flatMap(r=>arr(r.history).map(h=>({id:r.id,strategy:r.strategy,...h})))),'text/csv;charset=utf-8'));
$("download-trials").addEventListener('click',()=>download('newton-selected-trials.json',JSON.stringify(focused()||null,null,2)));
$("download-theory").addEventListener('click',()=>download('THEORY.md',campaign.documents?.['THEORY.md']||'Theory unavailable','text/markdown;charset=utf-8'));
window.newtonReport={campaign,selectedRows,selectedCase,verified,native,target,pairs,comparison,initialHash,history,trialRecords,render,switchPage,switchView,snapshot:()=>({page,view,renderNumber,rendered,selected:selectedRows().length})};
window.addEventListener('resize',()=>{for(const e of document.querySelectorAll('.chart'))if(e._fullLayout)Plotly.Plots.resize(e);});
evidence();await render();
})().catch(error=>{window.newtonReportLoadError=String(error?.stack||error);const notice=document.getElementById('selection-notice');if(notice){notice.style.display='block';notice.textContent='Report load failed: '+error.message;}console.error(error);});
