(async function bootReport(){
"use strict";
const embeddedData=document.getElementById("data");
let campaignText=embeddedData.textContent.trim();
if(embeddedData.dataset.encoding==="gzip-base64") {
 if(typeof DecompressionStream!=="function")throw new Error("This report uses lossless gzip data. Open it in a modern browser with native DecompressionStream support, or regenerate with --data-encoding plain.");
 const binary=atob(campaignText),bytes=new Uint8Array(binary.length);
 for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
 campaignText=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
 if(new TextEncoder().encode(campaignText).length!==Number(embeddedData.dataset.jsonBytes))throw new Error("Embedded report data length check failed");
}
const campaign = JSON.parse(campaignText);
campaignText="";
const $ = id => document.getElementById(id);
const FLUXES = campaign.settings.fluxes;
const FLUX_NAME = {rusanov:"Rusanov",hll:"HLL",roe:"Roe",stadtke:"Stadtke · corrected"};
for(const method of FLUXES) {
 const options=campaign.settings.flux_options?.[method];
 if(options?.flux==="osher") FLUX_NAME[method]=`Osher · ${options.integration==="sonic_partitioned"?"sonic-partitioned":"Gauss"} ${options.quadrature_points??5}`;
}
const PALETTE = ["#6a00a8","#9b179e","#bd3786","#d8576b","#ed7953","#fca636"];
const COLORS = Object.fromEntries(FLUXES.map((flux,index)=>[flux,PALETTE[index%PALETTE.length]]));
const FIELD_INFO = {
 p:{label:"Pressure",unit:"MPa",scale:1e-6},rho:{label:"Density",unit:"kg/m³",scale:1},
 u:{label:"Velocity",unit:"m/s",scale:1},T:{label:"Temperature",unit:"K",scale:1},
 h:{label:"Enthalpy",unit:"kJ/kg",scale:1e-3},M_euler:{label:"Euler Mach number",unit:"—",scale:1},
 Q:{label:"Raw vapor mass quality",unit:"—",scale:1},s:{label:"Table entropy",unit:"kJ/(kg K)",scale:1e-3},
 mass_flow:{label:"Numerical face mass flow",unit:"kg/s",scale:1},
 shock_position:{label:"Shock position",unit:"mm",scale:1e3}
};
const unique = a => [...new Set(a)];
const finite = n => typeof n === "number" && Number.isFinite(n);
const maxFinite = a => {const b=a.flat(Infinity).filter(finite);return b.length?Math.max(...b):null;};
const number = (value,digits=3) => !finite(value)?"—":Math.abs(value)>0&&(Math.abs(value)<1e-3||Math.abs(value)>=1e5)?value.toExponential(digits):value.toLocaleString("en-GB",{maximumFractionDigits:digits});
const percent = value => finite(value)?`${number(100*value,3)}%`:"—";
const escape = s => String(s??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;");
const phys = c => c.fluid_case || c.physical_case || c.id.replace(/_(adapted|shock)$/," ").trim();
const regime = c => c.regime || (c.id.endsWith("_shock")?"shock":"adapted");
const rows = campaign.results || [];
const cases = campaign.cases || [];
const caseById = new Map(cases.map(c=>[c.id,c]));
let page = "solution";
let renderNumber = 0;
let rendered = 0;
let lastPlots = [];

function reconstruction(row) {return row.reconstruction === "constant"?"Constant":"Linear MUSCL";}
function seriesName(row) {return `${FLUX_NAME[row.flux]||row.flux} · ${reconstruction(row)}`;}
function selectedCase() {return cases.find(c=>phys(c)===$("physical-case").value&&regime(c)===$("regime").value);}
function selectedFluxes() {return [...$("methods").querySelectorAll("input:checked")].map(input=>input.value);}
function selectedRows(allMeshes=false) {
 const c=selectedCase(); if(!c)return [];
 return rows.filter(row=>row.case_id===c.id&&selectedFluxes().includes(row.flux)&&
  ($("reconstruction").value==="both"||row.reconstruction===$("reconstruction").value)&&
  (allMeshes||Number(row.cells)===Number($("mesh").value)))
  .sort((a,b)=>a.cells-b.cells||FLUXES.indexOf(a.flux)-FLUXES.indexOf(b.flux)||String(a.reconstruction).localeCompare(String(b.reconstruction)));
}
function fields(row) {const f=row.fields||{};return {...f,x:f.x||row.x||[]};}
function reference(c) {const f=c?.reference?.fields||c?.reference||{};return {...f,x:f.x||c?.reference?.x||[]};}
function valueArray(f,key) {if(Array.isArray(f[key]))return f[key];if(key==="M_euler"&&Array.isArray(f.M))return f.M;return [];}
function history(row,includeRejected=false) {return (row.history||[]).filter(h=>includeRejected||(h.accepted!==false&&h.phase!=="rejected"));}
function stepCount(row) {const h=history(row);return Math.max(0,...h.map((x,i)=>Number(x.step??i)));}
function rejectedCount(row) {return history(row,true).filter(h=>h.accepted===false||h.phase==="rejected").length;}
function historyName(row) {return seriesName(row)+(row._attemptLabel?` · ${row._attemptLabel}`:row.attempts?.length?" · final attempt":"");}
function selectedHistoryRows() {
 const selected=selectedRows();
 if($("attempt-mode").value!=="all")return selected;
 return selected.flatMap(row=>[...(row.attempts||[]).map((attempt,index)=>({...row,...attempt,_retained:true,_attemptLabel:`retained attempt ${index+1}`})),{...row,_attemptLabel:row.attempts?.length?"final attempt":""}]);
}
function error(row,field,norm="l1") {
 if(field==="mass_flow") {
  const v=row.audit?.conservation?.numerical_face_mass_flow_relative_reference_error;
  return norm==="l1"&&finite(v)?v:null;
 }
 if(field==="shock_position") {
  const v=row.audit?.shock?.position_error_over_divergent_length;
  return norm==="l1"&&finite(v)?Math.abs(v):null;
 }
 const v=row.errors?.[field];return finite(v?.[`normalized_${norm}`])?v[`normalized_${norm}`]:null;
}
function finalNorm(row) {return maxFinite(row.metrics?.equation_norm||row.audit?.residual?.equation_norm||history(row).at(-1)?.equation_norm||[]);}
function status(row) {
 if(row.reason==="not_completed"||row.reason==="not_run")return {label:"Not run",css:"partial"};
 if(row.reason==="study_collapsed_cfl_plateau")return {label:"Stopped · CFL plateau",css:"fail"};
 if(row.reason==="study_wall_time_budget")return {label:"Stopped · time budget",css:"fail"};
 if(row.success&&row.audit?.passed===true)return {label:"Verified",css:"pass"};
 if(row.success)return {label:row.audit?.passed===false?"Audit failed":"Native success",css:row.audit?.passed===false?"fail":"partial"};
 return {label:row.reason==="blocked"||row.blocked?"Blocked":"Not converged",css:"fail"};
}
function lineStyle(row) {return {color:COLORS[row.flux]||"#777",width:row._retained?1:1.9,dash:row._retained?"dot":row.reconstruction==="constant"?"dash":"solid"};}
function theme() {
 const css=getComputedStyle(document.documentElement);
 return {paper:css.getPropertyValue("--paper").trim(),ink:css.getPropertyValue("--ink").trim(),muted:css.getPropertyValue("--muted").trim(),grid:css.getPropertyValue("--grid").trim(),reference:css.getPropertyValue("--reference").trim()};
}
function baseLayout(title,xTitle,yTitle,extra={}) {
 const t=theme();
 return {title:{text:title,x:.06,xanchor:"left",font:{size:14,color:t.ink}},paper_bgcolor:t.paper,plot_bgcolor:t.paper,
  font:{family:'"Segoe UI",Arial,sans-serif',size:11,color:t.muted},margin:{l:66,r:22,t:57,b:56},
  xaxis:{title:{text:xTitle,standoff:12},gridcolor:t.grid,zeroline:false,ticks:"outside",tickcolor:t.grid},
  yaxis:{title:{text:yTitle,standoff:10},gridcolor:t.grid,zeroline:false,ticks:"outside",tickcolor:t.grid},
  showlegend:false,hovermode:"closest",hoverlabel:{font:{size:11}},
  uirevision:`${selectedCase()?.id}-${$("mesh").value}-${document.documentElement.dataset.theme}`,...extra};
}
function graph(id,traces,layout) {
 const target=$(id);
 if(layout.yaxis?.type==="log")Object.assign(layout.yaxis,{tickformat:".0e",exponentformat:"e",showexponent:"all"});
 if(!traces.length) {
  if(target._fullLayout)Plotly.purge(target);
  target.innerHTML='<div class="empty-chart">No saved data for this selection.<br>Choose another method or inspect the run status below.</div>';
  return Promise.resolve();
 }
 if(!target._fullLayout)target.innerHTML="";
 lastPlots.push(id);
 return Plotly.react(target,traces,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["select2d","lasso2d"],toImageButtonOptions:{format:"svg",filename:`co2_${selectedCase()?.id}_${id}`}});
}
function shockPosition(c) {
 if(finite(c?.shock_position))return c.shock_position;
 if(finite(c?.shock_x))return c.shock_x;
 if(finite(c?.shock?.x))return c.shock.x;
 const x=reference(c).x;
 for(let i=1;i<x.length;i++)if(x[i]===x[i-1])return x[i];
 return null;
}
function shapes(c,additional=[]) {
 const p=shockPosition(c);let s=[{type:"line",xref:"x",yref:"paper",x0:0,x1:0,y0:0,y1:1,line:{color:theme().muted,width:.7,dash:"dot"}}];
 if(regime(c)==="shock"&&finite(p))s.push({type:"line",xref:"x",yref:"paper",x0:p*1e3,x1:p*1e3,y0:0,y1:1,line:{color:theme().muted,width:.8,dash:"dash"}});
 return s.concat(additional);
}
function renderFields() {
 const c=selectedCase(), selected=selectedRows(), exact=reference(c);
 const jobs=[];
 for(const key of ["p","rho","u","T","M_euler","Q"]) {
  const f=FIELD_INFO[key],traces=[],exactValues=valueArray(exact,key);
  if(exact.x?.length&&exactValues.length)traces.push({type:"scatter",mode:"lines",name:"Continuous reference",x:exact.x.map(x=>x*1e3),y:exactValues.map(x=>finite(x)?x*f.scale:null),line:{color:theme().reference,width:2,dash:"dot"},hovertemplate:`Reference<br>x=%{x:.3f} mm<br>${f.label}=%{y:.6g} ${f.unit}<extra></extra>`});
  for(const row of selected) {
   const a=fields(row),v=valueArray(a,key);if(!a.x.length||!v.length)continue;
   traces.push({type:"scatter",mode:row.cells<=100?"lines+markers":"lines",name:seriesName(row),x:a.x.map(x=>x*1e3),y:v.map(x=>finite(x)?x*f.scale:null),line:lineStyle(row),marker:{size:3,symbol:row.reconstruction==="constant"?"circle-open":"circle"},opacity:row.success?1:.65,hovertemplate:`${escape(seriesName(row))}<br>${escape(status(row).label)}<br>x=%{x:.3f} mm<br>${f.label}=%{y:.6g} ${f.unit}<extra></extra>`});
  }
  jobs.push(graph(`field-${key}`,traces,baseLayout(f.label,"Axial position x [mm]",`${f.label} [${f.unit}]`,{shapes:shapes(c)})));
 }
 return Promise.all(jobs);
}
function acceptedHistoryTrace(row,key,mode) {
 const hs=history(row),x=[],y=[];
 for(let i=0;i<hs.length;i++) {
  const h=hs[i];let val;
  if(key==="residual")val=mode==="relative"?h.overall_relative:mode==="ratio"?h.equation_max_ratio:maxFinite(h.equation_norm||[]);
  else if(key==="boundary")val=h.boundary_max;
  else if(key==="global")val=maxFinite(h.global_defect||[]);
  else val=h[key];
  if(finite(val)){x.push(h.step??i);y.push(val>0?val:null);}
 }
 if(!x.length)return null;
 return {type:"scatter",mode:"lines",name:historyName(row),x,y,line:lineStyle(row),hovertemplate:`${escape(historyName(row))}<br>Step %{x}<br>%{y:.5e}<extra></extra>`};
}
function renderHistory() {
 const selected=selectedHistoryRows(),mode=$("residual-mode").value;
 const all=(key)=>selected.map(row=>acceptedHistoryTrace(row,key,mode)).filter(Boolean);
 const resLabel=mode==="relative"?"Largest relative equation norm":mode==="ratio"?"Largest equation stopping ratio":"Maximum equation RMS [fixed scale]";
 const residualLayout=baseLayout("Original spatial residual","Accepted stage step",resLabel);
 residualLayout.yaxis.type="log";
 const limits=campaign.settings?.steady?.residual||campaign.settings?.residual||{};
 const threshold=mode==="relative"?(limits.relative_tolerance??1e-6):mode==="ratio"?1:(limits.absolute_tolerance??1e-9);
 residualLayout.shapes=[{type:"line",xref:"paper",yref:"y",x0:0,x1:1,y0:threshold,y1:threshold,line:{color:theme().muted,dash:"dot",width:1}}];
 residualLayout.annotations=[{xref:"paper",yref:"y",x:1,y:Math.log10(threshold),text:mode==="absolute"?"absolute target (relative alternative also applies)":"target",showarrow:false,xanchor:"right",yanchor:"bottom",font:{size:9,color:theme().muted}}];
 const boundary=all("boundary"),global=all("global");
 global.forEach(trace=>{trace.line={...trace.line,width:1,dash:"dot"};trace.name+=" · global";});
 const bLayout=baseLayout("Boundary and global conservation","Accepted stage step","Normalized defect");bLayout.yaxis.type="log";
 const cfl=[],inner=[];
 for(const row of selected) {
  const hs=history(row,true),accepted=hs.filter(h=>h.accepted!==false&&h.phase!=="rejected");
  if(accepted.length)cfl.push({type:"scatter",mode:"lines",name:historyName(row),x:accepted.map((h,i)=>h.step??i),y:accepted.map(h=>finite(h.cfl)&&h.cfl>0?h.cfl:null),line:lineStyle(row),hovertemplate:`${escape(historyName(row))}<br>Step %{x}<br>CFL %{y:.5g}<extra></extra>`});
  const rejected=hs.filter(h=>h.accepted===false||h.phase==="rejected");
  if(rejected.length)cfl.push({type:"scatter",mode:"markers",name:`${seriesName(row)} · rejected`,x:rejected.map(h=>h.step),y:rejected.map(h=>h.cfl),marker:{symbol:"x",size:6,color:COLORS[row.flux]},hovertemplate:`${escape(seriesName(row))}<br>Rejected trial at step %{x}<br>CFL %{y:.5g}<extra></extra>`});
  if(accepted.some(h=>finite(h.inner_iterations)))inner.push({type:"scatter",mode:"lines",name:historyName(row),x:accepted.map((h,i)=>h.step??i),y:accepted.map(h=>finite(h.inner_iterations)?h.inner_iterations:null),line:lineStyle(row),hovertemplate:`${escape(historyName(row))}<br>Step %{x}<br>Inner iterations %{y}<extra></extra>`});
 }
 const cLayout=baseLayout("Accepted CFL and rejected trials","Accepted stage step","Effective CFL");cLayout.yaxis.type="log";
 return Promise.all([graph("history-residual",all("residual"),residualLayout),graph("history-boundary",boundary.concat(global),bLayout),graph("history-cfl",cfl,cLayout),graph("history-inner",inner,baseLayout("Newton iterations per accepted step","Accepted stage step","Inner iterations"))]);
}
function gridGroups() {
 const groups=new Map();
 for(const row of selectedRows(true)){const key=`${row.flux}|${row.reconstruction}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
 return [...groups.values()].map(a=>a.sort((x,y)=>x.cells-y.cells));
}
function gridTrace(group,kind) {
 const r=group[0],field=$("error-field").value;
 const y=group.map(row=>kind==="error"?error(row,field):kind==="steps"?stepCount(row):kind==="time"?row.elapsed_s:finalNorm(row));
 if(!y.some(finite))return null;
 return {type:"scatter",mode:"lines+markers",name:seriesName(r),x:group.map(x=>x.cells),y:y.map(v=>finite(v)&&v>0?v:null),line:lineStyle(r),marker:{color:COLORS[r.flux],size:7,symbol:group.map(row=>status(row).css==="pass"?(row.reconstruction==="constant"?"circle-open":"circle"):"x")},customdata:group.map(row=>[status(row).label,row.reason||""]),hovertemplate:`${escape(seriesName(r))}<br>%{x} cells<br>%{y:.6g}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>`};
}
function renderGrid() {
 const groups=gridGroups(),key=$("error-field").value;
 const scalar=key==="mass_flow"||key==="shock_position";
 const errorAxis=key==="mass_flow"?"Relative numerical mass-flow error":key==="shock_position"?"Absolute position error / divergent length":"Normalized L₁ error";
 $("grid-error-heading").textContent=scalar?"Relative error":"Normalized L₁";
 $("grid-l2-heading").textContent=scalar?"—":"Normalized L₂";
 $("grid-linf-heading").textContent=scalar?"—":"Normalized L∞";
 $("error-definition").textContent=key==="mass_flow"?"Numerical mass-flow error is |mean numerical face ṁ / reference ṁ − 1|. It measures accuracy of the conserved numerical flow rate, not the spatial spread of that flux and not the physical cell-state mass-flow error. This scalar diagnostic has no L₂ or L∞ columns.":key==="shock_position"?"Shock position is estimated from the largest positive adjacent-cell pressure jump in the divergent section. The plotted value is |estimated x_s − reference x_s| / divergent length; its resolution is limited by the mesh spacing. It is available only for internal-shock cases and has no L₂ or L∞ columns.":"Primitive-field errors are volume-weighted numerical cell states versus the continuous reference at physical cell centers, with the declared fixed normalization scale. These are not errors from an exact cell-average primitive inversion.";
 const plots=[["grid-error","error",`${FIELD_INFO[key].label}: analytic comparison`,errorAxis,true],["grid-iterations","steps","Iterations after coarse-to-fine initialization","Accepted stage steps",false],["grid-time","time","Single-run stage wall time","Stage wall time [s]",false],["grid-residual","residual","Final original spatial residual","Maximum equation RMS [fixed scale]",true]];
 const jobs=plots.map(([id,kind,title,label,log])=>{const layout=baseLayout(title,"Physical mesh cells",label);layout.xaxis.type="log";layout.xaxis.tickvals=[50,100,200,400,800];layout.xaxis.ticktext=["50","100","200","400","800"];if(log)layout.yaxis.type="log";return graph(id,groups.map(g=>gridTrace(g,kind)).filter(Boolean),layout);});
 $("grid-table").innerHTML=selectedRows(true).map(row=>`<tr><td>${row.cells}</td><td><span class="dot" style="background:${COLORS[row.flux]}"></span>${escape(seriesName(row))}</td><td><span class="badge ${status(row).css}">${status(row).label}</span></td><td>${number(error(row,key))}</td><td>${number(error(row,key,"l2"))}</td><td>${number(error(row,key,"linf"))}</td><td>${stepCount(row)}</td><td>${number(row.elapsed_s,2)} s</td></tr>`).join("");
 return Promise.all(jobs);
}
function initials(row) {
 const text=typeof row.initialization==="string"?row.initialization:row.initialization?.method||row.initialization?.kind||row.initialization?.source||"—";
 const source=row.source_cells??row.initialization?.source_cells;
 return source?`${text} · ${source} → ${row.cells}`:text;
}
function renderSummary() {
 const c=selectedCase(),selected=selectedRows(),passed=selected.filter(r=>status(r).css==="pass").length,native=selected.filter(r=>r.success).length;
 $("kpi-success").textContent=`${passed} / ${selected.length}`;
 $("kpi-native").textContent=`${native} native success · ${selected.length-passed} not verified`;
 $("kpi-residual").textContent=number(maxFinite(selected.map(finalNorm)));
 $("kpi-error").textContent=percent(maxFinite(selected.map(r=>error(r,"mass_flow"))));
 const source=unique(selected.map(r=>r.source_cells??r.initialization?.source_cells).filter(finite));
 const notRun=selected.every(r=>r.reason==="not_completed"||r.reason==="not_run");
 const restarts=selected.filter(row=>row.attempts?.length).length;
 $("kpi-multigrid").textContent=!notRun&&Number($("mesh").value)===50?"Cold start":restarts?"Grid sequence + retry":"Coarse to fine";
 $("kpi-sequence").textContent=(source.length?`${source.join(" / ")} to ${$("mesh").value} cells - transferred state`:`${$("mesh").value} cells - cold initialization`)+(restarts?`; ${restarts} cold restart${restarts===1?"":"s"}`:"");
 const bad=selected.filter(r=>status(r).css!=="pass");
 $("selection-notice").style.display=bad.length||!selected.length?"block":"none";
 $("selection-notice").textContent=!selected.length?"No recorded run matches this selection. Choose another method or mesh.":notRun?"The selected mesh stages have not completed.":`${bad.length} selected run${bad.length===1?" is":"s are"} not independently verified. Inspect the status and audits.`;
 const p0=c?.p0??c?.settings?.p0,T0=c?.T0??c?.settings?.T0,pb=c?.backpressure??c?.p_back;
 const sc=shockPosition(c),geo=c?.geometry||{};
 $("solution-context").textContent=`${c?.label||c?.id||"No physical case"} · ${$("mesh").value} cells · reservoir ${number(p0/1e6,4)} MPa / ${number(T0,2)} K · ${regime(c)==="adapted"?"reference adapted pressure":"imposed backpressure"} ${number(pb/1e6,5)} MPa${regime(c)==="shock"&&finite(sc)?` · reference shock x = ${number(sc*1e3,3)} mm`:""}`;
 $("run-table").innerHTML=selected.map(row=>`<tr><td><span class="dot" style="background:${COLORS[row.flux]}"></span>${escape(seriesName(row))}</td><td><span class="badge ${status(row).css}">${status(row).label}</span><span class="status-sub" title="${escape(row.reason)}">${escape(row.reason||"")}</span></td><td>${percent(error(row,"p"))}</td><td>${percent(error(row,"u"))}</td><td>${percent(error(row,"mass_flow"))}</td><td>${number(finalNorm(row))}</td><td>${stepCount(row)} / ${rejectedCount(row)}</td><td>${number(row.elapsed_s,2)} s</td><td title="${escape(initials(row))}">${escape(initials(row))}</td></tr>`).join("");
 $("case-settings").innerHTML=cases.map(c=>{const g=c.geometry||{},t=c.table||c.settings?.table||{};const rs=c.regime||regime(c);return `<article class="settings-card"><h4>${escape(c.label||c.id)}</h4><dl><dt>Operating mode</dt><dd>${rs==="adapted"?"Adapted expansion":"Internal shock"}</dd><dt>Reservoir pressure</dt><dd>${number((c.p0??c.settings?.p0)/1e6,4)} MPa</dd><dt>Reservoir temperature</dt><dd>${number(c.T0??c.settings?.T0,2)} K</dd><dt>Backpressure</dt><dd>${number((c.backpressure??c.p_back)/1e6,5)} MPa</dd><dt>Exit / throat area</dt><dd>${number(g.exit_ratio,3)}</dd><dt>Table resolution</dt><dd>100 × 100</dd>${finite(t.p_min)?`<dt>Pressure table bounds</dt><dd>${number(t.p_min/1e6)}–${number(t.p_max/1e6)} MPa</dd>`:""}${finite(t.h_min)?`<dt>Enthalpy table bounds</dt><dd>${number(t.h_min/1e3)}–${number(t.h_max/1e3)} kJ/kg</dd>`:""}</dl></article>`;}).join("");
 $("provenance").textContent=JSON.stringify({summary:campaign.summary,settings:campaign.settings,chains:campaign.chains,metadata:campaign.metadata,report_provenance:campaign.report_provenance,note:"Full states and every history row are embedded and complete JSON download; this panel shows the settings and provenance."},null,2);
 $("setup-note").innerHTML=`<strong>Matched property tables.</strong><p>Each FVM case and its continuous reference use the same JAXProp bicubic table with 100 x 100 nodes. The table bounds and fingerprints are recorded in the case settings and provenance. Every flow-state property call uses the bicubic backend.</p>`;
 const mismatchFor=caseItem=>caseItem?.baseline_audit?.profiles?.[regime(caseItem)==="adapted"?"adapted":"internal_shock"]?.table_sound_speed_relative_mismatch_max;
 const acousticDescription=cases.map(caseItem=>`${phys(caseItem)==="supercritical"?"Supercritical":"Two-phase"} ${regime(caseItem)}: ${percent(mismatchFor(caseItem))}`).join("; ");
 $("acoustic-note").textContent=`Reference maxima of |a_table/a_Euler - 1|: ${acousticDescription}. The conservative reference derives its acoustic speed from density; production wave speeds and boundary characteristics use the independently interpolated table sound speed. Numerical convergence and property-table consistency are separate assessments.`;
 const cold=campaign.settings?.cold_start;
 if(cold?.adapted==="table_mach_ramp")$("cold-start-note").textContent=`The adapted 50-cell stage uses a rough cold ramp: pressure and velocity vary linearly from the reservoir to the outlet guess, with inlet u/a₀ = ${number(cold.inlet_velocity_over_a0,3)}, target outlet table Mach ${number(cold.outlet_mach_table,3)}, and h = h₀ − u²/2. No analytic area–Mach solution, choked mass flow or shock location is used. Shock cases use the production default cold state. Finer grids receive the preceding successful mesh state through the production grid-sequence driver.`;
 const budgetRows=rows.filter(r=>r.execution_controls),budget=budgetRows[0]?.execution_controls;
 $("execution-note").innerHTML=budget?`<strong>Execution limits.</strong><p>The production step ceiling is 16,000. The execution profile permits ${number(campaign.settings.execution_limits?.osher_wall_seconds??budget.wall_seconds,0)} s per Osher attempt and ${number(campaign.settings.execution_limits?.default_wall_seconds??budget.wall_seconds,0)} s for other methods, checked at callbacks. A ${number(budget.plateau_window,0)}-step collapsed-CFL progress check also applies. A budget stop is unsuccessful. Convergence always requires the original spatial residual, boundary and global conservation tests to pass. Per-stage execution controls and measured solve times are included in the downloadable data.</p>`:"";
 $("selected-audits").textContent=JSON.stringify(selected.map(row=>({id:row.id,case_id:row.case_id,flux:row.flux,reconstruction:row.reconstruction,cells:row.cells,scheme:row.scheme,success:row.success,reason:row.reason,initialization:row.initialization,source_cells:row.source_cells,execution_controls:row.execution_controls,history_rows:row.history?.length||0,attempts:(row.attempts||[]).map(a=>({reason:a.reason,success:a.success,elapsed_s:a.elapsed_s,history_rows:a.history?.length||0,metrics:a.metrics})),metrics:row.metrics,audit:row.audit,errors:row.errors})),null,2);
}
async function render() {
 const own=++renderNumber;lastPlots=[];renderSummary();
 if(page==="solution")await renderFields();
 else if(page==="convergence")await renderHistory();
 else if(page==="refinement")await renderGrid();
 if(own===renderNumber)rendered=own;
}
function switchPage(name) {
 page=name;
 document.querySelectorAll(".page").forEach(el=>el.classList.toggle("active",el.id===name));
 document.querySelectorAll(".tabs button").forEach(el=>el.classList.toggle("active",el.dataset.page===name));
 render();
}
function download(name,type,text) {
 const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function csvCell(v) {const s=v===null||v===undefined?"":typeof v==="object"?JSON.stringify(v):String(v);return /[",\n\r]/.test(s)?`"${s.replaceAll('"','""')}"`:s;}
function csv(header,records) {return [header,...records].map(row=>row.map(csvCell).join(",")).join("\r\n")+"\r\n";}
function downloadFields() {
 const keys=["x","p","rho","u","T","h","M_euler","M_table","a_euler","a_table","Q","s"],out=[];
 const c=selectedCase(),exact=reference(c);
 for(let i=0;i<exact.x.length;i++)out.push([c.id,"reference","reference",$("mesh").value,"reference",...keys.map(key=>valueArray(exact,key)[i])]);
 for(const row of selectedRows()) {const f=fields(row);for(let i=0;i<f.x.length;i++)out.push([row.case_id,row.flux,row.reconstruction,row.cells,status(row).label,...keys.map(key=>valueArray(f,key)[i])]);}
 download(`co2_${c.id}_${$("mesh").value}_fields.csv`,"text/csv;charset=utf-8",csv(["case_id","flux","reconstruction","cells","status",...keys],out));
}
function downloadHistory() {
 const out=[];
 for(const row of selectedHistoryRows())for(const h of history(row,true))out.push([row.case_id,row.flux,row.reconstruction,row.cells,row._attemptLabel||"final attempt",h.step,h.phase,h.accepted,h.retry,h.cfl,h.dt,...(h.equation_norm||[null,null,null]),h.overall_relative,h.equation_max_ratio,h.boundary_max,maxFinite(h.global_defect||[]),h.inner_iterations]);
 download(`co2_${selectedCase().id}_${$("mesh").value}_history.csv`,"text/csv;charset=utf-8",csv(["case_id","flux","reconstruction","cells","attempt","step","phase","accepted","retry","cfl","dt_s","mass_equation_norm","momentum_equation_norm","energy_equation_norm","overall_relative","equation_max_ratio","boundary_max","global_defect_max","inner_iterations"],out));
}
function initialize() {
 const physical=unique(cases.map(phys));
 $("physical-case").innerHTML=physical.map(p=>`<option value="${escape(p)}">${p==="supercritical"?"Supercritical CO₂":p==="lettieri_equilibrium"?"Two-phase CO₂ · Lettieri-type":escape(p)}</option>`).join("");
 const meshes=unique(rows.map(r=>Number(r.cells))).sort((a,b)=>a-b);
 $("mesh").innerHTML=(meshes.length?meshes:[50,100,200,400,800]).map(n=>`<option value="${n}">${n} cells</option>`).join("");
 $("mesh").value=String(meshes.includes(800)?800:meshes.at(-1)||50);
 $("methods").innerHTML=FLUXES.map(f=>`<label class="method-chip"><input type="checkbox" value="${f}" checked><span class="dot" style="background:${COLORS[f]}"></span>${FLUX_NAME[f]}</label>`).join("");
 $("campaign-count").textContent=rows.length.toLocaleString("en-GB");
 const verified=rows.filter(r=>status(r).css==="pass").length;
 $("campaign-flux-counts").innerHTML=FLUXES.map(flux=>{const stages=rows.filter(r=>r.flux===flux),passed=stages.filter(r=>status(r).css==="pass").length;return `<span class="hero-flux-count" data-flux="${flux}" data-verified="${passed}" data-total="${stages.length}" title="Independently verified stages in this campaign"><span class="dot" style="background:${COLORS[flux]}"></span>${FLUX_NAME[flux]}<strong>${passed} / ${stages.length}</strong></span>`;}).join("");
 $("campaign-state").textContent=`${verified} independently verified mesh stages`;
 const savedTheme=(()=>{try{return localStorage.getItem("co2-fvm-theme");}catch{return null;}})();
 document.documentElement.dataset.theme=savedTheme||"light";
 for(const id of ["physical-case","regime","mesh","reconstruction","methods","residual-mode","attempt-mode","error-field"])$(id).addEventListener("change",render);
 document.querySelectorAll(".tabs button").forEach(button=>button.addEventListener("click",()=>switchPage(button.dataset.page)));
 $("theme").addEventListener("click",()=>{const t=document.documentElement.dataset.theme==="dark"?"light":"dark";document.documentElement.dataset.theme=t;try{localStorage.setItem("co2-fvm-theme",t);}catch{}render();});
 $("download-data").addEventListener("click",()=>download("co2_fvm_report_data.json","application/json",JSON.stringify(campaign,null,2)+"\n"));
 $("download-fields").addEventListener("click",downloadFields);$("download-history").addEventListener("click",downloadHistory);
 render();
}
window.nozzleReport={campaign,selectedRows,selectedHistoryRows,selectedCase,switchPage,render,status,errorValue:error,downloadFields,downloadHistory,snapshot:()=>({page,renderNumber,rendered,plotIds:lastPlots,caseId:selectedCase()?.id,mesh:Number($("mesh").value),rows:selectedRows().length})};
initialize();
})().catch(error=>{
 window.nozzleReportLoadError=String(error?.message||error);
 document.getElementById("campaign-state").textContent="Report data could not be loaded";
 const notice=document.getElementById("selection-notice");notice.style.display="block";notice.textContent=window.nozzleReportLoadError;
 console.error(error);
});
