(function bootAnnularReport(){
"use strict";
const data=JSON.parse(document.getElementById("report-data").textContent);
const $=id=>document.getElementById(id);
const INFO={
 p:{label:"Static pressure",unit:"kPa",scale:1e-3},
 rho:{label:"Density",unit:"kg/m³",scale:1},
 v_m:{label:"Meridional velocity",unit:"m/s",scale:1},
 v_theta:{label:"Tangential velocity",unit:"m/s",scale:1},
 T:{label:"Static temperature",unit:"K",scale:1},
 M_m:{label:"Meridional Mach number",unit:"—",scale:1},
 h:{label:"Static enthalpy",unit:"kJ/kg",scale:1e-3},
 h0:{label:"Stagnation enthalpy",unit:"kJ/kg",scale:1e-3},
 s:{label:"Specific entropy",unit:"J/(kg K)",scale:1},
 alpha_deg:{label:"Flow angle",unit:"deg",scale:1},
 M:{label:"Total Mach number",unit:"—",scale:1},
 mdot:{label:"Cell-state mass flow",unit:"kg/s",scale:1}
};
const GROUPS={primary:["p","rho","v_m","v_theta","T","M_m"],
 energy:["h","h0","s","alpha_deg","M","mdot"],all:Object.keys(INFO)};
const eqNames=["Mass","Meridional momentum","Angular momentum","Energy"];
const palette=["#6a00a8","#9b179e","#cc4778","#ed7953","#fca636"];
const number=n=>!Number.isFinite(n)?"—":Math.abs(n)>0&&(Math.abs(n)<1e-3||Math.abs(n)>=1e5)?n.toExponential(3):n.toLocaleString("en-GB",{maximumFractionDigits:4});
const esc=s=>String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
const max=a=>Math.max(...a.filter(Number.isFinite));
const broadcast=(a,n)=>Array.isArray(a)?a:Array(n).fill(a);
let page="solution",renderNumber=0,rendered=0,chain=Promise.resolve(),activeFields=[];
const drawn=new Set();

function variant(){return data.variants.find(v=>v.id===$("case").value);}
function run(){return variant().runs.find(r=>String(r.cells)===$("grid").value&&r.method_id===$("method").value);}
function theme(){
 const css=getComputedStyle(document.documentElement);
 return Object.fromEntries(["paper","ink","muted","grid","line"].map(k=>[k,css.getPropertyValue(`--${k}`).trim()]));
}
function color(c){
 if(document.documentElement.dataset.theme!=="dark")return c;
 const rgb=[1,3,5].map(i=>parseInt(c.slice(i,i+2),16));
 return `rgb(${rgb.map(v=>Math.round(v*.65+255*.35)).join(",")})`;
}
function layout(title,yTitle,extra={}){
 const t=theme();
 return {title:title?{text:title,x:.06,xanchor:"left",font:{size:13,color:t.ink}}:undefined,
  paper_bgcolor:t.paper,plot_bgcolor:t.paper,font:{family:'"Segoe UI",Arial,sans-serif',size:11,color:t.muted},
  margin:{l:69,r:20,t:title?48:16,b:51},
  xaxis:{title:{text:"Meridional arclength [mm]",standoff:10},gridcolor:t.grid,zeroline:false,ticks:"outside"},
  yaxis:{title:{text:yTitle,standoff:8},gridcolor:t.grid,zeroline:false,ticks:"outside",exponentformat:"power"},
  legend:{orientation:"h",x:0,y:1.04,xanchor:"left",yanchor:"bottom",font:{size:10}},
  hovermode:"closest",...extra};
}
async function plot(id,traces,options){
 const config={responsive:true,displaylogo:false,scrollZoom:false,displayModeBar:window.innerWidth>760?"hover":false,
  modeBarButtonsToRemove:["select2d","lasso2d"],toImageButtonOptions:{format:"png",scale:2,filename:`annular_${id}`}};
 await Plotly.react($(id),traces,options,config);
 drawn.add(id);
}
function markerIndices(n,skip){
 const step=Math.max(1,Math.floor(Number(skip)||0)+1),indices=[];
 for(let i=0;i<n;i+=step)indices.push(i);
 if(n&&indices.at(-1)!==n-1)indices.push(n-1);
 return indices;
}
function absoluteError(row,key){return row.fields[key].map((y,i)=>Math.abs(y-row.reference_fields[key][i]));}
function errorTrace(row,key){
 const values=absoluteError(row,key),info=INFO[key];
 return {type:"scatter",mode:"lines",name:"|FVM − NozzleX|",x:row.m.map(x=>x*1e3),
  y:values.map(v=>v===0?null:v*info.scale),customdata:values,
  line:{color:color(variant().color),width:1.65},connectgaps:false,
  hovertemplate:`m = %{x:.5f} mm<br>Absolute error = %{y:.5e} ${info.unit}<extra></extra>`};
}
function profileTraces(row,key){
 const v=variant(),ref=v.reference,info=INFO[key],traces=[],selection=$("display").value;
 if(selection!=="fvm")traces.push({type:"scatter",mode:"lines",name:"NozzleX · space marching",
  x:ref.m.map(x=>x*1e3),y:ref.fields[key].map(y=>y*info.scale),line:{color:color(v.color),width:2.1},
  hovertemplate:`m = %{x:.5f} mm<br>${info.label} = %{y:.7g} ${info.unit}<extra>NozzleX</extra>`});
 if(selection!=="space"){
  const indices=markerIndices(row.cells,$("skip").value);
  // This trace is last so its white faces sit above the space-marching line.
  traces.push({type:"scatter",mode:"markers",name:`FVM · ${row.cells} cells`,
   x:indices.map(i=>row.m[i]*1e3),y:indices.map(i=>row.fields[key][i]*info.scale),customdata:indices,
   marker:{color:"#ffffff",size:6.5,symbol:"circle",line:{color:color(v.color),width:1.6}},
   hovertemplate:`Cell %{customdata}<br>m = %{x:.5f} mm<br>${info.label} = %{y:.7g} ${info.unit}<extra>FVM</extra>`});
 }
 return traces;
}
function ensureFieldCards(keys){
 if(keys.join("|")===activeFields.join("|"))return;
 for(const key of activeFields)for(const prefix of ["field","error"]){
  const id=`${prefix}-${key}`;if($(id)){Plotly.purge($(id));drawn.delete(id);}
 }
 $("field-plots").replaceChildren();
 for(const key of keys){
  const card=document.createElement("article");card.className="field-card";
  const heading=document.createElement("h3");heading.className="field-title";heading.textContent=INFO[key].label;
  card.appendChild(heading);
  for(const [prefix,css] of [["field","field-chart"],["error","error-chart"]]){
   const div=document.createElement("div");div.id=`${prefix}-${key}`;div.className=css;card.appendChild(div);
  }
  $("field-plots").appendChild(card);
 }
 activeFields=[...keys];
}
function linkRanges(key,range){
 const first=$(`field-${key}`),second=$(`error-${key}`);
 if(first._annularLinked)return;
 let linking=false;
 function link(other,event){
  if(linking)return;
  let update=null;
  if(event["xaxis.range[0]"]!==undefined)update={"xaxis.range":[event["xaxis.range[0]"],event["xaxis.range[1]"]],"xaxis.autorange":false};
  else if(event["xaxis.range"])update={"xaxis.range":event["xaxis.range"],"xaxis.autorange":false};
  else if(event["xaxis.autorange"])update={"xaxis.range":range,"xaxis.autorange":false};
  if(update){linking=true;Promise.resolve(Plotly.relayout(other,update)).finally(()=>{linking=false;});}
 }
 first.on("plotly_relayout",e=>link(second,e));second.on("plotly_relayout",e=>link(first,e));
 first._annularLinked=true;
}
async function renderFields(row){
 const keys=GROUPS[$("field-group").value];ensureFieldCards(keys);
 const range=[0,variant().reference.m.at(-1)*1e3],revision=`${variant().id}|${row.cells}|${row.method_id}`;
 for(const key of keys){
  const info=INFO[key],options=layout("",`${info.label} [${info.unit}]`,{uirevision:revision});
  options.margin.t=37;options.xaxis.range=range;
  const err=layout(`Absolute error [${info.unit}]`,info.unit,{showlegend:false,uirevision:revision});
  err.yaxis.type="log";err.xaxis.range=range;err.margin={l:69,r:20,t:41,b:47};
  const zeros=absoluteError(row,key).filter(v=>v===0).length;
  if(zeros)err.annotations=[{xref:"paper",yref:"paper",x:1,y:1,xanchor:"right",yanchor:"top",showarrow:false,
   text:`${zeros} exact zero${zeros===1?"":"s"} omitted`,font:{size:9,color:theme().muted},bgcolor:theme().paper}];
  await plot(`field-${key}`,profileTraces(row,key),options);
  await plot(`error-${key}`,[errorTrace(row,key)],err);
  linkRanges(key,range);
 }
}
async function renderGeometry(){
 const g=variant().reference.geometry,options=layout("Imported cubic midline and walls","Radius r [m]");
 options.xaxis.title.text="Axial coordinate z [m]";
 options.yaxis.scaleanchor="x";options.yaxis.scaleratio=1;options.legend.y=-.28;
 await plot("geometry-shape",[
  {type:"scatter",mode:"lines",x:g.z_hub,y:g.r_hub,name:"Hub",line:{color:theme().ink,width:2}},
  {type:"scatter",mode:"lines",x:g.z_shroud,y:g.r_shroud,name:"Shroud",line:{color:theme().muted,width:2}},
  {type:"scatter",mode:"lines",x:g.z,y:g.r,name:"Midline",line:{color:color(variant().color),width:1.6,dash:"dash"}}
 ],options);
 await plot("geometry-area",[{type:"scatter",mode:"lines",x:variant().reference.m.map(x=>x*1e3),
  y:g.A.map(a=>a/g.A[0]),name:"A / A inlet",line:{color:color(variant().color),width:2}}],
  layout("Normal flow area","A / A inlet",{showlegend:false}));
}
function history(row){return row.history.filter(h=>h.accepted!==false&&Array.isArray(h.equation_norm));}
function positive(values){return values.map(v=>Number.isFinite(v)&&v>0?v:null);}
async function renderConvergence(row){
 const h=history(row),steps=h.map(r=>r.step);
 const residual=eqNames.map((name,k)=>({type:"scatter",mode:"lines",name,x:steps,y:positive(h.map(r=>r.equation_norm[k])),line:{color:color(palette[k]),width:1.8}}));
 const normLayout=layout("Original spatial equation residuals","Fixed-scale RMS");normLayout.yaxis.type="log";
 normLayout.shapes=[{type:"line",xref:"paper",x0:0,x1:1,y0:1e-9,y1:1e-9,line:{color:theme().muted,width:1,dash:"dot"}}];
 await plot("history-residual",residual,normLayout);
 const budgets=[{name:"Boundary",y:h.map(r=>r.boundary_max)},
  {name:"Global conservation",y:h.map(r=>max(r.global_defect))}].map((r,i)=>({type:"scatter",mode:"lines",name:r.name,x:steps,y:positive(r.y),line:{color:color(palette[i+1]),width:1.8}}));
 const budgetLayout=layout("Independent boundary and global checks","Normalized defect");budgetLayout.yaxis.type="log";
 budgetLayout.shapes=normLayout.shapes;await plot("history-budget",budgets,budgetLayout);
 const refLayout=layout("Mesh refinement against NozzleX","Max normalized conservative L₁");
 refLayout.xaxis.title.text="Number of cells";refLayout.xaxis.type="log";refLayout.yaxis.type="log";
 const rows=variant().runs.filter(r=>r.method_id===row.method_id).sort((a,b)=>a.cells-b.cells);
 refLayout.shapes=[{type:"line",xref:"paper",x0:0,x1:1,y0:1e-3,y1:1e-3,line:{color:theme().muted,width:1,dash:"dot"}}];
 await plot("grid-error",[{type:"scatter",mode:"lines+markers",name:variant().label,x:rows.map(r=>r.cells),
  y:rows.map(r=>max(r.errors.conservative_normalized_l1)),line:{color:color(variant().color),width:2},
  marker:{size:7,color:"white",line:{color:color(variant().color),width:1.5}}}],refLayout);
 const loads=row.wall_loads,x=row.m.map(m=>m*1e3);
 await plot("closure-components",["cf_wall","cf_diffusion","cf_curvature"].map((key,i)=>({type:"scatter",mode:"lines",name:["Wall","Diffusion","Curvature"][i],
  x,y:broadcast(loads[key],row.cells),line:{color:color(palette[i]),width:1.8}})),layout("Imported friction and loss components","Fanning / empirical coefficient"));
 await plot("wall-heat",[{type:"scatter",mode:"lines",name:"Wall heat",x,y:broadcast(loads.q_w,row.cells).map(v=>v*1e-3),line:{color:color(variant().color),width:2}}],
  layout("Heat flux into the fluid","q wall [kW/m²]",{showlegend:false}));
 const cflLayout=layout("Pseudo-time CFL schedule","CFL");cflLayout.xaxis.title.text="Accepted update / rejected trial";cflLayout.yaxis.type="log";
 await plot("history-cfl",[{type:"scatter",mode:"lines",name:"Accepted",x:steps,y:positive(h.map(r=>r.cfl)),line:{color:color(variant().color),width:1.8}},
  {type:"scatter",mode:"markers",name:"Rejected trial",x:row.history.filter(r=>r.accepted===false).map(r=>r.step),
   y:row.history.filter(r=>r.accepted===false).map(r=>r.cfl),marker:{symbol:"x",color:color(palette[3]),size:7}}],cflLayout);
}
function updateSummary(row){
 const shown=markerIndices(row.cells,$("skip").value).length;
 $("marker-count").textContent=`${shown} / ${row.cells} FVM markers · ${row.cells} error stations · endpoints retained`;
 $("kpi-success").textContent=row.success?"Converged":row.reason;
 $("kpi-method").textContent=`${row.cells} cells · ${row.method_label}`;
 $("kpi-residual").textContent=number(max(row.metrics.equation_norm));
 $("kpi-error").textContent=number(max(row.errors.conservative_normalized_l1));
 $("kpi-pressure").textContent=`${number(variant().reference.fields.p.at(-1)*1e-3)} kPa`;
 $("solution-context").textContent=`${variant().label} · ${row.cells} cells · ${row.method_label} · NozzleX Dopri5 reference`;
 $("convergence-context").textContent=`${variant().label} · ${row.cells} cells · original saved convergence and closure data`;
 const accepted=row.history.filter(h=>h.phase==="accepted").length,rejected=row.history.filter(h=>h.accepted===false).length;
 $("run-table").innerHTML=`<tr><td>${esc(variant().label)} / ${row.cells}</td><td>${esc(row.method_label)}</td><td><span class="badge ${row.success?"pass":"fail"}">${esc(row.reason)}</span></td><td>${number(row.errors.p.l1)}</td><td>${number(row.errors.v_m.l1)}</td><td>${number(row.errors.inlet_Mach_error)}</td><td>${accepted} / ${rejected}</td><td>${esc(row.initialization)}</td></tr>`;
}
function render(){
 const ticket=++renderNumber;
 chain=chain.then(async()=>{
  if(ticket!==renderNumber)return;
  const row=run();if(!row)throw Error("No saved run matches the selection");
  updateSummary(row);
  if(page==="solution")await renderFields(row);
  else if(page==="geometry")await renderGeometry();
  else if(page==="convergence")await renderConvergence(row);
  rendered=ticket;
 }).catch(error=>{window.annularReportLoadError=String(error);$("solution-context").textContent=`Report error: ${error}`;console.error(error);throw error;});
 return chain;
}
function switchPage(id){
 page=id;
 document.querySelectorAll(".page").forEach(p=>p.classList.toggle("active",p.id===id));
 document.querySelectorAll("[data-page]").forEach(b=>b.classList.toggle("active",b.dataset.page===id));
 return render();
}
function fillChoices(){
 const v=variant(),oldGrid=$("grid").value,oldMethod=$("method").value;
 const cells=[...new Set(v.runs.map(r=>r.cells))].sort((a,b)=>a-b);
 $("grid").replaceChildren(...cells.map(n=>new Option(`${n} cells`,String(n))));
 $("grid").value=cells.includes(Number(oldGrid))?oldGrid:String(cells.at(-1));
 const methods=[...new Map(v.runs.map(r=>[r.method_id,r.method_label]))];
 $("method").replaceChildren(...methods.map(([id,label])=>new Option(label,id)));
 $("method").value=methods.some(([id])=>id===oldMethod)?oldMethod:methods[0][0];
 $("method").disabled=methods.length===1;
 $("skip").max=String(Number($("grid").value)-1);
 $("skip").value=String(Math.min(Number($("skip").value)||0,Number($("skip").max)));
}
function download(name,text,type){
 const url=URL.createObjectURL(new Blob([text],{type}));
 const a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),30000);
}
function fieldsCSV(){
 const r=run(),keys=Object.keys(INFO),header=["m_m",...keys.flatMap(k=>[`${k}_fvm`,`${k}_nozzlex`,`${k}_absolute_error`])];
 const errors=Object.fromEntries(keys.map(k=>[k,absoluteError(r,k)]));
 const rows=r.m.map((x,i)=>[x,...keys.flatMap(k=>[r.fields[k][i],r.reference_fields[k][i],errors[k][i]])]);
 return [header,...rows].map(row=>row.join(",")).join("\n");
}
function historyCSV(){
 const r=run(),keys=["step","phase","accepted","cfl","dt","inner_iterations"];
 return [keys.join(","),...r.history.map(h=>keys.map(k=>h[k]??"").join(","))].join("\n");
}
function fillEvidence(){
 $("campaign-count").textContent=`${data.variants.flatMap(v=>v.runs).filter(r=>r.success).length} / ${data.variants.flatMap(v=>v.runs).length}`;
 $("campaign-table").innerHTML=data.variants.map(v=>{
  const r=[...v.runs].sort((a,b)=>b.cells-a.cells)[0];
  return `<tr><td>${esc(v.label)}</td><td>${r.cells}</td><td>${r.success?"Converged":esc(r.reason)}</td><td>${number(max(r.errors.conservative_normalized_l1))}</td><td>${number(r.errors.inlet_Mach_error)}</td><td>${number(v.reference.fields.p.at(-1)*1e-3)}</td></tr>`;
 }).join("");
 $("reference-table").innerHTML=data.variants.flatMap(v=>v.reference_audits.map(r=>`<tr><td>${esc(v.label)}</td><td>${r.tolerance.toExponential(0)}</td><td>${number(r.audit.mass_variation)}</td><td>${number(r.audit.entropy_integration_error)}</td><td><span class="badge ${r.audit.passed?"pass":"fail"}">${r.audit.passed?"Pass":"Fails strict target"}</span></td></tr>`)).join("");
 $("provenance").textContent=JSON.stringify({summary:data.summary,geometry:data.geometry_audit,presentation:data.presentation},null,2);
}
$("case").replaceChildren(...data.variants.map(v=>new Option(v.label,v.id)));
$("case").value=data.variants.some(v=>v.id==="aungier")?"aungier":data.variants[0].id;
fillChoices();fillEvidence();
$("case").addEventListener("change",()=>{fillChoices();render();});
$("grid").addEventListener("change",()=>{fillChoices();render();});
for(const id of ["method","display","field-group"])$(id).addEventListener("change",render);
$("skip").addEventListener("change",()=>{$("skip").value=String(Math.max(0,Math.min(Number($("skip").max),Math.floor(Number($("skip").value)||0))));render();});
document.querySelectorAll("[data-page]").forEach(b=>b.addEventListener("click",()=>switchPage(b.dataset.page)));
$("theme").addEventListener("click",()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==="dark"?"light":"dark";render();});
$("download-fields").addEventListener("click",()=>download(`annular_${variant().id}_n${run().cells}_fields.csv`,fieldsCSV(),"text/csv;charset=utf-8"));
$("download-history").addEventListener("click",()=>download(`annular_${variant().id}_n${run().cells}_history.csv`,historyCSV(),"text/csv;charset=utf-8"));
$("download-data").addEventListener("click",()=>download("annular_plotly_report_data.json",JSON.stringify(data),"application/json"));
window.annularReport={data,INFO,absoluteError,profileTraces,errorTrace,markerIndices,fieldsCSV,run,variant,render,switchPage,
 snapshot:()=>({page,renderNumber,rendered,case:$("case").value,cells:Number($("grid").value),method:$("method").value,
  display:$("display").value,skip:Number($("skip").value),fields:[...activeFields],theme:document.documentElement.dataset.theme})};
render();
})();
