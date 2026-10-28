/* Interactive presentation of retained evidence. No numerical solver runs here. */
(() => {
  "use strict";
  const bundle = JSON.parse(document.getElementById("report-data").textContent);
  const data = bundle.benchmark;
  const names = {production: "Internal BCs", coupled: "Coupled BC equations"};
  let active = "results", revision = 0, queue = Promise.resolve(), ready = Promise.resolve();
  let resolveMath;
  const mathReady = new Promise(resolve => { resolveMath = resolve; });
  const plots = new Map();
  const $ = id => document.getElementById(id);
  window.addEventListener("report-math-ready", () => resolveMath());
  function palette() {
    const css = getComputedStyle(document.documentElement);
    const get = key => css.getPropertyValue(key).trim();
    return {bg:get("--panel"),ink:get("--ink"),grid:get("--grid"),line:get("--line"),muted:get("--muted"),production:get("--source"),coupled:get("--working")};
  }
  function frame(id, xTitle, yTitle, height=365) {
    const p = palette();
    const axis = title => ({title:{text:title,font:{size:11}},color:p.ink,tickfont:{size:10},showline:true,mirror:true,linecolor:p.line,gridcolor:p.grid,zeroline:false,automargin:true});
    return {paper_bgcolor:p.bg,plot_bgcolor:p.bg,font:{family:'"Segoe UI",Arial,sans-serif',size:11,color:p.ink},height,
      margin:{l:62,r:18,t:18,b:88},xaxis:axis(xTitle),yaxis:axis(yTitle),hovermode:"closest",
      legend:{orientation:"h",x:0,y:-.27,font:{size:9}},uirevision:`${id}:${revision}`};
  }
  const config = id => ({responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"],toImageButtonOptions:{format:"svg",filename:id}});
  async function plot(id, traces, layout) {
    await Plotly.react($(id),traces,layout,config(id));
    plots.set(id,{traces,layout});
  }
  async function draw() {
    const p = palette(), work = [];
    data.cases.forEach((c,i) => {
      const residual=[], pressure=[];
      c.runs.forEach(r => {
        const line={color:p[r.mode],width:r.mode==="production"?2.7:1.8,dash:r.mode==="production"?"solid":"dash"};
        residual.push({type:"scatter",mode:"lines",x:r.history.map(h=>h.update),y:r.history.map(h=>h.residual_rms),name:names[r.mode],line});
        pressure.push({type:"scatter",mode:"lines",x:r.x_m.slice(1,-1),y:r.final_state.slice(1,-1).map(row=>row[0]/1000),name:names[r.mode],line});
      });
      const first=c.runs[0];
      pressure.push({type:"scatter",mode:"lines",x:first.x_m.slice(1,-1),y:first.pressure_exact_Pa.slice(1,-1).map(v=>v/1000),name:"Analytical",line:{color:p.muted,width:1.2,dash:"dot"}});
      const layout=frame(`residual-${i}`,"Accepted implicit update","Combined residual RMS");
      layout.yaxis.type="log";
      layout.shapes=[{type:"line",xref:"paper",x0:0,x1:1,y0:1e-8,y1:1e-8,line:{color:p.muted,width:1,dash:"dot"}}];
      work.push(plot(`residual-${i}`,residual,layout));
      work.push(plot(`pressure-${i}`,pressure,frame(`pressure-${i}`,"Axial position [m]","Static pressure [kPa]")));
    });
    const timings=[];
    for (const [mode,shift] of [["production",-.12],["coupled",.12]]) {
      const x=[],y=[],means=[],upper=[],lower=[],positions=[];
      data.cases.forEach((c,i)=> {
        const samples=c.measured_runs.filter(r=>r.mode===mode);
        samples.forEach((r,j)=> {x.push(i+shift+(j-(samples.length-1)/2)*.008);y.push(r.solve_s);});
        const t=c.timings[mode];positions.push(i+shift);means.push(t.mean_s);upper.push(t.max_s-t.mean_s);lower.push(t.mean_s-t.min_s);
      });
      timings.push({type:"scatter",mode:"markers",x,y,name:names[mode]+" samples",marker:{color:p[mode],size:5,opacity:.45},hovertemplate:"%{y:.5f} s<extra>%{fullData.name}</extra>"});
      timings.push({type:"scatter",mode:"markers",x:positions,y:means,name:names[mode]+" mean/range",marker:{color:p[mode],size:9,symbol:"diamond"},error_y:{type:"data",symmetric:false,array:upper,arrayminus:lower,color:p[mode],thickness:1.5,width:5}});
    }
    const timingLayout=frame("timings","Nozzle case","Complete measured solve [s]",460);
    timingLayout.xaxis.tickmode="array";timingLayout.xaxis.tickvals=data.cases.map((_,i)=>i);timingLayout.xaxis.ticktext=data.cases.map(c=>c.id);
    timingLayout.yaxis.rangemode="tozero";
    work.push(plot("timings",timings,timingLayout));
    await Promise.all(work);
    $("report-status").textContent=`${data.cases.length} cases; ${data.cases.reduce((n,c)=>n+c.measured_runs.length,0)} measured solves; recorded data unchanged`;
  }
  function renderFigures() {queue=queue.then(draw);return queue;}
  function activate(tab,target=tab) {
    if (!["results","theory","recommendations"].includes(tab)) tab="results";
    active=tab;document.body.dataset.page=tab;document.documentElement.dataset.reportReady="false";
    document.querySelectorAll("[data-report-page]").forEach(e=>{e.hidden=e.dataset.reportPage!==tab;});
    document.querySelectorAll("[data-report-nav]").forEach(e=>{e.hidden=e.dataset.reportNav!==tab;});
    document.querySelectorAll("[data-report-tab]").forEach(e=>{if(e.dataset.reportTab===tab)e.setAttribute("aria-current","page");else e.removeAttribute("aria-current");});
    $("sidebar-label").textContent={results:"Results",theory:"Theory & implementation",recommendations:"Recommendations"}[tab];
    ready=Promise.all([mathReady,tab==="results"?renderFigures():null]).then(()=>{
      if(active!==tab)return;
      document.documentElement.dataset.reportReady="true";
      if(target===tab)window.scrollTo({top:0,left:0,behavior:"instant"});
      else $(target)?.scrollIntoView({behavior:"instant"});
    }).catch(error=>{document.documentElement.dataset.reportError=error.message;console.error(error);});
    return ready;
  }
  function navigate() {const target=decodeURIComponent(location.hash.slice(1))||"results";return activate(target.split("-")[0],target);}
  document.addEventListener("click",event=>{
    const link=event.target.closest('a[href^="#"]');if(!link)return;
    const target=link.getAttribute("href").slice(1),tab=target.split("-")[0];
    if(!["results","theory","recommendations"].includes(tab))return;
    event.preventDefault();history.pushState(null,"",`#${target}`);activate(tab,target);
  });
  window.addEventListener("hashchange",navigate);
  document.querySelectorAll("[data-theme-choice]").forEach(button=>{
    button.setAttribute("aria-pressed",String(button.dataset.themeChoice===document.documentElement.dataset.theme));
    button.addEventListener("click",()=>{
      const theme=button.dataset.themeChoice;document.documentElement.dataset.theme=theme;
      try {localStorage.setItem("nozzle-bc-residual.report.theme",theme);} catch {}
      document.querySelectorAll("[data-theme-choice]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));activate(active);
    });
  });
  $("reset-plots").addEventListener("click",()=>{revision++;activate("results");});
  function download(text,type,name) {const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}
  $("download-csv").addEventListener("click",()=>download(bundle.timing_csv,"text/csv;charset=utf-8","BC_residual_timings.csv"));
  $("download-json").addEventListener("click",()=>download(JSON.stringify(data,null,2),"application/json","BC_residual_evidence.json"));
  $("license-button").addEventListener("click",()=>{const d=$("license-details");d.hidden=false;d.open=!d.open;d.scrollIntoView();});
  $("print-report").addEventListener("click",async()=>{document.body.classList.add("printing");await Promise.all([mathReady,renderFigures()]);window.print();document.body.classList.remove("printing");activate(active);});
  window.bcResidualReport={bundle,plots,selectTab:activate,ready:()=>ready,get active(){return active;}};
  navigate();
})();