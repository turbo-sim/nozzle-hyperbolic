/* Local-only report UI. No solver, fetch, CDN, analytics, or external service. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const themeKey = "euler-annular.report.theme";
  const methods = ["constant", "muscl_van_leer"];
  const labels = ["Analytical reference", "FVM (constant reconstruction)", "FVM (linear / MUSCL reconstruction)"];
  const names = {
    radial_outflow: "Radial outflow · low Mach",
    radial_inflow: "Radial inflow · subsonic",
    sonic_exit: "Radial inflow · sonic exit",
    swirl_diffusion: "Swirl-dominated diffusion",
    conical_subsonic: "Conical channel",
    curved_subsonic: "Curved NURBS channel",
    supersonic_expansion: "Supersonic radial expansion",
    shock35: "Shocked outflow · 35%",
    shock70: "Shocked outflow · 70%"
  };
  const explanations = {
    radial_outflow: "Subsonic pressure recovery in a constant-width radial channel.",
    radial_inflow: "Subsonic acceleration toward a smaller radius, with constant normal width.",
    sonic_exit: "The choking limit of a constant-width radial inflow; an internal smooth transonic branch is unavailable.",
    swirl_diffusion: "Total inlet Mach exceeds one while meridional Mach remains subsonic; the swirled flow diffuses smoothly.",
    conical_subsonic: "A straight conical midline with width defined by the cant and wall semi-divergence angles.",
    curved_subsonic: "A quarter-circle NURBS midline with constant normal width and nonzero inlet swirl.",
    supersonic_expansion: "Meridionally supersonic radial outflow accelerates through the diverging passage.",
    shock35: "A prescribed supersonic inlet, internal shock at 35% of the channel, then subsonic diffusion.",
    shock70: "The same supersonic inlet at a lower backpressure, placing the internal shock at 70%."
  };
  const escape = value => String(value ?? "—").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
  const fmt = (value, digits=3) => Number.isFinite(value) ? value.toExponential(digits) : "n/a";
  const pretty = value => Number.isFinite(value) ? value.toLocaleString("en", {maximumFractionDigits:3}) : "n/a";
  const badge = (ok, text) => `<span class="badge ${ok ? "pass" : "fail"}">${ok ? "✓" : "!"} ${escape(text ?? (ok ? "Met" : "Not met"))}</span>`;
  const plots = new Map();
  let DATA, state;
  function palette() {
    const css = getComputedStyle(document.documentElement);
    const value = name => css.getPropertyValue(name).trim();
    return {bg:value("--panel"), ink:value("--ink"), muted:value("--muted"),
      axis:value("--line"), grid:value("--grid"), accent:value("--accent"),
      colors:[document.documentElement.dataset.theme === "dark" ? "#b277ef" : "#6a00a8", "#cc4778", "#fca636"]};
  }
  function frame(extra={}) {
    const p=palette();
    return {paper_bgcolor:p.bg, plot_bgcolor:p.bg,
      font:{family:'"Segoe UI", Arial, sans-serif',size:11,color:p.ink},
      margin:{l:67,r:23,t:42,b:56}, hovermode:"closest", dragmode:"zoom",
      hoverlabel:{bgcolor:p.bg,bordercolor:p.axis,font:{family:'"Segoe UI", sans-serif',color:p.ink,size:12}},
      legend:{orientation:"h",x:0,y:1.12,xanchor:"left",yanchor:"bottom",font:{size:10},
        bgcolor:"rgba(0,0,0,0)",groupclick:"togglegroup"},
      showlegend:true, ...extra};
  }
  function axis(title, extra={}) {
    const p=palette();
    return {title:{text:title,standoff:9,font:{size:11}},showgrid:true,gridcolor:p.grid,
      gridwidth:0.8,zeroline:false,showline:true,linecolor:p.axis,linewidth:1,
      mirror:true,ticks:"outside",ticklen:4,tickcolor:p.axis,tickfont:{size:10},
      automargin:true,exponentformat:"power",...extra};
  }
  function config(name) {
    return {responsive:true,displaylogo:false,scrollZoom:false,
      modeBarButtonsToRemove:["lasso2d","select2d"],
      toImageButtonOptions:{format:"svg",filename:name,width:1300,height:730,scale:1}};
  }
  async function plot(id, traces, layout, filename) {
    const el=$(id);
    if (!el) return;
    await Plotly.react(el,traces,layout,config(filename));
    plots.set(id,{traces,layout,filename});
  }
  function scientificTrace(index, x, y, extra={}) {
    const p=palette();
    return {type:"scatter",x,y,name:labels[index],legendgroup:String(index),
      mode:"lines", zorder:index===0 ? 2 : 1,
      line:{color:p.colors[index],width:index===0 ? 2.4 : 2,dash:["dash","dash","solid"][index]},
      connectgaps:false,meta:{method:index===0 ? "analytical" : methods[index-1]},
      hovertemplate:"m = %{x:.6g} m<br>%{y:.7g}<extra>%{fullData.name}</extra>", ...extra};
  }
  async function fieldFigure(id, mesh, fields, kind) {
    const narrow=$(id).clientWidth<620, columns=narrow ? 1 : 3, rows=Math.ceil(fields.length/columns);
    const gapX=0.115, gapY=narrow ? 0.073 : 0.20;
    const width=(1-gapX*(columns-1))/columns, height=(1-gapY*(rows-1))/rows;
    const layout=frame({margin:{l:narrow ? 69 : 72,r:27,t:65,b:53},
      height:narrow ? 1580 : ($(id).clientWidth<930 ? 610 : 660),
      uirevision:`${kind}:${kind==="fields" ? state.caseId : "contact"}:${mesh.cells}`,
      legend:{orientation:"h",x:0,y:narrow ? 1.028 : 1.08,xanchor:"left",yanchor:"bottom",font:{size:narrow ? 9 : 11},groupclick:"togglegroup"}});
    const traces=[];
    fields.forEach((f,i)=>{
      const col=i%columns,row=Math.floor(i/columns),suffix=i===0 ? "" : String(i+1);
      const left=col*(width+gapX),top=1-row*(height+gapY),bottom=top-height;
      layout[`xaxis${suffix}`]=axis("Meridional arclength [m]",{domain:[left,left+width],anchor:`y${suffix}`});
      layout[`yaxis${suffix}`]=axis(f.unit ? `${f.label} [${f.unit}]` : f.label,{domain:[bottom,top],anchor:`x${suffix}`});
      const ys=[mesh.reference[f.key],...methods.map(m=>mesh.methods[m].fields[f.key])];
      // A flat analytic field needs a physical, non-offset range for readable axes.
      if (ys[0].every(v=>Math.abs(v-ys[0][0])<=1e-12*Math.max(1,Math.abs(v)))) {
        const center=ys[0][0];
        const spread=Math.max(...ys.flat().map(v=>Math.abs(v-center)));
        const half=1.1*Math.max(.005*Math.abs(center),1e-6,spread);
        layout[`yaxis${suffix}`].range=[center-half,center+half];
        layout[`yaxis${suffix}`].exponentformat="none";
      }
      // Complete numerical lines first and analytical reference above; no solution markers.
      [1,2,0].forEach(k=>traces.push(scientificTrace(k,mesh.m,ys[k],{xaxis:`x${suffix}`,yaxis:`y${suffix}`,showlegend:i===0})));
    });
    await plot(id,traces,layout,`${kind}_${state.caseId}_${mesh.cells}`);
  }
  function currentCase() { return DATA.cases.find(c=>c.id===state.caseId); }
  function currentMesh() { return currentCase().meshes.find(m=>m.cells===state.cells); }
  function directionMarkers(arrows, horizontal, vertical, color, minimumLength=0) {
    // Marker sizes are pixels: short projected chords must not hide arrowheads.
    const visible=arrows.filter(a=>Math.hypot(
      a.head[horizontal]-a.tail[horizontal],a.head[vertical]-a.tail[vertical])>minimumLength);
    if(!visible.length)return null;
    return {type:"scatter",mode:"markers",zorder:5,
      x:visible.map(a=>a.head[horizontal]),y:visible.map(a=>a.head[vertical]),
      marker:{symbol:"arrow",size:15,color,angleref:"up",
        angle:visible.map(a=>Math.atan2(a.head[horizontal]-a.tail[horizontal],
          a.head[vertical]-a.tail[vertical])*180/Math.PI),
        line:{color,width:.5}},
      name:"Flow direction",legendgroup:"paths",showlegend:false,hoverinfo:"skip",
      meta:{role:"flow-direction",fixed_pixel_size:15}};
  }
  async function channelView(view) {
    const channelId=state.channelCase==="current" ? state.caseId : state.channelCase;
    const record=DATA.channels.cases[channelId],g=record.geometry,p=palette();
    const dark=document.documentElement.dataset.theme==="dark";
    const wall=dark ? "#a0b2ca" : "#647892",blue=dark ? "#88b7ff" : "#2462d1";
    const traces=[],annotations=[];
    let layout=frame({height:430,margin:{l:70,r:25,t:45,b:55},
      uirevision:`channel:${channelId}:${view}`});
    $("channel-description").textContent=`${names[channelId] ?? (channelId==="transient" ? "Transient constant-radius annulus" : "Nonuniform radial outflow")} · L = ${pretty(g.length)} m · radius ${pretty(Math.min(...g.r))}–${pretty(Math.max(...g.r))} m`;
    if(view==="annular") {
      const theta=Array.from({length:49},(_,i)=>2*Math.PI*i/48);
      for(const side of ["minus","plus"]){
        const w=record.walls[side];
        traces.push({type:"surface",x:theta.map(()=>w.x),
          y:theta.map(t=>w.r.map(r=>r*Math.cos(t))),z:theta.map(t=>w.r.map(r=>r*Math.sin(t))),
          colorscale:[[0,wall],[1,wall]],opacity:.16,showscale:false,
          hoverinfo:"skip",name:"Channel wall",showlegend:false,
          lighting:{ambient:.8,diffuse:.6,specular:.1}});
        for(const k of [0,w.x.length-1]) traces.push({type:"scatter3d",mode:"lines",
          x:theta.map(()=>w.x[k]),y:theta.map(t=>w.r[k]*Math.cos(t)),z:theta.map(t=>w.r[k]*Math.sin(t)),
          line:{color:wall,width:2},hoverinfo:"skip",showlegend:false});
      }
      record.paths_3d.forEach((path,i)=>{
        const color=blue;
        const arrowLength=.025*Math.max(g.length,2*Math.max(...g.r));
        traces.push({type:"scatter3d",mode:"lines",x:path.x,y:path.y,z:path.z,
          line:{color,width:2},opacity:1,
          name:"Illustrative flow paths",legendgroup:"paths",showlegend:i===0,
          hovertemplate:"x = %{x:.5f} m<br>y = %{y:.5f} m<br>z = %{z:.5f} m<extra>Approximate quasi-1D flow direction</extra>"});
        traces.push({type:"cone",x:path.arrows.map(a=>a.head[0]),y:path.arrows.map(a=>a.head[1]),z:path.arrows.map(a=>a.head[2]),
          u:path.arrows.map(a=>arrowLength*a.direction[0]),v:path.arrows.map(a=>arrowLength*a.direction[1]),w:path.arrows.map(a=>arrowLength*a.direction[2]),
          colorscale:[[0,color],[1,color]],showscale:false,sizemode:"raw",
          anchor:"tip",hoverinfo:"skip",showlegend:false});
      });
      layout={...layout,margin:{l:0,r:0,t:34,b:0},
        scene:{aspectmode:"data",bgcolor:p.bg,camera:{eye:{x:1.7,y:1.55,z:1.1},up:{x:0,y:0,z:1}},
          xaxis:axis("Axial x [m]",{backgroundcolor:p.bg}),
          yaxis:axis("Transverse y [m]",{backgroundcolor:p.bg}),
          zaxis:axis("Transverse z [m]",{backgroundcolor:p.bg})},
        legend:{orientation:"h",x:.02,y:1.02,xanchor:"left",font:{size:10}}};
    } else if(view==="frontal") {
      // Look along the axial x direction: project the same 3D paths onto (y,z).
      const theta=Array.from({length:97},(_,i)=>2*Math.PI*i/96);
      const radii=[...record.walls.minus.r,...record.walls.plus.r];
      const inner=Math.min(...radii),outer=Math.max(...radii);
      const circle=radius=>({x:theta.map(t=>radius*Math.cos(t)),y:theta.map(t=>radius*Math.sin(t))});
      const a=circle(outer),b=circle(inner);
      traces.push({type:"scatter",mode:"lines",
        x:[...a.x,...b.x.slice().reverse(),a.x[0]],y:[...a.y,...b.y.slice().reverse(),a.y[0]],
        fill:"toself",fillcolor:dark ? "rgba(136,183,255,.08)" : "rgba(36,98,209,.05)",
        line:{width:0},hoverinfo:"skip",showlegend:false});
      [inner,outer].forEach((radius,i)=>{
        const c=circle(radius);
        traces.push({type:"scatter",mode:"lines",x:c.x,y:c.y,line:{color:wall,width:1.5},
          name:"Projected passage envelope",legendgroup:"walls",showlegend:i===0,hoverinfo:"skip"});
      });
      record.paths_3d.forEach((path,i)=>{
        traces.push({type:"scatter",mode:"lines",x:path.y,y:path.z,line:{color:blue,width:2},
          name:"Illustrative flow paths",legendgroup:"paths",showlegend:i===0,
          hovertemplate:"y = %{x:.5f} m<br>z = %{y:.5f} m<extra>Axial projection of the 3D flow path</extra>"});
        const markers=directionMarkers(path.arrows,1,2,blue,1e-12*outer);
        if(markers)traces.push(markers);
      });
      if(!traces.some(t=>t.meta?.role==="flow-direction"))annotations.push({
        xref:"paper",yref:"paper",x:.5,y:1.04,showarrow:false,
        text:"Flow is axial: no transverse direction in this projection",font:{size:10,color:blue}});
      layout={...layout,xaxis:axis("Transverse y [m]"),
        yaxis:axis("Transverse z [m]",{scaleanchor:"x",scaleratio:1}),annotations};
    } else if(view==="meridional") {
      const a=record.walls.minus,b=record.walls.plus;
      traces.push({type:"scatter",mode:"lines",x:[...a.x,...b.x.slice().reverse(),a.x[0]],y:[...a.r,...b.r.slice().reverse(),a.r[0]],
        fill:"toself",fillcolor:dark ? "rgba(136,183,255,.08)" : "rgba(36,98,209,.05)",
        line:{width:0},hoverinfo:"skip",showlegend:false});
      for(const side of ["minus","plus"]) traces.push({type:"scatter",mode:"lines",x:record.walls[side].x,y:record.walls[side].r,
        name:side==="minus" ? "Channel walls" : "Channel wall",legendgroup:"walls",showlegend:side==="minus",line:{color:wall,width:2},hoverinfo:"skip"});
      record.cross_gap_paths.forEach((path,i)=>{
        traces.push({type:"scatter",mode:"lines",x:path.x,y:path.r,line:{color:blue,width:2},name:"Illustrative direction paths",legendgroup:"paths",showlegend:i===0,hoverinfo:"skip"});
        const markers=directionMarkers(path.arrows,0,1,blue);
        if(markers)traces.push(markers);
      });
      layout={...layout,xaxis:axis("Axial coordinate x [m]"),yaxis:axis("Radius r [m]",{scaleanchor:"x",scaleratio:1}),annotations};
    }
    try {
      await plot(`channel-${view}`,traces,layout,`channel_${channelId}_${view}`);
    } catch(error) {
      if(view!=="annular")throw error;
      $("channel-annular").textContent="The 3D renderer is unavailable on this device. The meridional view remains available.";
    }
  }
  async function renderChannel() {
    await Promise.all([channelView("meridional"),channelView("frontal"),channelView("annular")]);
    $("channel-caption").textContent="Left: equal-scale meridional section of the actual NURBS walls. Center: frontal projection along the axial x direction, using the same (y,z) paths as the 3D view and the projected passage envelope. Right: the revolved annular channel. All paths and arrows share one color. The 2D arrowheads have a fixed pixel size and are drawn above the streamlines to remain visible. Arrows follow increasing meridional distance, including radial inflow; purely axial motion is labeled in the frontal view because it has no transverse direction. Swirl paths use saved finest-grid MUSCL velocities and illustrate quasi-1D flow direction without resolving a cross-gap velocity profile.";
  }
  function methodRows(mesh) { return methods.map(m=>mesh.methods[m]); }
  function historyRows(history, accepted=true) {
    return (accepted ? history.accepted : history.rejected).map(row=>Object.fromEntries(history.columns.map((key,i)=>[key,row[i]])));
  }
  async function convergence(mesh) {
    const traces=methods.map((method,i)=>{
      const rows=historyRows(mesh.methods[method].history);
      const relative=rows.every(row=>Array.isArray(row.equation_relative));
      const y=rows.map(row=>relative ? Math.max(...row.equation_relative.filter((_,j)=>row.relative_active[j]))
        : Math.max(row.spatial_rms,row.boundary_max,row.conservation_max));
      return scientificTrace(i+1,rows.map(row=>row.step),y.map(value=>Number.isFinite(value)&&value>0?value:null),{
        hovertemplate:"Accepted update %{x}<br>Overall R %{y:.6e}<extra>%{fullData.name}</extra>"});
    });
    const monitored=methods.every(method=>mesh.methods[method].monitoring);
    const target=monitored?mesh.methods.constant.monitoring.policy.relative_tolerance:mesh.settings.tolerance;
    $("convergence-note").textContent=monitored?
      "R is the maximum active initial-relative equation norm. It starts at one and must reach 10^-9. Absolute residual, boundary, global conservation, regime and configured monitor checks must also pass; R alone does not certify success.":
      "Historical definition: maximum of spatial RMS, boundary maximum and global defect. These are not initial-normalized residuals.";
    await plot("convergence-plot",traces,frame({xaxis:axis("Accepted updates (initial state = 0)"),
      yaxis:axis(monitored?"Overall relative residual R":"Historical overall defect",{type:"log"}),
      margin:{l:79,r:22,t:49,b:58},uirevision:`overall:${state.caseId}:${state.cells}`,
      shapes:[{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:target,y1:target,
        line:{color:palette().muted,width:1,dash:"dot"}}]}),`overall_${state.caseId}_${state.cells}`);
    $("convergence-checks").innerHTML=methods.map((method,i)=>{
      const run=mesh.methods[method],m=run.metrics,p=run.monitoring?.policy;
      if(!p)return "";
      const active=m.equation_relative.filter((_,j)=>m.relative_active[j]);
      return `<div class="badges"><strong>${i===0?"Constant":"MUSCL"}</strong> `+
        badge(active.every(value=>value<=p.relative_tolerance),"Relative target")+" "+
        badge(m.equation_norm.every(value=>value<=p.absolute_tolerance),"Absolute safeguard")+" "+
        badge(m.boundary_passed,"Boundary")+" "+badge(m.global_passed,"Global balances")+" "+
        badge(m.monitors_passed,"Integral monitors")+" "+badge(run.root_pass,"Complete convergence")+
        `<span class="subtext">${escape(run.message)}</span></div>`;
    }).join("");
  }
  async function equationConvergence(mesh) {
    const sampled=methods.map(m=>historyRows(mesh.methods[m].history));
    const equationView=sampled.every(rows=>rows.length&&rows.every(r=>Array.isArray(r.equation_relative)));
    if(equationView){
      const quantities=["R1 mass","R2 meridional","R3 angular","R4 energy"];
      const colors=[palette().colors[0],"#b12a90","#e16462","#fca636"],traces=[];
      quantities.forEach((name,j)=>methods.forEach((method,i)=>{
        const rows=sampled[i],active=rows[0].relative_active[j];
        traces.push({type:"scatter",mode:"lines",x:rows.map(r=>r.step),
          y:rows.map(r=>active?r.equation_relative[j]:null),
          name:`${name} / ${i===0?"const":"MUSCL"}${active?"":" (abs.)"}`,
          meta:{method,quantity:j},line:{color:colors[j],width:2,dash:i===0?"dash":"solid"},
          hovertemplate:"Accepted update %{x}<br>R / R(initial) %{y:.6e}<extra>%{fullData.name}</extra>"});
      }));
      const tol=mesh.methods.constant.monitoring.policy.relative_tolerance;
      $("equation-residual-note").textContent="Each curve is a volume-weighted RMS of cell equation defects, divided by its own initial value. R1: mass; R2: meridional momentum; R3: angular momentum; R4: energy. Initially satisfied equations use absolute checks and have no invented relative curve. Colors match the global balances; dashed = constant, solid = MUSCL.";
      await plot("equation-residual-plot",traces,frame({xaxis:axis("Accepted updates (initial state = 0)"),
        yaxis:axis("Per-equation residual / initial residual",{type:"log"}),
        margin:{l:79,r:22,t:126,b:58},uirevision:`convergence:${state.caseId}:${state.cells}`,
        legend:{orientation:"h",entrywidthmode:"fraction",entrywidth:.5,x:0,y:1.04,xanchor:"left",yanchor:"bottom",font:{size:10}},
        shapes:[{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:tol,y1:tol,line:{color:palette().muted,width:1,dash:"dot"}}]}),`convergence_${state.caseId}_${state.cells}`);
      return;
    }
    $("equation-residual-note").textContent="Per-equation relative histories were not recorded in this historical dataset. No missing equation curves are inferred.";
    const traces=methods.map((m,i)=>{
      const rows=historyRows(mesh.methods[m].history);
      const y=rows.map(r=>Math.max(r.spatial_rms,r.boundary_max,r.conservation_max));
      const t=scientificTrace(i+1,rows.map(r=>r.step),y);
      t.mode="lines";
      t.hovertemplate="Accepted update %{x}<br>Original defect %{y:.5e}<extra>%{fullData.name}</extra>";
      return t;
    });
    const p=palette(),tol=mesh.settings.tolerance;
    await plot("equation-residual-plot",traces,frame({xaxis:axis("Accepted implicit updates"),
      yaxis:axis("Max(original spatial, BC, balance)",{type:"log"}),
      margin:{l:79,r:22,t:49,b:58},uirevision:`convergence:${state.caseId}:${state.cells}`,
      shapes:[{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:tol,y1:tol,line:{color:p.muted,width:1,dash:"dot"}}],
      annotations:[{xref:"paper",x:.99,yref:"y",y:Math.log10(tol),text:"10⁻⁸ target",showarrow:false,xanchor:"right",yshift:11,font:{color:p.muted,size:10}}]}),`convergence_${state.caseId}_${state.cells}`);
  }
  async function integralMonitors(mesh){
    const definitions=[{id:"cp-monitor-plot",key:"Cp_mean",label:"Mean Cp along domain"},
      {id:"mdot-monitor-plot",key:"mdot_out",label:"Outlet numerical mass flow [kg/s]"}];
    let available=0;
    for(const definition of definitions){
      const traces=[];
      methods.forEach((method,i)=>{
        const rows=historyRows(mesh.methods[method].history).filter(r=>Number.isFinite(r.monitors?.[definition.key]?.value));
        if(!rows.length)return;
        available++;
        traces.push(scientificTrace(i+1,rows.map(r=>r.step),rows.map(r=>r.monitors[definition.key].value),{
          customdata:rows.map(r=>[r.monitors[definition.key].range,r.monitors[definition.key].threshold,r.monitors[definition.key].stable]),
          hovertemplate:"Accepted update %{x}<br>Value %{y:.8g}<br>Window range %{customdata[0]:.3e}<br>Threshold %{customdata[1]:.3e}<br>Stable %{customdata[2]}<extra>%{fullData.name}</extra>"}));
      });
      await plot(definition.id,traces,frame({xaxis:axis("Accepted implicit updates (initial state = 0)"),
        yaxis:axis(definition.label),margin:{l:79,r:22,t:49,b:58},
        uirevision:`${definition.key}:${state.caseId}:${state.cells}`,
        annotations:traces.length?[]:[{xref:"paper",yref:"paper",x:.5,y:.5,text:"Not recorded in these saved histories",showarrow:false,font:{color:palette().muted,size:13}}]}),`${definition.key}_${state.caseId}_${state.cells}`);
    }
    $("integral-monitor-note").textContent=available?"Configured engineering quantities sampled at the initial and every accepted state. Cp uses the instantaneous inlet pressure and fixed reservoir pressure; mass flow uses the actual numerical outlet face flux. Stationarity is the peak-to-peak range over a complete accepted-state window, compared with the configured fixed-scale threshold. These checks supplement residual convergence; stable output alone does not certify the equations.":"The legacy saved runs did not record engineering-monitor trajectories. No curves are reconstructed from final fields.";
    $("engineering-monitor-row").hidden=!available;
  }
  async function cflHistory(mesh) {
    const traces=methods.map((method,i)=>{
      const history=mesh.methods[method].history;
      const rows=[...historyRows(history).map(row=>({...row,accepted:true})),
        ...historyRows(history,false).map(row=>({...row,accepted:false}))]
        .sort((a,b)=>a.step-b.step || (a.retry??0)-(b.retry??0)).filter(row=>row.phase!=="initial"&&Number.isFinite(row.cfl));
      return scientificTrace(i+1,rows.map(row=>row.step),rows.map(row=>row.cfl),{
        customdata:rows.map(row=>[row.retry??0,row.accepted?"accepted":"rejected"]),
        hovertemplate:"Update %{x}<br>CFL %{y:.6g}<br>Retry %{customdata[0]}; %{customdata[1]}<extra>%{fullData.name}</extra>"});
    });
    await plot("cfl-plot",traces,frame({xaxis:axis("Accepted-update index (retries share x)"),
      yaxis:axis("Recorded trial CFL",{type:"log"}),margin:{l:76,r:22,t:49,b:58},
      uirevision:`cfl:${state.caseId}:${state.cells}`}),`cfl_${state.caseId}_${state.cells}`);
  }
  async function refinement() {
    const c=currentCase(),shocked=c.is_shocked;
    $("refinement-case").textContent=names[c.id];
    const x=c.meshes.map(m=>m.cells);
    const graph=async(id,metric,title,limit)=>{
      const traces=methods.map((m,i)=>{
        const y=c.meshes.map(mesh=>metric(mesh.methods[m].errors));
        return scientificTrace(i+1,x,y,{mode:"lines",
          hovertemplate:"N = %{x}<br>Error %{y:.6e}<extra>%{fullData.name}</extra>"});
      });
      const layout=frame({xaxis:axis("Physical cells N",{type:"log",tickmode:"array",tickvals:x,ticktext:x.map(String)}),
        yaxis:axis(title,{type:"log"}),margin:{l:72,r:24,t:47,b:57},uirevision:`${id}:${c.id}`,
        shapes:shocked ? [] : [{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:limit,y1:limit,line:{color:palette().muted,width:1,dash:"dot"}}]});
      await plot(id,traces,layout,`${id}_${c.id}`);
    };
    await graph("refinement-plot",e=>e.Cp_mean_absolute,"Mean absolute Cp error along domain",0.01);
    await graph("conservative-plot",e=>Math.max(...e.conservative_L1),"Max conservative L₁ error",1e-3);
    const coarse=c.meshes.find(m=>m.cells===200),fine=c.meshes.find(m=>m.cells===400);
    const orders=methods.map(m=>Math.log2(coarse.methods[m].errors.Cp_mean_absolute/fine.methods[m].errors.Cp_mean_absolute));
    $("refinement-order").textContent=`Meridional-length mean absolute Cp difference over all physical cells. Observed order, N = 200 → 400: constant ${orders[0].toFixed(3)}, MUSCL ${orders[1].toFixed(3)}. ${shocked ? "The shock acceptance criterion uses location; Cp is supplementary." : c.id==="sonic_exit" ? "The sonic-exit limit is reported separately from strictly smooth branches." : "This is a two-grid observation, not an assumed formal order."}`;
  }
  async function renderCase(options={}) {
    const mesh=currentMesh(),meta=mesh.metadata;
    document.querySelectorAll("#mesh-select button").forEach(b=>b.setAttribute("aria-pressed",String(Number(b.dataset.cells)===state.cells)));
    $("case-description").innerHTML=`<strong>${escape(names[state.caseId])}</strong> · backpressure ${pretty(meta.backpressure/1000)} kPa · α<sub>in</sub> ${pretty(meta.alpha_in*180/Math.PI)}°`;
    const bothConverged=methodRows(mesh).every(r=>r.root_pass);
    $("case-status").innerHTML=`${badge(bothConverged,bothConverged?"Both converged":"Convergence target missed")} ${badge(mesh.methods.constant.accuracy_pass,"Constant accuracy")} ${badge(mesh.methods.muscl_van_leer.accuracy_pass,"MUSCL accuracy")}`;
    const worst=k=>Math.max(...methodRows(mesh).map(r=>r.metrics[k]));
    $("selected-metrics").innerHTML=[["Max original spatial RMS",fmt(worst("spatial_rms"))],["Max separate BC defect",fmt(worst("boundary_max"))],["Max integrated balance",fmt(worst("conservation_max"))],["Physical boundary modes",`${meta.inlet_regime} → ${meta.outlet_regime}`]].map(([k,v])=>`<div><small>${escape(k)}</small><strong>${escape(v)}</strong></div>`).join("");
    $("convergence-context").textContent=`${names[state.caseId]??state.caseId}; N = ${mesh.cells}; constant and MUSCL.`;
    const work=[fieldFigure("field-plot",mesh,DATA.fields,"fields"),convergence(mesh),equationConvergence(mesh),cflHistory(mesh),integralMonitors(mesh),conservation(mesh),refinement()];
    if(options.geometry!==false)work.push(renderChannel());
    await Promise.all(work);
  }
  function accuracyDetail(c,r) {
    if(c.is_shocked)return `Shock position: ${fmt(r.errors.shock_error_cells)} cells; required <= 2 cells. Mean |delta Cp| ${fmt(r.errors.Cp_mean_absolute)}, maximum ${fmt(r.errors.Cp_max)} (supplementary; no shock Cp limit).`;
    const q=r.errors.conservative_L1;
    return `Mean |delta Cp| ${fmt(r.errors.Cp_mean_absolute)}; required <= 0.01 over the domain (${(r.errors.Cp_mean_absolute/0.01).toFixed(2)}x limit). Supplementary maximum ${fmt(r.errors.Cp_max)}. Scaled L1 [mass, meridional momentum, angular momentum, energy]: [${q.map(v=>fmt(v)).join(", ")}]; each required < 1.000e-3. Worst component ${fmt(Math.max(...q))} (${(Math.max(...q)/1e-3).toFixed(2)}x limit).`;
  }
  function fillAccuracy() {
    $("accuracy-table").querySelector("tbody").innerHTML=DATA.cases.map(c=>{
      const mesh=c.meshes.find(m=>m.cells===400);
      return methods.map((method,i)=>{
        const r=mesh.methods[method];
        return `<tr><td><button class="case-button" data-case="${escape(c.id)}">${escape(names[c.id])}</button><span class="subtext">${i===0?"Constant":"Linear (MUSCL)"} reconstruction</span></td><td>${escape(accuracyDetail(c,r))}</td><td>${badge(r.root_pass,r.root_pass?"Converged":"Convergence target missed")} ${badge(r.accuracy_pass,r.accuracy_pass?"Accuracy target met":"Accuracy target missed")}</td></tr>`;
      }).join("");
    }).join("");
    $("accuracy-table").querySelectorAll("button[data-case]").forEach(b=>b.addEventListener("click",async()=>{
      state.caseId=b.dataset.case;state.cells=400;$("case-select").value=state.caseId;
      revealSection($("fields"));await renderCase();$("fields").scrollIntoView({behavior:reducedMotion()?"instant":"smooth"});
    }));
  }
  async function conservation(mesh) {
    const quantities=["Mass","Meridional momentum","Angular momentum","Energy"];
    const colors=[palette().colors[0],"#b12a90","#e16462","#fca636"];
    const traces=[],missing=[],histories={};
    methods.forEach((method,i)=>{
      const rows=historyRows(mesh.methods[method].history);
      const complete=rows.length>0&&rows.every(r=>Array.isArray(r.integrated_balance_scaled)&&r.integrated_balance_scaled.length===4&&r.integrated_balance_scaled.every(Number.isFinite));
      if(complete)histories[method]=rows;else missing.push(i===0?"constant":"linear (MUSCL)");
    });
    quantities.forEach((name,j)=>methods.forEach((method,i)=>{
      const rows=histories[method];if(!rows)return;
      traces.push({type:"scatter",mode:"lines",x:rows.map(r=>r.step),
        y:rows.map(r=>Math.abs(r.integrated_balance_scaled[j])),
        name:`${name} / ${i===0?"constant":"MUSCL"}`,meta:{method,quantity:j},
        line:{color:colors[j],width:2,dash:i===0?"dash":"solid"},
        hovertemplate:"Accepted update %{x}<br>|Scaled balance defect| %{y:.6e}<extra>%{fullData.name}</extra>"});
    }));
    $("conservation-note").textContent=missing.length?`Per-equation histories were not recorded for ${missing.join(" and ")} in these saved runs. No curves are inferred from the scalar maximum or final state.`:"Absolute boundary-flux difference minus integrated sources, divided by fixed nonzero flux scales. Momentum includes geometric, centrifugal and any prescribed wall sources. Only the selected mesh's solve is plotted; coarse-grid initialization stages are separate recorded solves. Exact zeros are omitted by the logarithmic axis; all raw values remain downloadable.";
    await plot("conservation-plot",traces,frame({xaxis:axis("Accepted updates (initial state = 0)"),yaxis:axis("Absolute global defect / fixed flux scale",{type:"log"}),
      margin:{l:79,r:24,t:126,b:58},uirevision:`balance:${state.caseId}:${state.cells}`,
      legend:{orientation:"h",entrywidthmode:"fraction",entrywidth:.5,x:0,y:1.04,xanchor:"left",yanchor:"bottom",font:{size:10}},
      annotations:traces.length?[]:[{xref:"paper",yref:"paper",x:.5,y:.5,text:"Per-equation histories not recorded",showarrow:false,font:{color:palette().muted,size:14}}]}),`conservation_${state.caseId}_${state.cells}`);
  }
  function shockRows() {
    const sources=DATA.shocks.sources ?? [];
    let peak=-Infinity;
    $("shock-table").querySelector("tbody").innerHTML=sources.filter(s=>s.cells===400).flatMap(s=>s.methods.map(run=>({s,run}))).map(({s,run})=>{
      const upstream=run.upstream;
      const downstream=run.downstream;
      const quantities=run.planar_jump;
      const local=run.excluded_shock_region;
      // Normalized by the audited analytical shock-window maximum, never global pmax.
      const excess=local.pressure_peak_over_reference_window_max;
      if(Number.isFinite(excess)) peak=Math.max(peak,excess);
      const method=run.method;
      return `<tr><td><button class="case-button" data-shock="${escape(s.case)}">${escape(names[s.case])}</button><span class="subtext">${method==="constant" ? "Constant" : "MUSCL"}</span></td><td class="numeric">${fmt(Math.abs(run.location_error_local_cells))}</td><td class="numeric">${pretty(upstream.M_m)}</td><td class="numeric">${pretty(downstream.M_m)}</td><td class="numeric">${pretty(quantities.entropy_rise)}</td><td class="numeric ${excess>0.01 ? "danger" : ""}">${Number.isFinite(excess) ? `${excess>=0?"+":""}${(100*excess).toFixed(3)}%` : "n/a"}</td></tr>`;
    }).join("");
    if(Number.isFinite(peak)) $("shock-peak-heading").textContent=`${(100*peak).toFixed(3)}% local pressure excess`;
    $("shock-table").querySelectorAll("button[data-shock]").forEach(b=>b.addEventListener("click",async()=>{
      state.caseId=b.dataset.shock;state.cells=400;$("case-select").value=state.caseId;revealSection($("fields"));await renderCase();$("fields").scrollIntoView();
    }));
  }
  function auditContent() {
    const all=DATA.cases.flatMap(c=>c.meshes.flatMap(m=>methods.map(k=>m.methods[k])));
    const max=k=>Math.max(...all.map(r=>r.metrics[k]));
    const relative=all.flatMap(run=>(run.metrics.equation_relative??[]).filter((_,j)=>run.metrics.relative_active[j]));
    const absolute=all.flatMap(run=>run.metrics.equation_norm??[run.metrics.spatial_rms]);
    $("audit-cards").innerHTML=[
      ["Relative equation reduction",relative.length?Math.max(...relative):NaN,1e-9,"Worst active final Rk across all solves; failures remain visible"],
      ["Absolute equation safeguard",Math.max(...absolute),1e-8,"Largest fixed-scale equation norm across all solves"],
      ["Independent boundary rows",max("boundary_max"),1e-8,"Separate maximum; eight algebraic rows have zero storage"],
      ["Assembled global balance",max("conservation_max"),1e-8,"Boundary flux difference minus integrated sources"]
    ].map(([name,value,target,note])=>`<div class="panel audit-card">${badge(value<=target,value<=target?"Target met":"Target missed")}<h3>${escape(name)}</h3><p class="mono">${fmt(value)} <span class="muted">&le; ${fmt(target,0)}</span></p><small>${escape(note)}</small></div>`).join("");
    const wall=DATA.wall_sources;
    $("wall-source-evidence").innerHTML=wall
      ?`${badge(wall.passed,"Source checks passed")} <p>Independent homogeneous shear, heating and combined-source checks: ${Object.keys(wall.homogeneous??{}).length}. Full steady solver checks with independently assembled wall-force, torque and heat budgets: ${Object.keys(wall.steady??{}).length}. These verify prescribed inputs to the placeholder functions; they do not supply empirical friction or heat-transfer correlations.</p>`
      :"Wall-source verification data were not present when this report was built.";
    $("nonuniform-table").querySelector("tbody").innerHTML=DATA.nonuniform.flatMap(mesh=>methods.map((m,i)=>{const r=mesh.methods[m];return `<tr><td>${i===0?"Constant":"MUSCL"}</td><td>${badge(r.root_pass)}</td><td class="numeric">${fmt(r.errors.Cp_mean_absolute)}</td><td class="numeric">${fmt(Math.max(...r.errors.conservative_L1))}</td><td>${badge(r.accuracy_pass)}</td></tr>`;})).join("");
    const failed=DATA.diagnostics.narrow_width.failed;
    const rus=DATA.diagnostics.rusanov_diagnostic;
    $("robustness-details").innerHTML=`<p>The inherited entropy-width factor 10⁻⁶ failed ${failed.length} recorded cold starts. The selected factor is 0.1; both reconstructions use the same value and the original tolerance. Wider-width runs and a Rusanov diagnostic were saved separately.</p><ul>${failed.map(r=>`<li>${escape(names[r.case])}, N = ${r.cells}, ${r.method==="constant"?"constant":"MUSCL"}: ${escape(r.message)}.</li>`).join("")}</ul><p>The separate Rusanov campaign passes ${rus.root_passes}/${rus.entries.length} root checks and ${rus.accuracy_passes}/${rus.entries.length} shock-location checks. It is an explicit diagnostic, not a hidden fallback in the characteristic results.</p><p>Independent dense/colored Jacobian and dense/banded linear checks at the failed narrow-width state exclude a structural assembly error in that diagnosis. Severe local nonlinear curvature stalls the admissible Armijo trials. All negative evidence remains in the downloaded payload.</p>`;
    const p=DATA.provenance,versions=DATA.cases[0].meshes[0].metadata.versions ?? {};
    const date=new Date(p.generated_at_utc);
    $("generated-date").textContent=date.toLocaleDateString("en-GB",{day:"numeric",month:"long",year:"numeric",timeZone:"Europe/Paris"});
    const settings=DATA.cases[0].meshes[0].settings;
    const cap=DATA.cases[0].meshes[0].methods.constant.effective_options.newton_iterations;
    const rows=[
      ["Report snapshot",date.toLocaleString("en-GB",{timeZone:"Europe/Paris",timeZoneName:"short"})],
      ["Runtime libraries",Object.entries(versions).map(([k,v])=>`${k} ${v}`).join(" · ")],
      ["Primary evidence",`${DATA.summary.primary_files} case/mesh files · ${DATA.summary.method_entries} numerical solutions · full plotted samples`],
      ["Independent audit",`${DATA.audits ? (DATA.audits.audit_pass ? "Integrity passed" : "Integrity failed") : "Audit not present"} · source unchanged during audit: ${DATA.audits?.source_unchanged_during_audit ?? "n/a"}`],
      ["Steady controller", `Initial CFL ${settings.cfl}; growth ${settings.cfl_growth??settings.growth}; shrink ${settings.cfl_shrink??settings.shrink}; Newton cap ${cap}; fixed accepted-step growth`],
      ["HTML data loading",`${p.input_files.length} immutable inputs checked · no solver or test rerun`],
      ["Embedded data SHA-256",window.REPORT_BUILD?.payload_sha256 ?? "See downloaded provenance"],
      ["Offline presentation",`Plotly.js 2.35.2 · embedded result data · no external fonts, CDN or telemetry`]
    ];
    $("provenance-table").querySelector("tbody").innerHTML=rows.map(([k,v])=>`<tr><td>${escape(k)}</td><td><code>${escape(v)}</code></td></tr>`).join("");
    $("source-manifest").textContent=JSON.stringify({source_sha256:p.current_source_sha256,input_files:p.input_files,nonfinite:p.nonfinite_source_values},null,2);
  }
  function reducedMotion() { return matchMedia("(prefers-reduced-motion: reduce)").matches; }
  function revealSection(target) {
    for(let node=target?.parentElement;node;node=node.parentElement){
      if(node.tagName==="DETAILS")node.open=true;
    }
    target?.querySelectorAll?.(":scope > details.result-section").forEach(box=>{box.open=true;});
  }
  function initResultSections() {
    const boxes=[...document.querySelectorAll("#report-page-results details.result-section")];
    $("results-expand")?.addEventListener("click",()=>boxes.forEach(box=>{box.open=true;}));
    $("results-collapse")?.addEventListener("click",()=>boxes.forEach(box=>{box.open=false;}));
    document.querySelectorAll("#report-page-results details").forEach(box=>box.addEventListener("toggle",()=>{
      if(box.open)setTimeout(()=>{
        if(box.id==="box-fields"&&DATA)renderCase();
        else box.querySelectorAll(".js-plotly-plot").forEach(node=>Plotly.Plots?.resize(node));
      },0);
    }));
    let printState=[];
    window.addEventListener("beforeprint",()=>{
      printState=[...document.querySelectorAll("#report-page-results details")].map(box=>[box,box.open]);
      printState.forEach(([box])=>{box.open=true;});
    });
    window.addEventListener("afterprint",()=>printState.forEach(([box,open])=>{box.open=open;}));
  }
  async function setTheme(theme,persist=true) {
    document.documentElement.dataset.theme=theme;
    document.querySelectorAll("[data-theme-choice]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.themeChoice===theme)));
    if(persist) {try{localStorage.setItem(themeKey,theme);}catch{}}
    if(DATA) await Promise.all([renderCase()]);
  }
  function downloadData() {
    const payload={...DATA,report_build:window.REPORT_BUILD};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload)],{type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="euler_annular_report_data.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function exportFigure(id,format) {
    const entry=plots.get(id); if(!entry)return;
    await Plotly.downloadImage($(id),{format,filename:entry.filename,width:1300,height:id==="field-plot"?740:450,scale:format==="png"?2:1});
  }
  async function navigate(hash=window.location.hash,scroll=true) {
    const key=decodeURIComponent((hash||"#overview").replace(/^#/,""));
    const target=$(key)||$("overview");
    revealSection(target);
    const page=target.closest("[data-report-page]")?.dataset.reportPage||"results";
    document.querySelectorAll("[data-report-page]").forEach(el=>{el.hidden=el.dataset.reportPage!==page;});
    document.querySelectorAll("[data-page-nav]").forEach(el=>{el.hidden=el.dataset.pageNav!==page;});
    document.querySelectorAll("[data-page-choice]").forEach(el=>{
      if(el.dataset.pageChoice===page)el.setAttribute("aria-current","page");else el.removeAttribute("aria-current");
    });
    document.body.dataset.page=page;
    $("sidebar-page-label").textContent={results:"Results",theory:"Theory",implementation:"Implementation",notes:"Notes & findings"}[page];
    if(state)state.page=page;
    if(page==="results"&&DATA)await renderCase();
    if(page!=="results"){
      if(window.MathJax?.startup?.promise)await window.MathJax.startup.promise;
      if(window.MathJax?.typesetPromise)await window.MathJax.typesetPromise([target.closest("[data-report-page]")]);
    }
    if(scroll)target.scrollIntoView({behavior:"instant",block:"start"});
  }
  function downloadEvidence() {
    const record=JSON.parse($("embedded-downloads").textContent).controller_evidence;
    const bytes=Uint8Array.from(atob(record.base64),c=>c.charCodeAt(0));
    const url=URL.createObjectURL(new Blob([bytes],{type:record.mime}));
    const a=document.createElement("a");a.href=url;a.download=record.filename;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function navObserver() {
    const links=[...document.querySelectorAll(".sidebar nav a[href^='#']")];
    const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){links.forEach(a=>a.classList.toggle("active",a.hash===`#${entry.target.id}`));}}),{rootMargin:"-12% 0px -70% 0px"});
    links.forEach(a=>{const target=document.getElementById(decodeURIComponent(a.hash.slice(1)));if(target)observer.observe(target);});
  }
  async function init() {
    const initialTheme=document.documentElement.dataset.theme;
    await setTheme(initialTheme,false);
    document.querySelectorAll("[data-theme-choice]").forEach(b=>b.addEventListener("click",()=>setTheme(b.dataset.themeChoice)));
    $("print-report")?.addEventListener("click",()=>window.print());
    navObserver();
    if(!$("report-data"))return;
    initResultSections();
    DATA=JSON.parse($("report-data").textContent);
    state={caseId:DATA.cases[0].id,cells:400,channelCase:"current"};
    window.ANNULAR_REPORT={data:DATA,state,plots,renderCase,renderChannel,setTheme,navigate,downloadEvidence};
    $("case-select").innerHTML=DATA.cases.map(c=>`<option value="${escape(c.id)}">${escape(names[c.id]??c.label)}</option>`).join("");
    $("mesh-select").innerHTML=DATA.meshes.map(n=>`<button data-cells="${n}" aria-pressed="${n===400}">${n}</button>`).join("");
    $("case-select").addEventListener("change",async e=>{state.caseId=e.target.value;await renderCase();});
    document.querySelectorAll("#mesh-select button").forEach(b=>b.addEventListener("click",async()=>{state.cells=Number(b.dataset.cells);await renderCase();}));
    $("export-fields-svg").addEventListener("click",()=>exportFigure("field-plot","svg"));
    $("export-fields-png").addEventListener("click",()=>exportFigure("field-plot","png"));
    $("download-data").addEventListener("click",downloadData);
    $("channel-case").addEventListener("change",async e=>{state.channelCase=e.target.value;await renderChannel();});
    $("export-channel-png").addEventListener("click",()=>exportFigure("channel-annular","png"));
    $("download-controller-evidence").addEventListener("click",downloadEvidence);
    window.addEventListener("hashchange",()=>navigate());
    fillAccuracy();shockRows();auditContent();
    await Promise.all([renderCase()]);
    await navigate(window.location.hash,Boolean(window.location.hash));
    document.body.dataset.reportReady="true";
    window.dispatchEvent(new CustomEvent("annular-report-ready"));
    let timeout;
    window.addEventListener("resize",()=>{clearTimeout(timeout);timeout=setTimeout(async()=>{await Promise.all([renderCase()]);},180);});
  }
  document.addEventListener("DOMContentLoaded",()=>init().catch(error=>{
    console.error(error);document.body.dataset.reportError=error.message;
    const el=document.createElement("div");el.className="fatal";el.textContent=`The report could not finish rendering: ${error.message}. Saved data is retained in the embedded report payload.`;document.querySelector("main").prepend(el);
  }));
})();
