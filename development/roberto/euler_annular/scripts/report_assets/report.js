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
      marker:{color:"white",symbol:"circle",size:9,line:{color:p.colors[index],width:1.8}},
      connectgaps:false,meta:{method:index===0 ? "analytical" : methods[index-1], marker_stride:index===0 ? null : state.stride},
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
      // Full numerical lines first, dashed analytical reference above, markers last.
      [1,2,0].forEach(k=>traces.push(scientificTrace(k,mesh.m,ys[k],{xaxis:`x${suffix}`,yaxis:`y${suffix}`,showlegend:i===0})));
      [1,2].forEach(k=>{
        const offset=k===1 ? 0 : Math.floor(state.stride/2);
        const indices=mesh.m.map((_,j)=>j).filter(j=>j%state.stride===offset);
        traces.push(scientificTrace(k,indices.map(j=>mesh.m[j]),indices.map(j=>ys[k][j]),{
          mode:"markers",zorder:3,xaxis:`x${suffix}`,yaxis:`y${suffix}`,showlegend:false,
          meta:{method:methods[k-1],marker_stride:state.stride,marker_offset:offset}}));
      });
    });
    await plot(id,traces,layout,`${kind}_${state.caseId}_${mesh.cells}`);
  }
  function currentCase() { return DATA.cases.find(c=>c.id===state.caseId); }
  function currentMesh() { return currentCase().meshes.find(m=>m.cells===state.cells); }
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
        path.arrows.forEach(a=>{
          // A purely axial chord has no directional arrow in this projection.
          if(Math.hypot(a.head[1]-a.tail[1],a.head[2]-a.tail[2])<=1e-12*outer)return;
          annotations.push({x:a.head[1],y:a.head[2],ax:a.tail[1],ay:a.tail[2],
            xref:"x",yref:"y",axref:"x",ayref:"y",showarrow:true,text:"",arrowcolor:blue,
            arrowwidth:1.6,arrowhead:2,arrowsize:1.3});
        });
      });
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
        path.arrows.forEach(a=>annotations.push({x:a.head[0],y:a.head[1],ax:a.tail[0],ay:a.tail[1],
          xref:"x",yref:"y",axref:"x",ayref:"y",showarrow:true,text:"",arrowcolor:blue,arrowwidth:1.6,arrowhead:2,arrowsize:1.3}));
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
    $("channel-caption").textContent="Left: equal-scale meridional section of the actual NURBS walls. Center: frontal projection along the axial x direction, using the same (y,z) paths as the 3D view and the projected passage envelope. Right: the revolved annular channel. All paths and direction arrows share one color. Arrows follow increasing meridional distance, including radial inflow; purely axial motion has no frontal direction arrow. Swirl paths use saved finest-grid MUSCL velocities and illustrate quasi-1D flow direction without resolving a cross-gap velocity profile.";
  }
  function methodRows(mesh) { return methods.map(m=>mesh.methods[m]); }
  function historyRows(history, accepted=true) {
    return (accepted ? history.accepted : history.rejected).map(row=>Object.fromEntries(history.columns.map((key,i)=>[key,row[i]])));
  }
  function selectedDiagnostics(mesh) {
    const isShock=currentCase().is_shocked;
    $("selected-case-type").textContent=isShock ? "Shock location criterion" : "Original smooth-flow accuracy criteria";
    const rows=methods.map((m,i)=>{
      const r=mesh.methods[m];
      return `<tr><td>${i===0 ? "Constant" : "MUSCL"}</td><td class="numeric">${fmt(r.errors.Cp_max)}</td><td class="numeric">${fmt(Math.max(...r.errors.conservative_L1))}</td><td>${badge(r.accuracy_pass)}</td></tr>`;
    }).join("");
    const settings=mesh.settings;
    const controllerText=settings.controller==="cfl_recommended"
      ? `Fixed-growth CFL controller: initial ${settings.cfl}, accepted-step multiplier ${settings.growth}, failure multiplier ${settings.shrink}; max CFL ${settings.cfl_max}. Maximum ${settings.newton_iterations} Newton corrections per timestep; both inner tolerances remain 1e-10. Initial dt: ${fmt(mesh.methods.constant.controller_audit?.initial_dt)} / ${fmt(mesh.methods.muscl_van_leer.controller_audit?.initial_dt)} s (constant / linear).`
      : "Historical production work-based CFL controller; see saved settings in the data download.";
    const work=methodRows(mesh).map(r=>{
      const rows=[...historyRows(r.history),...historyRows(r.history,false)];
      return {newton:rows.reduce((n,row)=>n+row.inner_iterations,0),backtracks:rows.reduce((n,row)=>n+row.rejected_newton_trials,0)};
    });
    const rejected=methodRows(mesh).reduce((n,r)=>n+r.history.counts.rejected,0);
    const counts=methodRows(mesh).map(r=>r.history.counts.accepted);
    $("case-diagnostics").innerHTML=`<p style="font-size:12px;color:var(--muted)">${escape(explanations[state.caseId])}</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>Max Cp error</th><th>Max q L₁</th><th>Criterion</th></tr></thead><tbody>${rows}</tbody></table></div><p class="muted" style="font-size:11px">${escape(controllerText)}</p><p class="muted" style="font-size:11px">Total Newton corrections: ${work[0].newton.toLocaleString()} / ${work[1].newton.toLocaleString()}; line-search backtracks: ${work[0].backtracks.toLocaleString()} / ${work[1].backtracks.toLocaleString()} (constant / linear, including rejected attempts).</p><p class="muted" style="font-size:11px;margin-top:15px">Accepted updates: constant ${counts[0].toLocaleString()}, MUSCL ${counts[1].toLocaleString()}. Rejected attempts: ${rejected.toLocaleString()} total.</p><p class="muted" style="font-size:11px">${isShock ? "The criterion here checks shock location only. Cp and conservative errors remain shown in full, including the discontinuity and any overshoot." : "Both Cp and all conservative L₁ components must satisfy the original 10⁻³ target. A converged residual does not imply that this accuracy target is met."}</p>`;
  }
  async function convergence(mesh) {
    const traces=methods.map((m,i)=>{
      const rows=historyRows(mesh.methods[m].history);
      const y=rows.map(r=>Math.max(r.spatial_rms,r.boundary_max,r.conservation_max));
      const t=scientificTrace(i+1,rows.map(r=>r.step),y);
      t.mode="lines+markers";
      const offset=i===0 ? 0 : Math.floor(state.stride/2);
      t.marker.size=y.map((_,j)=>j%state.stride===offset ? 9 : 0);

      t.hovertemplate="Accepted update %{x}<br>Original defect %{y:.5e}<extra>%{fullData.name}</extra>";
      return t;
    });
    const p=palette(),tol=mesh.settings.tolerance;
    await plot("convergence-plot",traces,frame({xaxis:axis("Accepted implicit updates"),
      yaxis:axis("Max(original spatial, BC, balance)",{type:"log"}),
      margin:{l:79,r:22,t:49,b:58},uirevision:`convergence:${state.caseId}:${state.cells}`,
      shapes:[{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:tol,y1:tol,line:{color:p.muted,width:1,dash:"dot"}}],
      annotations:[{xref:"paper",x:.99,yref:"y",y:Math.log10(tol),text:"10⁻⁸ target",showarrow:false,xanchor:"right",yshift:11,font:{color:p.muted,size:10}}]}),`convergence_${state.caseId}_${state.cells}`);
  }
  async function refinement() {
    const c=currentCase(),shocked=c.is_shocked;
    $("refinement-case").textContent=names[c.id];
    const x=c.meshes.map(m=>m.cells);
    const graph=async(id,metric,title)=>{
      const traces=methods.map((m,i)=>{
        const y=c.meshes.map(mesh=>metric(mesh.methods[m].errors));
        return scientificTrace(i+1,x,y,{marker:{color:"white",symbol:"circle",size:9,line:{color:palette().colors[i+1],width:1.8}},mode:"lines+markers",
          hovertemplate:"N = %{x}<br>Error %{y:.6e}<extra>%{fullData.name}</extra>"});
      });
      const layout=frame({xaxis:axis("Physical cells N",{type:"log",tickmode:"array",tickvals:x,ticktext:x.map(String)}),
        yaxis:axis(title,{type:"log"}),margin:{l:72,r:24,t:47,b:57},uirevision:`${id}:${c.id}`,
        shapes:shocked ? [] : [{type:"line",xref:"paper",x0:0,x1:1,yref:"y",y0:1e-3,y1:1e-3,line:{color:palette().muted,width:1,dash:"dot"}}]});
      await plot(id,traces,layout,`${id}_${c.id}`);
    };
    await graph("refinement-plot",e=>e.Cp_max,"Maximum absolute Cp error");
    await graph("conservative-plot",e=>Math.max(...e.conservative_L1),"Max conservative L₁ error");
    const coarse=c.meshes.find(m=>m.cells===200),fine=c.meshes.find(m=>m.cells===400);
    const orders=methods.map(m=>Math.log2(coarse.methods[m].errors.Cp_max/fine.methods[m].errors.Cp_max));
    $("refinement-order").textContent=`Observed Cp order, N = 200 → 400: constant ${orders[0].toFixed(3)}, MUSCL ${orders[1].toFixed(3)}. ${shocked ? "A maximum norm across a discontinuity need not decrease; the original shock criterion uses location." : c.id==="sonic_exit" ? "The sonic-exit limit is reported separately from strictly smooth branches." : "This is a two-grid observation, not an assumed formal order."}`;
  }
  async function renderCase(options={}) {
    const mesh=currentMesh(),meta=mesh.metadata;
    document.querySelectorAll("#mesh-select button").forEach(b=>b.setAttribute("aria-pressed",String(Number(b.dataset.cells)===state.cells)));
    $("case-description").innerHTML=`<strong>${escape(names[state.caseId])}</strong> · backpressure ${pretty(meta.backpressure/1000)} kPa · α<sub>in</sub> ${pretty(meta.alpha_in*180/Math.PI)}°`;
    $("case-status").innerHTML=`${badge(methodRows(mesh).every(r=>r.root_pass),"Both roots met")} ${badge(mesh.methods.constant.accuracy_pass,"Constant accuracy")} ${badge(mesh.methods.muscl_van_leer.accuracy_pass,"MUSCL accuracy")}`;
    const worst=k=>Math.max(...methodRows(mesh).map(r=>r.metrics[k]));
    $("selected-metrics").innerHTML=[["Max original spatial RMS",fmt(worst("spatial_rms"))],["Max separate BC defect",fmt(worst("boundary_max"))],["Max integrated balance",fmt(worst("conservation_max"))],["Physical boundary modes",`${meta.inlet_regime} → ${meta.outlet_regime}`]].map(([k,v])=>`<div><small>${escape(k)}</small><strong>${escape(v)}</strong></div>`).join("");
    selectedDiagnostics(mesh);
    const work=[fieldFigure("field-plot",mesh,DATA.fields,"fields"),convergence(mesh),conservation(mesh),refinement()];
    if(options.geometry!==false)work.push(renderChannel());
    await Promise.all(work);
  }
  function accuracyDetail(c,r) {
    if(c.is_shocked)return `Shock position: ${fmt(r.errors.shock_error_cells)} cells; required <= 2 cells. Cp error ${fmt(r.errors.Cp_max)} (no shock Cp limit).`;
    const q=r.errors.conservative_L1;
    return `Max |delta Cp| ${fmt(r.errors.Cp_max)}; required < 1.000e-3. Scaled L1 [mass, meridional momentum, angular momentum, energy]: [${q.map(v=>fmt(v)).join(", ")}]; each required < 1.000e-3. Worst component ${fmt(Math.max(...q))} (${(Math.max(...q)/1e-3).toFixed(2)}x limit); Cp ${(r.errors.Cp_max/1e-3).toFixed(2)}x limit.`;
  }
  function fillAccuracy() {
    $("accuracy-table").querySelector("tbody").innerHTML=DATA.cases.map(c=>{
      const mesh=c.meshes.find(m=>m.cells===400);
      return methods.map((method,i)=>{
        const r=mesh.methods[method];
        return `<tr><td><button class="case-button" data-case="${escape(c.id)}">${escape(names[c.id])}</button><span class="subtext">${i===0?"Constant":"Linear (MUSCL)"} reconstruction</span></td><td>${escape(accuracyDetail(c,r))}</td><td>${badge(r.accuracy_pass,r.accuracy_pass?"Accuracy target met":"Accuracy target missed")}</td></tr>`;
      }).join("");
    }).join("");
    $("accuracy-table").querySelectorAll("button[data-case]").forEach(b=>b.addEventListener("click",async()=>{
      state.caseId=b.dataset.case;state.cells=400;$("case-select").value=state.caseId;
      await renderCase();$("fields").scrollIntoView({behavior:reducedMotion()?"instant":"smooth"});
    }));
  }
  async function conservation(mesh) {
    const quantities=["Mass","Meridional momentum","Angular momentum","Energy"];
    const colors=[palette().colors[0],"#b12a90","#e16462","#fca636"];
    const traces=[];
    const missing=[];
    methods.forEach((method,i)=>{
      const rows=historyRows(mesh.methods[method].history);
      const complete=rows.length>0&&rows.every(r=>Array.isArray(r.integrated_balance_scaled)&&r.integrated_balance_scaled.length===4&&r.integrated_balance_scaled.every(Number.isFinite));
      if(!complete){missing.push(i===0?"constant":"linear (MUSCL)");return;}
      quantities.forEach((name,j)=>traces.push({type:"scatter",mode:"lines",x:rows.map(r=>r.step),
        y:rows.map(r=>Math.abs(r.integrated_balance_scaled[j])),
        name:`${name} / ${i===0?"constant":"linear"}`,line:{color:colors[j],width:2,dash:i===0?"dash":"solid"},
        hovertemplate:"Accepted update %{x}<br>|Scaled balance defect| %{y:.6e}<extra>%{fullData.name}</extra>"}));
    });
    $("conservation-note").textContent=missing.length?`Per-equation histories were not recorded for ${missing.join(" and ")} in these saved runs. No curves are inferred from the scalar maximum or final state. Rerun with the prepared conservation recorder to populate this figure.`:"Absolute boundary-flux difference minus integrated sources, divided by fixed nonzero flux scales. Momentum includes geometric and centrifugal sources. Exact zeros are omitted by the logarithmic axis; all raw values remain downloadable.";
    await plot("conservation-plot",traces,frame({xaxis:axis("Accepted implicit updates"),yaxis:axis("Absolute scaled balance defect",{type:"log"}),
      margin:{l:79,r:24,t:95,b:58},uirevision:`balance:${state.caseId}:${state.cells}`,
      annotations:traces.length?[]:[{xref:"paper",yref:"paper",x:.5,y:.5,text:"Per-equation histories pending a new simulation run",showarrow:false,font:{color:palette().muted,size:14}}]}),`conservation_${state.caseId}_${state.cells}`);
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
      state.caseId=b.dataset.shock;state.cells=400;$("case-select").value=state.caseId;await renderCase();$("fields").scrollIntoView();
    }));
  }
  function auditContent() {
    const all=DATA.cases.flatMap(c=>c.meshes.flatMap(m=>methods.map(k=>m.methods[k])));
    const max=k=>Math.max(...all.map(r=>r.metrics[k]));
    $("audit-cards").innerHTML=[
      ["Original spatial equations",max("spatial_rms"),"Fixed-scale spatial RMS across all 72 solutions"],
      ["Independent boundary rows",max("boundary_max"),"Separate maximum; eight algebraic rows have zero storage"],
      ["Assembled global balance",max("conservation_max"),"Boundary flux difference minus integrated sources"]
    ].map(([name,value,note])=>`<div class="panel audit-card">${badge(value<1e-8,"Target met")}<h3>${escape(name)}</h3><p class="mono">${fmt(value)} <span class="muted">&lt; 10⁻⁸</span></p><small>${escape(note)}</small></div>`).join("");
    $("nonuniform-table").querySelector("tbody").innerHTML=DATA.nonuniform.flatMap(mesh=>methods.map((m,i)=>{const r=mesh.methods[m];return `<tr><td>${i===0?"Constant":"MUSCL"}</td><td>${badge(r.root_pass)}</td><td class="numeric">${fmt(r.errors.Cp_max)}</td><td class="numeric">${fmt(Math.max(...r.errors.conservative_L1))}</td><td>${badge(r.accuracy_pass)}</td></tr>`;})).join("");
    const failed=DATA.diagnostics.narrow_width.failed;
    const rus=DATA.diagnostics.rusanov_diagnostic;
    $("robustness-details").innerHTML=`<p>The inherited entropy-width factor 10⁻⁶ failed ${failed.length} recorded cold starts. The selected factor is 0.1; both reconstructions use the same value and the original tolerance. Wider-width runs and a Rusanov diagnostic were saved separately.</p><ul>${failed.map(r=>`<li>${escape(names[r.case])}, N = ${r.cells}, ${r.method==="constant"?"constant":"MUSCL"}: ${escape(r.message)}.</li>`).join("")}</ul><p>The separate Rusanov campaign passes ${rus.root_passes}/${rus.entries.length} root checks and ${rus.accuracy_passes}/${rus.entries.length} shock-location checks. It is an explicit diagnostic, not a hidden fallback in the characteristic results.</p><p>Independent dense/colored Jacobian and dense/banded linear checks at the failed narrow-width state exclude a structural assembly error in that diagnosis. Severe local nonlinear curvature stalls the admissible Armijo trials. All negative evidence remains in the downloaded payload.</p>`;
    const p=DATA.provenance,versions=DATA.cases[0].meshes[0].metadata.versions ?? {};
    const date=new Date(p.generated_at_utc);
    $("generated-date").textContent=date.toLocaleDateString("en-GB",{day:"numeric",month:"long",year:"numeric",timeZone:"Europe/Paris"});
    const rows=[
      ["Report snapshot",date.toLocaleString("en-GB",{timeZone:"Europe/Paris",timeZoneName:"short"})],
      ["Runtime libraries",Object.entries(versions).map(([k,v])=>`${k} ${v}`).join(" · ")],
      ["Primary evidence",`${DATA.summary.primary_files} case/mesh files · ${DATA.summary.method_entries} numerical solutions · full plotted samples`],
      ["Independent audit",`${DATA.audits.audit_pass ? "Integrity passed" : "Integrity failed"} · source unchanged during audit: ${DATA.audits.source_unchanged_during_audit}`],
      ["Steady controller", `Initial CFL ${DATA.cases[0].meshes[0].settings.cfl}; growth ${DATA.cases[0].meshes[0].settings.growth}; shrink ${DATA.cases[0].meshes[0].settings.shrink}; Newton cap ${DATA.cases[0].meshes[0].settings.newton_iterations}; fixed accepted-step growth`],
      ["HTML data loading",`${p.input_files.length} immutable inputs checked · no solver or test rerun`],
      ["Embedded data SHA-256",window.REPORT_BUILD?.payload_sha256 ?? "See downloaded provenance"],
      ["Offline presentation",`Plotly.js 2.35.2 · embedded result data · no external fonts, CDN or telemetry`]
    ];
    $("provenance-table").querySelector("tbody").innerHTML=rows.map(([k,v])=>`<tr><td>${escape(k)}</td><td><code>${escape(v)}</code></td></tr>`).join("");
    $("source-manifest").textContent=JSON.stringify({source_sha256:p.current_source_sha256,input_files:p.input_files,nonfinite:p.nonfinite_source_values},null,2);
  }
  function reducedMotion() { return matchMedia("(prefers-reduced-motion: reduce)").matches; }
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
    DATA=JSON.parse($("report-data").textContent);
    state={caseId:DATA.cases[0].id,cells:400,stride:11,markerSkip:10,channelCase:"current"};
    window.ANNULAR_REPORT={data:DATA,state,plots,renderCase,renderChannel,setTheme,navigate,downloadEvidence};
    $("case-select").innerHTML=DATA.cases.map(c=>`<option value="${escape(c.id)}">${escape(names[c.id]??c.label)}</option>`).join("");
    $("mesh-select").innerHTML=DATA.meshes.map(n=>`<button data-cells="${n}" aria-pressed="${n===400}">${n}</button>`).join("");
    $("case-select").addEventListener("change",async e=>{state.caseId=e.target.value;await renderCase();});
    document.querySelectorAll("#mesh-select button").forEach(b=>b.addEventListener("click",async()=>{state.cells=Number(b.dataset.cells);await renderCase();}));
    $("marker-skip").addEventListener("input",async e=>{
      const raw=e.target.value,skipped=Number(raw);
      const valid=raw.trim()!==""&&Number.isSafeInteger(skipped)&&skipped>=0&&skipped<Number.MAX_SAFE_INTEGER;
      e.target.setAttribute("aria-invalid",String(!valid));
      if(!valid){$("marker-spacing-note").textContent="Enter a whole number ≥ 0";return;}
      state.markerSkip=skipped;state.stride=skipped+1;
      $("marker-spacing-note").textContent=`Every ${state.stride} point${state.stride===1?"":"s"} · ${skipped} skipped`;
      await Promise.all([renderCase({geometry:false})]);
    });
    $("export-fields-svg").addEventListener("click",()=>exportFigure("field-plot","svg"));
    $("export-fields-png").addEventListener("click",()=>exportFigure("field-plot","png"));
    $("download-data").addEventListener("click",downloadData);
    $("channel-case").addEventListener("change",async e=>{state.channelCase=e.target.value;await renderChannel();});
    $("export-channel-png").addEventListener("click",()=>exportFigure("channel-annular","png"));
    $("download-controller-evidence").addEventListener("click",downloadEvidence);
    window.addEventListener("hashchange",()=>navigate());
    fillAccuracy();shockRows();auditContent();
    await Promise.all([renderCase()]);
    await navigate(window.location.hash,false);
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
