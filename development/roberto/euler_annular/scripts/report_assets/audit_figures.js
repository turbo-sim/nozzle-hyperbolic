/* Executable JavaScript-data audit with a minimal DOM and a Plotly recorder.
 * This checks figure definitions and UI state transitions; it is not a visual
 * browser test. It drives no desktop/browser application or external service.
 * Run through the local Node REPL with vm, passing source/html and an assert.
 */
async function auditFigureDefinitions({vm, source, html, assert, setTimeout, clearTimeout}) {
  assert(typeof setTimeout==="function" && typeof clearTimeout==="function","Pass Node timer functions explicitly to the recorder");
  const payload=JSON.parse(html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const elements=new Map(),listeners={},events={};
  class Element {
    constructor(id="") { this.id=id;this.dataset={};this.value="";this.textContent="";this._html="";this.attributes={};this.clientWidth=1100;this.handlers={};this.classList={toggle(){},add(){},remove(){}}; }
    get innerHTML(){return this._html;}
    set innerHTML(value){this._html=value;}
    setAttribute(key,value){this.attributes[key]=value;}
    removeAttribute(key){delete this.attributes[key];}
    closest(){return this.parentPage??null;}
    click(){lastClicked=this;}
    addEventListener(name,fn){this.handlers[name]=fn;}
    querySelector(){return this.body ??= new Element(this.id+"-tbody");}
    querySelectorAll(){return [];}
    scrollIntoView(){}
    prepend(){}
  }
  let lastClicked=null,lastBlob=null,mathPageCalls=0;
  const element=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
  for(const match of html.matchAll(/\bid="([^"]+)"/g))element(match[1]);
  const pageMatches=[...html.matchAll(/<section class="report-page" data-report-page="([^"]+)" id="([^"]+)"/g)];
  const pageNodes=pageMatches.map((m,i)=>{
    const page=element(m[2]);page.dataset.reportPage=m[1];
    const part=html.slice(m.index,pageMatches[i+1]?.index??html.indexOf("<footer>"));
    for(const id of part.matchAll(/\bid="([^"]+)"/g))element(id[1]).parentPage=page;
    return page;
  });
  const choiceNodes=pageNodes.map(p=>{const e=new Element();e.dataset.pageChoice=p.dataset.reportPage;return e;});
  const navNodes=pageNodes.map(p=>{const e=new Element();e.dataset.pageNav=p.dataset.reportPage;return e;});
  element("report-data").textContent=JSON.stringify(payload);
  element("embedded-downloads").textContent=html.match(/<script id="embedded-downloads" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const markers=[5,2].map(stride=>{const el=new Element();el.dataset.stride=String(stride);return el;});
  const channelViews=["annular","meridional","height"].map(view=>{const el=new Element();el.dataset.view=view;return el;});
  const meshButtons=[50,100,200,400].map(n=>{const el=new Element();el.dataset.cells=String(n);return el;});
  const themes=["light","dark"].map(theme=>{const el=new Element();el.dataset.themeChoice=theme;return el;});
  const root={dataset:{theme:"light"}};
  let observedHeading=false;
  const numericLink={hash:"#1-geometry-and-transported-quantities"};
  const document={documentElement:root,body:new Element("body"),getElementById:id=>elements.get(id)??null,
    querySelectorAll(selector){if(selector==="[data-report-page]")return pageNodes;if(selector==="[data-page-choice]")return choiceNodes;if(selector==="[data-page-nav]")return navNodes;if(selector.startsWith(".sidebar"))return [numericLink];if(selector.includes("data-theme-choice"))return themes;if(selector==="#marker-select button")return markers;if(selector==="#mesh-select button")return meshButtons;if(selector==="#channel-view button")return channelViews;return [];},
    querySelector(selector){assert(!selector?.startsWith("#1-"),"Unsafe numeric ID selector");return new Element();},createElement(){return new Element();},
    addEventListener(name,fn){listeners[name]=fn;}};
  const tokens={"--panel":"#fff","--ink":"#18314d","--muted":"#62758b","--line":"#d9e2ee","--grid":"#e6edf6","--accent":"#2462d1"};
  const context={document,console,location:{hash:"#overview"},MathJax:{startup:{promise:Promise.resolve()},async typesetPromise(){mathPageCalls++;}},atob:v=>Buffer.from(v,"base64").toString("binary"),Plotly:{async react(el,traces,layout){el.data=traces;el.layout=layout;},async downloadImage(){}},
    getComputedStyle:()=>({getPropertyValue:name=>tokens[name]??""}),matchMedia:()=>({matches:false}),
    localStorage:{getItem(){return null;},setItem(){}},IntersectionObserver:class{observe(target){if(target.id==="1-geometry-and-transported-quantities")observedHeading=true;}},
    setTimeout,clearTimeout,CustomEvent:class{constructor(type){this.type=type;}},
    Blob:class{constructor(parts,options){this.parts=parts;this.options=options;}},URL:{createObjectURL(blob){lastBlob=blob;return "blob:local";},revokeObjectURL(){}},
    addEventListener(name,fn){events[name]=fn;},dispatchEvent(){},print(){}};
  context.window=context;
  vm.createContext(context);
  new vm.Script(source,{filename:"report.js"}).runInContext(context);
  await listeners.DOMContentLoaded();
  assert(!document.body.dataset.reportError,document.body.dataset.reportError);
  assert(document.body.dataset.reportReady==="true","Initialization did not complete");
  assert(observedHeading,"Numeric theory heading was not observed safely");
  const app=context.ANNULAR_REPORT;
  assert(!elements.has("marker-skip"),"Solution marker controls must be absent");
  for(const [hash,page] of [["#1-geometry-and-transported-quantities","theory"],["#spatial-discretization","theory"],["#use-reproduce","implementation"],["#notes-controller","notes"],["#overview","results"]]){
    await app.navigate(hash,false);
    assert(app.state.page===page&&document.body.dataset.page===page,"Internal chapter routing failed");
    pageNodes.forEach(el=>assert(el.hidden===(el.dataset.reportPage!==page),"Page visibility wrong"));
    navNodes.forEach(el=>assert(el.hidden===(el.dataset.pageNav!==page),"Sidebar visibility wrong"));
    choiceNodes.forEach(el=>assert((el.attributes["aria-current"]==="page")===(el.dataset.pageChoice===page),"Page tab status wrong"));
  }
  assert(mathPageCalls===4,"Math chapters were not typeset on navigation");
  context.location.hash="#use-fluids";
  await events.hashchange();
  assert(app.state.page==="implementation","Browser hashchange event did not switch pages");
  context.location.hash="#overview";
  await events.hashchange();
  app.downloadEvidence();
  assert(lastClicked.download==="EulerAnnular_Controller_Evidence.zip","Evidence download filename");
  assert(lastBlob.parts[0][0]===80&&lastBlob.parts[0][1]===75,"Embedded download is not a ZIP");

  const exact=(a,b,message)=>assert(JSON.stringify(a)===JSON.stringify(b),message);
  let figures=0,points=0,channels=0;
  function fields(mesh) {
    const traces=element("field-plot").data;
    assert(traces.length===18,"Six panels require three complete lines each");
    ["p","rho","alpha_deg","M_m","M_theta","M"].forEach((field,j)=>{
      const ts=traces.slice(3*j,3*j+3);
      ["constant","muscl_van_leer","analytical"].forEach((method,k)=>{
        const t=ts[k];
        exact(t.x,mesh.m,"Line coordinates must remain complete");
        exact(t.y,method==="analytical"?mesh.reference[field]:mesh.methods[method].fields[field],"Changed field values");
        assert(t.mode==="lines","Solution trends must have no markers");
        assert(t.line.dash===(k===1?"solid":"dash"),"Wrong reconstruction/reference dash");
        assert(t.zorder===(k===2?2:1),"Reference must overlay numerical lines");
        points+=t.y.length;
      });
    });
    figures++;
  }
  for(const c of payload.cases)for(const mesh of c.meshes){
    app.state.caseId=c.id;app.state.cells=mesh.cells;
    await app.renderCase({geometry:false});fields(mesh);
    const histories=element("convergence-plot").data;
    const equationHistories=element("equation-residual-plot").data;
    const cfl=element("cfl-plot").data;
    ["constant","muscl_van_leer"].forEach((method,i)=>{
      const h=mesh.methods[method].history;
      const rows=h.accepted.map(row=>Object.fromEntries(h.columns.map((key,j)=>[key,row[j]])));
      if(rows.length&&rows.every(r=>Array.isArray(r.equation_relative))){
        for(let j=0;j<4;j++){
          const t=equationHistories.find(t=>t.meta.method===method&&t.meta.quantity===j);
          assert(t&&t.mode==="lines","Missing equation residual trace");
          exact(t.x,rows.map(r=>r.step),"Dropped initial or accepted equation state");
          exact(t.y,rows.map(r=>rows[0].relative_active[j]?r.equation_relative[j]:null),"Changed relative residual values");
        }
        const expected=rows.map(row=>{
          let maximum=0;
          row.equation_relative.forEach((value,j)=>{if(row.relative_active[j])maximum=Math.max(maximum,value);});
          return maximum>0?maximum:null;
        });
        exact(histories[i].x,rows.map(r=>r.step),"Overall residual dropped an accepted state");
        exact(histories[i].y,expected,"Overall residual must be the largest active relative equation norm");
        assert(element("equation-residual-plot").layout.legend.entrywidth===.5,"Equation legend must have two columns");
        assert(element("convergence-plot").layout.shapes[0].y0===1e-9,"Overall target must be nine orders");
      }else{
        exact(histories[i].x,rows.map(r=>r.step),"Dropped accepted history");
        assert(histories[i].mode==="lines","Residual histories must be lines");
        exact(histories[i].y,rows.map(r=>Math.max(r.spatial_rms,r.boundary_max,r.conservation_max)),"Changed convergence data");
      }
      for(const [id,key] of [["cp-monitor-plot","Cp_mean"],["mdot-monitor-plot","mdot_out"]]){
        const monitored=rows.filter(r=>Number.isFinite(r.monitors?.[key]?.value));
        const t=element(id).data.find(t=>t.meta.method===method);
        if(monitored.length){
          assert(t,"Recorded engineering monitor omitted");
          exact(t.x,monitored.map(r=>r.step),"Monitor state indices changed");
          exact(t.y,monitored.map(r=>r.monitors[key].value),"Monitor values changed");
        }else assert(!t,"Invented missing integral monitor history");
      }
      const rejected=h.rejected.map(row=>Object.fromEntries(h.columns.map((key,j)=>[key,row[j]])));
      const attempted=[...rows,...rejected].sort((a,b)=>a.step-b.step||(a.retry??0)-(b.retry??0)).filter(r=>r.phase!=="initial"&&Number.isFinite(r.cfl));
      exact(cfl[i].x,attempted.map(r=>r.step),"CFL update/retry coordinates changed");
      exact(cfl[i].y,attempted.map(r=>r.cfl),"CFL must come from actual unsampled trials");
      assert(cfl[i].mode==="lines","CFL histories must be lines");
    });
    const balanceTraces=element("conservation-plot").data;
    let expectedCount=0;
    ["constant","muscl_van_leer"].forEach((method,i)=>{
      const h=mesh.methods[method].history;
      const rows=h.accepted.map(row=>Object.fromEntries(h.columns.map((key,j)=>[key,row[j]])));
      if(rows.length&&rows.every(r=>Array.isArray(r.integrated_balance_scaled))){
        for(let j=0;j<4;j++){
          const t=balanceTraces.find(t=>t.meta.method===method&&t.meta.quantity===j);expectedCount++;
          assert(t,"Missing per-equation history");
          exact(t.x,rows.map(r=>r.step),"Balance update coordinates");
          exact(t.y,rows.map(r=>Math.abs(r.integrated_balance_scaled[j])),"Balance component was changed");
          assert(t.line.dash===(i===0?"dash":"solid"),"Wrong balance style");
        }
      }
    });
    assert(balanceTraces.length===expectedCount,"Missing or fabricated conservation traces");
    assert(element("conservation-plot").layout.legend.entrywidth===.5,"Conservation legend must have two columns");
    if(expectedCount<8)assert(element("conservation-note").textContent.includes("not recorded"),"Missing data must be explained");
  }
  assert(!elements.has("case-diagnostics"),"Selected-case diagnostics pane must be removed");
  let visible2DArrowheads=0;
  function checkArrowMarkers(trace,arrows,h,v,color,minimumLength=0) {
    const expected=arrows.filter(a=>Math.hypot(a.head[h]-a.tail[h],a.head[v]-a.tail[v])>minimumLength);
    assert(trace&&trace.mode==="markers"&&trace.zorder===5,"Arrowheads must be drawn above the lines");
    assert(trace.marker.symbol==="arrow"&&trace.marker.size===15&&trace.marker.angleref==="up","Arrowheads must keep a visible pixel size");
    assert(trace.marker.color===color,"Arrowheads must use the streamline color");
    exact(trace.x,expected.map(a=>a.head[h]),"Arrowhead horizontal position");
    exact(trace.y,expected.map(a=>a.head[v]),"Arrowhead vertical position");
    trace.marker.angle.forEach((degrees,i)=>{
      const a=expected[i],dx=a.head[h]-a.tail[h],dy=a.head[v]-a.tail[v],length=Math.hypot(dx,dy),radians=degrees*Math.PI/180;
      assert(Math.abs(Math.sin(radians)-dx/length)<1e-12&&Math.abs(Math.cos(radians)-dy/length)<1e-12,"Arrowhead points against the projected flow");
    });
    visible2DArrowheads+=expected.length;
  }
  for(const id of Object.keys(payload.channels.cases)) {
    app.state.channelCase=id;await app.renderChannel();
    const record=payload.channels.cases[id];
    for(const view of ["meridional","frontal","annular"]){
      const traces=element("channel-"+view).data;
      assert(traces.length>0,"Missing simultaneous channel view");
      assert(traces.every(t=>!(t.name??"").includes("control")),"Control polygon/points shown");
      const pathsInView=traces.filter(t=>t.legendgroup==="paths"&&t.mode==="lines");
      assert(pathsInView.every(t=>t.line.color==="#2462d1"&&(t.opacity??1)===1),"Streamlines must share one color and opacity");
      assert((element("channel-"+view).layout.annotations??[]).filter(a=>a.showarrow).every(a=>a.arrowcolor==="#2462d1"),"2D arrow color differs");
      if(view==="annular") {
        const paths=traces.filter(t=>t.type==="scatter3d"&&t.legendgroup?.includes("path"));
        assert(paths.length===record.paths_3d.length,"3D paths missing");
        paths.forEach((t,i)=>{exact(t.x,record.paths_3d[i].x,"3D x");exact(t.y,record.paths_3d[i].y,"3D y");exact(t.z,record.paths_3d[i].z,"3D z");});
        const cones=traces.filter(t=>t.type==="cone");
        assert(cones.length===record.paths_3d.length,"Flow arrows missing");
        assert(cones.every(t=>t.colorscale.every(c=>c[1]==="#2462d1")),"3D arrow color differs");
      } else if(view==="frontal") {
        const paths=traces.filter(t=>t.legendgroup==="paths"&&t.mode==="lines");
        assert(paths.length===record.paths_3d.length,"Frontal paths missing");
        paths.forEach((t,i)=>{exact(t.x,record.paths_3d[i].y,"Frontal y projection");exact(t.y,record.paths_3d[i].z,"Frontal z projection");});
        const outer=Math.max(...record.walls.minus.r,...record.walls.plus.r);
        const arrows=record.paths_3d.flatMap(p=>p.arrows).filter(a=>Math.hypot(a.head[1]-a.tail[1],a.head[2]-a.tail[2])>1e-12*outer);
        const layout=element("channel-frontal").layout;
        assert(layout.yaxis.scaleanchor==="x"&&layout.yaxis.scaleratio===1,"Frontal axes distort the flow paths");
        const heads=traces.filter(t=>t.meta?.role==="flow-direction");
        assert(heads.length===record.paths_3d.length,"A frontal streamline has no direction marker trace");
        heads.forEach((t,i)=>checkArrowMarkers(t,record.paths_3d[i].arrows,1,2,"#2462d1",1e-12*outer));
        assert(heads.reduce((n,t)=>n+t.x.length,0)===arrows.length,"Projected arrowheads omitted");
      } else {
        assert(element("channel-meridional").layout.yaxis.scaleanchor==="x","Section is not equal scale");
        const heads=traces.filter(t=>t.meta?.role==="flow-direction");
        assert(heads.length===record.cross_gap_paths.length,"A meridional streamline has no direction marker trace");
        heads.forEach((t,i)=>checkArrowMarkers(t,record.cross_gap_paths[i].arrows,0,1,"#2462d1"));
      }
      channels++;
    }
  }
  // Synthetic UI fixture verifies future data handling, without solving any flow.
  app.state.caseId=payload.cases[0].id;app.state.cells=400;
  const mesh=payload.cases[0].meshes.find(m=>m.cells===400);
  const originals={};
  ["constant","muscl_van_leer"].forEach((method,i)=>{
    originals[method]=mesh.methods[method].history;
    mesh.methods[method].history={columns:["step","spatial_rms","boundary_max","conservation_max","integrated_balance_scaled","inner_iterations","rejected_newton_trials"],accepted:[[1,1,1,1,[1,-2,3,-4],5,0],[2,.1,.1,.1,[0,-.2,.3,-.4],4,0]],rejected:[],counts:{accepted:2,rejected:0}};
  });
  // Runtime payload was serialized into the VM: mutate the actual app object.
  const actualMesh=app.data.cases[0].meshes.find(m=>m.cells===400);
  const actualOriginals={};
  ["constant","muscl_van_leer"].forEach(method=>{actualOriginals[method]=actualMesh.methods[method].history;actualMesh.methods[method].history=mesh.methods[method].history;});
  await app.renderCase({geometry:false});
  const balances=element("conservation-plot").data;
  assert(balances.length===8,"Future histories need all four quantities and both methods");
  balances.forEach((t,i)=>{
    exact(t.x,[1,2],"Conservation iteration coordinates");
    const j=Math.floor(i/2);assert(t.y[0]===j+1&&Math.abs(t.y[1]-(j===0?0:(j+1)*.1))<1e-15,"Conservation absolute components");
    assert(t.line.dash===(i%2===0?"dash":"solid"),"Conservation reconstruction style");
    assert(t.line.color===balances[2*j].line.color,"Same quantity must use same color");
  });
  const priorMonitoring={};
  ["constant","muscl_van_leer"].forEach(method=>{
    priorMonitoring[method]=actualMesh.methods[method].monitoring;
    actualMesh.methods[method].monitoring={policy:{relative_tolerance:1e-9}};
    actualMesh.methods[method].history={columns:["step","phase","cfl","equation_relative","relative_active","monitors","spatial_rms","boundary_max","conservation_max"],
      accepted:[[0,"initial",.2,[1,1,0,1],[true,true,false,true],{Cp_mean:{value:.5},mdot_out:{value:1}},1,1,1],
        [1,"accepted",.2,[1e-10,3e-10,0,1e-10],[true,true,false,true],{Cp_mean:{value:.6},mdot_out:{value:1.1}},1e-10,1e-10,1e-10]],
      rejected:[],counts:{initial:1,accepted_updates:1,accepted:2,rejected:0}};
  });
  await app.renderCase({geometry:false});
  const relativeTraces=element("equation-residual-plot").data;
  assert(relativeTraces.length===8,"Need four equation curves for each reconstruction");
  relativeTraces.forEach(t=>{
    exact(t.x,[0,1],"Relative residual must include the initial state");
    exact(t.y,t.meta.quantity===2?[null,null]:t.meta.quantity===1?[1,3e-10]:[1,1e-10],"Relative residual/inactive equation values changed");
  });
  assert(element("convergence-plot").layout.shapes[0].y0===1e-9,"Nine-order residual target omitted");
  assert(element("equation-residual-plot").layout.legend.entrywidth===.5,"Equation legend must have two columns");
  assert(element("convergence-plot").data.length===2,"Overall plot needs one line per reconstruction");
  element("convergence-plot").data.forEach(t=>exact(t.y,[1,3e-10],"Overall relative maximum incorrect"));
  for(const [id,values] of [["cp-monitor-plot",[.5,.6]],["mdot-monitor-plot",[1,1.1]]]){
    assert(element(id).data.length===2,"Need both engineering monitor histories");
    element(id).data.forEach(t=>{exact(t.x,[0,1],"Monitor omitted initial state");exact(t.y,values,"Changed engineering quantity");});
  }
  assert(element("engineering-monitor-row").hidden===false,"Recorded engineering plots were hidden");
  ["constant","muscl_van_leer"].forEach(method=>{
    actualMesh.methods[method].history=actualOriginals[method];mesh.methods[method].history=originals[method];
    actualMesh.methods[method].monitoring=priorMonitoring[method];
  });
  await app.setTheme("dark");
  assert(element("field-plot").data[2].line.color==="#b277ef","Dark reference contrast");
  for(const view of ["meridional","frontal","annular"]){
    const traces=element("channel-"+view).data;
    assert(traces.filter(t=>t.legendgroup==="paths"&&t.mode==="lines").every(t=>t.line.color==="#88b7ff"),"Dark-theme streamline colors differ");
    assert((element("channel-"+view).layout.annotations??[]).filter(a=>a.showarrow).every(a=>a.arrowcolor==="#88b7ff"),"Dark-theme arrows differ");
    assert(traces.filter(t=>t.type==="cone").every(t=>t.colorscale.every(c=>c[1]==="#88b7ff")),"Dark 3D arrows differ");
    assert(traces.filter(t=>t.meta?.role==="flow-direction").every(t=>t.marker.color==="#88b7ff"&&t.marker.size===15),"Dark 2D arrowheads differ");
  }
  assert(!elements.has("transient-plot"),"Removed contact graph was rendered");
  assert(!elements.has("channel-plot"),"Old single geometry graph was rendered");
  return {passed:true,field_figures:figures,full_line_points_checked:points,channel_views:channels,visible_2D_arrowheads_checked:visible2DArrowheads,synthetic_future_balance_traces:8,synthetic_equation_residual_traces:8,synthetic_integral_monitor_traces:4,internal_pages_checked:4,embedded_evidence_download_checked:true,visual_browser_QA:false};
}
