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
  assert(app.state.stride===11 && app.state.markerSkip===10,"Default is ten skipped points");
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
  function fields(mesh,stride) {
    const traces=element("field-plot").data;
    assert(traces.length===30,"Six panels require three full lines and two marker overlays each");
    ["p","Cp","M_m","v_m","v_theta","rho"].forEach((field,j)=>{
      const ts=traces.slice(5*j,5*j+5);
      ["constant","muscl_van_leer","analytical"].forEach((method,k)=>{
        const t=ts[k];
        exact(t.x,mesh.m,"Line coordinates must remain complete");
        exact(t.y,method==="analytical"?mesh.reference[field]:mesh.methods[method].fields[field],"Changed field values");
        assert(t.mode==="lines","Full trend must be a line");
        assert(t.line.dash===(k===1?"solid":"dash"),"Wrong reconstruction/reference dash");
        assert(t.zorder===(k===2?2:1),"Reference must overlay numerical lines");
        points+=t.y.length;
      });
      ["constant","muscl_van_leer"].forEach((method,k)=>{
        const t=ts[3+k],offset=k===0?0:Math.floor(stride/2);
        const indices=mesh.m.map((_,i)=>i).filter(i=>i%stride===offset);
        exact(t.x,indices.map(i=>mesh.m[i]),"Wrong staggered marker positions");
        exact(t.y,indices.map(i=>mesh.methods[method].fields[field][i]),"Wrong marker values");
        assert(t.mode==="markers"&&t.zorder===3,"Markers must overlay all lines");
        assert(t.marker.color==="white"&&t.marker.symbol==="circle"&&t.marker.size===9,"Marker styling");
        assert(t.marker.line.width>=1.5,"Marker edge must remain legible");
      });
    });
    figures++;
  }
  for(const skip of [0,1,2,4,10,11,1000]) {
    const input=element("marker-skip");input.value=String(skip);
    await input.handlers.input({target:input});
    assert(app.state.stride===skip+1,"Skip input did not set stride");
    for(const c of payload.cases)for(const mesh of c.meshes){
      app.state.caseId=c.id;app.state.cells=mesh.cells;
      await app.renderCase({geometry:false});fields(mesh,skip+1);
      const histories=element("convergence-plot").data;
      ["constant","muscl_van_leer"].forEach((method,i)=>{
        const h=mesh.methods[method].history;
        const rows=h.accepted.map(row=>Object.fromEntries(h.columns.map((key,j)=>[key,row[j]])));
        exact(histories[i].x,rows.map(r=>r.step),"Dropped accepted history");
        const offset=i===0?0:Math.floor((skip+1)/2);
        exact(histories[i].marker.size,rows.map((_,j)=>j%(skip+1)===offset?9:0),"Convergence staggered markers");
        exact(histories[i].y,rows.map(r=>Math.max(r.spatial_rms,r.boundary_max,r.conservation_max)),"Changed convergence data");
      });
      const balanceTraces=element("conservation-plot").data;
      let expectedCount=0;
      ["constant","muscl_van_leer"].forEach((method,i)=>{
        const h=mesh.methods[method].history;
        const rows=h.accepted.map(row=>Object.fromEntries(h.columns.map((key,j)=>[key,row[j]])));
        if(rows.every(r=>Array.isArray(r.integrated_balance_scaled))){
          for(let j=0;j<4;j++){
            const t=balanceTraces[expectedCount++];
            exact(t.x,rows.map(r=>r.step),"Balance update coordinates");
            exact(t.y,rows.map(r=>Math.abs(r.integrated_balance_scaled[j])),"Balance component was changed");
            assert(t.line.dash===(i===0?"dash":"solid"),"Wrong balance style");
          }
        }
      });
      assert(balanceTraces.length===expectedCount,"Missing or fabricated conservation traces");
      if(expectedCount<8)assert(element("conservation-note").textContent.includes("not recorded"),"Missing data must be explained");
    }
  }
  for(const invalid of ["", "-1", "1.5", "NaN", "9007199254740991"]){
    const input=element("marker-skip"),old=app.state.stride;input.value=invalid;
    await input.handlers.input({target:input});assert(app.state.stride===old,"Invalid input changed stride");
  }
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
    const j=i%4;assert(t.y[0]===j+1&&Math.abs(t.y[1]-(j===0?0:(j+1)*.1))<1e-15,"Conservation absolute components");
    assert(t.line.dash===(i<4?"dash":"solid"),"Conservation reconstruction style");
    assert(t.line.color===balances[j].line.color,"Same quantity must use same color");
  });
  ["constant","muscl_van_leer"].forEach(method=>{actualMesh.methods[method].history=actualOriginals[method];mesh.methods[method].history=originals[method];});
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
  return {passed:true,field_figures:figures,full_line_points_checked:points,channel_views:channels,visible_2D_arrowheads_checked:visible2DArrowheads,synthetic_future_balance_traces:8,internal_pages_checked:4,embedded_evidence_download_checked:true,visual_browser_QA:false};
}
