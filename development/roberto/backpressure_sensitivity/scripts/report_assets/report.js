/* Interactive views of saved results. No numerical time integration runs here. */
(() => {
  'use strict';
  const bundle = JSON.parse(document.getElementById('report-data').textContent);
  const report = bundle.report, runs = report.runs, x = report.x_m, lm = report.landmarks;
  const descending = runs.filter(r => r.mode === 'descending');
  const cold = runs.filter(r => r.mode === 'cold');
  const restarts = runs.filter(r => r.mode === 'supersonic_restart');
  const $ = id => document.getElementById(id);
  let ink, muted, grid, axis;
  const cssColor = key => getComputedStyle(document.documentElement).getPropertyValue(key).trim();
  function readTheme() {ink=cssColor('--ink');muted=cssColor('--muted');grid=cssColor('--grid');axis=cssColor('--line');}
  readTheme();
  const numericalColor = bundle.colors[80], coldColor = bundle.colors[120], restartColor = bundle.colors[35];
  const fmt = (v, digits = 4) => Number.isFinite(v) ? Number(v).toFixed(digits) : '—';
  const exp = v => Number.isFinite(v) ? v.toExponential(2) : 'undefined at rest';
  const labels = {descending:'Descending continuation', cold:'Independent simple start', supersonic_restart:'Supersonic restart'};
  const regimes = {stagnant:'Stagnant', subsonic:'Subsonic', critical_sonic:'Sonic-throat limit', internal_shock:'Internal shock',
    shock_at_exit:'Shock at exit', overexpanded_external:'Overexpanded · external adjustment', adapted:'Adapted', underexpanded_external:'Underexpanded · external expansion'};
  const palette = ratio => {
    const nearest=descending.reduce((best,r,i)=>Math.abs(r.p_back_ratio-ratio)<Math.abs(descending[best].p_back_ratio-ratio)?i:best,0);
    return bundle.colors[Math.round(nearest*120/(descending.length-1))];
  };
  const reference = row => bundle.references[row.p_back_ratio.toFixed(12)];
  const throatIndex = report.area_m2.indexOf(Math.min(...report.area_m2));
  const figures = new Map();
  const config = {responsive:true, displaylogo:false, scrollZoom:false, doubleClick:'reset',
    modeBarButtonsToRemove:['select2d','lasso2d'], toImageButtonOptions:{format:'svg', filename:'nozzle_backpressure', scale:1}};

  // Mirrors TurboDash's Plotly simple_white framing and inward/mirrored axes.
  function set_plot_options(xTitle, yTitle, options = {}) {
    const ax = title => ({title:{text:title, font:{size:13}, standoff:12}, tickfont:{size:11}, color:ink,
      ticks:'inside', ticklen:6, tickwidth:1, showline:true, mirror:true, linewidth:1.15,
      linecolor:axis, showgrid:true, gridcolor:grid, zeroline:false, automargin:true});
    return {paper_bgcolor:cssColor('--panel'), plot_bgcolor:cssColor('--panel'), font:{family:'Segoe UI, Arial, sans-serif', size:12, color:ink},
      margin:{l:62,r:22,t:30,b:58}, height:340, hovermode:'closest', dragmode:'zoom',
      hoverlabel:{bgcolor:cssColor('--panel'), bordercolor:axis, font:{color:ink, family:'Segoe UI, Arial, sans-serif',size:12}},
      xaxis:ax(xTitle), yaxis:ax(yTitle), showlegend:false,
      legend:{orientation:'h', x:0, y:-.22, font:{size:10}, bgcolor:cssColor('--panel'), borderwidth:0},
      ...options};
  }
  function landmarkShapes() {
    return [lm.critical_ratio,lm.shock_exit_ratio,lm.adapted_ratio].map(v => ({type:'line', xref:'x', yref:'paper',
      x0:v,x1:v,y0:0,y1:1,line:{color:'#aabfce',width:1,dash:'dot'},layer:'below'}));
  }
  function pressureLayout(yTitle, range=[1.005,0]) {
    const layout = set_plot_options('Imposed backpressure p<sub>b</sub>/p<sub>0</sub>  → decreasing',yTitle);
    layout.xaxis.range = range;
    layout.shapes = landmarkShapes();
    return layout;
  }
  function trace(xv,yv,name,color=numericalColor,extra={}) {
    return {type:'scatter',mode:'lines+markers',x:xv,y:yv,name,
      line:{color,width:1.9},marker:{color,size:4},connectgaps:false,
      hovertemplate:'%{x:.7g}<br>%{y:.7g}<extra>%{fullData.name}</extra>',...extra};
  }
  async function draw(id,data,layout,revision=id) {
    layout.uirevision=revision;
    // Plotly mutates its input layout during zoom/relayout; keep reset ranges separate.
    figures.set(id,{data,layout:JSON.parse(JSON.stringify(layout))});
    await Plotly.react($(id),data,layout,config);
  }
  const px = descending.map(r => r.p_back_ratio);
  const xt = row => row.history.at(-1).spatial_rms;
  function operatingTraces(key, exactKey, refField) {
    const exact = exactKey ? descending.map(r=>r[exactKey]) : descending.map(r=>reference(r)[refField][throatIndex]);
    const traces = [trace(px,descending.map(r=>r[key]),'Numerical continuation'),
      trace(px,exact,'Analytical interior',cssColor('--reference'),{mode:'lines',line:{color:cssColor('--reference'),width:1.6,dash:'dash'}})];
    if ($('show-cold').checked) traces.push(trace(cold.map(r=>r.p_back_ratio),cold.map(r=>r[key]),labels.cold,coldColor,
      {mode:'markers',marker:{symbol:'circle-open',size:8,color:coldColor,line:{width:1.6}}}));
    if ($('show-restarts').checked) traces.push(trace(restarts.map(r=>r.p_back_ratio),restarts.map(r=>r[key]),labels.supersonic_restart,restartColor,
      {mode:'markers',marker:{symbol:'triangle-up',size:9,color:restartColor}}));
    return traces;
  }
  async function operatingMaps() {
    const specs=[['mass-chart','mdot_kg_s','mdot_exact_kg_s','Mass flow [kg/s]'],
      ['entropy-chart','entropy_generation_W_K','entropy_generation_exact_W_K','Interior entropy-flux rise [W/K]'],
      ['temperature-chart','exit_T_K','exit_T_exact_K','Last-interior temperature [K]'],
      ['pressure-chart','exit_p_ratio','exit_p_exact_ratio','Last-interior pressure p/p<sub>0</sub>'],
      ['throat-chart','throat_M',null,'Throat Mach number'],['exit-chart','exit_M',null,'Last-interior Mach number']];
    await Promise.all(specs.map(async ([id,key,exact,title])=>{
      let data=operatingTraces(key,exact,'M');
      if(id==='exit-chart') data[1].y=descending.map(r=>reference(r).M.at(-2));
      if(id==='entropy-chart') data.push(trace(px,descending.map(r=>r.shock_entropy_generation_including_exit_W_K),
        'Exact shock loss including exit plane',bundle.colors[90],{mode:'lines',line:{color:bundle.colors[90],width:1.2,dash:'dot'}}));
      await draw(id,data,pressureLayout(title));
    }));
    const choke=pressureLayout('Throat Mach number',[.942,.930]);
    choke.yaxis.range=[.74,1.03];
    choke.shapes.push({type:'rect',xref:'x',yref:'paper',x0:.9340,x1:.9342,y0:0,y1:1,fillcolor:'rgba(54,124,177,.13)',line:{width:0},layer:'below'});
    const release=pressureLayout('Last-interior Mach number',[.55,.40]);
    release.yaxis.range=[.4,2.3];
    release.shapes.push({type:'rect',xref:'x',yref:'paper',x0:.420,x1:.425,y0:0,y1:1,fillcolor:'rgba(54,124,177,.13)',line:{width:0},layer:'below'});
    const exit=operatingTraces('exit_M',null,'M');exit[1].y=descending.map(r=>reference(r).M.at(-2));
    await Promise.all([draw('choking-chart',operatingTraces('throat_M',null,'M'),choke),draw('release-chart',exit,release)]);
  }

  async function geometry() {
    const radius=report.area_m2.map(a=>1000*Math.sqrt(a/Math.PI)), mm=x.map(v=>1000*v);
    const layout=set_plot_options('Axial position [mm]','Equivalent radius [mm]',{height:270,margin:{l:60,r:20,t:28,b:55}});
    layout.xaxis.range=[0,200];layout.yaxis.range=[-13,13];
    layout.yaxis.scaleanchor='x';layout.yaxis.scaleratio=1;
    layout.shapes=[{type:'line',x0:100,x1:100,y0:-13,y1:13,line:{color:'#9bb0c0',dash:'dot',width:1}}];
    layout.annotations=[{x:100,y:9,text:'Throat · A<sub>t</sub> = 150 mm²',showarrow:false,font:{size:11,color:muted}}];
    await draw('geometry-chart',[
      trace(mm,radius,'Equivalent axisymmetric outline',bundle.colors[30],{mode:'lines',line:{color:bundle.colors[30],width:2}}),
      trace(mm,radius.map(v=>-v),'Lower outline',bundle.colors[30],{mode:'lines',line:{color:bundle.colors[30],width:2},
        fill:'tonexty',fillcolor:'rgba(33,108,165,.07)'})],layout);
  }

  let profileRows=descending, currentIndex=0, playTimer=null;
  function populateProfiles(ratio=.425) {
    profileRows=runs.filter(r=>r.mode===$('profile-mode').value);
    currentIndex=profileRows.reduce((best,r,i)=>Math.abs(r.p_back_ratio-ratio)<Math.abs(profileRows[best].p_back_ratio-ratio)?i:best,0);
    $('profile-select').replaceChildren(...profileRows.map((r,i)=>{
      const opt=document.createElement('option');opt.value=i;opt.textContent=`${fmt(r.p_back_ratio,8)} · ${regimes[r.regime_exact]}`;return opt;
    }));
    $('profile-index').max=profileRows.length-1;
    $('profile-index').value=currentIndex;$('profile-select').value=currentIndex;
    return profile();
  }
  function stopPlaying() { clearTimeout(playTimer);playTimer=null; $('profile-play').disabled=false; $('profile-stop').disabled=true; }
  async function profile() {
    const row=profileRows[currentIndex], ref=reference(row), showGhosts=$('profile-ghosts').checked, outlet=$('profile-outlet').checked;
    $('profile-index').value=currentIndex;$('profile-select').value=currentIndex;
    $('profile-index').setAttribute('aria-valuetext',`Backpressure ratio ${fmt(row.p_back_ratio,8)}`);
    $('profile-readout').textContent=`pᵦ/p₀ = ${fmt(row.p_back_ratio,8)} · ${labels[row.mode]} · expected: ${regimes[row.regime_exact]} · ${row.accepted_updates} updates · RMS ${exp(xt(row))}`;
    const cut=values=>showGhosts?values:values.slice(1,-1), stations=cut(x);
    const alt=$('profile-alt').checked?runs.filter(r=>r!==row&&Math.abs(r.p_back_ratio-row.p_back_ratio)<1e-9):[];
    await Promise.all([['p','Pressure p/p<sub>0</sub>',report.p0_Pa],['M','Mach number',1],['T','Temperature [K]',1],['ds','Entropy change s − s<sub>0</sub> [J/(kg K)]',1]].map(async ([key,title,scale])=>{
      const values=r=>cut(r.flow_state[key]).map(v=>v/scale);
      const data=[trace(stations,values(row),labels[row.mode],palette(row.p_back_ratio),{mode:'lines+markers',marker:{size:4,color:palette(row.p_back_ratio)}}),
        trace(stations,cut(ref[key]).map(v=>v/scale),'Analytical interior',cssColor('--reference'),{mode:'lines',line:{color:cssColor('--reference'),width:1.6,dash:'dash'}})];
      for(const other of alt) data.push(trace(stations,values(other),labels[other.mode],other.mode==='cold'?coldColor:restartColor,
        {mode:'lines+markers',line:{color:other.mode==='cold'?coldColor:restartColor,width:1.7,dash:'dot'},marker:{size:4,symbol:'circle-open'}}));
      if(showGhosts) data.push(trace([x[0],x.at(-1)],[row.flow_state[key][0]/scale,row.flow_state[key].at(-1)/scale],
        'Algebraic ghost states',palette(row.p_back_ratio),{mode:'markers',marker:{size:9,symbol:'square-open',color:palette(row.p_back_ratio)}}));
      const layout=set_plot_options('Axial position [m]',title,{showlegend:key==='p',height:360,margin:{l:65,r:18,t:25,b:key==='p'?94:58}});
      layout.xaxis.range=outlet?[.190,.2004]:[0,.2];
      layout.legend={orientation:'h',x:0,y:-.25,font:{size:10}};
      layout.shapes=[{type:'line',xref:'x',yref:'paper',x0:.1,x1:.1,y0:0,y1:1,line:{color:'#bacbd7',width:1,dash:'dot'},layer:'below'}];
      if(showGhosts) layout.shapes.push({type:'rect',xref:'x',yref:'paper',x0:.1995,x1:.2004,y0:0,y1:1,
        fillcolor:'rgba(88,119,143,.09)',line:{width:0},layer:'below'});
      await draw(`profile-${key}`,data,layout,`profile-${row.mode}-${currentIndex}-${outlet}-${showGhosts}-${$('profile-alt').checked}`);
    }));
  }
  async function playNext() {
    if(currentIndex>=profileRows.length-1){stopPlaying();return;}
    currentIndex++;await profile();
    if($('profile-play').disabled)playTimer=setTimeout(playNext,1000);
  }
  function initProfileControls() {
    $('profile-mode').addEventListener('change',()=>{stopPlaying();populateProfiles(profileRows[currentIndex].p_back_ratio);});
    for(const id of ['profile-index','profile-select']) $(id).addEventListener(id==='profile-index'?'input':'change',()=>{
      stopPlaying();currentIndex=Number($(id).value);profile();
    });
    for(const id of ['profile-outlet','profile-ghosts','profile-alt']) $(id).addEventListener('change',()=>profile());
    $('profile-play').addEventListener('click',()=>{if(currentIndex===profileRows.length-1)currentIndex=-1;
      $('profile-play').disabled=true;$('profile-stop').disabled=false;playNext();});
    $('profile-stop').addEventListener('click',stopPlaying);stopPlaying();
  }

  async function branchComparison() {
    const chosen=runs.filter(r=>Math.abs(r.p_back_ratio-.42)<1e-10&&r.mode!=='supersonic_restart');
    const data=chosen.map(r=>trace(x.slice(1,-1),r.flow_state.p.slice(1,-1).map(v=>v/report.p0_Pa),labels[r.mode],
      r.mode==='cold'?coldColor:numericalColor,{line:{color:r.mode==='cold'?coldColor:numericalColor,width:2},marker:{size:4}}));
    data.push(trace(x.slice(1,-1),reference(chosen[0]).p.slice(1,-1).map(v=>v/report.p0_Pa),'Analytical interior',cssColor('--reference'),
      {mode:'lines',line:{color:cssColor('--reference'),width:1.4,dash:'dash'}}));
    const layout=set_plot_options('Axial position [m]','Pressure p/p<sub>0</sub>',{height:390,showlegend:true,margin:{l:62,r:22,t:25,b:90}});
    layout.xaxis.range=[.190,.2];layout.legend={orientation:'h',x:0,y:-.22,font:{size:11}};
    await draw('branch-chart',data,layout);
  }
  const regionPredicate={all:()=>true,near_rest:r=>r>=.99,choking:r=>r>.93&&r<.945,
    internal_shock:r=>r<.93&&r>lm.shock_exit+.0021,
    shock_exit:r=>r>=.40&&r<=.54,
    external:r=>r<lm.shock_exit-.0021};
  async function histories() {
    const region=$('history-region').value, selected=descending.filter(r=>(regionPredicate[region]||regionPredicate.all)(r.p_back_ratio));
    const data=selected.map((r,i)=>{
      const color=bundle.colors[Math.round(i*120/Math.max(selected.length-1,1))];
      return trace(r.history.map(v=>v.update),r.history.map(v=>v.spatial_rms),fmt(r.p_back_ratio,8),color,
        {mode:r.history.length===1?'markers':'lines',line:{color,width:1.4},marker:{size:6,color},
          hovertemplate:'Update %{x}<br>RMS %{y:.3e}<extra>pᵦ/p₀ '+fmt(r.p_back_ratio,8)+'</extra>'});
    });
    const layout=set_plot_options('Accepted implicit updates','Fixed-scale spatial RMS',
      {height:480,showlegend:true,margin:{l:66,r:18,t:24,b:130}});
    layout.yaxis.type='log';layout.yaxis.range=[-18,0];layout.yaxis.tickformat='.0e';
    layout.legend={orientation:'h',x:0,y:-.19,font:{size:10}};
    layout.shapes=[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:1e-8,y1:1e-8,line:{color:muted,width:1,dash:'dot'}}];
    await draw('history-chart',data,layout,`history-${region}`);
  }
  async function diagnostics() {
    const mass=pressureLayout('Relative range of numerical face mass flux');mass.yaxis.type='log';mass.yaxis.tickformat='.0e';
    const data=[trace(px,descending.map(r=>r.face_mass_spread_relative>0?r.face_mass_spread_relative:null),'Descending continuation')];
    const enthalpy=pressureLayout('Maximum interior total-enthalpy error [%]');
    await Promise.all([draw('mass-balance-chart',data,mass),draw('enthalpy-chart',[
      trace(px,descending.map(r=>100*r.total_enthalpy_error_relative),'Descending continuation'),
      trace(cold.map(r=>r.p_back_ratio),cold.map(r=>100*r.total_enthalpy_error_relative),labels.cold,coldColor,
        {mode:'markers',marker:{symbol:'circle-open',size:8,color:coldColor,line:{width:1.5}}})],enthalpy)]);
  }

  function table() {
    const mode=$('table-mode').value, query=$('table-search').value.trim().toLowerCase();
    const filtered=runs.filter(r=>(mode==='all'||r.mode===mode)&&
      `${labels[r.mode]} ${r.p_back_ratio.toFixed(8)} ${regimes[r.regime_exact]}`.toLowerCase().includes(query));
    $('run-table-body').replaceChildren(...filtered.map(r=>{
      const tr=document.createElement('tr');
      const cells=[labels[r.mode],fmt(r.p_back_ratio,8),regimes[r.regime_exact],exp(xt(r)),String(r.accepted_updates),
        fmt(r.exit_M,5),fmt(r.mdot_kg_s,7),fmt(r.entropy_generation_W_K,5)];
      for(const value of cells){const td=document.createElement('td');td.textContent=value;tr.append(td);}
      return tr;
    }));
    $('table-count').textContent=`${filtered.length} of ${runs.length} states`;
  }
  function download(name,content,type){
    const url=URL.createObjectURL(new Blob([content],{type}));
    const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function provenance() {
    if($('dataset-hash'))$('dataset-hash').textContent=bundle.provenance.input_sha256['summary.json'];
    if($('runtime-version'))$('runtime-version').textContent=bundle.provenance.plotly_js_version;
    if($('provenance-content'))$('provenance-content').textContent=JSON.stringify({
      generated_from:'Saved baseline data; no simulations rerun', source_unchanged:report.source_unchanged,
      source_files_checked:bundle.provenance.source_files, versions:report.versions,
      settings:report.settings, wrapper_scaled_step_error:report.dynamic_vs_static_source_step_scaled_max_error,
      reference_checks:bundle.audit.reference_checks, input_sha256:bundle.provenance.input_sha256,
      plot_style_sources:bundle.provenance.style_sources},null,2);
  }
  async function boot() {
    if(typeof Plotly==='undefined')throw new Error('Embedded Plotly runtime did not initialize.');
    $('report-status').textContent='Preparing interactive figures…';
    for(const id of ['show-cold','show-restarts'])$(id).addEventListener('change',operatingMaps);
    initProfileControls();
    $('history-region').addEventListener('change',histories);
    $('table-mode').addEventListener('change',table);$('table-search').addEventListener('input',table);
    $('download-csv').addEventListener('click',()=>download('backpressure_operating_map.csv',bundle.csv,'text/csv;charset=utf-8'));
    $('download-json').addEventListener('click',()=>download('backpressure_results.json',JSON.stringify({report,audit:bundle.audit},null,2),'application/json'));
    $('reset-charts').addEventListener('click',()=>{
      for(const [id,figure] of figures){
        const update={'xaxis.autorange':!figure.layout.xaxis.range,'yaxis.autorange':!figure.layout.yaxis.range};
        if(figure.layout.xaxis.range)update['xaxis.range']=figure.layout.xaxis.range;
        if(figure.layout.yaxis.range)update['yaxis.range']=figure.layout.yaxis.range;
        Plotly.relayout($(id),update);
      }
    });
    provenance();table();
    const navLinks=[...document.querySelectorAll('.contents a')];
    let navQueued=false;
    function markSection(){
      navQueued=false;
      let current=navLinks[0];
      for(const link of navLinks){
        const section=document.querySelector(link.getAttribute('href'));
        if(section&&section.getBoundingClientRect().top<=160)current=link;
      }
      for(const link of navLinks){if(link===current)link.setAttribute('aria-current','true');else link.removeAttribute('aria-current');}
    }
    window.addEventListener('scroll',()=>{if(!navQueued){navQueued=true;requestAnimationFrame(markSection);}},{passive:true});
    markSection();
    await geometry();await operatingMaps();await populateProfiles();await branchComparison();await histories();await diagnostics();
    document.body.dataset.resultsReady='true';
    $('report-status').textContent=`Ready · ${runs.length} embedded states · interactive figures available offline`;
    window.addEventListener('beforeprint',()=>{stopPlaying();for(const id of figures.keys())Plotly.Plots.resize($(id));});
    document.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>{if(el.open)for(const id of figures.keys())Plotly.Plots.resize($(id));}));
    // Available for read-only QA and useful for advanced readers in the console.
    window.backpressureReport={bundle,figures,profile:()=>profileRows[currentIndex],stopAnimation:stopPlaying,
      refreshTheme:async()=>{readTheme();await geometry();await operatingMaps();await profile();await branchComparison();await histories();await diagnostics();}};
    window.dispatchEvent(new Event('report-results-ready'));
  }
  boot().catch(error=>{
    console.error(error);$('report-status').textContent='Interactive figures could not load: '+error.message;
    $('report-status').setAttribute('role','alert');document.documentElement.dataset.reportError=error.message;
  });
})();
