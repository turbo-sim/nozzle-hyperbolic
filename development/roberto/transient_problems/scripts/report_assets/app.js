'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const COLORS = ['#6a00a8', '#b12a90', '#e16462', '#fca636'];
const FIELDS = [
  {key:'p', label:'Pressure', symbol:'p', unit:'bar', factor:1e-5},
  {key:'rho', label:'Density', symbol:'ρ', unit:'kg/m³', factor:1},
  {key:'u', label:'Velocity', symbol:'u', unit:'m/s', factor:1},
  {key:'M', label:'Mach number', symbol:'M', unit:'–', factor:1},
  {key:'T', label:'Temperature', symbol:'T', unit:'K', factor:1}
];
const canvas = $('comparison');
const ctx = canvas.getContext('2d');
let activeCase = DATA.cases[0];
let frame = 0, playing = false, lastStamp = null, raf = null;
let recording = null, recordingChunks = [], exportLoop = false;
let recordingError = null;
const selected = new Set(activeCase.methods.map(method => method.id));
const methodStyle = new Map();
const extentCache = new Map();
for (const method of activeCase.methods) {
  const i = methodStyle.size;
  methodStyle.set(method.id, {color:COLORS[i % COLORS.length], dash:i===3?[7,4]:i===1?[8,3,2,3]:[], width:i>=2?2.5:2});
}

function number(value, digits=3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude >= 1e5 || magnitude < 1e-3 ? value.toExponential(2) : Number(value.toPrecision(digits)).toString();
}

function theme() {
  const dark = document.body.classList.contains('dark');
  return dark ? {bg:'#241d2d', fg:'#eee6f6', muted:'#baafc7', grid:'#493f54', reference:'#d1c8dc', fill:'#33253f'}
              : {bg:'#ffffff', fg:'#292334', muted:'#756b80', grid:'#e4deeb', reference:'#60576c', fill:'#f0eaf5'};
}

function indexAt(method, time) {
  const times = method.times || activeCase.times;
  let best = 0;
  for (let i=1; i<times.length; i++) if (Math.abs(times[i]-time)<Math.abs(times[best]-time)) best=i;
  return Math.abs(times[best]-time) <= Math.max(1e-12, Math.abs(time)*1e-9) ? best : -1;
}

function extent(field, currentOnly) {
  const cacheKey=activeCase.id+'/'+field.key;
  if(!currentOnly&&extentCache.has(cacheKey))return extentCache.get(cacheKey);
  let low = Infinity, high = -Infinity;
  function add(values) { for (const value of values) { const scaled=value*field.factor; low=Math.min(low,scaled);high=Math.max(high,scaled); } }
  for (const method of activeCase.methods) {
    const rows=method.fields[field.key];
    if (!rows) continue;
    if (currentOnly) { const i=indexAt(method,activeCase.times[frame]); if(i>=0)add(rows[i]); }
    else for(const row of rows)add(row);
  }
  const reference=activeCase.reference?.fields?.[field.key];
  if(reference) { if(currentOnly)add(reference[frame]);else for(const row of reference)add(row); }
  if(!Number.isFinite(low)||!Number.isFinite(high))return [0,1];
  const padding=Math.max((high-low)*.085,Math.abs(high)*.015,1e-8);
  const result=[low-padding,high+padding];
  if(!currentOnly)extentCache.set(cacheKey,result);
  return result;
}

function axes(box, title, unit, low, high, palette) {
  const [left,top,width,height]=box;
  const plot={left:left+74,top:top+43,width:width-105,height:height-95};
  const faces=activeCase.faces || activeCase.x;
  const xmin=faces[0],xmax=faces[faces.length-1];
  plot.X=value=>plot.left+(value-xmin)/(xmax-xmin)*plot.width;
  plot.Y=value=>plot.top+plot.height-(value-low)/(high-low)*plot.height;
  ctx.fillStyle=palette.fg;ctx.font='600 20px system-ui';ctx.fillText(title+' ['+unit+']',left+20,top+24);
  ctx.font='13px system-ui';ctx.lineWidth=1;ctx.setLineDash([]);
  for(let i=0;i<=4;i++) {
    const value=low+(high-low)*i/4,y=plot.Y(value);
    ctx.strokeStyle=palette.grid;ctx.beginPath();ctx.moveTo(plot.left,y);ctx.lineTo(plot.left+plot.width,y);ctx.stroke();
    ctx.fillStyle=palette.muted;ctx.textAlign='right';ctx.fillText(number(value,4),plot.left-10,y+4);
    const x=xmin+(xmax-xmin)*i/4;
    ctx.textAlign='center';ctx.fillText(number(x,3),plot.X(x),plot.top+plot.height+22);
  }
  ctx.strokeStyle=palette.muted;ctx.beginPath();ctx.moveTo(plot.left,plot.top);ctx.lineTo(plot.left,plot.top+plot.height);ctx.lineTo(plot.left+plot.width,plot.top+plot.height);ctx.stroke();
  ctx.fillStyle=palette.muted;ctx.textAlign='center';ctx.fillText('x [m]',plot.left+plot.width/2,plot.top+plot.height+43);ctx.textAlign='left';
  return plot;
}

function curve(plot, x, values, factor, color, width, dash) {
  ctx.save();ctx.beginPath();ctx.rect(plot.left,plot.top,plot.width,plot.height);ctx.clip();
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);
  for(let i=0;i<x.length;i++) { const px=plot.X(x[i]),py=plot.Y(values[i]*factor);if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py); }
  ctx.stroke();ctx.restore();ctx.setLineDash([]);
}

function diagnosticPanel(box, palette) {
  const [left,top,width,height]=box;
  ctx.fillStyle=palette.fg;ctx.font='600 20px system-ui';ctx.fillText('Geometry and endpoint checks',left+20,top+24);
  const area=activeCase.area_centers || activeCase.x.map(()=>1), a0=activeCase.physical_inputs?.inlet_area || area[0] || 1;
  let low=Infinity,high=-Infinity;for(const value of area){low=Math.min(low,value/a0);high=Math.max(high,value/a0);}
  const span=Math.max(high-low,.2), mini={left:left+38,top:top+53,width:width-76,height:70};
  const x0=activeCase.x[0],x1=activeCase.x[activeCase.x.length-1];
  ctx.beginPath();ctx.moveTo(mini.left,mini.top+mini.height);
  for(let i=0;i<area.length;i++)ctx.lineTo(mini.left+(activeCase.x[i]-x0)/(x1-x0)*mini.width,mini.top+mini.height-(area[i]/a0-low+.05*span)/(span*1.1)*mini.height);
  ctx.lineTo(mini.left+mini.width,mini.top+mini.height);ctx.closePath();ctx.fillStyle=palette.fill;ctx.fill();ctx.strokeStyle=palette.muted;ctx.lineWidth=1.5;ctx.stroke();
  ctx.fillStyle=palette.muted;ctx.font='12px system-ui';ctx.fillText('A / inlet A: '+number(low)+'–'+number(high)+' · B: BC defect · C: scaled budget',left+38,top+141);
  let row=top+171;
  const time=activeCase.times[frame];
  for(const method of activeCase.methods) {
    if(!selected.has(method.id))continue;
    const i=indexAt(method,time),style=methodStyle.get(method.id);
    ctx.fillStyle=style.color;ctx.fillRect(left+24,row-10,10,10);ctx.fillStyle=palette.fg;ctx.font='600 12px system-ui';ctx.fillText(method.label || method.id,left+43,row);
    ctx.font='12px system-ui';ctx.fillStyle=palette.muted;
    if(i<0)ctx.fillText('No computed frame at this time',left+43,row+17);
    else {
      const bc=method.metrics?.boundary_max?.[i];
      const budget=method.metrics?.conservation_budget?.[i] || [];
      const maximum=budget.reduce((acc,v)=>Math.max(acc,Math.abs(v)),0);
      const regime=method.outlet_regime?.[i] || (activeCase.id==='sod'?'transmissive':'not recorded');
      const mach=method.outlet_trace_mach?.[i],pressure=method.outlet_trace_pressure?.[i];
      ctx.fillText(regime+' · M '+number(mach)+' · p '+number(pressure===undefined?undefined:pressure*1e-5)+' bar · B '+number(bc)+' · C '+number(maximum),left+43,row+17);
    }
    row+=43;
  }
}

function render() {
  const palette=theme(),time=activeCase.times[frame],reference=activeCase.reference?.fields;
  ctx.fillStyle=palette.bg;ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle=palette.fg;ctx.font='600 23px system-ui';ctx.fillText(activeCase.label,25,31);
  ctx.fillStyle=palette.muted;ctx.font='15px system-ui';
  const tau=activeCase.reference_time?time/activeCase.reference_time:null;
  ctx.fillText('t = '+number(time*1e3,5)+' ms'+(tau!==null?' · τ = '+number(tau,4):'')+' · computed frame '+(frame+1)+'/'+activeCase.times.length,25,57);
  const top=79,panelWidth=canvas.width/2,panelHeight=(canvas.height-top)/3;
  for(let i=0;i<FIELDS.length;i++) {
    const field=FIELDS[i],box=[(i%2)*panelWidth,top+Math.floor(i/2)*panelHeight,panelWidth,panelHeight];
    const [low,high]=extent(field,$('autoscale').checked);
    const plot=axes(box,field.label,field.unit,low,high,palette);
    if($('exact').checked&&reference?.[field.key])curve(plot,activeCase.x,reference[field.key][frame],field.factor,palette.reference,2,[5,5]);
    for(const method of activeCase.methods) {
      if(!selected.has(method.id))continue;
      const index=indexAt(method,time);if(index<0)continue;
      const style=methodStyle.get(method.id);
      curve(plot,activeCase.x,method.fields[field.key][index],field.factor,style.color,style.width,style.dash);
    }
  }
  diagnosticPanel([panelWidth,top+2*panelHeight,panelWidth,panelHeight],palette);
  $('timeline').value=String(frame);
  $('time-label').textContent='t = '+number(time*1e3,5)+' ms'+(tau!==null?' · τ = '+number(tau,4):'')+' · '+(frame+1)+'/'+activeCase.times.length;
  const unavailable=activeCase.methods.filter(method=>selected.has(method.id)&&indexAt(method,time)<0);
  $('frame-status').textContent=unavailable.length?'Missing current-time frame: '+unavailable.map(method=>method.label||method.id).join(', '):'Saved computed states; no temporal interpolation';
}

function setFrame(index) {frame=Math.min(activeCase.times.length-1,Math.max(0,Number(index)));render();}
function pause() {playing=false;lastStamp=null;if(raf!==null)cancelAnimationFrame(raf);raf=null;$('play').textContent='Play';}
function tick(stamp) {
  if(!playing)return;
  const interval=1000/(20*Number($('speed').value));
  if(lastStamp===null)lastStamp=stamp;
  if(stamp-lastStamp>=interval) {
    const advance=Math.max(1,Math.floor((stamp-lastStamp)/interval));lastStamp+=advance*interval;
    if(frame+advance>=activeCase.times.length) {
      setFrame(activeCase.times.length-1);
      if(recording){pause();setTimeout(finishRecording,180);return;}
      if($('loop').checked)setFrame(0);else{pause();return;}
    } else setFrame(frame+advance);
  }
  raf=requestAnimationFrame(tick);
}
function play(){if(frame===activeCase.times.length-1)setFrame(0);playing=true;lastStamp=null;$('play').textContent='Pause';raf=requestAnimationFrame(tick);}

function methodControls() {
  const host=$('methods');host.innerHTML='';
  for(const method of activeCase.methods) {
    const label=document.createElement('label');label.className='method-option';
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=selected.has(method.id);checkbox.dataset.method=method.id;
    checkbox.addEventListener('change',()=>{if(checkbox.checked)selected.add(method.id);else selected.delete(method.id);render();});
    const swatch=document.createElement('span');swatch.className='swatch';swatch.style.borderColor=methodStyle.get(method.id).color;
    if(methodStyle.get(method.id).dash.length)swatch.style.borderTopStyle='dashed';
    const text=document.createElement('span');text.textContent=method.label||method.id;
    label.append(checkbox,swatch,text);host.append(label);
  }
  const label=document.createElement('span');label.className='method-option';label.textContent=activeCase.reference?.fields?'Dashed gray: independent exact reference':'No exact transient nozzle reference';host.append(label);
}

function showPage(id) {
  pause();
  const chosen=DATA.cases.find(caseData=>caseData.id===id);
  $('simulation').hidden=!chosen;$('theory').hidden=id!=='theory';$('evidence').hidden=id!=='evidence';
  document.querySelectorAll('[data-page]').forEach(button=>button.classList.toggle('active',button.dataset.page===id));
  if(!chosen)return;
  activeCase=chosen;frame=0;
  for(const method of activeCase.methods)if(!methodStyle.has(method.id))methodStyle.set(method.id,{color:COLORS[methodStyle.size%COLORS.length],dash:[],width:2});
  $('case-title').textContent=chosen.label;$('case-description').textContent=chosen.description||'';
  const settings=DATA.settings||{};
  $('case-settings').textContent=activeCase.x.length+' physical cells · '+activeCase.times.length+' frames · CFL '+(settings.cfl??settings.CFL??'recorded')+(settings.scheme?' · '+settings.scheme.flux.toUpperCase()+' / '+settings.scheme.reconstruction:'');
  $('timeline').max=String(chosen.times.length-1);
  $('exact').disabled=!chosen.reference?.fields;$('exact').checked=Boolean(chosen.reference?.fields);
  const physicalNote=chosen.id==='sod'?'The exact solution is sampled at each saved physical time. Transmissive endpoints are checked before waves reach the domain ends.':'The physical finite-wave outlet law is shared by all four methods. Coupled/segregated describes the numerical boundary treatment, not a change of physical model. Initial physical cells are stagnant; endpoint traces are completed after the pressure step.';
  $('case-notes').textContent=[chosen.notes||chosen.initial_trace_note,physicalNote].filter(Boolean).join(' ');
  methodControls();render();
}

function lockExport(locked) {
  for(const id of ['restart','previous','play','next','speed','loop','timeline','exact','autoscale','record'])$(id).disabled=locked||(id==='exact'&&!activeCase.reference?.fields);
  document.querySelectorAll('[data-page], [data-method]').forEach(element=>element.disabled=locked);
}
function finishRecording() {if(recording&&recording.state!=='inactive')recording.stop();}
function canRecord() {
  return typeof MediaRecorder!=='undefined'&&typeof canvas.captureStream==='function'&&
    (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')||MediaRecorder.isTypeSupported('video/webm'));
}
function recordAnimation() {
  if(!canRecord())return;
  pause();setFrame(0);exportLoop=$('loop').checked;$('loop').checked=false;$('speed').value='1';recordingChunks=[];recordingError=null;
  const stream=canvas.captureStream(20);
  const mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
  recording=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:6000000});
  recording.ondataavailable=event=>{if(event.data.size)recordingChunks.push(event.data);};
  recording.onstop=()=>{
    if(recordingChunks.length&&!recordingError) {
      const blob=new Blob(recordingChunks,{type:mime}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
      anchor.href=url;anchor.download=activeCase.id+'_boundary_treatments.webm';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1500);
    }
    for(const track of stream.getTracks())track.stop();recording=null;$('loop').checked=exportLoop;lockExport(false);
    $('export-status').textContent=recordingError?'Export unavailable: '+recordingError:recordingChunks.length?'WebM exported from computed frames':'No encoded video produced';
  };
  recording.onerror=event=>{pause();recordingError=event.error?.message||'recorder error';finishRecording();};
  lockExport(true);$('export-status').textContent='Recording computed animation…';recording.start();play();
}

function evidence() {
  const host=$('result-table');host.innerHTML='';
  for(const row of DATA.presentation_summary.methods) {
    const tr=document.createElement('tr'),steps=row.summary.accepted_steps??row.summary.steps??'n/a';
    for(const value of [row.case+' / '+row.label,String(row.success),String(steps),String(row.frames),number(row.boundary_max),number(row.conservation_budget_max)]) {
      const td=document.createElement('td');td.textContent=value;tr.append(td);
    }host.append(tr);
  }
  $('implicit-comparison').textContent=JSON.stringify(DATA.presentation_summary.implicit_comparisons,null,2);
  $('method-diagnostics').textContent=JSON.stringify(DATA.presentation_summary.methods.map(row=>({case:row.case,method:row.method,...row.summary})),null,2);
  const validation={source_matches_campaign:DATA.report_evidence_source_matches_campaign};
  const audit=DATA.report_evidence?.independent_audit;
  if(audit)validation.independent_audit={passed:audit.passed,records:audit.records.length,passed_records:audit.records.filter(row=>row.passed).length,implicit_comparisons:audit.implicit_comparisons,failed_records:audit.records.filter(row=>!row.passed),campaign_source_matches_current:audit.campaign_source_matches_current,campaign_client_matches_current:audit.campaign_client_matches_current};
  const tests=DATA.report_evidence?.production_tests;
  if(tests)validation.production_tests={passed:tests.passed,counts:tests.counts,source_unchanged:(tests.provenance||tests).source_unchanged};
  $('validation-summary').textContent=JSON.stringify(validation,null,2);
  const provenance={settings:DATA.settings,provenance:{...DATA.provenance}};
  if(provenance.provenance)for(const key of Object.keys(provenance.provenance))if(key.includes('source')&&typeof provenance.provenance[key]==='object'&&provenance.provenance[key]!==null)provenance.provenance[key]={files:Object.keys(provenance.provenance[key]).length,full_hashes:'embedded download'};
  $('provenance').textContent=JSON.stringify(provenance,null,2);
}

$('play').onclick=()=>playing?pause():play();$('previous').onclick=()=>{pause();setFrame(frame-1);};$('next').onclick=()=>{pause();setFrame(frame+1);};$('restart').onclick=()=>{pause();setFrame(0);};
$('timeline').addEventListener('input',()=>{pause();setFrame($('timeline').value);});
$('exact').addEventListener('change',render);$('autoscale').addEventListener('change',render);
$('theme').onclick=()=>{document.body.classList.toggle('dark');render();};
$('download').onclick=()=>{const blob=new Blob([JSON.stringify(DATA)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='transient_boundary_treatment_data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);};
$('record').onclick=recordAnimation;
if(!canRecord()){$('record').disabled=true;$('record').title='This browser does not support canvas WebM export; playback still works offline.';}
document.querySelectorAll('[data-page]').forEach(button=>button.onclick=()=>showPage(button.dataset.page));
document.querySelectorAll('[data-gif-target]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();const a=document.createElement('a');a.href=$(link.dataset.gifTarget).src;a.download=link.dataset.gifName;a.click();}));
evidence();showPage(activeCase.id);
