/* Run auditTransientReport(htmlText) in a JavaScript runtime.
 * Canvas calls are checked for finite coordinates; this is a control/runtime
 * audit, not browser layout or video-codec certification.
 */
function auditTransientReport(htmlText) {
  const dataText=htmlText.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
  const code=htmlText.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
  const parsed=JSON.parse(dataText),elements=new Map(),created=[];
  let canvasCalls=0,downloads=0,tracksStopped=0,requestId=0;
  const classes=()=>({values:new Set(),contains(value){return this.values.has(value);},toggle(value,force){const yes=force===undefined?!this.values.has(value):force;if(yes)this.values.add(value);else this.values.delete(value);return yes;}});
  function element(id) {
    return {id,children:[],dataset:{},style:{},classList:classes(),checked:false,disabled:false,hidden:false,value:'',textContent:'',events:{},
      set innerHTML(value){this.children=[];this.html=value;},get innerHTML(){return this.html||'';},
      append(...children){this.children.push(...children);},addEventListener(event,fn){this.events[event]=fn;},click(){downloads++;}};
  }
  const context={};
  for(const name of ['fillRect','clearRect','moveTo','lineTo','rect'])context[name]=(...args)=>{if(args.some(value=>!Number.isFinite(value)))throw new Error('Invalid canvas '+name);canvasCalls++;};
  for(const name of ['save','restore','beginPath','closePath','clip','fill','stroke','setLineDash'])context[name]=()=>{canvasCalls++;};
  context.fillText=(text,x,y)=>{if(!Number.isFinite(x)||!Number.isFinite(y)||String(text).includes('undefined'))throw new Error('Invalid canvas text');canvasCalls++;};
  const ids=['data','comparison','timeline','play','restart','previous','next','speed','loop','methods','exact','autoscale','theme','download','record','simulation','theory','evidence','case-title','case-description','case-settings','case-notes','time-label','frame-status','export-status','result-table','implicit-comparison','method-diagnostics','validation-summary','provenance'];
  for(const id of ids)elements.set(id,element(id));
  elements.get('data').textContent=dataText;elements.get('speed').value='1';elements.get('loop').checked=true;elements.get('exact').checked=true;
  const canvas=elements.get('comparison');canvas.width=1500;canvas.height=1130;canvas.getContext=()=>context;canvas.captureStream=()=>({getTracks:()=>[{stop(){tracksStopped++;}}]});
  const nav=[...parsed.cases.map(c=>c.id),'theory','evidence'].map(id=>{const item=element('nav-'+id);item.dataset.page=id;return item;});
  const gifLinks=[...htmlText.matchAll(/data-gif-target="([^"]+)" data-gif-name="([^"]+)"/g)].map(match=>{const link=element('gif-link-'+match[1]);link.dataset.gifTarget=match[1];link.dataset.gifName=match[2];const image=element(match[1]);image.src='data:image/gif;base64,runtime-fixture';elements.set(match[1],image);return link;});
  const document={body:{classList:classes()},getElementById:id=>elements.get(id),createElement:tag=>{const item=element(tag);created.push(item);return item;},querySelectorAll:query=>query==='[data-page]'?nav:query==='[data-gif-target]'?gifLinks:[...nav,...created.filter(item=>item.dataset.method)]};
  const Blob=function(parts,options){this.parts=parts;this.options=options;};
  const URL={createObjectURL:()=> 'blob:local',revokeObjectURL:()=>{}};
  const raf=()=>++requestId,cancel=()=>{},timeout=fn=>fn();
  class Recorder {
    static isTypeSupported(){return true;}
    constructor(){this.state='inactive';}
    start(){this.state='recording';}
    stop(){this.state='inactive';this.ondataavailable?.({data:{size:1}});this.onstop?.();}
  }
  const app=new Function('document','requestAnimationFrame','cancelAnimationFrame','setTimeout','Blob','URL','MediaRecorder',code+'\nreturn {showPage,setFrame,render,play,pause,tick,recordAnimation,getState:()=>({frame,playing,case:activeCase.id}),getData:()=>DATA};')(document,raf,cancel,timeout,Blob,URL,Recorder);
  if(JSON.stringify(app.getData().provenance)!==JSON.stringify(parsed.provenance))throw new Error('Presentation mutated embedded provenance');
  let renderedFrames=0,methodToggles=0;
  for(const caseData of parsed.cases) {
    app.showPage(caseData.id);
    for(let i=0;i<caseData.times.length;i++){app.setFrame(i);renderedFrames++;}
    const controls=elements.get('methods').children.filter(item=>item.children[0]?.dataset.method);
    for(const label of controls) {
      const input=label.children[0];input.checked=false;input.events.change();input.checked=true;input.events.change();methodToggles++;
    }
    elements.get('autoscale').checked=true;elements.get('autoscale').events.change();elements.get('autoscale').checked=false;
    elements.get('timeline').value='1';elements.get('timeline').events.input();if(app.getState().frame!==1)throw new Error('Scrubber failed');
    elements.get('previous').onclick();if(app.getState().frame!==0)throw new Error('Previous frame failed');
    elements.get('next').onclick();if(app.getState().frame!==1)throw new Error('Next frame failed');
    elements.get('restart').onclick();app.play();app.tick(0);app.tick(60);if(app.getState().frame!==1)throw new Error('Animation playback failed');app.pause();
    elements.get('loop').checked=false;app.play();app.tick(0);app.tick(1000000);if(app.getState().playing)throw new Error('End-of-time pause failed');
    elements.get('loop').checked=true;app.play();app.tick(0);app.tick(1000000);if(app.getState().frame!==0)throw new Error('Loop failed');app.pause();
    app.recordAnimation();app.tick(0);app.tick(1000000);if(tracksStopped<1)throw new Error('Video export control did not stop tracks');
  }
  app.showPage('theory');if(!elements.get('simulation').hidden||elements.get('theory').hidden)throw new Error('Theory navigation failed');
  app.showPage('evidence');if(elements.get('evidence').hidden)throw new Error('Evidence navigation failed');
  elements.get('theme').onclick();if(!document.body.classList.contains('dark'))throw new Error('Theme failed');
  elements.get('download').onclick();if(downloads<parsed.cases.length+1)throw new Error('Embedded download failed');
  for(const link of gifLinks)link.events.click({preventDefault(){}});
  return {passed:true,cases:parsed.cases.length,rendered_computed_frames:renderedFrames,method_toggles:methodToggles,
    finite_canvas_calls_checked:canvasCalls,play_pause_scrub:true,navigation:true,theme:true,embedded_download:true,
    mocked_video_event_path:true,embedded_gif_download_events:gifLinks.length,browser_layout_verified:false,video_codec_export_verified:false};
}
