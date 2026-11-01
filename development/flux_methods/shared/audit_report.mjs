/** Execute the real saved-data UI with DOM/Plotly stubs; never launch a browser. */
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const decode=s=>String(s).replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
class ClassList{constructor(s=''){this.items=new Set(s.split(/\s+/).filter(Boolean));}add(s){this.items.add(s);}remove(s){this.items.delete(s);}contains(s){return this.items.has(s);}toggle(s){if(this.items.has(s)){this.items.delete(s);return false;}this.items.add(s);return true;}}
class Element{
  constructor(id='',tag='',classes=''){this.id=id;this.tagName=tag.toUpperCase();this.classList=new ClassList(classes);this.dataset={};this.style={};this.multiple=false;this.options=[];this._value='';this._html='';this.textContent='';}
  set innerHTML(s){this._html=s;if(this.tagName==='SELECT'){this.options=[...s.matchAll(/<option value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/g)].map(m=>({value:decode(m[1]),text:decode(m[2]),selected:false}));if(!this.multiple&&this.options.length)this.options[0].selected=true;this._value='';}}
  get innerHTML(){return this._html;}
  set value(s){this._value=String(s);for(const o of this.options)o.selected=o.value===this._value;}
  get value(){return this.options.find(o=>o.selected)?.value??this._value;}
  get selectedOptions(){return this.options.filter(o=>o.selected);}
  click(){this.onclick?.({type:'click'});}
}

export function auditReport(html,testNegativeControls=true){
  const checks=[],check=(name,ok)=>{checks.push({name,passed:!!ok});assert.ok(ok,name);};
  const match=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/);
  assert.ok(match,'embedded data exists');const D=JSON.parse(match[1]);
  const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)],app=scripts.at(-1)[1];
  const markup=html.slice(0,match.index+match[0].indexOf('>')+1),elements=new Map();
  for(const m of markup.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)){const e=new Element(m[2],m[1],m[0].match(/class="([^"]*)"/)?.[1]);e.multiple=/\bmultiple\b/.test(m[0]);elements.set(e.id,e);}
  const get=id=>{assert.ok(elements.has(id),'known element '+id);return elements.get(id);};get('payload').textContent=match[1];get('range').innerHTML=html.match(/<select id="range">([\s\S]*?)<\/select>/)[1];
  const nav=[...markup.matchAll(/<button data-page="([^"]+)"[^>]*>/g)].map(m=>{const b=new Element('','button',m[0].match(/class="([^"]*)"/)?.[1]);b.dataset.page=m[1];return b;}),sections=[...markup.matchAll(/<section id="([^"]+)"[^>]*>/g)].map(m=>get(m[1])),body=new Element('body','body');
  const document={body,getElementById:get,querySelectorAll(s){if(s==='nav button[data-page]')return nav;if(s==='nav button[data-page],main section')return [...nav,...sections];throw Error(s);}};
  const plots=new Map(),Plotly={react(id,data,layout,config){assert.ok(config.responsive);for(const trace of data){assert.ok(Array.isArray(trace.x)&&Array.isArray(trace.y));assert.equal(trace.x.length,trace.y.length,'paired trace arrays');}plots.set(id,{data,layout});return Promise.resolve();}};
  let resized=0;const window={dispatchEvent(){resized++;}};
  new vm.Script(app,{filename:'actual-report-application.js'}).runInContext(vm.createContext({document,Plotly,window,Event:class{},console}),{timeout:10000});
  check('actual application initialized',!!window.REPORT);
  check('one global case dropdown',(html.match(/<select id="case">/g)||[]).length===1);
  check('four requested physical cases',JSON.stringify(get('case').options.map(o=>o.value).sort())===JSON.stringify(['euler_annular/shock35','euler_annular/shock70','euler_nozzle/shock_35','euler_nozzle/shock_70'].sort()));
  check('Plotly figures only',!/<img\b|<figure\b/i.test(markup));
  check('offline resources',!/<script[^>]+src\s*=|<link[^>]+href\s*=/i.test(markup));
  check('eight actual Plotly targets',['field0','field1','field2','field3','residual','cfl','work','refinementPlot'].every(id=>plots.has(id)));
  check('compilation diagnostic is not a timing claim',markup.includes('No matched performance timings are claimed.'));
  const auditTop=D.audits['fresh_audit.json']||D.audits['audit.json']||{},findingsValid=D.findings?.verified&&auditTop.complete===true&&['evaluation_epoch_stable','audit_valid','snapshot_source_match','snapshot_client_match','snapshot_case_match'].every(key=>auditTop[key]!==false);if(findingsValid){check('bound matrix findings visible beside profiles',get('findings').style.display==='block'&&D.findings.statements.every(text=>get('findings').innerHTML.includes(text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'))));}else check('incomplete or invalid audit suppresses complete-matrix claims',get('findings').style.display==='none'&&get('findings').innerHTML==='');
  if(D.epoch_recovery?.statement)check('execution and evaluation epoch scope visible',get('epochRecovery').textContent===D.epoch_recovery.statement);
  if(D.current_focused_status)check('evaluation test scope visible',get('focusedTests').textContent===D.current_focused_status);
  check('one declared public flux retained',D.records.every(r=>r.missing||r.flux===D.public_flux));
  if(D.records.length)check('initial selection contains saved attempts',window.REPORT.selected().length>0);
  const specifications=Object.fromEntries(D.report_profile.configurations.map(item=>[item.id,item.parameters]));
  check('configuration parameters match metadata',D.records.every(r=>r.missing||Object.entries(specifications[r.variant]).every(([k,value])=>JSON.stringify(r.parameters[k])===JSON.stringify(value))));
  if(auditTop.evaluation_epoch_stable===false||auditTop.audit_valid===false)check('invalid evaluation epoch is visible and never validated',!get('rows').innerHTML.includes('>validated<')&&get('rows').innerHTML.includes('failed aggregate audit'));
  const selected=()=>D.records.filter(r=>r.case===get('case').value&&r.cells===Number(get('cells').value)&&r.reconstruction===get('reconstruction').value&&String(r.alignment)===get('alignment').value&&get('variant').selectedOptions.some(o=>o.value===r.variant));
  function verify(){
    const rows=selected();assert.equal((get('rows').innerHTML.match(/<tr class=/g)||[]).length,rows.length,'table has exactly selected attempts');
    assert.ok(new Set(rows.map(r=>r.comparison_identity)).size<=1,'a comparison shares source/physical/common control identity');
    for(const [i,key] of ['p','velocity','T','h0'].entries()){
      const traces=plots.get('field'+i).data,names=new Set();
      for(const r of rows){if(!r.profile.fields[key])continue;const name=(D.labels[r.variant]||r.variant)+(r.success?'':' [unsuccessful]');names.add(name);const trace=traces.find(t=>t.name===name);assert.ok(trace);assert.equal(JSON.stringify(trace.x),JSON.stringify(r.profile.x),'coordinates remain exact');assert.equal(JSON.stringify(trace.y),JSON.stringify(r.profile.fields[key]),'profile values remain exact');}
      const base=rows.find(r=>r.profile.x.length)||rows[0],projection=traces.find(t=>t.name==='Conservative cell projection');
      if(base?.reference_cell.fields[key]){assert.ok(projection);assert.equal(JSON.stringify(projection.x),JSON.stringify(base.reference_cell.x),'projection coordinates exact');assert.equal(JSON.stringify(projection.y),JSON.stringify(base.reference_cell.fields[key]),'analytical conservative projection exact');}
      assert.ok(traces.every(t=>names.has(t.name)||['Analytical point branches','Conservative cell projection'].includes(t.name)),'no profile from another selection');
    }
    for(const [id,key] of [['residual','overall_relative'],['cfl','cfl']]){const traces=plots.get(id).data;assert.equal(traces.length,rows.length);rows.forEach((r,i)=>{assert.equal(JSON.stringify(traces[i].x),JSON.stringify(r.history.map(h=>h.step)));assert.equal(JSON.stringify(traces[i].y),JSON.stringify(r.history.map(h=>h[key])),'actual accepted history retained');});}
    for(const [i,key] of ['accepted_steps','rejected_attempts','newton_corrections'].entries())assert.equal(JSON.stringify(plots.get('work').data[i].y),JSON.stringify(rows.map(r=>r.work[key]??null)),'complete work totals retained');
    for(const trace of plots.get('refinementPlot').data){const expected=D.records.filter(r=>r.case===get('case').value&&r.reconstruction===get('reconstruction').value&&String(r.alignment)===get('alignment').value&&r.variant===trace.meta.configuration_id).sort((a,b)=>a.cells-b.cells);assert.equal(JSON.stringify(trace.x),JSON.stringify(expected.map(r=>r.cells)),'refinement meshes exact');assert.equal(JSON.stringify(trace.y),JSON.stringify(expected.map(r=>r.errors[get('errorMetric').value]??null)),'saved analytical refinement errors exact');const top=D.audits['fresh_audit.json']||D.audits['audit.json']||{};for(const [i,r] of expected.entries()){const a=r.fresh_audit,identity=r.audit_required_true||D.audit_required_true,knownFailure=!r.success||r.stale_audit||a?.error||a?.all_checks_passed===false||identity.some(k=>a?.[k]===false)||top.snapshot_source_match===false||top.snapshot_client_match===false||top.snapshot_case_match===false||top.evaluation_epoch_stable===false||top.audit_valid===false;const valid=!!a&&!r.stale_audit&&!a.error&&a.all_checks_passed===true&&identity.every(k=>a[k]===true)&&top.snapshot_source_match!==false&&top.snapshot_client_match!==false&&top.snapshot_case_match!==false&&top.evaluation_epoch_stable!==false&&top.audit_valid!==false;assert.equal(trace.marker.symbol[i],knownFailure?'x':valid?'circle':'diamond-open','fresh identity/physics audit controls refinement acceptance marker');}}
    return rows;
  }
  let selections=0;
  for(const c of D.cases){get('case').value=c;get('case').onchange();for(const n of get('cells').options.map(o=>o.value)){get('cells').value=n;for(const order of get('reconstruction').options.map(o=>o.value)){get('reconstruction').value=order;for(const a of get('alignment').options.map(o=>o.value)){get('alignment').value=a;get('alignment').onchange();verify();selections++;for(const v of get('variant').options.map(o=>o.value)){get('variant').value=v;get('variant').onchange();verify();}for(const o of get('variant').options)o.selected=true;get('variant').onchange();get('range').value='shock';get('range').onchange();const base=verify().find(r=>r.profile.x.length);if(base?.shock_position!=null&&base.profile.x.length>1)assert.equal(plots.get('field0').layout.xaxis.range.length,2);get('range').value='domain';get('range').onchange();}}}for(const metric of get('errorMetric').options.map(o=>o.value)){get('errorMetric').value=metric;get('errorMetric').onchange();for(const trace of plots.get('refinementPlot').data)assert.equal(trace.x.length,trace.y.length);}check('case-scoped profiles, histories and work '+c,true);}
  for(const b of nav){b.click();check('navigation '+b.dataset.page,get(b.dataset.page).classList.contains('active')&&sections.filter(s=>s.classList.contains('active')).length===1);}
  check('navigation requests Plotly resize',resized>=nav.length);get('theme').click();check('dark theme reaches Plotly',plots.get('field0').layout.font.color==='#e5ebf4');
  const zip=Buffer.from(html.match(/data:application\/zip;base64,([^"]+)/)[1],'base64');check('embedded downloadable evidence',zip.subarray(0,2).toString()==='PK');
  check('original evidence hashes retained',D.evidence_manifest.length>0&&get('manifest').textContent.includes(D.evidence_manifest[0].sha256));
  for(const [name,item] of Object.entries(D.supplemental_evidence||{})){const original=D.evidence_manifest.find(entry=>entry.path===name);assert.equal(original?.sha256,item.sha256,'supplemental filename and SHA bound to original manifest');assert.equal(original.bytes,item.bytes,'supplemental byte count bound');assert.ok(get('supplemental').textContent.includes(item.sha256),'supplemental hashes visible separately from fresh row checks');}
  check('supplemental evidence is distinct from fresh row audits',markup.includes('not fresh saved-row Operator evaluations'));
  if(testNegativeControls){for(const key of ['evaluation_epoch_stable','audit_valid','complete']){const invalid=JSON.parse(match[1]),top=invalid.audits['fresh_audit.json']||invalid.audits['audit.json'];top[key]=false;const replacement=JSON.stringify(invalid).replace(/</g,'\u003c'),negative=html.replace(match[0],'<script id="payload" type="application/json">'+replacement+'</script>');const result=auditReport(negative,false);check('negative aggregate audit control '+key,result.passed);}}
  return {passed:checks.every(c=>c.passed),checks,study:D.study,case_count:D.cases.length,record_count:D.records.length,selection_count:selections,browser_rendering_verified:false,scope:'Actual inline JavaScript with DOM/Plotly stubs: dropdown scope, exact saved arrays and histories, work and navigation.'};
}

export function auditFile(input,output){const html=fs.readFileSync(input,'utf8'),result=auditReport(html);result.report_sha256=createHash('sha256').update(html).digest('hex');if(output)fs.writeFileSync(output,JSON.stringify(result,null,2));return result;}
if(process.argv[2]){const file=process.argv[2],output=process.argv[3]||file.replace(/REPORT\.html$/,'outputs/verification/report_logic_audit.json');console.log(JSON.stringify(auditFile(file,output),null,2));}
