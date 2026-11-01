/** Execute the actual saved-data UI with DOM/Plotly stubs; no browser or solver. */
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mountReport,AuditEvent} from './report_dom.mjs';

const same=(a,b,message)=>assert.equal(JSON.stringify(a),JSON.stringify(b),message);
const active=(mount,navId)=>(mount.document.getElementById(navId)||mount.document.querySelector(navId)).querySelectorAll('button[aria-controls]').find(button=>button.getAttribute('aria-selected')==='true');
const change=(node,value)=>{node.value=value;node.dispatchEvent(new AuditEvent('change',{bubbles:true}));};
const stripScripts=html=>html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'');

export function auditShell(mount,{navId='report-tabs',panelSelector='.report-page'}={}){
  const {document,get,window,storage,html,scripts}=mount,checks=[];
  const check=(name,ok)=>{assert.ok(ok,name);checks.push({name,passed:true});};
  const nav=document.getElementById(navId)||document.querySelector(navId);assert.ok(nav,'primary tablist');const buttons=nav.querySelectorAll('button[aria-controls]'),panels=document.querySelectorAll(panelSelector);
  same(buttons.map(button=>button.textContent.trim().replace(/\s+/g,' ')),['Overview','Case explorer','Theory','Protocol & evidence'],'common primary navigation labels/order');
  check('four shared report pages',buttons.length===4&&panels.length===4);
  const navVisible=()=>{const selected=buttons.filter(button=>button.getAttribute('aria-selected')==='true');return selected.length===1&&panels.filter(panel=>!panel.hidden).length===1&&!get(selected[0].getAttribute('aria-controls')).hidden;};
  for(const button of buttons){button.click();check('primary navigation '+button.textContent.trim(),navVisible());}
  buttons.at(-1).dispatchEvent(new AuditEvent('keydown',{key:'Home'}));check('primary keyboard Home',active(mount,navId)===buttons[0]&&document.activeElement===buttons[0]&&navVisible());
  buttons[0].dispatchEvent(new AuditEvent('keydown',{key:'ArrowLeft'}));check('primary keyboard wraps',active(mount,navId)===buttons.at(-1)&&navVisible());
  buttons.at(-1).dispatchEvent(new AuditEvent('keydown',{key:'ArrowRight'}));check('primary keyboard next',active(mount,navId)===buttons[0]&&navVisible());
  const theme=get('report-theme');same(theme.options.map(option=>option.value),['auto','light','dark'],'shared theme modes');
  for(const mode of ['dark','light','auto']){change(theme,mode);check('theme preference '+mode,storage.get('nh-flux-report-theme')===mode&&document.documentElement.dataset.theme===(mode==='auto'?'light':mode));}
  check('single shared theme runtime',scripts.filter(script=>script.id==='report-theme-runtime').length===1&&!!window.FluxReportUI);
  check('common style embedded',/<style id="report-theme-css">[\s\S]+?<\/style>/.test(html));
  check('offline math runtime and license',(scripts.some(script=>script.id==='mathjax-runtime')&&scripts.some(script=>script.id==='mathjax-license'))||document.querySelectorAll('math').length>=7);
  check('offline resource packaging',!/<script[^>]+src\s*=|<link[^>]+(?:rel=["']stylesheet["'][^>]+href|href\s*=)/i.test(stripScripts(html)));
  check('explicit hidden page panels',panels.every(panel=>panel.getAttribute('role')==='tabpanel'));
  const download=get('download-toggle'),menu=get('download-menu');download.click();check('header downloads opens',menu.hidden===false);document.body.click();check('header downloads closes',menu.hidden===true);
  return checks;
}

export function auditReport(html,testNegativeControls=true){
  const mount=mountReport(html),{get,document,plots,window,payload:D}=mount,checks=[];
  const check=(name,ok)=>{assert.ok(ok,name);checks.push({name,passed:true});};
  check('actual application initialized',!!window.REPORT);
  checks.push(...auditShell(mount));
  check('one global case dropdown',(html.match(/<select id="case"[^>]*>/g)||[]).length===1);
  same(get('case').options.map(option=>option.value).sort(),['euler_annular/shock35','euler_annular/shock70','euler_nozzle/shock_35','euler_nozzle/shock_70'].sort(),'four requested physical cases');
  check('Plotly figures only',document.querySelectorAll('img,figure').length===0);
  check('compilation diagnostic is not a timing claim',html.includes('No matched performance timings are claimed.'));
  const outer=get('report-tabs').querySelectorAll('button[aria-controls]'),caseButton=outer.find(button=>button.dataset.page==='case'),inner=get('case-tabs').querySelectorAll('button[aria-controls]');
  const visit=name=>{caseButton.click();inner.find(button=>button.getAttribute('aria-controls')===name).click();};
  const innerVisible=()=>inner.filter(button=>button.getAttribute('aria-selected')==='true').length===1&&document.querySelectorAll('.case-page').filter(panel=>!panel.hidden).length===1;
  for(const button of inner){button.click();check('case navigation '+button.getAttribute('aria-controls'),innerVisible()&&!get(button.getAttribute('aria-controls')).hidden);}
  inner.at(-1).dispatchEvent(new AuditEvent('keydown',{key:'ArrowRight'}));check('case keyboard wraps',active(mount,'case-tabs')===inner[0]&&innerVisible());
  outer[0].click();check('case controls hidden outside explorer',!!get('case').closest('[hidden]'));caseButton.click();
  const top=D.audits['fresh_audit.json']||D.audits['audit.json']||{};
  const findingsValid=D.findings?.verified&&top.complete===true&&['evaluation_epoch_stable','audit_valid','snapshot_source_match','snapshot_client_match','snapshot_case_match'].every(key=>top[key]!==false);
  if(findingsValid)check('complete verified findings retained',!get('findings').hidden&&D.findings.statements.every(text=>get('findings').textContent.includes(text)));
  else check('invalid or incomplete evidence suppresses matrix claims',get('findings').hidden&&get('findings').innerHTML==='');
  check('one declared public flux retained',D.records.every(row=>row.missing||row.flux===D.public_flux));
  const specifications=Object.fromEntries(D.report_profile.configurations.map(item=>[item.id,item.parameters]));
  check('configuration parameters match metadata',D.records.every(row=>row.missing||Object.entries(specifications[row.variant]).every(([key,value])=>JSON.stringify(row.parameters[key])===JSON.stringify(value))));
  const selected=()=>D.records.filter(row=>row.case===get('case').value&&row.cells===Number(get('cells').value)&&row.reconstruction===get('reconstruction').value&&String(row.alignment)===get('alignment').value&&get('variant').selectedOptions.some(option=>option.value===row.variant));
  function verify(){
    const rows=selected();assert.equal(get('rows').querySelectorAll('tr').length,rows.length,'table has exactly selected attempts');assert.ok(new Set(rows.map(row=>row.comparison_identity)).size<=1,'matched source/physical/common controls');
    visit('profiles');
    for(const [i,key] of ['p','velocity','T','h0'].entries()){
      const traces=plots.get('field'+i).data,names=new Set();
      for(const row of rows){if(!row.profile.fields[key])continue;const name=(D.labels[row.variant]||row.variant)+(row.success?'':' [unsuccessful]');names.add(name);const trace=traces.find(trace=>trace.name===name);assert.ok(trace,'selected profile exists');same(trace.x,row.profile.x,'profile coordinates exact');same(trace.y,row.profile.fields[key],'profile values exact');}
      const base=rows.find(row=>row.profile.x.length)||rows[0],projection=traces.find(trace=>trace.name==='Conservative cell projection');if(base?.reference_cell.fields[key]){assert.ok(projection);same(projection.x,base.reference_cell.x,'projection coordinates exact');same(projection.y,base.reference_cell.fields[key],'conservative projection exact');}
      assert.ok(traces.every(trace=>names.has(trace.name)||['Analytical point branches','Conservative cell projection'].includes(trace.name)),'no foreign profile');
    }
    visit('convergence');
    for(const [id,key] of [['residual','overall_relative'],['cfl','cfl']]){const traces=plots.get(id).data;assert.equal(traces.length,rows.length);rows.forEach((row,i)=>{same(traces[i].x,row.history.map(item=>item.step),'actual history abscissa');same(traces[i].y,row.history.map(item=>item[key]),'actual accepted history');});}
    for(const [i,key] of ['accepted_steps','rejected_attempts','newton_corrections'].entries())same(plots.get('work').data[i].y,rows.map(row=>row.work[key]??null),'complete work totals');
    visit('refinement');
    for(const trace of plots.get('refinementPlot').data){const expected=D.records.filter(row=>row.case===get('case').value&&row.reconstruction===get('reconstruction').value&&String(row.alignment)===get('alignment').value&&row.variant===trace.meta.configuration_id).sort((a,b)=>a.cells-b.cells);same(trace.x,expected.map(row=>row.cells),'refinement meshes exact');same(trace.y,expected.map(row=>row.errors[get('errorMetric').value]??null),'saved analytical errors exact');for(const [i,row] of expected.entries()){const audit=row.fresh_audit,identity=row.audit_required_true||D.audit_required_true,failed=!row.success||row.stale_audit||audit?.error||audit?.all_checks_passed===false||identity.some(key=>audit?.[key]===false)||['snapshot_source_match','snapshot_client_match','snapshot_case_match','evaluation_epoch_stable','audit_valid'].some(key=>top[key]===false),valid=!!audit&&!row.stale_audit&&!audit.error&&audit.all_checks_passed===true&&identity.every(key=>audit[key]===true)&&['snapshot_source_match','snapshot_client_match','snapshot_case_match','evaluation_epoch_stable','audit_valid'].every(key=>top[key]!==false);assert.equal(trace.marker.symbol[i],failed?'x':valid?'circle':'diamond-open','audit governs acceptance marker');}}
    return rows;
  }
  let selections=0;
  for(const caseId of D.cases){change(get('case'),caseId);for(const cells of get('cells').options.map(option=>option.value)){change(get('cells'),cells);for(const order of get('reconstruction').options.map(option=>option.value)){change(get('reconstruction'),order);for(const alignment of get('alignment').options.map(option=>option.value)){change(get('alignment'),alignment);verify();selections++;for(const variant of get('variant').options.map(option=>option.value)){change(get('variant'),variant);verify();}get('config-all').click();change(get('range'),'shock');const base=verify().find(row=>row.profile.x.length);if(base?.shock_position!=null&&base.profile.x.length>1)assert.equal(plots.get('field0').layout.xaxis.range.length,2);change(get('range'),'domain');}}}for(const metric of get('errorMetric').options.map(option=>option.value)){change(get('errorMetric'),metric);visit('refinement');for(const trace of plots.get('refinementPlot').data)assert.equal(trace.x.length,trace.y.length);}check('case-scoped profiles, histories, work and errors '+caseId,true);}
  get('config-none').click();check('deselect all retains empty selection',window.REPORT.selected().length===0);get('config-all').click();check('select all restores configurations',window.REPORT.selected().length===D.variants.length);
  const checkbox=get('configuration-controls').querySelector('input[data-configuration]');assert.ok(checkbox);const before=window.REPORT.selected().length;checkbox.click();check('visible configuration checkbox changes selection',window.REPORT.selected().length===before-1);checkbox.click();
  visit('profiles');change(get('report-theme'),'dark');check('dark theme reaches Plotly',plots.get('field0').layout.font.color===window.FluxReportUI.palette().ink);change(get('report-theme'),'light');check('light theme reaches Plotly',plots.get('field0').layout.font.color===window.FluxReportUI.palette().ink);
  const zip=Buffer.from(html.match(/data:application\/zip;base64,([^"\s]+)/)[1],'base64');check('embedded evidence archive',zip.subarray(0,2).toString()==='PK');check('original evidence hashes retained',D.evidence_manifest.length>0&&get('manifest').textContent.includes(D.evidence_manifest[0].sha256));
  for(const [name,item] of Object.entries(D.supplemental_evidence||{})){const original=D.evidence_manifest.find(entry=>entry.path===name);assert.equal(original?.sha256,item.sha256,'supplement hash');assert.equal(original.bytes,item.bytes,'supplement bytes');}
  if(top.evaluation_epoch_stable===false||top.audit_valid===false)check('invalid epoch never promoted',!get('rows').innerHTML.includes('>validated<')&&get('rows').innerHTML.includes('failed aggregate audit'));
  if(testNegativeControls)for(const key of ['evaluation_epoch_stable','audit_valid','complete']){const invalid=JSON.parse(JSON.stringify(D)),aggregate=invalid.audits['fresh_audit.json']||invalid.audits['audit.json'];aggregate[key]=false;const replacement=JSON.stringify(invalid).replace(/</g,'\\u003c'),negative=html.replace(/(<script id="payload" type="application\/json">)[\s\S]*?(<\/script>)/,(_m,open,close)=>open+replacement+close);check('negative aggregate control '+key,auditReport(negative,false).passed);}
  return {passed:true,checks,study:D.study,record_count:D.records.length,case_count:D.cases.length,selection_count:selections,browser_rendering_verified:false,scope:'Actual presentation JavaScript, deterministic DOM/Plotly stubs; exact saved arrays, controls and evidence states. Browser layout and MathJax SVG rendering need separate visual QA.'};
}
export function auditFile(input,output){const html=fs.readFileSync(input,'utf8'),result=auditReport(html);result.report_sha256=createHash('sha256').update(html).digest('hex');if(output){fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');}return result;}
if(typeof process!=='undefined'&&process.argv[1]?.endsWith('audit_report.mjs')&&process.argv[2]){const input=process.argv[2],output=process.argv[3]||(input.endsWith('REPORT.html')?input.slice(0,-11)+'outputs/verification/report_logic_audit.json':input+'.audit.json');console.log(JSON.stringify(auditFile(input,output),null,2));}
