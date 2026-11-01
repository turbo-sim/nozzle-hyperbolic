/** Unified report audit: real presentation JavaScript, unchanged numerical evidence. */
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {auditReport,auditShell} from './audit_report.mjs';
import {mountReport,AuditEvent} from './report_dom.mjs';

const same=(a,b,message)=>assert.equal(JSON.stringify(a),JSON.stringify(b),message);
const sha=value=>createHash('sha256').update(value).digest('hex');
const change=(node,value)=>{node.value=value;node.dispatchEvent(new AuditEvent('change',{bubbles:true}));};
const number=value=>value===null||value===undefined||typeof value==='boolean'?null:(Number.isFinite(Number(value))?Number(value):null);
const max=values=>{const usable=values.map(number).filter(Number.isFinite);return usable.length?Math.max(...usable):null;};
const residual=sample=>{const direct=number(sample.residual??sample.spatial_rms??sample.residual_rms??sample.max_residual??sample.spatial_norm);if(direct!==null)return direct;const values=sample.equation_norm||sample.equation_norms||sample.equation_residual||sample.residuals||sample.norms;return max(Array.isArray(values)?values:Object.values(values||{}));};
const history=(row,field)=>row.history.map((sample,index)=>({x:number(sample.step??sample.iteration??sample.update??sample.accepted_step)??index,y:field==='residual'?residual(sample):number(sample.cfl??sample.CFL??sample.cfl_requested??sample.requested_cfl)})).filter(point=>Number.isFinite(point.x)&&Number.isFinite(point.y)&&point.y>0);
const close=(actual,expected,message)=>assert.ok(Math.abs(actual-expected)<=2e-13*Math.max(1,Math.abs(actual),Math.abs(expected)),message);

export function auditRoe(html,{sodRaw=null}={}){
  const mount=mountReport(html,{payloadId:'benchmark-data',applicationId:'benchmark-app'}),{window,document,get,plots,payload:D}=mount,api=window.benchmarkReport,checks=[];
  const check=(name,ok)=>{assert.ok(ok,name);checks.push({name,passed:true});};
  check('actual Roe application initialized',window.benchmarkReportReady===true&&!!api);
  checks.push(...auditShell(mount,{navId:'.page-tabs',panelSelector:'.page'}));
  check('steady recorded population retained',api.results===160&&api.cases===20&&D.campaign.results.length===160);
  check('verification scope never becomes timing evidence',D.campaign.verification_only===true&&get('campaign-subtitle').textContent.includes('Performance timing was not collected'));
  same(api.fluxes,['roe'],'sole Roe flux retained');
  const cases=get('case-select').options.map(option=>option.value);
  let selections=0,fieldSelections=0;
  function verify(){
    const rows=api.selectedRows(),field=get('field-select').value;
    assert.ok(rows.length,'selected saved attempts exist');
    assert.ok(rows.every(row=>row.caseId===get('case-select').value&&row.cells===Number(get('case-grid').value)&&String(row.cap)===get('case-cap').value),'selected case/mesh/cap scope');
    if(get('case-reconstruction').value!=='all')assert.ok(rows.every(row=>row.reconstruction===get('case-reconstruction').value),'selected reconstruction');
    const traces=plots.get('field-chart')?.data||[],numerical=traces.filter(trace=>trace.mode==='lines+markers'),expected=rows.filter(row=>row.state.fields[field]);
    assert.equal(numerical.length,expected.length,'all and only selected numerical field curves');
    expected.forEach((row,index)=>{same(numerical[index].x,row.state.x,'saved field coordinates');same(numerical[index].y,row.state.fields[field],'saved numerical values');});
    const references=traces.filter(trace=>trace.mode==='lines');
    for(const trace of references)assert.ok(rows.some(row=>JSON.stringify(trace.x)===JSON.stringify(row.reference.x)&&JSON.stringify(trace.y)===JSON.stringify(row.reference.fields[field])),'analytical curve is a saved selected reference');
    for(const [id,key] of [['residual-chart','residual'],['cfl-chart','cfl']]){const expected=rows.map(row=>history(row,key)).filter(points=>points.length),actual=plots.get(id)?.data||[];assert.equal(actual.length,expected.length,'history curve count '+key);expected.forEach((points,index)=>{same(actual[index].x,points.map(point=>point.x),'actual attempt abscissa '+key);same(actual[index].y,points.map(point=>point.y),'actual attempt history '+key);});}
    for(const trace of plots.get('refinement-chart')?.data||[])assert.equal(new Set(trace.x).size,trace.x.length,'refinement grids never mixed across cases');
    checkSelectionFlags(rows);
    return rows;
  }
  function checkSelectionFlags(rows){
    const evidence=JSON.parse(get('case-evidence').textContent);
    assert.equal(evidence.selected_results.length,rows.length,'all selected evidence records');
    evidence.selected_results.forEach((item,index)=>{const row=rows[index];assert.equal(item.status,row.status);assert.equal(item.branch_passed,row.raw.branch_passed);assert.equal(item.analytical_passed,row.raw.analytical_passed);same(item.metrics,row.metrics,'original metrics retained');same(item.errors,row.errors,'saved analytical errors retained');});
  }
  for(const caseId of cases){api.openCase(caseId);get('history-axis').value='step';for(const cells of get('case-grid').options.map(option=>option.value)){change(get('case-grid'),cells);for(const order of ['all','constant','muscl_van_leer']){change(get('case-reconstruction'),order);verify();selections++;}}change(get('case-reconstruction'),'all');for(const field of get('field-select').options.map(option=>option.value)){change(get('field-select'),field);verify();fieldSelections++;}check('case profiles, actual histories and evidence '+caseId,true);}
  change(get('report-theme'),'dark');check('Roe chart theme follows shared palette',plots.get('field-chart').layout.font.color===window.FluxReportUI.palette().ink);change(get('report-theme'),'auto');
  api.showPage('protocol');
  const pairs=D.roe_validation;assert.ok(pairs);
  check('both thermodynamic pair models remain visible',get('roe-validation-table').querySelectorAll('tr').length===pairs.pair_audits.length&&pairs.pair_audits.length===2);
  check('host derivative limitations remain explicit',get('roe-validation-jvp').textContent.includes('one-sided')&&get('roe-validation-jvp').textContent.includes('finite differences'));
  const flow=D.roe_flow_validation,sod=flow.sod,gas=flow.real_gas;
  check('all Sod criteria and attempts retained',get('roe-sod-table').querySelectorAll('tr').length===sod.records.length&&sod.records.length===12);
  check('strict Sod failures remain failures',get('roe-sod-table').querySelectorAll('tr').filter(row=>row.textContent.includes('failed; retained')).length===sod.records.filter(row=>row.passed!==true).length);
  check('all real-gas attempts retained',get('roe-real-gas-table').querySelectorAll('tr').length===gas.length&&gas.length===4);
  check('real-gas budget stops remain incomplete',get('roe-real-gas-table').querySelectorAll('tr').filter(row=>row.textContent.includes('budget stop; incomplete')).length===gas.filter(row=>row.budget_incomplete).length);
  check('real-gas steady status retained',get('roe-real-gas-table').querySelectorAll('tr').filter(row=>row.textContent.includes('steady verified')).length===gas.filter(row=>row.steady_verified).length);
  const packet=JSON.parse(get('roe-sod-data').textContent);assert.equal(packet.records.length,12);
  if(sodRaw){const raw=JSON.parse(sodRaw);assert.equal(packet.source_sha256,sha(sodRaw),'Sod display data bound to complete original JSON');same(packet.settings,raw.settings,'saved physical settings');for(const row of packet.records){const original=raw.records.find(item=>item.cells===row.cells&&item.method.id===row.method.id);assert.ok(original);for(const key of ['final_state','exact_final_conservative_cell_averages','physical_cell_centers'])same(row[key],original[key],'exact saved Sod '+key);assert.equal(row.passed,original.passed,'strict flag');same(row.classification,original.classification,'numerical criteria');}}
  let sodSelections=0;
  for(const cells of get('roe-sod-grid').options.map(option=>option.value)){
    change(get('roe-sod-grid'),cells);const rows=packet.records.filter(row=>row.cells===Number(cells)),physical=packet.settings.physical,first=rows[0],q=first.exact_final_conservative_cell_averages,left=physical.left,right=physical.right;
    const leftEnergy=q[0][2]/left.rho-.5*left.u**2,rightEnergy=q.at(-1)[2]/right.rho-.5*right.u**2,cvOverR=(leftEnergy-rightEnergy)/(left.p/left.rho-right.p/right.rho),datum=leftEnergy-cvOverR*left.p/left.rho,sound=Math.sqrt((1+1/cvOverR)*left.p/left.rho);
    const panels=['roe-sod-pressure','roe-sod-density','roe-sod-velocity'].map(id=>plots.get(id));assert.ok(panels.every(panel=>panel?.data.length===rows.length+1),'three populated saved Sod panels');
    for(const [index,row] of rows.entries()){
      const traces=panels.map(panel=>panel.data[index+1]);for(const trace of traces){same(trace.x,row.physical_cell_centers.map(x=>x/physical.length),'Sod physical coordinates');assert.equal(trace.meta.saved_cells,row.cells);assert.equal(trace.meta.saved_method,row.method.id);assert.equal(trace.meta.strict_criterion_passed,row.passed);assert.equal(trace.meta.numerical_passed,row.classification.numerical_passed);}
      row.final_state.slice(1,-1).forEach((state,cell)=>{close(traces[0].y[cell]*left.p,state[0],'Sod primitive pressure');close(traces[2].y[cell]*sound,state[1],'Sod primitive velocity');close(traces[1].y[cell]*left.rho*(state[2]-datum),(cvOverR+1)*state[0],'Sod density satisfies affine enthalpy law');});
    }
    const reference=panels.map(panel=>panel.data[0]);q.forEach((state,cell)=>{const p=reference[0].y[cell]*left.p,rho=reference[1].y[cell]*left.rho,u=reference[2].y[cell]*sound;close(rho,state[0],'Sod exact mass');close(rho*u,state[1],'Sod exact momentum');close(cvOverR*p+datum*rho+.5*rho*u*u,state[2],'Sod exact energy with retained datum');});sodSelections++;
  }
  check('three Sod grids use all saved arrays and criteria',sodSelections===3);
  check('original PNG is archival rather than visible UI',document.querySelectorAll('img').filter(element=>!element.closest('[hidden]')).length===0);
  api.showPage('theory');check('native offline Roe equations retained',get('page-theory').querySelectorAll('math').length>=7);
  return {passed:true,checks,study:'roe',record_count:160,case_count:20,selection_count:selections,field_selection_count:fieldSelections,sod_selection_count:sodSelections,browser_rendering_verified:false,scope:'Actual inline Roe JavaScript with shared DOM/Plotly stubs; numerical arrays and failure classifications are checked, no solve or EOS evaluation.'};
}

export function auditReports(root,{output=null}={}){
  root=path.resolve(root);const studies=path.join(root,'development/flux_methods'),results={},assets={};
  for(const study of ['stadtke','osher','roe']){const html=fs.readFileSync(path.join(studies,study,'REPORT.html'),'utf8');results[study]=study==='roe'?auditRoe(html,{sodRaw:fs.readFileSync(path.join(studies,'roe/outputs/roe_transient.json'),'utf8')}):auditReport(html);results[study].report_sha256=sha(html);const css=html.match(/<style id="report-theme-css">([\s\S]*?)<\/style>/)?.[1],runtime=html.match(/<script id="report-theme-runtime">([\s\S]*?)<\/script>/)?.[1];assert.ok(css&&runtime,'shared theme assets '+study);assets[study]={css_sha256:sha(css),runtime_sha256:sha(runtime)};}
  assert.equal(new Set(Object.values(assets).map(item=>item.css_sha256)).size,1,'all reports embed identical common theme CSS');assert.equal(new Set(Object.values(assets).map(item=>item.runtime_sha256)).size,1,'all reports embed identical common interaction runtime');
  const result={passed:true,reports:results,shared_assets:assets,browser_rendering_verified:false,scope:'Uniformity plus actual UI/control/data audits. Layout and browser math rendering are separate checks.'};
  for(const [key,name] of [['math_parser_evidence','math_parser_audit.json'],['preservation_evidence','preservation_audit.json']]){const file=path.join(studies,'shared/outputs/report_presentation_qa',name);if(fs.existsSync(file)){const audit=JSON.parse(fs.readFileSync(file,'utf8'));assert.ok(audit.passed,key+' passes');result[key]={file:path.relative(root,file).replaceAll('\\','/'),sha256:sha(fs.readFileSync(file)),audit};}}
  if(output){fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');}return result;
}
if(typeof process!=='undefined'&&process.argv[1]?.endsWith('audit_reports.mjs')){const root=process.argv[2]||process.cwd(),result=auditReports(root,{output:process.argv[3]||path.join(root,'development/flux_methods/shared/outputs/report_presentation_qa/unified_report_audit.json')});console.log(JSON.stringify(result,null,2));}
