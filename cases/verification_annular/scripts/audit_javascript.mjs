import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname,join} from 'node:path';
import vm from 'node:vm';

export async function auditReport(reportPath,outputPath){
  const html=await readFile(reportPath,'utf8');
  const app=html.match(/<script id="report-app">([\s\S]*?)<\/script>/)?.[1];
  if(!app)throw Error('Report application script is missing');
  new vm.Script(app,{filename:'annular-report-app.js'});
  const browser=JSON.parse(await readFile(join(dirname(reportPath),'outputs/browser_audit/audit.json'),'utf8'));
  const digest=createHash('sha256').update(html).digest('hex');
  const checks=browser.checks||[];
  const result={passed:browser.passed&&browser.html_sha256===digest,
    html_sha256:digest,syntax:true,pages:checks.filter(c=>c.name.startsWith('page_')&&c.passed).length,
    navigation:checks.filter(c=>c.name.startsWith('page_')).every(c=>c.passed),
    solver_visibility:checks.filter(c=>c.name.startsWith('solver_visibility_')).every(c=>c.passed),
    marker_skipping:checks.filter(c=>c.name.startsWith('marker_skip_')).every(c=>c.passed),
    theme:checks.some(c=>c.name==='dark_theme_white_marker_faces'&&c.passed),
    method:'JavaScript syntax plus matching actual headless Edge audit; no DOM stand-in'};
  await writeFile(outputPath,JSON.stringify(result,null,2));
  if(!result.passed)throw Error('Run the current headless browser audit before the JavaScript summary');
  return result;
}
