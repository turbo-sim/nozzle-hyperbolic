/** Offline JavaScript behavior smoke test with a small DOM stand-in. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {Blob} from 'node:buffer';

const html = await fs.readFile(new URL('../REPORT.html', import.meta.url), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
for (let i=0; i<scripts.length; i++) new vm.Script(scripts[i], {filename:`embedded-${i}.js`});
const markup = html.split('<script>')[0];
function classList(initial='') {
  const values = new Set(initial.split(/\s+/).filter(Boolean));
  return { contains: x => values.has(x), toggle(x, force) {
    const add = force === undefined ? !values.has(x) : force;
    if(add) values.add(x); else values.delete(x);
    return add;
  }};
}
const sections = [...markup.matchAll(/<section id="([^"]+)" class="([^"]*)">([\s\S]*?)<\/section>/g)].map(m => ({id:m[1], classList:classList(m[2]), begin:m.index, end:m.index+m[0].length}));
const elements = new Map();
for (const m of markup.matchAll(/<([a-zA-Z0-9]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const attributes = Object.fromEntries([...m[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(a=>[a[1],a[2]]));
  const owner = sections.find(s=>m.index>=s.begin && m.index<s.end);
  const element = {id:m[3], value:attributes.value || '0', max:attributes.max, src:attributes.src, textContent:'', handlers:{}, classList:classList(attributes.class), addEventListener(type, fn){this.handlers[type]=fn;}};
  Object.defineProperty(element,'offsetParent',{get(){return !owner || owner.classList.contains('active') ? {} : null;}});
  elements.set(element.id,element);
}
const buttons = [...markup.matchAll(/<button data-tab="([^"]+)" class="([^"]*)">/g)].map(m=>({dataset:{tab:m[1]},classList:classList(m[2])}));
const imageButtons = [...markup.matchAll(/<button data-save-image="([^"]+)" data-filename="([^"]+)">/g)].map(m=>({dataset:{saveImage:m[1],filename:m[2]}}));
const rendered=[],downloads=[],blobs=[];
const document = {body:{classList:classList()},getElementById:id=>elements.get(id)||null,querySelectorAll(selector){return selector==='section'?sections:selector==='[data-tab]'?buttons:selector==='[data-save-image]'?imageButtons:[];},createElement(tag){assert.equal(tag,'a');return {click(){downloads.push({filename:this.download,href:this.href});}};}};
const context = vm.createContext({document,Plotly:{react(id,data,layout){assert(elements.has(id));for(const t of data){assert.equal(t.x.length,t.y.length);}rendered.push({id,color:layout.paper_bgcolor});}},Blob,Uint8Array,atob:s=>Buffer.from(s,'base64').toString('binary'),URL:{createObjectURL(blob){blobs.push(blob);return 'blob:offline-test';},revokeObjectURL(){}},setTimeout:fn=>fn()});
vm.runInContext(scripts.at(-1),context,{timeout:20000});
assert.equal(buttons.length,6);
for(const button of buttons){button.onclick();assert(sections.find(s=>s.id===button.dataset.tab).classList.contains('active'));}
buttons.find(b=>b.dataset.tab==='steady').onclick();
elements.get('theme').onclick();assert(document.body.classList.contains('dark'));assert.equal(rendered.at(-1).color,'#211c2d');
buttons.find(b=>b.dataset.tab==='transient').onclick();
const slider=elements.get('frame');slider.value=slider.max;slider.handlers.input();assert.match(elements.get('frame-time').textContent,/ms$/);
for(const button of imageButtons) button.onclick();
elements.get('download').onclick();assert.equal(blobs.at(-1).type,'application/gzip');assert(downloads.some(d=>d.filename==='rotating_detonation_results.json.gz'));
assert.deepEqual(imageButtons.map(b=>b.dataset.filename).sort(),['periodic.gif','startup.gif','steady_contours.png']);
const result={passed:true,mode:'VM with a DOM stand-in; no browser rendering',embedded_scripts: scripts.length,navigation_tabs:buttons.length,render_calls:rendered.length,image_downloads:imageButtons.length,theme_checked:true,slider_checked:true,archive_download_checked:true};
await fs.writeFile(new URL('../outputs/javascript_audit.json',import.meta.url),JSON.stringify(result,null,2));
console.log(result);
