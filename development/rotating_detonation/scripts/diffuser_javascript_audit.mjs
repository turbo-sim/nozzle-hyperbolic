/** Exercise the offline report controls using a DOM stand-in, without a browser. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {Blob} from 'node:buffer';

const html = await fs.readFile(new URL('../REPORT.html', import.meta.url), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert.equal(scripts.length, 2);
for (const source of scripts) new vm.Script(source);
const markup = html.split('<script>')[0];
function classList(initial = '') {
  const values = new Set(initial.split(/\s+/).filter(Boolean));
  return {contains: value => values.has(value), toggle(value, force) {
    const add = force === undefined ? !values.has(value) : force;
    if (add) values.add(value); else values.delete(value);
    return add;
  }};
}
const sections = [...markup.matchAll(/<section id="([^"]+)" class="([^"]*)">([\s\S]*?)<\/section>/g)]
  .map(m => ({id: m[1], classList: classList(m[2]), begin: m.index, end: m.index + m[0].length}));
const elements = new Map();
for (const match of markup.matchAll(/<([a-zA-Z0-9]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const attributes = Object.fromEntries([...match[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(a => [a[1], a[2]]));
  const owner = sections.find(s => match.index >= s.begin && match.index < s.end);
  const element = {id: match[3], value: attributes.value || '0', max: attributes.max,
    src: attributes.src, textContent: '', handlers: {}, classList: classList(attributes.class),
    addEventListener(type, handler) {this.handlers[type] = handler;}};
  Object.defineProperty(element, 'offsetParent', {get() {
    return !owner || owner.classList.contains('active') ? {} : null;
  }});
  elements.set(element.id, element);
}
elements.get('frame-field').value = 'M';
const buttons = [...markup.matchAll(/<button data-tab="([^"]+)" class="([^"]*)">/g)]
  .map(m => ({dataset: {tab: m[1]}, classList: classList(m[2])}));
const imageButtons = [...markup.matchAll(/<button data-save-image="([^"]+)" data-filename="([^"]+)">/g)]
  .map(m => ({dataset: {saveImage: m[1], filename: m[2]}}));
const rendered = [], downloads = [], blobs = [];
const document = {body: {classList: classList()}, getElementById: id => elements.get(id) || null,
  querySelectorAll(selector) {
    return selector === 'section' ? sections : selector === '[data-tab]' ? buttons
      : selector === '[data-save-image]' ? imageButtons : [];
  }, createElement(tag) {
    assert.equal(tag, 'a');
    return {click() {downloads.push({filename: this.download, href: this.href});}};
  }};
const context = vm.createContext({document, Plotly: {react(id, data, layout) {
  assert(elements.has(id));
  for (const trace of data) {
    assert.equal(trace.x.length, trace.y.length);
    assert(trace.y.every(Number.isFinite));
  }
  rendered.push({id, color: layout.paper_bgcolor});
}}, Blob, Uint8Array, atob: s => Buffer.from(s, 'base64').toString('binary'),
URL: {createObjectURL(blob) {blobs.push(blob); return 'blob:offline-audit';}, revokeObjectURL() {}},
setTimeout: callback => callback()});
vm.runInContext(scripts.at(-1), context, {timeout: 30000});
assert.equal(buttons.length, 6);
for (const button of buttons) {
  button.onclick();
  assert(sections.find(s => s.id === button.dataset.tab).classList.contains('active'));
}
buttons.find(b => b.dataset.tab === 'steady').onclick();
elements.get('theme').onclick();
assert(document.body.classList.contains('dark'));
assert.equal(rendered.at(-1).color, '#211a2d');
buttons.find(b => b.dataset.tab === 'forced').onclick();
const slider = elements.get('frame');
slider.value = slider.max;
slider.handlers.input();
assert.match(elements.get('frame-time').textContent, /ms$/);
for (const key of ['p', 'M', 'M_m', 'T', 'v_m', 'alpha_deg', 'mdot']) {
  elements.get('frame-field').value = key;
  elements.get('frame-field').handlers.change();
}
for (const button of imageButtons) button.onclick();
assert.equal(imageButtons.length, 3);
assert(imageButtons.some(b => b.dataset.filename === 'startup.gif'));
assert(imageButtons.some(b => b.dataset.filename === 'late_transient.gif' || b.dataset.filename === 'periodic.gif'));
elements.get('download-data').onclick();
assert.equal(blobs.at(-1).type, 'application/gzip');
assert(downloads.some(d => d.filename === 'annular_diffuser_evidence.json.gz'));
const result = {passed: true, mode: 'JavaScript VM with a DOM stand-in; no browser rendering',
  embedded_scripts: scripts.length, chapters: buttons.length, render_calls: rendered.length,
  image_downloads: imageButtons.length, theme_checked: true, frame_slider_checked: true,
  all_field_selectors_checked: true, evidence_download_checked: true};
await fs.writeFile(new URL('../outputs/diffuser/javascript_audit.json', import.meta.url), JSON.stringify(result, null, 2));
console.log(result);
