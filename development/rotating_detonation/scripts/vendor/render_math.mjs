/** Prerender the retained LaTeX with the vendored MathJax3 package.
 * Usage: node --input-type=module -e "import('./render_math.mjs').then(m=>m.render())"
 * No installation, network request, browser or unpacked dependency tree is used.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import vm from 'node:vm';

const sha256=data=>createHash('sha256').update(data).digest('hex');
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');

export async function render(directory=fileURLToPath(new URL('.',import.meta.url))){
  const filename='mathjax-full-3.2.2.tgz';
  const archive=await fs.readFile(path.join(directory,filename));
  const tar=gunzipSync(archive), files=new Map();
  for(let cursor=0;cursor+512<=tar.length;){
    const header=tar.subarray(cursor,cursor+512);
    const name=header.subarray(0,100).toString().replace(/\0.*$/s,'');
    if(!name)break;
    const size=parseInt(header.subarray(124,136).toString().replace(/\0.*$/s,'').trim()||'0',8);
    const content=tar.subarray(cursor+512,cursor+512+size);
    if(name.startsWith('package/js/')||name==='package/LICENSE')files.set(name,content.toString());
    cursor+=512+Math.ceil(size/512)*512;
  }
  const cache=new Map();
  function requireModule(name){
    if(cache.has(name))return cache.get(name).exports;
    const source=files.get(name);
    if(source===undefined)throw new Error('Vendored MathJax module unavailable: '+name);
    const module={exports:{}};cache.set(name,module);
    const require=relative=>{
      if(!relative.startsWith('.'))throw new Error('Unexpected external MathJax dependency: '+relative);
      let resolved=path.posix.normalize(path.posix.join(path.posix.dirname(name),relative));
      if(!path.posix.extname(resolved))resolved+='.js';
      return requireModule(resolved);
    };
    const wrapper=new vm.Script('(function(module,exports,require,__filename,__dirname){\n'+source+'\n})',{filename:name}).runInThisContext();
    wrapper(module,module.exports,require,name,path.posix.dirname(name));
    return module.exports;
  }
  const {mathjax}=requireModule('package/js/mathjax.js');
  const {TeX}=requireModule('package/js/input/tex.js');
  const {SVG}=requireModule('package/js/output/svg.js');
  const {liteAdaptor}=requireModule('package/js/adaptors/liteAdaptor.js');
  const {RegisterHTMLHandler}=requireModule('package/js/handlers/html.js');
  requireModule('package/js/input/tex/ams/AmsConfiguration.js');
  const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
  const document=mathjax.document('',{InputJax:new TeX({packages:['base','ams']}),OutputJax:new SVG({fontCache:'none'})});
  const source=await fs.readFile(path.join(directory,'equations.json'));
  const equations=JSON.parse(source.toString());
  const manifest={engine:'MathJax',version:'3.2.2',format:'prerendered self-contained SVG, fontCache:none',
    source_url:'https://registry.npmjs.org/mathjax-full/-/mathjax-full-3.2.2.tgz',
    package_file:filename,package_sha256:sha256(archive),latex_source_file:'equations.json',
    latex_source_sha256:sha256(source),renderer_file:'render_math.mjs',
    renderer_sha256:sha256(await fs.readFile(path.join(directory,'render_math.mjs'))),formulas:{}};
  for(const [key,tex] of Object.entries(equations)){
    const node=document.convert(tex,{display:true});
    const html=adaptor.outerHTML(node);
    if(/data-mjx-error|merror/.test(html))throw new Error('MathJax error in '+key+': '+html);
    let svg=html.slice(html.indexOf('<svg'),html.lastIndexOf('</svg>')+6);
    svg=svg.replace(/^<svg\b([^>]*)>/,(_,attributes)=>'<svg'+attributes.replace(/\s(?:role|aria-label)="[^"]*"/g,'')+' role="img" aria-label="'+escape(tex)+'">');
    if(!svg.startsWith('<svg ')||/<(?:script|image|use)\b/.test(svg))throw new Error('SVG is not self-contained: '+key);
    const file='equation-'+key+'.svg';await fs.writeFile(path.join(directory,file),svg);
    manifest.formulas[key]={tex,svg_file:file,sha256:sha256(svg)};
  }
  await fs.writeFile(path.join(directory,'MATHJAX_LICENSE.txt'),files.get('package/LICENSE'));
  await fs.writeFile(path.join(directory,'mathjax-render-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  return {formulas:Object.keys(manifest.formulas).length,package_sha256:manifest.package_sha256,engine:manifest.engine,version:manifest.version};
}
