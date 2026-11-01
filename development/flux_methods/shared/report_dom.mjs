/** Minimal deterministic DOM/Plotly harness for presentation audits, not layout QA. */
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {Blob} from 'node:buffer';
import {URL} from 'node:url';

const decode=value=>String(value).replace(/&quot;/g,'"').replace(/&#39;|&apos;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&').replace(/&nbsp;/g,' ');
class Classes {
  constructor(owner){this.owner=owner;this.items=new Set();}
  set(value){this.items=new Set(String(value).split(/\s+/).filter(Boolean));}
  add(...values){values.forEach(v=>this.items.add(v));}
  remove(...values){values.forEach(v=>this.items.delete(v));}
  contains(value){return this.items.has(value);}
  toggle(value,force){const on=force??!this.items.has(value);on?this.add(value):this.remove(value);return on;}
  toString(){return [...this.items].join(' ');}
}
export class AuditEvent {
  constructor(type,options={}){this.type=type;Object.assign(this,options);this.defaultPrevented=false;}
  preventDefault(){this.defaultPrevented=true;}
  stopPropagation(){this.bubbles=false;}
}
const splitSelector=s=>s.trim().split(/\s+(?![^\[]*\])/).filter(Boolean);
function atom(element,selector){
  let s=selector.trim();
  if(s==='*')return true;
  for(const no of [...s.matchAll(/:not\(([^)]+)\)/g)]){if(atom(element,no[1]))return false;s=s.replace(no[0],'');}
  const tag=s.match(/^[a-zA-Z][\w-]*/)?.[0];if(tag&&element.tagName!==tag.toUpperCase())return false;
  const id=s.match(/#([\w-]+)/)?.[1];if(id&&element.id!==id)return false;
  for(const item of s.matchAll(/\.([\w-]+)/g))if(!element.classList.contains(item[1]))return false;
  for(const item of s.matchAll(/\[([^\]=\s]+)(?:\s*=\s*["']?([^\]"']*)["']?)?\]/g)){
    const actual=element.getAttribute(item[1]);if(actual===null)return false;if(item[2]!==undefined&&actual!==item[2].trim())return false;
  }
  return true;
}
function matches(element,selector){
  return selector.split(',').some(part=>{const pieces=splitSelector(part);if(!pieces.length||!atom(element,pieces.pop()))return false;let node=element.parentElement;while(pieces.length){const wanted=pieces.pop();while(node&&!atom(node,wanted))node=node.parentElement;if(!node)return false;node=node.parentElement;}return true;});
}
const voidTags=new Set(['AREA','BASE','BR','COL','EMBED','HR','IMG','INPUT','LINK','META','PARAM','SOURCE','TRACK','WBR']);
export class AuditElement {
  constructor(tag='div',owner=null){
    this.tagName=tag.toUpperCase();this.ownerDocument=owner;this.parentElement=null;this.children=[];this._attributes=new Map();this.dataset={};this.classList=new Classes(this);this.style={setProperty(k,v){this[k]=String(v);},getPropertyValue(k){return this[k]||'';}};this.listeners=new Map();this._text='';this._html='';this.hidden=false;this.multiple=false;this.selected=false;this.checked=false;this._value='';this.tabIndex=0;this.clientWidth=1200;this.clientHeight=500;this.offsetWidth=1200;
  }
  get id(){return this._attributes.get('id')||'';}
  set id(value){this.setAttribute('id',value);}
  get className(){return this.classList.toString();}
  set className(value){this.classList.set(value);}
  setAttribute(name,value){value=String(value);this._attributes.set(name,value);if(name==='class')this.classList.set(value);if(name==='hidden')this.hidden=true;if(name==='multiple')this.multiple=true;if(name==='selected')this.selected=true;if(name==='checked')this.checked=true;if(name==='value')this._value=value;if(name==='id')this.ownerDocument?.ids.set(value,this);if(name.startsWith('data-'))this.dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;}
  getAttribute(name){if(name==='class')return this.className||null;if(name==='hidden')return this.hidden?'':null;return this._attributes.get(name)??null;}
  hasAttribute(name){return this.getAttribute(name)!==null;}
  removeAttribute(name){this._attributes.delete(name);if(name==='hidden')this.hidden=false;}
  get attributes(){return [...this._attributes].map(([name,value])=>({name,value}));}
  get options(){return this.querySelectorAll('option');}
  get selectedOptions(){return this.options.filter(option=>option.selected);}
  get selectedIndex(){return this.options.findIndex(option=>option.selected);}
  set selectedIndex(index){this.options.forEach((option,i)=>{option.selected=i===Number(index);});}
  get value(){if(this.tagName==='SELECT')return this.selectedOptions[0]?.value??this.options[0]?.value??this._value;return this._value;}
  set value(value){this._value=String(value);if(this.tagName==='SELECT')this.options.forEach(option=>{option.selected=option.value===this._value;});}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  set textContent(value){this.children=[];this._text=String(value??'');this._html='';}
  get innerText(){return this.textContent;}
  set innerText(value){this.textContent=value;}
  get innerHTML(){return this._html;}
  set innerHTML(value){this._html=String(value);this.children=[];this._text='';parseFragment(this._html,this,this.ownerDocument);this.normalizeOptions();}
  normalizeOptions(){if(this.tagName==='SELECT'&&!this.multiple&&this.options.length&&!this.options.some(o=>o.selected))this.options[0].selected=true;}
  append(...nodes){for(let child of nodes){if(typeof child!=='object'){this._text+=String(child);continue;}child.parentElement=this;child.ownerDocument=this.ownerDocument;this.children.push(child);}this.normalizeOptions();}
  appendChild(child){this.append(child);return child;}
  replaceChildren(...nodes){this.children=[];this._text='';this._html='';this.append(...nodes);}
  remove(){if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(child=>child!==this);}
  addEventListener(type,callback){if(!this.listeners.has(type))this.listeners.set(type,[]);this.listeners.get(type).push(callback);}
  removeEventListener(type,callback){this.listeners.set(type,(this.listeners.get(type)||[]).filter(fn=>fn!==callback));}
  dispatchEvent(event){if(!event.target)event.target=this;event.currentTarget=this;this['on'+event.type]?.call(this,event);for(const callback of this.listeners.get(event.type)||[])callback.call(this,event);if(event.bubbles)this.parentElement?.dispatchEvent(event);return !event.defaultPrevented;}
  click(){if(this.tagName==='INPUT'&&this.getAttribute('type')==='checkbox'){this.checked=!this.checked;this.dispatchEvent(new AuditEvent('click',{bubbles:true}));this.dispatchEvent(new AuditEvent('change',{bubbles:true}));}else this.dispatchEvent(new AuditEvent('click',{bubbles:true}));}
  focus(){if(this.ownerDocument)this.ownerDocument.activeElement=this;}
  matches(selector){return matches(this,selector);}
  closest(selector){let node=this;while(node){if(matches(node,selector))return node;node=node.parentElement;}return null;}
  querySelectorAll(selector){const result=[];const visit=node=>{for(const child of node.children){if(matches(child,selector))result.push(child);visit(child);}};visit(this);return result;}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  getBoundingClientRect(){return {x:0,y:0,left:0,top:0,right:this.clientWidth,bottom:this.clientHeight,width:this.clientWidth,height:this.clientHeight};}
  scrollIntoView(){}
}
function parseFragment(markup,parent,document){
  const stack=[parent],tokens=String(markup).match(/<!--[\s\S]*?-->|<[^>]+>|[^<]+/g)||[];
  for(const token of tokens){
    if(token.startsWith('<!--')||token.startsWith('<!'))continue;
    if(token.startsWith('</')){const tag=token.match(/^<\/\s*([\w:-]+)/)?.[1]?.toUpperCase();let index=stack.length-1;while(index>0&&stack[index].tagName!==tag)index--;if(index>0)stack.length=index;continue;}
    if(token.startsWith('<')){const name=token.match(/^<\s*([\w:-]+)/)?.[1];if(!name)continue;const node=new AuditElement(name,document);const attrs=token.slice(token.indexOf(name)+name.length).replace(/\/?\s*>$/,'');for(const attribute of attrs.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g))node.setAttribute(attribute[1],decode(attribute[2]??attribute[3]??attribute[4]??''));stack.at(-1).append(node);if(!voidTags.has(node.tagName)&&!token.endsWith('/>'))stack.push(node);continue;}
    stack.at(-1)._text+=decode(token);
  }
}
export function makeDOM(html){
  const document={ids:new Map(),listeners:new Map(),readyState:'complete',activeElement:null};
  const root=new AuditElement('document',document);document.root=root;
  const markup=html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'').replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi,'');
  parseFragment(markup,root,document);
  document.documentElement=root.querySelector('html')||root;document.body=root.querySelector('body')||root;document.head=root.querySelector('head')||root;
  document.getElementById=id=>document.ids.get(id)||null;
  document.querySelectorAll=selector=>root.querySelectorAll(selector);document.querySelector=selector=>root.querySelector(selector);
  document.createElement=tag=>new AuditElement(tag,document);document.createTextNode=text=>{const node=new AuditElement('text',document);node.textContent=text;return node;};
  document.addEventListener=(type,callback)=>{if(!document.listeners.has(type))document.listeners.set(type,[]);document.listeners.get(type).push(callback);};
  document.dispatchEvent=event=>{for(const callback of document.listeners.get(event.type)||[])callback(event);};
  document.body.addEventListener('click',event=>document.dispatchEvent(event));
  return document;
}
const palette={light:{'--ink':'#183247','--paper':'#ffffff','--text':'#26364a','--muted':'#64748b','--panel':'#ffffff','--surface':'#ffffff','--border':'#dbe3ee','--grid':'#e4eaf2','--plot-bg':'#ffffff'},dark:{'--ink':'#e5eff8','--paper':'#192936','--text':'#e5ebf4','--muted':'#a8b6ca','--panel':'#17243a','--surface':'#17243a','--border':'#34445e','--grid':'#34445e','--plot-bg':'#17243a'}};
export function mountReport(html,{payloadId='payload',applicationId=null,storage={}}={}){
  const document=makeDOM(html),plots=new Map(),stored=new Map(Object.entries(storage)),events=new Map();
  const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].map(match=>({attrs:match[1],source:match[2],id:match[1].match(/\bid="([^"]+)"/)?.[1]}));
  for(const script of scripts){if(script.id){let node=document.getElementById(script.id);if(!node){node=document.createElement('script');node.id=script.id;document.body.append(node);}node.textContent=script.source;}}
  const payloadMatch=html.match(new RegExp('<script[^>]*id="'+payloadId+'"[^>]*>([\\s\\S]*?)<\\/script>'));
  assert.ok(payloadMatch,'report payload exists');
  let payload=document.getElementById(payloadId);if(!payload){payload=document.createElement('script');payload.id=payloadId;document.body.append(payload);}payload.textContent=payloadMatch[1];
  const window={document,console,Buffer,Promise,Event:AuditEvent,KeyboardEvent:AuditEvent,Blob,URL,setTimeout:callback=>{callback();return 1;},clearTimeout(){},requestAnimationFrame:callback=>{callback();return 1;},cancelAnimationFrame(){},innerWidth:1440,scrollY:0,scrollTo(){},location:{hash:'',search:''},localStorage:{getItem:key=>stored.get(key)??null,setItem:(key,value)=>stored.set(key,String(value)),removeItem:key=>stored.delete(key)},matchMedia:()=>({matches:false,addEventListener(){},removeEventListener(){}}),getComputedStyle(element){const theme=document.documentElement.dataset.theme==='dark'?'dark':'light';return {getPropertyValue:key=>element.style.getPropertyValue(key)||palette[theme][key]||'',display:element.hidden?'none':'block'};},MathJax:{typesetPromise:()=>Promise.resolve(),startup:{promise:Promise.resolve()}}};
  window.window=window;window.globalThis=window;
  window.addEventListener=(type,callback)=>{if(!events.has(type))events.set(type,[]);events.get(type).push(callback);};
  window.dispatchEvent=event=>{for(const callback of events.get(event.type)||[])callback(event);};
  window.Plotly={react(target,data,layout,config){const id=typeof target==='string'?target:target.id,element=document.getElementById(id);for(const trace of data){if(Array.isArray(trace.x)&&Array.isArray(trace.y))assert.equal(trace.x.length,trace.y.length,'paired trace arrays '+id);}plots.set(id,{data,layout,config});if(element){element.data=data;element.layout=layout;element.classList.add('js-plotly-plot');}return Promise.resolve();},newPlot(...args){return this.react(...args);},Plots:{resize(){return Promise.resolve();}},purge(){}};
  const context=vm.createContext(window);

  const common=scripts.find(script=>script.id==='report-theme-runtime'||script.id==='flux-report-ui'||/window\.FluxReportUI\s*=/.test(script.source));
  if(common)new vm.Script(common.source,{filename:'actual-report-theme.js'}).runInContext(context,{timeout:10000});
  const app=applicationId?scripts.find(script=>script.id===applicationId):scripts.at(-1);
  assert.ok(app,'report application exists');new vm.Script(app.source,{filename:'actual-report-application.js'}).runInContext(context,{timeout:10000});
  document.dispatchEvent(new AuditEvent('DOMContentLoaded'));window.dispatchEvent(new AuditEvent('load'));
  return {document,window,plots,storage:stored,payload:JSON.parse(payloadMatch[1]),get:id=>{const node=document.getElementById(id);assert.ok(node,'known element '+id);return node;},scripts,html};
}
