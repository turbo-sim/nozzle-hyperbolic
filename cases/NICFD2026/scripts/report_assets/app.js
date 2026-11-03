'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
const COLORS=['#6a00a8','#cc4778','#f89540'];
const INFO={p:['Pressure','bar',1e-5],alpha_g:['Vapour/gas volume fraction','–',1],u_g:['Gas velocity','m/s',1],u_l:['Liquid velocity','m/s',1],Ma_mix:['Mixture Mach number','–',1],mdot:['Total mass flow','kg/s',1],T_g:['Gas temperature','K',1],T_l:['Liquid temperature','K',1],h_g:['Gas enthalpy','kJ/kg',.001],h_l:['Liquid enthalpy','kJ/kg',.001],rho_g:['Gas density','kg/m³',1],rho_l:['Liquid density','kg/m³',1],quality_static:['Static gas mass fraction','–',1],quality_flow:['Gas mass-flow fraction','–',1],slip:['Velocity slip','–',1],a_mix:['Acoustic mixture speed','m/s',1]};
const GROUPS={flow:['p','alpha_g','u_g','u_l','Ma_mix','mdot'],thermal:['T_g','T_l','h_g','h_l','rho_g','rho_l'],mixture:['quality_static','quality_flow','slip','a_mix','Ma_mix','mdot']};
let renderNumber=0,rendered=0,activePage='profiles';
function theme(value){document.documentElement.dataset.theme=value;try{localStorage.setItem('nicfd2026-theme',value);}catch(_){}for(const v of ['light','dark'])$('theme-'+v).setAttribute('aria-pressed',String(v===value));if(DATA.kind==='case')render();}
$('theme-light').onclick=()=>theme('light');$('theme-dark').onclick=()=>theme('dark');
function css(key){return getComputedStyle(document.documentElement).getPropertyValue(key).trim();}
function layout(title,ylabel,extra={}){return {title:{text:title,font:{size:15}},paper_bgcolor:css('--paper'),plot_bgcolor:css('--paper'),font:{color:css('--ink'),family:'Segoe UI, Arial',size:11},margin:{l:70,r:22,t:50,b:58},xaxis:{title:{text:'Axial position [m]'},gridcolor:css('--grid'),zeroline:false},yaxis:{title:{text:ylabel},gridcolor:css('--grid'),zeroline:false},legend:{orientation:'h',y:-.22},...extra};}
function plot(id,traces,settings){return Plotly.react(id,traces,settings,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg'}});}
function run(){return DATA.runs.find(r=>r.mode===$('mode').value);}
function stage(){return run()?.stages.find(s=>String(s.cells)===$('grid').value);}
function good(value){return typeof value==='number'&&Number.isFinite(value);}
function interpolate(x,xs,ys){if(x<=xs[0])return ys[0];if(x>=xs.at(-1))return ys.at(-1);let a=0,b=xs.length-1;while(b-a>1){const m=(a+b)>>1;if(xs[m]<=x)a=m;else b=m;}return ys[a]+(ys[b]-ys[a])*(x-xs[a])/(xs[b]-xs[a]);}
function markers(n){const skip=Math.max(0,Math.floor(Number($('skip').value)||0)),step=Math.min(skip,n-1)+1;return [...new Set([...Array.from({length:Math.ceil(n/step)},(_,i)=>i*step),n-1])].filter(i=>i<n);}
function chooseGrid(){const old=$('grid').value;const stages=run()?.stages.filter(s=>s.fields)||[];$('grid').innerHTML=stages.length?stages.map(s=>`<option value="${s.cells}">${s.cells} cells${s.case_success?'':' · not reproduced'}</option>`).join(''):'<option value="">No current field</option>';$('grid').value=stages.some(s=>String(s.cells)===old)?old:String(stages.at(-1)?.cells||'');$('grid').disabled=!stages.length;}
function historyX(history){const axis=$('history-axis').value;let work=0;const prefix=new Map();for(const row of stage()?.history||history){work+=row.inner_iterations||0;prefix.set(row,work);}return history.map(h=>axis==='time'?h.elapsed_s:axis==='work'?prefix.get(h):h.step);}
async function render(){
 const number=++renderNumber,s=stage(),old=DATA.reference,group=$('field-group').value,visibility=$('visibility').value;
 const ids=GROUPS[group];$('profile-plots').innerHTML=ids.map(k=>`<div class="pair"><div id="field-${k}" class="chart"></div><div id="error-${k}" class="chart error"${s?.fields?'':' hidden'}></div></div>`).join('');
 $('selected-note').textContent=s?`${run().mode==='cold'?'Compilation-inclusive':'Replayed'} ${s.cells}-cell attempt: ${s.reason}. ${s.initialization}. ${s.case_success?'Convergence and regime checks passed.':'This plotted state is not a verified reproduction.'}`:'No current-solver field was returned. The published profiles remain available.';
 const current=s?.fields,promises=[],color=COLORS[Math.min(2,DATA.protocol.meshes.indexOf(s?.cells))]||COLORS[2];
 $('download-profile').disabled=!current;$('download-history').disabled=!s?.history?.length;
 for(const key of ids){const [label,unit,scale]=INFO[key],traces=[];if(visibility!=='current')traces.push({x:old.x,y:old[key].map(v=>v*scale),mode:'lines',name:'Published solution',line:{color:css('--ink'),width:2}});
  if(current&&visibility!=='published'){const indices=markers(current.x.length);traces.push({x:indices.map(i=>current.x[i]),y:indices.map(i=>good(current[key][i])?current[key][i]*scale:null),name:`Current · ${s.cells} cells${s.case_success?'':' · unconverged/unverified'}`,mode:'lines+markers',line:{color,width:1.4},marker:{color:'#fff',line:{color,width:1.3},size:5}});}
  promises.push(plot('field-'+key,traces,layout(label,unit)));
  const difference=current?current.x.map((x,i)=>good(current[key][i])?Math.abs(current[key][i]-interpolate(x,old.x,old[key]))*scale:null):[];
  promises.push(plot('error-'+key,current?[{x:current.x,y:difference.map(v=>v===0?null:v),mode:'lines',name:'Absolute profile difference',line:{color,width:1.6}}]:[],layout('Absolute difference',unit,{height:240,yaxis:{title:{text:unit},type:'log',gridcolor:css('--grid')},showlegend:false,margin:{l:70,r:22,t:43,b:53}})));
 }
 const r=DATA.reference,upper=r.r.map(v=>v*1000),lower=r.r.map(v=>-v*1000);
 promises.push(plot('geometry',[{x:r.x,y:lower,mode:'lines',line:{color:css('--muted'),width:1},showlegend:false},{x:r.x,y:upper,mode:'lines',fill:'tonexty',fillcolor:'#cc47781b',line:{color:css('--muted'),width:1.4},name:'Published nozzle geometry'}],layout('Nozzle geometry','Radius [mm]',{height:200,showlegend:false,margin:{l:70,r:22,t:45,b:45}})));
 const history=s?.history||[],accepted=history.filter(h=>h.accepted&&h.equation_norm),hx=historyX(accepted),axis=$('history-axis').value,xlabel=axis==='time'?'Solve time [s]':axis==='work'?'Cumulative Newton corrections':'Accepted step index';
 $('residual').parentElement.hidden=!history.length;$('history-axis').closest('label').hidden=!history.length;$('history-note').hidden=Boolean(history.length);$('history-note').textContent=DATA.property_audit.usable_on_published_profile?'No current-solver history was returned; consult the stage outcomes.':'Property-table initialization was rejected before a CFD solve started. There are no nonlinear iterations or convergence curves to display.';
 const residuals=DATA.equation_names.map((label,i)=>({x:hx,y:accepted.map(h=>h.equation_norm[i]>0?h.equation_norm[i]:null),mode:'lines',name:label,line:{color:['#5c01a6','#8b0aa5','#b52f8c','#d8576b','#ed7953','#fca636'][i]}}));
 promises.push(plot('residual',residuals,layout('Original selected spatial equations','Dimensionless norm',{xaxis:{title:{text:xlabel},gridcolor:css('--grid')},yaxis:{title:{text:'Dimensionless norm'},type:'log',gridcolor:css('--grid')}})));
 const defects=[['boundary_max','Boundary traces'],['constraint_max','Cell EOS constraints'],['global_defect','Global balance']].map(([key,label])=>({x:hx,y:accepted.map(h=>{const v=Array.isArray(h[key])?Math.max(...h[key].map(Math.abs)):h[key];return good(v)&&v>0?v:null;}),name:label,mode:'lines'}));
 promises.push(plot('defects',defects,layout('Independent algebraic / balance checks','Dimensionless defect',{xaxis:{title:{text:xlabel},gridcolor:css('--grid')},yaxis:{title:{text:'Dimensionless defect'},type:'log',gridcolor:css('--grid')}})));
 const trials=history.filter(h=>h.phase!=='initial');let work=0;
 promises.push(plot('cfl',[{x:trials.map((_,i)=>i+1),y:trials.map(h=>h.cfl),mode:'lines+markers',name:'Actual trial CFL',line:{color:COLORS[1]},marker:{symbol:trials.map(h=>h.accepted?'circle':'x'),size:5}}],layout('CFL and rejected attempts','Actual CFL',{xaxis:{title:{text:'Trial index'},gridcolor:css('--grid')},yaxis:{title:{text:'Actual CFL'},type:'log',gridcolor:css('--grid')}})));
 promises.push(plot('work',[{x:trials.map((_,i)=>i+1),y:trials.map(h=>work+=h.inner_iterations||0),mode:'lines',name:'All Newton corrections',line:{color:COLORS[0]}}],layout('Actual Newton work','Cumulative corrections',{xaxis:{title:{text:'Trial index'},gridcolor:css('--grid')}})));
 await Promise.all(promises);rendered=number;window.dispatchEvent(new CustomEvent('nicfd-rendered'));
}
function showPage(id){activePage=id;for(const page of document.querySelectorAll('.page'))page.hidden=page.id!==id;for(const tab of document.querySelectorAll('[data-page]'))tab.setAttribute('aria-selected',String(tab.dataset.page===id));if(DATA.kind==='case')render();}
function download(name,text,mime){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:mime}));a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(fields){const keys=Object.keys(fields).filter(k=>Array.isArray(fields[k])&&!Array.isArray(fields[k][0]));return keys.join(',')+'\n'+fields[keys[0]].map((_,i)=>keys.map(k=>fields[k][i]??'').join(',')).join('\n');}
if(DATA.kind==='case'){
 $('mode').innerHTML=DATA.runs.map(r=>`<option value="${r.mode}">${r.mode==='cold'?'Cold · compilation included':'Warm · same bindings'}</option>`).join('');$('mode').value=DATA.runs.at(-1)?.mode||'cold';chooseGrid();
 $('mode').onchange=()=>{chooseGrid();render();};for(const id of ['grid','field-group','visibility','skip','history-axis'])$(id).onchange=render;
 for(const tab of document.querySelectorAll('[data-page]'))tab.onclick=()=>showPage(tab.dataset.page);
 $('download-json').onclick=()=>download('NICFD2026_'+DATA.family+'_'+DATA.regime+'.json',JSON.stringify(DATA,null,2),'application/json');
 $('download-profile').onclick=()=>{if(stage()?.fields)download('current_profile.csv',csv(stage().fields),'text/csv');};
 $('download-reference').onclick=()=>download('published_reference.csv',csv(DATA.reference),'text/csv');
 $('download-history').onclick=()=>{const h=stage()?.history||[],rows=h.map(row=>{const flat={};for(const [key,value] of Object.entries(row)){if(Array.isArray(value))value.forEach((v,i)=>{if(typeof v!=='object')flat[key+'_'+(i+1)]=v;});else if(typeof value!=='object')flat[key]=value;}return flat;}),keys=[...new Set(rows.flatMap(Object.keys))];download('convergence_history.csv',keys.join(',')+'\n'+rows.map(row=>keys.map(k=>JSON.stringify(row[k]??'')).join(',')).join('\n'),'text/csv');};
 window.nicfdReport={data:DATA,stage,render,showPage,snapshot:()=>({renderNumber,rendered,activePage,mode:$('mode').value,cells:$('grid').value,fields:GROUPS[$('field-group').value],visibility:$('visibility').value}),markerIndices:markers};
}
let initialTheme='light';try{initialTheme=localStorage.getItem('nicfd2026-theme')||'light';}catch(_){}theme(initialTheme);
