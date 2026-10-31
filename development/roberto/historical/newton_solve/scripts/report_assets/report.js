/* The browser only displays embedded retained records; it solves no flow equations. */
(() => {
 'use strict';
 const D=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);
 const colors={implicit:'#6a00a8',ptc:'#cb4779',newton:'#fca636',hybr:'#9c179e',lm:'#cb4779',slsqp:'#e16462'};
 const titles={cold:'Cold starts',performance:'PTC cost',adapted:'Adapted starts',theory:'Convergence mechanism',notes:'Earlier investigations'};
 let active='cold',queue=Promise.resolve();const figures=new Map();
 const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'newton-study'}};
 function palette(){const s=getComputedStyle(document.documentElement);return {bg:s.getPropertyValue('--panel').trim(),ink:s.getPropertyValue('--ink').trim(),grid:s.getPropertyValue('--grid').trim(),line:s.getPropertyValue('--line').trim()};}
 function axis(text){const p=palette();return {title:{text,font:{size:10}},color:p.ink,tickfont:{size:10},showline:true,mirror:true,linecolor:p.line,gridcolor:p.grid,zeroline:false,automargin:true,exponentformat:'power',showexponent:'all',ticks:'outside',ticklen:4};}
 function layout(height=900){const p=palette();return {height,paper_bgcolor:p.bg,plot_bgcolor:p.bg,font:{family:'Arial,sans-serif',size:11,color:p.ink},margin:{l:66,r:25,t:65,b:105},legend:{orientation:'h',y:-.13,font:{size:10},groupclick:'togglegroup'},hovermode:'closest'};}
 async function plot(id,traces,l){await Plotly.react($(id),traces,l,config);figures.set(id,{traces,layout:l});}
 function markers(x,y,skip,offset){const mx=[],my=[];for(let i=offset%(skip+1);i<x.length;i+=skip+1){mx.push(x[i]);my.push(y[i]);}return {x:mx,y:my};}
 function curve(traces,r,name,color,metric,caseData,xa,ya,showlegend,skip,offset=0){
  const h=r.history;let x=h.map((v,i)=>metric==='time'?v.elapsed_s:(v.accepted_index??v.observation_index??i)),y=h.map(v=>v.residual_rms);
  if(metric==='pressure'){x=caseData.x_m.slice(1,-1);y=r.diagnostics.fields.p.slice(1,-1).map(v=>v/1000);}
  const failed=r.root_success===false,wrong=r.root_success&&r.same_shocked_root===false,group=name;
  traces.push({x,y,name:name+(failed?' · unconverged':wrong?' · wrong branch':''),legendgroup:group,showlegend,mode:'lines',line:{color,width:1.8,dash:offset%3===1?'dash':offset%3===2?'dot':'solid'},xaxis:xa,yaxis:ya});
  if(metric==='pressure')traces.push({...markers(x,y,skip,offset),name,legendgroup:group,showlegend:false,mode:'markers',marker:{size:7,color:palette().bg,line:{color,width:1.4}},hoverinfo:'skip',xaxis:xa,yaxis:ya});
  else traces.push({x:x.slice(-1),y:y.slice(-1),name,legendgroup:group,showlegend:false,mode:'markers',marker:{size:7,color,symbol:failed?'x':'circle'},xaxis:xa,yaxis:ya,hoverinfo:'skip'});
 }
 function frame(l,i,count,xlabel,ylabel,logY,title){
  const suffix=i?String(i+1):'',xa='x'+suffix,ya='y'+suffix;
  const xd=count===2?[[0,.435],[.565,1]][i]:[[0,.285],[.355,.64],[.715,1]][i%3];
  const yd=count===2?[.02,1]:[[.59,1],[.02,.43]][Math.floor(i/3)];
  l['xaxis'+suffix]={...axis(xlabel),domain:xd,anchor:ya};l['yaxis'+suffix]={...axis(ylabel),domain:yd,anchor:xa,type:logY?'log':'linear'};
  (l.annotations??=[]).push({xref:'paper',yref:'paper',x:(xd[0]+xd[1])/2,y:yd[1]+.065,text:title,showarrow:false,xanchor:'center',font:{size:12}});
  return [xa,ya,xd];
 }
 function threshold(l,xd,ya){(l.shapes??=[]).push({type:'line',xref:'paper',yref:ya,x0:xd[0],x1:xd[1],y0:1e-8,y1:1e-8,line:{color:palette().ink,dash:'dot',width:1}});}
 async function cold(){
  const metric=$('cold-metric').value,skip=Number($('cold-skip').value);if(!Number.isInteger(skip)||skip<0){$('cold-skip').setAttribute('aria-invalid','true');return;}$('cold-skip').removeAttribute('aria-invalid');
  const traces=[],l=layout();
  D.cases.forEach((c,i)=>{
   const [xa,ya,xd]=frame(l,i,5,metric==='pressure'?'Axial position x [m]':metric==='time'?'Recorded solve time [s]':'Accepted outer updates',metric==='pressure'?'Static pressure [kPa]':'Original spatial residual RMS',metric!=='pressure',c.name+'<br><span style="font-size:10px">pᵦ/p₀ = '+c.p_back_ratio.toFixed(6)+'</span>');
   c.runs.forEach((r,j)=>curve(traces,r,D.names[r.method],colors[r.method],metric,c,xa,ya,i===0,skip,j));
   if(metric==='pressure')traces.push({x:c.exact_x_m,y:c.exact_pressure_Pa.map(v=>v/1000),name:'Analytical reference',legendgroup:'Analytical',showlegend:i===0,mode:'lines',line:{color:palette().ink,width:2,dash:'dash'},xaxis:xa,yaxis:ya});else threshold(l,xd,ya);
  });await plot('cold-plot',traces,l);
 }
 async function performance(){
  const metric=$('performance-metric').value,traces=[],l=layout();
  D.cases.forEach((c,i)=>{
   const [xa,ya,xd]=frame(l,i,5,'Accepted outer update',metric==='inner'?'Actual inner Newton corrections':metric==='cfl'?'Actual accepted CFL':'Original spatial residual RMS',metric!=='inner',c.name);
   for(const [key,label,color,dash] of [['default','Implicit, original cap 8','#6a00a8','solid'],['one_inner','Implicit, one correction','#fca636','dash']]){
    const h=D.performance.replay[c.id][key].accepted_history;
    traces.push({x:h.map(v=>v.accepted_index),y:h.map(v=>metric==='inner'?v.inner_solves:metric==='cfl'?v.effective_cfl:v.residual_rms),name:label,legendgroup:label,showlegend:i===0,mode:'lines',line:{color,width:1.8,dash},xaxis:xa,yaxis:ya});
   }
   if(metric!=='inner'){
    const r=c.runs.find(v=>v.method==='ptc'),h=r.history;
    traces.push({x:h.map(v=>v.accepted_index),y:h.map(v=>metric==='cfl'?v.cfl:v.residual_rms),name:D.names.ptc,legendgroup:'PTC',showlegend:i===0,mode:'lines',line:{color:colors.ptc,width:1.8,dash:'dot'},xaxis:xa,yaxis:ya});
   }
   if(metric==='residual')threshold(l,xd,ya);
  });await plot('performance-plot',traces,l);
  const a=layout(470),curves=[];a.xaxis=axis('Accepted outer PTC updates');a.yaxis={...axis('Original spatial residual RMS'),type:'log'};
  const base=D.cases.find(c=>c.id==='shock35').runs.find(r=>r.method==='ptc');
  [['Baseline PTC',base],...Object.entries(D.performance.ablations)].forEach(([name,r],i)=>curves.push({x:r.history.map(v=>v.accepted_index),y:r.history.map(v=>v.residual_rms),name,mode:'lines',line:{color:['#6a00a8','#b12a90','#e16462','#fca636'][i],width:1.8,dash:['solid','dash','dot','dashdot'][i]}}));
  await plot('ablation-plot',curves,a);
 }
 async function adapted(){
  const metric=$('adapted-metric').value,seed=$('adapted-seed').value,skip=Number($('adapted-skip').value);if(!Number.isInteger(skip)||skip<0){$('adapted-skip').setAttribute('aria-invalid','true');return;}$('adapted-skip').removeAttribute('aria-invalid');
  const traces=[],l=layout(540);
  D.adapted.forEach((c,i)=>{
   const [xa,ya,xd]=frame(l,i,2,metric==='pressure'?'Axial position x [m]':metric==='time'?'Native solve/callback time [s]':'Accepted-state index',metric==='pressure'?'Static pressure [kPa]':'Original spatial residual RMS',metric!=='pressure',c.label);
   c.runs.filter(r=>r.initialization===seed).forEach((r,j)=>curve(traces,r,D.names[r.method],colors[r.method],metric,c,xa,ya,i===0,skip,j));
   if(metric==='pressure'){
    traces.push({x:c.x_m,y:c.reference.p.map(v=>v/1000),name:'Analytical shocked branch',legendgroup:'Analytical',showlegend:i===0,mode:'lines',line:{color:palette().ink,dash:'dash',width:2},xaxis:xa,yaxis:ya});
    traces.push({x:c.x_m.slice(1,-1),y:c.seeds[seed].fields.p.slice(1,-1).map(v=>v/1000),name:'Numerical starting field',legendgroup:'Seed',showlegend:i===0,mode:'lines',line:{color:'#7592ae',dash:'dot',width:2},xaxis:xa,yaxis:ya});
    traces.push({x:c.x_m.slice(1,-1),y:c.implicit_reference.diagnostics.fields.p.slice(1,-1).map(v=>v/1000),name:'Implicit shocked reference',legendgroup:'Reference',showlegend:i===0,mode:'lines',line:{color:colors.implicit,dash:'dashdot',width:1.8},xaxis:xa,yaxis:ya});
   }else threshold(l,xd,ya);
  });await plot('adapted-plot',traces,l);
 }
 function target(){let id='cold';try{id=decodeURIComponent(location.hash.slice(1))||id;}catch{}const el=$(id)||$('cold');return {el,page:el.closest('[data-report-page]')?.dataset.reportPage||'cold'};}
 async function typeset(page){await MathJax.startup.promise;await MathJax.typesetPromise([$(page)]);}
 function activate(scroll=true){const {el,page}=target();active=page;document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=p.dataset.reportPage!==page);document.querySelectorAll('[data-report-nav]').forEach(p=>p.hidden=p.dataset.reportNav!==page);document.querySelectorAll('[data-report-tab]').forEach(a=>{if(a.dataset.reportTab===page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});$('sidebar-label').textContent=titles[page];document.documentElement.dataset.reportReady='false';queue=queue.catch(console.error).then(async()=>{if(active!==page)return;if(page==='cold')await cold();else if(page==='performance'){await performance();await typeset(page);}else if(page==='adapted')await adapted();else await typeset(page);if(scroll)el.scrollIntoView({block:'start'});document.documentElement.dataset.reportReady='true';});return queue;}
 function theme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem('newton-study-report-theme',t);}catch{}document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.themeChoice===t));activate(false);}
 document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>theme(b.dataset.themeChoice));
 for(const id of ['cold-metric','cold-skip','performance-metric','adapted-metric','adapted-seed','adapted-skip'])$(id).addEventListener(id.endsWith('skip')?'input':'change',()=>activate(false));
 window.addEventListener('hashchange',()=>activate());
 $('print-report').onclick=async()=>{await queue;document.body.classList.add('printing');document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=false);await cold();await performance();await adapted();for(const p of ['performance','theory','notes'])await typeset(p);requestAnimationFrame(()=>requestAnimationFrame(()=>window.print()));};
 window.addEventListener('afterprint',()=>{document.body.classList.remove('printing');activate(false);});
 window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='p'){e.preventDefault();$('print-report').click();}});
 window.newtonStudyReport={data:D,figures,ready:()=>queue};theme(document.documentElement.dataset.theme);
})();
