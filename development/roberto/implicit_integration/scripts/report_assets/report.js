/* Plots and internal pages use only the embedded saved evidence. */
(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('report-data').textContent);
  const names={newton:'Newton caps',cfl100:'CFL / cap 100',cfl1:'CFL / cap 1',theory:'Theory and provenance'};
  let active='newton',queue=Promise.resolve(),generation=0;
  const $=id=>document.getElementById(id),figures=new Map();
  const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'implicit-integration'}};
  const metricInfo={updates:['Accepted outer update','Original spatial residual RMS','log'],work:['Cumulative Newton corrections, including rejects','Original spatial residual RMS','log'],time:['Recorded solve time [s]','Original spatial residual RMS','log'],cfl:['Accepted outer update','Actual CFL','log'],pressure:['Axial position x [m]','Static pressure [kPa]','linear'],inner:['Accepted outer update','Newton corrections in accepted step','linear']};
  function colors(){const s=getComputedStyle(document.documentElement);return {bg:s.getPropertyValue('--panel').trim(),ink:s.getPropertyValue('--ink').trim(),grid:s.getPropertyValue('--grid').trim(),line:s.getPropertyValue('--line').trim()};}
  function axis(title){const p=colors();return {title:{text:title,font:{size:10}},color:p.ink,tickfont:{size:10},showline:true,mirror:true,linecolor:p.line,gridcolor:p.grid,zeroline:false,automargin:true,exponentformat:'power',showexponent:'all',ticks:'outside',ticklen:4};}
  async function plot(id,data,layout){await Plotly.react($(id),data,layout,config);figures.set(id,{data,layout});}
  function selected(page,caseId){return D.studies[page].map(id=>D.runs[id]).filter(r=>!caseId||r.case===caseId);}
  function legendLabel(r,page){return page==='newton'?'Cap '+r.cap:r.label;}
  async function drawStudy(page){
    const skip=Number($(page+'-skip').value);
    if(!Number.isInteger(skip)||skip<0){$(page+'-skip').setAttribute('aria-invalid','true');return;}
    $(page+'-skip').removeAttribute('aria-invalid');
    const stride=skip+1,metric=$(page+'-metric').value,rejects=$(page+'-rejects').checked,p=colors();
    const [xt,yt,yscale]=metricInfo[metric],traces=[],annotations=[],shapes=[];
    const layout={paper_bgcolor:p.bg,plot_bgcolor:p.bg,font:{family:'Arial, sans-serif',size:11,color:p.ink},height:900,
      margin:{l:65,r:22,t:70,b:115},legend:{orientation:'h',x:0,y:-.14,font:{size:10},groupclick:'togglegroup'},hovermode:'closest'};
    const xdomains=[[0,.285],[.355,.64],[.715,1]],ydomains=[[.59,1],[.02,.43]];
    D.order.forEach((caseId,ci)=>{
      const suffix=ci?String(ci+1):'',xa='x'+suffix,ya='y'+suffix,xd=xdomains[ci%3],yd=ydomains[Math.floor(ci/3)],meta=D.cases[caseId];
      layout['xaxis'+suffix]={...axis(xt),domain:xd,anchor:ya};
      layout['yaxis'+suffix]={...axis(yt),domain:yd,anchor:xa,type:yscale};
      annotations.push({xref:'paper',yref:'paper',x:(xd[0]+xd[1])/2,y:yd[1]+.065,text:D.titles[caseId]+'<br><span style="font-size:10px">pᵦ/p₀ = '+(meta.p_back/meta.p0).toFixed(6)+'</span>',showarrow:false,xanchor:'center',font:{size:12}});
      selected(page,caseId).forEach((r,vi)=>{
        const h=r.history,label=legendLabel(r,page),color=D.palettes[page][vi],dash=['solid','dash','dot','dashdot'][vi%4];
        let x=h.accepted_index,y=h.residual_rms;
        if(metric==='work')x=h.cumulative_linear_solves;
        if(metric==='time')x=h.elapsed_s;
        if(metric==='cfl')y=h.effective_cfl;
        if(metric==='inner')y=h.inner_solves;
        if(metric==='pressure'){x=meta.x_m.slice(1,-1);y=r.pressure_Pa.slice(1,-1).map(v=>v/1000);}
        traces.push({x,y,name:label,legendgroup:label,showlegend:ci===0,mode:'lines',line:{color,dash,width:1.7},xaxis:xa,yaxis:ya,
          hovertemplate:label+'<br>%{x:.6g}<br>%{y:.6g}<extra>'+D.titles[caseId]+'</extra>'});
        if(metric==='pressure'){
          const mx=[],my=[];for(let i=vi%stride;i<x.length;i+=stride){mx.push(x[i]);my.push(y[i]);}
          traces.push({x:mx,y:my,name:label,legendgroup:label,showlegend:false,mode:'markers',marker:{size:7,color:p.bg,line:{color,width:1.5}},xaxis:xa,yaxis:ya,hoverinfo:'skip'});
        }else{
          const last=x.length-1;
          traces.push({x:[x[last]],y:[y[last]],name:label,legendgroup:label,showlegend:false,mode:'markers',marker:{size:6,color,symbol:r.root_success?'circle':'x'},xaxis:xa,yaxis:ya,hoverinfo:'skip'});
        }
        if(rejects&&(metric==='cfl'||metric==='inner')){
          const rx=[],ry=[];
          r.trials.accepted.forEach((yes,i)=>{if(!yes){rx.push(r.trials.anchor_index[i]+1);ry.push(metric==='cfl'?r.trials.effective_cfl[i]:r.trials.inner_solves[i]);}});
          traces.push({x:rx,y:ry,name:label+' rejected',legendgroup:label,showlegend:false,mode:'markers',marker:{size:6,color,symbol:'x'},xaxis:xa,yaxis:ya,hovertemplate:'Rejected attempt<br>%{x}<br>%{y}<extra>'+label+'</extra>'});
        }
      });
      if(metric==='pressure')traces.push({x:meta.exact_x_m,y:meta.exact_pressure_Pa.map(v=>v/1000),name:'Analytical reference',legendgroup:'Analytical',showlegend:ci===0,mode:'lines',line:{color:p.ink,dash:'dash',width:2},xaxis:xa,yaxis:ya});
      if(['updates','work','time'].includes(metric))shapes.push({type:'line',xref:'paper',yref:ya,x0:xd[0],x1:xd[1],y0:1e-8,y1:1e-8,line:{color:p.ink,dash:'dot',width:1}});
    });
    layout.annotations=annotations;layout.shapes=shapes;
    await plot(page+'-histories',traces,layout);
    const variants=selected(page,D.order[0]);
    const bars=[],barlayout={paper_bgcolor:p.bg,plot_bgcolor:p.bg,font:{family:'Arial, sans-serif',size:11,color:p.ink},height:520,barmode:'group',margin:{l:65,r:30,t:55,b:125},legend:{orientation:'h',y:-.27,font:{size:10}},
      xaxis:{...axis('Physical case'),domain:[0,.44]},yaxis:axis('All attempted Newton corrections'),
      xaxis2:{...axis('Physical case'),domain:[.56,1],anchor:'y2'},yaxis2:{...axis('Recorded solve time [s] · one sample'),anchor:'x2'}};
    variants.forEach((v,vi)=>{
      const all=selected(page).filter(r=>page==='newton'?r.cap===v.cap:r.cfl_variant===v.cfl_variant),label=legendLabel(v,page);
      for(const [key,xaxis,yaxis] of [['linear_solves','x','y'],['solve_s','x2','y2']])bars.push({type:'bar',x:all.map(r=>r.case),y:all.map(r=>r[key]),name:label,legendgroup:label,showlegend:xaxis==='x',marker:{color:D.palettes[page][vi]},xaxis,yaxis});
    });
    await plot(page+'-work',bars,barlayout);
  }
  function targetPage(){let key='newton';try{key=decodeURIComponent(location.hash.slice(1))||key;}catch{}const el=$(key)||$('newton');return {el,page:el.closest('[data-report-page]')?.dataset.reportPage||'newton'};}
  async function typeset(){await MathJax.startup.promise;await MathJax.typesetPromise([$('theory')]);}
  function activate(scroll=true){
    const {el,page}=targetPage();active=page;
    document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=p.dataset.reportPage!==page);
    document.querySelectorAll('[data-report-nav]').forEach(p=>p.hidden=p.dataset.reportNav!==page);
    document.querySelectorAll('[data-report-tab]').forEach(a=>{if(a.dataset.reportTab===page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    document.querySelectorAll('.sidebar a').forEach(a=>a.classList.toggle('active',a.hash==='#'+el.id));
    $('sidebar-label').textContent=names[page];document.title=names[page]+' · Implicit integration';
    document.documentElement.dataset.reportReady='false';
    queue=queue.catch(console.error).then(async()=>{
      if(active!==page)return;
      if(page==='theory')await typeset();else await drawStudy(page);
      if(scroll)el.scrollIntoView({block:'start'});
      document.documentElement.dataset.reportReady='true';
    });
    return queue;
  }
  function theme(value){document.documentElement.dataset.theme=value;try{localStorage.setItem('implicit-integration-report-theme',value);}catch{}
    document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.themeChoice===value));generation++;return activate(false);}
  document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>theme(b.dataset.themeChoice));
  for(const page of ['newton','cfl100','cfl1']){
    $(page+'-metric').onchange=()=>activate(false);
    $(page+'-skip').oninput=()=>activate(false);
    $(page+'-rejects').onchange=()=>activate(false);
    $(page+'-reset').onclick=()=>{generation++;activate(false);};
  }
  window.addEventListener('hashchange',()=>activate());
  $('print-report').onclick=async()=>{
    await queue;document.body.classList.add('printing');
    document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=false);
    await typeset();for(const page of ['newton','cfl100','cfl1'])await drawStudy(page);
    requestAnimationFrame(()=>requestAnimationFrame(()=>window.print()));
  };
  window.addEventListener('beforeprint',()=>document.body.classList.add('printing'));
  window.addEventListener('afterprint',()=>{document.body.classList.remove('printing');activate(false);});
  window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='p'){e.preventDefault();$('print-report').click();}});
  window.implicitIntegrationReport={data:D,figures,ready:()=>queue,activate};
  theme(document.documentElement.dataset.theme);
})();
