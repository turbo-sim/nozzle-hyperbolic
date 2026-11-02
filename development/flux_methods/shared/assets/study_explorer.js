/* Shared offline presentation helpers. Physical calculations do not live here. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=v=>v==null||!Number.isFinite(Number(v))?'—':Number(Number(v).toPrecision(4)).toString();
  const badge=(passed,yes='Pass',no='Miss')=>`<span class="badge ${passed?'pass':'warn'}">${escape(passed?yes:no)}</span>`;
  const stat=(title,value,note)=>`<article class="stat"><span>${escape(title)}</span><strong>${escape(value)}</strong><small>${escape(note)}</small></article>`;
  const colors=['#6a00a8','#a11b9b','#cc4778','#ed7953','#fca636'];
  let tasks=[];
  function layout(title,y) {
    const l=FluxReportUI.layout(title,y);l.margin={l:78,r:28,t:65,b:110};l.legend={orientation:'h',x:0,y:-.23,font:{size:10}};return l;
  }
  function plot(id,traces,l) {
    if(!traces.length)l.annotations=[{text:'No selected saved profiles',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
    const p=Plotly.react($(id),traces,l,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'svg',filename:id}});
    tasks.push(Promise.resolve(p));return p;
  }
  function mount(render) {
    let number=0,rendered=0,page='overview';
    function redraw(){const n=++number;tasks=[];render(page);Promise.all(tasks).then(()=>{if(n===number)rendered=n;});}
    FluxReportUI.init({onThemeChange:redraw});
    const tabs=FluxReportUI.bindTabs({nav:'#page-tabs',panels:'.report-page',onChange:id=>{page=id.replace('page-','');redraw();}});
    return {render:redraw,showPage:p=>tabs.activate('page-'+p),snapshot:()=>({page,renderNumber:number,rendered})};
  }
  window.StudyReportUI={$,escape,fmt,badge,stat,colors,layout,plot,mount};
})();
