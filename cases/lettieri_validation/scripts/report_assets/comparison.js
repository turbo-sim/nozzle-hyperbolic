/* Comparison component; all thermodynamic and numerical work is already saved. */
(() => {
  'use strict';
  window.LettieriComparison={install(D,U,context){
    if(!D.solver_comparison||!D.comparison_markup)return null;
    const {$,escape:E,fmt:F,badge:B,stat:S}=U,C=D.solver_comparison;
    const nav=$('page-tabs'),button=document.createElement('button');
    button.id='tab-comparison';button.setAttribute('role','tab');button.setAttribute('aria-controls','page-comparison');
    button.setAttribute('aria-selected','false');button.setAttribute('tabindex','-1');button.textContent='Solver comparison';
    nav.insertBefore(button,nav.firstChild);
    const panel=document.createElement('section');panel.id='page-comparison';panel.className='report-page';panel.hidden=true;
    panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','tab-comparison');panel.innerHTML=D.comparison_markup;
    $('content').appendChild(panel);
    const INFO={p:{label:'Static pressure',unit:'bar',scale:1e-5},u:{label:'Velocity',unit:'m/s',scale:1},
      rho:{label:'Density',unit:'kg/m³',scale:1},T:{label:'Static temperature',unit:'K',scale:1},
      M:{label:'Mach number',unit:'−',scale:1},liquid:{label:'Liquid mass fraction',unit:'−',scale:1}};
    const keys=Object.keys(INFO),physicsKeys=Object.keys(C.variants);
    let caseId=null,rowId=null,active=false,configuration='';
    const finite=v=>typeof v==='number'&&Number.isFinite(v),max=a=>Math.max(...(a||[]).filter(finite));
    const v=()=>C.variants[$('cmp-physics').value];
    const entry=()=>v()?.cases?.[caseId]||{};
    const available=()=>entry().fvm||[];
    const rows=()=>available().filter(r=>String(r.cells)===$('cmp-cells').value);
    const row=()=>available().find(r=>r.id===rowId)||null;
    const space=()=>entry().space||{};
    const color=()=>document.documentElement.dataset.theme==='dark'?'#c778b8':'#9b179e';
    const method=r=>[r.scheme?.flux?.toUpperCase(),r.scheme?.reconstruction,r.scheme?.time_integrator].filter(Boolean).join(' / ');
    const values=(fields,key)=>key==='liquid'?(fields?.quality||[]).map(q=>finite(q)&&q>=0&&q<=1?1-q:null):(fields?.[key]||[]);
    const indices=(n,skip)=>{const out=[],stride=Math.max(1,Math.floor(Number(skip)||0)+1);for(let i=0;i<n;i+=stride)out.push(i);if(n&&out.at(-1)!==n-1)out.push(n-1);return out;};
    function difference(r,key){
      const a=values(r?.fields,key),b=values(r?.reference_fields,key);
      return a.map((x,i)=>!finite(x)||!finite(b[i])?null:key==='liquid'?100*Math.abs(x-b[i]):b[i]===0?null:100*Math.abs((x-b[i])/b[i]));
    }
    function configure(id,force=false){
      caseId=id;
      const mark=[id,$('cmp-physics').value].join('|');
      if(!force&&mark===configuration)return;
      configuration=mark;
      const previous=$('cmp-cells').value,levels=[...new Set(available().map(r=>r.cells))].sort((a,b)=>a-b);
      $('cmp-cells').replaceChildren(...levels.map(n=>new Option(`${n} cells`,String(n))));
      $('cmp-cells').value=levels.includes(Number(previous))?previous:String(levels.at(-1)||'');
      configureRuns();
    }
    function configureRuns(){
      const rr=rows();
      $('cmp-run').replaceChildren(...rr.map(r=>new Option(`${r.stretch===1?'Uniform':`Stretch ${r.stretch}`} · ${method(r)||'FVM'}${r.success?'':' · unfinished'}`,r.id)));
      if(!rr.some(r=>r.id===rowId))rowId=rr.find(r=>r.success&&r.stretch===1)?.id||rr.at(-1)?.id||null;
      $('cmp-run').value=rowId||'';$('cmp-run').disabled=rr.length<2;
      $('cmp-skip').max=String(Math.max(0,(row()?.x_m?.length||1)-1));
      $('cmp-skip').value=String(Math.max(0,Math.min(Number($('cmp-skip').max),Math.floor(Number($('cmp-skip').value)||0))));
    }
    function profileLayout(key){
      const info=INFO[key],l=FluxReportUI.layout('',`${info.label} [${info.unit}]`);
      l.margin={l:78,r:22,t:43,b:54};l.xaxis.title={text:'Position relative to throat [mm]'};
      l.xaxis.range=[-32.013,66.355];l.legend={orientation:'h',x:0,y:1.04,yanchor:'bottom',font:{size:10}};
      if($('cmp-onset').checked)l.shapes=[{type:'line',x0:context.info().definition.onset_mm,x1:context.info().definition.onset_mm,
        yref:'paper',y0:0,y1:1,line:{color:FluxReportUI.palette().ink,dash:'dot',width:1}}];
      return l;
    }
    function profileTraces(r,key){
      const info=INFO[key],ss=space(),traces=[],show=$('cmp-solvers').value;
      if(show!=='fvm'&&ss.success&&ss.x_m?.length){
        traces.push({type:'scatter',mode:'lines',name:'NozzleX · space marching',x:ss.x_m.map(x=>x*1000),
          y:values(ss.fields,key).map(y=>finite(y)?y*info.scale:null),line:{color:color(),width:2.1},connectgaps:false});
      }
      if(key==='p'&&$('cmp-experiment').checked)traces.push(context.experimentTrace(context.info()));
      if($('cmp-cfd').checked&&(key==='p'||key==='liquid')){
        const ref=context.referenceTrace(context.info(),key==='p'?'pressure':'liquid_fraction');if(ref)traces.push(ref);
      }
      if(show!=='space'&&r?.x_m?.length){
        const selected=indices(r.x_m.length,$('cmp-skip').value),field=values(r.fields,key);
        traces.push({type:'scatter',mode:'markers',name:`FVM · ${r.cells} cells${r.success?'':' · unfinished'}`,
          x:selected.map(i=>r.x_m[i]*1000),y:selected.map(i=>finite(field[i])?field[i]*info.scale:null),
          customdata:selected,marker:{color:'#ffffff',size:6.5,symbol:'circle',line:{color:color(),width:1.7}},connectgaps:false});
      }
      return traces;
    }
    function errorTrace(r,key){
      const d=space().success?difference(r,key):[],label=key==='liquid'?'Absolute difference [percentage points]':'Relative difference [%]';
      return {type:'scatter',mode:'lines',name:label,x:(r?.x_m||[]).map(x=>x*1000),
        y:d.map(y=>y===0?null:y),customdata:d,line:{color:color(),width:1.7},connectgaps:false,
        hovertemplate:`x = %{x:.6g} mm<br>${label} = %{y:.6e}<extra></extra>`};
    }
    function linkZoom(key){
      const first=$('cmp-field-'+key),second=$('cmp-error-'+key);if(first._cmpZoom)return;
      let busy=false;
      function copy(target,event){
        if(busy)return;let range=null;
        if(event['xaxis.range'])range=event['xaxis.range'];
        else if(event['xaxis.range[0]']!==undefined)range=[event['xaxis.range[0]'],event['xaxis.range[1]']];
        else if(event['xaxis.autorange'])range=[-32.013,66.355];
        if(range){busy=true;Promise.resolve(Plotly.relayout(target,{'xaxis.range':range,'xaxis.autorange':false})).finally(()=>{busy=false;});}
      }
      first.on('plotly_relayout',e=>copy(second,e));second.on('plotly_relayout',e=>copy(first,e));first._cmpZoom=true;
    }
    function geometryPlot(){
      const g=D.geometry_profile;if(!g)return;
      const x=g.x_m.map(v=>v*1000),y=g.half_height_m.map(v=>v*1000),l=FluxReportUI.layout('','Wall coordinate [mm]');
      l.margin={l:78,r:22,t:30,b:55};l.xaxis.title={text:'Position relative to throat [mm]'};l.xaxis.range=[x[0],x.at(-1)];
      l.yaxis.range=[-7,7];l.showlegend=false;
      const throat=C.geometry?.geometric_throat_x_m;
      if(finite(throat))l.shapes=[{type:'line',x0:throat*1000,x1:throat*1000,yref:'paper',y0:0,y1:1,line:{color:FluxReportUI.palette().ink,dash:'dot',width:1}}];
      U.plot('cmp-geometry',[
        {type:'scatter',mode:'lines',name:'Upper wall',x,y,line:{color:color(),width:2},hovertemplate:'x = %{x:.5g} mm<br>y = %{y:.5g} mm<extra></extra>'},
        {type:'scatter',mode:'lines',name:'Lower wall',x,y:y.map(v=>-v),line:{color:color(),width:2},hovertemplate:'x = %{x:.5g} mm<br>y = %{y:.5g} mm<extra></extra>'},
        {type:'scatter',mode:'markers',name:'Corrected measured contour points',x:g.control_x_m.map(v=>v*1000),y:g.control_half_height_m.map(v=>v*1000),marker:{color:'#ffffff',size:4.5,line:{color:color(),width:1.1}}}
      ],l);
    }
    function sensitivity(){
      const ss=C.reference_sensitivity;if(!ss)return;
      const rows=Object.entries(ss.cases||{}).flatMap(([caseId,variants])=>Object.entries(variants).map(([physics,row])=>({caseId,physics,...row})));
      $('cmp-sensitivity').hidden=!rows.length;
      $('cmp-sensitivity-head').innerHTML='<tr><th>Case / wall physics</th><th>Status</th><th>Max p / u / ρ / T / M change [%]</th><th>|Critical flow change| [%]</th><th>Sonic location change [μm]</th><th>Common finite points</th></tr>';
      $('cmp-sensitivity-body').innerHTML=rows.map(r=>`<tr><td>${E(r.caseId)} / ${E(C.variants[r.physics]?.label||r.physics)}</td><td>${r.success?'Completed':E(r.reason||r.refined_reason||'Not completed')}</td><td>${['p','u','rho','T','M'].map(k=>F(100*r.fields?.[k]?.max_relative_difference)).join(' / ')}</td><td>${F(100*r.critical?.mass_flow_absolute_relative_change)}</td><td>${F(1e6*r.critical?.sonic_location_change_m)}</td><td>${E(r.common_finite_coordinate_count??'—')}</td></tr>`).join('');
      $('cmp-sensitivity-raw').textContent=JSON.stringify(ss,null,2);
    }
    function render(id){
      configure(id);
      const r=row(),ss=space(),critical=ss.critical||{},cp=context.info();
      $('cmp-cfd').disabled=!cp.upstream_cfd;
      const shown=indices(r?.x_m?.length||0,$('cmp-skip').value).length;
      $('cmp-heading').textContent=`${id} · ${v()?.label||$('cmp-physics').value}`;
      $('selection-summary').innerHTML=`<strong>${E(id)}</strong> · ${E(v()?.label||$('cmp-physics').value)} · ${r?r.cells+' cells':'no completed FVM run'} · ${E($('cmp-solvers').selectedOptions[0]?.textContent||'')}`;
      $('empty-selection').hidden=true;
      $('cmp-marker-count').textContent=`${shown} / ${r?.x_m?.length||0} FVM markers; every saved cell is retained in the difference curves and download.`;
      const flow=critical.mass_flow_kg_s??ss.fields?.mdot?.[0];
      $('cmp-status').innerHTML=S('Space-marching reference',ss.success?'Completed':'Not completed',ss.reason||'Recorded integration status')+
        S('Selected FVM root',r?.success?'Converged':r?.reason||'Not available',r?`${r.cells} cells · ${method(r)}`:'No saved FVM result')+
        S('Critical mass flow',F(flow),'kg/s · independent space march')+S('Critical inlet Mach',F(critical.inlet_mach),'From sonic regularity and reservoir constraints');
      const notices=[];
      if(!ss.success)notices.push('No completed space-marching reference is available. Reference curves and solver differences are suppressed; the failure evidence remains below.');
      if(r&&!r.success)notices.push('The selected FVM state is unfinished. Its plotted differences are diagnostic and are not verification conclusions.');
      if(r?.saved_verification?.verified===false)notices.push('The selected FVM state does not pass its saved original-equation and physical-budget audit.');
      if(!r)notices.push('No saved FVM grid is available for this case and wall-physics selection.');
      if(!cp.upstream_cfd)notices.push('No reference CFD curve was supplied for this operating case.');
      $('cmp-notice').textContent=notices.join(' ');$('cmp-notice').hidden=!notices.length;
      $('cmp-reference-mode').textContent=r?`${r.reference_sampling}. The tabulated property and sound-speed conventions are unchanged.`:'';
      geometryPlot();
      sensitivity();
      for(const key of keys){
        const profile=U.plot('cmp-field-'+key,profileTraces(r,key),profileLayout(key));
        const l=profileLayout(key);delete l.shapes;l.title={text:key==='liquid'?'Absolute difference [percentage points]':'Relative difference [%]',x:.06,font:{size:12}};
        l.margin={l:78,r:22,t:42,b:50};l.yaxis.title={text:key==='liquid'?'Percentage points':'Percent'};l.yaxis.type='log';l.yaxis.dtick=1;l.yaxis.tickformat='.3g';l.showlegend=false;
        const error=U.plot('cmp-error-'+key,r&&ss.success?[errorTrace(r,key)]:[],l);
        Promise.all([profile,error]).then(()=>linkZoom(key));
      }
      const numericalFlow=r?.saved_verification?.mass_flow?.numerical_inlet_kg_s??r?.numerical_mass_flow_kg_s;
      $('cmp-table').innerHTML=`<tr><td>NozzleX · ${E(ss.success?'completed':ss.reason||'not completed')}</td><td>${F(critical.inlet_mach)}</td><td>${F(flow)}</td><td>ODE conservation / integration checks below</td><td>Independent sonic regularity</td></tr>`+
        (r?`<tr><td>FVM · ${r.cells} cells · ${E(r.reason||'recorded')}</td><td>${F(r.inlet_mach)}</td><td>${F(numericalFlow)}</td><td>${F(max(r.metrics?.equation_norm))}</td><td>${E(r.initialization||'Preserved saved FVM state')}</td></tr>`:'');
      const audit=r?.saved_verification;
      $('cmp-mesh-heading').textContent=`Grid-sequencing accuracy · ${v()?.label||''}`;
      $('cmp-mesh-rows').innerHTML=Object.entries(v()?.cases||{}).map(([label,item])=>{
        const at=n=>(item.fvm||[]).filter(x=>x.cells===n&&x.success).at(-1),fine=at(200);
        const flow=fine?.saved_verification?.mass_flow?.numerical_inlet_kg_s,reference=item.space?.critical?.mass_flow_kg_s;
        const mass=finite(flow)&&finite(reference)&&reference!==0?100*Math.abs(flow/reference-1):null;
        return `<tr><td>${label===id?'<strong>'+E(label)+'</strong>':E(label)}</td>${[50,100,200].map(n=>'<td>'+F(at(n)?.errors?.p?.relative_percent_rms)+'</td>').join('')}<td>${F(item.previous_constant_comparison?.pressure_rms_percent)}</td><td>${F(mass)}</td></tr>`;
      }).join('');
      $('cmp-audits').innerHTML=`<tr><td>${F(ss.audits?.critical_equations_max)}</td><td>${F(100*ss.audits?.mass_flow_relative_range)}</td><td>${F(100*ss.audits?.energy_relative_range)}</td><td>${F(max(audit?.equation_l2||r?.metrics?.equation_norm))}</td><td>${F(audit?.boundary_max)}</td><td>${F(max(audit?.global_defect))}</td><td>${audit?B(audit.verified,'Pass','Does not pass'):'Not supplied'}</td></tr>`;
      $('cmp-critical').textContent=JSON.stringify({physics:$('cmp-physics').value,case:id,
        space:{success:ss.success,reason:ss.reason,critical:ss.critical,audits:ss.audits,source:ss.source,profile_station_normalization:ss.profile_station_normalization},
        retained_space_attempts:(entry().space_attempts||[]).map(a=>({success:a.success,reason:a.reason,critical:a.critical,audits:a.audits,source_file:a.source_file,settings:a.settings,profile_points:a.x_m?.length||0})),
        reference_refinements:(entry().reference_refinements||[]).map(a=>({success:a.success,reason:a.reason,critical:a.critical,audits:a.audits,source_file:a.source_file,settings:a.settings,profile_points:a.x_m?.length||0})),
        fvm:r?{id:r.id,cells:r.cells,success:r.success,reason:r.reason,scheme:r.scheme,metrics:r.metrics,
          saved_verification:r.saved_verification,errors:r.errors,reference_sampling:r.reference_sampling,source_epoch:r.source_epoch,initialization:r.initialization}:null},null,2);
    }
    function activate(page){
      active=page==='comparison';document.body.classList.toggle('comparison-mode',active);
    }
    function csv(){
      const r=row();if(!r)return '';
      const columns=['x_m',...keys.flatMap(k=>[`${k}_fvm`,`${k}_space`,`${k}_${k==='liquid'?'absolute_percentage_points':'relative_percent'}`])];
      const a=Object.fromEntries(keys.map(k=>[k,values(r.fields,k)])),b=Object.fromEntries(keys.map(k=>[k,values(r.reference_fields,k)]));
      const d=Object.fromEntries(keys.map(k=>[k,difference(r,k)]));
      return [columns.join(','),...r.x_m.map((x,i)=>[x,...keys.flatMap(k=>[a[k][i]??'',b[k][i]??'',d[k][i]??''])].join(','))].join('\n');
    }
    $('cmp-physics').replaceChildren(...physicsKeys.map(key=>new Option(C.variants[key].label||key,key)));
    $('cmp-physics').value=physicsKeys.includes('inviscid')?'inviscid':physicsKeys[0];
    $('cmp-physics').addEventListener('change',()=>{configure(context.info().id,true);context.redraw();});
    $('cmp-cells').addEventListener('change',()=>{configureRuns();context.redraw();});
    $('cmp-run').addEventListener('change',()=>{rowId=$('cmp-run').value;configureRuns();context.redraw();});
    for(const id of ['cmp-solvers','cmp-experiment','cmp-cfd','cmp-onset'])$(id).addEventListener('change',context.redraw);
    $('cmp-skip').addEventListener('change',()=>{configureRuns();context.redraw();});
    $('cmp-download').addEventListener('click',()=>{
      const url=URL.createObjectURL(new Blob([csv()],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');
      a.href=url;a.download=`lettieri_${caseId}_${$('cmp-physics').value}_N${row()?.cells||0}_solver_comparison.csv`;
      document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
    });
    return {render,activate,available:()=>!!physicsKeys.length,snapshot:()=>({active,physics:$('cmp-physics').value,caseId,cells:$('cmp-cells').value,rowId,solvers:$('cmp-solvers').value,skip:Number($('cmp-skip').value)}),
      row,space,difference,profileTraces,errorTrace,indices,csv};
  }};
})();
