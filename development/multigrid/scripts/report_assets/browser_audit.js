/* Instrumented copy only. Exercises a local report in a fresh headless browser. */
(() => {
  'use strict';
  const results = [], errors = [], blobs = new Map(), downloads = [];
  const output = document.createElement('pre');
  output.id = 'browser-qa-result';
  output.style.cssText = 'white-space:pre-wrap;max-height:none;padding:12px';
  document.body.appendChild(output);
  const write = value => { output.textContent = JSON.stringify(value, null, 2); };
  write({passed:false,status:'pending',report_sha256:window.BROWSER_QA_REPORT_SHA256});
  const check = (condition, name) => { if(!condition) throw new Error(name); results.push(name); if(results.length%5===0)write({passed:false,status:'running',report_sha256:window.BROWSER_QA_REPORT_SHA256,checks:results.length,results}); };
  const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
  const wait = ms => new Promise(resolve => setTimeout(resolve,ms));
  const settle = async () => { await wait(150); };
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  addEventListener('error',event=>errors.push(event.message || 'browser error'));
  addEventListener('unhandledrejection',event=>errors.push(String(event.reason)));

  async function run() {
    try {
      const api = window.MultigridReport, data = api.state.data;
      const dataBefore = JSON.stringify(data);
      check(document.documentElement.dataset.reportReady === 'true','report initialized');
      check(location.hash === '#explorer','actual direct explorer link opens');
      const allMethods=data.groups.flatMap(group=>group.methods);
      const passed=allMethods.filter(method=>api.original(method.representative||method.warmup||{})===true).length;
      check(document.getElementById('hero-count').textContent === `${passed} / ${allMethods.length}`,'deep-link hero has verified count');
      check(!document.getElementById('hero-context').textContent.includes('Loading'),'deep-link loading placeholder cleared');
      const complete=allMethods.filter(method=>method.samples?.length===method.requested_repeats&&method.samples.length>0);
      const eligible=complete.filter(method=>method.timing_eligible!==false&&method.samples.every(record=>record.success!==false&&api.original(record)===true));
      check(document.getElementById('kpi-eligible').textContent === `${eligible.length} / ${complete.length}`,'timing eligibility remains distinct from native convergence');

      for(const page of ['overview','explorer','theory','protocol']) {
        document.querySelector(`.navigation button[data-page="${page}"]`).click();
        await settle();
        check(document.querySelectorAll('.page.active').length === 1 && document.getElementById(page).classList.contains('active'),'page control: '+page);
        if(page === 'theory') {
          if(window.MathJax?.startup?.promise) await MathJax.startup.promise;
          if(window.MathJax?.typesetPromise) await MathJax.typesetPromise([document.getElementById('theory')]);
          check(document.querySelectorAll('#theory mjx-container svg').length > 0,'actual MathJax SVG mathematics renders');
        }
      }
      if(data.tuning?.records?.length){
        const pilots=data.tuning.records.filter(record=>record.kind!=='matched_cfl_diagnostic');
        const diagnostics=data.tuning.records.filter(record=>record.kind==='matched_cfl_diagnostic');
        check(document.querySelectorAll('#tuning-table tr').length===pilots.length,'every tuning pilot is displayed separately');
        check(document.querySelectorAll('#matched-table tr').length===diagnostics.length,'matched controller samples are separate from primary and pilot populations');
        check(same(JSON.parse(document.getElementById('tuning-record').textContent),data.tuning),'full tuning and diagnostic record values unchanged');
      }
      api.navigate('explorer'); await settle();
      for(const group of data.groups) {
        const caseControl = document.getElementById('physical-case');
        const reconControl = document.getElementById('reconstruction');
        caseControl.value = group.case_id;
        caseControl.dispatchEvent(new Event('change',{bubbles:true}));
        await settle();
        reconControl.value = group.reconstruction;
        reconControl.dispatchEvent(new Event('change',{bubbles:true}));
        await settle();
        check(api.currentGroup() === group,'case and reconstruction selection: '+group.case_id+'/'+group.reconstruction);
        const outcomes=document.querySelectorAll('#case-table tr');
        group.methods.forEach((method,index)=>{
          const levels=(method.representative||method.warmup||{}).work?.levels;
          if(Array.isArray(levels)&&levels.length&&levels.every(row=>Number.isFinite(row.rejected_steps))){
            const expected=levels.reduce((sum,row)=>sum+row.rejected_steps,0);
            const cells=outcomes[index].querySelectorAll('td');
            check(Number(cells[cells.length-1].textContent.replaceAll(',',''))===expected,'all-grid rejected implicit steps: '+group.case_id+'/'+group.reconstruction+'/'+method.method_id);
          }
        });
        const graph = document.getElementById('profile-p');
        check(graph._fullData?.length > 0 && graph.getBoundingClientRect().width > 0,'actual Plotly profile renders: '+group.case_id+'/'+group.reconstruction);
        for(const method of group.methods) {
          const rep = method.representative || method.warmup;
          const field = api.fields(rep || {});
          if(!field.p?.length) continue;
          const label = method.label || ({sequence:'Grid sequencing',fas_fmg:'FAS with coarse-to-fine initialization'}[method.method_id] || method.method_id);
          const trace = graph._fullData.find(item => item.name.startsWith(label));
          check(Boolean(trace),'actual numerical trace: '+group.case_id+'/'+group.reconstruction+'/'+method.method_id);
          check(same(Array.from(trace.y),field.p),'actual Plotly pressure array matches saved values: '+group.case_id+'/'+group.reconstruction+'/'+method.method_id);
        }
        const initialization=group.methods.flatMap(method=>api.initializationSummary(method.representative||method.warmup||{}));
        if(initialization.length){check(document.querySelectorAll('#initialization-table tr').length===initialization.length,'all initialization stages displayed: '+group.case_id+'/'+group.reconstruction);check(document.getElementById('history-caption').textContent.includes('before fine cycle 1'),'FMG work is explicit before fine cycles: '+group.case_id+'/'+group.reconstruction);}
        const sequence=group.methods.find(method=>method.method_id==='sequence');
        if(sequence){
          const record=sequence.representative||sequence.warmup, stages=api.sequenceStages(record);
          const scope=document.getElementById('history-scope'), axis=document.getElementById('residual-axis'), mode=document.getElementById('residual-mode');
          scope.value='all';axis.value='work';mode.value='absolute';scope.dispatchEvent(new Event('change',{bubbles:true}));await settle();
          let cumulative=0;
          for(const stage of stages){
            const trace=document.getElementById('residual-chart')._fullData.find(trace=>trace.name.includes(`N=${stage.cells} (${stage.accepted_steps} steps)`));
            check(Boolean(trace),'all-grid residual trace exists: '+group.case_id+'/'+group.reconstruction+'/'+stage.cells);
            check(trace.x[0]===cumulative&&trace.x.at(-1)===cumulative+stage.accepted_steps,'cumulative accepted-step indices: '+group.case_id+'/'+group.reconstruction+'/'+stage.cells);
            const values=stage.rows.filter(row=>row.phase!=='rejected'&&row.accepted!==false&&Math.max(...row.equation_norm)>0).map(row=>Math.max(...row.equation_norm));
            check(same(Array.from(trace.y),values),'saved grid residual values unchanged: '+group.case_id+'/'+group.reconstruction+'/'+stage.cells);
            cumulative+=stage.accepted_steps;
          }
          check(document.querySelectorAll('#sequence-stage-table tr').length===stages.length,'all grids in iteration ledger: '+group.case_id+'/'+group.reconstruction);
          scope.value='fine';scope.dispatchEvent(new Event('change',{bubbles:true}));await settle();
          const fine=document.getElementById('residual-chart')._fullData.filter(trace=>trace.name.includes('N='));
          check(fine.length===1&&fine[0].x[0]===0,'finest-only scope control: '+group.case_id+'/'+group.reconstruction);
          scope.value='all';axis.value='time';axis.dispatchEvent(new Event('change',{bubbles:true}));await settle();
          const progress=record.work?.stage_progress||[];
          const markers=document.getElementById('residual-chart')._fullData.filter(trace=>trace.name.includes('stage end'));
          check(markers.every(trace=>trace.mode==='markers'&&progress.some(row=>row.elapsed_s===trace.x[0])),'only actual stage-end wall times: '+group.case_id+'/'+group.reconstruction);
          axis.value='work';
          for(const value of ['absolute','relative','boundary','global']){mode.value=value;mode.dispatchEvent(new Event('change',{bubbles:true}));await settle();check(document.getElementById('residual-chart')._fullData.every(trace=>Array.from(trace.y).every(y=>y>0)),'residual quantity control: '+group.case_id+'/'+group.reconstruction+'/'+value);}
          mode.value='absolute';mode.dispatchEvent(new Event('change',{bubbles:true}));await settle();
        }
      }
      const counter=document.getElementById('work-metric');
      for(const value of Array.from(counter.options,option=>option.value)){
        counter.value=value;counter.dispatchEvent(new Event('change',{bubbles:true}));await settle();
        check(document.getElementById('work-chart')._fullData?.length>0,'work counter control: '+value);
      }

      const firstToggle = document.querySelector('#methods input');
      const toggledMethod = api.currentGroup().methods.find(method=>method.method_id===firstToggle.value);
      const toggleLabel = toggledMethod.label || firstToggle.value;
      firstToggle.click(); await settle();
      check(!api.state.enabled.has(firstToggle.value),'method checkbox disables method');
      check(!document.getElementById('profile-p')._fullData.some(trace=>trace.name.startsWith(toggleLabel)),'disabled method removed from actual Plotly');
      firstToggle.click(); await settle();
      check(api.state.enabled.has(firstToggle.value),'method checkbox restores method');

      for(const theme of ['dark','light','auto']) {
        document.getElementById('theme').value = theme;
        document.getElementById('theme').dispatchEvent(new Event('change',{bubbles:true}));
        await settle();
        check(api.state.theme === theme,'theme control: '+theme);
        if(theme!=='auto') check(document.documentElement.dataset.theme===theme,'actual document theme: '+theme);
        check(Boolean(document.getElementById('profile-p')._fullLayout?.paper_bgcolor),'actual Plotly theme layout available: '+theme);
      }

      const originalCreate = URL.createObjectURL.bind(URL);
      const originalRevoke = URL.revokeObjectURL.bind(URL);
      const originalClick = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = blob => {const url=originalCreate(blob);blobs.set(url,blob);return url;};
      URL.revokeObjectURL = () => {};
      HTMLAnchorElement.prototype.click = function(){if(this.download){downloads.push({name:this.download,url:this.href});return;}return originalClick.call(this);};
      for(const id of ['download-data','download-csv','download-fields','download-history']) document.getElementById(id).click();
      await settle();
      check(downloads.length===4 && blobs.size===4,'four real embedded Blob downloads produced without filesystem saves');
      const text = name => {const item=downloads.find(value=>value.name===name);const request=new XMLHttpRequest();request.open('GET',item.url,false);request.send();check(request.status===200,'local Blob URL read: '+name);return request.responseText;};
      check(same(JSON.parse(text('multigrid_complete_campaign.json')),data),'complete embedded JSON download matches data');
      const expectedSamples=data.groups.flatMap(group=>group.methods).reduce((sum,method)=>sum+method.samples.length,0);
      check(text('multigrid_timing_samples.csv').split('\r\n').length===expectedSamples+1,'CSV download preserves every measured sample');
      const downloadedFields=JSON.parse(text('multigrid_selected_profiles.json'));
      const selectedGroup=api.currentGroup();
      check(downloadedFields.case_id===selectedGroup.case_id && downloadedFields.reconstruction===selectedGroup.reconstruction,'selected profile download matches controls');
      for(const item of downloadedFields.methods){const method=selectedGroup.methods.find(method=>method.method_id===item.method_id);check(same(item.fields,api.fields(method.representative||method.warmup)),'profile Blob data matches saved record: '+item.method_id);}
      const downloadedHistory=JSON.parse(text('multigrid_selected_histories.json'));
      check(downloadedHistory.case_id===selectedGroup.case_id,'selected history download matches controls');
      for(const item of downloadedHistory.methods){const method=selectedGroup.methods.find(method=>method.method_id===item.method_id);check(same(item.fine_history,api.history(method.representative||method.warmup)),'history Blob data matches saved record: '+item.method_id);}
      for(const item of downloadedHistory.methods){const method=selectedGroup.methods.find(method=>method.method_id===item.method_id);check(same(item.stage_history,(method.representative||method.warmup).stage_history||[]),'all-grid history Blob data matches saved record: '+item.method_id);}
      HTMLAnchorElement.prototype.click=originalClick;URL.createObjectURL=originalCreate;URL.revokeObjectURL=originalRevoke;
      for(const url of blobs.keys()) originalRevoke(url);

      check(same(JSON.parse(dataBefore),data),'all browser interactions preserve embedded campaign values');
      const network=performance.getEntriesByType('resource').filter(entry=>/^https?:/i.test(entry.name)).map(entry=>entry.name);
      check(network.length===0,'zero HTTP(S) runtime resource requests');
      check(errors.length===0,'no uncaught browser application errors');
      write({passed:true,status:'complete',report_sha256:window.BROWSER_QA_REPORT_SHA256,browser:navigator.userAgent,actual_browser:true,real_plotly:true,real_mathjax:true,groups:data.groups.length,measured_samples:expectedSamples,startup_probes:data.startup_probes?.records?.length||0,checks:results.length,results,downloads:downloads.map(value=>value.name),network_requests:network,errors,note:'Isolated headless Chrome profile; original REPORT.html was not instrumented or modified.'});
      document.documentElement.dataset.browserQaStatus='passed';
    } catch(error) {
      write({passed:false,status:'failed',report_sha256:window.BROWSER_QA_REPORT_SHA256,browser:navigator.userAgent,actual_browser:true,checks:results.length,results,errors:[...errors,String(error.stack||error)]});
      document.documentElement.dataset.browserQaStatus='failed';
    }
  }
  if(window.MultigridReport && document.documentElement.dataset.reportReady==='true') run();
  else addEventListener('multigrid-report-ready',run,{once:true});
})();
