/* Shared document navigation and themes; numerical results are read-only. */
(() => {
  let active='results', ready=Promise.resolve(), resolveMath, resolveResults;
  const mathReady=new Promise(resolve=>{resolveMath=resolve;});
  const resultsReady=new Promise(resolve=>{resolveResults=resolve;});
  window.addEventListener('report-math-ready',()=>resolveMath());
  window.addEventListener('report-results-ready',()=>resolveResults());
  const $=id=>document.getElementById(id);
  function activate(tab,target=tab) {
    if(!['results','theory'].includes(tab))tab='results';
    active=tab;window.backpressureReport?.stopAnimation();
    document.body.dataset.page=tab;document.documentElement.dataset.reportReady='false';
    document.querySelectorAll('[data-report-page]').forEach(e=>{e.hidden=e.dataset.reportPage!==tab;});
    document.querySelectorAll('[data-report-nav]').forEach(e=>{e.hidden=e.dataset.reportNav!==tab;});
    document.querySelectorAll('[data-report-tab]').forEach(e=>{if(e.dataset.reportTab===tab)e.setAttribute('aria-current','page');else e.removeAttribute('aria-current');});
    $('sidebar-label').textContent=tab==='results'?'Results':'Theory & background';
    ready=Promise.all([mathReady,resultsReady]).then(async()=>{
      if(tab==='results')await window.backpressureReport.refreshTheme();
      if(active!==tab)return;
      document.documentElement.dataset.reportReady='true';
      if(target===tab)window.scrollTo({top:0,left:0,behavior:'instant'});
      else $(target)?.scrollIntoView({behavior:'instant'});
    }).catch(error=>{document.documentElement.dataset.reportError=error.message;console.error(error);});
    return ready;
  }
  function navigate(){const target=decodeURIComponent(location.hash.slice(1))||'results';return activate(target.startsWith('theory')?'theory':'results',target);}
  document.addEventListener('click',event=>{
    const a=event.target.closest('a[href^="#"]');if(!a)return;
    const target=a.getAttribute('href').slice(1);if(!$(target))return;
    event.preventDefault();history.pushState(null,'',`#${target}`);activate(target.startsWith('theory')?'theory':'results',target);
  });
  window.addEventListener('hashchange',navigate);
  document.querySelectorAll('[data-theme-choice]').forEach(button=>{
    button.setAttribute('aria-pressed',String(button.dataset.themeChoice===document.documentElement.dataset.theme));
    button.addEventListener('click',()=>{
      const theme=button.dataset.themeChoice;document.documentElement.dataset.theme=theme;
      document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
      try{localStorage.setItem('nozzle-backpressure-sensitivity.report.theme',theme);}catch{}
      activate(active);
    });
  });
  $('print-report').addEventListener('click',async()=>{
    await Promise.all([mathReady,resultsReady]);window.backpressureReport.stopAnimation();
    document.body.classList.add('printing');await window.backpressureReport.refreshTheme();window.print();document.body.classList.remove('printing');activate(active);
  });
  $('license-button').addEventListener('click',()=>{const d=$('license-details');d.hidden=false;d.open=!d.open;d.scrollIntoView();});
  window.backpressureDocument={selectTab:activate,ready:()=>ready,get active(){return active;}};
  navigate();
})();