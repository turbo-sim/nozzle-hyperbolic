/* Shared offline report interactions: theme, accessible tabs and Plotly tokens. */
(() => {
  'use strict';
  const KEY = 'nh-flux-report-theme';
  const modes = ['auto', 'light', 'dark'];
  let preference = 'auto', controller = null, change = () => {};
  const media = typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function resolved() { return preference === 'auto' ? (media?.matches ? 'dark' : 'light') : preference; }
  function palette() {
    const dark = document.documentElement.dataset.theme === 'dark';
    return dark ? {canvas:'#111c27',paper:'#192936',ink:'#e5eff8',muted:'#adc0ce',line:'#354d60',grid:'#304759',accent:'#a4c8fa',bg:'#111c27',card:'#192936',fg:'#e5eff8',border:'#354d60'} :
      {canvas:'#f1f5f9',paper:'#ffffff',ink:'#183247',muted:'#586e81',line:'#dbe4ed',grid:'#e1e9f1',accent:'#3568ac',bg:'#f1f5f9',card:'#ffffff',fg:'#183247',border:'#dbe4ed'};
  }
  function resize() {
    window.dispatchEvent(new Event('resize'));
    if (window.Plotly?.Plots?.resize) document.querySelectorAll('.js-plotly-plot').forEach(element => { if (!element.closest('[hidden]')) window.Plotly.Plots.resize(element); });
  }
  function apply(notify=true) {
    const theme = resolved();
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    if (document.body) {
      document.body.classList.remove('dark');
      if (theme === 'dark') document.body.classList.add('dark');
    }
    const select = document.getElementById('report-theme');
    if (select) select.value = preference;
    if (notify) change(theme, preference);
  }
  function setTheme(mode) {
    preference = modes.includes(mode) ? mode : 'auto';
    try { localStorage.setItem(KEY, preference); } catch (_error) { /* Local files may disable storage. */ }
    apply();
  }
  function init({onThemeChange=()=>{}}={}) {
    change = onThemeChange;
    try { const saved = localStorage.getItem(KEY); preference = modes.includes(saved) ? saved : 'auto'; } catch (_error) { preference = 'auto'; }
    const select = document.getElementById('report-theme');
    if (select && !select.dataset.reportThemeBound) {
      select.addEventListener('change', () => setTheme(select.value));
      select.dataset.reportThemeBound = 'true';
    }
    if (media && !controller) {
      const listener = () => { if (preference === 'auto') apply(); };
      if (typeof media.addEventListener === 'function') media.addEventListener('change', listener);
      else if (typeof media.addListener === 'function') media.addListener(listener);
    }
    controller = {setTheme,getTheme:()=>({preference,resolved:resolved()}),resize};
    apply(false);
    return controller;
  }
  function layout(title='',y='') {
    const p = palette();
    return {title:{text:title,font:{size:17}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
      font:{family:'Segoe UI, Arial, sans-serif',size:13,color:p.ink},
      margin:{l:78,r:25,t:55,b:83},xaxis:{title:{text:'Flow coordinate [m]',standoff:12},gridcolor:p.grid,zerolinecolor:p.grid,automargin:true},
      yaxis:{title:{text:y,standoff:12},gridcolor:p.grid,zerolinecolor:p.grid,automargin:true},
      legend:{orientation:'h',x:0,y:-.24,font:{size:11},bgcolor:'rgba(0,0,0,0)'},
      hovermode:'closest',hoverlabel:{font:{family:'Segoe UI, Arial, sans-serif',size:13}},colorway:['#6a00a8','#9c179e','#cc4778','#ed7953']};
  }
  function bindTabs({nav,panels,onChange=()=>{}}) {
    nav = typeof nav === 'string' ? document.querySelector(nav) : nav;
    const buttons = Array.from(nav.querySelectorAll('button[aria-controls]'));
    panels = typeof panels === 'string' ? Array.from(document.querySelectorAll(panels)) : Array.from(panels || buttons.map(button => document.getElementById(button.getAttribute('aria-controls'))));
    function activate(button,focus=false) {
      const target = button.getAttribute('aria-controls');
      buttons.forEach(item => { const active = item === button; item.setAttribute('aria-selected',String(active)); item.setAttribute('tabindex',active?'0':'-1'); item.classList.toggle('active',active); });
      panels.forEach(panel => { const active = panel.id === target; panel.hidden = !active; panel.classList.toggle('active',active); });
      if (focus) button.focus();
      onChange(target);
      resize();
    }
    buttons.forEach((button,index) => {
      button.addEventListener('click', () => activate(button));
      button.addEventListener('keydown', event => {
        let next = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + buttons.length - 1) % buttons.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = buttons.length - 1;
        if (next !== null) { event.preventDefault(); activate(buttons[next],true); }
      });
    });
    activate(buttons.find(button=>button.getAttribute('aria-selected')==='true') || buttons[0]);
    return {activate:id=>{const button=buttons.find(item=>item.getAttribute('aria-controls')===id);if(button)activate(button);}};
  }
  window.FluxReportUI = {init,setTheme,palette,layout,bindTabs,resize,themeKey:KEY};
})();
