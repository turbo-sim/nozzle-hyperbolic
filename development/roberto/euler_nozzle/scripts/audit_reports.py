"""Check solitary HTML artifacts using an isolated headless Edge profile.

No live browser, user profile, external server or numerical solve is involved.
"""

from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

from .baseline_common import STUDY, save

EDGE=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.body={}
        self.math_errors=[]
        self.math_count=0

    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=="body":
            self.body=values
        if tag=="mjx-container":
            self.math_count+=1
        if values.get("data-mml-node")=="merror":
            self.math_errors.append(values)


INSTRUMENT=r"""
<script>
window.addEventListener('error',e=>{document.body.dataset.auditError=String(e.message);});
window.addEventListener('unhandledrejection',e=>{document.body.dataset.auditError=String(e.reason);});
async function checkReport(){
  try {
    if(window.MathJax&&MathJax.startup)await MathJax.startup.promise;
    await render();
    const duplicates=[...document.querySelectorAll('[id]')].map(el=>el.id).filter((id,i,a)=>a.indexOf(id)!==i);
    const cases=[...$('case-select').options].map(el=>el.value);
    const meshes=[...$('mesh-select').options].map(el=>el.value);
    let checked=0;
    for(const condition of cases){
      $('case-select').value=condition;
      $('mesh-select').value=meshes.at(-1);
      await render();
      for(const el of document.querySelectorAll('.js-plotly-plot')){
        if(!el.data.length)throw new Error('Empty plot '+el.id);
        for(const trace of el.data){if(trace.x.length!==trace.y.length)throw new Error('Length mismatch '+el.id);}
      }
      checked++;
    }
    $('case-select').value=cases[0];
    for(const mesh of meshes){$('mesh-select').value=mesh;await render();checked++;}
    const details=[...document.querySelectorAll('details.block')];
    for(const detail of details){detail.open=false;if(detail.open)throw new Error('Cannot collapse');detail.open=true;}
    $('all-grids').checked=true;tableRows();
    const allRows=$('table-content').querySelectorAll('tbody tr').length;
    $('all-grids').checked=false;tableRows();
    const fineRows=$('table-content').querySelectorAll('tbody tr').length;
    showPage('theory');
    if(document.querySelector('[data-report-page=theory]').hidden)throw new Error('Theory not visible');
    const mathOverflow=[...document.querySelectorAll('mjx-container[display=true]')].filter(e=>e.scrollWidth>e.clientWidth+3).length;
    showPage('reproduce');
    if(document.querySelector('[data-report-page=reproduce]').hidden)throw new Error('Reproduce not visible');
    showPage('results');
    document.documentElement.dataset.theme=location.hash.includes('dark')?'dark':'light';
    $('mesh-select').value=meshes.at(-1);
    $('case-select').value=steady?'shock_70':cases.at(-1);
    await render();
    document.body.dataset.browserAudit=JSON.stringify({checked,duplicates,sections:details.length,allRows,fineRows,mathOverflow,
      page:document.body.dataset.page,theme:document.documentElement.dataset.theme,
      plots:document.querySelectorAll('.js-plotly-plot').length,
      documentWidth:document.documentElement.scrollWidth,viewportWidth:innerWidth,
      error:document.body.dataset.auditError||null});
    // Edge's CLI can capture a blank compositor frame after a long scroll.
    // Crop the isolated artifact to the history section instead of scrolling.
    if(location.hash.includes('history')||location.hash.includes('fields')){
      const keep=location.hash.includes('history')?'history':'fields';
      for(const child of document.querySelector('[data-report-page=results]').children)
        if(child.id!==keep)child.style.display='none';
      window.scrollTo(0,0);
    }
  }catch(error){document.body.dataset.browserAudit=JSON.stringify({error:String(error),stack:error.stack});}
}
window.addEventListener('load',checkReport);
</script>
"""


def main():
    if not EDGE.exists():
        raise RuntimeError("Existing Edge installation is required for this optional artifact audit")
    output=STUDY/"outputs/html_audit"
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for kind,filename in (("steady","STEADY_REPORT.html"),("sod","SOD_REPORT.html")):
        source=STUDY/filename
        html=source.read_text(encoding="utf-8")
        assert '<script src=' not in html and '<link rel="stylesheet"' not in html
        isolated=output/f"standalone_{kind}.html"
        isolated.write_text(html.replace('</body>',INSTRUMENT+'</body>'),encoding="utf-8")
        for capture,hash_value,width in (("light","",1600),("dark","#history-dark",1280),("fields","#fields-light",1600)):
            stem=f"{kind}_{capture}"
            command=[str(EDGE),"--headless=new","--no-first-run","--disable-background-networking",
                     "--run-all-compositor-stages-before-draw",f"--user-data-dir={output/('profile_'+stem)}",
                     f"--window-size={width},1100","--virtual-time-budget=45000","--dump-dom",
                     f"--screenshot={output/(stem+'.png')}",isolated.as_uri()+hash_value]
            with (output/(stem+'.html')).open('wb') as stdout,(output/(stem+'.log')).open('wb') as stderr:
                run=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=58,check=False)
            parser=Page()
            parser.feed((output/(stem+'.html')).read_text(encoding="utf-8",errors="replace"))
            measured=json.loads(parser.body.get('data-browser-audit','{}'))
            assert run.returncode==0,(kind,capture,run.returncode)
            assert measured and not measured.get('error'),(kind,capture,measured)
            assert measured['plots']==14 and measured['sections']==8,measured
            assert not measured['duplicates'] and measured['mathOverflow']==0,measured
            assert measured['documentWidth']<=measured['viewportWidth']+1,measured
            assert measured['allRows']==(64 if kind=='steady' else 12),measured
            assert measured['fineRows']==(16 if kind=='steady' else 3),measured
            assert parser.math_count>5 and not parser.math_errors
            results.append(dict(kind=kind,math_nodes=parser.math_count,**measured))
            print(f"{kind} {capture}: selectors, tables, pages, 14 plots, math and layout passed",flush=True)
    save(output/'audit.json',dict(passed=True,records=results))


if __name__=='__main__':
    main()
