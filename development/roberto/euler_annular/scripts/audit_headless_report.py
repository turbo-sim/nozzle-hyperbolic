"""Render an isolated report copy with pre-existing headless Edge; no live UI.

This optional artifact check uses only the local Edge CLI, an empty study-local
profile and the embedded HTML. It does not access the user's browser profile,
install packages, control an existing app, or run numerical simulations.
"""

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from PIL import Image


STUDY = Path(__file__).resolve().parents[1]
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


class RenderedPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.body, self.math_errors, self.math_nodes = {}, [], 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "body":
            self.body = values
        if tag == "mjx-container":
            self.math_nodes += 1
        if values.get("data-mml-node") == "merror":
            self.math_errors.append(values)


def main():
    if not EDGE.is_file():
        raise RuntimeError("This optional audit requires the existing local Edge installation")
    output = STUDY/"outputs/html_report/headless_render"
    output.mkdir(parents=True, exist_ok=True)
    source = STUDY/"REPORT.html"
    target = output/"standalone.html"
    instrumentation = """<style>html{scroll-behavior:auto!important}*,*::before,*::after{animation:none!important;transition:none!important}</style>
<script>window.addEventListener('annular-report-ready',async()=>{
let sectionChecks=null;
const original=location.hash;
if(document.body.dataset.page==='results'){
  const boxes=[...document.querySelectorAll('#report-page-results details.result-section')];
  document.getElementById('results-collapse').click();
  const allClosed=boxes.every(box=>!box.open);
  await window.ANNULAR_REPORT.navigate('#fields',false);
  const navigationReopens=document.getElementById('box-fields').open;
  document.getElementById('results-expand').click();
  sectionChecks={count:boxes.length,allClosed,navigationReopens,allOpen:boxes.every(box=>box.open)};
  await window.ANNULAR_REPORT.navigate(original,false);
  window.scrollTo(0,0);
}
if(original==='#conservation'){
  document.getElementById('box-overview').hidden=true;
  document.getElementById('fields').hidden=true;
  document.querySelector('.section-controls').hidden=true;
  window.scrollTo(0,0);
}
setTimeout(()=>{
const t=document.getElementById(location.hash.slice(1)),visible=e=>e.getBoundingClientRect().height>0;
const ids=['convergence-plot','cfl-plot','equation-residual-plot','conservation-plot'];
const rectangles=ids.map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {left:r.left,top:r.top,bottom:r.bottom,width:r.width};});
const [overall,cfl,equations,balances]=rectangles;
const twoByTwo=document.body.dataset.page!=='results'||(Math.abs(overall.top-cfl.top)<3&&Math.abs(equations.top-balances.top)<3&&equations.top>overall.bottom&&overall.left<cfl.left&&equations.left<balances.left);
document.body.dataset.renderAudit=JSON.stringify({page:document.body.dataset.page,scrollY:scrollY,
anchorTop:t?t.getBoundingClientRect().top:null,documentWidth:document.documentElement.scrollWidth,
viewportWidth:innerWidth,mathOverflow:[...document.querySelectorAll('.math-block')].filter(e=>visible(e)&&e.scrollWidth>e.clientWidth+2).length,
visiblePlots:[...document.querySelectorAll('.js-plotly-plot')].filter(visible).length,sectionChecks,twoByTwo,rectangles});},800);});</script>"""
    target.write_text(source.read_text(encoding="utf-8").replace("</body>",instrumentation+"</body>"),encoding="utf-8")
    records = []
    # CLI screenshot clipping can be unreliable at nonzero scroll offsets.
    # Capture the top of each page; layout measurements still cover every
    # visible theory equation, including the matrix/vector sections below.
    for label, anchor in (("results", "overview"), ("convergence", "conservation"), ("theory", "theory-overview")):
        screenshot, dom = output/f"{label}.png", output/f"{label}.html"
        command = [str(EDGE), "--headless=new", "--no-first-run", "--disable-background-networking",
                   f"--user-data-dir={output/'isolated_profile'}", f"--window-size=1600,{2400 if label=='convergence' else 2000}",
                   "--run-all-compositor-stages-before-draw", "--disable-threaded-scrolling",
                   "--virtual-time-budget=12000", "--dump-dom", f"--screenshot={screenshot}",
                   target.as_uri()+"#"+anchor]
        with dom.open("wb") as stdout, (output/f"{label}.log").open("wb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=55, check=False)
        parser = RenderedPage()
        parser.feed(dom.read_text(encoding="utf-8", errors="replace"))
        measured = json.loads(parser.body.get("data-render-audit","{}"))
        with Image.open(screenshot) as pixels:
            nonblank = any(high-low>10 for low,high in pixels.convert("RGB").getextrema())
        passed = bool(result.returncode==0 and screenshot.is_file() and
                      parser.body.get("data-report-ready")=="true" and
                      parser.body.get("data-theory-ready")=="true" and not parser.math_errors
                      and measured and measured["mathOverflow"]==0
                      and measured["documentWidth"]<=measured["viewportWidth"] and nonblank)
        if measured.get("sectionChecks") is not None:
            passed = passed and measured["twoByTwo"] and measured["sectionChecks"] == dict(
                count=8, allClosed=True, navigationReopens=True, allOpen=True)
        records.append(dict(view=label, anchor=anchor, exit_code=result.returncode,
            report_ready=parser.body.get("data-report-ready"),
            theory_ready=parser.body.get("data-theory-ready"), math_nodes=parser.math_nodes,
            math_errors=parser.math_errors, measured=measured, screenshot_nonblank=nonblank,
            screenshot=str(screenshot.relative_to(STUDY)), passed=passed))
        print(f"{label}: {passed}; report={parser.body.get('data-report-ready')}; math={parser.math_nodes}",flush=True)
    report = dict(passed=all(row["passed"] for row in records), isolated_copy=True,
                  empty_separate_profile=True, report_sha256=sha256(source.read_bytes()).hexdigest(),
                  records=records, qualification="Actual headless artifact rendering, separate from live-app UI automation; screenshots require direct visual inspection.")
    (output/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
