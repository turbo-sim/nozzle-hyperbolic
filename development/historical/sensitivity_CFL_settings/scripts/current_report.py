"""Standalone report for current shared-FVM sensitivity records; no solver imports."""

import gzip
import html
import json
from pathlib import Path

SCHEMA = "shared_fvm_sensitivity_v1"


def render(output):
    """Read a completed or failed campaign and plot every attempted solve."""
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != SCHEMA:
        raise ValueError("Use scripts.report for the pre-migration saved campaign")
    runs = []
    for row in summary["runs"]:
        path = (output / row["file"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Run file must stay within the selected campaign")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    assets = (
        Path(__file__).resolve().parents[2]
        / "sensitivity_backpressure/scripts/report_assets"
    )
    plotly = gzip.decompress(
        (assets / "vendor/plotly-2.35.2.min.js.gz").read_bytes()
    ).decode("utf-8")
    plotly = plotly.replace("</script", "<\\/script")
    payload = json.dumps(dict(summary=summary, runs=runs), allow_nan=False).replace(
        "<", "\\u003c"
    )
    options = "".join(
        f"<option>{html.escape(case)}</option>" for case in summary["cases"]
    )
    rows = "".join(
        f'<tr><td>{html.escape(row["run_id"])}</td><td>{row["success"]}</td>'
        f'<td>{html.escape(row["reason"])}</td><td>{row["accepted_steps"]}</td>'
        f'<td>{row["rejected_steps"]}</td><td>{row["inner_iterations"]}</td>'
        f'<td>{row["spatial_max"]:.3g}</td><td>{row["solve_s"]:.3f}</td></tr>'
        for row in summary["runs"]
    )
    text = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nozzle Newton and CFL sensitivity</title>
<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#202838}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr))}table{border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid #ddd;padding:.5rem;text-align:left}.table{overflow:auto}select{padding:.5rem}</style>
<h1>Nozzle Newton and CFL sensitivity</h1>
<p>Current shared FVM solver, fixed-scale equation, boundary and global checks.
CFL grows after accepted steps and shrinks after rejected trials. Each curve
starts from the same physical cold guess. Newton counts include rejected work.
Warm solve times exclude the separately recorded binding warm-up; these are
single observations. Analytical accuracy and discrete convergence are separate.</p>
<label>Physical case <select id="case">"""
        + options
        + """</select></label>
<label>Comparison <select id="group"><option value="newton">Newton caps at baseline CFL</option>
<option value="1">CFL policies at cap 1</option><option value="100">CFL policies at cap 100</option></select></label>
<div class="plots"><div id="pressure"></div><div id="residual"></div>
<div id="cfl"></div><div id="work"></div></div>
<h2>All attempted solves</h2><div class="table"><table><thead><tr>
<th>Run</th><th>Converged</th><th>Reason</th><th>Accepted steps</th><th>Rejected trials</th>
<th>Newton corrections</th><th>Maximum equation norm</th><th>Warm solve [s]</th></tr></thead><tbody>"""
        + rows
        + """</tbody></table></div>
<script id="report-data" type="application/json">"""
        + payload
        + """</script><script>"""
        + plotly
        + """</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent);
function draw(){
 const id=document.getElementById('case').value,group=document.getElementById('group').value;
 const cases=data.summary.cases[id];
 const selected=data.runs.filter(r=>r.case===id&&(group==='newton'?r.cfl_variant==='baseline':r.cap===Number(group)));
 const traces={pressure:[{x:cases.x_m,y:cases.reference_pressure,name:'Analytical',line:{color:'black',dash:'dash'}}],residual:[],cfl:[],work:[]};
 for(const run of selected){
  const name=`Cap ${run.cap}, ${run.cfl_variant}${run.success?'':' (failed)'}`;
  const accepted=run.history.filter(h=>h.accepted),trials=run.history.filter(h=>h.phase!=='initial');
  traces.pressure.push({x:cases.x_m,y:run.fields.p,name});
  traces.residual.push({x:accepted.map(h=>h.step),y:accepted.map(h=>Math.max(...h.equation_norm)),name});
  traces.cfl.push({x:trials.map((_,i)=>i+1),y:trials.map(h=>h.cfl),name,mode:'lines+markers',marker:{symbol:trials.map(h=>h.accepted?'circle':'x')}});
  let count=0;traces.work.push({x:trials.map((_,i)=>i+1),y:trials.map(h=>count+=h.inner_iterations),name});
 }
 for(const [key,title,x,y] of [['pressure','Final pressure','Position [m]','Pressure [Pa]'],['residual','Spatial residual','Accepted steps','Maximum equation norm'],['cfl','Actual CFL, including rejected trials','Trial index','CFL'],['work','Actual Newton work','Trial index','Cumulative corrections']])
  Plotly.react(key,traces[key],{title,xaxis:{title:x},yaxis:{title:y,...(['residual','cfl'].includes(key)?{type:'log'}:{})},height:430,margin:{l:70,r:20,t:55,b:65},legend:{orientation:'h'}},{responsive:true});
}
document.getElementById('case').onchange=draw;document.getElementById('group').onchange=draw;draw();
</script></html>"""
    )
    destination = output / "REPORT.html"
    destination.write_text(text, encoding="utf-8")
    print("Wrote", destination)
    return destination
