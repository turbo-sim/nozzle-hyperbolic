"""Build two standalone HTML reports from saved numerical baselines only."""

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
from html import escape
import json

import numpy as np

from .baseline_common import STUDY, save
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as setup
from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle

ASSETS=STUDY/"scripts/report_assets"
ANNULAR=STUDY.parent/"euler_annular/scripts/report_assets"


def panel(identifier,title,geometry=False):
    return (f'<div class="panel"><div class="panel-head"><h3>{title}</h3></div>'
            f'<div class="plot {"geometry-plot" if geometry else ""}" id="{identifier}"></div></div>')


def section(identifier,title,body):
    return f'<details class="block" id="{identifier}" open><summary>{title}</summary><div class="block-content">{body}</div></details>'


def prepare(report):
    data=deepcopy(report)
    if data["kind"]=="steady":
        data["geometry_area_dense"]=np.asarray(get_symmetric_nozzle(
            np.asarray(data["runs"][0]["dense_reference"]["x"]),**setup.NOZZLE).A).tolist()
    else:
        left=data["physical_setup"]["left"]
        gamma=data["runs"][0]["gamma"]
        a=np.sqrt(gamma*left["p"]/left["rho"])
        scales=np.array([left["rho"],left["rho"]*a,left["p"]/(gamma-1)])
        for run in data["runs"]:
            for snapshot in run["snapshots"]:
                ref=snapshot["reference"]
                rho,u,p=(np.asarray(ref[key]) for key in ("rho","u","p"))
                qref=np.stack([rho,rho*u,p/(gamma-1)+.5*rho*u**2],axis=-1)
                snapshot["conservative_L1"]=(np.mean(np.abs(np.asarray(snapshot["conservative"])-qref),axis=0)/scales).tolist()
    return data


def theory(steady):
    common=r"""
    <h2>Conserved equations and discretization</h2>
    <p>The inviscid, adiabatic quasi-one-dimensional Euler model assumes a fixed
    cross-sectional area and no swirl. This baseline has no wall friction or
    heat flux. The general-EOS source kernels are evaluated with
    <code>jaxprop.FluidPerfectGas(Air)</code>; the analytical solution uses the
    exact same backend values of $\gamma$ and $R$, rather than imposing $\gamma=1.4$.</p>
    $$\mathbf q=(\rho,\rho u,\rho E)^T,$$
    $$\mathbf F=(\rho u,\rho u^2+p,\rho u H)^T,$$
    $$\partial_t(A\mathbf q)+\partial_x(A\mathbf F)=(0,p A',0)^T.$$
    <p>Here $E=e+u^2/2$ and $H=E+p/\rho$. The independent analytical perfect-gas
    reference uses $e=p/[(\gamma-1)\rho]$. Face fluxes and integrated pressure-area
    sources form the actual finite-volume residual:</p>
    $$\mathbf R_i=\mathbf\Phi_{i+1/2}-\mathbf\Phi_{i-1/2}-\mathbf\Sigma_i.$$
    <p>Cell-center $\rho u A$ can vary through a smeared shock without indicating
    a loss of discrete conservation. The numerical face mass flux is the
    appropriate steady mass-flow diagnostic. No numerical markers are used in
    the field plots, and all original samples are retained in downloads.</p>
    """
    common=common.rstrip()+"\n"
    if steady:
        return common+r"""
        <h2>Backpressure regimes</h2>
        <p>One symmetric converging-diverging nozzle is used for all eight
        operating points: length 0.2 m, inlet/exit area 300 mm², throat area
        150 mm² at x = 0.1 m. Reservoir conditions are 200 kPa and 300 K.</p>
        $$\frac{A}{A^*}=\frac{1}{M}
        \left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}.$$
        <p>Subsonic flow uses the subsonic area–Mach root on both sides of the
        throat with a virtual critical area below the geometric throat area.
        The exactly sonic limiting branch has M = 1 at the throat and a subsonic
        exit. The two internal shocks are selected at 35% and 70% of the
        divergent length using normal-shock jump relations. The resulting back
        pressure is given to the numerical solver; shock position and analytical
        fields are never used as initial guesses.</p>
        <p>At the shock-at-exit pressure, the reference shown at x = L is the
        upstream supersonic limit. The downstream jump is outside the plotted
        interior. Below that backpressure, overexpanded external compression,
        adapted discharge, and underexpanded external expansion have the same
        ideal internal solution. This 1D internal-domain calculation does not
        resolve an exterior jet or its shock/expansion structure.</p>
        <h2>Current solver and reconstruction</h2>
        <p>Production regularized Städtke characteristic FVS, constant or
        primitive Van Leer MUSCL reconstruction, backward-Euler pseudo-time
        updates and one unchecked full Newton correction per update. Jacobians
        use colored AD and a banded solve. The regularization factor is 10⁻⁶;
        the optional conservative filter is off. Boundary states are eliminated
        using the current production inlet/outlet maps, including the smooth
        Mach-dependent pressure blend. These choices are deliberately recorded
        before harmonizing them with annular.</p>
        <p>The retained backpressure-study controller starts from the ordinary
        simple initialization for every case and grid independently. It uses
        the existing SER/growth and increment-based retry rules, and requires
        both its fixed-scale spatial RMS ≤ 10⁻⁸ and native source SSR &lt; 10⁻⁸.
        It refuses a timestep if all nine trials fail its increment/positivity
        checks; this differs from the production application's last-retry forced
        acceptance. The driver is identified explicitly in the provenance.
        This is not the future annular-style nine-order convergence policy.</p>
        <p>The legacy grids have 51, 101, 201 and 401 nodes, including two
        boundary slots: 49, 99, 199 and 399 physical points. This retains the
        established 201-node baseline and places a node at the throat. Nodes,
        boundary slots and control-volume extents must be preserved explicitly
        when comparing against the future common mesh; they are not annular's
        50/100/200/400-cell meshes. No grid sequencing or exact initialization
        is used in these independent baseline solves.</p>
        <h2>Local residuals versus global defects</h2>
        <p>The lower-left plot takes a volume-weighted RMS of each signed local
        equation after fixed dimensional scaling, then divides by its initial
        value. It measures whether all cells satisfy that equation. Its first
        value is one. These additional observations do not control this run.</p>
        $$r_{ij}=\frac{(L/a_0)R_{ij}}{V_i q_{j,\mathrm{ref}}},\qquad
        N_j=\sqrt{\frac{\sum_i V_i r_{ij}^2}{\sum_i V_i}}.$$
        $$R_{j,\mathrm{relative}}=N_j/N_{j,\mathrm{initial}}.$$
        <p>The lower-right plot sums the signed integrated rows before taking
        the absolute value. It checks inlet/outlet flux differences minus
        integrated geometry sources. Local positive and negative defects can
        cancel, so this is a complementary conservation check, not a substitute
        for the per-equation local residuals.</p>
        $$G_j=\frac{|\sum_i R_{ij}|}{A_t a_0 q_{j,\mathrm{ref}}},\qquad
        \mathbf q_{\mathrm{ref}}=\rho_0(1,a_0,a_0^2)^T.$$
        <p>The study stopping RMS has its own inherited cell-crossing scaling;
        do not compare its number directly to Nⱼ or Gⱼ. The native SSR and all
        raw diagnostics are retained in the full history download.</p>
        <h2>Accuracy and limits</h2>
        <p>Full-domain mean absolute pressure, density and Mach errors are
        normalized by p₀, ρ₀ and one respectively. Each must be below 0.03;
        boundary numerical mass-flux imbalance must be below 10⁻⁴. These retain
        the established example's criteria. Conservative-state L1 errors are
        additional diagnostics, with scales ρ₀(1,a₀,a₀²). No shock region is
        excluded from acceptance.</p>
        <p>A field L1 pass does not guarantee the correct value at a single
        sonic point: throat Mach is always reported separately. Shock smearing,
        a shock adjacent to the boundary, and the sonic-limit branch all deserve
        separate inspection. Convergence, conservation and analytical accuracy
        are independent conclusions.</p>
        """
    return common+r"""
    <h2>One physical problem, multiple times and meshes</h2>
    <p>The standard Sod diaphragm problem is sufficient for this first transient
    baseline: it contains a rarefaction, contact discontinuity and shock. It is
    run independently on 50, 100, 200 and 400 physical cells. Each simulation
    advances once through τ = t aL/L = 0.1, 0.2 and 0.3. These are snapshots of
    one physical problem, not separate initial-value cases.</p>
    <p>The one-metre constant-area domain has a diaphragm at 0.5 m, aligned
    with a face. Left state: ρ = 1 kg/m³, u = 0, p = 100 kPa. Right state:
    ρ = 0.125 kg/m³, u = 0, p = 10 kPa. Transmissive ghost cells impose open
    ends; the exact waves have not reached either end at the saved times.</p>
    <h2>Physical time and current numerical method</h2>
    <p>The existing production HLL flux is used with constant reconstruction,
    forward Euler and CFL 0.45, matching the established Sod example. The small
    development driver uses one global physical timestep, clipped <em>before</em>
    updating all cells to hit each requested snapshot exactly. It does not use
    the application's local pseudo-time driver. This report does not claim to
    verify the steady characteristic FVS method or a new MUSCL/RK transient path.</p>
    $$\Delta t=\min\left(\frac{\mathrm{CFL}\,\Delta x}{\max_i(|u_i|+a_i)},
    t_{\mathrm{output}}-t\right).$$
    $$\mathbf q_i^{n+1}=\mathbf q_i^n
    -\frac{\Delta t}{\Delta x}(\mathbf\Phi_{i+1/2}-\mathbf\Phi_{i-1/2}).$$
    <h2>Conservation is a transient budget</h2>
    <p>A changing solution is expected; steady residuals should not decay to
    zero. Instead, audit the physical clock, exact-solution errors, positivity,
    and time-integrated boundary-flux budgets at every step:</p>
    $$\mathbf B(t)=\sum_i\Delta x\,[\mathbf q_i(t)-\mathbf q_i(0)]
    -\sum_n\Delta t_n(\mathbf\Phi_{\mathrm{left}}^n-\mathbf\Phi_{\mathrm{right}}^n).$$
    <p>Scale these defects by L times (ρL, ρL aL, pL/(γ−1)). Momentum is not
    constant because unequal end pressures exert a net force. Requiring all
    integrated totals to remain constant would be an incorrect test.</p>
    <h2>Accuracy and later extensions</h2>
    <p>The exact Riemann solution is sampled at numerical cell centers. Mean
    absolute density, velocity and pressure errors use scales ρL, aL and pL,
    and each must be below 0.04. The normalized budget target is 10⁻¹⁰.
    Conservative-state L1 errors use the same scales as the integrated budgets
    divided by L. First-order diffusion is expected at the shock and contact;
    the refinement curves are not a claim of second-order accuracy.</p>
    <p>Additional Riemann problems, smooth-wave order tests, near-vacuum and
    strong-shock robustness checks would broaden verification later. They are
    useful additions, but not necessary to retain this controlled pre-refactor
    Sod baseline.</p>
    """


def build(kind):
    destination=STUDY/"outputs"/f"baseline_{kind}"
    raw=json.loads((destination/"summary.json").read_text(encoding="utf-8"))
    audit=json.loads((destination/"audit.json").read_text(encoding="utf-8"))
    data=prepare(raw)
    data["audit"]=audit
    steady=kind=="steady"
    finest=max(r["points"] if steady else r["physical_cells"] for r in data["runs"])
    if steady:
        final=[r for r in data["runs"] if r["points"]==finest]
        kpis=[("Steady residual targets",sum(r["converged"] for r in data["runs"]),len(data["runs"])),
              ("Finest-grid field accuracy",sum(r["accuracy_passed"] for r in final),len(final)),
              ("Independent saved-state audits",audit["passed_runs"],len(data["runs"]))]
        subtitle="Eight backpressures · constant and MUSCL · four independent meshes"
        interpretation=("<p><strong>All 64 runs meet the recorded steady root checks; 15 of 16 finest-grid results pass the field-error targets.</strong> "
            "The finest constant sonic-limit result fails the Mach L1 criterion. Even the MUSCL field-error pass retains throat Mach about 0.890 rather than 1; constant gives about 0.809. "
            "The 201-node MUSCL result remains about 0.846, consistent with the earlier example. The limiting sonic point remains an important migration check.</p>"
            "<p>Refinement reduces the broad smooth-flow errors, while MUSCL generally improves shock resolution and field errors. At a shock exactly at the exit, the numerical pressure boundary can retain a compression near the last cells. "
            "Below that threshold, the three ideal interior profiles coincide because pressure adjustment belongs to the exterior jet. Their numerical fields need not be bitwise identical after independent starts.</p>"
            "<p>The local normalized residuals and global conservation defects are different diagnostics. The former checks every cell through an RMS; the latter sums signed defects, allowing cancellation. "
            "Neither a field-accuracy pass nor the legacy stopping rule implies annular's future nine-order criterion has been met.</p>")
        condition=("Length 0.2 m; inlet/exit area 300 mm²; throat 150 mm². Reservoir p₀ = 200 kPa, T₀ = 300 K. "
                   "Inviscid, adiabatic, fixed geometry; no prescribed shock location in the numerical solve. All cases start independently from the ordinary simple rule.")
    else:
        final=next(r for r in data["runs"] if r["physical_cells"]==finest)
        kpis=[("Snapshot accuracy + budgets",sum(s["passed"] for r in data["runs"] for s in r["snapshots"]),sum(len(r["snapshots"]) for r in data["runs"])),
              ("Finest-grid snapshots",sum(s["passed"] for s in final["snapshots"]),len(final["snapshots"])),
              ("Independent saved-state audits",audit["passed_runs"],len(data["runs"]))]
        subtitle="One Riemann problem · three physical times · four meshes"
        interpretation=("<p><strong>The 100-, 200- and 400-cell runs pass every snapshot; the 50-cell run misses the velocity error target at later times.</strong> "
            "The coarse failures are retained. At τ = 0.3 the 400-cell normalized L1 errors are approximately 0.00830 in density, 0.00888 in velocity and 0.00602 in pressure. "
            "The finest snapshots all satisfy the 0.04 accuracy and 10⁻¹⁰ budget criteria.</p>"
            "<p>Errors decrease under refinement. The first-order HLL method diffuses the contact and shock, so a second-order rate should not be expected. "
            "Conservation budget defects stay near floating-point roundoff; this does not remove discretization error in the profiles.</p>"
            "<p>One standard Sod case is an appropriate first baseline because it exercises three wave families. Multiple times and mesh refinement make it more informative than one final curve. "
            "The physical-time history is intentionally different from the steady report: no steady convergence tolerance is imposed on moving waves.</p>")
        condition=("One-metre, unit-area domain; diaphragm at x = 0.5 m. Left: ρ = 1 kg/m³, p = 100 kPa; right: ρ = 0.125 kg/m³, p = 10 kPa; both initially at rest. "
                   "HLL / constant / forward Euler, CFL 0.45, transmissive ends; all exact waves remain inside the domain.")
    overview="".join(f'<div class="kpi"><div class="kpi-label">{title}</div><div class="kpi-value {"success" if passed==total else "warning"}">{passed}<span> / {total}</span></div></div>' for title,passed,total in kpis)
    results=(f'<div class="hero"><div class="hero-copy"><div class="eyebrow">EulerNozzle / verification before shared-engine migration</div><h1 id="title"></h1><p>{subtitle}</p>'
             '<div class="hero-meta"><span>Current production kernels</span><span>JAXProp Air · float64</span><span>29 October 2026</span></div></div></div>'
             f'<div class="kpis">{overview}</div><div class="callout">Fixed numerical evidence, before the planned coupled-boundary, common-mesh and Newton/convergence migration. '
             'Failed accuracy checks and actual returned fields are retained. Timings are descriptive single-run values, not a speed benchmark.</div>'
             '<div class="controls"><div class="control"><label for="case-select">'+('Operating condition' if steady else 'Physical-time snapshot')+'</label><select id="case-select"></select></div>'
             '<div class="control"><label for="mesh-select">Numerical resolution</label><select id="mesh-select"></select></div>'
             '<button id="download-selected">Download selected data</button></div><div class="selection-note" id="selection-note"></div>'+
             section('conditions','01 · Physical conditions and baseline method',f'<p>{condition}</p><p>Selections update every field, history and refinement plot. Close any section using its heading. Plot toolbar downloads retain vector SVG export.</p>')+
             section('geometry','02 · '+('Nozzle geometry' if steady else 'Initial diaphragm states'),'<div class="grid">'+panel('geometry-profile','Equivalent axisymmetric wall shape' if steady else 'Initial density',True)+panel('geometry-area','Cross-sectional area' if steady else 'Initial pressure',True)+'</div>')+
             section('fields','03 · Numerical and analytical flow fields','<p>Reference: dotted neutral line. Numerical results: colored lines without markers. '+('Boundary slots are included at the plotted endpoints; accuracy uses physical points only. Cell-center mass flow is a field, not the discrete conservation diagnostic.' if steady else 'The exact solution contains a left rarefaction, a contact discontinuity and a right shock.')+'</p><div class="grid" id="plots-fields"></div>')+
             section('history','04 · '+('Convergence and conservation' if steady else 'Physical clock, CFL and conservation'),'<div class="grid" id="plots-history"></div>')+
             section('refinement','05 · Error under mesh refinement','<p>Full-domain mean errors include discontinuities. Every mesh starts independently; no analytical field is used as an initial guess.</p><div class="grid">'+panel('errors','Primitive-field error under refinement')+panel('state-errors','Conservative-state error')+'</div>')+
             section('interpretation','06 · Interpretation and limitations',interpretation)+
             section('evaluation','07 · Overview of all cases','<p>By default the table shows the finest mesh. Full fields and actual histories for every mesh are included in the embedded download.</p><label><input type="checkbox" id="all-grids"> Show all meshes</label><div id="table-content"></div>')+
             section('evidence','08 · Evidence and reproducibility',f'<p>Source fingerprint: {len(data["provenance"]["source_hashes"])} production Python files. Source unchanged during the campaign: <strong>{data["source_unchanged"]}</strong>. '
                     f'Independent audit: <strong>{audit["passed_runs"]}/{len(data["runs"])} runs</strong>. Source equality is checked again during auditing.</p><p>Open the Reproduce page for commands, current numerical settings, source hashes and limitations.</p>'))
    commands=(f'.venv/Scripts/python.exe -u -m development.roberto.historical.euler_nozzle.scripts.run_baselines {kind}\n'
              f'.venv/Scripts/python.exe -m development.roberto.historical.euler_nozzle.scripts.audit_baselines {kind}\n'
              '.venv/Scripts/python.exe -m development.roberto.historical.euler_nozzle.scripts.report_baselines\n'
              '.venv/Scripts/python.exe -m development.roberto.historical.euler_nozzle.scripts.audit_reports')
    reproduce=(f'<h1>Reproduce this baseline</h1><p>Run from the repository root in its existing environment. The report builder reads saved results and does not rerun simulations. '
               'Keep old outputs when changing source; select a new output directory instead of overwriting baseline evidence.</p>'
               f'<pre>{escape(commands)}</pre><h2>Case configuration</h2><pre>{escape((STUDY/"cases"/f"{kind}.yaml").read_text(encoding="utf-8"))}</pre>'
               f'<h2>Numerical settings and driver identity</h2><pre>{escape(json.dumps(data["runs"][0].get("settings",dict(method=data["runs"][0].get("method"),CFL=data["runs"][0].get("CFL"))),indent=2))}</pre>'
               f'<h2>Independent audit</h2><pre>{escape(json.dumps(audit,indent=2))}</pre>'
               f'<h2>Version and source record</h2><pre>{escape(json.dumps(data["provenance"],indent=2))}</pre>')
    css=(ANNULAR/"report.css").read_text(encoding="utf-8")+'\n'+(ASSETS/"baseline.css").read_text(encoding="utf-8")
    js=(ASSETS/"baseline.js").read_text(encoding="utf-8")
    plotly=gzip.decompress((ANNULAR/"vendor/plotly-2.35.2.min.js.gz").read_bytes()).decode("utf-8")
    mathjax=gzip.decompress((ANNULAR/"vendor/mathjax-3.2.2-tex-svg.js.gz").read_bytes()).decode("utf-8")
    payload=json.dumps(data,allow_nan=False,separators=(",",":"),ensure_ascii=True).replace('<','\\u003c')
    licenses='\n\n'.join((ANNULAR/"vendor"/name).read_text(encoding="utf-8") for name in ("PLOTLY_LICENSE.txt","MATHJAX_LICENSE.txt"))
    html=f'''<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; worker-src blob:; object-src 'none'; base-uri 'none'">
<title>EulerNozzle {'steady' if steady else 'Sod'} baseline</title><style>{css}</style></head><body data-ready="false">
<header class="header"><div class="brand">eulernozzle <small>Baseline verification</small></div><nav class="page-switch" aria-label="Report pages"><button data-page-choice="results">01 · Results</button><button data-page-choice="theory">02 · Theory</button><button data-page-choice="reproduce">03 · Reproduce</button></nav><div class="header-actions"><button id="download-data">Full dataset</button><button id="print">Print / PDF</button><div class="theme-switch"><button data-theme-choice="light" aria-pressed="true">Light</button><button data-theme-choice="dark" aria-pressed="false">Dark</button></div></div></header>
<aside class="sidebar"><div class="eyebrow">On this page</div><nav>{''.join(f'<a href="#{id}">{label}</a>' for id,label in [('conditions','01 · Conditions'),('geometry','02 · Geometry / initial states'),('fields','03 · Flow fields'),('history','04 · History and budgets'),('refinement','05 · Refinement'),('interpretation','06 · Interpretation'),('evaluation','07 · Case evaluation'),('evidence','08 · Evidence')])}</nav><div class="sidebar-foot">Offline document<br>Fixed pre-refactor evidence</div></aside>
<main class="baseline-main"><article data-report-page="results">{results}</article><article class="theory-copy" data-report-page="theory" hidden><h1>Theory and numerical implementation</h1>{theory(steady)}</article><article data-report-page="reproduce" hidden>{reproduce}</article>
<footer>EulerNozzle · source unchanged · independent numerical/reference audits</footer><details><summary>Embedded third-party licenses</summary><pre>{escape(licenses)}</pre></details></main>
<script id="baseline-data" type="application/json">{payload}</script><script>{plotly}</script>
<script>window.MathJax={{loader:{{load:[]}},options:{{enableMenu:false}},tex:{{packages:['base','ams','newcommand','noundefined','configmacros'],inlineMath:[['$','$']],displayMath:[['$$','$$']]}},svg:{{fontCache:'local'}},startup:{{typeset:true}}}};</script><script>{mathjax}</script><script>{js}</script></body></html>'''
    output=STUDY/("STEADY_REPORT.html" if steady else "SOD_REPORT.html")
    output.write_text(html,encoding="utf-8")
    save(destination/"report_build.json",dict(report=output.name,sha256=sha256(output.read_bytes()).hexdigest(),
        summary_sha256=sha256((destination/"summary.json").read_bytes()).hexdigest(),audit_sha256=sha256((destination/"audit.json").read_bytes()).hexdigest(),
        presentation_input_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in
            [*sorted(ASSETS.glob('*')),ASSETS.parent/'report_baselines.py'] if p.is_file()}))
    print(f"Wrote {output.name}: {output.stat().st_size/1e6:.2f} MB")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("steady","sod","both"),default="both")
    args=parser.parse_args()
    for kind in (("steady","sod") if args.kind=="both" else (args.kind,)):
        build(kind)
