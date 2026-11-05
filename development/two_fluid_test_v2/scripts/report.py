"""Build the standalone ablation report from saved data only."""

import argparse
import base64
import gzip
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np

from .common import ROOT, STUDY, CASES, GRIDS, RECONSTRUCTIONS, VARIANTS, read, save, sha, group_name

LABELS = dict(baseline="Native baseline", inverse_old_sonic="Inverse-free / old sonic",
              sonic_only="New sonic only", both="Inverse-free + new sonic")
FIELDS = dict(p="Pressure [Pa]", Ma_mix="Mixture Mach [–]", alpha_g="Gas volume fraction [–]",
              u_g="Gas velocity [m/s]", u_l="Liquid velocity [m/s]", T_g="Gas temperature [K]",
              T_l="Liquid temperature [K]", mdot="Total mass flow [kg/s]")


def number(value):
    return "—" if value is None else f"{value:.4g}"


def maximum(values):
    return max(values) if values and all(v is not None for v in values) else None


def load(directory, preview=False):
    metadata = read(directory / "metadata.json")
    records, raw_sources = [], {}
    for regime, rec, cells in metadata["groups"]:
        folder = directory / "groups" / group_name(regime, rec, cells)
        for variant in VARIANTS:
            path = folder / (variant + ".json")
            if not path.exists():
                if preview:
                    path = folder / (variant + "__compile.json")
                if not path.exists():
                    continue
            raw = read(path)
            assert raw["source_unchanged"] and raw["source_before"] == metadata["source"]
            relative = path.relative_to(directory).as_posix()
            raw_sources[relative] = sha(path)
            record = {k: raw.get(k) for k in ("variant", "regime", "reconstruction", "cells", "phase",
                "success", "root_pass", "reason", "compilation_free", "warm_s", "elapsed_s", "compilation",
                "audit", "branch", "profile", "errors", "work", "scheme", "steady", "effective",
                "activity_before", "activity_after", "initial_state_sha256", "state_sha256", "exception")}
            record["fields"] = {k: raw["fields"][k] for k in ("x", *FIELDS)}
            record["history"] = [{k:r[k] for k in ("step", "phase", "accepted", "equation_norm", "requested_cfl",
                "dt", "inner_iterations", "rejected_newton_trials") if k in r} for r in raw["history"]]
            record.update(file=relative, raw_sha256=raw_sources[relative], initial_norm=raw["initial_norm"])
            excluded=folder/(variant+"__compile.json")
            if excluded.exists():
                e=read(excluded)
                record["excluded"]={k:e.get(k) for k in ("success","root_pass","reason","work","state_sha256","elapsed_s","compilation")}
                record["excluded"]["raw_sha256"]=sha(excluded)
            record["discarded_cache_event_attempts"]=len(raw.get("discarded_cache_event_attempts",[]))
            records.append(record)
    references = {}
    for regime in CASES:
        path = ROOT / "cases/NICFD2026/two_component" / regime / "reference.csv"
        values = np.genfromtxt(path, delimiter=",", names=True)
        references[regime] = {k:values[k].tolist() for k in values.dtype.names if k == "x" or k in FIELDS}
    checks = {}
    for name in ("kernel_audit.json", "saved_audit.json", "fluid_table_audit.json", "excluded_warm_audit.json"):
        path = STUDY / "outputs" / name
        if path.exists():
            checks[name] = read(path)
    campaign={name:read(directory/name) for name in ('execution.json','protocol_amendment_before_warm.json',
        'workers_compile.json','workers_warm.json','completion_all.json') if (directory/name).exists()}
    return dict(metadata=metadata, campaign=campaign, records=records, sources=raw_sources, references=references,
                checks=checks, preview=preview, fields=FIELDS, labels=LABELS)


def findings(data):
    records = data["records"]
    counts = {v:dict(attempts=0,roots=0,profiles=0,warm=0) for v in VARIANTS}
    for r in records:
        c=counts[r["variant"]];c["attempts"]+=1;c["roots"]+=bool(r["root_pass"])
        c["profiles"]+=r["profile"]["status"]=="matches";c["warm"]+=bool(r["compilation_free"])
    pairs=[]
    for regime,rec,n in data["metadata"]["groups"]:
        selected={r["variant"]:r for r in records if (r["regime"],r["reconstruction"],r["cells"])==(regime,rec,n)}
        if len(selected)==4:
            pairs.append(dict(regime=regime,reconstruction=rec,cells=n,
                roots={v:r["root_pass"] for v,r in selected.items()},
                reasons={v:r["reason"] for v,r in selected.items()}))
    failed=[p for p in pairs if not p["roots"]["baseline"]]
    rescued={v:sum(p["roots"][v] for p in failed) for v in VARIANTS[1:]}
    harmed={v:sum(p["roots"]["baseline"] and not p["roots"][v] for p in pairs) for v in VARIANTS[1:]}
    only_both=sum(not p["roots"]["inverse_old_sonic"] and not p["roots"]["sonic_only"] and p["roots"]["both"] for p in failed)
    intro=(f"The native baseline converged in {counts['baseline']['roots']}/{counts['baseline']['attempts']} configurations. "
        f"Keeping the old sonic law, the inverse-free package converged in {counts['inverse_old_sonic']['roots']}/{counts['inverse_old_sonic']['attempts']}; "
        f"sonic-only converged in {counts['sonic_only']['roots']}/{counts['sonic_only']['attempts']}; "
        f"the combined method converged in {counts['both']['roots']}/{counts['both']['attempts']}.")
    causal=(f"Among {len(failed)} baseline attempts without convergence within the declared limits and with all four arms available, inverse-free/old-sonic recovered {rescued['inverse_old_sonic']}, "
        f"sonic-only recovered {rescued['sonic_only']}, and both recovered {rescued['both']}. "
        f"In {only_both} of those configurations only the combined arm recovered. "
        f"Regressions from a converged baseline were {harmed['inverse_old_sonic']}, {harmed['sonic_only']} and {harmed['both']}, respectively.")
    if failed and rescued['inverse_old_sonic']==len(failed):
        causal+=" The inverse-free package recovered every unconverged baseline configuration in this tested set while retaining the old sonic law."
    elif failed and rescued['both']>rescued['inverse_old_sonic']:
        causal+=" The continuous sonic policy supplied additional recoveries within the inverse-free formulation; removing the inverse alone was not sufficient for every tested path."
    subsonic=[p for p in failed if p['regime']=='subsonic']
    if subsonic:
        causal+=f" Of these baseline failures, {len(subsonic)} occurred in the subsonic case; inverse-free/old-sonic recovered {sum(p['roots']['inverse_old_sonic'] for p in subsonic)}. The difficulty is therefore not confined to transonic expansion."
    causal+=" These are conditional implementation-package effects, not proof of universal convergence or of the unique cause of every rejected step."
    budget_stops=sum(r['reason']=='wall_budget' for r in records)
    if budget_stops:
        causal+=f" {budget_stops} warm attempts reached the wall budget; these are time-limited unresolved runs, not demonstrated stagnation."
    accuracy="Published-profile screening passed for "+"; ".join(
        f"{LABELS[v]}: {counts[v]['profiles']}/{counts[v]['roots']} converged roots" for v in VARIANTS)+". "
    accuracy+="A recovered root can still be too diffusive or displaced on a coarse mesh. The field comparisons and unchanged profile limits test that separately."
    example=[r for r in records if r['regime']=='supersonic_shock' and r['reconstruction']=='muscl_van_albada' and r['cells']==400]
    timing_example=''
    if len(example)==4 and all(r['compilation_free'] and r['root_pass'] for r in example):
        timing_example='For the 400-cell linear shock case, all four variants converged. Their single warm observations were '+', '.join(
            f"{LABELS[r['variant']]}: {number(r['warm_s'])} s ({r['work']['accepted_steps']} accepted steps)" for r in example)+'. The full matrix below also includes the cases with similar runtime and every failed attempt.'
    combined=[r for r in records if r['variant']=='both']
    if len(combined)==len(data['metadata']['groups']):
        accuracy+=" For the combined method, "+" and ".join(
            f"{sum(r['profile']['status']=='matches' for r in combined if r['reconstruction']==rec)}/{sum(r['reconstruction']==rec for r in combined)} {label} configurations pass"
            for rec,label in zip(RECONSTRUCTIONS,('constant','linear')))+"."
        for r in combined:
            if r['regime']=='supersonic_shock' and r['reconstruction']=='constant' and r['cells']==max(GRIDS) and r['root_pass'] and not r['profile']['checks']['gas_temperature']:
                accuracy+=f" Even at {r['cells']} cells, its constant shock gas-temperature error is {100*r['errors']['T_g']['normalized_rmse']:.3g}% against the 1% limit."
    return dict(counts=counts,pairs=pairs,baseline_failures=len(failed),rescued=rescued,harmed=harmed,only_both=only_both,intro=intro,causal=causal,accuracy=accuracy,timing_example=timing_example)


def figures(data, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    colors=dict(zip(VARIANTS,plt.colormaps['plasma'](np.linspace(.2,.8,4))))
    artifacts=[]
    for regime in CASES:
        for rec in RECONSTRUCTIONS:
            for field in ("p","T_g","residual"):
                fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
                for ax,n in zip(axes.flat,GRIDS):
                    rows=[r for r in data['records'] if (r['regime'],r['reconstruction'],r['cells'])==(regime,rec,n)]
                    for r in rows:
                        label=LABELS[r['variant']]+(" · unconverged" if not r['root_pass'] else "")
                        if field=="residual":
                            points=[(i,maximum(h.get('equation_norm'))) for i,h in enumerate(r['history']) if h.get('equation_norm')]
                            x=[p[0] for p in points];y=[max(p[1],1e-300) if p[1] is not None else np.nan for p in points]
                            label+=f" · {r['work']['accepted_steps']}/{r['work']['rejected_steps']} accepted/rejected"
                            ax.semilogy(x,y,color=colors[r['variant']],label=label)
                        else:
                            ax.plot(r['fields']['x'],r['fields'][field],color=colors[r['variant']],
                                alpha=1. if r['root_pass'] else .35,label=label)
                    if field!="residual":
                        ref=data['references'][regime]
                        ax.plot(ref['x'],ref[field],color='.5',linestyle='--',label='Published reference')
                    else:
                        ax.axhline(1e-9,color='.5',linestyle=':',linewidth=1)
                    ax.set(title=f"{n} physical cells",xlabel='Outer observations' if field=='residual' else 'Axial position [m]',
                           ylabel='Maximum equation RMS [–]' if field=='residual' else FIELDS[field])
                    ax.grid(alpha=.2);ax.legend(fontsize=6)
                fig.suptitle(regime.replace('supersonic_','')+' · '+rec+' · '+field)
                stem=regime+'__'+rec+'__'+field
                for ext in ('png','pdf'):
                    p=directory/(stem+'.'+ext);fig.savefig(p,dpi=170)
                    artifacts.append(dict(file=p.relative_to(STUDY).as_posix(),sha256=sha(p)))
                plt.close(fig)
    return artifacts


def build(directory, preview=False, plots=True):
    data=load(directory,preview)
    assert data['records'],"No saved records yet"
    if not preview:
        assert len(data['records'])==len(data['metadata']['groups'])*4,"Incomplete campaign"
        assert all(r['compilation_free'] and r['warm_s'] is not None for r in data['records']),"Warm samples required"
        assert data['checks'].get('saved_audit.json',{}).get('passed'),"Saved-result audit required"
        audited={r['file'].replace('\\','/'):r['sha256'] for r in data['checks']['saved_audit.json']['records']}
        assert audited==data['sources'],"Saved-result audit belongs to different records"
        kernel=data['checks'].get('kernel_audit.json',{})
        assert kernel.get('passed') and kernel['source']==data['metadata']['source'],"Matching kernel audit required"
    result=findings(data);data['findings']=result
    configuration=data['metadata']['configuration']
    budgets=(f"Callback-checked wall budgets are {configuration['warm_budget']:g} s for a warm solve and "
             f"{configuration['compile_budget']:g} s for an excluded compile/load solve. Budget stops are reported separately from stagnation and iteration limits.")
    figure_data=figures(data,STUDY/'outputs/report/plots') if plots else []
    rows=[]
    for r in data['records']:
        a=r['audit'];w=r['work']
        values=[r['regime'].replace('supersonic_',''),('Constant' if r['reconstruction']=='constant' else 'Linear (Van Albada)'),str(r['cells']),LABELS[r['variant']],
            'Pass' if r['root_pass'] else r['reason'],r['profile']['status'],number(maximum(a.get('equation_norm'))),
            number(a.get('global_max')),f"{w['accepted_steps']} / {w['rejected_steps']}",
            number(r['warm_s']) if r['compilation_free'] else '—',
            'time to solution' if r['root_pass'] else 'time to termination']
        rows.append(values)
    headers=['Case','Reconstruction','Cells','Variant','Root / branch','Profile','Max RMS','Global','Accepted / rejected','Warm [s]','Timing meaning']
    table='<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'
    table+=''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    intro_cause=("The earlier failures involved very large inverse actions of the physical-flux Jacobian K in admissible interior states, Newton directions with an extremely narrow descent region, and eventual CFL collapse. The native face flux and nonconservative correction both use that inverse. A separate hard sonic-activation switch produced a finite flux jump. This four-arm study tests replacing those inverse-based operations with primitive-state-jump viscosity and direct nonconservative path integration, smoothing the sonic policy alone, or combining both packages. Physical inputs, boundary equations and the implicit solver remain fixed.")
    markdown=['# Water–nitrogen: inverse/sonic ablation','',intro_cause,'',result['intro'],'',result['causal'],'',result['accuracy'],'',result['timing_example'],'',
        'Root convergence, physical branch, profile accuracy and runtime are separate. The reported warm times are one compilation-free observation per configuration. Failed runs show time until termination, not time to solution.','',
        budgets,'',
        '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
    markdown+=['| '+' | '.join(row)+' |' for row in rows]
    markdown+=['','All runs use independent prepared published-profile seeds, implicit backward Euler, colored AD/banded Newton and unchanged $10^{-9}$ absolute targets. No sequencing or multigrid is used.','',
        'The sonic policy maps to signed weights in the native operator and to viscosity magnitudes in the inverse-free operator. Therefore these are conditional package effects; the finite-jump mapping, NC path and individual regularization changes are not separately identified by four arms.','',
        'The new NC path can change nonconservative shock selection. Equal-slip viscosity is approximate, and stationary void contacts are diffused. Cold-start and perturbed-seed robustness are outside this campaign.','']
    for item in figure_data:
        if item['file'].endswith('.png'):
            markdown+=['!['+Path(item['file']).stem+']('+item['file']+')','']
    (STUDY/'RESULTS.md').write_text('\n'.join(markdown),encoding='utf-8')
    serialized=json.dumps(data,separators=(',',':'),allow_nan=False)
    payload=base64.b64encode(gzip.compress(serialized.encode(),mtime=0)).decode()
    vendor=ROOT/'development/flux_methods/shared/assets/vendor'
    plotly=gzip.decompress((vendor/'plotly-2.35.2.min.js.gz').read_bytes()).decode()
    mathjax=gzip.decompress((vendor/'mathjax-3.2.2-tex-svg.js.gz').read_bytes()).decode()
    license_text='\n\n'.join((vendor/name).read_text(encoding='utf-8') for name in ('PLOTLY_LICENSE.txt','MATHJAX_LICENSE.txt'))
    license_text+='\n\nNICFD2026 reference data, CC BY 4.0\n'+(ROOT/'cases/NICFD2026/LICENSE_UPSTREAM.txt').read_text(encoding='utf-8')
    html=TEMPLATE.replace('__INTRO__',escape(intro_cause)).replace('__RESULT__',escape(result['intro'])).replace('__CAUSAL__',escape(result['causal']))
    html=html.replace('__ACCURACY__',escape(result['accuracy']))
    html=html.replace('__EXAMPLE__',escape(result['timing_example']))
    html=html.replace('__BUDGETS__',escape(budgets))
    html=html.replace('__TABLE__',table).replace('__PAYLOAD__',payload).replace('__PLOTLY__',plotly).replace('__MATHJAX__',mathjax).replace('__LICENSE__',escape(license_text))
    html=html.replace('__PREVIEW__','<p class="alert">PRELIMINARY: excluded compilation records may be shown; warm timings are not yet available for every configuration.</p>' if preview else '')
    path=STUDY/'REPORT.html';path.write_text(html,encoding='utf-8')
    save(STUDY/'outputs/report/report_audit.json',dict(records=len(rows),expected=len(data['metadata']['groups'])*4,
        preview=preview,report_sha256=sha(path),report_bytes=path.stat().st_size,
        generator_sha256=sha(Path(__file__)),data_sha256=sha256(serialized.encode()).hexdigest(),
        raw_sources=data['sources'],figures=figure_data,findings=result))
    print('REPORT',len(rows),'records',path.stat().st_size,'bytes',sha(path),flush=True)


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Water–nitrogen ablation v2</title>
<style>:root{color-scheme:light;--bg:#f4f5f8;--fg:#20283c;--card:#fff;--line:#dce0e9}body.dark{color-scheme:dark;--bg:#161b29;--fg:#e4e8f2;--card:#222b3e;--line:#43506b}body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui;max-width:1450px;margin:auto;padding:28px}section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;margin:22px 0}h1{font-size:35px;line-height:1.2}button,select{font:inherit;padding:7px;margin:4px;border:1px solid var(--line);border-radius:6px;color:inherit;background:var(--card)}nav a{color:inherit;margin-right:20px}.scroll{overflow:auto;max-height:650px}thead{position:sticky;top:0;background:var(--card)}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid var(--line)}.plots{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:15px}.plot{height:510px;min-width:0}pre{font-size:12px;max-height:480px;overflow:auto}.muted{opacity:.8}.alert{color:#b64242;font-weight:700}@media(max-width:900px){body{padding:12px}.plots{display:block}}</style>
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'local'},options:{enableMenu:false},loader:{load:[]}};</script><script>__MATHJAX__</script></head><body>
<h1>Water–nitrogen nozzle: what removes the convergence failures?</h1>__PREVIEW__
<p>__INTRO__</p><p><strong>__RESULT__</strong></p><p>__CAUSAL__</p><p>__ACCURACY__</p>
<p>Three regimes · 50/100/200/400 physical cells · constant and linear reconstruction · four numerical variants · independent single-grid implicit solves.</p>
<p><strong>One warm timing per configuration.</strong> Setup, compilation/cache loading, final independent audits and output are excluded. Failed configurations report time until termination. These observations do not establish statistical speedups or equal-root performance comparisons.</p>
<p>__EXAMPLE__</p>
<nav><a href="#results">Results</a><a href="#explorer">Fields & convergence</a><a href="#theory">Formulations</a><a href="#protocol">Protocol & evidence</a><button id="theme">Toggle theme</button><button id="download">Download report evidence</button></nav>
<section id="results"><h2>All warm runs, including failures</h2><p>Root passes require the original selected unforced spatial, boundary, EOS, global and invariant targets at $10^{-9}$, together with the intended physical branch. Profile screening is independent.</p>__TABLE__</section>
<section id="explorer"><h2>Actual fields, residuals and warm time</h2><label>Regime <select id="regime"></select></label><label>Reconstruction <select id="reconstruction"></select></label><label>Physical cells <select id="cells"></select></label><label>Field <select id="field"></select></label><p id="empty" class="muted"></p><div class="plots"><div id="fields" class="plot"></div><div id="convergence" class="plot"></div></div><div id="timing" class="plot"></div><h3>Independent gates and work counts</h3><pre id="checks"></pre></section>
<section id="theory"><h2>What the four arms change</h2>
<p>The baseline is the actual package model and solver, not a copied implementation. The other arms reuse the same storage, physical flux, EOS, closures, virtual mass, physical boundaries and stabilization. Let $U=(p,u_g,u_\ell,\alpha_g,h_g,h_\ell)$ and $\alpha_\ell=1-\alpha_g$.</p>
<p>For full-slip states, the native flux is $F^*=\tfrac12(F_L+F_R)-\tfrac12K(\sum_k z_ks_kP_k)K^{-1}\Delta F$. The inverse-free flux is $F^*=\tfrac12(F_L+F_R)-\tfrac12J_{\rm policy}(\sum_kz_km_kP_k)\Delta U$.</p>
<table><tr><th>Variant</th><th>Inverse-based face/NC operations</th><th>Sonic weights</th></tr><tr><td>Native baseline</td><td>Retained, unchanged src/</td><td>Native hard expansion-crossing policy</td></tr><tr><td>Inverse-free / old sonic</td><td>Removed; primitive jump and $X\,dU$ path</td><td>$m_k=\lambda_ks_k^{\rm old}$</td></tr><tr><td>New sonic only</td><td>Retained</td><td>$s_k=\lambda_k/H_\delta(\lambda_k)$</td></tr><tr><td>Both</td><td>Removed</td><td>$m_k=H_\delta(\lambda_k)$</td></tr></table>
<p>The native scalar law switches on $\lambda_L&lt;0&lt;\lambda_R$. It uses width $\lambda_R-\lambda_L$ at those expansion crossings and background floor $10^{-6}a_{\rm mix}$ otherwise. The new continuous scalar magnitude uses $\delta=0.02a_{\rm mix}$:</p>
$$H_\delta(\lambda)=\begin{cases}|\lambda|,&|\lambda|\ge\delta,\\(\lambda^2+\delta^2)/(2\delta),&|\lambda|&lt;\delta.\end{cases}$$
<p>The native construction consumes signed weights and the inverse-free construction consumes magnitudes. Consequently these are conditional package effects: the scalar mapping and matrix contraction are not identical across the two rows. At zero speed, $\lambda s^{\rm old}=0$ while $H_\delta(0)=\delta/2$. The experiment does not separately identify the primitive jump, bridge, NC path and individual sonic-policy ingredients.</p>
<p>The old NC formula is $XK^{-1}\Delta F$. The new formula integrates $X(U)\,dU$ with two-point Gauss quadrature along a straight primitive path, splitting interface contributions between neighbors and including the within-cell reconstructed path. Phase mass NC rows vanish; momentum NC rows cancel. A common numerical flux telescopes exactly, while final finite-tolerance face-flow variation is not necessarily machine zero.</p>
<p>All arms retain native low-Mach floors 1 internally and 0 at the endpoints, with actual acoustic multiplier $z=\operatorname{clip}(|u_{\rm mix}|/a_{\rm mix},\mathrm{floor},1)$. Zero floor is not zero dissipation. The same 5/20 m/s slip blend is retained.</p>
<p>The grouped small-slip inverse-free viscosity is bounded but approximates the matrix absolute action; material Jordan terms can be omitted. The combined magnitude-based arm diffuses stationary void contacts, whereas the old-law magnitude vanishes at zero speed. The earlier manufactured test measured about first order for constant and 1.6 order for linear reconstruction. Nonconservative paths can change shock selection. With independently interpolated EOS tables the literal native NC smooth limit is $XK_{\rm policy}^{-1}K_{\rm AD}U_x$, whereas the prototype targets the intended $XU_x$ balance. It is not an algebraically equivalent conditioning-only transformation.</p></section>
<section id="protocol"><h2>Protocol and reproducibility</h2><p>Every configuration independently resets the same published profile with physical boundary preparation; no numerical solution is transferred between meshes or methods. These tests do not establish arbitrary cold-start robustness.</p><p>Implicit backward Euler, colored AD/banded Newton, 20 Newton corrections, 20 backtracks, CFL 2 with growth 1.2 and cap $10^6$, 2000 accepted steps, 64 unchanged-state stagnation steps. All fixed-scale absolute gates remain $10^{-9}$; published-profile limits remain 2% pressure, 5% phase velocities, 1% phase temperatures, 0.02 gas-fraction RMS and 2% mean mass flow.</p><p>__BUDGETS__</p><p>Excluded compilation runs may overlap. Warm samples run serially, after a discarded loading preflight. Each timed solve includes native monitoring and rejected work; device work is synchronized. A warm number requires zero observed JAX backend compile/load events. Any contaminated attempt is retained and one cache-ready retry is allowed. Python activity snapshots detect some competing workloads, not all desktop noise.</p><p>Full states, excluded histories, failures and source snapshots remain in the campaign directory. This single file embeds the compact fields, histories, numerical definitions and source hashes used for its plots and offered download. Reference provenance: <a href="https://github.com/alber-berg/twophase-nozzle-nicfd2026/tree/2dc04ad9d474afb727857139df89537fcadb5769/data">alber-berg/twophase-nozzle-nicfd2026</a>, pinned revision 2dc04ad9d474afb727857139df89537fcadb5769, CC BY 4.0.</p><pre id="metadata"></pre><details><summary>Third-party licenses</summary><pre>__LICENSE__</pre></details></section>
<script>__PLOTLY__</script><script type="application/octet-stream" id="payload">__PAYLOAD__</script><script>
const $=id=>document.getElementById(id),order=['baseline','inverse_old_sonic','sonic_only','both'],colors=['#6a00a8','#b12a90','#e16462','#fca636'];let DATA;
function layout(title,x,y){return {title:{text:title,font:{size:16},x:.5,xanchor:'center',xref:'container',y:.99,yanchor:'top',yref:'container'},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:document.body.classList.contains('dark')?'#e4e8f2':'#20283c'},margin:{t:125,r:22,b:75,l:80},xaxis:{title:x},yaxis:{title:y},legend:{orientation:'h',y:1.02,yanchor:'bottom',font:{size:10}},uirevision:title}}
function draw(){let rows=DATA.records.filter(r=>r.regime===$('regime').value&&r.reconstruction===$('reconstruction').value&&r.cells===Number($('cells').value)),field=$('field').value,fields=[],residuals=[],checks={};$('empty').textContent=rows.length?'':'No saved run for this selection yet.';
 for(let r of rows){let i=order.indexOf(r.variant),name=DATA.labels[r.variant],w=r.work;fields.push({x:r.fields.x,y:r.fields[field],name:name+(r.root_pass?'':' · unconverged'),line:{color:colors[i]},visible:r.root_pass?true:'legendonly'});let obs=r.history.map((h,i)=>({...h,observation:i})).filter(h=>h.equation_norm);residuals.push({x:obs.map(h=>h.observation),y:obs.map(h=>h.equation_norm.some(v=>v===null)?null:Math.max(1e-300,...h.equation_norm)),name:name+' · '+r.cells+' cells · '+w.accepted_steps+' accepted / '+w.rejected_steps+' rejected',line:{color:colors[i]},mode:'lines'});checks[name]={root:r.root_pass,reason:r.reason,original_checks:r.audit,branch:r.branch,profile:r.profile,errors:r.errors,work:r.work,warm_s:r.warm_s,compilation_free:r.compilation_free,compilation:r.compilation,source_sha256:r.raw_sha256,initial_sha256:r.initial_state_sha256,activity_before:r.activity_before,activity_after:r.activity_after,excluded:r.excluded,discarded_cache_event_attempts:r.discarded_cache_event_attempts};}
 let ref=DATA.references[$('regime').value];if(ref?.[field])fields.push({x:ref.x,y:ref[field],name:'Published reference',line:{color:'#8992a5',dash:'dash'}});const options={responsive:true,displaylogo:false};Plotly.react('fields',fields,layout(DATA.fields[field],'Axial position [m]',DATA.fields[field]),options);let l=layout('Unforced residual: actual trajectories','Outer observations','Maximum equation RMS [–]');l.yaxis.type='log';l.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:1e-9,y1:1e-9,line:{color:'#8992a5',dash:'dot'}}];Plotly.react('convergence',residuals,l,options);Plotly.react('timing',[{x:rows.map(r=>DATA.labels[r.variant]),y:rows.map(r=>r.warm_s),type:'bar',marker:{color:rows.map(r=>colors[order.indexOf(r.variant)]),opacity:rows.map(r=>r.root_pass?1:.4)},text:rows.map(r=>r.root_pass?'converged':'failed: '+r.reason),textposition:'auto'}],layout('Single warm observation; failed bars are time to termination','Variant','Warm elapsed [s]'),options);$('checks').textContent=JSON.stringify(checks,null,2);}
async function init(){let bytes=Uint8Array.from(atob($('payload').textContent.trim()),c=>c.charCodeAt(0));DATA=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());window.REPORT_DATA=DATA;for(let [id,key]of [['regime','regime'],['reconstruction','reconstruction'],['cells','cells']])$(id).innerHTML=[...new Set(DATA.records.map(r=>r[key]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b))).map(v=>'<option value="'+v+'">'+({subsonic:'Subsonic',supersonic_shock:'Shock',supersonic_adapted:'Adapted expansion',constant:'Constant',muscl_van_albada:'Linear (Van Albada)'}[v]||v)+'</option>').join('');$('field').innerHTML=Object.entries(DATA.fields).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('');for(let id of ['regime','reconstruction','cells','field'])$(id).onchange=draw;$('theme').onclick=()=>{document.body.classList.toggle('dark');draw()};$('download').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(DATA)],{type:'application/json'}));a.download='two_fluid_v2_ablation_evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};$('metadata').textContent=JSON.stringify({metadata:DATA.metadata,campaign:DATA.campaign,raw_sources:DATA.sources,audits:DATA.checks,findings:DATA.findings},null,2);draw();if(window.MathJax?.startup?.promise)await MathJax.startup.promise;document.body.dataset.ready='true'}init().catch(e=>{document.body.dataset.error=e.stack;console.error(e)});
</script></body></html>'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=STUDY/'outputs/ablation_20261104')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    build(args.directory,args.preview,not args.no_plots)
