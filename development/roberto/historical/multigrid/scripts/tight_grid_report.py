"""Shared presentation for separately recorded tight-tolerance grid experiments."""

import csv
import html
import json
from pathlib import Path
import re
import statistics


def scope_body(body, prefix):
    ids = set(re.findall(r'\bid="([^"]+)"', body))
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', body)
    body = re.sub(
        r'\bfor="([^"]+)"',
        lambda m: f'for="{prefix}{m[1]}"' if m[1] in ids else m[0],
        body,
    )
    return re.sub(
        r'href="#([^"]+)"',
        lambda m: f'href="#{prefix}{m[1]}"' if m[1] in ids else m[0],
        body,
    )


def study_script(rr, payload, page, prefix=""):
    """Reuse all original plots with independent controls and unique DOM IDs."""
    js = rr.JS.replace(
        "const DATA=PAYLOAD;", "const DATA=" + rr.safe_json(payload) + ";"
    )
    js = js.replace("document.getElementById(", "el(")
    js = re.sub(r"Plotly\.react\('([^']+)'", lambda m: "draw(el('" + m[1] + "')", js)
    js = js.replace(
        "function redraw(){timing();historyPlot();profilePlot();stages();channel();}",
        "function redraw(){pending=[];timing();historyPlot();profilePlot();stages();channel();return Promise.all(pending);}",
    )
    js = js.replace(
        "window.redraw=redraw;redraw();",
        "window.gridStudies=window.gridStudies||{};window.gridStudies["
        + json.dumps(page)
        + "]={redraw,payload:DATA,ready:()=>Promise.all(pending)};if(!document.getElementById("
        + json.dumps(page)
        + ").hidden)redraw();",
    )
    head = (
        "(()=>{const prefix="
        + json.dumps(prefix)
        + ";const el=id=>document.getElementById(prefix+id);let pending=[];function draw(...args){const task=Plotly.react(...args);pending.push(task);return task;}\n"
    )
    return "<script>" + head + js + "\n})();</script>"


def build_page(
    rr, output, original_summary, original_groups, *, intervals=(40, 400), prefix="two-"
):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    hierarchy = [n + 1 for n in intervals]
    label = "Two grids" if len(hierarchy) == 2 else "Four grids"
    hierarchy_text = " / ".join(map(str, hierarchy))
    interval_text = " / ".join(map(str, intervals))
    assert summary["metadata"]["hierarchy_points"] == hierarchy
    assert (
        summary["metadata"]["coarse_tolerance"]
        == summary["metadata"]["fine_tolerance"]
        == 1e-8
    )
    groups = rr.group_runs(summary, output)
    payload = rr.make_payload(summary, groups)
    rr.static_figures(summary, groups, payload, output)
    measured = [r for r in summary["runs"] if not r["excluded"]]
    pairs = []
    for cid, recon, method in groups:
        if (cid, recon) in pairs:
            continue
        pairs.append((cid, recon))
    qualified = []
    for cid, recon in pairs:
        a = rr.measured(groups[cid, recon, "single_grid"])
        b = rr.measured(groups[cid, recon, "grid_sequence"])
        valid = all(r["root_success"] and r["same_root"] for r in a + b)
        ratio = statistics.mean(r["wall_s"] for r in a) / statistics.mean(
            r["wall_s"] for r in b
        )
        qualified.append((cid, recon, ratio, valid))
    finding = (
        f"{sum(r['root_success'] for r in measured)}/{len(measured)} measured solves reach the recorded target roots. "
        f"{sum(x[3] for x in qualified)}/{len(qualified)} case/reconstruction pairs satisfy the unchanged profile-agreement gate; "
        f"{sum(x[3] and x[2]>1 for x in qualified)} qualified pairs are faster and {sum(x[3] and x[2]<1 for x in qualified)} are slower than their fresh single-grid controls."
    )
    failed = [c for c in audit["comparisons"] if not c["pass"]]
    failure = (
        '<div class="callout" style="margin-top:18px"><div><strong>The complete audit has not passed.</strong> '
        + f"Root failures: {sum(not r['root'] for r in audit['results'] if not r['excluded'])}; "
        + f"integrity failures: {sum(not r['integrity_pass'] for r in audit['results'])}. "
        + "Unqualified comparisons: "
        + html.escape(
            ", ".join(
                c["case_id"]
                + " / "
                + c["reconstruction"]
                + " / "
                + c.get("method", "baseline")
                for c in failed
            )
        )
        + ". Failed agreement flags are retained; their time ratios are descriptive, without a validated same-root acceleration claim. No tolerance is relaxed.</div></div>"
    )
    body, csv_rows = rr.results_content(
        summary, groups, audit, finding_override=finding, failure_message=failure
    )
    body = body.replace(
        "Coarse startup, fine-grid convergence",
        "Two grids, equally tight convergence",
        1,
    )
    body = body.replace(
        "Coarse stages use a declared looser startup target and are never reported as final roots.",
        "Both coarse and fine stages use the same original spatial RMS target of 1e-8. These are steady stopping targets; the inner Newton cap remains one correction per pseudo-step.",
    )
    body = body[body.index('<div class="kpis">') :]
    hero = (
        '<div class="hero"><div class="hero-copy"><div class="eyebrow">New experiment / two grids / tight targets</div><h1>Two grids, equally tight convergence</h1><p>'
        + html.escape(finding)
        + '</p><div class="hero-meta"><span>40 to 400 source intervals</span><span>41 and 401 sampled points</span><span>Both spatial RMS targets: 1e-8</span><span>Fresh repeated timing controls</span></div></div></div>'
    )
    intro = '<section id="experiment"><div class="section-head"><div><span class="section-index">NEW / TWO-GRID HYPOTHESIS</span><h2>40 to 400 intervals, tight convergence on both</h2><p>The requested sizes use the existing source nodal convention: 41 and 401 sampled points, including two algebraic ghost slots, hence 39 and 399 interior unknown cells. The fine grid is identical to the original benchmark. Only the startup hierarchy and coarse residual target change; physics, reconstruction, transfer and timestep controls are retained.</p></div></div><div class="callout"><div><strong>Protocol:</strong> solve the 41-point coarse equations to spatial RMS 1e-8, interpolate primitive variables once, reset the retained SER controller, then solve the 401-point equations to the same 1e-8 target. A fresh cold single-grid control is measured in the same campaign. Both constant and MUSCL reconstructions cover shock35, shock70 and the subsonic control, with one excluded complete solve plus ten measured solves per configuration.</div></div></section>'
    valid_faster = [x for x in qualified if x[3] and x[2] > 1]
    valid_slower = [x for x in qualified if x[3] and x[2] < 1]
    outcomes = '<div class="callout"><div><strong>Measured conclusion.</strong> '
    outcomes += (
        "Qualified faster pairs: "
        + html.escape(
            ", ".join(
                f"{cid} / {rr.RECON[recon]} ({ratio:.2f}x)"
                for cid, recon, ratio, valid in valid_faster
            )
            or "none"
        )
        + ". "
    )
    outcomes += (
        "Qualified slower pairs: "
        + html.escape(
            ", ".join(
                f"{cid} / {rr.RECON[recon]} ({1/ratio:.2f}x the single-grid time)"
                for cid, recon, ratio, valid in valid_slower
            )
            or "none"
        )
        + ". "
    )
    more_work = sum(
        statistics.mean(
            r["fine_iterations"]
            for r in rr.measured(groups[cid, recon, "grid_sequence"])
        )
        > statistics.mean(
            r["fine_iterations"]
            for r in rr.measured(original_groups[cid, recon, "grid_sequence"])
        )
        for cid, recon in pairs
        if cid.startswith("shock")
    )
    outcomes += f"{more_work}/4 shocked case/reconstruction combinations use more fine-grid updates than the original looser-coarse sequence. This is a measured trajectory difference; no isolated explanation or universal acceleration claim follows.</div></div>"
    body = hero + intro + outcomes + body
    stage_rows = []
    comparison_rows = []
    for cid, recon in pairs:
        runs = rr.measured(groups[cid, recon, "grid_sequence"])
        for index, points in enumerate(hierarchy):
            stages = [r["stages"][index] for r in runs if len(r["stages"]) > index]
            stage_rows.append(
                [
                    cid,
                    rr.RECON[recon],
                    str(points),
                    str(len(stages)),
                    rr.fmt(
                        statistics.mean(s["wall_s"] + s["transfer_s"] for s in stages),
                        5,
                    ),
                    rr.fmt(statistics.mean(s["iterations"] for s in stages), 5),
                    "1e-8",
                    rr.fmt(max(s["diagnostics"]["residual_rms"] for s in stages), 4),
                    str(
                        all(
                            s["native_success"]
                            and s["diagnostics"]["residual_rms"] <= 1e-8
                            for s in stages
                        )
                    ),
                ]
            )
        old = rr.measured(original_groups[cid, recon, "grid_sequence"])
        fine = rr.measured(groups[cid, recon, "single_grid"])
        comparison_rows.append(
            [
                cid,
                rr.RECON[recon],
                rr.fmt(statistics.mean(r["wall_s"] for r in old), 5),
                rr.fmt(statistics.mean(r["wall_s"] for r in runs), 5),
                rr.fmt(statistics.mean(r["wall_s"] for r in fine), 5),
                rr.fmt(statistics.mean(r["fine_iterations"] for r in old), 5),
                rr.fmt(statistics.mean(r["fine_iterations"] for r in runs), 5),
            ]
        )
    agreement_rows = []
    for cid, recon in pairs:
        records = [
            r
            for r in audit["results"]
            if not r["excluded"]
            and r["case_id"] == cid
            and r["reconstruction"] == recon
            and r["method"] == "grid_sequence"
        ]
        agreement_rows.append(
            [
                cid,
                rr.RECON[recon],
                rr.fmt(max(r["baseline_scaled_state_difference"] for r in records), 5),
                "1e-4",
                rr.fmt(
                    max((r["baseline_shock_difference_cells"] or 0) for r in records), 4
                ),
                str(all(r["matched_solution"] for r in records)),
            ]
        )
    body += (
        '<section id="agreement"><div class="section-head"><div><span class="section-index">UNRELAXED / PROFILE AGREEMENT</span><h2>Same equations and branch, but not always the same discrete profile</h2></div></div>'
        + rr.table(
            [
                "Case",
                "Reconstruction",
                "Largest scaled primitive difference",
                "Agreement threshold",
                "Shock-station difference [fine spacings]",
                "All measured profiles match?",
            ],
            agreement_rows,
        )
        + '<p class="plot-caption">The reference is the first measured cold single-grid state in this campaign. All repeated endpoints, source spatial equations and physical branch flags remain audited. Profile differences beyond 1e-4 disqualify same-root performance claims even when the shock proxy agrees and both residuals are small. The cause or uniqueness of those discrete profiles has not been established.</p></section>'
    )
    body += (
        '<section id="tight-stages"><div class="section-head"><div><span class="section-index">TIGHT TARGETS / EVERY STAGE</span><h2>Cost and endpoint residual at each mesh</h2></div></div>'
        + rr.table(
            [
                "Case",
                "Reconstruction",
                "Points",
                "Measured stages",
                "Mean stage + transfer [s]",
                "Mean accepted updates",
                "Target spatial RMS",
                "Largest endpoint RMS",
                "All targets met?",
            ],
            stage_rows,
        )
        + '<p class="plot-caption">Stage costs include their transfer time; small Python bookkeeping between stages remains in the total timer. Native success and independently audited spatial residuals are separate from analytical accuracy. Both meshes solve their own discrete equations; a tight coarse root is not a fine-grid root.</p></section>'
    )
    same_source = (
        original_summary["metadata"]["source_hashes_after"]
        == summary["metadata"]["source_hashes_before"]
    )
    body += (
        '<section id="historical-comparison"><div class="section-head"><div><span class="section-index">CONTEXT / ORIGINAL FOUR-GRID STUDY</span><h2>Compare the work, qualify cross-campaign timings</h2></div></div>'
        + rr.table(
            [
                "Case",
                "Reconstruction",
                "Original four-grid mean [s]",
                "New two-grid mean [s]",
                "Fresh single-grid mean [s]",
                "Original fine updates",
                "New fine updates",
            ],
            comparison_rows,
        )
        + f"<p>Recorded production-source fingerprints agree across campaigns: <strong>{same_source}</strong>. The original four-grid solve used 51/101/201/401 points and coarse target 1e-5. This study changes both the hierarchy and coarse target, so it is not an ablation of mesh count alone. The original means are historical, not freshly interleaved with this campaign; use the new paired single-grid controls for performance claims.</p><p>Skipping levels removes intermediate work, but the larger mesh jump and tightly converged coarse state can change the fine-grid trajectory. SER also establishes a new state-dependent residual reference on transfer. These are possible explanations for a changed fine-stage cost, not separately verified causes. Original profile-agreement failures remain documented on the Original tab.</p></section>"
    )
    with (output / "timing_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    runner = "run_two_grid" if len(hierarchy)==2 else "run_four_grid"
    folder = "two_grid_tight" if len(hierarchy)==2 else "four_grid_tight"
    body = body.replace("scripts.run_demo\n", f"scripts.{runner}\n")
    body = body.replace("scripts.audit_results\n", f"scripts.audit_results --output development/roberto/historical/multigrid/outputs/{folder}\n")
    body, downloads = rr.embed_downloads(
        body, output, name_prefix="two_grid_" if len(hierarchy) == 2 else "four_grid_"
    )
    # Parameterize the shared two-grid wording without duplicating plotting code.
    body = body.replace("Both meshes solve their own", "Every mesh solves its own")
    if len(hierarchy) == 4:
        body = body.replace(
            "New experiment / two grids / tight targets",
            "New experiment / four grids / tight targets",
        )
        body = body.replace(
            "Two grids, equally tight convergence",
            "Four grids, equally tight convergence",
        )
        body = body.replace(
            "40 to 400 source intervals", "50 / 100 / 200 / 400 source intervals"
        )
        body = body.replace(
            "41 and 401 sampled points", "51 / 101 / 201 / 401 sampled points"
        )
        body = body.replace(
            "Both spatial RMS targets: 1e-8", "All four spatial RMS targets: 1e-8"
        )
        body = body.replace(
            "NEW / TWO-GRID HYPOTHESIS", "NEW / FOUR-GRID TIGHT-TOLERANCE RERUN"
        )
        body = body.replace(
            "40 to 400 intervals, tight convergence on both",
            "50 / 100 / 200 / 400 intervals, one tolerance throughout",
        )
        body = body.replace(
            "39 and 399 interior unknown cells",
            "49 / 99 / 199 / 399 interior unknown cells",
        )
        body = body.replace(
            "Only the startup hierarchy and coarse residual target change",
            "Relative to the original four-grid study, only the coarse residual target changes",
        )
        body = body.replace(
            "solve the 41-point coarse equations to spatial RMS 1e-8, interpolate primitive variables once, reset the retained SER controller, then solve the 401-point equations to the same 1e-8 target",
            "solve the 51, 101, 201 and 401-point equations sequentially to spatial RMS 1e-8, interpolating primitive variables and resetting the retained SER controller at each mesh transition",
        )
        body = body.replace("New two-grid mean [s]", "Tight four-grid mean [s]")
        body = body.replace(
            "This study changes both the hierarchy and coarse target, so it is not an ablation of mesh count alone.",
            "This rerun retains the original four-grid hierarchy and changes the coarse target from 1e-5 to 1e-8. The two-grid study uses a different hierarchy but the same tight targets.",
        )
        body = body.replace(
            "Skipping levels removes intermediate work, but the larger mesh jump and tightly converged coarse state can change the fine-grid trajectory.",
            "Tightening coarse convergence can change the transferred state and fine-grid trajectory without changing the fine-grid equations.",
        )
    return scope_body(body, prefix), payload, downloads, summary, audit


def compare_tight_experiments(rr, four_output, two_output):
    """Compare saved tight-target campaigns without claiming paired timing."""
    four_summary = json.loads(
        (four_output / "summary.json").read_text(encoding="utf-8")
    )
    two_summary = json.loads((two_output / "summary.json").read_text(encoding="utf-8"))
    four = rr.group_runs(four_summary, four_output)
    two = rr.group_runs(two_summary, two_output)
    rows = []
    for setup in four_summary["setups"]:
        cid, recon = setup["case_id"], setup["reconstruction"]
        seq4 = rr.measured(four[cid, recon, "grid_sequence"])
        seq2 = rr.measured(two[cid, recon, "grid_sequence"])
        cold4 = rr.measured(four[cid, recon, "single_grid"])
        cold2 = rr.measured(two[cid, recon, "single_grid"])
        avg = lambda runs, key: statistics.mean(r[key] for r in runs)
        matched4 = all(r["root_success"] and r["same_root"] for r in seq4 + cold4)
        matched2 = all(r["root_success"] and r["same_root"] for r in seq2 + cold2)
        rows.append(
            [
                cid,
                rr.RECON[recon],
                rr.fmt(avg(seq4, "wall_s"), 5),
                rr.fmt(avg(seq2, "wall_s"), 5),
                rr.fmt(avg(cold4, "wall_s") / avg(seq4, "wall_s"), 4),
                rr.fmt(avg(cold2, "wall_s") / avg(seq2, "wall_s"), 4),
                rr.fmt(avg(seq4, "fine_iterations"), 5),
                rr.fmt(avg(seq2, "fine_iterations"), 5),
                rr.fmt(avg(seq4, "equivalent_fine_updates"), 5),
                rr.fmt(avg(seq2, "equivalent_fine_updates"), 5),
                str(matched4) + " / " + str(matched2),
            ]
        )
    return (
        '<section id="four-tight-comparison"><div class="section-head"><div><span class="section-index">SAME TOLERANCE / TWO AND FOUR GRIDS</span><h2>Equal-target comparison with the two-grid campaign</h2></div></div>'
        + rr.table(
            [
                "Case",
                "Reconstruction",
                "Four-grid mean [s]",
                "Two-grid mean [s]",
                "Four-grid single/sequence ratio",
                "Two-grid single/sequence ratio",
                "Four-grid fine updates",
                "Two-grid fine updates",
                "Four-grid equivalent work",
                "Two-grid equivalent work",
                "Profile gates: four / two",
            ],
            rows,
        )
        + (
            "<p>Every stage in both campaigns uses spatial RMS 1e-8 and both finish on the same 401-point equations. Four-grid startup visits 51/101/201/401 points; two-grid startup visits 41/401, so the coarsest mesh size also differs. Each campaign measures its own cold single-grid control with ten repeats and one excluded full solve.</p>"
            "<p>The campaigns were executed sequentially, not interleaved with one another. Raw means therefore remain cross-campaign observations; normalized ratios use the paired control within each campaign. Any ratio with a failed profile gate is descriptive. Fine-update counts and fine-equivalent attempted Newton work expose changes in convergence independently of wall-time variability. These data do not isolate a universal optimum hierarchy.</p></section>"
        )
    )
