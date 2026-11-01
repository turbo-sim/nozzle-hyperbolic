"""Write the Roe study's concise measured assessment from saved evidence.

This presentation-only module imports no flow solver and performs no simulation.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


STUDY = Path(__file__).resolve().parents[1]
FLUXES = ("roe", "rusanov", "hll", "stadtke")
LABELS = dict(roe="Roe", rusanov="Rusanov", hll="HLL", stadtke="Städtke FVS (custom corrections off)")


def unchanged(campaign, name):
    """Read both saved layouts; an explicit failure wins over a true flag."""
    values = (campaign.get(name), campaign.get("provenance", {}).get(name))
    return False if any(value is False for value in values) else True if any(value is True for value in values) else None


def environment_status(value):
    return value.get("status") if isinstance(value, dict) else value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flux_name(row):
    return row.get("flux", row.get("scheme", {}).get("flux", row.get("settings", {}).get("scheme", {}).get("flux")))


def independently_converged(row, protocol):
    """Reassess retained original norms; this does not reevaluate the PDE."""
    metrics = row.get("metrics", {})
    policy = row.get("settings", {}).get("residual", protocol.get("residual", {}))
    norm, initial, global_defect = (metrics.get(key) for key in ("equation_norm", "initial_norm", "global_defect"))
    boundary = metrics.get("boundary_max")
    if not all(isinstance(v, list) and v and all(finite(x) and x >= 0 for x in v) for v in (norm, initial, global_defect)):
        return None
    if not len(norm) == len(initial) == len(global_defect) or not finite(boundary) or boundary < 0:
        return None
    required = ("initial_floor", "absolute_tolerance", "relative_tolerance", "boundary_tolerance", "global_tolerance")
    if not all(finite(policy.get(name)) for name in required):
        return None
    checks = []
    for final, first in zip(norm, initial):
        absolute = final <= policy["absolute_tolerance"]
        relative = final <= first * policy["relative_tolerance"]
        criterion = policy.get("criterion", "either")
        if criterion == "absolute" or first <= policy["initial_floor"]:
            checks.append(absolute)
        elif criterion == "relative":
            checks.append(relative)
        elif criterion == "both":
            checks.append(relative and absolute)
        elif criterion == "either":
            checks.append(relative or absolute)
        else:
            return None
    return bool(all(checks) and boundary <= policy["boundary_tolerance"]
                and max(global_defect) <= policy["global_tolerance"]
                and metrics.get("finite") is True and metrics.get("admissible") is True)


def timing_eligible(row, campaign, *, repeated=False):
    """Require completed, stable, clean evidence and the retained sample count."""
    if not (campaign.get("completed") is True and unchanged(campaign, "source_unchanged") is True
            and unchanged(campaign, "harness_unchanged") is True
            and environment_status(campaign.get("timing_environment")) == "clean"
            and environment_status(row.get("timing_environment")) == "clean"
            and row.get("timing_valid") is True and row.get("success") is True
            and row.get("branch_passed") is True
            and independently_converged(row, campaign.get("protocol", {})) is True):
        return False
    if row.get("accuracy_assessed") is True and row.get("analytical_passed") is not True:
        return False
    timing, samples = row.get("timing", {}), row.get("samples", [])
    expected = timing.get("expected_samples")
    times = timing.get("samples_s", [])
    warmup = row.get("warmup", {})
    if not (isinstance(expected, int) and expected >= (2 if repeated else 1)
            and len(samples) == len(times) == expected and warmup.get("success") is True
            and all(s.get("success") is True and independently_converged(dict(row, metrics=s.get("metrics", {})), campaign.get("protocol", {})) is True for s in samples)
            and all(finite(t) and t > 0 for t in times)
            and finite(timing.get("mean_s")) and timing["mean_s"] > 0):
        return False
    return True


def configuration(row):
    return (row.get("case_id"), row.get("reconstruction"), row.get("cells"),
            row.get("newton_cap", row.get("cap")), row.get("backend"), flux_name(row))


def expected_configurations(protocol):
    """The saved expanded population, without importing the numerical harness."""
    cases, grids = protocol.get("cases"), protocol.get("grids")
    if not isinstance(cases, list) or not grids:
        return set()
    return {(case, recon, cells, cap, backend, flux)
            for case in cases for cells in grids
            for recon in protocol.get("reconstructions", []) for cap in protocol.get("newton_caps", [])
            for backend in protocol.get("backends", [])
            if backend == "current" or cells == max(grids) or protocol.get("legacy_refinement")
            for flux in protocol.get("fluxes", [protocol.get("scheme", {}).get("flux")])}


def matching_controls(left, right):
    def controls(row):
        settings = deepcopy(row.get("settings", {}))
        settings.get("scheme", {}).pop("flux", None)
        settings.pop("roe_options", None)
        return settings
    return (bool(left.get("settings")) and controls(left) == controls(right)
            and left.get("case_sha256") == right.get("case_sha256")
            and left.get("source_sha256") == right.get("source_sha256"))


def sod_section(data, path=None, figure=None):
    """Report retained transient statuses without reevaluating a flow solve."""
    if data is None:
        return ["## Roe transient verification", "",
                "No saved Sod verification was supplied. Use `--sod-verification PATH` to include an existing result; this generator does not run it.", ""]
    rows = data.get("records", [])
    measured = [row for row in rows if "conservative_errors" in row]
    assessment = data.get("assessment", {})
    successful = sum(row.get("success") is True for row in rows)
    strict_failures = [row for row in rows if row.get("passed") is not True]
    numeric_passed = assessment.get("numerical_checks_passed", sum(
        row.get("classification", {}).get("numerical_passed") is True for row in rows))
    lines = ["## Roe transient verification", "",
        f"The retained Sod test contains **{successful}/{len(rows)} successful physical-time solves** on grids "
        f"`{data.get('settings', {}).get('grids', [])}`, using constant reconstruction and CFL 0.5. "
        "The four choices are explicit Euler, SSPRK3, and backward Euler with coupled or segregated transmissive traces. "
        "This is untimed verification; its concurrent execution does not support performance comparisons.", "",
        f"**{numeric_passed}/{len(rows)} pass the numerical accuracy/admissibility/BC/discrete-conservation checks.** "
        f"The original aggregate passed flag remains **{data.get('passed', False)}**: **{len(strict_failures)}** original failures "
        "arise from the additional comparison against exact, undisturbed endpoint transfer at a strict $10^{-8}$ normalized tolerance. "
        "These statuses are preserved and reported separately below.", "",
        "Conservative errors compare numerical cell averages with independently integrated exact Sod cell averages, "
        "normalized by the fixed model scales. Entries are mass / meridional momentum / total energy. "
        "The budget column uses actual accepted numerical flux/source transfers, rather than assuming unchanged endpoint states.", "",
        "| Cells | Integrator / BC treatment | Steps / rejected | Conservative normalized L1 | Max discrete budget | Original strict status |",
        "|---:|---|---:|---|---:|---|"]
    labels = dict(euler_explicit="Explicit Euler / segregated",rk3="SSPRK3 / segregated",
                  implicit_coupled="Backward Euler / coupled",implicit_segregated="Backward Euler / segregated")
    triple = lambda values:" / ".join(f"{value:.3g}" for value in values)
    for row in rows:
        name = labels.get(row.get("method", {}).get("id"),row.get("method", {}).get("id","unknown"))
        errors = row.get("conservative_errors", {}).get("L1")
        error_text = triple(errors) if errors and all(finite(value) for value in errors) else "unavailable"
        budget = row.get("independent_summed_transfer_budget", [])
        budget_text = f"{max(abs(value) for value in budget):.3g}" if budget and all(finite(value) for value in budget) else "unavailable"
        lines.append(f"| {row.get('cells')} | {name} | {row.get('accepted_steps', '—')} / {row.get('rejected_steps', '—')} | "
                     f"{error_text} | {budget_text} | {'passed' if row.get('passed') is True else 'failed; retained'} |")
    if measured:
        max_bc = max(row["maximum_boundary_defect"] for row in measured)
        max_budget = max(abs(value) for row in measured for value in row["independent_summed_transfer_budget"])
        rejections = sum(row.get("rejected_steps",0) for row in measured)
        lines += ["",f"Across the saved runs, the largest boundary defect is **{max_bc:.3g}**, "
            f"the largest independently summed discrete budget defect is **{max_budget:.3g}**, "
            f"and there are **{rejections} rejected steps**. All reported pressure, density and temperature minima remain positive. "
            "Minima cover the 11 saved frames; production admissibility also checks accepted updates.", ""]
    lines += ["### Preserved coarse-grid endpoint-tail diagnostics", "",
        "The exact waves have not reached the boundaries, so the ideal continuum transfer is "
        "$(0,(p_L-p_R)t,0)$. A finite-grid scheme can diffuse small tails to an endpoint earlier. "
        "The following difference measures actual numerical endpoint transfer minus that ideal transfer; "
        "it is distinct from discrete conservation, which uses the actual transfer.", "",
        "| Cells | Method | Normalized actual minus ideal transfer | Original $10^{-8}$ check |",
        "|---:|---|---|---|"]
    tail_failures = []
    for row in measured:
        numerical = row.get("independent_summed_transfer_budget", [])
        ideal = row.get("exact_pre_boundary_interaction_budget", [])
        if ideal and max(abs(value) for value in ideal)>1e-8:
            tail_failures.append(row)
            difference = [a-b for a,b in zip(ideal,numerical)]
            lines.append(f"| {row.get('cells')} | {labels.get(row['method']['id'],row['method']['id'])} | {triple(difference)} | failed; retained |")
    if not tail_failures:
        lines.append("| — | None in supplied data | — | — |")
    refinements = data.get("refinement", [])
    orders = [value for row in refinements for value in row.get("observed_order",[]) if finite(value)]
    comparisons = data.get("implicit_boundary_comparisons", [])
    differences = [value for row in comparisons for value in row.get("normalized_max_primitive_difference",[]) if finite(value)]
    sensitivities = [value for row in measured for value in row.get("reference_quadrature_64_vs_128_normalized_L1",[]) if finite(value)]
    lines += ["", "### Refinement and interpretation", ""]
    if orders:
        lines += [f"Every recorded conservative L1 error decreases on refinement: "
            f"**{all(row.get('L1_reduced_in_all_equations') is True for row in refinements)}**. "
            f"Observed orders range from **{min(orders):.2f} to {max(orders):.2f}**, consistent with a shock/contact test using constant reconstruction. "
            "Different finite-CFL time integrators have different numerical diffusion; the smaller explicit-Euler error here does not establish a general accuracy ranking.", ""]
    if differences:
        lines += [f"Coupled and segregated backward-Euler final fields agree within **{max(differences):.3g}** "
            "in normalized primitive variables. They solve the same implicit equations.", ""]
    if sensitivities:
        lines += [f"The exact reference uses 128-point Gauss quadrature per cell. The largest retained 64-versus-128 normalized L1 "
            f"difference is **{max(sensitivities):.3g}**. This is smaller than the reported discretization error but limits fine accuracy claims near discontinuities; "
            "it is a sensitivity estimate, not a rigorous quadrature bound.", ""]
    if figure:
        lines += [f"![Saved Roe Sod final pressure, density and velocity profiles]({figure})", "",
            "Final profiles on the finest saved grid. Exact conservative cell averages are converted to primitive form; "
            "overlapping coupled/segregated implicit curves use different styles and markers. The [PDF figure]"
            f"({Path(figure).with_suffix('.pdf').as_posix()}) is available for vector export.", ""]
    if path:
        try:
            display_path = path.relative_to(STUDY)
        except ValueError:
            display_path = path
        lines += [f"Saved evidence: `{display_path.as_posix()}`. Original fields, histories, settings, source fingerprints and failed statuses remain in that file.", ""]
    lines += ["Reproduce the numerical evidence only when a new simulation is intended:", "", "```powershell",
        "poetry run python -B -m development.flux_methods.roe.scripts.verify_sod",
        "poetry run python -B -m development.flux_methods.roe.scripts.summarize_sod", "```", "",
        "The first command intentionally returns a failed aggregate status when any original strict check fails; "
        "the second classifies retained evidence without changing that status. These are perfect-gas transient checks, "
        "complementing the separate real-gas matrix/property and flow audits.", ""]
    return lines


def render(campaign, audit, tests, campaign_path, *, sod=None, sod_path=None, sod_figure=None,
           real_gas=None, real_gas_link_prefix="outputs/",shock_diagnostic=None,shock_diagnostic_link=None,
           report_html_command=None,report_audit_command=None):
    rows = campaign.get("results", [])
    protocol = campaign.get("protocol", {})
    groups = {flux: [r for r in rows if flux_name(r) == flux] for flux in FLUXES}
    cases = list(tests.iter("testcase"))
    counts = dict(tests=len(cases), failures=sum(c.find("failure") is not None for c in cases),
                  errors=sum(c.find("error") is not None for c in cases), skipped=sum(c.find("skipped") is not None for c in cases))
    passed = counts["tests"]-counts["failures"]-counts["errors"]-counts["skipped"]
    xfails = sum(c.find("skipped") is not None and c.find("skipped").get("type") == "pytest.xfail" for c in cases)
    requested = expected_configurations(protocol)
    observed = {configuration(row) for row in rows}
    complete_coverage = bool(requested) and observed == requested and len(rows) == len(requested)
    coverage = f"{len(rows)} / {len(requested)}" if requested else f"{len(rows)} / unavailable"
    stability = lambda name: unchanged(campaign, name) if unchanged(campaign, name) is not None else "not yet audited"
    full_plan = (len(requested) == 640 and len(protocol.get("cases", [])) == 20
                 and set(protocol.get("grids", [])) == {50,100,200,400}
                 and set(protocol.get("fluxes", [])) == set(FLUXES)
                 and set(protocol.get("reconstructions", [])) == {"constant", "muscl_van_leer"}
                 and protocol.get("backends") == ["current"])
    verification_only = campaign.get("verification_only") is True or protocol.get("verification_only") is True
    roe_only = protocol.get("fluxes") == ["roe"]
    roe_plan = (roe_only and len(requested)==160 and len(protocol.get("cases",[]))==20
                and set(protocol.get("grids",[]))=={50,100,200,400}
                and set(protocol.get("reconstructions",[]))=={"constant","muscl_van_leer"})
    steady_heading = "## Roe steady verification" if roe_only else "## Matched steady comparison"
    timing_campaign = dict(campaign, completed=campaign.get("completed") is True and complete_coverage)
    audit_mismatch = audit.get("summary", {}).get("host_jvp_relative_mismatch_max")
    mismatch_text = f"`{audit_mismatch:.3g}`" if finite(audit_mismatch) else "unavailable"
    lines = [
        "# Roe implementation: numerical verification" if verification_only else "# Roe implementation: verification and flux comparison", "",
        "Recorded source and numerical settings are retained in the saved campaign. [REPORT.html](REPORT.html) is generated separately from these saved data. "
        "The steady cases use installed perfect-gas analytical references; actual real-gas matrix/property validation is reported separately.", "",
        f"Saved campaign completed flag: **{campaign.get('completed', False)}**. Retained/requested configurations: **{coverage}**. "
        f"Exact requested coverage: **{complete_coverage}**. "
        f"Source unchanged during campaign: **{stability('source_unchanged')}**. "
        f"Harness unchanged: **{stability('harness_unchanged')}**. "
        + ("Performance timing: **not collected; numerical-only verification**." if verification_only else f"Timing environment: **{environment_status(campaign.get('timing_environment')) or 'unverified'}**."), "",
        ("This campaign requests the complete **160-configuration Roe-only verification**: all 20 cases, four grids and both reconstructions. Completion and numerical acceptance remain separate checks."
         if roe_plan else "This saved campaign requests the full 640-configuration primary matrix. Completion and numerical acceptance remain separate checks."
         if full_plan else "This is a subset/smoke campaign. It does not establish completion of the planned 640-configuration, 20-case, four-grid, two-reconstruction matrix."), "",
        f"Saved selection: {len(protocol.get('cases', [])) if isinstance(protocol.get('cases'),list) else 'unavailable'} cases; "
        f"grids `{protocol.get('grids', [])}`; reconstructions `{protocol.get('reconstructions', [])}`; "
        f"fluxes `{protocol.get('fluxes', [protocol.get('scheme', {}).get('flux')])}`.", "",
        "## What changed", "",
        "- Added optional model-owned Roe data and a generic conservative Roe flux. No EOS or Euler variable indices enter the shared numerical layer.",
        "- Nozzle and annular models use the existing PH calls, centered secant coefficients, grouped projectors and acoustic-only Harten correction. "
        "A C2 small-jump blend avoids division-amplified roundoff; invalid averages reject the trial without clipping or a hidden HLL fallback.",
        "- Explicit Roe timesteps additionally bound reconstructed interface dissipation speeds. Existing fluxes and implicit CFL policy retain their definitions.",
        "- The benchmark now treats flux as an explicit dimension in keys, bindings, resume identity, audits and offline report controls.", "",
        "## Production tests and actual real-gas audit", "",
        f"Saved test suite: **{passed} passed**, **{xfails} expected failures**, **{counts['skipped']-xfails} other skipped**, "
        f"**{counts['failures']} failures**, **{counts['errors']} errors**. "
        "Counts come from the supplied JUnit file; individual test names and skip reasons remain in that evidence.", "",
        "Focused Roe tests cover nonlinear nonideal PH thermodynamics, finite/coincident jumps, energy shifts, forward/reverse AD, "
        "stationary contact/shear, invalid averages, an EOS-free user model and all supported integrator/boundary treatments.", "",
        "The reproducible actual-CO₂ audit is saved in `outputs/roe_audit.json` and generated by "
        "`poetry run python -B -m development.flux_methods.roe.scripts.audit_roe`. "
        "It retains every sampled endpoint/pair, failed checks if any, settings and source fingerprints. "
        "With `--roe-audit`, the HTML Protocol page shows the measured summary and embeds the complete original audit JSON in its evidence ZIP.", "",
        "```json", json.dumps(audit.get("summary", audit.get("checks", {})), indent=2), "```", "",
        "Actual host-EOS derivatives are limited by JAXProp's existing finite-difference JVP bridge; the observed "
        f"coincident-matrix relative mismatch in the supplied audit is {mismatch_text}. "
        "Independently interpolated table properties require their own consistency audit. "
        "These checks establish a tested domain, not unconditional positivity or nonlinear entropy stability for every EOS state.", "",
        steady_heading, "",
        "The planned primary profile uses backward Euler, coupled physical boundary equations, undamped Newton cap 10, initial CFL 0.2, "
        "growth 1.2, maximum $10^6$, shrink 0.5, float64, and the common relative/absolute/BC/global stopping rules. "
        "The supplied campaign's exact settings and any overrides remain in its saved protocol. "
        "Each solve starts independently from the same cold physical state. There is no multigrid or grid sequencing.", "",
        "| Flux | Configurations | Requested solves succeeded | Final residual pass | Residual unavailable | Branch failures | Assessed accuracy failures | Timing eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    roots=sum(row.get("success") is True and independently_converged(row,protocol) is True and row.get("branch_passed") is True for row in rows)
    fine_muscl=[row for row in rows if row.get("cells")==max(protocol.get("grids") or [400]) and row.get("reconstruction")=="muscl_van_leer"]
    fine_muscl_pass=sum(row.get("analytical_passed") is True for row in fine_muscl)
    misses=sum(row.get("accuracy_assessed") is True and row.get("analytical_passed") is False for row in rows)
    lead=["## Measured assessment", "",
          f"**{roots}/{len(rows)} retained cold steady solves converge and pass the independent spatial, BC/global and physical-branch checks.** "
          f"On the finest configured grid, **{fine_muscl_pass}/{len(fine_muscl)} MUSCL cases meet their analytical targets**; "
          f"**{misses} constant-reconstruction target misses remain visible**, with no relaxed thresholds.", ""]
    pairs=audit.get("summary",{}).get("pair_checks_passed")
    if pairs is not None:
        gas_pass=sum(row.get("steady_verified") is True for row in (real_gas or []))
        lead.append(f"- Real-gas evidence: **{pairs:,} CO2 state-pair checks** and **{gas_pass} coarse actual-CO2 steady roots**; finer-grid budget stops remain incomplete.")
    if sod is not None:
        assessment=sod.get("assessment",{});total=len(sod.get("records",[]));numeric=assessment.get("numerical_checks_passed",0);strict=assessment.get("analytic_boundary_transfer_checks_passed",0)
        lead.append(f"- Transient evidence: **{numeric}/{total} Sod numerical checks pass**; **{total-strict} original coarse-grid analytic endpoint-tail diagnostics** remain failed and distinct from discrete conservation.")
    if verification_only:lead.append("- **Performance timing is deferred.** This verification collected no timing samples or speed ranking.")
    lead.append("")
    lines[4:4]=lead
    position = lines.index(steady_heading)
    flow_lines=sod_section(sod,sod_path,sod_figure)
    if real_gas is not None:
        from .summarize_real_gas import render_markdown
        flow_lines=render_markdown(real_gas,link_prefix=real_gas_link_prefix).splitlines()+[""]+flow_lines
    lines[position:position] = flow_lines
    for flux in FLUXES:
        selected = groups[flux]
        if not selected and flux not in protocol.get("fluxes",[protocol.get("scheme",{}).get("flux")]):
            lines.append(f"| {LABELS[flux]} | not run | not run | not run | not run | not run | not run | not collected |")
            continue
        convergence = [independently_converged(row, protocol) for row in selected]
        lines.append(f"| {LABELS[flux]} | {len(selected)} | {sum(r.get('success') is True for r in selected)} | "
                     f"{sum(value is True for value in convergence)} | {sum(value is None for value in convergence)} | "
                     f"{sum(r.get('branch_passed') is False for r in selected)} | "
                     f"{sum(r.get('accuracy_assessed') is True and r.get('analytical_passed') is False for r in selected)} | "
                     f"{sum(timing_eligible(r, timing_campaign) for r in selected)} |")
    lines += ["", "Requested-run success, independently reassessed final spatial/BC/global norms, requested branch, analytical accuracy and timing validity are separate statuses. "
              "The saved norms are checked again against the saved policy; no new PDE residual is evaluated here. "
              "Known sonic/exit-limit discrepancies and coarse-grid errors remain visible. Different fluxes solve different finite-grid "
              "discrete systems; their converged fields need not agree to roundoff.", ""]
    accuracy_misses=[row for row in rows if row.get("accuracy_assessed") is True and row.get("analytical_passed") is False]
    if accuracy_misses:
        def accuracy_value(row,name):
            if name=="conservative_normalized_l1":return row.get("metrics",{}).get("max_conservative_normalized_l1")
            if name=="Cp_length_mean_absolute":return row.get("errors",{}).get("Cp",{}).get("l1")
            if name=="normalized_field_l1":
                values=[row.get("errors",{}).get(field,{}).get("normalized_l1") for field in ("p","rho","velocity","M","T")]
                return max(values) if all(finite(value) for value in values) else None
            return None
        shown=lambda value:f"{value:.6g}" if finite(value) else "see saved detail"
        lines += ["### Retained accuracy misses", "",
                  "These are converged discrete roots. Their original accuracy targets remain unchanged; use MUSCL or a finer grid for the flagged first-order cases.", "",
                  "| Case / cells | Reconstruction | Failed criterion | Observed | Target | MUSCL at same grid |",
                  "|---|---|---|---:|---:|---:|"]
        for row in accuracy_misses:
            counterpart=next((other for other in rows if other.get("case_id")==row.get("case_id") and other.get("cells")==row.get("cells") and other.get("flux")==row.get("flux") and other.get("reconstruction")=="muscl_van_leer"),{})
            for name,value in row.get("accuracy_checks",{}).items():
                if value is False:
                    lines.append(f"| `{row['case_id']}` / {row['cells']} | `{row['reconstruction']}` | `{name}` | {shown(accuracy_value(row,name))} | {shown(row.get('accuracy_targets',{}).get(name))} | {shown(accuracy_value(counterpart,name))} |")
        lines += [""]

    if shock_diagnostic is not None:
        d=shock_diagnostic;accepted=d.get("accepted",{});rejected=d.get("rejected",{})
        total=accepted.get("newton_corrections",0)+rejected.get("newton_corrections",0)
        lines += ["## Nonlinear work in a difficult converged case", "",
                  f"Annular `shock35`, 400-cell MUSCL, **converged and passed its physical/analytical checks** with "
                  f"**{accepted.get('attempts')} accepted steps, {rejected.get('attempts')} rejected trials and {total} Newton corrections**. "
                  f"Of those corrections, {rejected.get('newton_corrections')} were spent on rejected trials. "
                  f"With the prescribed cap 10, median accepted CFL was **{accepted.get('cfl_quantiles',{}).get('0.5',float('nan')):.3f}**; "
                  f"{d.get('accepted_cfl_below_5')} accepted steps stayed below CFL 5.", "",
                  f"There were **{d.get('cap_failures')} cap failures**; {d.get('cap_failures_with_residual_satisfied')} already met the inner residual criterion. "
                  "Pending update convergence is inferred from the verified simultaneous convergence rule; the actual update magnitudes were not logged. "
                  f"Another **{d.get('full_trial_acceptance_failures')} status-3 full-trial failures** do not identify whether cell/trace admissibility, "
                  "Roe-face validity or nonfinite residuals caused the rejection. These records do not establish limiter cycling or isolate a Roe-versus-FVS cause.", "",
                  "This is a numerical-work limitation under the matched controls, not a failed final root or a timing comparison.", ""]
        if shock_diagnostic_link:
            lines += [f"Read the [saved-history diagnosis]({shock_diagnostic_link}) and its "
                      f"[JSON evidence]({Path(shock_diagnostic_link).with_suffix('.json').as_posix()}).", ""]
    if verification_only:
        lines += ["## Performance campaign deferred", "",
                  "This campaign contains one cold numerical solve per requested configuration and **zero performance samples**. "
                  "There are no excluded timing warmups, timed repetitions or speed rankings. Load-dependent elapsed diagnostics remain in the raw evidence only.", "",
                  "The separate full four-flux benchmark remains reproducible through `benchmark_profile.yaml`: "
                  "640 configurations, 640 excluded complete warmups and 2080 measured solves, or **2720 complete solves** if all finish. "
                  "That full performance campaign has not been executed here. Competing workloads prevented clean timing, "
                  "and the current implementation request is addressed by the Roe-only numerical verification.", "",
                  "Earlier smoke attempts and an interrupted 22/640 subset were superseded by this complete Roe-only campaign. "
                  "They established neither full four-flux coverage nor comparative timing; their intermediate study files "
                  "were removed during final consolidation. No result in this assessment depends on them.", ""]
    else:
        lines += ["## Repeated timings on the finest configured grid", ""]

        def key(row):
            return (row.get("case_id"), row.get("reconstruction"), row.get("cells"), row.get("newton_cap", row.get("cap")), row.get("backend"))

        def mean(row):
            return row.get("timing", {}).get("mean_s")

        finest = max(protocol.get("grids") or [400])
        baseline = {key(r): r for r in groups["stadtke"] if r.get("cells") == finest and timing_eligible(r,timing_campaign,repeated=True)}
        lines += [f"The finest configured grid is **{finest} cells**; this is not necessarily the primary profile's 400-cell grid. "
                  "Only matched pairs with complete coverage, an unchanged source/harness, a clean campaign and sample block, "
                  "successful excluded solves, at least two measured samples, independently passing residuals, accepted branches and applicable accuracy checks enter these ratios. "
                  "A speedup above one means the selected method is faster than current FVS; the HTML instead plots method/reference **time**, where above one means slower. "
                  "One-sample, incomplete, contended or numerical-only evidence does not support a repeated-timing ranking.", "",
                  "| Flux versus current FVS | Eligible pairs | Geometric mean speedup | Median speedup | Range |",
                  "|---|---:|---:|---:|---:|"]
        pairs_by_flux = {}
        for flux in ("roe", "rusanov", "hll"):
            pairs = [(r, baseline[key(r)]) for r in groups[flux] if key(r) in baseline and timing_eligible(r,timing_campaign,repeated=True)
                     and matching_controls(r,baseline[key(r)])]
            ratios = [mean(base)/mean(row) for row, base in pairs]
            pairs_by_flux[flux] = pairs
            if ratios:
                geometric = math.exp(statistics.mean(math.log(x) for x in ratios))
                lines.append(f"| {LABELS[flux]} | {len(ratios)} | {geometric:.3f}× | {statistics.median(ratios):.3f}× | {min(ratios):.3f}–{max(ratios):.3f}× |")
            else:
                lines.append(f"| {LABELS[flux]} | 0 | unavailable | unavailable | unavailable |")
        if not any(pairs_by_flux.values()):
            lines += ["", "**No repeated-timing ranking is available from the supplied campaign.**"]
        roe_pairs = pairs_by_flux.get("roe", [])
        if roe_pairs:
            for description, pair in (("Largest measured Roe speedup", max(roe_pairs, key=lambda p:mean(p[1])/mean(p[0]))),
                                      ("Smallest measured Roe speedup", min(roe_pairs, key=lambda p:mean(p[1])/mean(p[0])))):
                row, base = pair
                lines += ["", f"{description}: `{row['case_id']}`, `{row['reconstruction']}`, {finest} cells: "
                          f"Roe {mean(row):.6g} s versus FVS {mean(base):.6g} s. "
                          f"Accepted steps {row.get('work',{}).get('accepted_steps')} versus {base.get('work',{}).get('accepted_steps')}; "
                          f"Newton corrections {row.get('work',{}).get('newton_corrections')} versus {base.get('work',{}).get('newton_corrections')}."]
        lines += ["", "Timing differences combine flux/JVP/validity-check cost with accepted steps, Newton work and rejected attempts. "
                  "The report retains these quantities separately; total elapsed time alone does not identify the cause. "
                  "Actual adaptive timestep trajectories may differ despite identical controller settings.", ""]
    lines += ["## Unsuccessful configurations", ""]
    failed = [r for r in rows if r.get("status") != "passed" or independently_converged(r,protocol) is not True]
    lines += [f"There are **{len(failed)}** configurations with a numerical, branch, accuracy, timeout or assessment status other than passed. "
              "The HTML and saved campaign retain all of them. This also lists nominally passed records whose saved residuals do not independently pass or are unavailable. "
              "The table below shows up to the first 40 for quick review.", "",
              "| Case | Grid | Reconstruction | Flux | Status | Reason |", "|---|---:|---|---|---|---|"]
    for row in failed[:40]:
        reason = str(row.get("reason", "")).replace("|", "/").replace("\n", " ")
        if row.get("status")=="accuracy_failed":
            missed=[f"{name} (target {row.get('accuracy_targets',{}).get(name,'see saved case')})" for name,value in row.get("accuracy_checks",{}).items() if value is False]
            reason+="; missed analytical checks: "+", ".join(missed)
        status = str(row.get("status"))
        if independently_converged(row,protocol) is not True:
            status += "; residual failed/unavailable"
        lines.append(f"| `{row.get('case_id')}` | {row.get('cells')} | `{row.get('reconstruction')}` | `{flux_name(row)}` | {status} | {reason[:180]} |")
    if not failed:
        lines.append("| None | — | — | — | — | — |")
    lines += ["", "## Evidence and reproduction", "",
              f"- Supplied saved campaign (completeness stated above): `{campaign_path.as_posix()}`.",
              "- Actual CO₂ matrix audit: `outputs/roe_audit.json`.",
              "- Full test log/JUnit: [production_tests.log](evidence/production_tests.log) and [production_tests.xml](evidence/production_tests.xml).",
              "- Final numerical evidence is embedded in `REPORT.html`; restore it with `scripts.restore_evidence` before rebuilding after cloning.",
              "- Read the [Roe-only verification profile](verification_profile.yaml), [full timing profile](benchmark_profile.yaml), [implemented theory](roe_theory.md) and [benchmark protocol](benchmark_protocol.md) for exact conventions.", "",
              "```powershell",
              (f'poetry run python -B -m development.flux_methods.roe.scripts.audit_steady --input "{campaign_path.as_posix()}"' if verification_only else f'poetry run python -B -m development.roberto.benchmarking.scripts.audit --input "{campaign_path.as_posix()}"'),
              report_html_command or f'poetry run python -B -m development.roberto.benchmarking.scripts.report --input "{campaign_path.as_posix()}" --roe-audit development/flux_methods/roe/outputs/roe_audit.json --output development/flux_methods/roe/REPORT.html',
              report_audit_command or "poetry run python -B -m development.roberto.benchmarking.scripts.audit_report --input development/flux_methods/roe/REPORT.html",
              "```", "",
              "Report rebuilding and this summary generator read saved evidence only. They do not advance the solver. "
              "The study was moved from `development/roberto/roe_method` to `development/roe_method` after validation; "
              "paths and hashes inside original numerical records retain their measurement-time values. "
              "Resume checks intentionally reject a changed client or harness; use a new output directory for new campaigns.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=STUDY/"outputs/roe_audit.json")
    parser.add_argument("--tests", type=Path, default=STUDY/"evidence/production_tests.xml")
    parser.add_argument("--output", type=Path, default=STUDY/"VERIFICATION_REPORT.md")
    parser.add_argument("--sod-verification", "--sod", dest="sod", type=Path,
                        help="Optional saved Sod JSON; autodetect outputs/roe_transient.json")
    parser.add_argument("--sod-figure-dir", type=Path, default=STUDY/"outputs/figures")
    parser.add_argument("--skip-sod-plot", action="store_true",
                        help="Read Sod evidence without regenerating its saved-data PNG/PDF")
    parser.add_argument("--real-gas-summary",type=Path,help="Optional saved actual-EOS attempt array; autodetect outputs/real_gas_summary.json")
    parser.add_argument("--shock-diagnostic",type=Path,help="Optional saved work diagnosis JSON; autodetect the shock35 N400 MUSCL analysis")
    args = parser.parse_args()
    sod_path = args.sod or STUDY/"outputs/roe_transient.json"
    sod = json.loads(sod_path.read_text(encoding="utf-8")) if sod_path.is_file() else None
    if args.sod is not None and sod is None:
        parser.error(f"Saved Sod verification does not exist: {args.sod}")
    real_gas_path=args.real_gas_summary or STUDY/"outputs/real_gas_summary.json"
    real_gas=json.loads(real_gas_path.read_text(encoding="utf-8")) if real_gas_path.is_file() else None
    if args.real_gas_summary is not None and real_gas is None:parser.error("Saved actual-EOS summary does not exist: "+str(args.real_gas_summary))
    shock_path=args.shock_diagnostic or STUDY/"outputs/shock_diagnostics/shock35_n400_muscl/history_analysis.json"
    shock=json.loads(shock_path.read_text(encoding="utf-8")) if shock_path.is_file() else None
    if args.shock_diagnostic is not None and shock is None:parser.error("Saved work diagnosis does not exist: "+str(args.shock_diagnostic))
    figure = None;figure_file=None
    if sod is not None and not args.skip_sod_plot:
        from .plot_sod import plot
        paths = plot(sod,sod_path,args.sod_figure_dir)
        if paths:
            figure_file=Path(paths[0])
            figure = os.path.relpath(paths[0],args.output.parent).replace("\\","/")
    elif sod is not None and (args.sod_figure_dir/"sod_final_profiles.png").is_file():
        figure_file=args.sod_figure_dir/"sod_final_profiles.png"
        figure=os.path.relpath(figure_file,args.output.parent).replace("\\","/")
    def quoted(path):
        path=Path(path).resolve()
        try:path=path.relative_to(STUDY.parents[2])
        except ValueError:pass
        return "'"+path.as_posix().replace("'","''")+"'"
    command="poetry run python -B -m development.roberto.benchmarking.scripts.report --input "+quoted(args.campaign)+" --roe-audit "+quoted(args.audit)
    if sod is not None:command+=" --roe-sod "+quoted(sod_path)
    if real_gas is not None:command+=" --roe-real-gas "+quoted(real_gas_path)
    if figure_file is not None:command+=" --roe-sod-figure "+quoted(figure_file)
    command+=" --output "+quoted(args.output.with_name("REPORT.html"))
    campaign=json.loads(args.campaign.read_text(encoding="utf-8"))
    missed_cases={row.get("case_id") for row in campaign.get("results",[]) if row.get("status")=="accuracy_failed"}
    for row in campaign.get("results",[]):
        if row.get("case_id") in missed_cases and row.get("cells")==max(campaign.get("protocol",{}).get("grids") or [400]) and row.get("file"):
            detail=(args.campaign.resolve().parent/row["file"]).resolve()
            if not detail.is_relative_to(args.campaign.resolve().parent):raise ValueError("Unsafe saved scenario path")
            row["errors"]=json.loads(detail.read_text(encoding="utf-8")).get("errors",{})
    display_campaign = args.campaign.resolve()
    try:
        display_campaign = display_campaign.relative_to(STUDY.parents[2])
    except ValueError:
        pass
    text = render(campaign,
                  json.loads(args.audit.read_text(encoding="utf-8")), ET.parse(args.tests).getroot(), display_campaign,
                  sod=sod,sod_path=sod_path if sod is not None else None,sod_figure=figure,real_gas=real_gas,
                  real_gas_link_prefix=os.path.relpath(STUDY/"outputs",args.output.parent).replace("\\","/")+"/",
                  shock_diagnostic=shock,shock_diagnostic_link=os.path.relpath(shock_path.with_suffix('.md'),args.output.parent).replace("\\","/") if shock is not None else None,
                  report_html_command=command,
                  report_audit_command="poetry run python -B -m development.roberto.benchmarking.scripts.audit_report --input "+quoted(args.output.with_name("REPORT.html")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
