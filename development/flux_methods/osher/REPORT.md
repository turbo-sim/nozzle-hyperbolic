# Osher path-flux configuration study: saved verification

Open [REPORT.html](REPORT.html) for the offline Plotly case explorer.

## Matrix findings

- All 128 preserved solve records pass current unified-API Operator, branch, cold-state and numeric-origin checks. This is a configuration metadata view, not 128 new cold solves.
- Immediate-neighbor pressure peaks occur in 0/128 profiles. Branch excess still reaches 3.356% and maximum total-enthalpy defect 3.632%; these are distinct diagnostics.
- Pressure conservative-cell L1 improves N50 to N400 in 32/32 series, but only 92/96 adjacent refinements improve. No uniform monotone-refinement or global-order claim follows.
- More quadrature points do not uniformly improve solution accuracy. Points per path or per captured panel are not spatial order, and nonlinear work is not a timing ranking.

N400 annular shock70 (constant) maximum relative total-enthalpy defects for Gauss3/Gauss5/Gauss9/sonic-partitioned5: 3.206%/3.181%/3.179%/3.179%.
N400 annular shock70 (muscl_van_leer) maximum relative total-enthalpy defects for Gauss3/Gauss5/Gauss9/sonic-partitioned5: 1.259%/1.294%/1.289%/1.29%.
Accepted/rejected/Newton totals across 32 matched configurations: Osher · Gauss 3 5044/908/40719; Osher · Gauss 5 5024/903/40719; Osher · Gauss 9 5010/899/40507; Osher · sonic-partitioned 5 5060/913/41049. These are nonlinear-work counts, not a timing ranking.

Retained: 128 solution records; 128 recorded successes; 128 hash-joined evaluation audits; 128 with passing physical/identity checks. Saved population completed flag: True; evaluation population complete: True.

Preserved historical solves and clean derived configuration metadata, with separate current unified-API Operator audits; not a new cold-solve campaign.

The four requested nozzle/annular shock cases, both reconstructions and every retained configuration remain explicit selections. Missing or unsuccessful runs remain visible. Coverage, original-equation convergence and analytical accuracy are separate checks.

Refinement uses saved errors against analytical conservative cell projections; point-reference comparisons remain separate. See [THEORY.md](THEORY.md) for the common flux and its explicit parameters.

This report embeds current saved-row Operator evaluations. No separate production-test or matrix-postprocessing supplement is included.

This is numerical verification, not a performance benchmark. Compilation-inclusive elapsed diagnostics do not establish a cost ranking. Residual/CFL plots show actual accepted histories; accepted steps, rejected attempts and all Newton corrections are separate work measures.

The HTML embeds all plotted values, clean audit summaries, manifests and Plotly. The download contains current-facing derived data and documents; immutable historical raw records stay archive-only. Structural and executable application audits are separate from browser rendering.

Browser rendering remains unverified. Installed Chrome static rendering was blocked by Windows IPC permissions and no image exporter was installed; no bypass was attempted.

Historical execution source hashes and origin hashes remain separate from current unified-API evaluation hashes. Numeric states, references, errors, histories and work are preserved; metadata migration and current Operator checks do not constitute new cold solves. Historical raw JSON is archive-only.
