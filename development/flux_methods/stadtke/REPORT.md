# Stadtke FVS and custom corrections: saved verification

Open [REPORT.html](REPORT.html) for the offline Plotly case explorer.

## Matrix findings

- All 128 saved solves and their recorded-epoch fresh original-equation/branch/identity checks pass in this unshifted production matrix.
- Endpoint-only has 0/32 immediate-neighbor pressure peaks; both-on has 0/32. This is a tested local-shape result, not a universal no-peak theorem.
- Both-on has larger conservative-cell-projection pressure L1 error than the original in 30/32 matched configurations, and larger maximum total-enthalpy error in 25/32. There is no blanket analytical-accuracy improvement. Median corrected/original error ratios are 1.44 for pressure and 1.722 for total enthalpy.
- The original MUSCL results contain one local pressure peak: 0.0998816% in euler_annular/shock35 at N400. The large shifted MUSCL example belongs to archived historical evidence, not this unshifted matrix.

Pressure conservative-cell L1 ratios span 0.5707–186.7; the median absolute change is 69.33 Pa. Maximum relative total-enthalpy error ratios span 0.9147–91.96; the median change is 0.4466 percentage points.
Large ratios need an absolute scale: euler_annular/shock70, N400, muscl_van_leer, pressure cell L1 changes from 0.27692 to 51.7085 Pa; maximum relative total-enthalpy error changes from 0.0250728% to 2.3057%. These are deterministic matched errors, not statistical timing evidence.

Retained: 128 solution records; 128 recorded successes; 128 hash-joined evaluation audits; 128 with passing physical/identity checks. Saved population completed flag: True; evaluation population complete: True.

These native cold solves and fresh checks belong to their recorded source/case/client epoch. They are not a current-checkout certification or a migrated Osher campaign.

The four requested nozzle/annular shock cases, both reconstructions and every retained configuration remain explicit selections. Missing or unsuccessful runs remain visible. Coverage, original-equation convergence and analytical accuracy are separate checks.

Refinement uses saved errors against analytical conservative cell projections; point-reference comparisons remain separate. See [THEORY.md](THEORY.md) for the common flux and its explicit parameters.

Listed supplemental production-test/API checks and matrix postprocessing are separate from fresh row audits. Their compact summaries and exact original JSON bytes are embedded with filename and SHA-256 bindings.

This is numerical verification, not a performance benchmark. Compilation-inclusive elapsed diagnostics do not establish a cost ranking. Residual/CFL plots show actual accepted histories; accepted steps, rejected attempts and all Newton corrections are separate work measures.

The HTML embeds all plotted values, rendered mathematical notes, clean audit summaries, manifests, Plotly and MathJax. The download contains current-facing derived data and documents; immutable historical raw records stay archive-only. Structural and executable application audits are separate from browser rendering.

Browser rendering remains unverified. Structural, archive and executable application checks are separate from a browser preview.

Flow and fresh-audit results refer to their recorded source/case/client epoch, not current-checkout byte identity.
