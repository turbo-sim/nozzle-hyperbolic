# Turbo Expo 2025: current HEM reproduction

For the distinction between this paper reproduction and the broader validation
datasets on `main`, see [VALIDATION_COVERAGE.md](VALIDATION_COVERAGE.md).

This application reproduces the operating points and result comparisons of
Cioffi et al., *A Hyperbolic One-Dimensional Model for Two-Phase Flows in
Converging-Diverging Nozzles*, [GT2025-153250](https://doi.org/10.1115/GT2025-153250),
using the current `roberto-branch` implementation. Open [REPORT.html](REPORT.html)
for the portable, offline report, and [RESULTS.md](RESULTS.md) for the saved
numerical coverage and scalar comparisons. The HTML embeds the settings, reference data,
actual final states, convergence histories and exact source snapshots.

The matrix contains 23 operating points and 52 case/method combinations:

- Six perfect-gas conditions, including subsonic and adapted flow, pressure
  ratio 1.56, and shocks at 25%, 50% and 75% of the divergent length; Rusanov,
  Roe and HLL on 500 cells.
- Three frictionless MDM conditions S1–S3; Rusanov and HLL on 250 cells.
- Lettieri L1 and L5, each with smooth-wall friction and without friction;
  Rusanov and HLL on 200 cells.
- Petruccelli P1–P4, with circular section and 2 μm roughness; Rusanov and HLL
  on 600 cells.
- Berana B1–B3 with 2 μm roughness, plus B3 at 0, 1 and 3 μm;
  Rusanov and HLL on 1600 cells.

The five additional Rusanov mesh studies use the Appendix A grids, including
the distinct 400-cell perfect-gas and 250-cell Lettieri finest meshes.
[case_definitions.json](case_definitions.json) maps every figure and table to
its cases. [provenance.json](provenance.json) records the source hierarchy and
unresolved historical inconsistencies. Original workbook bytes are pinned to
the paper-version Git revision, with independent SHA-256 and Git-blob identities
in [data/source_manifest.json](data/source_manifest.json).

The spatial reconstruction is piecewise constant, as in the paper. The
current production Rusanov/HLL fluxes and the current Roe implementation for
perfect gas are selected explicitly. The time integrator is coupled backward
Euler (`euler-implicit`) with damped Newton corrections and the production
colored/banded linear algebra. No explicit integration is used. The wall
correlation uses the production Haaland Darcy law, Beattie two-phase viscosity,
zero wall heat transfer and the physical hydraulic diameter.

Run from the repository root in its existing Poetry environment:

```powershell
poetry run python -B -m cases.TurboExpo2025.scripts.run --resume
poetry run python -B -m cases.TurboExpo2025.scripts.run_mesh_studies --resume
poetry run python -B -m cases.TurboExpo2025.scripts.prepare_phase_diagrams
poetry run python -B -m cases.TurboExpo2025.scripts.run --case L5 --cells 50 --flux hll
poetry run python -B -m cases.TurboExpo2025.scripts.run --case P1 --grids 50,100,200,400 --resume
poetry run python -B -m cases.TurboExpo2025.scripts.summarize
poetry run python -B -m cases.TurboExpo2025.scripts.audit
poetry run python -B -m cases.TurboExpo2025.scripts.report
poetry run python -B -m cases.TurboExpo2025.scripts.audit_report
```

`--cells` is an explicit pilot override. `--grids` prolongs only independently
verified solutions of the same case and method. A failed coarse state is saved;
the requested finest mesh then receives an independent cold attempt. All
attempts have separate directories and are never overwritten. `--resume` skips
only records with identical effective numerical settings and source hashes.
Property preparation is separate from solve timing. Recorded wall times are
individual compilation-inclusive observations, without a performance claim.
Some case campaigns execute concurrently; these observations must not be used
as controlled cross-method timing comparisons. Supersonic cases use an
explicit cold pressure datum $0.25p_0$ while retaining an extrapolated physical
outlet without prescribed backpressure.

The report and summary commands use saved data only. They do not import an EOS
or advance a simulation. The numerical audit checks coverage, settings, actual
saved states and the recorded independent equation targets; the browser audit
copies the HTML alone, disables networking, exercises its controls and checks
the downloaded evidence archive.

Use `scripts.audit --require-converged` for a stricter gate requiring every
primary and Appendix mesh to recover its intended regime. It returns a
nonzero status for the two coarse Berana branch discrepancies recorded in
[RESULTS.md](RESULTS.md), while the ordinary evidence audit checks that those
outcomes are preserved correctly.

Convergence uses every original spatial balance, boundary equations, global
numerical-flux budgets, EOS admissibility and the intended physical branch.
The dimensionless, volume-weighted spatial RMS uses fixed reservoir scales:

$$
R_k=\left[\frac{\sum_i V_i\left(t_\mathrm{ref}r_{ik}/(V_iq_{\mathrm{ref},k})\right)^2}{\sum_iV_i}\right]^{1/2}.
$$

Every equation must satisfy both $R_k\leq10^{-9}$ and its selected relative
reduction of $10^{-6}$; boundary and global defects must each be at most
$10^{-9}$. Small Newton corrections alone do not establish convergence.
Actual residuals and unsuccessful states remain available in the report.
Published-data agreement is a separate comparison, with no artificial acceptance
tolerance fitted to the new results.

The current model and common-face finite-volume geometry differ from the
historical area-weighted node formulation. The current stationary adiabatic
wall removes momentum without removing total energy. Native equilibrium HEOS
bicubic properties replace the publication-era property path; interpolation
stencils are audited before solving. The CO₂ table has eight unchanged
Grüneisen coefficient cells with nonfinite values. Their locations are saved,
and the native finite-$G$ admissibility check rejects states there; all other
required thermodynamic stencils are finite. Independent JIT probes confirmed
rejection at 40 interior points in those eight cells. The initial failed
whole-rectangle preparation is retained. No property is filled or substituted.
Beattie wall viscosity uses documented tabulated saturated phase transport up
to $0.98p_\mathrm{crit}$; native mixture viscosity applies outside that range.
These are recorded formulation
differences, consistent with using the current code.

Several limits prevent an exact historical regression. The original simulation
NPZ files were stored outside Git, and P1–P4 reference CFD profiles were not
recovered. Separately reviewed digitizations of the visible Figure 9 HEM and
CFD curve segments supply approximate P1–P4 comparisons. Their stroke and
calibration bounds, omitted overlaps/gaps, source-image hashes and visual
overlays are embedded in the report; these samples do not replace the original
profiles. The historical HEM curves in that figure use Rusanov. The extraction
and visual-review workflow is documented in
[scripts/digitize_paper.py](scripts/digitize_paper.py); it never advances CFD.

Spinelli's absolute scale and exact coordinate mapping are uncertain;
the primary comparison follows the archived normalized coordinates, while a
clearly marked alternative alignment is excluded from scoring. The recovered
perfect-gas contour reproduces the paper's area ratio and critical mass flow,
but does not reproduce every tabulated shock-entropy value. Its archived
subsonic backpressure selects a weak shock on that contour; `resolve_case`
records an explicit adjustment to 0.2% above the sonic limiting exit pressure
to recover the intended all-subsonic case. Earlier attempts retain their
original conditions.

The Lettieri Table 3 values labeled kg/s in the paper follow the archived
half-channel flow per unit depth. For rectangular depth $b=6.48$ mm, the
comparable value is $\dot m/(2b)$ in kg/(s m); the actual full-channel kg/s is
reported separately. Historical field values, experimental uncertainties and
unavailable CFD series are not fabricated. A converged current HEM result may
still disagree with nonequilibrium condensation measurements.

The report supplies evidence for reviewing a later merge into `main`.
Generating it does not merge, commit, push or publish the branch.
