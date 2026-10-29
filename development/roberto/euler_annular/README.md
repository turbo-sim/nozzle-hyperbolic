# EulerAnnular verification

The production model retains angular-momentum storage, with third conserved
variable $\rho\ell=\rho r v_\theta$. Named `shear_stress_wall` and `heat_flux_wall`
placeholders now supply zero-default or prescribed scalar/per-cell profiles.
They apply meridional drag, angular wall torque and wall heating; no empirical
friction or heat-transfer correlation is assumed.

Open **[REPORT.html](REPORT.html)**. Copying that single file is enough to read
it offline. Its internal pages contain:

- **Results:** nine cases, 50/100/200/400 physical cells, analytical/constant/MUSCL
  fields for pressure, density, flow angle and three Mach diagnostics; retained
  meridional/frontal/3D geometry views; residual/CFL histories, four conservation
  histories, refinement errors and a final overview table.
- **Theory:** general-EOS equations, eigenanalysis, flux/reconstruction,
  complete implicit residual, Newton machinery and simultaneous boundary blocks.
- **Implementation:** public API, fluid and NURBS contracts, shared code and
  reproduction commands.
- **Notes:** controller evidence, full-campaign comparison, physics qualifications,
  retained failures and limitations. The original 42 controller experiments are
  available as an embedded ZIP download.

The current report uses `outputs/grid_sequence_monitored_full/`: all nine cases
rerun on 50/100/200/400 cells with constant and MUSCL reconstruction. Every active
equation must reduce by nine orders from its actual mesh-specific initial norm,
with separate $10^{-8}$ absolute, boundary and global safeguards and enabled
Cp/mass-flow stationarity checks. The 50-cell guess is cold; successful coarse
fields initialize finer meshes through bounded primitive interpolation. A failed
pair forces a cold restart on the next mesh. This is initialization, not a
conservative time remap or nonlinear FAS correction. The earlier complete
`outputs/grid_sequence/` campaign remains archived with its original histories.

All 72 new solves are retained: 68 meet every criterion; four N400 MUSCL runs
(radial outflow/inflow, conical and curved subsonic) stagnate at roundoff above
their relative target. After 64 bitwise-unchanged accepted states at maximum CFL,
the monitored driver records a failed stagnation exit rather than repeating an
identical deterministic solve. Set `run.stagnation_steps: 0` to disable that exit.

The Results page has eight native collapsible boxes, Expand all/Collapse all
controls and automatic reopening through sidebar links. Its 2×2 convergence view
shows overall $R=\max_kR_k$ and actual trial CFL above local relative equation
norms and global conservation defects. Local RMS norms prevent cell cancellation;
global defects sum signed imbalances and use fixed reference flux scales. They
are complementary checks. Cp and mass-flow plots are in an optional box below.

Each mesh resets the fixed controller to CFL 0.2, with growth 1.2, shrink 0.5 and
at most eight Newton corrections; both strict inner tolerances remain
$10^{-10}$. Production solver defaults remain separate. The numerical method
accepts general JAXProp fluids; the verification campaign uses a perfect-gas
backend and zero wall sources. Separate `outputs/wall_sources/` checks cover
prescribed nonzero shear and heat.

## Folder structure

```text
euler_annular/
  README.md
  REPORT.html              one standalone document with four internal pages
  old_notes/               completed questionnaire, plan and planning snapshot
  scripts/                 verification, audits and one HTML report generator
    report_assets/         editable HTML chapters, CSS/JS and embedded libraries
  outputs/                 numerical data, figures, audits and provenance
```

The retired CFL investigation folder and duplicate Markdown theory/specification
notes have been removed. Their essential findings are incorporated in the
report. Current numerical data and previous campaigns are retained in `outputs/`.
The implementation/environment snapshot is in
`outputs/provenance/implementation_manifest.json`.

## Rebuild the report without rerunning simulations

Run from the repository root in the existing Poetry environment:

```powershell
poetry run python -m development.roberto.euler_annular.scripts.reassess_accuracy
poetry run python -m development.roberto.euler_annular.scripts.write_html_report --verification-folder outputs/grid_sequence_monitored_full
poetry run python -m development.roberto.euler_annular.scripts.audit_html_report
```

Edit chapter sources in `scripts/report_assets/` and rebuild. All styles,
Plotly/MathJax libraries, result data and offered downloads are embedded. The
static audit checks the saved samples, math, links, source hashes and embedded
evidence. JavaScript figure/navigation audits are recorded in `outputs/html_report/`;
they are distinct from browser visual inspection.

## Rerun the current verification campaign

```powershell
poetry run python -m development.roberto.euler_annular.scripts.verify_grid_sequence
poetry run python -m development.roberto.euler_annular.scripts.audit_results --output development/roberto/euler_annular/outputs/grid_sequence_monitored_full
poetry run python -m development.roberto.euler_annular.scripts.audit_grid_sequence
poetry run python -m development.roberto.euler_annular.scripts.audit_shocks --folder development/roberto/euler_annular/outputs/grid_sequence_monitored_full
poetry run python -m development.roberto.euler_annular.scripts.write_manifest
poetry run python -m development.roberto.euler_annular.scripts.write_html_report
poetry run python -m development.roberto.euler_annular.scripts.audit_html_report
```

The campaign attempts all case/mesh pairs, retains failed states and returns a
failing status if any convergence or accuracy target is missed. A failed coarser
pair never seeds a finer mesh. `--resume` checks exact source, YAML, seed and
parent fingerprints; `--aggregate-only` rebuilds summaries without solves.
The equation relative target is $10^{-9}$; the separate absolute safeguard is
$10^{-8}$. Neither is relaxed to make a failure pass. Earlier campaigns remain
under `outputs/grid_sequence/` and `outputs/grid_sequence_monitored/`.

Cp accuracy uses the meridional-length mean absolute difference throughout the
domain, with a limit of 0.01. Maximum Cp error remains a diagnostic. Accuracy and
convergence are assessed separately on the actual returned fields. A dedicated
`reassess_accuracy --folder ...` command can write a reassessment overlay without
changing raw states, histories or parent hashes. Current finest-grid accuracy
passes 16/18 assessments: constant radial inflow misses mean Cp; constant
supersonic expansion misses conservative energy L1.

The existing `.venv/Scripts/python.exe` may replace `poetry run python`.

## YAML cases and a single-case runner

The physical definitions previously embedded in Python now live in
[`cases/verification.yaml`](cases/verification.yaml). Its twelve named overrides
extend [`cases/common.yaml`](cases/common.yaml), which holds the default fluid,
geometry, mesh, numerical settings, residual policy, quantity monitors and CFL
settings. The nine standard verification cases retain exactly the same geometry,
reference invariants, boundary data and shock locations. Numerical initial states
remain uniform physical cold guesses. The analytical reference selects the
verification boundary data; it does not initialize the numerical field.

Run one case from the repository root:

```powershell
poetry run python development/roberto/euler_annular/run_case.py --case radial_outflow --cells 100 --reconstruction linear --print-every 10
poetry run python -m development.roberto.euler_annular.scripts.run_case --case shock35 --cells 200 --reconstruction constant
poetry run python development/roberto/euler_annular/run_case.py --list-cases
poetry run python development/roberto/euler_annular/run_case.py --case curved_subsonic --dry-run
```

`linear` selects `muscl_van_leer`. The runner also accepts `--config`,
`--max-steps` and `--output` (a directory or explicit JSON filename). YAML paths
may be absolute, relative to the current working directory, or relative to this
study. Overrides are saved in the resolved input data. The table prints the
initial state, accepted-update index, CFL, timestep, inner Newton work, four
initial-normalized equation norms (R1–R4), their active-equation maximum R and
separate boundary/global checks, within 100 characters per line. Integral
quantities remain in the saved history and stopping policy, outside the printed
table. Rejected attempts are explicitly marked as retries; their
unevaluated outer diagnostics remain unavailable. Results include full history,
final fields, residual definitions and fixed scales, monitor stability, YAML
source hashes and numerical source fingerprints. Nonfinite failure diagnostics
are exported as JSON null with their original kind and location recorded.

The verification runner currently supports the perfect-gas reference cases and
the radial, conical and circular-arc geometries listed in the YAML schema. This
does not restrict the general-EOS production solver. Wall shear and heat inputs
in this case runner are prescribed scalar values in Pa and W/m² respectively.
Empirical wall correlations remain outside the implementation.

The default quantity monitors are the length-mean pressure-recovery coefficient
and outlet numerical mass flux. Add or replace entries under `monitors`, for
example:

```yaml
monitors:
  - name: Cp_exit
    quantity: pressure_coefficient
    reduction: outlet
    convergence: true
    window: 20
    absolute_tolerance: 1.0e-8
    relative_tolerance: 1.0e-6
    reference: 1.0
  - name: mdot_in
    quantity: mass_flow
    reduction: inlet
    convergence: false
```

Cp supports `outlet` and `length_mean`; mass flow supports `inlet`, `outlet`
and `mean` over numerical faces. Cp uses the current inlet trace pressure and
the case's fixed total pressure. Mass flow comes from the actual reconstructed
numerical mass flux, rather than a separate cell-centre approximation. The
length mean is $\sum_i\Delta m_i C_{p,i}/\sum_i\Delta m_i$. It is a monitored
solution quantity, separate from the mean absolute Cp error against the reference.

With `convergence: true`, a quantity must satisfy the complete accepted-sample
window test

$$
\max_{\text{window}}J-\min_{\text{window}}J
\leq \text{absolute_tolerance}
+\text{relative_tolerance}\,\text{reference}.
$$

The reference is fixed and positive, so a quantity crossing zero is handled
without dividing by its current value. `convergence: false` records and prints
the quantity without making its stationarity a stopping requirement. `monitors:
[]` disables quantity monitors. Stability alone never establishes convergence:
every transport equation and the separate boundary/global constraints must pass.

The new residual policy requires nine orders of reduction for each initially
active equation, together with the fixed absolute safeguard. Each mesh in a
grid sequence starts its own normalization at its actual interpolated initial
state. Initially zero or roundoff-level equations use an explicitly labelled
absolute check. A nine-order target from an already accurate transferred state
can reach the numerical floor; any resulting nonconverged exit is retained.
Existing saved histories keep their original residual definition and are not
relabeled as new-policy results.

The residual reducers, monitor-window logic, plain-text table and strict JSON
serializer live under `src/nozzle_hyperbolic/solvers/` and are model-independent.
The observable formulas and steady driver adapter currently connect only to
EulerAnnular; EulerNozzle behavior is unchanged.

Audit YAML input validation and physical-definition compatibility without solves:

```powershell
poetry run python -m development.roberto.euler_annular.scripts.audit_case_config
```
