# Roe verification and four-flux benchmark protocol

This is the executable protocol for the implemented Roe method. The physical
cases come from `nozzle_hyperbolic.verification`; the shared runner, assessment,
auditor and offline renderer live in
[`development/historical/benchmarking`](../../historical/benchmarking/README.md).
No solver, geometry or analytical reference is copied into this study.

The completed evidence contains **160 untimed Roe configurations**. All 160
converged to admissible states on their intended branches. At 400 cells,
20/20 MUSCL cases and 16/20 constant cases met their analytical accuracy targets.
The four-flux timing campaign below is configured but has **not** been completed;
the Roe-only results establish no comparative speed ranking. See
[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) for the measured findings and
[roe_theory.md](roe_theory.md) for the construction being tested.

## Case population and methods

The installed catalog contains 20 inviscid steady cases with perfect-gas
analytical references:

| Model | Case suffixes; prefix each with the model name and `/` |
|---|---|
| `euler_nozzle` | `subsonic`, `sonic_limit`, `shock_35`, `shock_70`, `shock_exit`, `overexpanded`, `adapted`, `underexpanded` |
| `euler_annular` | `radial_outflow_m005`, `radial_outflow`, `radial_outflow_m020`, `radial_inflow`, `radial_outflow_swirl`, `conical_subsonic`, `curved_subsonic`, `sonic_exit`, `swirl_diffusion`, `supersonic_expansion`, `shock35`, `shock70` |

`radial_inflow` denotes decreasing physical radius along forward meridional
flow. The expanded case list and original YAML bytes/hashes are recorded with
each campaign; a future catalog addition must not silently alter a resumed run.

[`benchmark_profile.yaml`](benchmark_profile.yaml) selects the current production
backend with `roe`, `rusanov`, `hll` and `stadtke` (the original FVS, with
`custom_jump_correction: false` and `custom_endpoint_envelope: false`), grids
of 50/100/200/400 physical cells, and constant/Van Leer MUSCL
reconstruction. The two algebraic endpoint traces are additional unknowns.
This gives $20\times4\times4\times2=640$ configurations. Different fluxes can
have different finite-grid roots; compare analytical accuracy and refinement
rather than requiring roundoff agreement between their final states.

Every grid, method and repetition starts independently from the same
model-owned cold initialization. There is **no multigrid, grid sequencing,
continuation, analytical initial guess or reuse of a previous solution**.
Analytical references set the physical operating condition and assess accuracy;
they are not numerical initial states.

## Matched numerical controls

Both committed profiles resolve the following controls explicitly:

| Control | Value |
|---|---|
| Integration and BC treatment | Backward Euler; coupled boundary residuals |
| Newton | Undamped; at most 10 corrections; strict convergence required |
| Inner tolerances | Scaled update $10^{-10}$ **and** residual $10^{-12}$ |
| Jacobian and correction solve | Colored AD; block-banded direct linear solver |
| Entropy parameter; conservative filter | `delta_factor=0.1`; filter off |
| CFL | Initial 0.2; accepted-step growth 1.2; maximum $10^6$; failure shrink 0.5 |
| Limits | 16,000 accepted-step budget; 14 retries; stagnation window 64 |
| Spatial stopping norm | Per-equation, volume-weighted dimensionless L2 |
| Spatial stopping rule | Relative reduction $10^{-6}$ **or** absolute $10^{-9}$ |
| Independent checks | Boundary and global defects each at most $10^{-9}$ |
| Initially negligible equations | Absolute criterion below initial floor $10^{-14}$ |
| Precision and threads | JAX float64; recorded single-thread CPU library settings |
| Worker timeout | 300 seconds for a complete worker operation |

The spatial tolerances use fixed physical reference scales; they are not raw
SI residuals. Inner Newton convergence, original spatial convergence, BC/global
conservation and analytical accuracy are separate tests. Reaching the Newton
cap without convergence rejects the timestep. Equal adaptive-controller rules
do not imply equal accepted CFL histories for different numerical fluxes.

Roe additionally records the model-owned `centered_secant_projection_v1`
averaging, `acoustic_harten` entropy policy, `reject` invalid-face policy,
reference scales and resolved small-jump policy. Invalid final faces prevent
acceptance; there is no hidden replacement flux or sound-speed clipping.

## Timing and acceptance

Each configuration receives one excluded **complete cold solve** to warm its
execution path. Bindings/kernels are then reused, but the initial state,
normalization, controller and history are reset before every measured solve.
The coarser grids receive one measured solve; 400-cell configurations receive
ten. If all configurations finish, this is **640 excluded solves plus 2,080
measured solves: 2,720 complete cold solves**.

Methods execute sequentially in rotating order, including an offset between
configuration blocks. JAX work is synchronized before timing ends. The timer
covers the public steady driver, monitoring, accepted and rejected work;
binding/JIT warmup, reference projection, final independent assessment, file
output and rendering are outside the measured loop. Failed warmups and missing
samples remain explicit. The runner checks for competing research workloads
before and after timing blocks; contended samples cannot support speed claims.

Report all raw repeated times and their minimum/mean/maximum. A one-sample
coarse-grid result is not a timing distribution. Plot an actual representative
trajectory, not an average or best-so-far envelope. Keep accepted steps, Newton
corrections including rejected work, and rejected attempts separate.

The worker independently samples the original spatial residual and computes
physical branch/conservation checks and analytical errors after a solve.
Conservative errors use the reference's finite-volume cell projection;
primitive pointwise comparisons retain their own meaning and scales. Errors
are retained on every grid, while case accuracy targets apply at the declared
recommended finest resolution (400 cells here). Known sonic/exit limits remain
visible. Failure, timeout, wrong-branch and accuracy-miss records must not be
discarded, relabeled as passes or repaired by relaxing targets.

Only complete, source-stable, clean, numerically/physically accepted comparisons
are eligible for timing rankings; assessed accuracy must pass too. Repeated
rankings require multiple retained samples. State the eligible population beside
any aggregate comparison. Source, case, profile, harness and machine identities
are recorded; resume only an identical population and revision. Use a fresh
output directory after source or numerical changes.

## Commands

Run from the repository root using the existing Poetry environment. First
generate the PH matrix audit; it checks state pairs, not flow convergence:

```powershell
poetry run python -B -m development.flux_methods.roe.scripts.audit_roe
```

The numerical-only profile performs one cold solve for each of the 160 Roe
configurations and deliberately collects no performance samples:

```powershell
poetry run python -B -u -m development.flux_methods.roe.scripts.verify_steady --config development/flux_methods/roe/verification_profile.yaml --output development/flux_methods/roe/outputs/roe_steady_verification
poetry run python -B -m development.flux_methods.roe.scripts.audit_steady --input development/flux_methods/roe/outputs/roe_steady_verification/summary.json
```

For the full four-flux campaign, use an idle machine and a fresh output name.
The shared runner contains its outputs within the maintained benchmark study:

```powershell
poetry run python -B -u -m development.historical.benchmarking.scripts.run --config development/flux_methods/roe/benchmark_profile.yaml --output outputs/roe_flux_v1 --wait-for-idle
poetry run python -B -m development.historical.benchmarking.scripts.audit --input outputs/roe_flux_v1/summary.json
poetry run python -B -m development.historical.benchmarking.scripts.report --input development/historical/benchmarking/outputs/roe_flux_v1/summary.json --output development/flux_methods/roe/REPORT.html
```

Add `--resume` to the run command only when the recorded identities match.
The auditor checks saved identities, coverage, flags and retained norms without
reevaluating the PDE. Add `--require-numerical-pass` when any numerical or
analytical miss should produce a failing exit status. Generate the report even
when that stricter acceptance check fails so the failures remain reviewable.

The canonical rebuild of the delivered report, including saved CO₂ and Sod
evidence, is documented in [README.md](README.md). Rendering uses saved results
and does not advance simulations. The HTML embeds its runtime, data and evidence
downloads for offline use; check its navigation, selectors, plots, equations,
themes and downloads after regenerating it.

Actual-EOS state-pair and flow checks are separate from this perfect-gas
analytical matrix. The completed CO₂ evidence includes 6,000 matrix pairs and
two converged five-cell flows; the two 25-cell attempts stopped at their wall
budgets. Neither those coarse flows nor the perfect-gas campaign establish
arbitrary-EOS positivity, phase-switch robustness or real-gas grid convergence.
