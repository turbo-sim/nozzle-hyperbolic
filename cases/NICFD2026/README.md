# NICFD 2026 published-solution reproduction

## Convergence diagnostics

The optional diagnostic client observes the production implicit Newton algorithm
with JAX debug callbacks. It records corrections, linear defects, admissibility,
evaporation-pin gate changes and source fingerprints. Settings overrides are
explicit and saved with each probe; the production implementation is unchanged:

```bash
poetry run python -B -m cases.NICFD2026.scripts.diagnose_convergence --help
poetry run python -B -m cases.NICFD2026.scripts.diagnose_convergence --regime subsonic --cells 400 --cfl 2
poetry run python -B -m cases.NICFD2026.scripts.diagnose_convergence --regime subsonic --cells 400 --scan
```

The client uses the flashing-water family and explicitly starts from the published
reference profile. `--scan` compares existing implicit steps at fixed CFL values;
`--solve-best` additionally attempts a steady solve from that same initial state,
with `--limit-s` controlling its diagnostic wall-time limit. Records go to
`single_component/outputs/convergence_diagnosis/`. Compilation and debug callbacks
affect elapsed times, so these runs are diagnostics rather than performance
benchmarks or independent cold-start validation. Actual convergence and regime
checks remain in the saved outcomes. Only existing settings are exposed:
`--iterations`, `--backtracks`, `--reconstruction`, `--solve-cfl`, `--growth`
and `--cfl-max`. No Newton correction-size cap is added to the solver.

The 400-cell subsonic probes distinguish a successful implicit step from a
steady CFD solution. Starting at CFL 0.02 or 0.2 makes the first step converge,
while CFL 2 and the tested larger values fail their first Newton solve.
Starting at 0.02 with growth 1.2, with slower growth 1.02 and maximum 0.2,
or with fixed maximum 0.02 and 80 inner iterations still ends in stagnation.
The original spatial residual remains large as the controller reduces the
timestep enough for zero-correction inner solves. Lower CFL alone has not
resolved the failure. These observations concern this published-profile-seeded
subsonic probe, not a complete sensitivity campaign for all three regimes.
The first-order trial with 80 Newton corrections, 40 backtracks, CFL 0.02,
growth 1.02 and maximum 0.2 stops at its 300 s diagnostic solve limit after
21 accepted steps. Its maximum equation norm decreases from 0.03168 to
0.02113, but repeated retries reduce the CFL and it does not pass the
unchanged steady checks. It is not used to initialize a MUSCL continuation.

The first failed CFL-2 Newton trace has no evaporation-pin gate switches.
Its largest relative linear defect is below $4\times10^{-12}$; central finite
differences agree with the automatic derivative at sufficiently small steps.
This points to strong local nonlinearity rather than an inaccurate linear
solve, without proving the cause of every subsequent failure. Saved probe
summaries appear under **Convergence & work** in the corresponding HTML report;
the benchmark timings and opening verdict remain based on the original campaign.

For example, this uses only supported reconstruction and nonlinear/controller
settings, with unchanged final checks:

```bash
poetry run python -B -m cases.NICFD2026.scripts.diagnose_convergence --regime subsonic --cells 400 --solve-cfl 0.02 --growth 1.02 --cfl-max 0.2 --iterations 80 --backtracks 40 --reconstruction constant --max-steps 3000 --limit-s 300 --label first_order_conservative
```

## Published-case reports

Open [REPORT.html](REPORT.html) for the six-case index. Each standalone case
report begins with an overall statement about agreement, then provides old/new
profiles, absolute differences, convergence, actual nonlinear work, timings,
property checks and a comparison of the implementations. Each case report can
be copied alone and opened offline. The index links to those six files.

| Family | Subsonic | Supersonic with shock | Supersonic adapted |
|---|---|---|---|
| Flashing water | [Report](single_component/subsonic/REPORT.html) | [Report](single_component/supersonic_shock/REPORT.html) | [Report](single_component/supersonic_adapted/REPORT.html) |
| Nitrogen–water | [Report](two_component/subsonic/REPORT.html) | [Report](two_component/supersonic_shock/REPORT.html) | [Report](two_component/supersonic_adapted/REPORT.html) |

All three nitrogen–water cases converge on the intended regime and match the
published profiles within the declared criteria. The flashing-water cases use
a validated liquid-table configuration that removes their property initialization
failure without changing JAXProp. Their reports distinguish property validity,
CFD convergence, flow regime and agreement with the published solution.
The fresh water runs still stop without passing the original spatial-residual
checks. They are reported as unconverged; valid lookup tables alone do not
establish reproduction of the published CFD solutions.

The liquid table starts at **250 kJ/kg**, replacing the published 100 kJ/kg
lower bound. This avoids the generator's false spinodal-ratio trigger in cold
water, where the isothermal bulk modulus initially increases with enthalpy.
Every published liquid state is above 534.6 kJ/kg. The unchanged JAXProp API
generates a separate named cache with the same HEOS, metastable phase, pressure
range, upper enthalpy bound, 90 × 90 resolution and forward gradients. All 1,800
published liquid states pass the unchanged stencil checks; their maximum
temperature difference is below 0.000005 K. This changes the interpolation grid
and is declared in [protocol.json](protocol.json); the published YAML files remain
unchanged. It does not establish CFD convergence by itself.

The [supplementary repository](https://github.com/alber-berg/twophase-nozzle-nicfd2026/tree/2dc04ad9d474afb727857139df89537fcadb5769/data)
provides the unchanged `settings.yaml` and `reference.csv` in each case folder.
Upstream `phase_change/` is named `single_component/` here. SHA-256 hashes,
original paths and Git blob identities are in [provenance.json](provenance.json).
The data are CC BY 4.0; see [LICENSE_UPSTREAM.txt](LICENSE_UPSTREAM.txt).

Only published solutions are used as the old-code reference. Upstream has no
executable solver, timing logs or convergence histories. Old runtimes and speedup
are therefore unavailable. Fresh timings describe the current solver only.

The physical mapping preserves the published geometries, HEOS phase models,
source parameters, stabilizations and phase-pin selection. Two setup rules
missing from the YAML are resolved against publication-era source revision
`e185f571f8ea0512a7a16ce53a4b2458f72c33c2`: the original number-density area law
and the saturated ghost-vapor reservoir when equilibrium PT enthalpy is below
the metastable vapor table. The shocked nitrogen YAML omits gas `N_h`; the
explicit 145 × 200 within-family table from its subsonic settings is reused.
Both effective settings and numerical differences are embedded in the reports.
The historical project used an editable JAXProp checkout labelled 0.5.6, which
does not identify an immutable dependency revision. Alberto confirms his source
is current; reuse of an earlier generated cache remains a plausible explanation
for the difference from fresh generation. The configuration fix here uses the
installed package without modifying it or relaxing physical checks.

[protocol.json](protocol.json) declares the meshes, work budget and agreement
thresholds before the current campaign. Each case runs in a fresh process with
the persistent JAX compilation cache disabled. A compilation-inclusive hierarchy
starts at 100 cells from the published profile, then transfers only successful
states to 200 and 400 cells. The replay resets the initial state and controller
while retaining the same bindings. If a coarse stage fails, its state is kept
for inspection and an independent published-profile 400-cell attempt follows;
the unfinished state is never used as a restart. Reports include failed work.

Convergence requires the original selected spatial residual, local EOS,
boundary, global-balance and total-mass checks. The intended regime is checked
separately before profile agreement. Saturation-energy pins intentionally
change selected energy equations; the unmodified physical defects are also
saved and do not inherit a convergence claim from pinned rows. A profile close
to its published seed is insufficient evidence of reproduction.

Property preparation, output and final audits are excluded from solve timings.
Compilation events and cold/replay state agreement qualify the replay. These
are individual timing observations, not repeated performance distributions.
They do not establish independent cold-start recovery or grid independence.

From the repository root, using the existing Poetry environment:

```powershell
poetry run python -B -m cases.NICFD2026.scripts.run
poetry run python -B -m cases.NICFD2026.scripts.audit
poetry run python -B -m cases.NICFD2026.scripts.report
poetry run python -B -m cases.NICFD2026.scripts.audit_browser
```

The runner refuses to overwrite a completed case campaign. Generated states,
histories and logs live in each case's `outputs/`; property caches are shared
within each family. The report generator needs saved results and property
audit JSON but does not invoke CFD or EOS. Numerical, evidence and browser
audits are separate so presentation changes never rerun the simulations.
Passing evidence and browser audits verifies data/report consistency; successful
CFD convergence and profile agreement are assessed separately for each case.
There are no retained historical campaign folders or legacy clients.
