# NICFD 2026 published-solution reproduction

Open [REPORT.html](REPORT.html) for the six-case index. Each standalone case
report begins with an overall statement about agreement, then provides old/new
profiles, absolute differences, convergence, actual nonlinear work, timings,
property checks and a comparison of the implementations. Each case report can
be copied alone and opened offline. The index links to those six files.

| Family | Subsonic | Supersonic with shock | Supersonic adapted |
|---|---|---|---|
| Flashing water | [Report](single_component/subsonic/REPORT.html) | [Report](single_component/supersonic_shock/REPORT.html) | [Report](single_component/supersonic_adapted/REPORT.html) |
| Nitrogen–water | [Report](two_component/subsonic/REPORT.html) | [Report](two_component/supersonic_shock/REPORT.html) | [Report](two_component/supersonic_adapted/REPORT.html) |

In the saved campaign, all three nitrogen–water cases converge on the intended
regime and match the published profiles within the declared criteria. All three
flashing-water cases are rejected before a CFD solve: the installed JAXProp
metastable liquid-table generator freezes liquid properties near room
temperature across the required hot-liquid enthalpies. The model rejects those
copied stencils. Their reports show the published profiles and the property
diagnosis; no current solution or old timing is invented. Repairing and
independently validating that dependency is needed before a valid flashing
reproduction is possible.

The [supplementary repository](https://github.com/alber-berg/twophase-nozzle-nicfd2026/tree/2dc04ad9d474afb727857139df89537fcadb5769/data)
provides the unchanged `settings.yaml` and `reference.csv` in each case folder.
Upstream `phase_change/` is named `single_component/` here. SHA-256 hashes,
original paths and Git blob identities are in [provenance.json](provenance.json).
The data are CC BY 4.0; see [LICENSE_UPSTREAM.txt](LICENSE_UPSTREAM.txt).

Only published solutions are used as the old-code reference. Upstream has no
executable solver, timing logs or convergence histories. Old runtimes and speedup
are therefore unavailable. Fresh timings describe the current solver only.

The physical mapping preserves the published geometries, EOS specifications,
source parameters, stabilizations and phase-pin selection. Two setup rules
missing from the YAML are resolved against publication-era source revision
`e185f571f8ea0512a7a16ce53a4b2458f72c33c2`: the original number-density area law
and the saturated ghost-vapor reservoir when equilibrium PT enthalpy is below
the metastable vapor table. The shocked nitrogen YAML omits gas `N_h`; the
explicit 145 × 200 within-family table from its subsonic settings is reused.
Both effective settings and numerical differences are embedded in the reports.

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
There are no retained historical campaign folders or legacy clients.
