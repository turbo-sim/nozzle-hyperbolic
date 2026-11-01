# Shared steady verification and benchmark tooling

This client uses the production FVM engine and installed physical YAML cases.
It records original-equation convergence, physical branch, analytical accuracy
and complete-solve timing separately. It contains no copy of the production solver.

The final [Roe study](../../roe_method/README.md) supplies its implemented theory,
160-configuration verification evidence, offline report and four-flux protocol.
The complete Roe/Rusanov/HLL/characteristic-FVS performance campaign is prepared;
no completed four-flux timing ranking is claimed.

## Physical and numerical inputs

Physical geometry, EOS, boundaries, closures and accuracy targets are versioned
beside the nozzle/annular models. Load them through
[`nozzle_hyperbolic.verification`](../../../src/nozzle_hyperbolic/verification/README.md).
The catalog has eight nozzle and twelve annular inviscid perfect-gas reference
cases. Numerical meshes, reconstructions, fluxes, Newton/CFL controls, tolerances
and repetitions belong to profiles such as [`settings.yaml`](settings.yaml) and
[the Roe comparison profile](../../roe_method/benchmark_profile.yaml).

Schema 2 adds a flux dimension to configuration, worker and resume identities;
schema 1 native/current comparisons remain supported. The native adapter loads
only the tracked frozen `legacy/pre_generic_fvm/` snapshot. Incompatible native
flux sweeps are rejected. Physical cases and analytical references are reused,
not duplicated into benchmark scripts.

## Run, audit and render

Run from the repository root in the existing Poetry environment. These commands
perform numerical work and require a new output name after source or settings
change:

```powershell
poetry run python -B -u -m development.roberto.benchmarking.scripts.run --config development/roe_method/benchmark_profile.yaml --output outputs/roe_flux_v1 --wait-for-idle
poetry run python -B -m development.roberto.benchmarking.scripts.audit --input outputs/roe_flux_v1/summary.json
poetry run python -B -m development.roberto.benchmarking.scripts.report --input development/roberto/benchmarking/outputs/roe_flux_v1/summary.json --output development/roe_method/REPORT.html
```

Use `--resume` only for identical recorded source/cases/profile/harness/machine
identities. The auditor reads saved evidence without advancing the PDE. Its
integrity result is separate from numerical/analytical acceptance; add
`--require-numerical-pass` when failed cases must cause a failing exit status.

Every timing configuration starts cold, with one excluded complete solve before
measured repetitions. The Roe profile uses one measured solve on coarse meshes
and ten at 400 cells, sequential rotating method order, no grid sequencing, and
an idle-environment gate. All failed/rejected work remains in the record.
Incomplete, contended, one-sample or verification-only populations cannot support
repeated timing rankings. See the [full protocol](../../roe_method/benchmark_protocol.md)
for counts, matched controls and acceptance definitions.

The renderer accepts optional `--baseline`, `--roe-audit`, `--roe-sod`,
`--roe-real-gas` and `--roe-sod-figure` saved evidence. It imports no flow solver
and creates one standalone offline HTML with summary, case explorer, protocol
and native-MathML Roe theory. Plotly, data, original evidence downloads and
licenses are embedded; no CDN or local server is needed.

Check a report with:

```powershell
poetry run python -B -m development.roberto.benchmarking.scripts.audit_report --input development/roe_method/REPORT.html
poetry run python -B -m development.roberto.benchmarking.scripts.check_flux_sweep
```

The second command exercises campaign dispatch and reporting with explicitly
synthetic data; it performs no physical solve. The report auditor can prepare an
instrumented offline browser check using `--prepare-browser PATH`; inspect the
actual browser runtime result as well as the static structure audit.

## Modules and generated artifacts

- `run.py`, `worker.py`, `legacy_adapter.py`: campaign, isolated workers and
  frozen-native compatibility.
- `common.py`, `environment.py`: identities, serialization and timing environment.
- `audit.py`, `check_flux_sweep.py`: evidence, population and dispatch checks.
- `report.py`, `report_assets/`, `audit_report.py`: offline report and browser audit.
- `snapshot.py`, `baseline.py`: explicit source/case preservation and baseline export.

Generated numerical data, logs and browser checks live under ignored
`outputs/<campaign>/`; human-facing reports live at the study root. Historical
native/current campaigns are separate evidence for their own recorded versions.
The final Roe report embeds its own retained numerical evidence and provides a
restore client, so that report can be rebuilt after cloning without new solves.
