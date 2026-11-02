# Roberto: verification and numerical studies

The maintained steady verification and performance suite is
[`benchmarking/`](benchmarking/README.md). It uses versioned physical YAML cases
packaged beside the nozzle and annular models in `src/nozzle_hyperbolic/`.
The reusable `nozzle_hyperbolic.verification` API builds cold physical problems
and independent analytical reference projections. Numerical methods, meshes,
timing repetitions and controller settings belong to the benchmark profile.

| Study | Purpose |
|---|---|
| [`local_timestep/`](local_timestep/README.md) | Experimental HLL timing comparison: limited/inconsistent gains; local stepping is not recommended as a production default |
| [`benchmarking/`](benchmarking/README.md) | Twenty steady nozzle/annular cases, constant/MUSCL, independent grid refinement, analytical errors and current-code timings |
| [`../benchmark_legacy/`](../benchmark_legacy/README.md) | Six representative shock/supersonic cases comparing original Stadtke in current and frozen native code |
| [`../flux_methods/`](../flux_methods/README.md) | Production Stadtke corrections, Osher path rules and general-EOS Roe theory, verification and offline reports |
| [`transient_problems/`](transient_problems/README.md) | Sod and nozzle pressure release: explicit Euler, SSPRK3, coupled and segregated backward Euler, conservation budgets, and offline animations |
| [`numerical_fixes/`](numerical_fixes/README.md) | Retained projector, sonic regularization, and limiter investigations; inspect imports before using historical clients |
| [`historical/`](historical/README.md) | Earlier nozzle, annular, backpressure, boundary residual, Newton, implicit integration, and grid-sequencing studies |

The actual pre-integration production source is preserved at the repository
root in [`legacy/pre_generic_fvm/`](../../legacy/pre_generic_fvm/README.md).
Its manifest identifies the frozen native baseline used by the benchmark;
keep adapters outside that snapshot. Current production code lives in `src/`.

Run the maintained steady suite from the repository root:

```powershell
.venv/Scripts/python.exe -B -m development.roberto.benchmarking.scripts.run --output outputs/revision_2 --wait-for-idle
.venv/Scripts/python.exe -B -m development.roberto.benchmarking.scripts.audit --input outputs/revision_2/summary.json
.venv/Scripts/python.exe -B -m development.roberto.benchmarking.scripts.snapshot --input outputs/revision_2/summary.json --attach
.venv/Scripts/python.exe -B -m development.roberto.benchmarking.scripts.report --input development/roberto/benchmarking/outputs/revision_2/summary.json --baseline development/roberto/benchmarking/baselines/initial_20261101.json
```

The benchmark README describes resume, subset, baseline-comparison and report
options. Reports live at each study's root; generated numerical data and logs
remain under `outputs/`. Saved historical evidence keeps its measurement-time
paths and source fingerprints. Older clients may require retired solver APIs;
their reports describe their recorded versions.

Use the [architecture guide](../../docs/architecture/ARCHITECTURE.md) for current
package responsibilities and the [source verification guide](../../src/nozzle_hyperbolic/verification/README.md)
for physical case definitions and analytical assumptions.
