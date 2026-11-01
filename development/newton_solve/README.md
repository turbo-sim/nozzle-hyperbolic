# Direct Newton and implicit Euler for shocked Euler nozzle flow

This study tests whether the current solver can reach the shocked perfect-gas
nozzle solution directly with PySolverView Newton, and compares it with implicit
Euler from exactly the same initial field. The two target shocks lie at 35% and
70% of the divergent section. The interactive, standalone [report](REPORT.html)
separates convergence of the discrete equations, the physical shock branch and
agreement with the continuous analytical reference.

See [RESULTS.md](RESULTS.md) for the measured outcomes and their interpretation.

The implementation and results live in `development/newton_solve`. The previous
study in `development/roberto/historical/newton_solve` is retained untouched.
This study does not modify `src/` or use a historical solver adapter.

## Controlled comparison

| Setting | Coverage |
|---|---|
| Physical targets | Packaged `euler_nozzle/shock_35` and `euler_nozzle/shock_70` |
| Fluxes | Rusanov, HLL, Stadtke with jump correction and endpoint envelope enabled |
| Reconstruction | Constant; linear MUSCL with Van Leer limiter |
| Outlet equations | Acoustic; finite wave |
| Initial fields | Uniform reservoir state at rest; converged numerical adapted expansion |
| Target mesh | 200 physical cells, plus two algebraic endpoint traces |
| Solution strategies | Implicit Euler; unshifted custom PySolverView Newton |
| Baseline attempts | 96: 48 matched configurations, two strategies each |
| Precision | JAX float64, CPU |

Both cases use the packaged symmetric nozzle: length 0.2 m, inlet/exit area
$3\times10^{-4}$ m², throat area $1.5\times10^{-4}$ m², reservoir pressure
200 kPa and temperature 300 K. The throat is at 0.1 m; the target shocks are at
0.135 and 0.170 m. The analytical solution supplies each backpressure and the
verification profiles; it is never used as an initial field.

| Target | Analytical shock $x_s$ (m) | Analytical backpressure (Pa) |
|---|---:|---:|
| `nozzle_shock35` | 0.135 | 177667.62651413813 |
| `nozzle_shock70` | 0.170 | 141744.48279960960 |

The stagnant start sets every cell and both traces to reservoir pressure and
enthalpy, with velocity exactly zero. The adapted start is a separate numerical
implicit-Euler solution of `euler_nozzle/adapted`. Six such seeds are prepared,
one for each flux/reconstruction pair, with 50 → 100 → 200 cell grid sequencing.
Each 200-cell seed is reused without interpolation or outlet adjustment for
both target shocks, both outlet formulations and both solution strategies.
The target solves themselves start directly on 200 cells: direct Newton has no
implicit-Euler prefix or shocked coarse-grid warm start.

The two strategies receive bit-identical full initial arrays. Direct Newton
also records a hash of its first actual residual evaluation to check this
property after coordinate scaling. A match is necessary for the comparison;
it does not imply that both strategies follow the same trajectory.

## Numerical policy

[settings.json](settings.json) is the executable policy. Implicit Euler uses
coupled algebraic boundary traces, a colored Jacobian, a banded linear solve,
up to ten Newton corrections per timestep and backtracking. CFL starts at 0.2,
grows by 1.2 up to $10^6$, and shrinks by 0.5 on rejected steps. There is no
conservative filter. The timestep cap is 16,000 accepted steps.

Direct Newton uses installed PySolverView's custom `pysolver/newton_raphson`
backend, an exact dense JAX Jacobian, strict full-rank Newton directions and
Armijo backtracking. Its first step fraction is 0.25, growing by 1.5 toward
one. The main campaign permits **400 accepted direct-Newton updates**. The
initial evaluation is update zero; convergence at update 400 is allowed. A
run that has not converged at that point is failed and is not extended.
Rejected line-search trials are counted separately, with secondary safeguards
of 10,000 total trials and 20,000 residual evaluations. It solves the original coupled
steady equations without a storage shift.
There is no velocity perturbation, residual clipping, analytical seed or
fallback nonlinear method. All main runs use the same initial step policy.

Both strategies use the same fixed physical scales and volume-weighted
per-equation spatial RMS policy: relative reduction $10^{-6}$ **or** absolute
norm $10^{-9}$ for every equation, with independent boundary and global
conservation limits $10^{-9}$. Initially zero equations use the absolute limit.
Final verification evaluates the original spatial and boundary equations afresh.
A native optimizer flag alone cannot establish convergence. PySolverView can
return native status 5 when the common stopping callback has found a verified
root; that status and the physical checks are retained separately.

Each attempt has a 600-second observation budget, checked at solver callbacks;
an active compilation, Jacobian or line search can finish before the next check.
Failed attempts and their last states are retained. The report shows accepted
updates and rejected trials separately. The [theory](THEORY.md) documents the
equations, outlet laws, scaling and interpretation.

## Reproduce and inspect

Run from the repository root using the existing environment:

```powershell
.venv\Scripts\python.exe -B -u -m development.newton_solve.scripts.run
.venv\Scripts\python.exe -B -m development.newton_solve.scripts.audit_campaign
.venv\Scripts\python.exe -B -m development.newton_solve.scripts.verify_saved
.venv\Scripts\python.exe -B -m development.newton_solve.scripts.audit_direct --full --output development/newton_solve/outputs/audit_direct_full.json
```

The runner preserves every saved attempt, including failures. A selected
configuration can be run with:

```powershell
.venv\Scripts\python.exe -B -u -m development.newton_solve.scripts.run --case nozzle_shock35 --outlet acoustic --flux hll --reconstruction constant --start stagnant --jobs 1 --no-report
```

Rebuild the HTML from saved evidence, without advancing a flow solution:

```powershell
.venv\Scripts\python.exe -B -m development.newton_solve.scripts.run --assemble-only
.venv\Scripts\python.exe -B -m development.newton_solve.scripts.audit_report --help
```

`outputs/provenance.json` records the actual working-tree source, numerical
clients, settings, dependency versions and Git revision. Copies of the executed
files live in `outputs/source/`. Source and client hashes must still match on
resume; changed numerical inputs require a separate campaign. Root states,
initial fields, histories, unsuccessful trials, audits and analytical comparison
data are saved under `outputs/`; target records use lossless `.json.gz`
compression. The campaign index references these full records. The standalone
report embeds initial/final fields and scalar histories; native coordinate and
residual vectors for every backend event remain in the full disk records.
Report presentation files can be updated independently of frozen numerical inputs.

Failed and unfinished runs retain their convergence histories and final
accepted flow profiles in the report. Their stopping reasons and residual
levels remain visible alongside successful solutions.

Durations describe single trajectories, including compilation and monitoring.
They are diagnostic, not a speed ranking: direct Newton uses dense PySolverView
linear algebra, implicit Euler uses the production colored/banded path, and up
to two independent workers share the CPU. Adapted-seed preparation is reported
separately and is part of the cost of an adapted-start workflow.
