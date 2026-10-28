# Speed-up of the single-phase FVS solver: report

2026-10-07.  Scope: the single-phase / HEM Städtke FVS Newton solver,
`make_jax_stadtke_step` and the `fvs_newton` loop of `fvm_euler_1d` in
`nozzle_hyperbolic/fvm_solver/fvm.py`, driven by `dev_scripts/main.py`.
The two-fluid solver (`two_fluid/`, `main_two_fluid.py`) is not touched; its
report is `SPEEDUP_REPORT.md`.

**Status:** local working tree, not committed.  Every change is a YAML switch
in `numerical_settings`.  **With the keys absent the code is bit-for-bit the
git HEAD version** (verified, §6).  The five existing `fvs_newton` settings
files now turn ON the switches that change results only at round-off; the SER
ramp stays OFF there (`ser_beta: 0.0`).

---

## 1. Headline numbers

HEM water flashing nozzle (`settings_nozzle_hem.yaml`: SMD geometry, MUSCL van
Leer, colored Jacobian, `newton_iter` 8, CFL 0.5, `eps_filter` 0.05), P-cores:

| Run | Steps | Wall | vs HEAD |
|---|---|---|---|
| N=500, git HEAD (3.83 s/step) | 15,000–20,000 | **~16–21 h** (extrapolated) | — |
| N=500, exact switches (§3.1–3.2) | 19,974 | **325 s** | ~200× |
| N=500, + SER ramp (§3.3), converged to SSR 3e-11 | 2,566 | **66 s** | |
| N=100, git HEAD (67 ms/step) | 19,317 | **~22 min** | — |
| N=100, exact switches | 19,317 | **84 s** | ~15× |
| N=100, + SER ramp (same stop criterion) | 345 | **2.6 s** | |
| N=100, + SER ramp, converged to SSR 3e-11 | 458 | **2.9 s** | |

The N=500 runs without SER stop on the plateau test at SSR 3e-6 after a long
limit-cycle phase whose end is round-off sensitive: 14,957 steps in the
prototype, 19,974 in the ported code (they differ at ~1e-10 per step through the
`newton_tol` scaling).  HEAD (3.83 s/step measured through the same driver;
not run to the end) would need a similar 15,000–20,000 steps.  With SER the
count is robust.

Per iteration on the other `fvs_newton` cases (first 40–60 steps, HEAD vs new,
same driver):

| Settings file | N | HEAD | New | Gain |
|---|---|---|---|---|
| `settings_subsonic_pipe.yaml` (`jacobian` absent = dense AD) | 150 | 5,180 ms | 10.9 ms | 475× |
| `settings.yaml` (Lettieri, CO2) | 200 | 1,273 ms | 15.5 ms | 82× |
| `settings_nozzle_hem.yaml` | 100 | 74 ms | 10.4 ms | 7× |
| `settings_nozzle_validation.yaml` (Air), full run, 484 steps | 100 | 21.3 s | 1.6 s | 13× |

---

## 2. Where the time went (profile of git HEAD)

The bottleneck is **not** the one of the two-fluid solver.

| Piece (HEM) | N=100 | N=500 |
|---|---|---|
| Colored Jacobian (15 JVP seeds + dense assembly) | 1.9 ms | 6.6 ms |
| Residual | 0.4 ms | 1.0 ms |
| **Dense `jnp.linalg.solve`** on the 3(N−2) system | ~7 ms in the step | **~500 ms** |
| Step (8 Newton iterations) | 65–75 ms | **3,830 ms** |

- `jacobian: colored` only made the Jacobian *computation* O(N).  `_jac_colored`
  then de-interleaved it into a full dense (3(N−2))² matrix, and the Newton
  update was a dense LU, O(N³): 294×294 at N=100, 1494×1494 at N=500 (~99 % of
  the step).  The two-fluid solver already had banded solvers (E1b, E3-banded,
  2026-07-16); they were never ported here.
- Without a `jacobian` key the default is `dense` = full reverse-mode AD
  Jacobian: 5.2 s per iteration at N=150 (`settings_subsonic_pipe.yaml`).
- At CFL 0.5 Newton reaches round-off in 5–6 iterations in transients and in
  1–3 near the steady state; `newton_iter` 8 always ran 8.
- The driver's increment check (dp, dh, du) ran as ~20 eager JAX ops:
  0.8 ms per step — negligible at 70 ms/step, ~15 % of a 5 ms step.
- Step count: at CFL 0.5 the HEM run spends ~10,000 of its 19,317 steps
  limit-cycling at SSR ~1e-5 (N=100).

---

## 3. Changes

### 3.1 Solver — faster steps, same result (`numerical_settings`)

| YAML key (default → recommended) | What it does | Gain |
|---|---|---|
| `linear_solver: dense → banded` | 3×3 blocks read straight from the colored JVP columns (no dense matrix); cells grouped in super-blocks of `stencil width` cells (6×6 for MUSCL) → block-tridiagonal; block Thomas in a `lax.scan`, one LU per super-block, back substitution x_i = g_i − G_i·x_{i+1}.  O(N) instead of O(N³).  Uses the colored compression whatever `jacobian` says. | step 75 → 12.5 ms (N=100), **3,830 → 54 ms (N=500)** |
| `newton_tol: 0.0 → 1.0e-8` | `lax.while_loop`: stop when max\|ΔU/U_ref\| < tol (U_ref: \|p\| per cell, max\|u\|+1 m/s, max\|h\|); `newton_iter` stays the cap | 8 → 5–6 iterations (transient), 1–3 (near steady) |
| `flux_local_jac: false → true` | `jax.custom_jvp` on the face flux: each face linearized once (exact 3×6 block), every colored seed becomes a mat-vec | Jacobian −20 % |
| `cell_local_jac: false → true` | same for the per-cell EOS outputs (ρ, a, μ; 3×3 block) | a few % |

Two-fluid switches with no counterpart here: `fast_wb_flux` (no `wb`
non-conservative term) and `flux_single_branch` (no `lax.cond` in the
single-phase flux).  The local-Jacobian switches gain far less than in the
two-fluid solver (75 → 24 ms there) because the single-phase flux is cheap (3
EOS calls, 3×3 solves, no SVD) and needs 15 seeds instead of 40.

The banded LU pivots only inside each super-block (no pivoting across blocks).
It matched the dense LU to ~1e-12 on every case tested, including the SER runs
at CFL 50.  It relies on the same sparsity as the colored Jacobian, so
`jacobian_validate: true` remains the guard.

### 3.2 Driver (`fvm_euler_1d`, `main.py`)

| YAML key | What it does | Gain |
|---|---|---|
| `fast_increments: false → true` | jitted dp/dh/du check (same formulas) | 0.8 → 0.07 ms per step |
| `cpu_affinity: null → "0-15"` | `main.py` pins the process to the P-cores (`dev_scripts/cpu_affinity.py`) | up to ~2×, depending on scheduling |

### 3.3 Fewer steps — same steady equations, different path

| YAML key | What it does |
|---|---|
| `ser_beta: 0.0 → 1.0`, `ser_cfl_max: 50` | SER / pseudo-transient continuation, as in the two-fluid driver: CFL_eff = CFL·clip((SSR₁/SSR)^β, 1, cfl_max/CFL), SSR₁ = SSR after the first step.  dt still grows at most 1.2× per step and the increment checks / halvings remain.  Steady cases only (it gives up time accuracy). |

`ser_cfl_max: 500` was too aggressive on HEM N=500: the run reached CFL 500 at
SSR ~5e-7, kicked back to 1e-3 and kept cycling (8,000 steps, not converged).
50 converged cleanly on every case.

---

## 4. Recommended settings

```yaml
numerical_settings:
  jacobian: colored
  linear_solver: banded
  newton_tol: 1.0e-8
  flux_local_jac: true
  cell_local_jac: true
  fast_increments: true
  ser_beta: 1.0          # steady cases; omit (0.0) to reproduce old runs / transients
  ser_cfl_max: 50.0
  cpu_affinity: "0-15"
```

Everything except `ser_beta` is already in the five existing `fvs_newton`
settings files.  Add `ser_beta: 1.0` for new steady studies; also lower
`residuals` (§7.1).

---

## 5. Results

Step time at the same state and dt (HEM: after 3 marched steps; result vs
HEAD ≤ 1.4e-14 for every row):

| Switches | HEM N=100 | HEM N=500 | Air nozzle N=100 |
|---|---|---|---|
| git HEAD (`linear_solver: dense`, 8 / 8 / 5 Newton its) | 74.8 ms | 3,833 ms | 50.9 ms |
| `linear_solver: banded` | — | 53.8 ms (71×) | 8.3 ms (6.1×) |
| banded + `newton_tol` | — | 31.1 ms (123×) | 5.7 ms (9.0×) |
| `flux_local_jac` + `cell_local_jac` | 67.3 ms (1.1×) | 3,957 ms (1.0×) | 36.1 ms (1.4×) |
| local Jacobians + banded | 12.5 ms (6.0×) | 49.7 ms (77×) | 6.7 ms (7.6×) |
| all four (Newton its at this state) | 8.4 ms (8.9×, 6 its) | 25.9 ms (148×, 5 its) | 5.1 ms (10×, 3 its) |

Pieces (HEM): colored Jacobian 1.9 → 1.3 ms (N=100), 6.6 → 5.1 ms (N=500);
banded solve 0.25 ms (N=100), 1.1 ms (N=500); LAPACK `gbsv` through
`pure_callback` was tried as well (0.8 / 1.25 ms) and is slower than the JAX
block Thomas.  Near the steady state `newton_tol` needs 1 Newton iteration:
2.7 ms/step at N=100.

Whole runs (HEM, real driver, ported code):

| Run | Steps | SSR at stop | Wall | ms/step |
|---|---|---|---|---|
| N=100, exact switches + `newton_tol` | 19,317 | 3.0e-6 (plateau) | 84 s | 4.3 |
| N=100, + SER | 345 | 1.4e-6 (plateau) | 2.6 s | 7.4 |
| N=100, + SER, `residuals: 1e-11` | 458 | 2.8e-11 | 2.9 s | 6.4 |
| N=500, exact switches + `newton_tol` | 19,974 | 3.0e-6 (plateau) | 325 s | 16.3 |
| N=500, + SER, `residuals: 1e-11` | 2,566 | 3.0e-11 | 66 s | 25.9 |
| N=500, git HEAD, 10 steps | — | — | 38 s | 3,831 |

The steps with SER cost more on average because Newton needs more iterations
at CFL 50 than near-steady at CFL 0.5.

---

## 6. Verification

- **Defaults vs git HEAD** (HEAD `fvm.py` loaded side by side, same driver, 40–60
  steps): final state bit-identical (`np.array_equal`) on
  `settings_nozzle_hem.yaml` (N=100), `settings_nozzle_validation.yaml`,
  `settings_subsonic_pipe.yaml` and `settings.yaml`.
- **Switches on vs HEAD** (same runs): max relative difference 7.6e-13,
  2.3e-14, 3.7e-11, 5.4e-13.  Full Air validation run: same 484 steps, final
  state 1.6e-14.
- **Per piece:** local Jacobians vs colored 6e-16 (residual identical); banded
  solve vs dense LU 1.6e-12 (N=500) – 3.8e-12 (N=100).
- **Whole run, HEM N=100:** the SSR history of the first 300 steps matches HEAD
  to 1e-11 (exact switches) and 4e-12 (+ `newton_tol`); both whole runs take
  19,317 steps and end within 4e-10 (range-normalised) of each other.  HEAD
  itself was not run to the end (~22 min).
- **SER steady state:** the SER-converged HEM state (SSR 2.8e-11) marched for
  6,000 steps at plain CFL 0.5 keeps falling to SSR 6e-14 — a genuine, stable
  steady state of the same discrete equations.
- **Analytical validation:** normal shock in the Lettieri nozzle, §8.

---

## 7. Findings worth knowing

### 7.1 The HEM plateau stop is not converged

At CFL 0.5 the HEM run limit-cycles at SSR 1e-5 – 2e-5 and stops on the plateau
test at SSR ~3e-6 (`residuals: 1e-6` → stop at 3e-6).  That state is
1.7e-3 (N=100) and **6e-3 (N=500, at x ≈ 0.45 m, the thermal throat)** of the
p range away from the steady state that SER reaches at SSR 3e-11.  Use SER and
`residuals` ≤ 1e-9 for anything quantitative near the thermal throat.

### 7.2 Pre-existing issues seen during the tests (not caused by these changes)

- `settings_shock_tube.yaml` fails in the initial condition with HEAD code too:
  the PT root-find (`get_state(PT_INPUTS, p_L, T_L)`, T_L = 400 K) does not
  converge on the Air 100×60 table.
- `settings_nozzle_validation.yaml` (Air, symmetric nozzle) ends with the outlet
  pressure at 5,050 Pa = 1.01·`p_min` of the table (clipped) and a 2 % face
  mass-flux blip at x ≈ 0.30 m — identical in HEAD.
- `get_lettieri_nozzle` reads `Lettieri_nozzle.xlsx` from the cwd; the file was
  missing from this repo (copied from `../branch/nozzle-hyperbolic-andrea`).

### 7.3 Caveats

- `newton_tol > 0` uses `lax.while_loop`, which JAX cannot reverse-differentiate
  (forward mode and the implicit-function-theorem adjoint are unaffected).
- Do not use `ser_beta > 0` for transients (shock tube, `t2`-limited runs).

---

## 8. Validation: normal shock in the Lettieri nozzle (Air LUT)

`dev_scripts/validation_fvs_shock_nozzle.py` with
`settings/settings_fvs_validation_lettieri_air.yaml`:

- Lettieri nozzle (A_in/A* = 17.1, A_exit/A* = 1.69, throat at x/L = 0.325),
  Air from the bicubic LUT (100×80), p0 = 2 bar, T0 = 295.15 K, N = 300.
- Shocks at 25, 50, 75 % of the divergent-section length.  The back pressure
  for each comes from the exact quasi-1D solution computed with the same EOS
  (`nozzle_hyperbolic/fvm_solver/nozzle_exact_real_gas.py`: real-gas
  isentropes + Rankine–Hugoniot; within ~5e-4 of the perfect-gas area–Mach and
  normal-shock relations for this air).
- Start: choked, fully subsonic initial condition (`initialization: subsonic`);
  the shock forms and travels to its position — it is not placed by the IC.
- All speed-up switches on, SER β = 1.

Run `results/fvs_validation_lettieri_nozzle_Air_N300_20261007_130934`
(dx = 0.329 mm):

| Shock at | p_out/p0 | M1 → M2 | x_s exact | x_s numerical | error | p/p0 error mean / max | M error mean / max | Steps | Wall |
|---|---|---|---|---|---|---|---|---|---|
| 25 % | 0.82638 | 1.510 → 0.698 | 16.56 mm | 16.57 mm | +0.03 dx | 1.5e-4 / 4.8e-3 | 2.8e-4 / 7.8e-3 | 2,221 | 31 s |
| 50 % | 0.71927 | 1.750 → 0.628 | 33.16 mm | 33.26 mm | +0.30 dx | 1.5e-4 / 4.8e-3 | 2.9e-4 / 7.8e-3 | 1,613 | 25 s |
| 75 % | 0.61187 | 1.942 → 0.588 | 49.76 mm | 49.77 mm | +0.04 dx | 1.5e-4 / 4.8e-3 | 3.0e-4 / 7.8e-3 | 1,329 | 21 s |

All three converged to SSR ≤ 3e-10 (plus ~6 s compilation and ~15 s for the
exact solution per case).

Errors over interior cells farther than 3 dx from the shock.  The largest
error away from the shock is at the throat cell (sonic point) in all three
cases; elsewhere ≤ 1e-3·p0.  Case 2's shock sits 0.3 dx inside a cell and
shows a one-cell post-shock overshoot; cases 1 and 3 land almost on a face.
Mass flow (median ρuA away from the shock) is +3.5e-4 above the exact value
(discrete throat).

Run it (cwd = `dev_scripts/`):

```
python validation_fvs_shock_nozzle.py [settings/settings_fvs_validation_lettieri_air.yaml]
```

### 8.1 Grid convergence: 3-grid GCI (N = 300 / 500 / 850)

`dev_scripts/gci_fvs_validation.py` (Celik et al. 2008, Fs = 1.25; the GCI
maths is imported from `cases/nicfd/grid_convergence_gci.py`).  h = 0.329
/ 0.197 / 0.116 mm, r32 = 1.669, r21 = 1.701.  The exact solution is known, so
every GCI band is checked against the true error of the fine grid.  Report:
`results/fvs_validation_gci_20261007_144830/`.

Observed order of the true mean error away from the shock (least squares over
the three grids):

| | 25 % | 50 % | 75 % |
|---|---|---|---|
| p/p0 | 1.73 | 1.83 | 1.93 |
| M | 1.92 | 1.99 | 2.06 |

GCI on mdot and on p/p0 and M at x/L = 0.10, 0.25, 0.40, mid post-shock and
x/L = 0.97 (11 QoIs per case; the upstream ones are identical in all cases):

- **mdot:** p_obs = 1.72, GCI_fine = 7.5e-5, true error 5.1e-5 (in the band);
  the Richardson-extrapolated value is within 9e-6 of the exact mass flow.
- **Upstream of the shock and Mach everywhere:** p_obs = 1.86–1.99, every band
  contains the true error, Richardson values within ≤ 1.6e-5 of the exact.
- **Post-shock pressure:** p_obs = 1.0–1.3 (Mach 1.5–1.6): the downstream state
  depends on the post-shock entropy, i.e. on where the shock sits inside its
  cell.  Bands still hold (GCI 0.003–0.005 %).
- **Shock position:** not monotone (50 % case: +0.30, +0.02, +0.82 dx), always
  within one cell — reported, no GCI.
- **The one failure — p/p0 at x/L = 0.10** (R = 0.029, true error 7.5e-6 vs
  GCI 1e-6): from the inlet to x/L ≈ 0.1 (M ≈ 0.03–0.07) N = 500 and 850 sit on
  the same ~8e-6·p0 plateau, and the N = 500 error changes sign at
  x/L ≈ 0.19.  It is not entropy loss (s = s0 to 3e-9 relative); it is a
  ~1.5 Pa pressure offset, 0.25 % of the local dynamic pressure, at a station
  where p/p0 = 0.997.  This low-Mach region is not in the asymptotic range on
  these grids; Mach at the same station converges normally (p_obs 1.98).

Output in `results/fvs_validation_<geometry>_<fluid>_N<Nx>_<timestamp>/`:
`shock_validation.png/.pdf`, one CSV per case (numerical + exact at the cell
centres), `summary.txt`, the settings file, `run_info.yaml`.

---

## 9. Files changed

| File | Change |
|---|---|
| `nozzle_hyperbolic/fvm_solver/fvm.py` | `make_jax_stadtke_step`: `linear_solver`, `newton_tol`, `flux_local_jac`, `cell_local_jac` (`_colored_columns`, `_jac_blocks`, `_banded_solve`, `_newton_dU`, custom JVPs).  `fvm_euler_1d`: forwards them, `fast_increments`, SER ramp, `[Speed-up]` log line, `CFL_eff` in the iteration lines, stores `parameters['U_puh_final']` and `parameters['run_stats']` |
| `nozzle_hyperbolic/fvm_solver/nozzle_exact_real_gas.py` | new: exact quasi-1D solution with real-gas EOS and normal shock |
| `dev_scripts/main.py` | reads the new keys; `cpu_affinity` pinning |
| `dev_scripts/cpu_affinity.py` | new: `pin_cpus()` (copy of the two-fluid driver's helper) |
| `dev_scripts/validation_fvs_shock_nozzle.py` | new: shock validation driver + figure |
| `dev_scripts/gci_fvs_validation.py` | new: 3-grid GCI of the validation runs vs the exact solution |
| `dev_scripts/settings/settings_fvs_validation_lettieri_air.yaml` | new: validation case |
| `dev_scripts/settings/settings{,_nozzle_hem,_nozzle_validation,_subsonic_pipe,_shock_tube}.yaml` | speed-up block (round-off-exact switches on, `ser_beta: 0.0`) |
| `dev_scripts/Lettieri_nozzle.xlsx` | geometry data (was missing) |
