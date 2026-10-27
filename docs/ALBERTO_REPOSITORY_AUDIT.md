# Alberto branch: repository and implementation audit

Date: 2026-10-26.

This audit records inherited findings for the structural refactor on `roberto-branch`. It does not authorize or implement numerical fixes. Function bodies, comments, docstrings, and the three expected-failure tests are to be preserved during this stage. Detailed JAX, JIT, and automatic-differentiation findings are recorded separately in [JAX_IMPLEMENTATION_AUDIT.md](JAX_IMPLEMENTATION_AUDIT.md).

## Provenance and scope

- Alberto source: `alberto-branch`, commit `6fe085c68c5c4c944868f1fd4e4f08f34c29b5e7`.
- Andrea comparison source: `andrea-branch`, commit `bcfe6ae1ec8abd71528e57b218ac8291dc9461fa`.
- All source locations below refer to the original Alberto tree and its line numbers, before relocation. They can be inspected with `git show alberto-branch:<path>` even after a file moves.
- The checkout already contained user changes to `pyproject.toml`, `poetry.lock`, and the shock-validation Python/YAML files, plus untracked `case_roberto/`, `case_study/cases_roberto/`, and `docs/`. These are not attributed to Alberto's committed source or to this audit.
- Review used source inspection and small isolated executions of the original source. It did not rerun the expensive research cases or certify their numerical results.

The current structural destination is `models/euler_nozzle/` for EulerNozzle physics, initialization, boundaries, fluxes, and its existing solver implementation; `models/two_fluid/` for TwoFluid; model-local `closures/` directories for correlations; and `thermodynamics/` for recovery helpers. Analytical references live in `models/euler_nozzle/analytical/` and `models/two_fluid/analytical.py`. EulerAnnular has design material in the local checkout but no implemented FVM solver in the Alberto commit. Creating its namespace is not implementation or validation of that solver.

## Baseline test result

The existing bounded test suite, run before the structural changes with the repository virtual environment and `pytest tests -q`, produced:

| Result | Count | Interpretation |
|---|---:|---|
| Passed | 25 | Existing checks passed in this environment. |
| Expected failure | 3 | The metastability restrictions intentionally documented with `xfail(strict=True)`. |
| Failed | 1 | `test_lowmach_flux_dissipation`: `KeyError: 'T'` in the ideal-gas test fixture path. |

The failure is an inherited fixture/interface mismatch: `tests/test_fvs.py`, `_IdealGas.get_state`, returns only `rho`, `a`, and `G` (line 37), whereas `nozzle_hyperbolic/fvm_solver/two_fluid/matrices.py`, `phase_props`, reads `st["T"]` (line 76). This prevents that test from reaching its dissipation assertion; it is not evidence that the dissipation calculation failed. Neither the fixture nor the implementation is repaired in this refactor.

`tests/test_fvs.py` checks ideal-gas matrix and EOS identities, partition properties, uniform and supersonic fluxes, MUSCL reconstruction, low-Mach dissipation, and a ramped-inlet helper. These checks do not establish real-fluid recovery, full transient accuracy, annular behavior, or end-to-end differentiation.

`tests/test_mass_transfer_metastability.py` deliberately records a closure discrepancy. `calculate_mass_transfer_lv` uses raw heat-transfer contributions, whereas the test's Städtke reference clamps driving temperatures to their metastable sides. The three stable-phase quadrant tests are strict expected failures. Preserve them as explicit documentation of the current model choice; changing the correlation would be a separate physical-model decision.

## Confirmed inherited errors

### A1. The NumPy Euler driver cannot reach its timestep calculation

**Evidence:** `nozzle_hyperbolic/fvm_solver/fvm.py`, `fvm_euler_1d`, lines 2307 and 2707.

The Städtke-only branch assigns a local function named `_compute_dt` through the return values of `make_jax_stadtke_step`. Python therefore treats `_compute_dt` as local throughout `fvm_euler_1d`. The separate NumPy path calls `_compute_dt(u, c, dx, CFL)` without executing that assignment. The module-level function of the same name is shadowed.

**Reproduction:** execute the unchanged baseline definitions of `fvm_euler_1d`, `_compute_dt`, `_bc_inlet_perfect`, `_bc_outlet_perfect`, and `_source_term` from an AST, with NumPy/math/time and the existing air heat capacities supplied as globals. A five-cell uniform positive-state case with `perfect_gas=True`, `space_method='rusanov'`, `time_method='euler'`, and one requested iteration raises:

```text
UnboundLocalError: cannot access local variable '_compute_dt' where it is not associated with a value
```

This is a confirmed runtime error for the tested perfect-gas path. The same scoping problem applies to the general-EOS Roe path, which also selects the NumPy loop. Relocating functions without changing their bodies does not fix this error.

### A2. The area-Mach derivative helper is incorrect

**Evidence:** `nozzle_hyperbolic/fvm_solver/nozzle_exact_solution.py:7` and `nozzle_hyperbolic/fvm_solver/exact_solution.py:7`, `areaMachDerivative`.

Both copies omit the derivative contribution from the Mach-dependent factor multiplying $1/M$. Isolated execution of the baseline helper, compared with a central difference of `areaMachRelation` at $\gamma=1.4$ with step $10^{-6}$, gives:

| Mach number | Helper result | Central difference |
|---:|---:|---:|
| 0.5 | -2.6796875 | -1.9140625 |
| 1.0 | -1.0 | approximately 0 |
| 2.0 | -0.84375 | 1.40625 |

The local `f_prime` variable currently is not passed to `fsolve` in `nozzle_exact2`. This is a confirmed helper error, not a demonstrated error in the current `fsolve` solutions. Preserve both copies during the structural move; deliberate deduplication and correction belong to a later stage.

### A3. Exact-nozzle initialization assumes ten points downstream of the throat

**Evidence:** `nozzle_hyperbolic/fvm_solver/nozzle_exact_solution.py:67` and `nozzle_hyperbolic/fvm_solver/exact_solution.py:69`, `nozzle_exact2`.

Both functions evaluate `x[A_cr_index+10]` before deciding whether a shock calculation is required. There is no check that the index exists.

**Reproduction:** isolated baseline `nozzle_exact_solution.py`, five coordinates from 0 to 1, `A_nodes=[2,1.5,1,1.5,2]`, `gamma=1.4`, `p_out=9e4`, and `total_bc=[300,1e5]` raises:

```text
IndexError: index 12 is out of bounds for axis 0 with size 5
```

The tested error is independent of the FVM solver. Larger grids can also fail if their minimum-area point lies too near the downstream boundary.

## Source-observed numerical and reporting issues

These findings follow from the implementation or its explicit existing comments. They have not been certified by new full-solver experiments.

| ID | Alberto source and observation | Consequence / later check |
|---|---|---|
| A4 | `fvm.py:2757-2763`, `fvm_euler_1d`: after an ordinary step writes state/time at index `n+1` and increments `n`, the return slices are `Q[:, :, :n]` and `t_vec[:n]`. | The newest calculated state is excluded. For one ordinary completed step, only the initial sample would be returned. A1 currently blocks normal execution of this path; retain this as a separate source-observed issue. |
| A5 | `fvm.py:2706-2733`, NumPy RK3/RK4 branches: source `S` is evaluated once and reused at every intermediate stage. Boundary values are set before the stages. | Nominal Runge-Kutta order cannot be inferred for the full state-dependent source/boundary problem. Later temporal-convergence checks should exercise those terms. |
| A6 | `flux.py:20-41`, `state_from_rho_u`: any root-solver exception selects the pressure-bracket midpoint. `_prim_real` at lines 239-244 substitutes previous-face pressure/sound speed, or `1e5` Pa and `100` m/s. `fvm.py:914-919`, `_real_gas_state_and_friction`, supplies whole-field default properties on batch EOS failure. | Invalid or failed thermodynamic recovery can produce plausible-looking flux inputs without a structured failure status. Retain current behavior now; future recovery contracts should make validity explicit. |
| A7 | `two_fluid/solver.py:1588-1595`, ghost-gas momentum pin: comments explicitly identify timestep-dependent converged states and a nonzero steady residual floor for the algebraic blend. | Relevant only when momentum pinning is enabled and `pin_mom_tau=None`. Existing measured values in the source are historical reports, not newly reproduced results. |
| A8 | `two_fluid/solver.py`, final steady-residual calculation around lines 2584-2610, normalizes by local absolute conserved variables and supports `ssr_mode='all'` or `'mass'`. `dev_scripts/OPEN_QUESTIONS.md`, section 2, records sensitivity to nearly absent phases. | Residual values and stopping criteria need interpretation alongside conservation diagnostics and phase masses. The historical note's proposed changes are not implemented or endorsed as a proven correction here. |
| A9 | `fvm.py`, `make_jax_step`, `_LIMS['muscl_superbee']`: the JAX lambda takes the maximum of two already-signed candidates. For slopes `(1,2)` it yields `2`, but for `(-1,-2)` it yields `-1`. | The JAX limiter is asymmetric under sign reversal. The standalone NumPy `_superbee` selects by absolute magnitude and does not have this defect; the two implementations differ. |
| A10 | `flux.py`, `F_HLLC`, recovers star-state internal energy from the EOS at star density/pressure; `fvm.py`, `_jax_F_hllc`, uses a conservative jump-condition energy expression. | NumPy and JAX implementations are not identical algorithms. Equivalence and conservation need dedicated checks before consolidation. This observation alone does not certify which implementation is appropriate for every EOS. |
| A11 | `fvm.py`, `euler_roe_matrix`, uses averaged sound speed and a perfect-gas-style contact eigenvector structure for a general EOS. | A general-EOS Roe linearization and its flux-difference property are not established by this implementation. Record as a mathematical verification requirement, not a newly demonstrated numerical failure. |

The JAX audit covers tracer errors, derivative behavior, and implementation-specific transformation limits. No statement here that a kernel is written in JAX implies that its enclosing run driver or all parameter sensitivities support JIT/AD.

## Repository and documentation inconsistencies

### Configuration descriptions have drifted

- `dev_scripts/OPEN_QUESTIONS.md`, sections 3-4, describes `w_char='auto'` and EOS rows as candidates for promotion. The current `dev_scripts/settings/defaults.yaml:52-55` already selects `eos_rows: true`, `eos_rows_banded: true`, and `w_char: auto`. Distinguish function defaults from application defaults and treat the old note as historical.
- `two_fluid/solver.py:1491` describes `eps_ghost_k2` as default 4. Its function default at line 150 and `dev_scripts/settings/defaults.yaml:84` are 0.
- `two_fluid/sources.py`, `TwoFluidClosure.sigma_Ql`, has a comment saying heat goes to gas, although this field denotes liquid-phase heat. This is a comment inconsistency to preserve and record now.
- `fit_nozzle_shape.py`, `fit_nozzle_shape`, documents two return values but returns the polynomial coefficients, polynomial, and spline. Existing callers unpack three values.

### Reproduction material is incomplete in the tracked tree

Code and tests cite files absent from Alberto's tracked tree:

- `hem_nozzle_implementation_spec.md` (`tests/test_fvs.py:4`).
- `stadtke_enthalpy_frame_derivation_latest.md` (`two_fluid/__init__.py:18` and `matrices.py:4`).
- `stadtke_twofluid_implementation_pipeline_with_sources.md` (`two_fluid/sources.py:12`).
- `mass_flow_diagnostic.md` (`fvm.py:2531`).
- `dev_scripts/debug_projection_tri.py` (`two_fluid/solver.py:298,2360-2367`).

`.gitignore` excludes `debug_*`, `check_*`, `implementation_notes/`, fluid tables, and result directories. Some missing material may exist in other local environments; absence from this commit does not establish that it never existed. It does mean a clean checkout cannot reproduce all cited certification probes from the tracked files alone.

### Packaging and test discovery depend on the environment

- Committed `pyproject.toml` uses `jaxprop = { path = "../jaxprop/", develop = true }`. That sibling directory does not exist relative to this checkout. The pre-existing local user edit already selects a registry version; it must not be overwritten or described as a refactoring change.
- `nozzle_hyperbolic/fit_nozzle_shape.py:37` directly imports `pandas`; configuration/drivers directly import `yaml`. Neither is declared as a direct dependency in this manifest. Excel reads also require an installed reader engine. Current transitive dependencies or environment contents can mask this portability issue.
- The available virtual environment does not have this package installed: importing `nozzle_hyperbolic` from outside the checkout without adding its source path raises `ModuleNotFoundError`. It also lacks `poetry-core` and `build`, so an installed-wheel verification was not performed. The baseline package was likewise available to the comparison probes through an explicit source path. This is an environment limitation, not evidence of a relocation regression.
- `nozzle_hyperbolic/__init__.py:10` changes the process exception hook. The EulerNozzle and TwoFluid solver modules enable JAX x64 at import. Plotting and geometry-fitting dependencies are eagerly imported through package initializers. The structural move preserves these existing effects.
- `two_fluid/sources.py:25-27` mutates `sys.path` to import bare `source_two_fluid`. The refactor must route this import to the model-local correlation module without leaving the source stranded at its old path. This is import plumbing, not a numerical fix.
- Bare repository-wide pytest discovery includes research scripts named `test_*.py`. For example, `validation_cases/Spinelli2019/test_plot_data.py:8` reads `cases_summary.xlsx` relative to the process working directory at import and constructs plots. Its quoted f-string syntax also requires Python 3.12, whereas the manifest permits 3.11. `pytest tests` is the bounded existing unit-suite command; it does not certify every research script.

## Architecture observations that constrain the refactor

EulerNozzle currently has both perfect-gas and general-EOS paths within the same implementation. `flux.py` already contains Rusanov, Roe, HLL, HLLC, and Lax-Friedrichs variants on Alberto's branch, alongside the Städtke FVS implementation in `fvm.py`. Therefore, FDS-style options are not entirely absent from Alberto. Future integration of Andrea's methods still requires explicit comparison of state, flux, area-weighting, EOS, and boundary conventions.

TwoFluid uses six physical primitive variables, six balance variables, and eight Newton unknowns. A shared numerical layer must not assume that these spaces have equal dimensions. `state.py` and `matrices.py` belong with TwoFluid; the interphase correlations in `source_two_fluid.py` belong in that model's `closures/`. Physical boundary implementations also belong with their model, even when their arrays are passed to a common numerical driver.

Generic reconstruction, Jacobian assembly, linear algebra, and iteration control are candidates for shared infrastructure when their mathematical contracts match. Model-specific residual assembly, nonconservative terms, closure placement, state recovery requirements, and boundary dependence must remain explicit. Current nested implementations may remain intact until a later behavior-preserving extraction is agreed.

Geometry and reference solutions can be relocated without changing their formulas. Their existing sampling assumptions, hardcoded reference constants, and duplicate implementations should remain visible rather than being silently reconciled during a structural change.

No NozzleX source is imported or changed as part of this audit or the present refactor. Future reuse requires numerical comparisons after the standalone package has a stable structure and representative case baselines.
