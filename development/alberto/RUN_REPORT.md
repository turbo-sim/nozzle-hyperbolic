# Alberto legacy files: classification and execution audit

Date: 2026-10-28. Scope: the collection now under `development/alberto/` only, with existing source imported from
the project environment. No numerical model, production source, default YAML
setting, or dependency version was changed.

**True means the stated execution/check completed. False means it did not
complete in this audit.** A False caused by a timeout or missing input is not
proof that the script is broken or needs a numerical code modification. Helper
coverage is explicit; merely importing a helper was not treated as a full test.

The audit below predates the subsequent Alberto/Roberto directory consolidation.
Its source links and commands were updated; the recorded tests were **not rerun**.
Raw logs and JSON preserve measurement-time paths and settings.

## Python programs, helpers and cluster launcher

| File | Purpose | Works | Observed result / coverage |
|---|---|---|---|
| [euler_nozzle/main.py](euler_nozzle/main.py) | Legacy single-fluid Euler/HEM nozzle driver | **False** | Default HEM case did not reach the solver: real-fluid table generation was stopped after about 258 s. No code failure established. |
| [euler_nozzle/shock_tube_fvs.py](euler_nozzle/shock_tube_fvs.py) | Explicit FVS Sod shock tube with tabulated air | **True** | Full default script completed in 49 s; finite L1 errors printed. No accuracy thresholds are asserted by this legacy script. |
| [two_fluid/main_two_fluid.py](two_fluid/main_two_fluid.py) | Two-fluid flashing nitrogen default campaign | **False** | Default nitrogen case stopped during table generation after 90 s. Additional 401-point ideal-gas run progressed through about 650 steps before its 120 s limit. A separate 21-point/3-step smoke completed including CSV and plotting. |
| [two_fluid/main_two_fluid_grid_sequence.py](two_fluid/main_two_fluid_grid_sequence.py) | Coarse-to-fine two-fluid restart campaign | **False** | Executed with the existing nitrogen configuration; coarse-level property-table generation exceeded 90 s. Full grid ladder remains unverified. |
| [two_fluid/config_loader.py](two_fluid/config_loader.py) | Resolve settings inheritance and typed configuration | **True** | All 17 complete two-fluid case YAMLs load. The six finite-difference configurations also load; defaults are inherited fragments. |
| [two_fluid/initial_conditions.py](two_fluid/initial_conditions.py) | Build prescribed, cold, restart or external-solver initial fields | **True** | Uniform, stagnation and simple initializers returned finite (7,6) arrays. Restart/external NozzleX pathways are not certified. |
| [two_fluid/settings/settings_two_fluid.py](two_fluid/settings/settings_two_fluid.py) | Legacy Python configuration wrapper | **True** | Executed wrapper and loaded its original YAML. |
| [two_fluid/settings/settings_phase_change.py](two_fluid/settings/settings_phase_change.py) | Legacy phase-change configuration wrapper | **True** | Executed wrapper and loaded its original YAML. |
| [sensitivity/sensitivity_study.py](sensitivity/sensitivity_study.py) | Solve and differentiate interphase-closure parameters | **False** | Fixed stale interfacial_area import to the existing original law. Default attempt then exceeded 90 s in property-table generation; no completed sensitivity result. |
| [sensitivity/sensitivity_closure.py](sensitivity/sensitivity_closure.py) | Differentiate an existing saved converged two-fluid field | **False** | No saved converged two-fluid run is available in its expected folder. Default lookup raises IndexError on the empty run list; supply a valid run directory. |
| [sensitivity/sensitivity_plots.py](sensitivity/sensitivity_plots.py) | Render sensitivity profiles and diagnostics | **True** | Rendered the field-sensitivity plotting function with canned data. A tight_layout warning remains. |
| [support/cpu_affinity.py](support/cpu_affinity.py) | Pin a solver process to selected logical CPUs | **True** | Pinned only a disposable child to one CPU allowed by its current affinity. |
| [support/run_info.py](support/run_info.py) | Save code, environment and input provenance | **True** | Wrote and read back real provenance, versions and input checksum. |
| [hpc/run_job.sh](hpc/run_job.sh) | LSF cluster launcher for the legacy Euler driver | **False** | Bash syntax check passed. Full Linux/LSF job is not run on Windows and was not submitted. |

## Important distinctions

- The self-contained Sod script executes, but that does not certify its
  numerical accuracy. It reports density/velocity/pressure L1 errors of about
  0.0436 kg/m?, 50.46 m/s and 5492 Pa, without pass/fail tolerances. Its numerical
  method and reference assumptions have not been adjusted.
- The two-fluid **default** remains False/unverified in the table. The extra
  **smoke is True for execution only**: it uses a copied ideal-gas configuration
  with 21 points and three steps, under `outputs/audit/`, and exercises setup,
  compilation, stepping, diagnostics, solution export and plots. It deliberately
  does not meet steady convergence, and prints transient/checkerboard diagnostics.
  Original settings files were not edited. This proves an executable driver
  path, not a converged flashing-flow or mesh-sequence solution.
- A full existing 401-point gas-isentropic case also made finite timestepping
  progress before its time limit. No solver exception was observed there.
- Table-generation limits are prerequisites/coverage limitations. Prepared EOS
  caches and larger execution budgets are required to certify those defaults.
- Closure sensitivities require a **converged**, compatible saved two-fluid
  field. The short smoke is not a valid substitute, and archived Euler NPZs
  are not compatible inputs. Missing data were not fabricated to obtain a pass.
- The HPC entry is an environment limitation, not a diagnosed numerical error.
  Its syntax was checked with `bash -n`; no cluster job was submitted.

## Repairs made, and scope preserved

All repairs are import/path or launch-path changes associated with relocation:

1. Replaced sibling imports of `config_loader`, `initial_conditions`,
   `run_info`, `cpu_affinity` and `sensitivity_plots` with canonical package
   imports. Removed obsolete `sys.path` insertions; launch with `python -m`.
2. Kept configurations next to their model drivers and the full two-fluid
   defaults chain together. Corrected the sensitivity driver's default config
   path and the saved-run search to the new output root.
3. Corrected the project JAX cache path, provenance repository root, grid
   subprocess module/working directory, and generated output paths.
4. Repaired the stale sensitivity-law import using
   `closures.source_two_fluid.interfacial_area_original as interfacial_area`.
   The Nb/Nd number-density law and its existing call signatures are retained;
   it was not replaced by a different polynomial/parabola closure.
5. Updated the archived cluster launcher's repository/environment/module paths.
   Its resource and notification directives were preserved.

No time-integration, flux, reconstruction, closure formula, derivative method,
convergence tolerance or reference solution was repaired. All `src/` Python
hashes match the pre-audit snapshot. Original YAML and historical Markdown
files were moved byte-for-byte. Historical notes still contain old names and
commands; use the new README commands for current execution.

Several other case-study wrappers elsewhere in the repository still refer to
the already-obsolete `dev_scripts/` location. They were outside this requested
directory audit and were not silently rewritten, especially where unrelated
working edits exist. No compatibility claim is made for those external callers.

## Configuration inventory

All 35 YAML files parse. All 17 complete two-fluid cases and six sensitivity
finite-difference cases pass the typed configuration loader. The five Euler
configurations are parsed with their separate legacy schema; the seven defaults
files are inheritance fragments, not standalone simulations. No full physical
solve is implied by a configuration-load pass.

| Configuration | Kind | Check |
|---|---|---|
| [euler_nozzle/settings/settings.yaml](euler_nozzle/settings/settings.yaml) | Euler case | True: YAML parse only |
| [euler_nozzle/settings/settings_nozzle_hem.yaml](euler_nozzle/settings/settings_nozzle_hem.yaml) | Euler case | True: YAML parse only |
| [euler_nozzle/settings/settings_nozzle_validation.yaml](euler_nozzle/settings/settings_nozzle_validation.yaml) | Euler case | True: YAML parse only |
| [euler_nozzle/settings/settings_shock_tube.yaml](euler_nozzle/settings/settings_shock_tube.yaml) | Euler case | True: YAML parse only |
| [euler_nozzle/settings/settings_subsonic_pipe.yaml](euler_nozzle/settings/settings_subsonic_pipe.yaml) | Euler case | True: YAML parse only |
| [sensitivity/settings/settings_fd_nb_hi.yaml](sensitivity/settings/settings_fd_nb_hi.yaml) | Sensitivity finite-difference case | True: typed loader |
| [sensitivity/settings/settings_fd_nb_hi2.yaml](sensitivity/settings/settings_fd_nb_hi2.yaml) | Sensitivity finite-difference case | True: typed loader |
| [sensitivity/settings/settings_fd_nb_lo.yaml](sensitivity/settings/settings_fd_nb_lo.yaml) | Sensitivity finite-difference case | True: typed loader |
| [sensitivity/settings/settings_fd_nb_lo2.yaml](sensitivity/settings/settings_fd_nb_lo2.yaml) | Sensitivity finite-difference case | True: typed loader |
| [sensitivity/settings/settings_fd_ndhi.yaml](sensitivity/settings/settings_fd_ndhi.yaml) | Sensitivity finite-difference case | True: typed loader |
| [sensitivity/settings/settings_fd_ndlo.yaml](sensitivity/settings/settings_fd_ndlo.yaml) | Sensitivity finite-difference case | True: typed loader |
| [two_fluid/settings/defaults.yaml](two_fluid/settings/defaults.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_flashing.yaml](two_fluid/settings/defaults_flashing.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_flashing_cold_start.yaml](two_fluid/settings/defaults_flashing_cold_start.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_gas_dominant.yaml](two_fluid/settings/defaults_gas_dominant.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_legacy.yaml](two_fluid/settings/defaults_legacy.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_liquid.yaml](two_fluid/settings/defaults_liquid.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/defaults_two_component.yaml](two_fluid/settings/defaults_two_component.yaml) | Defaults fragment | True: YAML and inherited-chain use |
| [two_fluid/settings/settings_flashing_nitrogen_simoneau.yaml](two_fluid/settings/settings_flashing_nitrogen_simoneau.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_flashing_nitrogen_simoneau_test.yaml](two_fluid/settings/settings_flashing_nitrogen_simoneau_test.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_flashing_smd.yaml](two_fluid/settings/settings_flashing_smd.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_flashing_smd_space_marching_comparison.yaml](two_fluid/settings/settings_flashing_smd_space_marching_comparison.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_flashing_smd_supersonic_test.yaml](two_fluid/settings/settings_flashing_smd_supersonic_test.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_phase_change.yaml](two_fluid/settings/settings_phase_change.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_phase_change_new.yaml](two_fluid/settings/settings_phase_change_new.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_single_phase_nozzle.yaml](two_fluid/settings/settings_single_phase_nozzle.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid.yaml](two_fluid/settings/settings_two_fluid.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_gas_isentropic.yaml](two_fluid/settings/settings_two_fluid_gas_isentropic.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_gas_shock.yaml](two_fluid/settings/settings_two_fluid_gas_shock.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_gas_subsonic.yaml](two_fluid/settings/settings_two_fluid_gas_subsonic.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_liquid_bernoulli.yaml](two_fluid/settings/settings_two_fluid_liquid_bernoulli.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_liquid_bernoulli_symmetric.yaml](two_fluid/settings/settings_two_fluid_liquid_bernoulli_symmetric.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_nozzle_validation.yaml](two_fluid/settings/settings_two_fluid_nozzle_validation.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_nozzle_validation_two_comonent.yaml](two_fluid/settings/settings_two_fluid_nozzle_validation_two_comonent.yaml) | Two-fluid case | True: typed loader |
| [two_fluid/settings/settings_two_fluid_nozzle_validation_two_component_test_static.yaml](two_fluid/settings/settings_two_fluid_nozzle_validation_two_component_test_static.yaml) | Two-fluid case | True: typed loader |

## Other original files

- [two_fluid/OPEN_QUESTIONS.md](two_fluid/OPEN_QUESTIONS.md): historical two-fluid
  correctness questions; preserved without claiming they are resolved now.
- [two_fluid/SPEEDUP_REPORT.md](two_fluid/SPEEDUP_REPORT.md): historical two-fluid
  performance notes, with old measurements and path/status claims.
- [euler_nozzle/SPEEDUP_REPORT_FVS_SINGLE_PHASE.md](euler_nozzle/SPEEDUP_REPORT_FVS_SINGLE_PHASE.md):
  historical single-phase FVS performance notes.
- Three existing `shock_roe_*_N500.npz` outputs were preserved under
  `outputs/archived_results/paper_reproduction/`. Data files are not executables
  and were not counted as script passes.

## Reproduction and evidence

Run from the repository root in the Poetry environment, headlessly:

```powershell
poetry run python -m development.alberto.audit.scripts.run_checks --timeout 90
```

The initial Euler property-table attempt was stopped after about 258 seconds;
most subsequent heavy attempts used a 90-second cap. The additional original
ideal-gas campaign used 120 seconds; the reduced smoke completed in 26 seconds.
Headless plotting and single-thread BLAS are audit environment choices, not
changes to the physical/numerical defaults. No pytest tests were added.

Detailed commands, logs and outcomes:

- [Execution records](outputs/audit/results.json), including the HPC preflight.
- [Additional full-case/smoke records](outputs/audit/results_extra.json).
- [Typed case loads](outputs/audit/configurations.json) and
  [all-YAML inventory](outputs/audit/yaml_inventory.json).
- [Original-to-new file map and hashes](outputs/audit/relocation_inventory.json).
- Logs, provenance and a sample rendered sensitivity figure are alongside them.

Generated evidence is ignored through the repository's `outputs/` rule. This
report and the relocated source/settings are reviewable changes; no commit
was made by this audit.
