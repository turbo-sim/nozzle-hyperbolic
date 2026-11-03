# Physical cases and reference data

This directory groups the former `case_study/` applications and
`validation_cases/` collection under one parent. Production solvers live in
[`src/nozzle_hyperbolic/`](../src/nozzle_hyperbolic/); numerical investigations
and refactoring checks live in [`development/`](../development/README.md).

| Directory | Contents |
|---|---|
| [NICFD2026/](NICFD2026/README.md) | All six published flashing-water and nitrogen–water cases, current-solver measurements and standalone comparison reports |
| [lettieri_validation/](lettieri_validation/README.md) | Five-case equilibrium HEM comparison, measured contour, original L1/L5 Excel/CFD and transcribed published L2–L4 pressure data |
| [validation/](validation/notes.md) | Literature data from Barana, Simoneau, Spinelli, and Zhu/Elbel, plus data processing and plotting scripts |
| [validation_shock_fvs/](validation_shock_fvs/) | Lettieri nozzle FVS shock comparison, settings and nozzle geometry |
| [data_assimilation/](data_assimilation/) | Inverse-designed CO2 nozzle runs, staged cold starts and interfacial-area postprocessing |
| [nicfd/](nicfd/) | Grid-convergence analysis and publication figure scripts |
| [r1233zde_inlet_sensitivity/](r1233zde_inlet_sensitivity/README.md) | R1233zd(E) flashing-nozzle inlet studies and property-table utilities |
| [flashing-nozzle-data/](flashing-nozzle-data/README.md) | Flashing-nozzle dataset metadata, indexing and thermodynamic plotting tools |

The historical folder name `validation_shock_fvs` is retained. Comparison with
an analytical solution is verification; comparison with experiments is
validation. The [reference-data notes](validation/notes.md) discuss that distinction.

Imports referring to the former `case_study` package now use `cases`, for example
`cases.nicfd.grid_convergence_gci`. Update local bookmarks and commands to use
`cases/<case>/` and `cases/validation/<dataset>/`.

This consolidation preserves case settings, function bodies and saved data;
only locations, package references and associated documentation were updated.
No case programs or simulations were rerun. Older scripts retain their existing
working-directory assumptions and pre-existing dependencies: in particular,
the data-assimilation wrappers still refer to the retired `dev_scripts/`
driver. Their execution needs a separate review against the maintained
[`development/alberto/two_fluid/`](../development/alberto/two_fluid/) driver.
This directory move does not establish that those legacy cases run successfully.

Generated outputs remain ignored through the repository `.gitignore`. Saved
run metadata retains its original paths as historical provenance.
