# Lettieri equilibrium HEM validation

This case uses the shared `NozzleModel` and generic FVM solver, equilibrium
JAXProp CO2 bicubic properties, and exactly zero wall shear and heat transfer.
It is separate from the nonequilibrium two-fluid port. The experiment contains
delayed condensation; disagreement of an equilibrium calculation is a physical
validation result, not automatically a solver defect.

Run from the repository root in its existing Poetry environment:

```powershell
poetry run python -B -m cases.lettieri_validation.scripts.audit
poetry run python -B -m cases.lettieri_validation.scripts.run --prepare-only
poetry run python -B -m cases.lettieri_validation.scripts.run --case L1 --cells 24 --max-steps 400
poetry run python -B -m cases.lettieri_validation.scripts.run --case all --cells 100
poetry run python -B -m cases.lettieri_validation.scripts.verify_results
poetry run python -B -m cases.lettieri_validation.scripts.report
poetry run python -B -m cases.lettieri_validation.scripts.audit_report
```

The last campaign command should follow a successful small solver check.
`--stretch 1.6` supplies a nonuniform grid. `--restart <solution.npz>` explicitly
interpolates an earlier saved state, recording the restart path. Each requested
run retains settings, source hashes, table hash/preparation cost, accepted and
rejected history, final state and original residual checks in `outputs/`.
Preparation is recorded separately from solve time. A failed run remains failed;
its pressure errors are diagnostics of an unfinished state, not validation.

## Property preparation and admissibility

The selected table is Andrea's equilibrium CO2/HEOS bicubic configuration:
200 by 200 nodes, $h=50$–850 kJ/kg and $p=0.1$–300 bar. Its first generation
reported 29,440 successful evaluations out of 40,000; JAXProp skips unsupported
states elsewhere in this broad rectangle. Finite arrays alone do not prove a
populated table. The case therefore exposes the fixed operating rectangle
$h=300$–550 kJ/kg, $p=10$–100 bar to the existing model admissibility check.
Every interpolation corner and coefficient supporting that rectangle is audited;
density, temperature and sound-speed nodes must additionally be positive.
The source interpolants and tabulated derivative fields are unchanged.
There are no property substitutions, extrapolation, clamping or density-derived
EOS experiments. An out-of-domain candidate is rejected. The early unguarded
L1 pilot remains labeled as a preflight result rather than accepted validation.

The existing model checks for finite positive thermodynamic properties, returned
HP coordinates and its existing entropy-derivative condition remain active.
The study neither changes their tolerances nor investigates EOS consistency.

## Data and revisions

The unchanged `.xlsx` files in `data/` are copied from fetched Andrea revision
`bcfe6ae1ec8abd71528e57b218ac8291dc9461fa`. The audit checks each file against
its Git blob. `Case1.xlsx` and `Case5.xlsx` contain 13 pressure measurements
each. Their pressures exactly match the first repeat of publication cases 1
and 5. Their axial positions are digitizations, slightly different between cases.
`Case1_CFD.xlsx` and `Case5_CFD.xlsx` contain independently sampled pressure
and liquid mass fraction columns; they are kept as separate series.

The source is Lettieri et al., *Characterization of Nonequilibrium Condensation
of Supercritical Carbon Dioxide in a de Laval Nozzle*, DOI
[10.1115/1.4038082](https://doi.org/10.1115/1.4038082), April 2018 issue,
published online 7 November 2017. The upstream folder label `Lettieri2018`
and figure label “2017” refer to this same publication. The
[MIT record](https://hdl.handle.net/1721.1/114689) identifies the final article;
the [author-posted copy](https://www.researchgate.net/publication/320208117_Characterization_of_Nonequilibrium_Condensation_of_Supercritical_Carbon_Dioxide_in_a_de_Laval_Nozzle)
provided the inspected Appendix Tables 2–4. `data/publication.json` transcribes
the first repeat for all five cases. No L2–L4 original Excel or reference CFD
files were found in either fetched branch or the local repositories.

| Label | Publication case/repeat | Total pressure (bar) | Total temperature (K) | Condensation onset (mm) |
|---|---|---:|---:|---:|
| L1 | 1/1 | 58.96 | 314.67 | 3.748 |
| L2 | 2/1 | 65.35 | 311.99 | 1.660 |
| L3 | 3/1 | 73.53 | 313.60 | -0.505 |
| L4 | 4/1 | 79.99 | 313.94 | -3.462 |
| L5 | 5/1 | 84.74 | 313.88 | -4.675 |

Use the Appendix's achieved reservoir conditions, not the rounded target
conditions of its Table 1. Andrea rounded L1/L5 temperatures to 314.7/313.9 K;
this validation retains the measured precision. The outlet is extrapolated
supersonic outflow and must pass the model's final regime audit. The supplied
`0.25*p0` pressure is only a cold-start guess; it is not an invented measured
backpressure and does not enter the supersonic boundary equations.

## Geometry and mass-flow conventions

The 33 measured wall points define half-height $y(x)$. The publication's
Table 4 axial coordinate is reversed, $x=-z$, to make flow run left to right.
The exact interval is $[-0.032013,0.066355]$ m. The original Excel file is
preserved, but its inlet half-height of 0.00638 m is explicitly corrected to
Table 4's 0.00635 m when constructing the spline. All other coordinates agree.
No synthetic nozzle fit or degree-10 polynomial is used: the upstream active
geometry is the not-a-knot cubic spline, whose antiderivative supplies exact
cell volumes and whose face-area differences supply integrated geometry sources.

The configured depth $b=6.48$ mm is Andrea's full-channel convention: $A=2by$.
The primary paper's experimental-apparatus section reports a 20 mm² throat
area. Dividing by Table 4's full throat height $2(1.5431)$ mm gives 6.4805 mm,
consistent with this selected depth to the precision of the published area.
This resolves the factor of two without adopting a secondary source's differing
depth. In this zero-friction quasi-1D case a constant depth multiplier changes
dimensional mass flow, but not the pressure/velocity profile. Report both

$$
\dot m_{\rm full}=\rho u(2by),\qquad
\dot m'_{\rm half}=\frac{\dot m_{\rm full}}{2b}=\rho u y.
$$

The second quantity has units kg/(s m), rather than kg/s. Andrea's quoted
Romei/Persico values 29.43/48.94 for L1/L5 use this half-channel per-depth
convention. They describe a separate CFD model, not measured mass flow and not
an acceptance target for the present zero-friction calculation. The CFD
workbooks are attributed by Andrea to Romei/Persico (2021),
[DOI 10.1016/j.applthermaleng.2021.116816](https://doi.org/10.1016/j.applthermaleng.2021.116816).
This attribution and the mass-flow values are corroborated by Cioffi et al.,
[GT2025-153250](https://doi.org/10.1115/GT2025-153250), Figure 8 and Table 3.
The digitization procedure and precise source figure in the 2021 article were
not supplied, so that part of the provenance remains incomplete.

## Remaining measurement limitations

Exact pressure-tap coordinates were not recovered from primary tabulated data.
Both original L1/L5 coordinate series round to the same nominal positions
$-28,-21,\ldots,56$ mm. L2–L4 use those explicitly approximate positions for
comparison, with the exact Table 2 pressures. No new measurement coordinates
or uncertainties are claimed. The original L1/L5 digitized positions are used
unchanged. Coordinate and pressure uncertainty fields remain null; error bars
are omitted. A rigorous uncertainty-weighted validation needs the original tap
coordinates and uncertainty metadata. No experimental mass-flow measurements
or L2–L4 reference CFD series were recovered.

See [REPORT.md](REPORT.md) for the actual run status and limitations.
