# Physical verification cases

The physical case definitions are installed YAML resources beside their models:

- `models/euler_nozzle/cases/`: eight nozzle operating points from unchoked
  subsonic flow through internal/exit shocks and external-adjustment regimes.
- `models/euler_annular/cases/`: twelve radial, conical and curved annular cases,
  including swirl, supersonic expansion, shocks and a sonic endpoint.

The YAML files specify geometry, fluid, inlet/outlet physics, analytical branch
selection and declared accuracy targets. They contain no reconstruction,
integrator, CFL schedule or nonlinear solver settings. Mesh and numerical
profiles belong to the verification/benchmark client.

```python
import jax
jax.config.update("jax_enable_x64", True)

from nozzle_hyperbolic.verification import available_cases, build_case, load_case

definition = load_case("euler_annular/shock35")
problem = build_case(definition, cells=200)
model = problem.model
initial = problem.initial_state  # Model-owned cold guess, including two traces.
analytical_points = problem.reference_fields()
analytical_cells = problem.cell_reference()
```

`available_cases()` returns qualified `model/id` strings. `CaseDefinition`
provides the resource name and SHA-256 of the exact YAML bytes; `to_dict()` and
`data` return independent copies. Invalid/unknown keys and incompatible physical
choices are rejected. Application mappings can be validated with
`case_from_dict()` without adding a numerical solver dependency.

`VerificationProblem` exposes the production model and mesh, physical geometry,
fluid, analytical reference, resolved pressure and shock location. Its
`legacy_payload` contains only resolved plain data for an isolated legacy
worker, including fixed scales and the same cold state; it contains no live
current-package model object.

Use `numerical_fields(full)` for physical-cell numerical diagnostics.
`reference_fields(coordinates)` returns independent analytical point values.
`cell_reference()` instead integrates analytical conserved densities with the
physical volume measure, splitting cells at shocks and special geometry
locations. It provides `W`, conserved averages `q`, integrated contents, and
fields of the equivalent projected state. `reference_cell_state()`,
`reference_cell_contents()` and `reference_cell_fields()` expose these products
separately. An equivalent conservative cell state is not a point sample or an
average of primitive variables. Analytical fields must never become an
unreported numerical warm start.

All packaged steady cases currently use inviscid calorically perfect-gas
references. Numerical model calls still use their JAXProp EOS. Perfect-gas
formulae in this verification module are reference projection/recovery, outside
the evolution machinery. Annular pressure coefficient diagnostics use the same
fixed analytical inlet pressure and reservoir pressure for both numerical and
reference fields; zero-swirl fields retain nonzero physical normalization scales.

Accuracy targets retain their historical meaning at the declared recommended
finest mesh. Report coarser-mesh errors and refinement trends independently.
Known sonic/exit-limit cases carry explicit `known_limit` metadata; these notes
do not waive convergence, physical branch checks or accuracy targets.

For smooth annular cases, `conservative_normalized_l1` means the physical-volume
mean absolute conservative-density error divided by the model's fixed physical
conservative scale, with a target below `1e-3`; it is not relative to the mean
analytical value of each component. The fixed scale remains meaningful for zero
swirl. The companion Cp target is a physical-arclength mean absolute error no
greater than `0.01`. Annular shock cases retain the historical shock-location
target of two cells, while all field and conservative errors remain reported.
