# Code organization on roberto-branch

Date: 2026-10-26.

This step reorganizes **nozzle_hyperbolic only**. It does not integrate NozzleX,
port code from Andrea, implement EulerAnnular, change numerical algorithms,
or repair the issues recorded in [the repository audit](ALBERTO_REPOSITORY_AUDIT.md)
and [the JAX audit](JAX_IMPLEMENTATION_AUDIT.md).

## Baseline and preservation

`roberto-branch` starts at Alberto's local branch tip, `6fe085c`. Andrea was
reviewed at `bcfe6ae`. Both local tips matched their existing remote-tracking
refs when inspected; no remote update was requested or fetched. These histories
have no common Git ancestor in this non-shallow checkout. Similar source is
therefore described as source overlap, not Git ancestry.

The existing Roberto notes/cases and annular design documents are retained on
this branch. The initial local changes to `pyproject.toml`, `poetry.lock`, and the
two shock-validation files are preserved separately from the refactoring.

Existing functions, nested functions, classes, comments, and docstrings are
preserved. Self-contained top-level definitions move verbatim; only module-level
imports and filesystem paths are rewired. The original factory signatures and
configuration keys remain unchanged. Some old comments describe earlier names
or behavior; the audit documents clarify them without rewriting historical text.

The existing flat package layout is retained to preserve checkout-based scripts
and current packaging. A `src/` conversion is optional later, alongside an
installation/entry-point migration. Internal ownership is independent of that
choice and can later be transplanted beneath `nozzlex`.

## Current implemented layout

```text
nozzle_hyperbolic/
  __init__.py                     existing public imports and side effects
  fit_nozzle_shape.py             existing fitting utility, unchanged
  plot_settings.py                existing plotting utility, unchanged
  models/
    euler_nozzle/                EulerNozzle model family
      __init__.py
      equations.py              physical fluxes, eigenvalues, retained Roe matrices
      state.py                  conservative / (p,u,h) conversion
      initialization.py         existing initial-condition builders
      boundaries.py             standalone Euler boundary routines
      sources.py                equation sources and state/friction evaluation
      closures/wall.py          wall and equilibrium-mixture correlations
      reconstruction.py         model-dependent primitive reconstruction
      flux.py                   retained NumPy FDS/Riemann fluxes
      flux_jax.py               existing JAX Euler FVS and FDS/Riemann fluxes
      fvm.py                    assembly, intact factories and run driver
      analytical/
        __init__.py
        exact_solution.py       retained alternative nozzle reference
        nozzle_exact_solution.py
                                retained perfect-gas nozzle reference
        nozzle_exact_real_gas.py
                                retained real-fluid nozzle reference
        shock_tube.py           exact Sod shock-tube solution
    two_fluid/                   TwoFluid model family
      __init__.py                existing public TwoFluid exports
      state.py                  six-variable physical/storage transformations
      matrices.py               thermodynamics bundle, model matrices/eigenstructure
      sources.py                source assembly and closure evaluation
      closures/source_two_fluid.py
                                interphase drag, heat, mass and area laws
      boundaries.py             existing ramped-inlet function
      flux.py                   model FVS and nonconservative correction
      solver.py                 intact eight-variable Newton/FVM factory
      analytical.py             isentropic, shock, subsonic and Bernoulli references
    euler_annular/               reserved EulerAnnular namespace
      __init__.py                explicitly states implementation is pending
      boundaries.py             ownership placeholder; no boundary implementation
      closures/__init__.py       ownership placeholder; no closure implementation
  geometry/duct.py               unchanged geometry definitions/sampling
  thermodynamics/
    recovery.py                 existing state inversions
    perfect_gas_props.py        retained property wrapper
    perfect_gas_props_functions.py
  numerics/
    reconstruction.py          array limiters and reconstruction arithmetic
    linear.py                  equilibrated solve and least-squares kernels
  solvers/time_marching/cfl.py  existing shared CFL helper
  fvm_solver/                  compatibility modules for original import paths
    Lettieri_nozzle.xlsx        original data location retained
```

`EulerNozzle`, `TwoFluid`, and `EulerAnnular` are the intended model names.
This move does **not** introduce classes with those names. Existing factories and
functions remain the callable API. A perfect gas is a thermodynamic choice within
EulerNozzle; the existing specialized perfect-gas routines are retained for
compatibility and comparison.

Analytical and semi-analytical solutions belong to the physical model they
describe. Euler nozzle and shock-tube references live in
`models/euler_nozzle/analytical/`; the two-fluid dominant-phase references live
in `models/two_fluid/analytical.py`. Development and case-study scripts import
these implementations. There is no top-level reference-solution package.

The standalone Euler boundary routines and the TwoFluid ramped inlet now have
model-local homes. Other boundaries, local limiters, residuals, and Newton
utilities remain nested inside their original model-specific factories.
Extracting those requires changes to the enclosing function bodies and is
outside this step. In particular, moving generic kernels does not claim that
the two large factories already use one shared Newton implementation.

`models/two_fluid/sources.py` retains its existing `sys.path` setup, adjusted to
the new closure location, while its import uses the canonical relative package
path. Removing the path side effect is a later cleanup. Historical bare imports
are not the proposed public API.

## Relocation and compatibility

| Original location under `nozzle_hyperbolic/` | Canonical location |
|---|---|
| `fvm_solver/fvm.py` | `models/euler_nozzle/fvm.py`, with top-level groups extracted as listed above |
| `fvm_solver/flux.py` | `models/euler_nozzle/flux.py` |
| `fvm_solver/two_fluid/{state,matrices,flux,sources,solver}.py` | `models/two_fluid/`, with ramped inlet and equilibrated linear helpers extracted |
| `fvm_solver/source_two_fluid.py` | `models/two_fluid/closures/source_two_fluid.py` |
| `fvm_solver/geometry.py` | `geometry/duct.py` |
| `fvm_solver/perfect_gas_props*.py` | `thermodynamics/` |
| `fvm_solver/{exact_solution,nozzle_exact_solution,nozzle_exact_real_gas}.py` | `models/euler_nozzle/analytical/` |

The analytical initial-condition builders formerly defined in
`development/initial_conditions.py` now live in `models/two_fluid/analytical.py`.
The driver module re-exports them, and the duplicated NICFD isentropic module
forwards to the same implementations. The plotting-specific subsonic reference
also lives with TwoFluid, preserving its existing behavior. The exact Sod
routine formerly defined in `development/shock_tube_fvs.py` and the vectorized
comparison from `development/main.py` now live in
`models/euler_nozzle/analytical/shock_tube.py`. The latter retains its supplied
EOS sound speeds and existing ideal-gas approximation.

Old module paths forward to the canonical module objects, including private
names explicitly imported by current tests and cases. The legacy TwoFluid
package re-exports its existing API. Extracted helpers are imported back into
their original canonical `fvm`, `solver`, or `flux` module so existing lazy Roe
imports and callers continue to resolve. Compatibility modules contain no
duplicate numerical implementations.

For new code, use imports such as:

```python
from nozzle_hyperbolic.models.euler_nozzle.fvm import make_jax_stadtke_step
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.models.two_fluid.analytical import isentropic_ic, bernoulli_ic
from nozzle_hyperbolic.models.two_fluid.solver import make_two_fluid_step
from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
```

Existing function names and signatures remain valid. Function `__module__`
metadata now identifies the canonical home; code that inspects implementation
paths or monkeypatches another module's imported globals may need adjustment.
The existing cases/tests do not require such behavior. The legacy property
wrapper still has its original optional/environment-dependent `barotropy`
import; relocation does not resolve undeclared dependencies.

## Ownership for the next steps

Keep **closures, boundary conditions, and analytical solutions with the model that defines their
meaning**. A wall law can be shared later only after its assumptions, units,
friction-factor convention, and inputs are reconciled. Avoid a top-level bucket
for all physical closures or boundaries.

| Shared infrastructure | Model or method ownership |
|---|---|
| Mesh metrics and geometry sampling | Geometry quantities required by each equation set |
| Array reconstruction and limiter arithmetic | Reconstruction variables, admissibility and boundary stencils |
| Linear algebra, Newton iteration and convergence reporting | Residual definition, EOS constraints, scaling and projections |
| Jacobian coloring machinery | Proven dependency radius and model-specific nonlocal terms |
| Time integration and CFL control | Wave-speed evaluation and physical/pseudo-time interpretation |
| Thermodynamic adapters and supported state recovery | Phase choice, state representation, required derivatives |
| Logging, output containers and run orchestration | Derived physical fields and conservation diagnostics |

The eventual common layers can grow to `numerics/{linear,nonlinear,jacobians}`,
`solvers/finite_volume`, and `solvers/time_marching`. Extract them incrementally
after defining inputs/outputs and verifying equivalence. A shared operation
should accept model callbacks/data rather than import a concrete physical model.
Model equations and closures should eventually be independent of plotting,
configuration parsing, and time or space integration.

Preserve three distinct state representations:

| Model | Physical primitives | Balance storage | Current solver unknowns |
|---|---|---|---|
| EulerNozzle | $(p,u,h)$, with retained conservative paths | 3 mass/momentum/energy densities | 3 primitive or conservative variables, depending on path |
| TwoFluid | $(p,u_g,u_l,\alpha_g,h_g,h_l)$ | 6 partial mass/momentum/energy densities | 8 Newton variables in the current factory |
| EulerAnnular, planned | $(p,v_m,v_\theta,h)$ | 4 balances including swirl | To be designed and checked against the annular documents |

A future model contract can provide storage, physical flux, physical sources,
wave information, and admissibility. TwoFluid additionally requires its
nonconservative coupling and EOS-constraint interface; six physical variables
must not be confused with eight nonlinear unknowns.

## Alberto FVS and Andrea FDS review

Andrea's `fvm_solver/flux.py` dispatches `rusanov`, `roe`, `hll`, `hllc`, and `lf`
through `F_numeric`. Alberto already contains these implementations: the diff of
that file between the reviewed tips changes only module-docstring whitespace.
Alberto also retains the corresponding NumPy/JAX driver paths. Preserving them
is preservation of Alberto's source, not a new port from Andrea.

| Existing method | State/array contract | Flux weighting | Important specialization |
|---|---|---|---|
| NumPy FDS/Riemann methods in `flux.py` | Conservative states, `(3, Nfaces)` | Already multiplied by interface area | Separate perfect/general-EOS paths; Roe imports model matrices lazily |
| JAX Rusanov/HLL/HLLC/LF in `flux_jax.py` | Conservative states, `(3, Nfaces)` plus supplied thermodynamics | Area weighting inside current helpers | No JAX Roe implementation in this path |
| EulerNozzle Städtke FVS | $(p,u,h)$, `(Nfaces, 3)` | Unweighted flux; caller applies `A_face` | EOS derivatives and model-specific spectral splitting |
| TwoFluid Städtke FVS | Six primitives at each face, scalar-face calls batched by caller | Geometry handled by assembly | Virtual-mass/eigenstructure choices and nonconservative correction |

Current `time_method='fvs_newton'` selects Euler FVS, mixing the spatial-method
choice with time/nonlinear integration. The legacy `implicit` FDS option is a
diagonal damping update; it is not the same algorithm as Newton backward Euler.
The primitive/conservative representations, face orientation, and area weighting
must be reconciled explicitly before sharing an operator. Otherwise it is easy
to apply area twice or use an incompatible state dimension.

Use FDS as a broad organizational label when discussing Andrea's branch, while
retaining precise algorithm names: Roe splitting, HLL/HLLC approximate Riemann
solvers, and local/global Lax-Friedrichs variants are different constructions.
Generic HLL/Rusanov algebra can eventually consume fluxes and wave bounds;
general-EOS Roe averaging, HLLC star states, and Städtke projectors remain
capability- and model-dependent.

### Suggested future option schema, not implemented

```yaml
model: euler_nozzle
spatial:
  family: fvs
  flux: stadtke
  reconstruction:
    order: 2
    variables: primitive_ph
    limiter: van_leer
time:
  method: backward_euler
  purpose: steady_pseudotime
nonlinear:
  method: newton
  jacobian: colored
linear:
  method: banded
```

An FDS/Riemann selection would choose a named flux independently of reconstruction,
time integration, nonlinear method, and linear solver. Define a supported-options
matrix and reject unsupported combinations when constructing a solver. Do not
silently route a requested JAX method to NumPy, or offer TwoFluid/EulerAnnular
methods merely because an Euler method has the same flux label.

Boundary data belong to the physical problem. Reconstruction order and ghost
stencil requirements belong to its numerical adapter. A prescribed reservoir
and outlet pressure must therefore not be encoded as a universal boundary
operator applied to all models.

## JAX, JIT and differentiation target

All future hyperbolic production kernels should use pure JAX operations with
explicit array/PyTree inputs. Keep configuration parsing, file I/O, plotting,
and host orchestration outside compiled kernels. Resolve method choices at
construction and make differentiation with respect to fluid, geometry, closure,
and boundary parameters explicit where supported.

JIT compilation, residual differentiation, differentiation through fixed steps,
and differentiation of a converged solution are separate capabilities. Test each
claimed capability. Bisection, stopped gradients, dynamic Newton loops, clipping,
and nonlocal reconstruction currently limit them; see the JAX audit for concrete
reproductions. This refactor preserves those implementations and does not make
the existing NumPy routines JAX-compatible by changing their folder name.

Treat local time stepping as steady pseudo-time acceleration. A physical
transient needs a common physical timestep and consistent stage states/sources.
Verify dense versus colored Jacobians only for configurations whose dependency
stencils actually match the coloring; global limiter statistics can invalidate
that assumption.

## Later reuse by NozzleX

After this package is unified and validated, the same models, closures,
thermodynamics, and continuous geometry can be consumed by time marching and
space marching under NozzleX. Each model can expose an additional steady system
$M(q,x)\,dq/dx=b(q,x)$ where mathematically supported. Space-marching inlet data
and time-marching boundary-value data remain distinct specifications of the same
physics.

Do not assume that a spatial TwoFluid implementation exists just because another
package has a directory with that name. Reconcile the actual equations first.
EulerAnnular also remains design work: the retained documents in `docs/` are
inputs to that implementation, not an executable unsteady solver.

## Verification recorded for the initial restructuring

The following results describe the initial relocation from Alberto's tree,
before the model renaming and extraction of analytical routines from scripts.
They are retained as baseline evidence, not as a claim that those scripts are
still byte-for-byte unchanged after the subsequent extraction.

- Source comparison against a snapshot taken before relocation verifies all
  **153 original top-level functions/classes**, including nested routines,
  retain exactly the same source text.
- All **2,211 original Python comments** and **156 docstrings** are retained.
- The **305 pre-existing non-package files** captured in that snapshot remain
  byte-for-byte unchanged, including the initial local edits and Roberto files.
- Package source compilation succeeds. Legacy-import and representative numerical
  checks are recorded in the workspace [compatibility inventory](../files_that_should_still_run.md).
- All **32 finite numerical outputs** in focused before/after probes match
  exactly (maximum absolute difference 0.0), including five FDS flux/residual
  choices and six-cell JIT EulerNozzle/TwoFluid Newton steps with analytical
  ideal-gas mocks. These checks also cover reconstruction, wall friction, CFL,
  EulerNozzle face mass flux, and its dense/colored Jacobian comparison.
- Before and after relocation, `python -m pytest tests -q -p no:cacheprovider`
  reports **25 passed, 3 expected failures, 1 failure**. The failure is the same
  missing `T` entry in the existing `_IdealGas` test fixture. It is documented,
  not repaired. This is preservation of the baseline, not a clean numerical
  certification.
- Full research campaigns and table-dependent validation runs are not claimed
  to have been executed. The compatibility inventory distinguishes executed
  checks from retained entry points and externally dependent cases.
- Installed-wheel verification was not performed: this virtual environment has
  neither the package installed nor `poetry-core`/`build`. Checks use the source
  checkout. The packaging files were not changed by this refactor.
- Git's whitespace check flags original trailing whitespace in moved code and
  the retained annular documents. Those lines are preserved deliberately; this
  step does not reformat existing function bodies or documents.
