# Demo examples

The `demos/` directory contains five standalone Python scripts. They construct
physical inputs, fluids, geometry, initial fields, boundary conditions and
solver settings directly in Python. Edit the grouped inputs near each script's
top, then run from the repository root. No YAML loader or command-line case
selection is required.

| Demo | Physical problem | Default spatial/time method |
|---|---|---|
| Euler nozzle | Subsonic, shocked and adapted air flow | 80 cells; MUSCL Van Leer, HLL, backward Euler |
| Annular Euler | Subsonic radial diffusion, internal shock and adapted expansion | 80 cells; MUSCL Van Leer, HLL, backward Euler |
| Two-fluid | NICFD nitrogen–water Elliot nozzle in three regimes | 100 cells; MUSCL Van Albada, model compatibility FVS, backward Euler |
| Shock tube | Diaphragm release and long-time wave washout | 100 cells; MUSCL Van Leer, HLL, SSPRK3 |
| Transient nozzle | Startup after an external-pressure drop | 100 cells; MUSCL Van Leer, HLL, SSPRK3 |

```bash
poetry run python demos/euler_nozzle/run.py
poetry run python demos/euler_annular/run.py
poetry run python demos/two_fluid/run.py
poetry run python demos/shock_tube/run.py
poetry run python demos/nozzle_transient/run.py
```

## Steady Euler nozzle

The {download}`nozzle script <../../demos/euler_nozzle/run.py>` constructs a
smooth converging-diverging nozzle with an air reservoir. Its three backpressure
regimes illustrate subsonic acceleration/deceleration, an internal normal shock
and adapted expansion. It compares the finite-volume result with independent
perfect-gas references and plots both flow fields and convergence diagnostics.
For a short initial run, reduce `NUMBER_OF_CELLS` and set
`CASES_TO_RUN = ("subsonic",)`.

This is a larger version of the [minimal example](getting_started/index.md).
The governing equations and reservoir/outlet laws are explained on the
[Euler nozzle page](theory/euler_nozzle.md).

## Annular Euler

The {download}`annular script <../../demos/euler_annular/run.py>` adds mean
radius, passage height, swirl and angular-momentum transport. The cases expose
how radial geometry and swirl influence pressure recovery, expansion and shock
location. Independent inviscid references provide comparisons within their
stated perfect-gas assumptions. The [annular theory page](theory/euler_annular.md)
explains the centrifugal source and the use of $\ell=r v_\theta$.

## Two-fluid nitrogen–water nozzle

The {download}`two-fluid script <../../demos/two_fluid/run.py>` constructs the
NICFD two-component Elliot-nozzle setup, with separate gas/liquid fluids and
heat/drag exchange. It compares subsonic, shocked and adapted profiles with
published CSV data retained in the repository.

The default `INITIALIZATION = "reference"` visibly interpolates the published
profile as a numerical initial guess. This is a seeded reproduction, not an
independent cold-start demonstration. `INITIALIZATION = "cold"` selects a
separate 10 m/s initial guess without a convergence guarantee. Property-table
construction and nonlinear solve costs can both be substantial. The
[two-fluid theory page](theory/two_fluid.md) explains its eight-state constrained
formulation and why its flux is explicitly model-specific.

## Shock-tube animation

The {download}`shock-tube script <../../demos/shock_tube/run.py>` starts with a
stationary diaphragm separating two perfect-gas states. It follows the
rarefaction, contact and shock, then continues until waves wash out through
approximate transmissive endpoints. An independent Riemann reference is an
exact comparison before boundary interaction. Afterward it is shown as an
unbounded-domain comparison rather than the finite-domain boundary solution.

The default movie uses 601 actual simulation states at 20 fps with uniform
physical-time spacing and no endpoint pauses, giving about 30 seconds of
constant-speed MP4/GIF playback. No intermediate states are invented by video
interpolation. Separate early-time comparison figures and arrays retain the
analytical reference and its boundary-interaction validity limit.

## Nozzle pressure-release animation

The {download}`transient-nozzle script <../../demos/nozzle_transient/run.py>`
follows startup after the downstream pressure drops. Its automatic Euler outlet
can move between pressure-driven, choked and fully outgoing regimes during the
transient. This avoids imposing the final steady Mach regime on every early
time. The movie allocates more frames to the first two acoustic transit times,
and includes initial/final pauses; playback time is intentionally nonuniform
relative to physical time. Titles retain the actual simulated time.

## Outputs and interpretation

Each script writes to its own ignored `outputs/` directory. Steady demos save
`flow.png/.pdf`, `convergence.png/.pdf`, per-case NPZ fields, JSON histories and
settings, and a summary. Figures use numerical markers and separate analytical
reference curves. Convergence plots separate spatial equations, global balance
and algebraic defects; rejected attempts remain recorded.

Transient demos save `transient.mp4`, history plots, selected frame figures,
`solution.npz` and JSON outcomes. The shock tube also saves a GIF. Movie export
requires FFmpeg on `PATH` or configured through `FFMPEG_BINARY`. An unfinished
run retains its actual reached state/time and reports failure.

The demo scripts configure a persistent JAX executable cache separately from
fluid-property tables. Keep their cache directories between runs when comparing
warm execution; disclose initialization and compilation when comparing timings.
Demonstration convergence does not establish mesh independence or validation of
all physical closures. See [convergence](numerics/convergence.md) for the checks
that define numerical success.
