# Standalone physical-model demos

Each folder contains one commented Python script. Edit the grouped inputs near
the top, then run it from the repository root:

???powershell
poetry run python demos/euler_nozzle/run.py
poetry run python demos/euler_annular/run.py
poetry run python demos/two_fluid/run.py
poetry run python demos/shock_tube/run.py
poetry run python demos/nozzle_transient/run.py
???

The existing environment can also be used directly, for example
?.venv/Scripts/python.exe -B demos/shock_tube/run.py?.

| Folder | Physical problem | Default method |
|---|---|---|
| [euler_nozzle](euler_nozzle/run.py) | Subsonic, shocked and adapted air nozzle flow | MUSCL Van Leer / HLL / backward Euler; 80 cells |
| [euler_annular](euler_annular/run.py) | Subsonic diffusion, internal shock and adapted radial expansion | MUSCL Van Leer / HLL / backward Euler; 80 cells |
| [two_fluid](two_fluid/run.py) | NICFD nitrogen?water Elliot nozzle, three regimes | MUSCL Van Albada / model-owned Stadtke flux / backward Euler; 100 cells |
| [shock_tube](shock_tube/run.py) | Diaphragm release, analytical comparison and long-time washout | MUSCL Van Leer / HLL / RK3; 100 cells |
| [nozzle_transient](nozzle_transient/run.py) | Nozzle start-up after an external-pressure drop | MUSCL Van Leer / HLL / RK3; 100 cells |

The scripts construct fluids, geometry, initial states, boundaries and solvers
in Python. They use ordinary configuration variables, with no CLI arguments
or YAML loaders. Method alternatives are explained beside the scheme.

## Editing inputs

For a smaller steady example, change:

???python
NUMBER_OF_CELLS = 40
CASES_TO_RUN = ("subsonic",)
???

Transient controls include ?END_TIME?, ?NUMBER_OF_FRAMES?, ?VIDEO_FPS?,
?INITIAL_HOLD_SECONDS? and ?FINAL_HOLD_SECONDS?. ?END_TIME = None? selects the
physical default. Units and the role of each physical input are explained
beside its definition.

## Publication figures and saved records

Outputs go to each demo's ignored ?outputs/? directory. Figures use
?set_plot_options? with consistent serif/STIX fonts, labelled units, panel
labels, light grids and legends outside the data. Static figures are saved
as 300-dpi PNGs and vector PDFs with embedded fonts.

Steady examples save combined ?flow.png/.pdf? and ?convergence.png/.pdf?,
per-case NPZ fields and JSON histories/settings, and ?summary.json?. Circles
denote finite-volume results; dashed curves denote independent references.
Two-fluid liquid curves have square markers and separate line styles.

Convergence panels separate original spatial residuals, global balance and
algebraic defects. Bottom crosses mark rejected attempts without inventing
residual values. Exact zero defects stay zero in saved data and are omitted
from logarithmic curves.

Transients save `transient.mp4`, `history.png/.pdf`, selected `frame_*.png/.pdf`,
`solution.npz` and JSON histories/outcomes. The shock-tube demo also exports
`transient.gif` with the same physical-time spacing. Failed runs retain the actual
reached time and state, then raise an error.

## Playback timing and analytical reference

The defaults record **601 actual simulation states** at **20 fps**. The shock
tube uses **uniform physical-time intervals** and no endpoint pauses, producing
about **30 s** of constant-speed MP4/GIF playback. Titles show actual physical
time; no intermediate flow states are invented by video interpolation.

The shock-tube movie includes an independent perfect-gas Riemann reference throughout.
Before interaction it is an exact comparison. Afterward it is grey and labelled
as an **unbounded-domain comparison**, since approximate transmissive boundaries
need not reproduce that infinite-domain solution after interaction.

The right shock reaches a boundary first for this preset, at about **0.90 ms**.
?reference_comparison.png/.pdf? compares pressure, density and velocity at three
earlier times. Analytical arrays and their validity limit are in ?reference.npz?.
The simulation still runs to about **0.06685 s**, three estimated fan-tail exit
times, so the resolved waves leave. Initial velocities remain zero.

The nozzle movie allocates 60% of its moving frames to the first two acoustic
transit times, making the pressure release easier to follow. It retains a
two-second initial hold and three-second final hold, giving about **35 s** total.

## Reusing JAX executables

The scripts configure persistent caching before creating JAX arrays. Keep
?outputs/jax_cache/? between runs. ?JAX_COMPILATION_CACHE_DIR? can select another
location; ?JAX_ENABLE_COMPILATION_CACHE=false? is respected.

Normal execution prints **one JAX status line per script**: compiled, loaded
from disk, or mixed, plus the observed compile/load time. Counts and times are
saved in ?compilation.json?. Leave ?VERBOSE_JAX = False? for normal use; true
enables kernel-level debugging.

Steady scripts retain a ?SteadyWorkspace?; transients retain an ?Operator?.
Reuse the same fixed model/scheme binding within a process. Across processes,
eligible executables load from disk. Actual cache hits are observed; existing
files alone are not treated as proof of reuse.

Times cover backend compilation/cache handling, rather than pure compiler CPU
time. Python tracing/lowering, property preparation, solving and rendering have
different costs. Changed shapes, precision, constants, methods, libraries or
hardware can require new builds. See
[JAX's cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

## Nitrogen?water initialization

The two-fluid demo uses 20 bar and 295.15 K reservoirs, gas mass-flow quality
0.5, inlet slip one, and backpressures 16, 12 and 0.986 bar. Drag and heat
transfer are active; mass transfer is off. Both phases use real-fluid HEOS
tables, cached separately under ?outputs/fluid_tables/?.

?INITIALIZATION = "reference"? visibly interpolates a published CSV as the
initial guess. Model settings are defined in the script.
?INITIALIZATION = "cold"? selects an independent 10 m/s guess, without a
cold-start convergence guarantee. Attribution and provenance are in the
[NICFD guide](../cases/NICFD2026/README.md).

Property tables, executable caches and flow initialization are separate.
Loading a kernel does not load a previous flow solution.

## MP4 prerequisite

Use FFmpeg on PATH or set ?FFMPEG_BINARY?:

???powershell
$env:FFMPEG_BINARY = 'C:/tools/ffmpeg/bin/ffmpeg.exe'
poetry run python demos/shock_tube/run.py
???

A portable executable at the demo's ?outputs/tools/ffmpeg.exe? is also recognized.
Videos use H.264/yuv420p at 1600 by 1000 pixels.

Measured results and verification scope are in [VALIDATION.md](VALIDATION.md).
