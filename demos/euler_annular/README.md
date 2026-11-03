# Steady annular Euler flow with swirl

Run `poetry run python demos/euler_annular/run.py` from the repository root.
[run.py](run.py) builds the radial geometry, mesh, fluid, inlet and outlet.

The channel radius increases from 0.1 to 0.2 m at a 0.005 m gap. Inlet angle is
15 degrees, converted to radians. Iteration variables are $(p,v_m,\ell,h)$ with
$\ell=r v_\theta$. The subsonic case uses a reservoir inlet. Shocked and adapted
expansion use a prescribed supersonic inlet; the diverging channel is not a
converging-diverging choking nozzle. The adapted exit is fully outgoing.

Default: 80 cells, MUSCL Van Leer/HLL, implicit Euler. Independent analytical
profiles determine boundary data and comparison curves; initial fields are
model-owned cold guesses. Combined flow/convergence plots and raw histories
are saved under `outputs/`.

Edit the grouped inputs at the top of run.py; the script takes no CLI arguments.
Figures use set_plot_options, 300-dpi PNG previews and vector PDFs. The JAX
cache report is one concise line. See the [shared guide](../README.md).
