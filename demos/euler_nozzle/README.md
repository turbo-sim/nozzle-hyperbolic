# Steady Euler nozzle

Run `poetry run python demos/euler_nozzle/run.py` from the repository root.
All three regimes are constructed in [run.py](run.py), with physical parameters
and method alternatives explained beside the relevant calls.

The 2 bar, 300 K reservoir feeds a 0.2 m smooth nozzle with inlet/throat area
ratio two. The analytical perfect-gas reference selects pressures and assesses
the result; it never supplies the initial profile. The adapted outlet is fully
outgoing: design backpressure is not an extra imposed supersonic condition.

Default: 80 cells, MUSCL Van Leer/HLL, implicit Euler. Outputs combine the three
regimes under `outputs/`. For a smaller run, set NUMBER_OF_CELLS = 40 and CASES_TO_RUN = ("subsonic",) near the top of run.py.
Convergence and actual regime selection should be assessed separately.

Edit the grouped inputs at the top of run.py; the script takes no CLI arguments.
Figures use set_plot_options, 300-dpi PNG previews and vector PDFs. The JAX
cache report is one concise line. See the [shared guide](../README.md).
