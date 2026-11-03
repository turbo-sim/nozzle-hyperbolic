# NICFD nitrogen–water nozzle

Run `poetry run python demos/two_fluid/run.py` from the repository root.
[run.py](run.py) contains all physical and numerical configuration; it does not
load YAML or import the larger NICFD case runner.

Gaseous nitrogen and liquid water enter the Elliot nozzle from a 20 bar,
295.15 K reservoir at gas mass-flow quality 0.5 and inlet slip one. Outlet
pressures are 16 bar, 12 bar and 0.986 bar. Interphase heat/drag is active and
mass transfer is disabled. The original number-density area correlation,
virtual mass, void filter and `wb` geometry pairing are explicit in the script.

Default: 100 cells, MUSCL Van Albada, model-owned Stadtke compatibility flux,
implicit Euler with at most twenty Newton corrections. JAXProp creates/reuses
nitrogen/water HEOS tables under `outputs/fluid_tables/`.

The default is an **explicit published-profile initial guess**, read from the
checked-in CSV for each regime. It supports a verified discrete-branch example;
it is not evidence of cold-start robustness. `INITIALIZATION = "cold"` selects
an independent 10 m/s inlet guess and retains any failure. The CSV is also
the independent comparison, not an exact solution of the current discretization.

Combined phase-flow/convergence plots, original solver histories, configured
closures/boundaries and final states are saved under `outputs/`.

Edit the grouped inputs at the top of run.py; the script takes no CLI arguments.
Figures use set_plot_options, 300-dpi PNG previews and vector PDFs. The JAX
cache report is one concise line. See the [shared guide](../README.md).
