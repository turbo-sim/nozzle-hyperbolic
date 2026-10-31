"""Repeat 50/100/200/400 interval grid sequencing with spatial RMS 1e-8 throughout.

These are 51/101/201/401 source points including two algebraic ghost slots.
The common runner times a fresh single-grid control and complete four-grid
sequence, with one excluded full solve plus ten measured solves per setup.
"""

from .run_two_grid import main

if __name__ == "__main__":
    main(
        intervals=(50, 100, 200, 400),
        folder="four_grid_tight",
        experiment="Four grids with equally tight endpoint residuals",
    )
