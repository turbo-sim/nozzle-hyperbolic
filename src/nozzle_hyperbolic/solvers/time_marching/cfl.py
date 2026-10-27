"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np


# ---------------------------------------------------------------------------
# CFL time step
# ---------------------------------------------------------------------------

def _compute_dt(u, c, dx, CFL):
    max_speed = np.max(np.abs(u) + c)
    return CFL * dx / max_speed if max_speed > 0.0 else CFL * dx
