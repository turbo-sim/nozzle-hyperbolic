"""Optional study-only monitoring of each accepted steady conservation balance.

Use ``with ConservationHistoryRecorder(solver, progress) as monitor:`` and pass
``callback=monitor`` to ``solver.solve_steady``. This reuses the production step,
flux and source operators without changing their equations or acceptance rules.
Monitoring adds flux/source evaluations; include that cost in future timings.
"""

import numpy as np


class ConservationHistoryRecorder:
    """Record signed, fixed-scale global defects in existing accepted rows.

    Ordering: mass, meridional momentum, angular momentum, energy. Momentum is
    a balance including geometric/centrifugal sources, not a constant flux.
    The solver instance must not be used concurrently inside this context.
    """

    def __init__(self, solver, progress=None):
        self.solver = solver
        self.progress = progress
        density, sound = solver.reference_density, solver.reference_sound_speed
        mesh = solver.mesh
        self.scales = float(mesh.A_face[0])*sound*np.asarray([
            density, density*sound, density*float(mesh.r_face[0])*sound,
            density*sound**2])
        self.candidate = None

    def __enter__(self):
        self.original_step = self.solver.step

        def monitored_step(state, dt):
            candidate, info = self.original_step(state, dt)
            self.candidate = candidate
            return candidate, info

        self.solver.step = monitored_step
        return self

    def __call__(self, row):
        # solve_steady calls its callback only after accepting this candidate.
        flux = np.asarray(self.solver.face_flux(self.candidate))
        source = np.asarray(self.solver.source(self.candidate))
        balance = flux[-1]-flux[0]-np.sum(source, axis=0)
        scaled = balance/self.scales
        if not np.all(np.isfinite(scaled)):
            raise ValueError("Nonfinite per-equation conservation monitor")
        if not np.isclose(np.max(np.abs(scaled)), row["conservation_max"],
                          rtol=1e-10, atol=1e-14):
            raise ValueError("Conservation monitor does not match solver audit")
        row["integrated_balance_scaled"] = scaled.tolist()
        if self.progress is not None:
            self.progress(row)

    def __exit__(self, *exc):
        self.solver.step = self.original_step
        self.candidate = None
