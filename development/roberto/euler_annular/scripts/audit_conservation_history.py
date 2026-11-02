"""Exercise study monitoring with prescribed arrays, without solver iterations.

Run as a module from the repo root. This checks bookkeeping and restoration,
not the numerical PDE implementation, which retains its existing verification.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .conservation_history import ConservationHistoryRecorder


def main():
    state = np.array([1., 2., 3., 4.])
    solver = SimpleNamespace(
        reference_density=2., reference_sound_speed=3.,
        mesh=SimpleNamespace(A_face=[5.], r_face=[7.]),
        step=lambda point, dt: (point + dt, "prescribed info"),
        face_flux=lambda point: np.array([point, 4*point]),
        source=lambda point: np.array([point, point]),
    )
    original = solver.step
    expected_scales = np.array([30., 90., 630., 270.])
    forwarded = []
    with ConservationHistoryRecorder(solver, forwarded.append) as recorder:
        np.testing.assert_array_equal(recorder.scales, expected_scales)
        candidate, info = solver.step(state, .25)
        np.testing.assert_array_equal(candidate, state + .25)
        assert info == "prescribed info"
        expected = candidate/expected_scales
        row = dict(step=1, accepted=True, conservation_max=max(abs(expected)))
        recorder(row)
        np.testing.assert_allclose(row["integrated_balance_scaled"], expected)
        assert forwarded[0] is row
        # A rejected attempt has no accepted callback and cannot append data.
        solver.step(state, .5)
        assert len(forwarded) == 1
    assert solver.step is original
    try:
        with ConservationHistoryRecorder(solver):
            raise RuntimeError("prescribed failure")
    except RuntimeError:
        pass
    assert solver.step is original
    report = dict(passed=True, numerical_iterations_run=0,
                  checks=["unchanged candidate/info", "fixed scales and four signed components",
                          "accepted callback forwarding", "rejected attempt not recorded",
                          "step restored on normal and exceptional exit"])
    output = Path(__file__).resolve().parents[1]/"outputs"/"html_report"/"recorder_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
