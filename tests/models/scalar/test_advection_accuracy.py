"""Scalar advection obeys physical wave direction and improves with refinement."""

import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

from tests._support.generic_models import _UserAdvection


@pytest.mark.parametrize("speed", [-0.7, 0.7])
@pytest.mark.parametrize("flux", ["hll", "rusanov"])
@pytest.mark.parametrize(
    "integrator", ["euler-implicit", "euler-explicit", "runge-kutta-3"]
)
def test_signed_user_advection_all_integrators_without_eigenstructure(
    speed, flux, integrator
):
    errors = []
    for count in (16, 32):
        model = _UserAdvection(speed, count)
        (final,) = solve_transient(
            model, (0.03,), Scheme(flux=flux, time_integrator=integrator), cfl=0.4
        )
        assert final.success, final.reason
        error = np.abs(
            final.state[1:-1, 0] - model.exact(model.mesh.centers, 0.03, speed)
        )
        errors.append(np.dot(model.mesh.volumes, error))
        assert np.max(np.abs(final.metrics["conservation_budget"])) < 1e-12
        assert np.max(np.abs(model.boundary_residual(final.state, final.time))) < 1e-12
    # First-order face reconstruction must converge toward the translated wave.
    assert errors[1] < 0.7 * errors[0], errors
