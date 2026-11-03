"""Transient pressure/sonic/outgoing outlet transitions through shared physics."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.equations import physical_flux
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_outlet import EulerOutlet
from nozzle_hyperbolic.solvers.boundaries import make_boundary_solver
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


class ShiftedPressureEOS:
    """Stiffened-pressure, shifted-enthalpy fixture with no public gas constants."""

    def __init__(self, offset):
        self._base = jxp.FluidPerfectGas(name="Air")
        self.offset = offset
        self.shift = -5e5
        self.h_min = self.shift + 2e4
        self.h_max = self.shift + 2e6

    def get_state(self, pair, first, second):
        if pair == jxp.PT_INPUTS:
            state = self._base.get_state(pair, first + self.offset, second)
        elif pair == jxp.HmassP_INPUTS:
            state = self._base.get_state(pair, first - self.shift, second + self.offset)
        else:
            raise AssertionError(
                "The transient acoustic law only needs PT setup and HP states"
            )
        result = {key: state[key] for key in ("p", "h", "rho", "a", "G", "s", "T")}
        result["p"] = result["p"] - self.offset
        result["h"] = result["h"] + self.shift
        return result


def _outlet(fluid, pressure, *, nvar=3, kind="automatic_outflow", mach=0.0):
    state = fluid.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    primitive = jnp.array([state["p"], mach * state["a"], state["h"]])
    scales = jnp.array([2e5, state["a"], state["a"] ** 2])
    if nvar == 4:
        primitive = jnp.array([primitive[0], primitive[1], 3.0, primitive[-1]])
        scales = jnp.array([2e5, state["a"], 70.0, state["a"] ** 2])
    outlet = EulerOutlet(
        fluid, pressure, scales, float(state["a"] ** 2 / 300), 0.5, kind=kind
    )
    full = jnp.tile(primitive, (6, 1))

    def residual(values, time):
        return jnp.stack(
            (
                (values[0] - values[1]) / scales,
                outlet.residual(values[-1], values[1:-1]),
            )
        )

    def admissible(values):
        return (
            jnp.all(jnp.isfinite(values))
            & jnp.all(values[:, 0] > 0)
            & jnp.all(values[:, -1] > fluid.h_min)
            & jnp.all(values[:, -1] < fluid.h_max)
        )

    solve = make_boundary_solver(
        residual, full.shape, scales, admissible=admissible, max_iterations=40
    )
    return outlet, full, solve


@pytest.mark.parametrize("offset", [0.0, 1e5])
@pytest.mark.parametrize("nvar", [3, 4])
def test_extreme_pressure_step_selects_reachable_eos_constraint_with_finite_ad(
    offset, nvar
):
    fluid = ShiftedPressureEOS(offset)
    outlet, old, solve = _outlet(fluid, 2e3, nvar=nvar)
    full, info = solve(old, 0.0)
    assert bool(info.converged & info.admissible), info
    np.testing.assert_array_equal(full[1:-1], old[1:-1])
    trace = full[-1]
    state = outlet.thermodynamics(trace)
    if offset == 0.0:
        np.testing.assert_allclose(trace[1], state["a"], rtol=2e-12)
        assert float(trace[0]) > 2e3
    else:
        # This stiffened-pressure EOS still has a large sound speed at low
        # actual pressure. The prescribed pressure remains subsonically
        # reachable, so imposing a perfect-gas critical ratio would be wrong.
        assert float(trace[0]) == pytest.approx(2e3)
        assert trace[1] < state["a"]
    np.testing.assert_allclose(
        state["s"], outlet.thermodynamics(old[-2])["s"], atol=3e-12
    )
    np.testing.assert_allclose(outlet.residual(trace, full[1:-1]), 0.0, atol=2e-12)
    assert bool(outlet.direct_flux(full[1:-1]))
    assert np.isfinite(
        jax.jacfwd(lambda value: outlet.residual(value, full[1:-1]))(trace)
    ).all()
    assert np.isfinite(
        jax.jacrev(lambda value: outlet.residual(value, full[1:-1]))(trace)
    ).all()


def test_mild_subsonic_pressure_step_matches_static_acoustic_trace():
    fluid = ShiftedPressureEOS(0.0)
    auto, old, complete = _outlet(fluid, 1.8e5)
    static, _, static_complete = _outlet(fluid, 1.8e5, kind="pressure")
    solved, info = complete(old, 0.0)
    reference, other = static_complete(old, 0.0)
    assert bool(info.converged & other.converged)
    np.testing.assert_allclose(solved, reference, rtol=2e-13, atol=2e-9)
    assert solved[-1, 0] == pytest.approx(1.8e5)
    assert solved[-1, 1] < auto.thermodynamics(solved[-1])["a"]
    assert bool(auto.direct_flux(solved[1:-1])) and not bool(
        static.direct_flux(solved[1:-1])
    )


def test_fully_outgoing_auto_acoustic_trace_is_independent_of_exterior_pressure():
    fluid = ShiftedPressureEOS(1e5)
    first, old, complete = _outlet(fluid, 2e3, mach=2.0)
    second, _, other_complete = _outlet(fluid, 1.9e5, mach=2.0)
    solved, info = complete(old, 0.0)
    other, other_info = other_complete(old, 0.0)
    assert bool(info.converged & other_info.converged)
    np.testing.assert_array_equal(solved, other)
    np.testing.assert_array_equal(solved[-1], first.extrapolate(old[1:-1]))
    assert not bool(first.direct_flux(solved[1:-1]))
    np.testing.assert_array_equal(
        first.residual(solved[-1], old[1:-1]), second.residual(solved[-1], old[1:-1])
    )


def _production_model(model_type, formulation="acoustic"):
    fluid = jxp.FluidPerfectGas(name="Air")
    state = fluid.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    if model_type == "nozzle":
        native = None
        mesh = ConstantAreaGeometry(length=0.1, area=0.003).build_mesh(4)
        old = jnp.tile(jnp.asarray([2e5, 0.0, state["h"]]), (6, 1))
        model = NozzleModel(
            mesh,
            fluid,
            p0=2e5,
            T0=300.0,
            back_pressure=2e3,
            outlet_regime="auto",
            outlet_formulation=formulation,
            initial_state=old,
        )
    else:
        mesh, native = build_annular_mesh(conical_channel(0.1, 0.2, 0.005, 0.0), 4)
        old = jnp.tile(jnp.asarray([2e5, 0.0, 0.0, state["h"]]), (6, 1))
        model = AnnularModel(
            mesh,
            native,
            fluid,
            inlet_data=dict(p0=2e5, T0=300.0, alpha_in=0.0),
            outlet_pressure=2e3,
            outlet_regime="auto",
            outlet_formulation=formulation,
            initial_state=old,
        )
    return model, old, native


@pytest.mark.parametrize("model_type", ["nozzle", "annular"])
@pytest.mark.parametrize("formulation", ["acoustic", "finite_wave"])
def test_model_auto_outlet_completes_stagnant_trace_and_applies_physical_flux(
    model_type, formulation
):
    model, old, native = _production_model(model_type, formulation)
    mesh, fluid = model.mesh, model.fluid
    operator = Operator(model, Scheme())
    solved, info = operator.complete_boundaries(old, 0.0)
    assert bool(info.converged & info.admissible), info
    assert bool(operator.is_admissible(solved))
    assert model.solution_audit(solved)
    np.testing.assert_array_equal(solved[1:-1], old[1:-1])
    sound = model.outlet.thermodynamics(solved[-1])["a"]
    np.testing.assert_allclose(solved[-1, 1], sound, rtol=2e-12)
    assert float(solved[-1, 0]) > 2e3
    expected = (
        model.face_physical_flux(solved[-1])
        if model_type == "nozzle"
        else physical_flux(solved[-1], native.r_face[-1], fluid)
    )
    np.testing.assert_allclose(
        operator.face_flux(solved)[-1], mesh.area_faces[-1] * expected, rtol=2e-13
    )


@pytest.mark.parametrize("model_type", ["nozzle", "annular"])
def test_transient_signed_startup_audit_is_distinct_from_steady_forward_audit(
    model_type,
):
    model, rest, _ = _production_model(model_type)
    sound = float(model.outlet.thermodynamics(rest[0])["a"])
    signed = (
        rest.at[0, 1]
        .set(-0.01 * sound)
        .at[2, 1]
        .set(-0.01 * sound)
        .at[-1, 1]
        .set(0.1 * sound)
    )
    assert model.transient_solution_audit(signed)
    assert not model.solution_audit(signed)
    # A supersonic reservoir trace changes the characteristic count. An
    # actual pressure-outlet inflow needs exterior thermodynamic data.
    assert not model.transient_solution_audit(signed.at[0, 1].set(-2 * sound))
    assert not model.transient_solution_audit(signed.at[-1, 1].set(-0.01 * sound))
    assert not model.transient_solution_audit(signed.at[2, 0].set(-1.0))
