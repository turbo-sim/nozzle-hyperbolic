"""Production model contracts, thermodynamic reference shifts and wall budgets."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import radial_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import (
    ConstantAreaGeometry,
    SymmetricNozzleGeometry,
)
from nozzle_hyperbolic.models.euler_nozzle.closures import HaalandWall, PrescribedWall
from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.models.scalar import AdvectionModel, BurgersModel


class ShiftedHPFluid:
    """Hide backend constants and unsupported recovery pairs from physical code."""

    def __init__(self, shift):
        self.shift = shift
        self._backend = jxp.FluidPerfectGas(name="Air")

    def get_state(self, pair, first, second):
        if pair not in (jxp.PT_INPUTS, jxp.HmassP_INPUTS):
            raise NotImplementedError(
                "Only PT setup and HP numerical calls are supported"
            )
        first = first - self.shift if pair == jxp.HmassP_INPUTS else first
        state = self._backend.get_state(pair, first, second)
        result = {
            key: state[key] for key in ("p", "h", "rho", "T", "s", "a", "G", "mu")
        }
        result["h"] = result["h"] + self.shift
        return result


@pytest.mark.parametrize("shift", [0.0, -5e5])
def test_nozzle_hp_only_storage_characteristics_and_reference_shift(shift):
    fluid = ShiftedHPFluid(shift)
    mesh = SymmetricNozzleGeometry().build_mesh(6)
    model = NozzleModel(mesh, fluid, back_pressure=1.7e5)
    initial = model.initial_state()
    assert bool(model.cell_admissible(initial[1:-1]))
    if shift < 0:
        assert np.all(np.asarray(initial[:, 2]) < 0)
    face = initial[:-1]
    projectors, eigenvalues, _ = model.face_characteristics(face, face, 0.1)
    scale = model.conservative_scales
    np.testing.assert_allclose(
        np.sum(projectors, axis=-3) * scale[None, :] / scale[:, None],
        np.broadcast_to(np.eye(3), (7, 3, 3)),
        atol=3e-13,
    )
    # Independent AD of F(W) and Q(W) must give the same conservative eigenspaces.
    state = initial[2]
    storage_jac = jax.jacfwd(model.face_conserved)(state)
    flux_jac = jax.jacfwd(model.face_physical_flux)(state)
    matrix = flux_jac @ jnp.linalg.inv(storage_jac)
    projectors, speeds, _ = model.face_characteristics(state, state, 0.1)
    for projector, speed in zip(projectors, speeds):
        defect = (
            (matrix @ projector - speed * projector) * scale[None, :] / scale[:, None]
        )
        np.testing.assert_allclose(defect, 0.0, atol=4e-12)


def test_nozzle_rest_balance_and_prescribed_wall_energy_budget():
    fluid = ShiftedHPFluid(-4e5)
    mesh = SymmetricNozzleGeometry().build_mesh(
        faces=np.array([0.0, 0.017, 0.073, 0.12, 0.2])
    )
    rest = NozzleModel(mesh, fluid, back_pressure=2e5)
    full = rest.initial_state()
    pressure_flux = full[0, 0] * np.asarray(mesh.area_faces)
    source = rest.integrated_source(full)
    np.testing.assert_allclose(np.diff(pressure_flux), source[:, 1], atol=2e-14)
    wall = NozzleModel(
        mesh, fluid, back_pressure=2e5, wall_closure=PrescribedWall(tau_w=3.0, q_w=7.0)
    )
    moving = full.at[:, 1].set(-12.0)
    difference = wall.integrated_source(moving) - rest.integrated_source(moving)
    measure = 2 * np.sqrt(np.pi * np.asarray(mesh.area_centers)) * mesh.widths
    np.testing.assert_allclose(difference[:, 1], 3.0 * measure, atol=2e-14)
    np.testing.assert_allclose(difference[:, 2], 7.0 * measure, atol=2e-14)
    np.testing.assert_array_equal(difference[:, 0], 0.0)
    assert bool(wall.cell_admissible(moving[1:-1]))


def test_nozzle_haaland_zero_speed_derivative_and_source_timestep():
    fluid = ShiftedHPFluid(0.0)
    mesh = ConstantAreaGeometry(length=0.2, area=1e-4).build_mesh(4)
    closure = HaalandWall(hydraulic_diameter=0.01)
    model = NozzleModel(mesh, fluid, back_pressure=2e5, wall_closure=closure)
    full = model.initial_state().at[:, 1].set(0.0)
    derivative = jax.jacfwd(
        lambda speed: model.integrated_source(full.at[:, 1].set(speed))[:, 1]
    )(0.0)
    state = model.thermodynamic_state(full[1:-1])
    expected = -model._wall_measure * 8 * state["mu"] / 0.01
    np.testing.assert_allclose(derivative, expected, rtol=2e-14)
    cap = float(model.explicit_timestep(full))
    assert np.isfinite(cap) and cap > 0


def _annular(fluid, **options):
    geometry = radial_channel(0.1, 0.2, 0.005)
    mesh, native = build_annular_mesh(geometry, 5)
    return AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=dict(p0=2e5, T0=300.0, alpha_in=0.2),
        outlet_pressure=1.7e5,
        **options
    )


def test_annular_transport_frame_wall_torque_and_heating():
    fluid = ShiftedHPFluid(-5e5)
    baseline = _annular(fluid)
    wall = _annular(fluid, tau_w=3.0, q_w=7.0)
    full = baseline.initial_state().at[:, 1].set(-12.0)
    assert bool(wall.cell_admissible(full[1:-1]))
    delta = wall.integrated_source(full) - baseline.integrated_source(full)
    vm = np.asarray(full[1:-1, 1])
    theta = np.asarray(full[1:-1, 2] / baseline._r)
    measure = 2 * np.asarray(baseline.mesh.volumes) / np.asarray(baseline._height)
    speed = np.sqrt(vm**2 + theta**2)
    np.testing.assert_allclose(delta[:, 1], -measure * 3 * vm / speed, atol=2e-14)
    np.testing.assert_allclose(
        delta[:, 2], -baseline._r * measure * 3 * theta / speed, atol=2e-14
    )
    np.testing.assert_allclose(delta[:, 3], measure * 7, atol=2e-14)
    face = full[:-1]
    transported = baseline.flux_to_conserved_frame(baseline.face_physical_flux(face))
    np.testing.assert_allclose(
        transported[:, 2], transported[:, 0] * face[:, 2], rtol=2e-14
    )


@pytest.mark.parametrize("model_name", ["nozzle", "annular"])
def test_user_initial_state_and_full_filter_contract(model_name):
    fluid = ShiftedHPFluid(0.0)
    if model_name == "nozzle":
        mesh = ConstantAreaGeometry().build_mesh(5)
        maker = lambda **kw: NozzleModel(mesh, fluid, back_pressure=1.7e5, **kw)
    else:
        maker = lambda **kw: _annular(fluid, **kw)
    baseline = maker()
    values = baseline.initial_state()
    model = maker(
        initial_state=lambda points: values,
        filter_velocity_cutoff=5.0,
        filter_velocity_width=1.0,
    )
    np.testing.assert_array_equal(model.initial_state(), values)
    assert model.filter_conserved(values).shape == values.shape
    signal = model.filter_signal_speed(values)
    assert signal.shape == (model.mesh.n_cells + 1,)
    assert np.all(np.isfinite(signal) & (signal >= 0))
    with pytest.raises(ValueError, match="full"):
        maker(initial_state=values[1:-1]).initial_state()


@pytest.mark.parametrize("factory", [AdvectionModel, BurgersModel])
def test_scalar_model_rejects_silent_cell_count_truncation(factory):
    with pytest.raises(ValueError, match="integer"):
        factory(3.5)
