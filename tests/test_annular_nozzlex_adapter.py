"""Optional NozzleX adapter: all available policies, JIT and source stencil."""

import pytest
pytest.importorskip("nozzlex")
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.annular import radial_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.closures import NozzleXWall
from nozzle_hyperbolic.solvers.fvm import Operator,Scheme


@pytest.fixture(scope="module")
def setup():
    fluid=jxp.FluidPerfectGas("Air")
    mesh,native=build_annular_mesh(radial_channel(.1,.2,.015),4)
    return fluid,mesh,native


@pytest.mark.parametrize("friction",[
    dict(type="zero_friction"),dict(type="constant_friction_factor",Cf=.005),
    dict(type="aungier",roughness=1e-6,Re_transition=2300,Re_width=500)])
@pytest.mark.parametrize("heat",[dict(type="adiabatic"),dict(type="reynolds_analogy",T_wall=260)])
def test_imported_laws_have_finite_local_AD_and_correct_force_torque_heat(setup,friction,heat):
    fluid,mesh,native=setup
    wall=NozzleXWall(native,dict(friction_model=friction,heat_model=heat),alpha_in_deg=60)
    model=AnnularModel(mesh,native,fluid,inlet_data=dict(p0=101325,T0=300,alpha_in=np.pi/3),
                       outlet_pressure=95000,wall_closure=wall)
    state=model.initial_state().at[:,1].set(70).at[:,2].set(.1*120)
    loads=wall.fields(state[1:-1],model._r,fluid)
    source=jax.jit(model.integrated_source)(state)
    measure=2*mesh.volumes/native.b
    # Angular torque contains the radius; heating never includes shear work.
    np.testing.assert_allclose(source[:,2],-model._r*measure*loads["tau_t"],rtol=2e-14)
    np.testing.assert_allclose(source[:,3],measure*loads["q_w"],rtol=2e-14)
    assert np.isfinite(jax.jacfwd(model.integrated_source)(state)).all()
    assert np.all(np.asarray(loads["tau_m"])>=0)
    if heat["type"]=="reynolds_analogy" and friction["type"]!="zero_friction":
        assert np.all(np.asarray(loads["q_w"])<0)


def test_imported_source_keeps_colored_jacobian_equivalent_to_dense(setup):
    fluid,mesh,native=setup
    options=dict(friction_model=dict(type="aungier",roughness=1e-6,Re_transition=2300,Re_width=500),
                 heat_model=dict(type="reynolds_analogy",T_wall=260))
    wall=NozzleXWall(native,options,alpha_in_deg=60)
    model=AnnularModel(mesh,native,fluid,inlet_data=dict(p0=101325,T0=300,alpha_in=np.pi/3),
                      outlet_pressure=95000,wall_closure=wall)
    operator=Operator(model,Scheme(flux="hll",reconstruction="muscl_van_leer"))
    state=model.initial_state().at[:,1].set(jnp.linspace(80,40,6))
    colored=np.asarray(operator.jacobian(state.ravel(),state.ravel(),1e-5,0.0))
    dense=np.asarray(operator.dense_jacobian(state.ravel(),state.ravel(),1e-5,0.0))
    scale=np.tile(model.variable_scales,6)
    np.testing.assert_allclose(colored*scale,dense*scale,rtol=3e-10,atol=3e-10)
