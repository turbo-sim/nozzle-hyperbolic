"""Independent identities and small contract checks for the frozen Alberto port."""

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry, SymmetricNozzleGeometry
from nozzle_hyperbolic.models.two_fluid import (
    TwoFluidModel, BoundaryConfig, ClosureConfig, CompatibilityConfig,
    StabilizationConfig, conserved, primitives, from_primitives,
    PhasePinConfig,
)
from nozzle_hyperbolic.models.two_fluid._alberto_matrices import (
    phase_props, common_quantities, J_matrix, K_matrix, G_h_matrix, X_matrix, flux_vector,
)
from nozzle_hyperbolic.models.two_fluid._alberto_sources import (
    TwoFluidClosure, D_VE_algebric, D_VE_algebric_two_component, D_VE_conservative, D_VE_area,
    two_fluid_closure,
)
from nozzle_hyperbolic.models.two_fluid._alberto_flux import compatible_flux_solve, nonconservative_correction

jax.config.update("jax_enable_x64", True)


def model(*, velocity=25., alpha=0.4, mesh=None, **kwargs):
    fluid = jxp.FluidPerfectGas(name="Air")
    return TwoFluidModel(mesh or ConstantAreaGeometry().build_mesh(3), fluid, fluid,
                         initial_primitives=[2e5, velocity, velocity, alpha, 3e5, 2.8e5], **kwargs)


def physical_data(U):
    fluid = jxp.FluidPerfectGas(name="Air")
    eg, el = phase_props(fluid, U[0], U[4]), phase_props(fluid, U[0], U[5])
    cq = common_quantities(U, eg, el, 0.5)
    return fluid, eg, el, cq


def raw_flux(m, W):
    return m.numerical_face_flux(W[:-1], W[1:], SimpleNamespace(flux="model_compatibility"))


def test_eight_state_storage_and_intrinsic_eos_constraints():
    m = model(alpha=1e-8)
    W = m.initial_state()
    np.testing.assert_allclose(m.cell_constraints(W), 0., atol=1e-14)
    np.testing.assert_allclose(from_primitives(primitives(W), m.fluid_g, m.fluid_l), W, rtol=2e-15)
    # Hold h fixed while perturbing intrinsic gas density by 1%; its residual
    # stays O(rho), independent of the tiny gas volume fraction.
    altered = W.at[:, 2].multiply(1.01)
    altered = altered.at[:, 6].set(altered[:, 2]*primitives(W)[:, 4]-W[:, 0]*W[:, 1])
    np.testing.assert_allclose(m.cell_constraints(altered)[:, 0]/m.constraint_scales[0], 0.01, rtol=1e-12)
    assert conserved(W).shape == (5, 6)


def test_j_k_x_differential_identities():
    U = jnp.array((2e5, 45., 21., 0.4, 3e5, 2.8e5))
    fluid, eg, el, cq = physical_data(U)
    J = jax.jacfwd(lambda u: conserved(from_primitives(u, fluid, fluid)))(U)
    K = jax.jacfwd(lambda u: flux_vector(u, phase_props(fluid, u[0], u[4]), phase_props(fluid, u[0], u[5])))(U)
    np.testing.assert_allclose(J_matrix(U, eg, el), J, rtol=3e-13, atol=1e-10)
    np.testing.assert_allclose(K_matrix(U, eg, el), K, rtol=3e-13, atol=1e-10)
    np.testing.assert_allclose(X_matrix(U, eg, el, cq), J@G_h_matrix(U, eg, el, cq)-K, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("velocity", (0., 25., -25.))
def test_rest_forward_reversal_constant_state_and_finite_ad(velocity):
    m = model(velocity=velocity)
    W = m.initial_state()
    F = raw_flux(m, W)
    np.testing.assert_allclose(F, m.face_physical_flux(W[:-1]), rtol=2e-13, atol=1e-9)
    np.testing.assert_allclose(m.integrated_nonconservative(W, F), 0., atol=1e-10)
    derivative = jax.jacfwd(lambda left: m.numerical_face_flux(left, W[1:], SimpleNamespace(flux="model_compatibility")))(W[:-1])
    assert np.isfinite(np.asarray(derivative)).all()


def test_singular_k_requires_consistent_rhs_and_unique_nc_action():
    U = jnp.array((2e5, 0., 0., 0.4, 3e5, 2.8e5))
    _, eg, el, cq = physical_data(U)
    K, X = K_matrix(U, eg, el), X_matrix(U, eg, el, cq)
    direction = jnp.array((100., 2., -1., 0.01, 35., -28.))
    rhs = K@direction
    recovered = compatible_flux_solve(K, rhs, U)
    np.testing.assert_allclose(K@recovered, rhs, atol=1e-9)
    np.testing.assert_allclose(X@recovered, X@direction, atol=1e-9)
    assert np.isnan(np.asarray(compatible_flux_solve(K, rhs.at[4].add(1e4), U))).all()


@pytest.mark.parametrize("nonuniform", (False, True))
def test_rest_geometry_balance_and_stationary_void_contact(nonuniform):
    faces = np.linspace(0., 0.2, 5)
    if nonuniform:
        faces = 0.2*(faces/0.2)**1.4
    mesh = SymmetricNozzleGeometry().build_mesh(faces=faces)
    m = model(velocity=0., mesh=mesh, compatibility=CompatibilityConfig(nc_form="wb"))
    U = primitives(m.initial_state()).at[:, 3].set(jnp.array((0.2, 0.2, 0.4, 0.6, 0.8, 0.8)))
    W = from_primitives(U, m.fluid_g, m.fluid_l)
    F = raw_flux(m, W)
    residual = mesh.area_faces[1:, None]*F[1:]-mesh.area_faces[:-1, None]*F[:-1]+m.integrated_nonconservative(W, F)-m.integrated_source(W)
    np.testing.assert_allclose(residual, 0., atol=1e-10)


def test_exchange_sources_match_primitive_transform_and_pressure_work():
    U = jnp.array((2e5, 45., 21., 0.4, 3e5, 2.8e5))
    _, eg, el, cq = physical_data(U)
    closure = TwoFluidClosure(0.3, -2e4, -4e4, -100., 0., 0., 0., 0., U[2], 31., 5e5, 3e5, 0.)
    source = D_VE_algebric(U, cq, closure, eg, el, 0.5)
    transformed = D_VE_conservative(U, eg, el, cq, closure, 0.5)
    np.testing.assert_allclose(source, transformed, rtol=1e-12, atol=1e-9)
    assert abs(float(source[0]+source[1])) < 1e-14
    assert abs(float(source[2]+source[3])) < 1e-14
    area = D_VE_area(U, cq, SimpleNamespace(dlnA_dx=0.01), eg, el)
    assert float(area[4]+area[5]) == 0.
    no_mass = closure._replace(sigma_Mg=0., sigma_Ql=2e4)
    np.testing.assert_allclose(D_VE_algebric(U, cq, no_mass, eg, el, 0.5),
                               D_VE_algebric_two_component(U, cq, no_mass, eg, el, 0.5), atol=1e-12)


def test_stagnation_trace_has_quality_slip_enthalpy_entropy_and_eos():
    m0 = model()
    U = primitives(m0.initial_state())[0]
    W = m0.initial_state()[0]
    quality = float(W[2]*U[1]/(W[2]*U[1]+W[4]*U[2]))
    entropy = float(m0.fluid_g.get_state(jxp.HmassP_INPUTS, U[4], U[0])["s"])
    bc = BoundaryConfig(inlet="stagnation", H_g0=float(U[4]+U[1]**2/2), H_l0=float(U[5]+U[2]**2/2),
                        s0=entropy, quality=quality, slip=1., outlet="characteristic", back_pressure=2e5)
    m = model(boundaries=bc)
    np.testing.assert_allclose(m.boundary_residual(m.initial_state(), 0.), 0., atol=1e-14)
    jac = jax.jacfwd(lambda z: m.boundary_residual(m.initial_state().at[0].set(z), 0.)[0])(m.initial_state()[0])
    assert np.linalg.matrix_rank(np.asarray(jac)*m.variable_scales[None, :]) == 8


@pytest.mark.parametrize("alpha", (1e-8, 1-1e-8))
def test_open_phase_limits_are_finite_without_clipping(alpha):
    m = model(alpha=alpha)
    assert bool(m.cell_admissible(m.initial_state()))
    assert np.isfinite(np.asarray(raw_flux(m, m.initial_state()))).all()
    with pytest.raises(ValueError, match="violates"):
        model(alpha=0. if alpha < 0.5 else 1.)


def test_filters_off_by_default_separate_from_eos_and_boundaries():
    m = model()
    W = m.initial_state()
    np.testing.assert_array_equal(m.stabilization_flux(W, raw_flux(m, W)), 0.)
    assert np.isinf(float(m.explicit_timestep(W)))
    active = model(stabilization=StabilizationConfig(void_strength=0.05, thin_phase_strength=0.01))
    assert 0 < float(active.explicit_timestep(W)) < np.inf
    delta = active.stabilization_flux(W, raw_flux(active, W))
    np.testing.assert_array_equal(delta[jnp.array((0, delta.shape[0]-1))], 0.)


def test_unsupported_flux_and_nonfinite_configuration_rejected():
    m = model()
    with pytest.raises(ValueError, match="model_compatibility"):
        m.numerical_face_flux(m.initial_state()[:-1], m.initial_state()[1:], SimpleNamespace(flux="hll"))
    with pytest.raises(ValueError, match="finite"):
        CompatibilityConfig(lowmach_diss=float("nan"))
    with pytest.raises(ValueError, match="slip regularization"):
        ClosureConfig(smooth_slip=True,smooth_slip_u0=0.)


def test_frozen_upstream_flux_and_nc_fixture_agreement():
    """Fixture was evaluated directly from git objects at its recorded SHA."""
    fixture = json.loads((Path(__file__).parent/"data/two_fluid_alberto_reference.json").read_text())
    m = model()
    assert fixture["revision"] == m.reference_revision
    for case in fixture["cases"]:
        left, right = jnp.array(case["left"]), jnp.array(case["right"])
        WL = from_primitives(left[None], m.fluid_g, m.fluid_l)
        WR = from_primitives(right[None], m.fluid_g, m.fluid_l)
        actual = m.numerical_face_flux(WL, WR, SimpleNamespace(flux="model_compatibility"))[0]
        np.testing.assert_allclose(actual, case["flux"], rtol=2e-12, atol=1e-8)
        _, eg, el, cq = physical_data(left)
        nc = nonconservative_correction(left, jnp.array(case["df"]), eg, el, cq)
        np.testing.assert_allclose(nc, case["nc"], rtol=2e-12, atol=1e-8)


def test_momentum_pin_is_explicitly_equation_changing_and_dt_consistent():
    m = model(alpha=1e-6, pins=PhasePinConfig(momentum=True, momentum_alpha=1e-5, momentum_time=1e-4))
    W = m.initial_state().at[:, 3].add(2.)
    rows = jnp.ones((m.mesh.n_cells, 6))
    steady = m.transform_cell_equations(W, rows)
    for dt in (1e-6, 0.02):
        np.testing.assert_allclose(m.transform_cell_equations(W, dt*rows, dt=dt)/dt, steady, rtol=1e-14)
    np.testing.assert_array_equal(steady[:, [0, 1, 3, 4, 5]], rows[:, [0, 1, 3, 4, 5]])
    assert m.steady_only
    with pytest.raises(ValueError, match="implicit steady"):
        m.validate_scheme(SimpleNamespace(flux="model_compatibility", time_integrator="runge-kutta-3"))
    with pytest.raises(ValueError, match="momentum_time"):
        PhasePinConfig(momentum=True)


def test_coupled_and_segregated_implicit_steps_solve_same_two_fluid_traces():
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
    m0 = model()
    U = primitives(m0.initial_state())[0]
    W = m0.initial_state()[0]
    entropy = float(m0.fluid_g.get_state(jxp.HmassP_INPUTS, U[4], U[0])["s"])
    bc = BoundaryConfig(inlet="stagnation", H_g0=float(U[4]+U[1]**2/2), H_l0=float(U[5]+U[2]**2/2),
                        s0=entropy, quality=float(W[2]/(W[2]+W[4])), slip=1.,
                        outlet="characteristic", back_pressure=199900.)
    m = model(boundaries=bc)
    solutions = []
    for treatment in ("coupled", "segregated"):
        op = Operator(m, Scheme(flux="model_compatibility", boundary_treatment=treatment,
                                newton_iterations=12, newton_backtracking=True))
        new, info = op.step(m.initial_state(), 1e-7, 1e-7)
        assert bool(info.converged & info.admissible)
        np.testing.assert_allclose(m.boundary_residual(new, 1e-7), 0., atol=2e-9)
        np.testing.assert_allclose(m.cell_constraints(new)/m.constraint_scales, 0., atol=2e-9)
        solutions.append(new)
    np.testing.assert_allclose(np.asarray(solutions[0])/m.variable_scales, np.asarray(solutions[1])/m.variable_scales, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize("options", [
    {"area_law": name} for name in ("original", "polynomial", "parabola", "segments")
] + [
    {"drag_bubbly": name, "drag_droplet": name} for name in ("schiller_naumann", "ishii_zuber")
] + [
    {"regime_blend": name, "regime_htc": True} for name in ("linear", "smoothstep")
] + [
    {"htc_liquid_bubbly": name} for name in ("ranz_marshall", "jakob", "max", "liao")
] + [
    {"htc_liquid_droplet": name, "regime_htc": True} for name in ("internal", "jakob", "max")
])
def test_every_closure_selector_has_finite_values_and_ad(options):
    """Synthetic phase/saturation data exercise kernels, not fluid validation."""
    fluid = jxp.FluidPerfectGas(name="Air")
    def evaluate(U):
        eg, el = phase_props(fluid, U[0], U[4]), phase_props(fluid, U[0], U[5])
        return jnp.array(two_fluid_closure(U, eg, el, fluid, fluid, 260., 5e5, 2.5e5, **options))
    for slip in (0., 12., -12.):
        U = jnp.array((2e5, 20.+slip, 20., .4, 3e5, 2.8e5))
        value, derivative = evaluate(U), jax.jacfwd(evaluate)(U)
        assert np.isfinite(np.asarray(value)).all(), options
        assert np.isfinite(np.asarray(derivative)).all(), (options, slip)


def test_interfacial_parameter_overrides_are_not_shadowed():
    fluid = jxp.FluidPerfectGas(name="Air")
    U = jnp.array((2e5, 32., 20., 1e-8, 3e5, 2.8e5))
    eg, el = phase_props(fluid, U[0], U[4]), phase_props(fluid, U[0], U[5])
    a = two_fluid_closure(U, eg, el, fluid, fluid, 260., 5e5, 2.5e5,
                          area_law="original", area_params={"d_min_b": 5e-6})
    b = two_fluid_closure(U, eg, el, fluid, fluid, 260., 5e5, 2.5e5,
                          area_law="original", area_params={"d_min_b": 5e-5})
    assert float(a.Fv_g) != float(b.Fv_g)


@pytest.mark.parametrize("law", ("schiller_naumann", "ishii_zuber"))
def test_drag_has_analytic_stokes_derivative_at_literal_rest(law):
    from nozzle_hyperbolic.models.two_fluid._alberto_closures import calculate_drag_force_one_component, ishii_zuber_mixture_viscosity
    alpha, mu_l, mu_g, diameter = .1, 1e-3, 1e-5, 1e-4
    fn = lambda slip: calculate_drag_force_one_component(alpha,1-alpha,10.,1000.,mu_g,mu_l,slip,0.,diameter,
                                                         drag_bubbly=law,drag_droplet=law)
    viscosity = mu_l if law=="schiller_naumann" else ishii_zuber_mixture_viscosity(mu_l,mu_g,alpha)
    assert float(fn(0.)) == 0.
    np.testing.assert_allclose(jax.grad(fn)(0.),18*viscosity*alpha/diameter**2,rtol=1e-14)
    for slip in (-5.,5.):
        re = 1000.*abs(slip)*diameter/viscosity
        factor, exponent = (.15,.687) if law=="schiller_naumann" else (.1,.75)
        expected = .75/diameter*abs(slip)*slip*24/re*(1+factor*re**exponent)*alpha*1000.
        np.testing.assert_allclose(fn(slip),expected,rtol=1e-14)


@pytest.mark.parametrize("velocities", ((0.,20.),(20.,0.),(0.,-20.),(-20.,0.)))
def test_one_phase_at_rest_has_unique_compatible_flux_and_nc_action(velocities):
    U = jnp.array((2e5,*velocities,.4,3e5,2.8e5))
    fluid, eg, el, cq = physical_data(U)
    m = TwoFluidModel(ConstantAreaGeometry().build_mesh(3),fluid,fluid,initial_primitives=U)
    W = m.initial_state()
    flux = raw_flux(m,W)
    np.testing.assert_allclose(flux,m.face_physical_flux(W[:-1]),rtol=1e-12,atol=1e-9)
    K, X = K_matrix(U,eg,el), X_matrix(U,eg,el,cq)
    y = jnp.array((10.,.01,-.02,.001,3.,-2.))
    recovered = compatible_flux_solve(K,K@y,U)
    np.testing.assert_allclose(X@recovered,X@y,rtol=1e-11,atol=1e-9)


def test_saturation_required_for_closures_and_independent_energy_pin():
    with pytest.raises(ValueError,match="saturation tables"):
        model(closures=ClosureConfig())
    with pytest.raises(ValueError,match="saturation tables"):
        model(pins=PhasePinConfig(saturation_energy=True))


def test_binary_saturation_pin_does_not_replace_mass_equations():
    class Phase:
        def __init__(self, enthalpy):
            self.fluid = jxp.FluidPerfectGas(name="Air")
            self.enthalpy = enthalpy
        def get_state(self, *args):
            return self.fluid.get_state(*args)
        def get_state_saturation(self, pressure):
            return {"enthalpy":jnp.zeros_like(pressure)+self.enthalpy,
                    "temperature":jnp.zeros_like(pressure)+260.}
    fg, fl = Phase(5e5), Phase(2.5e5)
    m = TwoFluidModel(ConstantAreaGeometry().build_mesh(3),fg,fl,
                      initial_primitives=(2e5,20.,20.,1e-4,3e5,2.4e5),
                      pins=PhasePinConfig(saturation_energy=True))
    W=m.initial_state()
    rows=jnp.ones((3,6))
    selected=m.transform_cell_equations(W,rows,dt=.1)
    np.testing.assert_array_equal(selected[:,[0,1,2,3,5]],rows[:,[0,1,2,3,5]])
    np.testing.assert_allclose(selected[:,4],m.mesh.volumes*W[1:-1,2]*(3e5-5e5))
    U=primitives(W).at[:,3].set(.1)
    real_phase=from_primitives(U,fg,fl)
    np.testing.assert_array_equal(m.transform_cell_equations(real_phase,rows,dt=.1),rows)


def test_two_fluid_nonuniform_reconstructed_colored_jacobian_matches_dense():
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
    mesh=ConstantAreaGeometry().build_mesh(faces=np.linspace(0,1,5)**1.3)
    m=model(mesh=mesh,stabilization=StabilizationConfig(void_strength=.01))
    U=primitives(m.initial_state())
    # Strictly monotone slopes in every primitive avoid the limiter's genuine
    # nondifferentiable flat-state corner during directional finite differences.
    U=U+jnp.asarray(mesh.points)[:,None]*jnp.array((100.,2.,3.,.01,100.,200.))[None,:]
    W=from_primitives(U,m.fluid_g,m.fluid_l)
    op=Operator(m,Scheme(flux="model_compatibility",reconstruction="muscl_van_albada"))
    for face in op.reconstruct(W):
        np.testing.assert_allclose(m.cell_constraints(face)/m.constraint_scales,0.,atol=3e-15)
    args=(W.ravel(),W.ravel(),1e-6,0.)
    dense=np.asarray(op.dense_jacobian(*args))
    colored=np.asarray(op.jacobian(*args))
    scale=np.tile(m.variable_scales,W.shape[0])
    np.testing.assert_allclose(colored*scale,dense*scale,rtol=2e-11,atol=2e-11)
    direction=jnp.asarray(scale*np.sin(np.arange(W.size)))*1e-3
    eps=1e-4
    fd=(op.residual(args[0]+eps*direction,*args[1:])-op.residual(args[0]-eps*direction,*args[1:]))/(2*eps)
    np.testing.assert_allclose(dense@direction,fd,rtol=2e-5,atol=2e-8)


def test_subsonic_inflow_prescribes_five_primitive_values_and_extrapolates_pressure():
    target=(1.9e5,23.,24.,.3,2.9e5,2.7e5)
    m=model(boundaries=BoundaryConfig(inlet="subsonic_inflow",inlet_state=target,
                                      outlet="pressure",back_pressure=1.99e5))
    U=primitives(m.initial_state()).at[0].set(jnp.asarray(target).at[0].set(2e5))
    W=from_primitives(U,m.fluid_g,m.fluid_l)
    np.testing.assert_allclose(m.boundary_residual(W,0.)[0],0.,atol=2e-15)
    changed=W.at[0,0].add(100.)
    assert float(m.boundary_residual(changed,0.)[0,0]) != 0.
    with pytest.raises(ValueError,match="six primitive"):
        BoundaryConfig(inlet="subsonic_inflow",inlet_state=(1.,2.))
    with pytest.raises(ValueError,match="finite"):
        BoundaryConfig(inlet="static_state",inlet_state=(1.,2.,3.,.4,float("nan"),6.))


@pytest.mark.parametrize("options", (
    {"area_params":{"d_min":float("nan")}},
    {"area_params":{"ai_ctrl":[100.,float("inf")]}},
    {"area_params":{"d_min":-1.}},
    {"area_params":{"unrecognized":1.}},
    {"area_law":"segments","area_params":{"alpha_pts":[0.,.7,.5,1.]}},
    {"area_law":"segments","area_params":{"ai_pts":[100.,200.]}},
))
def test_interfacial_area_configuration_rejects_invalid_parameters(options):
    with pytest.raises(ValueError):
        ClosureConfig(**options)


def test_bicubic_mask_rejects_unpopulated_nodes_and_nonfinite_coefficients():
    from nozzle_hyperbolic.models.two_fluid.model import _table_stencil_mask
    table={name:{"value":np.ones((4,4)),"coeffs":np.ones((3,3,16))}
           for name in ("density","temperature","speed_of_sound","gruneisen","entropy")}
    fluid=SimpleNamespace(table=table,N_h=4,N_p=4)
    np.testing.assert_array_equal(_table_stencil_mask(fluid,False),np.ones((3,3),bool))
    table["density"]["value"][0,0]=np.nan
    table["temperature"]["coeffs"][2,2,0]=np.inf
    expected=np.ones((3,3),bool)
    expected[0,0]=expected[2,2]=False
    np.testing.assert_array_equal(_table_stencil_mask(fluid,False),expected)
