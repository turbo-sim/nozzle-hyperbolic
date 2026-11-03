"""Equation, zero-storage boundary, lagged-step and Jacobian checks; no campaign."""

import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[name]="1"

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator,Scheme
from nozzle_hyperbolic.verification import build_case
from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from .adapter import local_operator
from .common import STUDY,save,source_hashes


def main():
    before = source_hashes()
    records=[]
    for reconstruction in ("constant","muscl_van_leer"):
        problem=build_case("euler_nozzle/shock_35",cells=6)
        base=Operator(problem.model,Scheme(flux="hll",reconstruction=reconstruction,newton_backtracking=True))
        local=local_operator(base)
        full=np.asarray(problem.initial_state).copy()
        x=np.linspace(0.,1.,len(full))
        full[:,0]*=1-.03*x*x
        full[:,1]=30+20*x+3*x*x
        full[:,2]*=1+.01*x+.004*x*x
        full=jnp.asarray(full)
        assert bool(base.is_admissible(full))
        scalar=jnp.asarray(1e-5)
        uniform=jnp.full((6,1),scalar)
        args=(full.ravel(),(.999*full).ravel(),uniform,jnp.asarray(0.))
        scalar_res=np.asarray(base.residual(args[0],args[1],scalar,args[3]))
        np.testing.assert_allclose(base.residual(*args),scalar_res,atol=1e-14,rtol=1e-14)
        times=local.local_wave_times(full)
        np.testing.assert_allclose(jnp.min(times),base.timestep(full,1.).wave_dt,rtol=1e-15)
        vector=local.local_cell_steps(full,scalar)
        np.testing.assert_allclose(vector[:,0],scalar*times/jnp.min(times),rtol=1e-15)
        residual=np.asarray(base.residual(full.ravel(),args[1],vector,0.)).reshape(base.shape)
        scale=np.asarray(problem.mesh.volumes)[:,None]*problem.model.conservative_scales
        row_scale=np.maximum(1.,np.asarray(vector)/problem.model.reference_time)
        expected=((np.asarray(base.storage(full[1:-1]))-np.asarray(base.storage(.999*full[1:-1])))
                  /row_scale+(np.asarray(vector)/row_scale)*np.asarray(base.spatial_residual(full)))/scale
        np.testing.assert_allclose(residual[1:-1],expected,rtol=1e-13,atol=1e-14)
        np.testing.assert_array_equal(residual[[0,-1]],np.asarray(base.boundary_residual(full,0.)))
        other=np.asarray(base.residual(full.ravel(),(.9*full).ravel(),vector*3.,0.)).reshape(base.shape)
        np.testing.assert_array_equal(residual[[0,-1]],other[[0,-1]])
        blocks=make_block_operators(base.residual,*base.shape,base.radius,jacobian_mode="colored",linear_solver="banded")
        args=(full.ravel(),args[1],vector,jnp.asarray(0.))
        colored=np.asarray(jax.jit(blocks["colored_jacobian"])(*args))
        dense=np.asarray(jax.jit(blocks["dense_jacobian"])(*args))
        jacobian_error=float(np.max(abs(colored-dense)))
        np.testing.assert_allclose(colored,dense,rtol=1e-11,atol=1e-13)
        direction=np.sin(np.arange(full.size)+.3)*np.tile(np.asarray(problem.model.variable_scales),len(full))
        eps=1e-6
        fd=(np.asarray(base.residual(args[0]+eps*direction,*args[1:]))-np.asarray(base.residual(args[0]-eps*direction,*args[1:])))/(2*eps)
        exact=dense@direction
        relative=float(np.max(abs(fd-exact))/max(np.max(abs(exact)),1.))
        assert relative<2e-6,relative
        records.append(dict(reconstruction=reconstruction,passed=True,uniform_step_equivalence=True,
            boundary_zero_storage=True,lagged_local_weights=True,colored_dense_max_error=jacobian_error,
            directional_fd_relative_error=relative,wave_time_ratio=float(jnp.max(times)/jnp.min(times))))
        print(records[-1],flush=True)
    assert before==source_hashes()
    out=STUDY/"outputs"
    out.mkdir(exist_ok=True)
    save(out/"implementation_checks.json",dict(passed=True,checks=records,source_unchanged=True))


if __name__=="__main__": main()
