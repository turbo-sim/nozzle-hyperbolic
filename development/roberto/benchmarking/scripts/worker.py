"""Isolated backend process; one complete warmup and cold repeated solves.

The current backend executes the production public driver. Only its operator
and sampler factories are reused in memory to exclude binding/JIT from repeated
solver-loop timing. Every solve resets state, residual normalization, controller
and history. Legacy evolution uses frozen kernels and a matched strict controller.
"""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from contextlib import redirect_stdout
import json
import importlib.util
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

import numpy as np
from .common import clean, ROOT


def current_monitor_module():
    """Load generic current reductions without mixing physical package trees."""
    name = "benchmark_current_residual_policy"
    if name not in sys.modules:
        path = ROOT / "src/nozzle_hyperbolic/solvers/residuals.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def legacy_sampler(problem, policy):
    import jax
    import jax.numpy as jnp
    shared = current_monitor_module()
    reduce_residuals, assess_residuals = shared.reduce_residuals, shared.assess_residuals

    @jax.jit
    def reduction(raw, boundary):
        values = reduce_residuals(raw, jnp.asarray(problem.volumes),
                                  jnp.asarray(problem.conservative_scales), problem.reference_time,
                                  weighting=policy.weighting)
        return values, jnp.max(jnp.abs(boundary)), jnp.abs(jnp.sum(raw, axis=0)/jnp.asarray(problem.global_flux_scales))

    def sample(state, initial=None):
        raw, boundary = problem.spatial_residual(state), problem.boundary_residual(state)
        reduced, bc, global_defect = jax.device_get(reduction(raw, boundary))
        norms = np.asarray(reduced[policy.norm]); initial = norms if initial is None else initial
        assessment = jax.device_get(assess_residuals(norms, initial, policy))
        valid = bool(problem.admissible(state))
        finite = bool(assessment["finite"]) and np.isfinite(bc) and np.isfinite(global_defect).all()
        return dict(equation_norm=norms, equation_relative=np.asarray(assessment["equation_relative"]),
                    equation_l1=np.asarray(reduced["l1"]), equation_l2=np.asarray(reduced["l2"]),
                    equation_linf=np.asarray(reduced["linf"]), worst_cell=np.asarray(reduced["worst_cell"]),
                    overall_relative=float(np.max(np.where(assessment["relative_active"], assessment["equation_relative"], 0.))),
                    equation_max_ratio=float(assessment["equation_max_ratio"]),
                    all_equations_passed=bool(assessment["all_equations_passed"]),
                    boundary_max=float(bc), global_defect=np.asarray(global_defect),
                    boundary_passed=bool(bc<=policy.boundary_tolerance),
                    global_passed=bool(np.max(global_defect)<=policy.global_tolerance),
                    finite=bool(finite), admissible=valid)
    return sample


def done(metrics):
    return all(metrics[key] for key in ("finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed"))


def solve_legacy(problem, sample, settings):
    import jax
    import jax.numpy as jnp
    state=jnp.asarray(problem.initial_state,dtype=jnp.float64)
    history=[];start=perf_counter();cfl=settings.cfl;accepted=0;unchanged=0;reason="max_steps"
    if not bool(problem.admissible(state)):
        return dict(state=state,success=False,reason="invalid_initial_state",history=[],metrics={})
    current=sample(state);initial=np.asarray(current["equation_norm"])
    history.append(dict(step=0,phase="initial",accepted=True,cfl=cfl,dt=0.,elapsed_s=perf_counter()-start,**current))
    while accepted<settings.max_steps and not done(current):
        okay=False
        for retry in range(settings.max_retries+1):
            wave_dt=float(problem.wave_dt(state));dt=cfl*wave_dt
            if not np.isfinite(dt) or dt<=0:
                reason="invalid_timestep";break
            candidate,info=problem.step(state,jnp.float64(dt));info=jax.device_get(info)
            okay=bool(info.converged and info.admissible)
            trial=sample(candidate,initial) if okay else None
            okay=okay and trial["finite"] and trial["admissible"]
            row=dict(step=accepted+1,retry=retry,phase="accepted" if okay else "rejected",accepted=bool(okay),
                     cfl=cfl,requested_cfl=cfl,dt=dt,inner_iterations=int(info.iterations),
                     inner_status=int(info.status),inner_residual=float(info.residual_norm),
                     rejected_newton_trials=int(info.rejected_trials),elapsed_s=perf_counter()-start)
            if okay:
                unchanged=unchanged+1 if np.array_equal(np.asarray(candidate),np.asarray(state)) else 0
                state=candidate;accepted+=1;current=trial;history.append(dict(**row,**current))
                cfl=min(settings.cfl_max,cfl*settings.cfl_growth);break
            history.append(row);cfl*=settings.cfl_shrink
        if not okay:
            if reason!="invalid_timestep":reason="retry_limit"
            break
        if settings.stagnation_steps>0 and unchanged>=settings.stagnation_steps and not done(current):
            reason="stagnation";break
        if accepted%250==0:
            print(f'  {accepted:5d} R={current["overall_relative"]:.3e} CFL={cfl:.2e}',file=sys.stderr,flush=True)
    final=sample(state,initial);success=done(final)
    return dict(state=state,success=success,reason="converged" if success else reason,history=history,metrics=final)


def conserved_from_fields(fields, model_name):
    rho,p,h=fields["rho"],fields["p"],fields["h"]
    if model_name=="euler_nozzle":
        u=fields["u"];return np.column_stack((rho,rho*u,rho*(h+.5*u*u)-p))
    vm,vt,ell=fields["v_m"],fields["v_theta"],fields["ell"]
    return np.column_stack((rho,rho*vm,rho*ell,rho*(h+.5*(vm*vm+vt*vt))-p))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--backend",choices=("current","legacy"),required=True)
    args=parser.parse_args()
    import jax
    jax.config.update("jax_enable_x64",True)
    import jax.numpy as jnp
    if args.backend=="legacy":
        from .legacy_adapter import load_legacy_package,bind_legacy
        load_legacy_package()
        ResidualPolicy = current_monitor_module().ResidualPolicy
    else:
        from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
    # The frozen config class is not used to select numerical methods, only a
    # namespace for the common strict outer controller's resolved settings.
    from types import SimpleNamespace
    cache={}

    def bind(command):
        if cache:
            cache.clear();jax.clear_caches()
        profile=command["profile"];policy=ResidualPolicy(**profile["residual"])
        settings=SimpleNamespace(residual=policy,**profile["steady"])
        if args.backend=="current":
            from nozzle_hyperbolic.verification import build_case
            from nozzle_hyperbolic.solvers.fvm import Scheme,SteadySettings,Operator
            from nozzle_hyperbolic.solvers.fvm import drivers
            physical=build_case(command["case_id"],command["cells"])
            scheme=Scheme(reconstruction=command["reconstruction"],newton_iterations=command["cap"],**profile["scheme"])
            settings=SteadySettings(residual=policy,**profile["steady"])
            operator=Operator(physical.model,scheme);sample=drivers.sampler(operator,policy)
            roe_metadata=dict(getattr(physical.model,"roe_metadata",{})) if scheme.flux=="roe" else None
            if scheme.flux=="roe":
                if not roe_metadata:
                    raise ValueError("Roe model did not declare its averaging/entropy/invalid-state policy")
                for key,value in profile.get("roe_options",{}).items():
                    if roe_metadata.get(key)!=value:
                        raise ValueError(f"Requested Roe {key}={value!r}; bound model declares {roe_metadata.get(key)!r}")
            initial=np.asarray(command["payload"]["initial_state"],dtype=float)
            def solve():
                started=perf_counter()
                with patch.object(drivers,"Operator",lambda model,selected:operator),patch.object(drivers,"sampler",lambda op,selected:sample):
                    solution=drivers.solve_single_grid(physical.model,scheme,settings,initial_state=initial,
                         callback=lambda row:row.update(elapsed_s=perf_counter()-started))
                return dict(state=solution.state,success=solution.success,reason=solution.reason,
                            history=solution.history,metrics=solution.metrics)
            cache.update(problem=physical,operator=operator,solve=solve,fields=physical.numerical_fields,
                         x=np.asarray(physical.mesh.centers),volumes=np.asarray(physical.mesh.volumes),model=physical.definition.model_name,
                         metadata=dict(backend="current",entrypoint="solvers.fvm.solve_single_grid",bound_operator_reuse=True,
                           flux=scheme.flux,roe=roe_metadata,
                           boundary_treatment=scheme.effective_boundary_treatment,newton_backtracking=scheme.newton_backtracking,
                           package_file=str(__import__('nozzle_hyperbolic').__file__)))
        else:
            if profile["scheme"]["time_integrator"]!="euler-implicit" or profile["scheme"]["boundary_treatment"] not in ("auto","coupled") or profile["scheme"].get("filter_strength",0)!=0:
                raise NotImplementedError("UnsupportedLegacyCase: native comparison requires implicit, native boundary treatment and no filter")
            problem=bind_legacy(command["payload"],cells=command["cells"],reconstruction=command["reconstruction"],newton_cap=command["cap"],
                   newton_backtracking=profile["scheme"]["newton_backtracking"],update_tolerance=profile["scheme"]["newton_update_tolerance"],
                   residual_tolerance=profile["scheme"]["newton_residual_tolerance"],delta_factor=profile["scheme"]["delta_factor"],
                   flux=profile["scheme"]["flux"],jacobian_mode=profile["scheme"]["jacobian_mode"],linear_solver=profile["scheme"]["linear_solver"],
                   custom_jump_correction=profile["scheme"].get("custom_jump_correction",False),
                   custom_endpoint_envelope=profile["scheme"].get("custom_endpoint_envelope",False),
                   max_backtracks=profile["scheme"]["max_backtracks"])
            sample=legacy_sampler(problem,policy)
            cache.update(problem=problem,solve=lambda:solve_legacy(problem,sample,settings),fields=problem.fields,
                         x=np.asarray(problem.coordinates),volumes=np.asarray(problem.volumes),model=problem.model_name,metadata=problem.metadata)
        return dict(metadata=cache["metadata"],x=cache["x"],volumes=cache["volumes"])

    def solve():
        started=perf_counter();result=cache["solve"]();jax.block_until_ready(result["state"])
        state=np.asarray(result["state"]);elapsed=perf_counter()-started
        fields={key:np.asarray(value) for key,value in cache["fields"](state).items() if isinstance(value,(np.ndarray,jax.Array,list,float,int))}
        fields["coordinate"]=cache["x"]
        q=conserved_from_fields(fields,cache["model"])
        initial=next((row for row in result["history"] if row.get("phase")=="initial"),{})
        result["metrics"]["initial_norm"]=initial.get("equation_norm")
        work=dict(accepted_steps=sum(row.get("phase")=="accepted" for row in result["history"]),
                  rejected_steps=sum(row.get("phase")=="rejected" for row in result["history"]),
                  newton_corrections=sum(row.get("inner_iterations",0) for row in result["history"]),
                  rejected_newton_trials=sum(row.get("rejected_newton_trials",0) for row in result["history"]),
                  max_inner_iterations=max((row.get("inner_iterations",0) for row in result["history"]),default=0))
        diagnostics=None
        if cache["metadata"].get("flux")=="roe":
            # Face audits are outside the complete-solve timing boundary.
            try:
                face_data=jax.device_get(cache["operator"].roe_data(jnp.asarray(state)))
                diagnostics={}
                for key,value in face_data.items():
                    if key=="projectors":continue
                    values=np.asarray(value)
                    diagnostics[key]=dict(shape=list(values.shape),finite=bool(np.isfinite(values).all()),
                                          minimum=float(np.min(values)),maximum=float(np.max(values)))
                diagnostics["invalid_faces"]=int(np.count_nonzero(~np.asarray(face_data["valid"],dtype=bool)))
            except Exception as error:
                diagnostics=dict(error=repr(error))
        return dict(elapsed_s=elapsed,success=result["success"],reason=result["reason"],metrics=result["metrics"],roe_diagnostics=diagnostics,
                    work=work,state=state,fields=fields,q=q,x=cache["x"],volumes=cache["volumes"],history=result["history"])

    print(json.dumps(dict(ready=True,backend=args.backend)),flush=True)
    for line in sys.stdin:
        try:
            command=json.loads(line)
            if command["command"]=="shutdown":
                print('{"ok":true}',flush=True);break
            with redirect_stdout(sys.stderr):
                result=bind(command) if command["command"]=="bind" else solve() if command["command"]=="solve" else None
            print(json.dumps(clean(dict(ok=True,result=result)),allow_nan=False),flush=True)
        except Exception as error:
            print(json.dumps(dict(ok=False,error=repr(error),traceback=traceback.format_exc())),flush=True)

if __name__=="__main__":
    main()
