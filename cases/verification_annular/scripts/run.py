"""Execute the imported NozzleX ODE and production annular FVM comparison."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzlex
import numpy as np
import yaml

from nozzlex.vaneless_channel.vaneless_channel_model import VanelessChannel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.closures.nozzlex import NozzleXWall
from nozzle_hyperbolic.solvers.fvm import (
    Scheme,SteadySettings,ResidualPolicy,solve_single_grid,
)
from .geometry import shared_geometry

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
OUT=STUDY/"outputs"


def plain(value):
    if isinstance(value,dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value,(jax.Array,np.ndarray)):
        return plain(np.asarray(value).tolist())
    if isinstance(value,np.generic):
        return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return str(value)
    return value


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False),encoding="utf-8")


def provenance():
    xroot=Path(nozzlex.__file__).resolve().parents[1]
    sources={}
    for label,folder in (("hyperbolic",ROOT/"src"),("nozzlex",xroot/"nozzlex"),
                         ("study",STUDY/"scripts")):
        for p in folder.rglob("*.py"):
            sources[f"{label}/{p.relative_to(folder).as_posix()}"]=sha256(p.read_bytes()).hexdigest()
    packages={name:importlib.metadata.version(name) for name in
              ("jax","jaxlib","jaxprop","nurbspy","nozzlex","diffrax","numpy","scipy")}
    return dict(sources=sources,packages=packages,platform=platform.platform(),
                dtype="float64",devices=[str(d) for d in jax.devices()],
                nozzlex_root=str(xroot),python=platform.python_version())


def reference(channel,handle,coordinates,tolerance):
    config=deepcopy(channel)
    config["solver_options"].update(rtol=tolerance,atol=tolerance,max_steps=10000,n_points=1001)
    fluid=jxp.FluidPerfectGas("Air",T_ref=300,p_ref=101325)
    model=VanelessChannel.from_dict(config,fluid)
    start=perf_counter()
    fields=jax.device_get(model.solve(geometry_handle=handle,
                                     save_coordinates=jnp.asarray(coordinates)))
    elapsed=perf_counter()-start
    mdot=np.asarray(fields["d"]*fields["v_m"]*fields["A"])
    audit=dict(mass_variation=float(np.ptp(mdot)/abs(mdot[0])),
               entropy_integration_error=float(np.max(abs(fields["s"]-fields["s_int"]))),
               M_m_min=float(np.min(fields["Ma_m"])),M_m_max=float(np.max(fields["Ma_m"])),
               h0_relative_variation=float(np.ptp(fields["h0"])/abs(fields["h0"][0])))
    audit["passed"]=bool(audit["mass_variation"]<1e-7 and
        audit["entropy_integration_error"]<1e-5 and 0<audit["M_m_min"]<audit["M_m_max"]<1)
    return dict(fields=fields,tolerance=tolerance,elapsed_compile_inclusive_s=elapsed,audit=audit)


def compare(model,state,ref):
    fields=model.fields(jnp.asarray(state))
    m=model.native_mesh.m_centers
    indices=np.searchsorted(ref["fields"]["m"],m)
    np.testing.assert_allclose(np.asarray(ref["fields"]["m"])[indices],m,rtol=0,atol=1e-14)
    keys=dict(p="p",v_m="v_m",v_theta="v_t",h="h",rho="d",M="Ma",M_m="Ma_m",
              T="T",h0="h0",s="s",alpha_deg="alpha")
    selected={k:np.asarray(ref["fields"][v])[indices] for k,v in keys.items()}
    selected["mdot"]=selected["rho"]*selected["v_m"]*model.mesh.area_centers
    scales=dict(p=model.reference_pressure,v_m=model.reference_sound_speed,
                v_theta=model.reference_sound_speed,h=model.reference_sound_speed**2,
                rho=model.reference_density,M=1,M_m=1,T=300,h0=model.reference_sound_speed**2,
                s=model.reference_sound_speed**2/300,alpha_deg=60,mdot=selected["mdot"][0])
    weights=model.mesh.volumes/np.sum(model.mesh.volumes)
    errors={k:dict(l1=float(weights@abs(np.asarray(fields[k])-expected)),
                   normalized_l1=float(weights@abs(np.asarray(fields[k])-expected)/scales[k]),
                   linf=float(np.max(abs(np.asarray(fields[k])-expected))))
            for k,expected in selected.items()}
    # Pointwise primitive comparisons above are distinct from conservative errors.
    refW=np.column_stack((selected["p"],selected["v_m"],
                         selected["v_theta"]*model.native_mesh.r_center,selected["h"]))
    from nozzle_hyperbolic.models.euler_annular.state import primitive_to_conservative
    q=np.asarray(primitive_to_conservative(jnp.asarray(state[1:-1]),model._r,model.fluid))
    qr=np.asarray(primitive_to_conservative(jnp.asarray(refW),model._r,model.fluid))
    errors["conservative_normalized_l1"]=(weights@abs((q-qr)/model.conservative_scales)).tolist()
    errors["inlet_Mach_error"]=float(np.hypot(state[0,1],state[0,2]/model._r_full[0])/
        model.fluid.get_state(jxp.HmassP_INPUTS,state[0,3],state[0,0])["a"]-.8)
    return plain(fields),selected,errors


def run(variants=None,grids=None):
    config=yaml.safe_load((STUDY/"case.yaml").read_text(encoding="utf-8"))
    before=provenance()
    source=Path(before["nozzlex_root"])/config["nozzlex_case"]
    base=yaml.safe_load(source.read_text(encoding="utf-8"))
    fluid=jxp.FluidPerfectGas("Air",T_ref=300,p_ref=101325)
    channel=VanelessChannel.from_dict(base,fluid)
    physical,handle,geometry_audit=shared_geometry(channel,config["numerics"]["geometry_map_points"])
    grids=grids or config["numerics"]["grids"]
    meshes={n:build_annular_mesh(physical,n,faces=physical.length*
                np.linspace(0,1,n+1)**config["numerics"]["mesh_stretch"]) for n in grids}
    coordinates=np.unique(np.concatenate([np.linspace(0,physical.length,1001)]+
        [mesh.centers for mesh,_ in meshes.values()]))
    results=dict(config=config,base_case=base,source_case=str(source),
                 source_case_sha256=sha256(source.read_bytes()).hexdigest(),
                 provenance_before=before,geometry_audit=geometry_audit,variants={})
    save(OUT/"setup.json",results)
    scheme=Scheme(flux=config["numerics"]["flux"],reconstruction=config["numerics"]["reconstruction"],
                  time_integrator="euler-implicit",newton_iterations=15,newton_backtracking=True)
    policy=ResidualPolicy(criterion="absolute",absolute_tolerance=1e-9,
                          boundary_tolerance=1e-9,global_tolerance=1e-9)
    settings=SteadySettings(residual=policy,**{key:config["numerics"][key] for key in
                           ("max_steps","cfl","cfl_growth","cfl_max")})
    for name in variants or config["variants"]:
        print(f"REFERENCE {name}",flush=True)
        variant=deepcopy(base)
        variant["model_options"]=deepcopy(config["variants"][name])
        refs=[]
        for tol in config["numerics"]["reference_tolerances"]:
            record=reference(variant,handle,coordinates,tol)
            refs.append(record)
            save(OUT/name/f"nozzlex_{tol:.0e}.json",record)
            print("  tol",tol,record["audit"],flush=True)
        ref=refs[-1]
        tolerance_change={k:float(np.max(abs(refs[0]["fields"][k]-ref["fields"][k])))
                          for k in ("p","v_m","v_t","h","Ma_m")}
        last_change={k:float(np.max(abs(refs[-2]["fields"][k]-ref["fields"][k])))
                     for k in ("p","v_m","v_t","h","Ma_m")}
        vresult=dict(references=[dict(tolerance=r["tolerance"],audit=r["audit"],
                     elapsed_compile_inclusive_s=r["elapsed_compile_inclusive_s"]) for r in refs],
                     reference_tolerance_change=tolerance_change,
                     reference_last_refinement_change=last_change,grids=[])
        results["variants"][name]=vresult
        previous=None
        for n in grids:
            mesh,native=meshes[n]
            wall=NozzleXWall(native,variant["model_options"],alpha_in_deg=base["operating_conditions"]["alpha_in"])
            seed=None
            if previous is not None and previous.success:
                seed=np.column_stack([np.interp(mesh.points,previous.operator.mesh.points,
                                       previous.state[:,k]) for k in range(4)])
            model=AnnularModel(mesh,native,fluid,
                inlet_data=dict(p0=base["operating_conditions"]["p0_in"],
                                T0=base["operating_conditions"]["T0_in"],
                                alpha_in=np.deg2rad(base["operating_conditions"]["alpha_in"])),
                outlet_pressure=float(ref["fields"]["p"][-1]),wall_closure=wall)
            print(f"HYPERBOLIC {name} N={n}",flush=True)
            initial=np.asarray(model.initial_state() if seed is None else seed)
            start=perf_counter()
            solution=solve_single_grid(model,scheme,settings,initial_state=initial)
            elapsed=perf_counter()-start
            fields,reference_fields,errors=compare(model,solution.state,ref)
            flux=np.asarray(solution.operator.face_flux(jnp.asarray(solution.state)))
            sources=np.asarray(model.integrated_source(jnp.asarray(solution.state)))
            budget=flux[-1]-flux[0]-sources.sum(axis=0)
            record=dict(cells=n,success=solution.success,reason=solution.reason,
                        metrics=solution.metrics,state=solution.state,initial_state=initial,
                        initialization="uniform cold guess" if seed is None else "previous coarser FVM solution",
                        history=solution.history,elapsed_compile_inclusive_s=elapsed,
                        fields=fields,reference_fields=reference_fields,errors=errors,
                        m=mesh.centers,faces=mesh.faces,volumes=mesh.volumes,
                        budget=budget,budget_normalized=budget/model.global_flux_scales,
                        face_flux=flux,sources=sources,
                        wall_loads=wall.fields(jnp.asarray(solution.state[1:-1]),model._r,fluid),
                        scheme=asdict(scheme),settings=asdict(settings))
            save(OUT/name/f"hyperbolic_n{n}.json",record)
            vresult["grids"].append({k:record[k] for k in ("cells","success","reason","errors",
                "metrics","elapsed_compile_inclusive_s","budget_normalized")})
            save(OUT/"results.json",results)
            print(" ",solution.success,solution.reason,"max q error",
                  max(errors["conservative_normalized_l1"]),flush=True)
            previous=solution
            if not solution.success:
                break
    results["provenance_after"]=provenance()
    changed={k for k,v in before["sources"].items()
             if results["provenance_after"]["sources"].get(k)!=v}
    results["sources_changed_during_run"]=sorted(changed)
    results["all_solved"]=all(len(v["grids"])==len(grids) and all(r["success"] for r in v["grids"])
                                 and v["references"][-1]["audit"]["passed"]
                                 for v in results["variants"].values())
    save(OUT/"results.json",results)
    print("ALL SOLVED:",results["all_solved"],flush=True)
