"""Independent local-kernel, stencil and saved-result checks for the ablation."""

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from .common import ROOT, STUDY, CASES, VARIANTS, read, save, sha, sources


def kernels(output, cache):
    from .run import setup, build
    fluids, _ = setup(cache)
    import jax
    import jax.numpy as jnp
    from nozzle_hyperbolic.models.two_fluid.flux import fvs_interface_flux
    from nozzle_hyperbolic.models.two_fluid import primitives
    from nozzle_hyperbolic.models.two_fluid.matrices import phase_props, common_quantities, X_matrix
    from cases.NICFD2026.scripts.case import build_case, specification
    from .numerics import face, path_nc, data
    from development.two_fluid_test_v1.scripts.kernels import inverse_free_face

    before = sources()
    result = dict(checks=[], native_equivalence=[], v1_equivalence=[], sonic_sweep=[], jacobians=[])

    def check(name, condition):
        result["checks"].append(dict(name=name, passed=bool(condition)))
        save(output, result)
        if not condition:
            raise AssertionError(name)

    def relative(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(abs(a-b)) / max(float(np.max(abs(b))), 1.))

    for regime in CASES:
        base, scheme, _, _ = build_case(specification("two_component", regime), fluids, 50,
            initialization="reference", reconstruction="constant", newton_iterations=20)
        values = primitives(base.initial_state())
        pairs = [(values[i], values[i+1]) for i in (0, 14, 24, 29, 39, 50)]
        cfg = base.compatibility
        for floor in (0., 1.):
            native = jax.jit(lambda a,b: fvs_interface_flux(a,b,base.fluid_g,base.fluid_l,base.virtual_mass,
                delta_factor=cfg.delta_factor, lowmach_diss=floor,du_blend_lo=cfg.du_blend_lo,du_blend_hi=cfg.du_blend_hi))
            cloned = jax.jit(lambda a,b: face(a,b,base.fluid_g,base.fluid_l,base.virtual_mass,cfg,floor,False,False))
            both = jax.jit(lambda a,b: face(a,b,base.fluid_g,base.fluid_l,base.virtual_mass,cfg,floor,True,True))
            v1 = jax.jit(lambda a,b: inverse_free_face(a,b,base.fluid_g,base.fluid_l,base.virtual_mass,
                lowmach_diss=floor,du_blend_lo=cfg.du_blend_lo,du_blend_hi=cfg.du_blend_hi)[0])
            error = max(relative(cloned(a,b), native(a,b)) for a,b in pairs)
            v1_error = max(relative(both(a,b), v1(a,b)) for a,b in pairs)
            result["native_equivalence"].append(dict(regime=regime,acoustic_floor=floor,max_relative_error=error))
            result["v1_equivalence"].append(dict(regime=regime,acoustic_floor=floor,max_relative_error=v1_error))
            check(regime+f" legacy clone equals src floor={floor}", error < 1e-11)
            check(regime+f" both equals boundary-matched v1 floor={floor}", v1_error < 1e-11)

    gas, liquid, vm = base.fluid_g, base.fluid_l, base.virtual_mass
    u = jnp.array((1e6, 0., 0., .9, 280000., 95000.))
    sound = float(.5 * (data(u,gas,liquid,vm)[3][0][2] - data(u,gas,liquid,vm)[3][0][3]))
    for variant in VARIANTS:
        inv, sonic = variant in ("inverse_old_sonic","both"), variant in ("sonic_only","both")
        function = jax.jit(lambda a,b: face(a,b,gas,liquid,vm,cfg,1.,inv,sonic))
        jumps = []
        for epsilon in (1e-2,1e-4,1e-6):
            right = u.at[1:3].set(sound+10.)
            fm = function(u.at[1:3].set(sound-epsilon),right)
            fp = function(u.at[1:3].set(sound+epsilon),right)
            jumps.append(relative(fp,fm))
        result["sonic_sweep"].append(dict(variant=variant,epsilons=[1e-2,1e-4,1e-6],relative_flux_jumps=jumps))
        check(variant+" sonic policy is continuous" if sonic else variant+" retains old sonic jump",
              jumps[-1] < jumps[0]*.01 if sonic else jumps[-1] > 1e-6)
        for slip in (0.,5.,20.,40.):
            point = u.at[1].set(80.+slip).at[2].set(80.)
            jac = jax.jacfwd(lambda x: function(x,point))(point)
            check(variant+f" finite AD at slip {slip}", np.isfinite(jac).all())

    left = u.at[1:3].set(jnp.array((100.,80.)))
    right = left.at[0].set(8e5).at[3].set(.94)
    nc = np.asarray(path_nc(left,right,gas,liquid,vm))
    check("NC mass rows vanish", np.array_equal(nc[:2],np.zeros(2)))
    check("NC mixture momentum cancels", nc[2]+nc[3] == 0.)
    check("NC path antisymmetry", relative(path_nc(right,left,gas,liquid,vm),-nc) < 1e-12)

    for variant in ("sonic_only", "inverse_old_sonic"):
        model, scheme, _, ws, _ = build(fluids,"supersonic_adapted","muscl_van_albada",8,variant)
        flat = model.initial_state().ravel()
        arguments = (flat,jnp.float64(1e-6),jnp.float64(0.))
        colored, dense = np.asarray(ws.operator.jacobian(flat,*arguments)),np.asarray(ws.operator.dense_jacobian(flat,*arguments))
        error = relative(colored,dense)
        result["jacobians"].append(dict(variant=variant,relative_error=error,radius=ws.operator.radius))
        check(variant+" colored AD equals dense AD", error < 1e-11 and np.isfinite(colored).all())
    check("Physical and study numerical source unchanged", before == sources())
    result.update(passed=True,complete=True,source=before,audit_sha256=sha(Path(__file__)))
    save(output,result)
    print("KERNEL AUDIT PASSED",len(result["checks"]),flush=True)


def saved(directory, output):
    import zipfile
    metadata = read(directory / "metadata.json")
    checks = []
    def check(name, passed):
        checks.append(dict(name=name,passed=bool(passed)))
        if not passed:
            raise AssertionError(name)
    check("Current numerical/physical source matches campaign",sources() == metadata["source"])
    with zipfile.ZipFile(directory/"source.zip") as archive:
        from hashlib import sha256
        for group in metadata["source"].values():
            for name,digest in group["files"].items():
                check("Archived source "+name,sha256(archive.read(name)).hexdigest()==digest)
    counts = {v:dict(attempts=0,roots=0,profiles=0,warm_clean=0) for v in VARIANTS}
    records=[]
    for regime,rec,n in metadata["groups"]:
        from .common import group_name
        folder=directory/"groups"/group_name(regime,rec,n)
        seed=read(folder/"seed.json")
        check("Seed hash "+folder.name,sha256(np.asarray(seed["state"],dtype=np.float64).tobytes()).hexdigest()==seed["state_sha256"])
        for variant in VARIANTS:
            path=folder/(variant+".json");d=read(path)
            check("Source and seed "+str(path),d["source_unchanged"] and d["source_before"]==metadata["source"] and d["initial_state_sha256"]==seed["state_sha256"])
            check("Final state hash "+str(path),sha256(np.asarray(d["state"],dtype=np.float64).tobytes()).hexdigest()==d["state_sha256"])
            check("Single-grid implicit "+str(path),d["scheme"]["time_integrator"]=="euler-implicit")
            check("Clean warm timing "+str(path),d["compilation_free"] and d["compilation"]["backend_requests"]==0 and d["warm_s"] is not None)
            if d["root_pass"]:
                a=d["audit"]
                check("Original root gates "+str(path),d["success"] and a["passed"] and d["branch"]["passed"] and max(a["equation_norm"])<=1e-9 and all(a[k]<=1e-9 for k in ("boundary_max","constraint_max","global_max","invariant_max")))
            if d["profile"]["status"]=="matches":
                check("Profile gates "+str(path),d["root_pass"] and all(d["profile"]["checks"].values()))
            warmup=read(folder/(variant+"__compile.json"))
            check("Excluded seed unchanged "+str(path),warmup["initial_state_sha256"]==d["initial_state_sha256"])
            for key,phase in (("accepted_steps","accepted"),("rejected_steps","rejected")):
                check(key+" "+str(path),d["work"][key]==sum(r["phase"]==phase for r in d["history"]))
            c=counts[variant];c["attempts"]+=1;c["roots"]+=d["root_pass"];c["profiles"]+=d["profile"]["status"]=="matches";c["warm_clean"]+=d["compilation_free"]
            records.append(dict(file=str(path.relative_to(directory)),sha256=sha(path)))
    save(output,dict(passed=True,checks=checks,counts=counts,records=records,audit_sha256=sha(Path(__file__))))
    print("SAVED AUDIT PASSED",len(checks),counts,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("kernels","saved"))
    parser.add_argument("--directory",type=Path,default=STUDY/"outputs/ablation_20261104")
    parser.add_argument("--cache",type=Path,default=ROOT/".jax_cache")
    args=parser.parse_args()
    if args.mode=="kernels":
        kernels(STUDY/"outputs/kernel_audit.json",args.cache)
    else:
        saved(args.directory,STUDY/"outputs/saved_audit.json")
