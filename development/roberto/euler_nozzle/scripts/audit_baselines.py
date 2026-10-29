"""Audit saved steady operators and independently replay Sod's discrete budgets."""

import argparse
import json

import jax.numpy as jnp
import numpy as np

from .baseline_common import STUDY, save, source_hashes
from .run_baselines import ObservedNozzle, fields
from development.roberto.backpressure_sensitivity.scripts.reference import PerfectGasReference
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact


def audit_steady(data):
    records=[]
    for points in data["config"]["grid_points"]:
        for method in data["config"]["reconstructions"]:
            model=ObservedNozzle(points,method)
            ref=PerfectGasReference(model.fluid)
            for row in (r for r in data["runs"] if r["points"]==points and r["reconstruction"]==method):
                full=jnp.asarray(row["primitive"])
                pb=jnp.float64(row["case"]["back_pressure_ratio"]*ref.p0)
                rms,native=(float(v) for v in model.metrics(full,pb))
                diagnostics=model.diagnostics(full,pb)
                numerical=fields(model,np.asarray(full))
                exact=ref.profile(model.x,row["case"]["back_pressure_ratio"])
                for key,scale in dict(p=ref.p0,rho=model.rho0,u=model.a0,M=1,T=ref.T0).items():
                    np.testing.assert_allclose(numerical[key],row["numerical"][key],rtol=2e-12,atol=2e-12*scale)
                    err=float(np.mean(np.abs(numerical[key][1:-1]-exact[key][1:-1]))/scale)
                    np.testing.assert_allclose(err,row["normalized_L1"][key],rtol=1e-10,atol=1e-12)
                tail=row["history"][-1]
                np.testing.assert_allclose([rms,native],[tail["spatial_rms"],tail["source_ssr"]],rtol=2e-6,atol=1e-13)
                np.testing.assert_allclose(diagnostics["global_defect"],tail["global_defect"],rtol=2e-6,atol=1e-12)
                np.testing.assert_allclose(diagnostics["equation_norm"],tail["equation_norm"],rtol=2e-6,atol=1e-12)
                assert bool(rms<=1e-8 and native<1e-8)==row["converged"]
                assert float(diagnostics["boundary_map_defect"])<1e-10
                assert all(h["accepted"] or h["retry"]<9 for h in row["trials"])
                expected_updates=list(range(len(row["history"])))
                assert [h["update"] for h in row["history"]]==expected_updates
                accepted=[h for h in row["trials"] if h["accepted"]]
                assert [h["update"] for h in accepted]==expected_updates[1:]
                records.append(dict(case=row["case"]["id"],points=points,method=method,
                                    recomputed_rms=rms,recomputed_native_ssr=native,passed=True))
    assert len(records)==len(data["runs"])==64
    return records


def audit_sod(data):
    """Independent NumPy HLL replay with recorded dt; not a second baseline solver."""
    setup=data["physical_setup"]
    left,right=setup["left"],setup["right"]
    records=[]
    for row in data["runs"]:
        cells,gamma=row["physical_cells"],row["gamma"]
        dx=setup["length_m"]/cells
        x=(np.arange(cells)+.5)*dx
        mask=x<setup["diaphragm_m"]
        rho=np.where(mask,left["rho"],right["rho"])
        p=np.where(mask,left["p"],right["p"])
        u=np.where(mask,left["u"],right["u"])
        q=np.stack([rho,rho*u,p/(gamma-1)+.5*rho*u*u])
        initial=dx*q.sum(axis=1)
        a_left=np.sqrt(gamma*left["p"]/left["rho"])
        scales=setup["length_m"]*np.array([left["rho"],left["rho"]*a_left,left["p"]/(gamma-1)])
        influx=np.zeros(3)
        snapshots={s["steps"]:s for s in row["snapshots"]}
        time=0.
        for history in row["history"]:
            bc=np.pad(q,((0,0),(1,1)),mode="edge")
            rho=bc[0]
            u=bc[1]/rho
            p=(gamma-1)*(bc[2]-.5*rho*u*u)
            a=np.sqrt(gamma*p/rho)
            physical_flux=np.stack([rho*u,rho*u*u+p,(bc[2]+p)*u])
            sl=np.minimum(u[:-1]-a[:-1],u[1:]-a[1:])
            sr=np.maximum(u[:-1]+a[:-1],u[1:]+a[1:])
            fl,fr=physical_flux[:,:-1],physical_flux[:,1:]
            flux=np.where(sl>=0,fl,np.where(sr<=0,fr,
                (sr*fl-sl*fr+sl*sr*(bc[:,1:]-bc[:,:-1]))/(sr-sl)))
            dt=history["dt_s"]
            assert dt>0
            np.testing.assert_allclose(dt*np.max(np.abs(u)+a)/dx,history["cfl"],rtol=1e-12)
            assert history["cfl"]<=row["CFL"]*(1+1e-12)
            q-=dt/dx*(flux[:,1:]-flux[:,:-1])
            influx+=dt*(flux[:,0]-flux[:,-1])
            time+=dt
            np.testing.assert_allclose(time,history["time_s"],rtol=1e-12)
            budget=(dx*q.sum(axis=1)-initial-influx)/scales
            np.testing.assert_allclose(budget,history["conservation_error"],atol=5e-14)
            if history["iteration"] in snapshots:
                snapshot=snapshots[history["iteration"]]
                np.testing.assert_allclose(q.T,snapshot["conservative"],rtol=2e-12,atol=1e-12)
                exact=dict(zip(("rho","u","p"),sod_exact(x,time,setup["diaphragm_m"],
                    left["rho"],left["u"],left["p"],right["rho"],right["u"],right["p"],gamma)))
                rho=q[0];u=q[1]/rho;p=(gamma-1)*(q[2]-.5*rho*u*u)
                for key,values,scale in (("rho",rho,left["rho"]),("u",u,a_left),("p",p,left["p"])):
                    error=np.mean(np.abs(values-exact[key]))/scale
                    np.testing.assert_allclose(error,snapshot["normalized_L1"][key],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(time,snapshot["tau"]*setup["length_m"]/a_left,rtol=1e-12)
        records.append(dict(cells=cells,steps=len(row["history"]),independent_numpy_replay=True,
                            maximum_final_budget=float(np.max(np.abs(budget))),passed=True))
    assert len(records)==4
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("steady","sod"))
    kind=parser.parse_args().kind
    destination=STUDY/"outputs"/f"baseline_{kind}"
    data=json.loads((destination/"summary.json").read_text(encoding="utf-8"))
    assert data["provenance"]["source_hashes"]==source_hashes(),"Source differs from baseline"
    records=audit_steady(data) if kind=="steady" else audit_sod(data)
    save(destination/"audit.json",dict(kind=kind,passed_runs=sum(r["passed"] for r in records),
        source_unchanged=True,records=records,
        scope="Saved-state consistency, source residuals and independent reference errors" if kind=="steady"
        else "Independent NumPy replay of every recorded timestep, final fields, clock and boundary-flux budgets"))
    print(f"{kind}: {len(records)}/{len(records)} saved-state audits passed")


if __name__=="__main__":
    main()
