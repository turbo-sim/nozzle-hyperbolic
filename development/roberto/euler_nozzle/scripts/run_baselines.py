"""Run current-source steady and Sod baselines without modifying production code.

The steady controller is reused from the retained backpressure study. The
production FVS/Newton/BC/AD/banded operators are called unchanged. Additional
residual samples observe accepted states; they do not change stopping rules.
"""

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from .baseline_common import STUDY, config, provenance, save, source_hashes, steady_cases, validate_saved_campaign
from development.roberto.historical.backpressure_sensitivity.scripts.run_sweep import ImplicitNozzle, RMS_TOL
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import PerfectGasReference
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as steady_setup
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_sod as sod_setup
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.models.euler_nozzle.flux_jax import _jax_F_hll
from nozzle_hyperbolic.solvers.residuals import reduce_residuals


class ObservedNozzle(ImplicitNozzle):
    """Add recorded diagnostics to the existing study controller, not its RHS."""
    def __init__(self, points, reconstruction):
        super().__init__(points)
        self.settings["reconstruction"] = reconstruction
        self.samples = []
        qscale = jnp.array([self.rho0, self.rho0*self.a0, self.rho0*self.a0**2])
        volumes = jnp.asarray(self.area[1:-1]*self.dx)
        flux_scale = steady_setup.NOZZLE["A_throat"] * qscale * self.a0
        native_metrics = self.metrics

        @jax.jit
        def diagnostics(full, pb):
            ops = self.factory(pb)[4]
            residual = ops["steady_residual"](full[1:-1].ravel()).reshape(-1, 3)
            norms = reduce_residuals(residual, volumes, qscale,
                                     steady_setup.NOZZLE["L"]/self.a0)["l2"]
            st = self.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
            mapped = ops["state_from_interior"](full[1:-1].ravel())
            bc = jnp.max(jnp.abs((full-mapped)[jnp.array([0, full.shape[0]-1])]) /
                         jnp.array([steady_setup.P_STAGNATION, self.a0, self.h0]))
            return dict(equation_norm=norms, global_defect=jnp.abs(jnp.sum(residual, axis=0)/flux_scale),
                        boundary_map_defect=bc,
                        wave_max=jnp.max(jnp.abs(full[1:-1, 1])+st["a"][1:-1]))
        self.diagnostics = diagnostics

        def observe(full, pb):
            sample = jax.device_get(diagnostics(full, pb))
            self.samples.append({key: np.asarray(value).tolist() for key, value in sample.items()})
            return native_metrics(full, pb)
        self.metrics = observe


def fields(model, full):
    st = model.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
    rho, sound = np.asarray(st["rho"]), np.asarray(st["a"])
    return dict(p=full[:, 0], rho=rho, u=full[:, 1], M=full[:, 1]/sound,
                T=np.asarray(st["T"]), h0=full[:, 2]+0.5*full[:, 1]**2)


def steady_run(model, reference, case, cfg):
    pb = case["back_pressure_ratio"]*reference.p0
    model.samples = []
    initial = model.cold_initial(pb)
    full, stats = model.integrate(initial, pb, cfg["max_updates"])
    samples = model.samples
    initial_norm = np.asarray(samples[0]["equation_norm"])
    for row, sample in zip(stats["history"], samples, strict=True):
        row.update(sample)
        row["equation_relative"] = np.divide(sample["equation_norm"], initial_norm,
                                              out=np.zeros(3), where=initial_norm > 0).tolist()
    # A trial's CFL uses the pre-update wave speed, including rejected trials.
    for trial in stats["trials"]:
        trial["cfl"] = trial["dt_s"]*samples[trial["update"]-1]["wave_max"]/model.dx
    numerical = fields(model, full)
    exact = reference.profile(model.x, case["back_pressure_ratio"])
    dense = reference.profile(np.linspace(0, reference.length, 1601), case["back_pressure_ratio"])
    physical = slice(1, -1)
    scales = dict(p=reference.p0, rho=model.rho0, u=model.a0, M=1., T=reference.T0)
    errors = {key: float(np.mean(np.abs(numerical[key][physical]-exact[key][physical]))/scale)
              for key, scale in scales.items()}
    q_num = np.asarray(model.factory(pb)[4]["conservative_state"](jnp.asarray(full)))
    q_exact = np.stack([exact["rho"], exact["rho"]*exact["u"],
                        exact["p"]/(reference.gamma-1)+0.5*exact["rho"]*exact["u"]**2], axis=-1)
    q_scales = model.rho0*np.array([1, model.a0, model.a0**2])
    q_error = np.mean(np.abs(q_num[physical]-q_exact[physical]), axis=0)/q_scales
    mass = np.asarray(model.face_flux(jnp.asarray(full), jnp.float64(pb)))
    imbalance = float(abs(mass[-1]-mass[0])/max(abs(mass[0]), 1e-30))
    accuracy = bool(max(errors[key] for key in ("p", "rho", "M")) <
                    cfg["accuracy"]["normalized_mean_absolute_error"])
    throat = int(np.argmin(model.area))
    record = dict(case=case, points=len(model.x), physical_cells=len(model.x)-2,
                  reconstruction=model.settings["reconstruction"], settings=model.settings,
                  initialization="independent ordinary simple start; no exact field or continuation",
                  x=model.x, area=model.area, primitive=full, numerical=numerical,
                  reference={key: exact[key] for key in (*scales, "h0") if key in exact},
                  dense_reference={key: dense[key] for key in ("x", *scales)},
                  analytical_regime=exact["regime"], shock_exact_m=exact["shock_x"],
                  normalized_L1=errors, conservative_L1=q_error,
                  relative_boundary_mass_imbalance=imbalance, face_mass_flux=mass,
                  accuracy_passed=accuracy, conservation_passed=imbalance <
                  cfg["accuracy"]["relative_boundary_mass_imbalance"],
                  throat_M_numerical=float(numerical["M"][throat]),
                  throat_M_reference=float(exact["M"][throat]),
                  **stats)
    record["passed"] = bool(stats["converged"] and accuracy and record["conservation_passed"])
    return record


def run_steady(cfg, destination, points=None, case_ids=None):
    if cfg["spatial_rms_tolerance"] != RMS_TOL:
        raise ValueError("This baseline pins the retained controller's RMS tolerance; changing it requires a separate protocol")
    path = destination/"summary.json"
    current = source_hashes()
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(
        kind="steady", config=cfg, provenance=provenance(), runs=[],
        backend="production make_jax_stadtke_step; retained ImplicitNozzle study controller",
        stopping="fixed study RMS <= 1e-8 AND native SSR < 1e-8; not annular nine-order stopping")
    validate_saved_campaign(report,cfg)
    done = {(r["case"]["id"],r["points"],r["reconstruction"]) for r in report["runs"]}
    for count in (points or cfg["grid_points"]):
        for reconstruction in cfg["reconstructions"]:
            model = ObservedNozzle(count, reconstruction)
            reference = PerfectGasReference(model.fluid)
            report["landmarks"] = reference.landmarks()
            for case in steady_cases(reference, cfg["cases"]):
                if case_ids and case["id"] not in case_ids:
                    continue
                if (case["id"],count,reconstruction) in done:
                    continue
                print(f'Steady {case["id"]} nodes={count} {reconstruction}', flush=True)
                record = steady_run(model, reference, case, cfg)
                report["runs"].append(record)
                if source_hashes() != current:
                    raise RuntimeError("Production source changed during baseline")
                report["source_unchanged"] = True
                save(path, report)
                print(f'  converged={record["converged"]} accuracy={record["accuracy_passed"]} '
                      f'updates={len(record["history"])-1} Mth={record["throat_M_numerical"]:.6f}', flush=True)
    return report


def sod_run(cells, cfg):
    setup = sod_setup
    fluid = jxp.FluidPerfectGas(name="Air")
    gamma = float(fluid.constants.gamma)
    sound_left = np.sqrt(gamma*setup.LEFT["p"]/setup.LEFT["rho"])
    dx = setup.LENGTH/cells
    x = (np.arange(cells)+.5)*dx
    left = x < setup.DIAPHRAGM
    rho = np.where(left, setup.LEFT["rho"], setup.RIGHT["rho"])
    u = np.where(left, setup.LEFT["u"], setup.RIGHT["u"])
    p = np.where(left, setup.LEFT["p"], setup.RIGHT["p"])
    initial = np.stack([rho,rho*u,p/(gamma-1)+.5*rho*u**2])
    q = jnp.asarray(np.pad(initial,((0,0),(1,1)),mode="edge"))
    scales = setup.LENGTH*np.array([setup.LEFT["rho"],setup.LEFT["rho"]*sound_left,
                                   setup.LEFT["p"]/(gamma-1)])
    initial_total = dx*initial.sum(axis=1)

    @jax.jit
    def step(old, remaining):
        bc = old.at[:,0].set(old[:,1]).at[:,-1].set(old[:,-2])
        rho = bc[0]
        velocity = bc[1]/rho
        pressure = (gamma-1)*(bc[2]-.5*rho*velocity**2)
        sound = fluid.get_state(jxp.DmassP_INPUTS,rho,pressure)["a"]
        speed = jnp.max(jnp.abs(velocity)+sound)
        dt = jnp.minimum(setup.CFL*dx/speed,remaining)
        area = jnp.ones(bc.shape[1])
        flux = _jax_F_hll(bc[:,:-1],bc[:,1:],pressure[:-1],sound[:-1],
                          pressure[1:],sound[1:],area[:-1],area[1:])
        new = bc.at[:,1:-1].add(-dt/dx*(flux[:,1:]-flux[:,:-1]))
        return new,dt,dt*(flux[:,0]-flux[:,-1]),speed

    snapshots, history = [], []
    integral = np.zeros(3)
    time = 0.
    started = perf_counter()
    args = (setup.DIAPHRAGM,setup.LEFT["rho"],setup.LEFT["u"],setup.LEFT["p"],
            setup.RIGHT["rho"],setup.RIGHT["u"],setup.RIGHT["p"],gamma)
    for tau in cfg["output_tau"]:
        target = tau*setup.LENGTH/sound_left
        while time < target:
            q,dt,influx,speed = step(q,jnp.float64(target-time))
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0:
                raise RuntimeError("Invalid Sod timestep")
            integral += np.asarray(influx)
            time = min(target,time+dt)
            qn = np.asarray(q[:,1:-1])
            budget = (dx*qn.sum(axis=1)-initial_total-integral)/scales
            history.append(dict(iteration=len(history)+1,time_s=time,tau=time*sound_left/setup.LENGTH,
                                dt_s=dt,cfl=dt*float(speed)/dx,conservation_error=budget))
        rho = qn[0]
        u = qn[1]/rho
        p = (gamma-1)*(qn[2]-.5*rho*u**2)
        numerical = dict(rho=rho,u=u,p=p,M=u/np.sqrt(gamma*p/rho),
                         T=p/(rho*float(fluid.constants.R)))
        if not np.isfinite(qn).all() or np.min(rho) <= 0 or np.min(p) <= 0:
            raise RuntimeError("Nonpositive or nonfinite Sod state")
        exact = dict(zip(("rho","u","p"),sod_exact(x,time,*args)))
        dense_x = np.linspace(0,setup.LENGTH,4001)
        dense = dict(zip(("rho","u","p"),sod_exact(dense_x,time,*args)))
        for profile in (exact,dense):
            profile["M"] = profile["u"]/np.sqrt(gamma*profile["p"]/profile["rho"])
            profile["T"] = profile["p"]/(profile["rho"]*float(fluid.constants.R))
        for key in ("rho","u","p"):
            np.testing.assert_allclose(dense[key][[0,-1]],[setup.LEFT[key],setup.RIGHT[key]],rtol=1e-10)
        errors = {k:float(np.mean(np.abs(numerical[k]-exact[k]))/scale)
                  for k,scale in dict(rho=setup.LEFT["rho"],u=sound_left,p=setup.LEFT["p"]).items()}
        snapshots.append(dict(tau=tau,time_s=time,steps=len(history),x=x,conservative=qn.T,
                              numerical=numerical,reference=exact,dense_reference=dict(x=dense_x,**dense),
                              normalized_L1=errors,normalized_conservation_error=budget,
                              passed=max(errors.values())<cfg["accuracy"]["normalized_mean_absolute_error"]
                              and np.max(np.abs(budget))<cfg["accuracy"]["normalized_conservation_error"]))
    return dict(physical_cells=cells,gamma=gamma,method="HLL / constant / forward Euler",
                CFL=setup.CFL,snapshots=snapshots,history=history,elapsed_s=perf_counter()-started,
                passed=all(s["passed"] for s in snapshots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("steady","sod"))
    parser.add_argument("--output",type=type(STUDY))
    parser.add_argument("--grids",type=int,nargs="+")
    parser.add_argument("--cases",nargs="+")
    args = parser.parse_args()
    destination = args.output or STUDY/"outputs"/f"baseline_{args.kind}"
    cfg = config(args.kind)
    before = source_hashes()
    if args.kind == "steady":
        run_steady(cfg,destination,args.grids,args.cases)
    else:
        path=destination/"summary.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(kind="sod",config=cfg,provenance=provenance(),runs=[],
                      physical_setup=dict(length_m=sod_setup.LENGTH,diaphragm_m=sod_setup.DIAPHRAGM,
                                          left=sod_setup.LEFT,right=sod_setup.RIGHT))
        validate_saved_campaign(report,cfg)
        done={row["physical_cells"] for row in report["runs"]}
        for cells in args.grids or cfg["physical_cells"]:
            if cells in done:
                continue
            print(f"Sod {cells} cells",flush=True)
            report["runs"].append(sod_run(cells,cfg))
            report["source_unchanged"] = source_hashes()==before
            if not report["source_unchanged"]:
                raise RuntimeError("Production source changed during baseline")
            save(path,report)
            print(f'  passed={report["runs"][-1]["passed"]}',flush=True)


if __name__ == "__main__":
    main()
