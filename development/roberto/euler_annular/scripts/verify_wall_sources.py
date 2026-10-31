"""Verify prescribed shear/heat terms without assuming an empirical wall law.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.verify_wall_sources

The homogeneous source problem has an independent exact solution before the
velocity reaches zero: density and velocity direction stay fixed, speed falls
linearly under constant shear magnitude, and total energy rises linearly under
constant heat flux. Constant-radius steady solver runs then check the original
PDE, every boundary equation and independently reconstructed wall budgets.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.models.euler_annular.analytical.transient import exact_homogeneous

from nozzle_hyperbolic.models.euler_annular import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.sources import integrated_source
from nozzle_hyperbolic.models.euler_annular.state import conservative_to_primitive
from .recommended_controller import solve_recommended
from .verify_annular import fingerprints, write_json


STUDY = Path(__file__).resolve().parents[1]
LINESTYLES = {"shear": "--", "heat": ":", "combined": "-"}




def homogeneous_checks(fluid, output):
    initial = dict(density=1.2, temperature=300., v_m=30., v_theta=-40., height=.005, radius=.1)
    cases = {"shear": dict(tau_w=2., q_w=0.),
             "heat": dict(tau_w=0., q_w=1000.),
             "combined": dict(tau_w=2., q_w=1000.)}
    times = np.linspace(0, .01, 101)
    records, trajectories = {}, {}
    for name, wall in cases.items():
        settings = dict(initial, **wall)
        q0, _, _ = exact_homogeneous(0., fluid, **settings)
        rows, defects, recovery_errors = [], [], []
        for time in times:
            exact, primitive, temp = exact_homogeneous(time, fluid, **settings)
            source = np.asarray(integrated_source(
                jnp.asarray(primitive), initial["radius"], 0., .003, .003, 1., fluid,
                height=initial["height"], **wall))
            # Direction is fixed, so this exact solution also satisfies every
            # backward-Euler source update, not just the differential equation.
            scale = np.maximum(abs(q0), np.array([1., 1., 1., 1.]))
            defects.append(float(np.max(abs(exact-q0-time*source)/scale)))
            recovered = np.asarray(conservative_to_primitive(
                exact, initial["radius"], fluid, pressure_bounds=(1e4, 1e6)))
            recovery_errors.append(float(np.max(abs(recovered-primitive)
                                                   /np.maximum(abs(primitive), 1.))))
            rows.append(dict(time_s=float(time), speed_m_per_s=float(np.hypot(primitive[1], primitive[2]/initial["radius"])),
                             temperature_K=float(temp), total_energy_J_per_kg=float(exact[3]/exact[0])))
        final = rows[-1]
        initial_energy = rows[0]["total_energy_J_per_kg"]
        expected_energy = initial_energy+2*wall["q_w"]*times[-1]/(initial["density"]*initial["height"])
        assert max(defects) < 5e-13
        assert max(recovery_errors) < 5e-13
        assert np.isclose(final["total_energy_J_per_kg"], expected_energy, rtol=1e-14)
        if wall["tau_w"]:
            assert final["speed_m_per_s"] < rows[0]["speed_m_per_s"]
        assert final["temperature_K"] > initial["temperature"]
        records[name] = dict(settings=settings, maximum_scaled_BE_source_defect=max(defects),
                             maximum_scaled_primitive_recovery_error=max(recovery_errors),
                             initial=rows[0], final=final, passed=True)
        trajectories[name] = rows
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for color, (name, rows) in zip(plt.cm.plasma(np.linspace(.2, .8, 3)), trajectories.items()):
        for axis, key in zip(axes, ("speed_m_per_s", "temperature_K", "total_energy_J_per_kg")):
            axis.plot(times, [row[key] for row in rows], label=name, color=color,
                      linestyle=LINESTYLES[name])
    for axis, label in zip(axes, ("Speed [m/s]", "Temperature [K]", "Total energy [J/kg]")):
        axis.set(xlabel="Physical time [s]", ylabel=label)
        axis.grid(alpha=.25)
    axes[0].legend()
    for extension in ("png", "pdf"):
        fig.savefig(output/f"homogeneous_sources.{extension}", dpi=180)
    plt.close(fig)
    write_json(output/"homogeneous.json", dict(records=records, trajectories=trajectories))
    return records


def independent_budget(solver, state, tau_w, q_w):
    """Reconstruct physical wall loads from EOS fields, never solver.source()."""
    mesh = solver.mesh
    fields = {name: np.asarray(value) for name, value in solver.fields(state).items()}
    velocity = np.stack((fields["v_m"], fields["v_theta"]), axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    direction = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity),
                          where=speed[:, None] > 0)
    wall_measure = 2*mesh.volumes/mesh.b
    exact_source = np.zeros((mesh.n_cells, 4))
    exact_source[:, 1] = -wall_measure*tau_w*direction[:, 0]
    exact_source[:, 2] = -mesh.r_center*wall_measure*tau_w*direction[:, 1]
    exact_source[:, 3] = wall_measure*q_w
    numerical_flux = np.asarray(solver.face_flux(state))
    balance = numerical_flux[-1]-numerical_flux[0]-np.sum(exact_source, axis=0)
    ref = np.array([solver.reference_density,
                    solver.reference_density*solver.reference_sound_speed,
                    solver.reference_density*mesh.r_face[0]*solver.reference_sound_speed,
                    solver.reference_density*solver.reference_sound_speed**2])
    flux_scale = mesh.A_face[0]*solver.reference_sound_speed*ref
    scaled = balance/flux_scale
    source_error = np.max(abs(np.asarray(solver.source(state))-exact_source))
    # The third equation already transports angular momentum and wall torque.
    torque_balance = balance[2]
    return dict(integrated_source=np.sum(exact_source, axis=0),
                flux_difference=numerical_flux[-1]-numerical_flux[0],
                balance=balance, balance_scaled=scaled,
                maximum_scaled_defect=float(np.max(abs(scaled))),
                torque_balance_N_m=float(torque_balance),
                independent_vs_model_source_max=float(source_error), fields=fields)


def steady_checks(fluid, output, cells, max_steps):
    geometry = conical_channel(length=.05, r_in=.1, b_in=.005, phi=0.)
    mesh = geometry.mesh(cells)
    cases = {"shear": dict(tau_w=1., q_w=0.),
             "heat": dict(tau_w=0., q_w=1000.),
             "combined": dict(tau_w=1., q_w=1000.)}
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for color, (name, wall) in zip(plt.cm.plasma(np.linspace(.2, .8, 3)), cases.items()):
        solver = make_annular_solver(fluid, mesh,
            inlet_data=dict(p0=200000., T0=300., alpha_in=.3), outlet_pressure=199000.,
            reconstruction="muscl_van_leer", newton_iterations=8, **wall)
        initial = solver.initial_state()
        solution, controller = solve_recommended(solver, initial, max_steps=max_steps)
        budget = independent_budget(solver, solution.state, **wall)
        fields = budget.pop("fields")
        metrics = solver._audit(solution.state)
        passed = (solution.success and max(metrics[key] for key in
                  ("spatial_rms", "boundary_max", "conservation_max")) < 1e-8
                  and budget["maximum_scaled_defect"] < 1e-8
                  and budget["independent_vs_model_source_max"] < 1e-10)
        if solution.success and wall["tau_w"]:
            passed = passed and bool(fields["ell"][-1] < fields["ell"][0])
        if solution.success and wall["q_w"]:
            passed = passed and bool(fields["h0"][-1] > fields["h0"][0])
        record = dict(settings=dict(wall, cells=cells, length_m=.05, radius_m=.1,
                                    height_m=.005, p0_Pa=200000., T0_K=300.,
                                    outlet_pressure_Pa=199000., alpha_in_rad=.3),
                      success=solution.success, message=solution.message,
                      passed=passed, metrics=metrics, independent_budget=budget,
                      controller=controller, state=solution.state,
                      fields=fields, initial_state=initial, history=solution.history)
        write_json(output/f"steady_{name}.json", record)
        results[name] = {key: value for key, value in record.items()
                         if key not in ("state", "fields", "initial_state", "history")}
        for axis, key in zip(axes, ("p", "ell", "h0")):
            axis.plot(mesh.m_centers, fields[key], color=color, label=name,
                      linestyle=LINESTYLES[name])
        print(f"wall {name}: success={solution.success}, passed={passed}, "
              f"independent budget={budget['maximum_scaled_defect']:.3e}", flush=True)
    for axis, label in zip(axes, ("Pressure [Pa]", "Angular momentum [m²/s]", "Total enthalpy [J/kg]")):
        axis.set(xlabel="Meridional arclength [m]", ylabel=label)
        axis.grid(alpha=.25)
    axes[0].legend()
    for extension in ("png", "pdf"):
        fig.savefig(output/f"steady_wall_sources.{extension}", dpi=180)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs"/"wall_sources")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = fingerprints()
    fluid = jxp.FluidPerfectGas(name="Air")
    homogeneous = homogeneous_checks(fluid, args.output)
    steady = steady_checks(fluid, args.output, args.cells, args.max_steps)
    passed = all(item["passed"] for item in homogeneous.values()) and all(
        item["passed"] for item in steady.values())
    assert before == fingerprints(), "Numerical sources changed during wall verification"
    report = dict(passed=passed, prescribed_sources_only=True,
                  empirical_wall_closure_implemented=False,
                  homogeneous=homogeneous, steady=steady,
                  source_fingerprints=before,
                  driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/"audit.json", report)
    print(json.dumps(dict(passed=passed, output=str(args.output)), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
