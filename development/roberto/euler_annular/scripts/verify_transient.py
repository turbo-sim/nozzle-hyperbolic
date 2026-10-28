"""Physical-time advection of a compact entropy contact in a uniform annulus.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.verify_transient

The general-EOS production solver advances both reconstructions with the same
physical CFL. Independent split Gaussian quadrature supplies exact conservative
cell averages. An accepted-step callback audits the actual backward-Euler
storage/boundary-flux/source budget without duplicating the time integrator.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.models.euler_annular import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.state import primitive_to_conservative

from .verify_annular import LABELS, RECONSTRUCTIONS, REPOSITORY, STUDY, fingerprints, write_json


def density_pulse(m, time, density_base, settings):
    """Exact compact C-infinity density contact, translated at prescribed v_m."""
    coordinate = ((np.asarray(m)-settings["center"]-settings["v_m"]*time)
                  /settings["half_width"])
    bump = np.zeros_like(coordinate)
    active = np.abs(coordinate) < 1.0
    bump[active] = np.exp(1.0-1.0/(1.0-coordinate[active]**2))
    return density_base*(1.0+settings["density_amplitude"]*bump)


def cell_reference(fluid, mesh, time, density_base, settings, order=32):
    """Independent physical-volume quadrature; no numerical flux/state-map use."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    center = settings["center"]+settings["v_m"]*time
    edges = (center-settings["half_width"], center+settings["half_width"])
    coordinates, quadrature_weights, owners = [], [], []
    area = 2.0*np.pi*settings["radius"]*settings["width"]
    for index, (lower, upper) in enumerate(zip(mesh.m_faces[:-1], mesh.m_faces[1:])):
        splits = [lower, *(edge for edge in edges if lower < edge < upper), upper]
        for a, b in zip(splits[:-1], splits[1:]):
            coordinates.extend(0.5*(a+b)+0.5*(b-a)*nodes)
            quadrature_weights.extend(area*0.5*(b-a)*weights)
            owners.extend([index]*order)
    coordinates, quadrature_weights = map(np.asarray, (coordinates, quadrature_weights))
    owners = np.asarray(owners, dtype=int)
    density = density_pulse(coordinates, time, density_base, settings)
    pressure = np.full_like(density, settings["pressure"])
    state = fluid.get_state(jxp.DmassP_INPUTS, density, pressure)
    enthalpy = np.asarray(state["h"])
    v_m, v_theta = settings["v_m"], settings["v_theta"]
    ell = settings["radius"]*v_theta
    kinetic = 0.5*(v_m**2+v_theta**2)
    values = np.stack((density, density*v_m, density*ell,
                       density*(enthalpy+kinetic)-pressure), axis=-1)
    integrated = np.zeros((mesh.n_cells, 4))
    np.add.at(integrated, owners, quadrature_weights[:, None]*values)
    conservative = integrated/mesh.volumes[:, None]
    density_average = conservative[:, 0]
    # For this perfect-gas constant-pressure contact, the conservative cell
    # projection preserves p, v_m and ell. Recover h through the backend's
    # density/pressure input pair, independently of the numerical state map.
    projected = fluid.get_state(jxp.DmassP_INPUTS, density_average,
                                np.full(mesh.n_cells, settings["pressure"]))
    enthalpy_average = np.asarray(projected["h"])
    energy_check = density_average*(enthalpy_average+kinetic)-settings["pressure"]
    np.testing.assert_allclose(energy_check, conservative[:, 3], rtol=2e-12, atol=2e-9)
    W = np.column_stack((np.full(mesh.n_cells, settings["pressure"]),
                         np.full(mesh.n_cells, v_m), np.full(mesh.n_cells, ell),
                         enthalpy_average))
    fields = dict(p=W[:, 0], v_m=W[:, 1], ell=W[:, 2], h=W[:, 3],
                  v_theta=np.full(mesh.n_cells, v_theta), rho=density_average,
                  T=np.asarray(projected["T"]), s=np.asarray(projected["s"]),
                  h0=enthalpy_average+kinetic)
    return dict(W=W, q=conservative, integrated=integrated, fields=fields,
                quadrature_order=order, quadrature_split_at_pulse_edges=True)


def centroid(density, mesh, density_base):
    excess = mesh.volumes*(np.asarray(density)-density_base)
    amount = np.sum(excess)
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("A positive finite pulse excess is required for its centroid")
    return float(np.sum(excess*mesh.m_centers)/amount)


def verify_mesh(fluid, geometry, cells, settings, output, max_steps):
    mesh = geometry.mesh(cells)
    base = fluid.get_state(jxp.PT_INPUTS, settings["pressure"], settings["temperature"])
    density_base, enthalpy_base, entropy_base = (float(base[key]) for key in ("rho", "h", "s"))
    kinetic = 0.5*(settings["v_m"]**2+settings["v_theta"]**2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, enthalpy_base+kinetic, entropy_base)
    inlet = dict(h0=enthalpy_base+kinetic, s0=entropy_base, p0=float(reservoir["p"]),
                 alpha_in=np.arctan2(settings["v_theta"], settings["v_m"]))
    initial = cell_reference(fluid, mesh, 0.0, density_base, settings)
    exact = cell_reference(fluid, mesh, settings["final_time"], density_base, settings)
    base_W = np.asarray([settings["pressure"], settings["v_m"],
                         settings["radius"]*settings["v_theta"], enthalpy_base])
    full_initial = np.vstack((base_W, initial["W"], base_W))
    result = dict(case="compact_entropy_contact", cells=cells, m=mesh.m_centers,
                  m_faces=mesh.m_faces, volumes=mesh.volumes,
                  initial_reference=initial, reference=exact, initial_state=full_initial,
                  settings=dict(settings, max_steps=max_steps), methods={},
                  source_hashes=fingerprints(),
                  versions={name: metadata.version(name) for name in
                            ("jax", "jaxprop", "nurbspy", "numpy", "scipy")})
    adapter = REPOSITORY/"src/nozzle_hyperbolic/thermodynamics/adapters.py"
    if adapter.is_file():
        result["source_hashes"][str(adapter.relative_to(REPOSITORY))] = sha256(adapter.read_bytes()).hexdigest()
    for reconstruction in RECONSTRUCTIONS:
        solver = make_annular_solver(fluid, mesh, inlet_data=inlet,
                                     outlet_pressure=settings["pressure"], reconstruction=reconstruction,
                                     delta_factor=settings["delta_factor"])
        starting = jnp.asarray(full_initial)
        initial_bc = np.asarray(solver.boundary_residual(starting))
        content_scale = np.sum(mesh.volumes)*np.asarray([
            solver.reference_density, solver.reference_density*solver.reference_sound_speed,
            solver.reference_density*settings["radius"]*solver.reference_sound_speed,
            solver.reference_density*solver.reference_sound_speed**2])
        initial_content = np.asarray(solver.storage(starting[1:-1])).sum(axis=0)
        budget = dict(boundary=np.zeros(4), source=np.zeros(4), previous=initial_content.copy(), steps=[])

        def audit_step(record, state):
            dt = float(record["dt"])
            flux = np.asarray(solver.face_flux(state))
            source = np.asarray(solver.source(state)).sum(axis=0)
            content = np.asarray(solver.storage(state[1:-1])).sum(axis=0)
            boundary = flux[-1]-flux[0]
            defect = content-budget["previous"]+dt*(boundary-source)
            budget["boundary"] += dt*boundary
            budget["source"] += dt*source
            budget["previous"] = content
            budget["steps"].append(dict(
                time=float(record["time_new"]), dt=dt, content=content,
                boundary_flux=boundary, integrated_source=source,
                step_defect=defect, step_defect_scaled=defect/content_scale))

        solution = solver.advance(starting, settings["final_time"], cfl=settings["cfl"],
                                   max_steps=max_steps, callback=audit_step)
        state = jnp.asarray(solution.state)
        fields = {key: np.asarray(value) for key, value in solver.fields(state).items()}
        conservative = np.asarray(primitive_to_conservative(state[1:-1], mesh.r_center, fluid))
        q_scale = content_scale/np.sum(mesh.volumes)
        q_error = (np.sum(mesh.volumes[:, None]*np.abs(conservative-exact["q"]), axis=0)
                   /content_scale)
        field_scales = dict(rho=density_base, p=settings["pressure"], T=settings["temperature"],
                            v_m=solver.reference_sound_speed, v_theta=solver.reference_sound_speed,
                            ell=solver.scales["ell"], h=solver.scales["h"],
                            h0=solver.scales["h"], s=solver.scales["s"])
        field_errors = {key: float(np.sum(mesh.volumes*np.abs(fields[key]-reference))
                                    /np.sum(mesh.volumes)/field_scales[key])
                        for key, reference in exact["fields"].items()}
        final_content = np.asarray(solver.storage(state[1:-1])).sum(axis=0)
        total_defect = final_content-initial_content+budget["boundary"]-budget["source"]
        initial_centroid = centroid(initial["fields"]["rho"], mesh, density_base)
        final_centroid = centroid(fields["rho"], mesh, density_base)
        exact_centroid = centroid(exact["fields"]["rho"], mesh, density_base)
        time_reached = np.isclose(solution.physical_time, settings["final_time"], rtol=0,
                                 atol=16*np.finfo(float).eps)
        changed = bool(np.max(np.abs(conservative-initial["q"])/q_scale) > 1e-10)
        movement = final_centroid > initial_centroid
        root_checks = dict(
            success=solution.success, requested_time_reached=bool(time_reached),
            state_changed=changed, pulse_moves_forward=bool(movement),
            initial_boundary_max=float(np.max(np.abs(initial_bc))),
            accepted_steps=len(budget["steps"]),
            accepted_dt_sum=float(sum(row["dt"] for row in budget["steps"])))
        result["methods"][reconstruction] = dict(
            checks=root_checks, message=solution.message, physical_time=solution.physical_time,
            effective_options=asdict(solver.options),
            state=solution.state, fields=fields, history=solution.history,
            initial_boundary_constraints=initial_bc,
            final_boundary_constraints=np.asarray(solver.boundary_residual(state)),
            conservative_L1=q_error, field_L1=field_errors,
            density_min=float(np.min(fields["rho"])), density_max=float(np.max(fields["rho"])),
            centroid=dict(initial=initial_centroid, final=final_centroid, exact=exact_centroid,
                          displacement=final_centroid-initial_centroid,
                          displacement_error=final_centroid-exact_centroid),
            physical_budget=dict(
                convention="Q_final-Q_initial+sum(dt*(Fout_new-Fin_new-S_new))",
                initial_content=initial_content, final_content=final_content,
                content_change=final_content-initial_content,
                boundary_integral=budget["boundary"], source_integral=budget["source"],
                content_scales=content_scale, cumulative_defect=total_defect,
                cumulative_defect_scaled=total_defect/content_scale,
                accepted_steps=budget["steps"]))
        print(f"Transient N={cells} {reconstruction}: success={solution.success}, "
              f"steps={len(budget['steps'])}, rho_L1={field_errors['rho']:.6g}, "
              f"shift={final_centroid-initial_centroid:.6g} m, "
              f"budget={np.max(np.abs(total_defect/content_scale)):.3g}", flush=True)
    write_json(output/f"contact_{cells}.json", result)
    plot_fields(result, output)
    return result


def plot_fields(result, output):
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, 3))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    fields = (("rho", "Density [kg/m3]"), ("T", "Temperature [K]"),
              ("s", "Entropy [J/(kg K)]"), ("p", "Static pressure [Pa]"),
              ("v_m", "Meridional velocity [m/s]"), ("v_theta", "Tangential velocity [m/s]"))
    for ax, (key, label) in zip(axes.ravel(), fields):
        ax.plot(result["m"], result["reference"]["fields"][key], color=colors[0],
                label=LABELS[0], lw=1.8)
        for index, (method, style, marker) in enumerate(zip(RECONSTRUCTIONS, ("--", "-."), ("o", "^")), 1):
            run = result["methods"][method]
            legend = LABELS[index]+(" (failed)" if not run["checks"]["success"] else "")
            ax.plot(result["m"], run["fields"][key], style, marker=marker,
                    color=colors[index], markevery=max(1, result["cells"]//20), ms=3, label=legend)
        ax.set(xlabel="Meridional arclength [m]", ylabel=label)
        if np.ptp(result["reference"]["fields"][key]) < 1e-12:
            center = float(result["reference"]["fields"][key][0])
            extent = max(0.005*abs(center), 1e-6,
                         *(float(np.max(np.abs(np.asarray(run["fields"][key])-center)))
                           for run in result["methods"].values()))
            ax.set_ylim(center-1.1*extent, center+1.1*extent)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Entropy-contact advection, N={result['cells']}, t={result['settings']['final_time']:.6g} s")
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"contact_{result['cells']}_fields.{suffix}", dpi=160)
    plt.close(fig)


def write_report(results, acceptance, output):
    lines = ["# Physical-time entropy-contact audit", "",
             "A compact density/entropy pulse advects in a constant-radius NURBS annulus.",
             "Pressure and both velocity components stay constant in the exact solution.",
             "The same production physical-time driver advances both reconstructions at CFL 0.5.",
             "References are independent physical-volume conservative cell integrals, not point samples.",
             "Every accepted backward-Euler step contributes its actual numerical boundary flux/source budget.", "",
             f"Entropy regularization delta_factor={results[0]['settings']['delta_factor']:.6g} is recorded in each method's effective options.", "",
             "| Cells | Reconstruction | Success | Time [s] | Steps | Density L1 / base density | Shift [m] | Max scaled budget defect |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for result in results:
        for method, run in result["methods"].items():
            lines.append(f"| {result['cells']} | {method} | {run['checks']['success']} | "
                         f"{run['physical_time']:.6g} | {run['checks']['accepted_steps']} | "
                         f"{run['field_L1']['rho']:.6e} | {run['centroid']['displacement']:.6g} | "
                         f"{np.max(np.abs(run['physical_budget']['cumulative_defect_scaled'])):.3e} |")
    lines.extend(["", f"Time/state/movement checks passed: **{acceptance['all_physical_checks']}**.",
                  f"Density L1 decreases across the requested meshes: **{acceptance['density_error_decreases']}**.",
                  "No unrequested field-accuracy threshold was invented. Errors and budget defects are retained as measured.",
                  "Backward Euler is first order in physical time. Fixed-CFL mesh refinement changes space and time together;",
                  "this does not independently establish temporal order or validate every transient boundary regime.", ""])
    for result in results:
        lines.extend([f"## {result['cells']} physical cells", "",
                      f"![Analytical, constant and MUSCL fields](contact_{result['cells']}_fields.png)", "",
                      f"[PDF figure](contact_{result['cells']}_fields.pdf).", ""])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--delta-factor", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs"/"transient")
    args = parser.parse_args()
    if (any(cells < 20 for cells in args.cells) or not np.isfinite(args.cfl) or args.cfl <= 0
            or not np.isfinite(args.delta_factor) or args.delta_factor <= 0):
        parser.error("Use at least 20 cells, positive CFL and a finite positive entropy width")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(length=1.0, radius=0.2, width=0.005, pressure=1e5,
                    temperature=300.0, v_m=80.0, v_theta=20.0,
                    center=0.35, half_width=0.10, density_amplitude=0.20,
                    final_time=0.10/80.0, cfl=args.cfl, delta_factor=args.delta_factor)
    fluid = jxp.FluidPerfectGas(name="Air")
    geometry = conical_channel(settings["length"], settings["radius"], settings["width"], 0.0)
    results = [verify_mesh(fluid, geometry, cells, settings, args.output, args.max_steps)
               for cells in sorted(set(args.cells))]
    all_checks = all(all(run["checks"][key] for key in
                        ("success", "requested_time_reached", "state_changed", "pulse_moves_forward"))
                     for result in results for run in result["methods"].values())
    trends = {method: [result["methods"][method]["field_L1"]["rho"] for result in results]
              for method in RECONSTRUCTIONS}
    decrease = (len(results) > 1 and all(
        all(next_error < error for error, next_error in zip(errors[:-1], errors[1:]))
        for errors in trends.values()))
    acceptance = dict(all_physical_checks=all_checks, density_error_decreases=decrease,
                      density_L1_trends=trends, success=all_checks and decrease,
                      driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/"acceptance.json", acceptance)
    write_report(results, acceptance, args.output)
    return 0 if acceptance["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
