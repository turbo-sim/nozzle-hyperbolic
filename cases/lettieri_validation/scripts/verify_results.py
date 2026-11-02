"""Re-evaluate saved fields and numerical flux budgets without advancing them."""

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from nozzle_hyperbolic.solvers.convergence_output import strict_json
from .data import ROOT
from .run import make_mesh, prepare_fluid, provenance


def main():
    output = ROOT / "outputs"
    source = provenance()
    fluid, prop = prepare_fluid(output)
    for result_file in sorted(output.glob("*/result.json")):
        record = json.loads(result_file.read_text(encoding="utf-8"))
        state_file = result_file.with_name("solution.npz")
        if not state_file.exists():
            continue
        saved = np.load(state_file)
        state = jnp.asarray(saved["state"])
        mesh, width = make_mesh(record["cells"], record["stretch"])
        case = record["definition"]
        model = NozzleModel(mesh, fluid, p0=case["p0_bar"]*1e5, T0=case["T0_K"],
                            back_pressure=0.25*case["p0_bar"]*1e5, outlet_regime="supersonic",
                            wall_closure=None, wall_perimeter=2*(width+2*mesh.geometry(mesh.centers)))
        op = Operator(model, Scheme(**record["scheme"]))
        raw = np.asarray(op.spatial_residual(state))
        faces = np.asarray(op.face_flux(state))
        source_terms = np.asarray(model.integrated_source(state))
        scale = np.asarray(model.conservative_scales)
        dimensionless = raw * model.reference_time / (mesh.volumes[:, None]*scale)
        l2 = np.sqrt(np.sum(mesh.volumes[:, None]*dimensionless**2, axis=0)/np.sum(mesh.volumes))
        boundary_max = float(np.max(np.abs(np.asarray(op.boundary_residual(state, 0.0)))))
        signed_budget = faces[-1] - faces[0] - source_terms.sum(axis=0)
        global_defect = np.abs(signed_budget / model.global_flux_scales)
        identity_defect = np.abs((raw.sum(axis=0)-signed_budget)/model.global_flux_scales)
        endpoint = state[jnp.array([0, -1])]
        endpoint_mach = np.asarray(endpoint[:, 1]/model.thermodynamic_state(endpoint)["a"])
        original_tol = record["steady"]["residual"]
        absolute_pass = bool(np.all(l2 <= original_tol["absolute_tolerance"]) and
                             boundary_max <= original_tol["boundary_tolerance"] and
                             np.all(global_defect <= original_tol["global_tolerance"]))
        admissible = bool(op.is_admissible(state))
        branch = bool(model.solution_audit(state))
        verification = dict(source=source, property_table_sha256=prop["table_sha256"],
                            original_solver_success=record["success"], absolute_targets_pass=absolute_pass,
                            admissible=admissible, physical_branch_pass=branch,
                            verified=absolute_pass and admissible and branch and bool(np.all(identity_defect<1e-12)),
                            equation_l2=l2, boundary_max=boundary_max, global_defect=global_defect,
                            signed_boundary_source_budget=signed_budget, telescoping_identity_defect=identity_defect,
                            endpoint_mach=endpoint_mach,
                            mass_flow=dict(numerical_inlet_kg_s=float(faces[0,0]), numerical_outlet_kg_s=float(faces[-1,0]),
                                           numerical_range_kg_s=[float(faces[:,0].min()),float(faces[:,0].max())],
                                           half_channel_per_depth_kg_s_m=float(faces[0,0]/(2*width)),
                                           cell_physical_mean_kg_s=float(np.mean(saved["mdot"])),
                                           note="Cell rho*u*A differs from the conservative numerical face flux on a finite grid"))
        result_file.with_name("verification.json").write_text(strict_json(verification), encoding="utf-8")
        print(f"{result_file.parent.name}: verified={verification['verified']}; numerical mass flow={faces[0,0]:.8g} kg/s", flush=True)


if __name__ == "__main__":
    main()
