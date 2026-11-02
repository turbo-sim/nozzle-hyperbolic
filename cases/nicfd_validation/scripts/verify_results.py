"""Independently audit saved fields without advancing the flow solution."""

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator
from .case import ROOT, specification, prepare_fluids, build_case
from .run import write_json
from .run import regime_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
    paths = args.runs or [ROOT / protocol["campaign"]]
    fluids, _ = prepare_fluids(specification("subsonic"))
    rows = []
    for path in paths:
        if path.is_dir():
            path = path / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, digest in record["source_hashes"].items():
            if name.startswith("src/"):
                assert hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest() == digest, name
        for stage in record["stages"]:
            if "fields" not in stage:
                continue
            model, scheme, settings, _ = build_case(stage["source_configuration"], fluids, stage["cells"],
                initialization=stage["effective"]["initialization"],
                stretch=record["arguments"]["stretch"],
                newton_iterations=stage["effective"]["scheme"]["newton_iterations"],
                area_law=stage["effective"]["closures"]["area_law"])
            op = Operator(model, scheme)
            saved = np.load(path.parent / stage["fields"])
            W = jnp.asarray(saved["state"])
            residual = np.asarray(op.physical_spatial_residual(W))
            scaled = residual * model.reference_time / (
                model.mesh.volumes[:, None] * model.conservative_scales)
            independent_l2 = np.sqrt(np.sum(model.mesh.volumes[:, None] * scaled**2, axis=0)
                                     / sum(model.mesh.volumes))
            np.testing.assert_allclose(independent_l2, stage["metrics"]["equation_l2"], rtol=2e-10, atol=1e-14)
            constraints = np.asarray(model.cell_constraints(W[1:-1])) / model.constraint_scales
            bc = np.asarray(op.boundary_residual(W, 0.))
            global_defect = np.sum(residual, axis=0) / model.global_flux_scales
            invariant = model.invariant_projections @ np.sum(residual, axis=0) / model.invariant_flux_scales
            contributions = {key: np.sum(np.asarray(value), axis=0)
                             for key, value in op.residual_contributions(W).items()}
            assembled = (contributions["conservative"] + contributions["stabilization"]
                         + contributions["nonconservative"] - contributions["source"])
            np.testing.assert_allclose(assembled, np.sum(residual, axis=0), rtol=1e-9, atol=1e-8)
            row = dict(run=str(path.parent.relative_to(ROOT)), regime=stage["regime"], cells=stage["cells"],
                original_success=stage["success"], reason=stage["reason"], equation_l2=independent_l2,
                constraint_max=float(np.max(abs(constraints))), boundary_max=float(np.max(abs(bc))),
                signed_global_defect=global_defect, signed_invariant_defect=invariant,
                integrated_contributions=contributions, admissible=bool(op.is_admissible(W)))
            policy = settings.residual
            initial = np.asarray(stage["history"][0]["equation_norm"])
            active = initial > policy.initial_floor
            # Reproduce the declared default either criterion independently.
            transport_ok = np.all((independent_l2 <= policy.absolute_tolerance) |
                                  (active & (independent_l2 <= initial * policy.relative_tolerance)))
            row["fresh_success"] = bool(transport_ok and row["admissible"]
                and np.all(np.isfinite(residual)) and row["constraint_max"] <= policy.constraint_tolerance
                and row["boundary_max"] <= policy.boundary_tolerance
                and max(abs(global_defect)) <= policy.global_tolerance
                and max(abs(invariant)) <= policy.global_tolerance)
            assert row["fresh_success"] == stage["success"], row
            row["regime_audit"] = regime_audit(saved, stage["regime"])
            row["fresh_case_success"] = bool(row["fresh_success"] and row["regime_audit"]["passed"])
            rows.append(row)
            print(stage["regime"], stage["cells"], "confirmed", stage["reason"],
                  "intended regime", row["regime_audit"]["passed"], flush=True)
    write_json(args.output or ROOT / protocol["verification"], dict(stages=rows))
    print("Saved-state independent residual/budget audits passed:", len(rows))


if __name__ == "__main__":
    main()
