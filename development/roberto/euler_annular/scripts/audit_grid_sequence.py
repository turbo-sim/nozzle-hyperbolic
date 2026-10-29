"""Independently check the saved coarse-to-fine initialization and root contract.

No Newton or timestep solve runs here. The PDE audit is performed separately
by audit_results; this audit independently reconstructs interpolation weights,
checks parent fingerprints and verifies every saved stage reaches its target.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .verify_annular import (CASES, DEFAULT_CASES, DEFAULT_MESHES, RECONSTRUCTIONS,
                            STUDY, fingerprints, write_json)


def interpolate_independently(points, values, targets):
    lower = np.searchsorted(points, targets, side="right")-1
    lower = np.clip(lower, 0, len(points)-2)
    theta = (targets-points[lower])/(points[lower+1]-points[lower])
    return values[lower]+theta[:, None]*(values[lower+1]-values[lower])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=STUDY/"outputs/grid_sequence_monitored_full")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--cells", nargs="+", type=int, default=list(DEFAULT_MESHES))
    args = parser.parse_args()
    if not args.cells or args.cells != sorted(set(args.cells)) or min(args.cells) < 2:
        parser.error("Use a strictly increasing sequence of physical cell counts")
    records = []
    source = fingerprints()
    for name in args.cases:
        previous, parent_path, failed_path = None, None, None
        for count in args.cells:
            path = args.directory/f"{name}_{count}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["source_hashes"] == source, f"Source drift: {path}"
            assert data["initialization"]["hierarchy"] == args.cells
            assert data["initialization"]["level"] == args.cells.index(count)
            controller = data["settings"]["controller"]
            assert controller in ("cfl_recommended", "equation_monitored")
            monitored = controller == "equation_monitored"
            yaml_hashes = data.get("case_config_provenance", {}).get("source_sha256", {})
            if monitored:
                assert yaml_hashes, f"Missing YAML provenance: {path}"
                for config_path, expected_sha in yaml_hashes.items():
                    assert sha256(Path(config_path).read_bytes()).hexdigest() == expected_sha, f"YAML drift: {config_path}"
                assert data["settings"]["residual_policy"] == data["case_config"]["residuals"]
                assert data["settings"]["tolerance"] == data["case_config"]["residuals"]["absolute_tolerance"]
                assert data["initialization"]["relative_tolerance"] == data["case_config"]["residuals"]["relative_tolerance"]
                assert data["initialization"]["residual_normalization"] == "fresh initial equation norms on each mesh"
            targets = np.r_[data["m_faces"][0],data["m"],data["m_faces"][-1]]
            if previous is not None:
                assert all(run["success"] for run in previous["methods"].values()), f"Finer stage used an unconverged parent: {parent_path}"
                assert data["initialization"]["parent_file"] == parent_path.name
                assert data["initialization"]["parent_sha256"] == sha256(parent_path.read_bytes()).hexdigest()
                assert data["backpressure"] == previous["backpressure"]
                assert data["reference_invariants"] == previous["reference_invariants"]
                old_points = np.r_[previous["m_faces"][0],previous["m"],previous["m_faces"][-1]]
            else:
                assert data["initialization"]["parent_file"] is None
                assert data["settings"]["initial_state"] == "uniform physical cold guess"
                if failed_path is not None:
                    assert data["initialization"]["strategy"] == "uniform cold restart after failed coarser stage"
                    assert data["initialization"]["failed_coarser_file"] == failed_path.name
                    assert data["initialization"]["failed_coarser_sha256"] == sha256(failed_path.read_bytes()).hexdigest()
            for method in RECONSTRUCTIONS:
                run = data["methods"][method]
                initial = np.asarray(run["initial_state"])
                if monitored:
                    metric = run["metrics"]
                    metadata = run["monitoring"]
                    policy = data["settings"]["residual_policy"]
                    assert metadata["policy"] == policy
                    selected = np.asarray(metric["equation_norm"])
                    initial_norm = np.asarray(metadata["initial_equation_norms"])
                    active = initial_norm > policy["initial_floor"]
                    relative = np.divide(selected, initial_norm, out=np.zeros_like(selected), where=initial_norm > 0)
                    np.testing.assert_array_equal(active, metric["relative_active"])
                    np.testing.assert_allclose(relative, metric["equation_relative"], rtol=1e-13, atol=0)
                    root = bool(run["success"] and metric["admissible"]
                        and np.isfinite(selected).all()
                        and np.all(selected <= policy["absolute_tolerance"])
                        and np.all(relative[active] <= policy["relative_tolerance"])
                        and metric["boundary_max"] <= policy["boundary_tolerance"]
                        and metric["conservation_max"] <= policy["global_tolerance"]
                        and all(not row["required"] or row["stable"] for row in metric["monitors"].values()))
                else:
                    root = bool(run["success"] and run["metrics"]["admissible"] and max(
                        run["metrics"][key] for key in ("spatial_rms","boundary_max","conservation_max"))
                        < data["settings"]["tolerance"])
                assert not run["success"] or root, f"Recorded success violates original stage target: {path} {method}"
                transfer_error = 0.
                if previous is not None:
                    old = np.asarray(previous["methods"][method]["state"])
                    expected = interpolate_independently(old_points,old,targets)
                    scale = np.maximum(np.max(abs(old),axis=0),[1.,1.,1.,1.])
                    transfer_error = float(np.max(abs(initial-expected)/scale))
                    assert transfer_error < 1e-13, f"Transfer differs: {path} {method}"
                    assert np.all(initial >= old.min(axis=0)-1e-12*scale)
                    assert np.all(initial <= old.max(axis=0)+1e-12*scale)
                    np.testing.assert_array_equal(initial[[0,-1]],old[[0,-1]])
                else:
                    # Uniform physical velocity means ell/r, rather than ell,
                    # is uniform in the prescribed cold start on varying r.
                    np.testing.assert_allclose(initial[1:-1,0],initial[1,0],rtol=0,atol=0)
                    np.testing.assert_allclose(initial[1:-1,1],initial[1,1],rtol=0,atol=0)
                    np.testing.assert_allclose(initial[1:-1,3],initial[1,3],rtol=0,atol=0)
                    np.testing.assert_allclose(initial[1:-1,2]/np.asarray(data["r"]),
                                               initial[1,2]/data["r"][0],rtol=2e-14,atol=2e-14)
                history = run["history"]
                if history:
                    assert history[0]["cfl"] == .2
                initial_samples = [row for row in history if row.get("phase") == "initial"]
                if monitored:
                    assert len(initial_samples) == 1 and history[0] is initial_samples[0]
                    assert initial_samples[0]["step"] == 0 and initial_samples[0]["accepted"]
                    np.testing.assert_allclose(initial_samples[0]["equation_norm"], metadata["initial_equation_norms"], rtol=0, atol=0)
                else:
                    assert not initial_samples
                updates = [row for row in history if row["accepted"] and row.get("phase") != "initial"]
                attempts = [row for row in history if row.get("phase") != "initial"]
                assert len(updates) == run["controller_audit"]["accepted"]
                assert len(attempts) == run["controller_audit"]["attempts"]
                records.append(dict(case=name,cells=count,method=method,root=root,
                    parent_cells=previous["cells"] if previous else None,
                    maximum_scaled_transfer_error=transfer_error,
                    accepted=run["controller_audit"]["accepted"],
                    rejected=run["controller_audit"]["rejected"],
                    residual_definition=run.get("monitoring", {}).get("definition") if monitored else "legacy spatial_rms",
                    yaml_source_sha256=yaml_hashes,
                    input_sha256=sha256(path.read_bytes()).hexdigest()))
            if all(run["success"] for run in data["methods"].values()):
                previous, parent_path, failed_path = data, path, None
            else:
                previous, parent_path, failed_path = None, None, path
    result = dict(passed=True,numerical_solutions=len(records),records=records,
        all_roots_met=all(row["root"] for row in records),
        converged_solutions=sum(row["root"] for row in records),
        maximum_scaled_transfer_error=max(row["maximum_scaled_transfer_error"] for row in records),
        selected_cases=args.cases, grid_sequence=args.cells,
        source_hashes=source,source_unchanged=source==fingerprints(),
        qualification="This checks initialization, stage targets and history bookkeeping without advancing a solve. The separate audit_results report re-evaluates original PDE and boundary operators. Primitive interpolation is an initialization transfer, not a conservative physical-time remap or FAS correction.")
    assert result["source_unchanged"]
    write_json(args.directory/"grid_sequence_audit.json",result)
    print(json.dumps({key:value for key,value in result.items() if key not in ("records","source_hashes")},indent=2))


if __name__=="__main__":
    main()
