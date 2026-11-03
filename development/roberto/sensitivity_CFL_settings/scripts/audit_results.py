"""Audit current sensitivity states, solver defects and actual Newton counts."""

import argparse
import json
from pathlib import Path

import numpy as np
from nozzle_hyperbolic.verification import build_case, case_from_dict

from development.roberto.sensitivity_backpressure.scripts.problem import (
    ImplicitNozzle,
    RMS_TOL,
    SCHEMA,
    scheme,
)
from development.roberto.sensitivity_backpressure.scripts.run_sweep import hashes


def audit(output):
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != SCHEMA:
        raise ValueError("Select a current shared-FVM campaign")
    assert (
        hashes() == summary["source_before"] == summary["source_after"]
    ), "Production source changed"
    results = []
    for row in summary["runs"]:
        path = (output / row["file"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Run file escapes campaign directory")
        run = json.loads(path.read_text(encoding="utf-8"))
        problem = ImplicitNozzle(
            summary["settings"]["points"], numerical_scheme=scheme(row["cap"])
        )
        physical = summary["cases"][row["case"]]["physical_case"]
        # Resolve backpressure from the installed physical case, independently
        # of the saved numerical field and reported outcome.
        pressure = build_case(
            case_from_dict(physical), cells=problem.mesh.n_cells
        ).back_pressure
        metrics = problem.workspace(pressure).sample(np.asarray(run["state"]))
        trials = [h for h in run["history"] if h["phase"] != "initial"]
        assert all(0 <= h["inner_iterations"] <= row["cap"] for h in trials)
        assert sum(h["inner_iterations"] for h in trials) == row["inner_iterations"]
        assert sum(h["accepted"] for h in trials) == row["accepted_steps"]
        assert sum(not h["accepted"] for h in trials) == row["rejected_steps"]
        for key in ("equation_norm", "global_defect"):
            np.testing.assert_allclose(
                metrics[key], run["metrics"][key], rtol=1e-10, atol=1e-12
            )
        np.testing.assert_allclose(
            metrics["boundary_max"], row["boundary_max"], rtol=1e-10, atol=1e-12
        )
        if row["success"]:
            assert metrics["finite"] and metrics["admissible"]
            assert np.max(metrics["equation_norm"]) <= RMS_TOL
            assert (
                metrics["boundary_passed"]
                and metrics["constraints_passed"]
                and metrics["global_passed"]
            )
        results.append(
            dict(run_id=row["run_id"], passed=True, converged=row["success"])
        )
    assert results, "No attempted solves in campaign"
    record = dict(passed=True, audited_runs=len(results), results=results)
    (output / "audit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/shared_fvm",
    )
    audit(parser.parse_args().output)
