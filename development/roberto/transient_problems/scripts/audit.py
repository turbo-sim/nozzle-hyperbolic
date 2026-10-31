"""Fresh field/BC audits and independent conservation bookkeeping of saved runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from .common import (
    ROOT,
    STUDY,
    save,
    provenance,
    finish,
    fingerprints,
    client_fingerprints,
)
from .fixtures import fixture, fields, reference_fields


def audit_method(model, case, row, settings):
    path = ROOT / row["artifact"]
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
        raise RuntimeError(f"Artifact checksum changed: {path}")
    op = Operator(model, Scheme(**row["scheme"]))
    states = np.asarray(artifact["states"])
    times = np.asarray(row["times"])
    scale = np.asarray(artifact["normalized_content_scales"])
    initial = np.asarray(op.storage(jnp.asarray(states[0, 1:-1]))).sum(axis=0)
    accepted = [item for item in artifact["history"] if item["accepted"]]
    transfer, position = np.zeros(model.nvar), 0
    boundary, budget, field_error = 0.0, 0.0, 0.0
    all_admissible, frame_audits = True, []
    for index, (time, state) in enumerate(zip(times, states)):
        while position < len(accepted) and accepted[position]["time"] <= time:
            transfer += np.asarray(accepted[position]["integrated_transfer"])
            position += 1
        full = jnp.asarray(state)
        actual = fields(model, full)
        norm = float(jnp.max(jnp.abs(op.boundary_residual(full, time))))
        valid = bool(op.is_admissible(full))
        totals = np.asarray(op.storage(full[1:-1])).sum(axis=0)
        independently_summed = (totals - initial - transfer) / scale
        saved = np.asarray(row["metrics"]["conservation_budget"][index])
        difference = max(
            float(np.max(np.abs(actual[name] - np.asarray(row["fields"][name][index]))))
            / case["field_scales"][name]
            for name in actual
        )
        frame_audits.append(
            dict(
                time=time,
                boundary_max=norm,
                admissible=valid,
                independent_budget=independently_summed,
                stored_budget_difference=float(
                    np.max(np.abs(independently_summed - saved))
                ),
                normalized_saved_field_difference=difference,
            )
        )
        boundary = max(boundary, norm)
        budget = max(budget, float(np.max(np.abs(independently_summed))))
        field_error = max(field_error, difference)
        all_admissible &= valid
    cfl_max = max((item["cfl"] for item in accepted), default=0.0)
    counts_ok = len(states) == settings["frames"] and np.array_equal(
        times, np.asarray(case["times"])
    )
    initial_unchanged = np.array_equal(
        states[0, 1:-1], np.asarray(artifact["initial_state_before_completion"])[1:-1]
    )
    passed = bool(
        row["success"]
        and counts_ok
        and initial_unchanged
        and all_admissible
        and boundary <= 1.0e-9
        and budget <= 1.0e-9
        and field_error <= 1.0e-13
        and cfl_max <= settings["cfl"] * (1.0 + 1.0e-12)
    )
    result = dict(
        case=case["id"],
        method=row["id"],
        frames=len(states),
        passed=passed,
        all_admissible=all_admissible,
        initial_physical_cells_unchanged=initial_unchanged,
        maximum_boundary_defect=boundary,
        maximum_independent_conservation_defect=budget,
        maximum_saved_field_difference=field_error,
        maximum_actual_cfl=cfl_max,
        accepted_steps=len(accepted),
        rejected_steps=len(artifact["history"]) - len(accepted),
        frame_audits=frame_audits,
    )
    if case["id"] == "sod":
        ref = reference_fields(model, "sod", settings, times[-1])
        errors = {
            name: float(
                np.mean(np.abs(np.asarray(row["fields"][name][-1]) - ref[name]))
            )
            / case["field_scales"][name]
            for name in ("p", "rho", "u")
        }
        result["exact_final_L1"] = errors
        result["exact_accuracy_passed"] = all(
            value <= 0.04 for value in errors.values()
        )
        result["passed"] &= result["exact_accuracy_passed"]
    else:
        config = settings["nozzle_step"]
        p = np.asarray(row["fields"]["p"])
        u = np.asarray(row["fields"]["u"])
        released = bool(
            np.all(u[0] == 0.0)
            and np.all(p[0] == config["reservoir_pressure"])
            and np.mean(p[-1]) < 0.95 * config["reservoir_pressure"]
            and row["outlet_regime"][-1] == "fully outgoing"
            and row["outlet_trace_mach"][-1] > 1.0
        )
        result["pressure_release_and_supersonic_exit_passed"] = released
        result["passed"] &= released
    jax.clear_caches()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=STUDY / "outputs/verification/summary.json"
    )
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs/verification/audit.json"
    )
    args = parser.parse_args()
    campaign = json.loads(args.input.read_text(encoding="utf-8"))
    data = dict(
        provenance=provenance(), records=[], implicit_comparisons=[], passed=False
    )
    data["campaign_source_matches_current"] = (
        campaign["provenance"]["source_before"] == fingerprints()
    )
    data["campaign_client_matches_current"] = (
        campaign["provenance"]["client_before"] == client_fingerprints()
    )
    for case in campaign["cases"]:
        model, _ = fixture(case["id"], campaign["settings"])
        for row in case["methods"]:
            try:
                result = audit_method(model, case, row, campaign["settings"])
            except Exception as exc:
                result = dict(
                    case=case["id"], method=row["id"], passed=False, exception=repr(exc)
                )
            data["records"].append(result)
            print(case["id"], row["id"], result["passed"], flush=True)
        methods = {row["id"]: row for row in case["methods"]}
        if {"implicit_coupled", "implicit_segregated"}.issubset(methods):
            first, second = (
                methods[key] for key in ("implicit_coupled", "implicit_segregated")
            )
            difference = {
                name: float(
                    np.max(
                        np.abs(
                            np.asarray(first["fields"][name])
                            - np.asarray(second["fields"][name])
                        )
                    )
                )
                / case["field_scales"][name]
                for name in first["fields"]
            }
            data["implicit_comparisons"].append(
                dict(
                    case=case["id"],
                    normalized_maximum_field_difference=difference,
                    tolerance=1.0e-8,
                    passed=max(difference.values()) <= 1.0e-8,
                )
            )
    data["passed"] = bool(
        campaign.get("completed")
        and campaign.get("passed")
        and campaign["provenance"]["source_unchanged"]
        and campaign["provenance"]["client_unchanged"]
        and data["campaign_source_matches_current"]
        and data["campaign_client_matches_current"]
        and data["records"]
        and all(item["passed"] for item in data["records"])
        and all(item["passed"] for item in data["implicit_comparisons"])
    )
    finish(data)
    data["passed"] &= (
        data["provenance"]["source_unchanged"]
        and data["provenance"]["client_unchanged"]
    )
    save(args.output, data)
    assert data["passed"], f"Independent audit failed; inspect {args.output}"
    print("All fresh transient audits passed", flush=True)


if __name__ == "__main__":
    main()
