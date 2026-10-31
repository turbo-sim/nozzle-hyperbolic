"""Run both physical transients through the selectable production boundary solver."""

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import traceback

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, solve_transient
from .common import (
    STUDY,
    ROOT,
    save,
    provenance,
    finish,
    fingerprints,
    client_fingerprints,
)
from .fixtures import METHODS, fixture, fields, reference_fields, trace_diagnostics


def execute(model, case, settings, method, output):
    scheme = Scheme(
        **settings["scheme"],
        time_integrator=method["integrator"],
        boundary_treatment=method["boundary_treatment"],
    )
    # Algebraic traces have no storage: impose t=0+ constraints while retaining
    # the exact stagnant/diaphragm physical initial field in every method.
    setup = Operator(model, scheme)
    raw_initial = jnp.asarray(model.initial_state())
    initial, initial_info = setup.complete_boundaries(raw_initial, 0.0)
    initial_info = jax.device_get(initial_info)
    if not bool(initial_info.converged and initial_info.admissible):
        raise RuntimeError(f"Initial boundary completion failed: {initial_info}")
    if not np.array_equal(np.asarray(initial[1:-1]), np.asarray(raw_initial[1:-1])):
        raise AssertionError("Boundary completion changed physical initial cells")
    initial_totals = np.asarray(setup.storage(initial[1:-1])).sum(axis=0)
    snapshots = solve_transient(
        model,
        case["times"][1:],
        scheme,
        initial_state=initial,
        cfl=settings["cfl"],
        max_steps=settings["max_steps"],
    )
    states = [np.asarray(initial)] + [snapshot.state for snapshot in snapshots]
    times = [0.0] + [snapshot.time for snapshot in snapshots]
    values = [fields(model, state) for state in states]
    boundaries = [float(jnp.max(jnp.abs(setup.boundary_residual(initial, 0.0))))]
    boundaries += [float(snapshot.metrics["boundary_max"]) for snapshot in snapshots]
    budgets = [np.zeros(model.nvar)] + [
        np.asarray(snapshot.metrics["conservation_budget"]) for snapshot in snapshots
    ]
    traces = [trace_diagnostics(model, state) for state in states]
    last = snapshots[-1] if snapshots else None
    history = [] if last is None else last.history
    accepted = [row for row in history if row["accepted"]]
    success = bool(
        len(snapshots) == len(case["times"]) - 1
        and all(snapshot.success for snapshot in snapshots)
    )
    min_velocity = min(float(np.min(state[1:-1, 1])) for state in states)
    record = dict(
        **method,
        scheme=asdict(scheme),
        effective_boundary_treatment=setup.effective_boundary_treatment,
        success=success,
        reason=last.reason if last is not None else "no_outputs",
        times=times,
        fields={
            name: np.stack([value[name] for value in values]) for name in values[0]
        },
        metrics=dict(
            boundary_max=boundaries,
            conservation_budget=budgets,
            minimum_pressure=[float(np.min(value["p"])) for value in values],
            minimum_density=[float(np.min(value["rho"])) for value in values],
            minimum_temperature=[float(np.min(value["T"])) for value in values],
            min_velocity=[float(np.min(value["u"])) for value in values],
        ),
        outlet_regime=[trace["mode"] for trace in traces],
        outlet_trace_pressure=[trace["pressure"] for trace in traces],
        outlet_trace_mach=[trace["mach"] for trace in traces],
        summary=dict(
            accepted_steps=len(accepted),
            rejected_steps=sum(not row["accepted"] for row in history),
            elapsed_s=0.0 if last is None else last.elapsed_s,
            outer_newton_iterations=sum(row["inner_iterations"] for row in accepted),
            inner_iteration_semantics=(
                "Outer physical Newton; nested boundary work is additional"
                if method["id"] == "implicit_segregated"
                else (
                    "Full coupled Newton"
                    if method["id"] == "implicit_coupled"
                    else "Cell storage recovery and stage boundary Newton iterations"
                )
            ),
            maximum_boundary_defect=max(boundaries),
            maximum_conservation_defect=float(np.max(np.abs(budgets))),
            minimum_velocity=min_velocity,
            maximum_actual_cfl=max((row["cfl"] for row in accepted), default=0.0),
            minimum_actual_cfl=min((row["cfl"] for row in accepted), default=0.0),
            initial_trace_completed=True,
        ),
    )
    if case["id"] == "sod":
        exact = reference_fields(model, "sod", settings, times[-1])
        record["summary"]["final_exact_L1"] = {
            name: float(
                np.mean(np.abs(values[-1][name] - exact[name]))
                / case["field_scales"][name]
            )
            for name in ("p", "rho", "u", "M", "T")
        }
    artifact = output / f"{case['id']}__{method['id']}.json"
    save(
        artifact,
        dict(
            record=record,
            states=states,
            initial_state_before_completion=raw_initial,
            initial_totals=initial_totals,
            history=history,
            equation_names=model.equation_names,
            conservative_scales=model.conservative_scales,
            normalized_content_scales=np.sum(model.mesh.volumes)
            * np.asarray(model.conservative_scales),
        ),
    )
    record["artifact"] = artifact.relative_to(ROOT).as_posix()
    record["artifact_sha256"] = sha256(artifact.read_bytes()).hexdigest()
    print(
        f"  {case['id']} {method['id']}: success={success}; steps={len(accepted)}; "
        f"BC={max(boundaries):.2e}; budget={record['summary']['maximum_conservation_defect']:.2e}",
        flush=True,
    )
    jax.clear_caches()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("sod", "nozzle_step"),
        default=["sod", "nozzle_step"],
    )
    parser.add_argument(
        "--methods", nargs="+", choices=[method["id"] for method in METHODS]
    )
    parser.add_argument("--cells", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--nozzle-times", type=float)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/verification")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    settings = yaml.safe_load((STUDY / "settings.yaml").read_text(encoding="utf-8"))
    for name in ("cells", "frames"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if settings["cells"] < 2 or settings["frames"] < 2:
        raise ValueError("Use at least two cells and two output frames")
    if args.nozzle_times is not None:
        if not np.isfinite(args.nozzle_times) or args.nozzle_times <= 0:
            raise ValueError("Positive finite nozzle duration required")
        settings["nozzle_step"]["end_acoustic_times"] = args.nozzle_times
    selected = [
        method
        for method in METHODS
        if args.methods is None or method["id"] in args.methods
    ]
    settings["selection"] = dict(
        cases=args.cases, methods=[method["id"] for method in selected]
    )
    path = args.output / "summary.json"
    data = (
        json.loads(path.read_text(encoding="utf-8"))
        if args.resume and path.exists()
        else dict(
            schema_version=1,
            settings=settings,
            provenance=provenance(),
            cases=[],
            exceptions=[],
            completed=False,
        )
    )
    if (
        data["settings"] != settings
        or data["provenance"]["source_before"] != fingerprints()
        or data["provenance"]["client_before"] != client_fingerprints()
    ):
        raise RuntimeError("Source/settings differ; select a new output directory")

    def checkpoint():
        finish(data)
        save(path, data)
        if (
            not data["provenance"]["source_unchanged"]
            or not data["provenance"]["client_unchanged"]
        ):
            raise RuntimeError("Execution source changed; partial evidence retained")

    for case_id in args.cases:
        model, case = fixture(case_id, settings)
        previous = next((item for item in data["cases"] if item["id"] == case_id), None)
        if previous is None:
            case.update(
                methods=[],
                initial_trace_completed=True,
                initial_trace_note="t=0+ algebraic traces are completed; physical initial cells are unchanged.",
            )
            reference = [
                reference_fields(model, case_id, settings, time)
                for time in case["times"]
            ]
            if reference[0] is not None:
                case["reference"] = dict(
                    fields={
                        name: np.stack([value[name] for value in reference])
                        for name in reference[0]
                    }
                )
            data["cases"].append(case)
        else:
            case = previous
        for method in selected:
            if args.resume and any(
                row["id"] == method["id"] and row.get("success")
                for row in case["methods"]
            ):
                continue
            print(case_id, method["id"], "CFL", settings["cfl"], flush=True)
            try:
                record = execute(model, case, settings, method, args.output)
                case["methods"] = [
                    row for row in case["methods"] if row["id"] != method["id"]
                ] + [record]
            except Exception as exc:
                data["exceptions"].append(
                    dict(
                        case=case_id,
                        method=method["id"],
                        exception=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                case["methods"].append(
                    dict(
                        **method,
                        success=False,
                        reason=repr(exc),
                        fields={},
                        metrics={},
                        summary={},
                    )
                )
                print(traceback.format_exc(), flush=True)
            checkpoint()
    data["completed"] = True
    data["passed"] = bool(
        not data["exceptions"]
        and all(
            method["success"] for case in data["cases"] for method in case["methods"]
        )
    )
    checkpoint()
    assert data["passed"], f"Transient study has failures; inspect {path}"


if __name__ == "__main__":
    main()
