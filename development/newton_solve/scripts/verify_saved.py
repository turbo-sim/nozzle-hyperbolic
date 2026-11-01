"""Re-evaluate every saved target with frozen source in a fresh process.

This executable does not import a report generator, call a solver or advance
time. It reconstructs canonical physical cases and their saved spatial scheme,
then independently evaluates the final states, including unconverged ones.

Run only after the campaign has finished writing its records::

    python -B -m development.newton_solve.scripts.verify_saved

Saved/fresh convergence flags and individual Boolean checks must agree exactly.
Floating diagnostics use ``atol=5e-12`` and ``rtol=1e-8`` to allow final-bit
compiler differences near roundoff. This comparison envelope never changes
the original convergence policy: fresh flags still use its unchanged targets.
JSON nulls in numerical arrays are treated as classified nonfinite entries;
their positions must reproduce, while the fresh classification is retained.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
from time import perf_counter
import traceback

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator, ResidualPolicy, Scheme, Solution

from .audit import evaluate_solution
from .common import (
    OUTPUT, configure_compilation_cache, identity, load, profile,
    require_source, run_id, save,
)
from .problem import target_context


ATOL = 5e-12
RTOL = 1e-8


def _numeric_comparison(saved, fresh):
    """Compare finite diagnostics and nonfinite positions without changing flags."""
    try:
        old = np.asarray(saved, dtype=float)
        new = np.asarray(fresh, dtype=float)
    except (TypeError, ValueError) as error:
        return dict(passed=False, error=str(error))
    if old.shape != new.shape:
        return dict(passed=False, saved_shape=old.shape, fresh_shape=new.shape)
    finite_old, finite_new = np.isfinite(old), np.isfinite(new)
    finite_pattern = bool(np.array_equal(finite_old, finite_new))
    both = finite_old & finite_new
    difference = np.abs(new[both] - old[both])
    allowed = ATOL + RTOL * np.abs(old[both])
    finite_values = bool(np.all(difference <= allowed))
    return dict(
        passed=bool(finite_pattern and finite_values),
        finite_pattern_matches=finite_pattern,
        finite_values_match=finite_values,
        maximum_absolute_difference=float(difference.max()) if difference.size else 0.0,
        maximum_comparison_ratio=float(np.max(difference / allowed)) if difference.size else 0.0,
        fresh_nonfinite=dict(
            nan=np.argwhere(np.isnan(new)).tolist(),
            positive_infinity=np.argwhere(np.isposinf(new)).tolist(),
            negative_infinity=np.argwhere(np.isneginf(new)).tolist(),
        ),
    )


def _target_paths(include_variants=True):
    paths = list((OUTPUT / "runs").glob("*.json"))
    paths.extend((OUTPUT / "runs").glob("*.json.gz"))
    if include_variants:
        paths.extend((OUTPUT / "variants").glob("*/runs/*.json"))
        paths.extend((OUTPUT / "variants").glob("*/runs/*.json.gz"))
    return sorted(paths)


def _file_hash(path):
    """Hash the saved bytes in bounded chunks, including compressed evidence."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_record(path):
    """Read one record and promptly discard large native trial/history payloads."""
    payload = load(path)
    names = (
        "id", "case_id", "outlet", "flux", "reconstruction", "cells", "strategy", "start",
        "state", "initial_state", "scheme", "init_hash", "success", "native_success", "reason",
        "verified", "target_correct", "audit", "source_sha256", "client_sha256", "settings_sha256",
    )
    row = {name: payload.get(name) for name in names}
    direct = payload.get("direct")
    if direct is not None:
        row["direct"] = {name: direct.get(name) for name in (
            "original_integrated_spatial_residual", "original_boundary_residual", "original_steady_residual"
        )}
    # Initial norms are independently recomputed from initial_state. Release
    # large histories before constructing another physical/spatial graph.
    del direct, payload
    return row


def verify_saved(*, include_variants=True, allow_incomplete=False):
    """Fresh original-equation audit of a stable set of saved final states."""
    started = perf_counter()
    provenance = load(OUTPUT / "provenance.json")
    require_source(provenance)
    settings = profile()
    policy = ResidualPolicy(**settings["residual"])
    configure_compilation_cache()
    paths = _target_paths(include_variants)
    file_hashes = {path: _file_hash(path) for path in paths}
    expected = {run_id(*config) for config in itertools.product(
        settings["cases"], settings["outlets"], settings["fluxes"],
        settings["reconstructions"], settings["starts"], settings["strategies"]
    )}
    baseline_ids = set()
    operators = {}
    details = []
    for path in paths:
        record_started = perf_counter()
        detail = dict(id=path.name.removesuffix(".gz").removesuffix(".json"), path=path.relative_to(OUTPUT).as_posix(),
                      saved_file_sha256=file_hashes[path], checks={})
        try:
            row = _audit_record(path)
            detail["id"] = row["id"]
            if path.parent == OUTPUT / "runs":
                baseline_ids.add(row["id"])
            detail["checks"]["frozen_input_identity"] = all(
                row.get(key) == provenance[key]
                for key in ("source_sha256", "client_sha256", "settings_sha256")
            )
            full = np.asarray(row["state"], dtype=float)
            initial = np.asarray(row["initial_state"], dtype=float)
            cells = int(row["cells"])
            expected_shape = (cells + 2, 3)
            if full.shape != expected_shape or initial.shape != expected_shape:
                raise ValueError(f"Expected recorded final and initial full states {expected_shape}; got {full.shape}, {initial.shape}")
            scheme = Scheme(**row["scheme"])
            expected_scheme = Scheme(flux=row["flux"], reconstruction=row["reconstruction"], **settings["scheme"])
            detail["checks"]["scheme_matches_campaign"] = scheme == expected_scheme
            detail["checks"]["target_mesh_matches_campaign"] = cells == settings["cells"]
            detail["checks"]["saved_initial_hash"] = identity(initial) == row["init_hash"]
            key = (row["case_id"], row["outlet"], row["flux"], row["reconstruction"], cells, identity(row["scheme"]))
            if key not in operators:
                context = target_context(row["case_id"])
                model = context.build_model(cells, row["outlet"])
                operators[key] = (context, Operator(model, scheme))
            context, operator = operators[key]
            solution = Solution(full, bool(row.get("success", False)), row.get("reason", "saved_state"),
                                [], {},
                                elapsed_s=0.0, operator=operator)
            fresh = evaluate_solution(context, solution, policy, initial_state=initial)
            old_audit, new_audit = row["audit"], fresh["audit"]
            detail["checks"].update(
                saved_verified_consistent=bool(row["verified"]) == bool(old_audit["passed"]),
                verified_flag_reproduced=bool(row["verified"]) == bool(new_audit["passed"]),
                target_flag_reproduced=row.get("target_correct") == fresh["target_correct"],
                individual_audit_checks_reproduced=old_audit["checks"] == new_audit["checks"],
                failed_check_names_reproduced=sorted(old_audit["failed_checks"]) == sorted(new_audit["failed_checks"]),
            )
            numeric = {}
            for name in (
                "equation_norm", "equation_initial_norm", "equation_l1", "equation_l2",
                "equation_linf", "equation_relative", "boundary_max", "global_defect",
            ):
                numeric[name] = _numeric_comparison(old_audit["residual"][name], new_audit["residual"][name])
            detail["checks"]["residual_diagnostics_reproduced"] = all(value["passed"] for value in numeric.values())
            detail["checks"]["equation_decisions_reproduced"] = all(
                np.array_equal(np.asarray(old_audit["residual"][name]), np.asarray(new_audit["residual"][name]))
                for name in ("equation_passed", "relative_active", "absolute_satisfied", "relative_satisfied")
            )
            raw = np.asarray(operator.spatial_residual(full))
            boundary = np.asarray(operator.boundary_residual(full, 0.0))
            direct = row.get("direct")
            direct_comparisons = {}
            if direct is not None:
                direct_comparisons = dict(
                    original_integrated_spatial_residual=_numeric_comparison(direct.get("original_integrated_spatial_residual"), raw),
                    original_boundary_residual=_numeric_comparison(direct.get("original_boundary_residual"), boundary),
                    original_steady_residual=_numeric_comparison(
                        direct.get("original_steady_residual"), np.asarray(operator.steady_residual(full.ravel(), 0.0))
                    ),
                )
                detail["checks"]["direct_original_rows_reproduced"] = all(value["passed"] for value in direct_comparisons.values())
            detail.update(
                saved_verified=bool(row["verified"]), fresh_verified=bool(new_audit["passed"]),
                saved_target_correct=row.get("target_correct"), fresh_target_correct=fresh["target_correct"],
                saved_native_success=row.get("native_success"),
                numerical_comparisons=numeric, direct_row_comparisons=direct_comparisons,
                fresh_audit=new_audit, fresh_errors=fresh["errors"],
                original_integrated_spatial_residual=raw, original_boundary_residual=boundary,
            )
        except Exception:
            detail["checks"]["reevaluation_completed"] = False
            detail["exception"] = traceback.format_exc()
        detail["elapsed_seconds"] = perf_counter() - record_started
        detail["passed"] = bool(all(detail["checks"].values()))
        details.append(detail)
        print(f"VERIFY {detail['id']} reproducible={detail['passed']} "
              f"saved_root={detail.get('saved_verified')} fresh_root={detail.get('fresh_verified')} "
              f"target={detail.get('fresh_target_correct')}", flush=True)
    require_source(provenance)
    stable_files = (_target_paths(include_variants) == paths and all(
        _file_hash(path) == expected_hash for path, expected_hash in file_hashes.items()
    ))
    checks = dict(
        baseline_coverage=allow_incomplete or baseline_ids == expected,
        baseline_ids_requested=baseline_ids <= expected,
        nonempty_records=bool(paths), saved_records_unchanged=stable_files,
        every_saved_state_reproduced=all(row["passed"] for row in details),
    )
    result = dict(
        passed=bool(all(checks.values())), checks=checks,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=provenance["source_sha256"], client_sha256=provenance["client_sha256"],
        settings_sha256=provenance["settings_sha256"],
        verifier_sha256=_file_hash(Path(__file__)),
        expected_baseline_runs=len(expected), verified_saved_records=len(paths),
        missing_baseline_runs=sorted(expected - baseline_ids),
        operator_instances=len(operators),
        numerical_comparison=dict(absolute_tolerance=ATOL, relative_tolerance=RTOL,
                                  purpose="Reproduction envelope only; original convergence thresholds are unchanged and Boolean decisions must reproduce exactly"),
        residual_policy=settings["residual"],
        execution=dict(fresh_process=True, physical_time_steps=0, nonlinear_solves=0,
                       include_variants=include_variants, allow_incomplete=allow_incomplete,
                       jax_enable_x64=bool(jax.config.x64_enabled)),
        elapsed_seconds=perf_counter() - started, records=details,
    )
    save(OUTPUT / "verify_saved.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicit partial-coverage audit; still check every saved state")
    args = parser.parse_args()
    result = verify_saved(include_variants=not args.baseline_only, allow_incomplete=args.allow_incomplete)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "missing_baseline_runs")}, indent=2))
    if not result["passed"]:
        raise SystemExit("Saved-state original-equation re-evaluation failed; inspect outputs/verify_saved.json")


if __name__ == "__main__":
    main()
