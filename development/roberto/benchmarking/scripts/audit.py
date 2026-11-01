"""Audit saved benchmark evidence without binding or advancing a flow solver.

Provenance/completeness success is distinct from numerical acceptance: a complete
campaign that faithfully retains failed cases can pass this audit. The numerical
and analytical failure counts remain explicit in its result.
"""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .common import ROOT, STUDY, SNAPSHOT, identity, source_snapshot, harness_snapshot, save


STATUSES = {"passed", "failed", "unsupported", "timeout", "accuracy_failed", "branch_failed", "assessment_failed"}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _array(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.ndim == 1 and result.size and np.isfinite(result).all() else None


def _close(left, right):
    return _finite(left) and _finite(right) and math.isclose(left, right, rel_tol=3e-12, abs_tol=3e-14)


def _environment_status(value):
    status = value.get("status") if isinstance(value, dict) else value
    if isinstance(status, str):
        status = status.strip().lower()
    return status if status in ("clean", "contended", "unverified") else "unverified"


def _configuration_id(case, reconstruction, cells, cap, backend, flux=None):
    key=f'{case.replace("/", "__")}__{reconstruction}__n{cells}__cap{cap}__{backend}'
    return key if flux is None else key+'__flux_'+flux


def expected_configurations(protocol):
    """Independently enumerate every requested case/mesh/backend combination."""
    cases = protocol["cases"]
    grids = protocol["grids"]
    maximum = max(grids)
    result = {}
    for case in cases:
        for cells in grids:
            backends = protocol["backends"] if cells == maximum or protocol.get("legacy_refinement") else [item for item in protocol["backends"] if item == "current"]
            for reconstruction in protocol["reconstructions"]:
                for cap in protocol["newton_caps"]:
                    for backend in backends:
                      for flux in protocol.get("fluxes",[protocol.get("scheme",{}).get("flux")]):
                        key = _configuration_id(case, reconstruction, cells, cap, backend,flux if "fluxes" in protocol else None)
                        if key in result:
                            raise ValueError(f"Duplicate requested configuration: {key}")
                        result[key] = dict(case_id=case, cells=cells, reconstruction=reconstruction,
                                           cap=cap, backend=backend,flux=flux,
                                           expected_samples=protocol["repeats"] if cells == maximum else protocol["refinement_repeats"])
    return result


def _residual_acceptance(metrics, policy):
    """Recheck the saved ORIGINAL spatial norms and independent BC/global limits.

    This does not reevaluate the PDE or trust the saved all_equations_passed
    flag. Full residual evaluation was performed outside timing by the worker's
    production sampler; here the retained reductions are assessed independently.
    """
    norm, initial = _array(metrics.get("equation_norm")), _array(metrics.get("initial_norm"))
    global_defect = _array(metrics.get("global_defect"))
    boundary = metrics.get("boundary_max")
    if norm is None or initial is None or global_defect is None or norm.shape != initial.shape or global_defect.shape != norm.shape:
        return dict(passed=False, available=False, reason="missing/nonfinite spatial, initial, or global vectors")
    if np.any(norm < 0) or np.any(initial < 0) or np.any(global_defect < 0) or not _finite(boundary) or boundary < 0:
        return dict(passed=False, available=False, reason="invalid negative/nonfinite norms")
    active = initial > policy["initial_floor"]
    absolute = norm <= policy["absolute_tolerance"]
    relative = norm <= initial * policy["relative_tolerance"]
    criterion = policy.get("criterion", "either")
    if criterion == "absolute":
        equations = absolute
    elif criterion == "relative":
        equations = np.where(active, relative, absolute)
    elif criterion == "both":
        equations = np.where(active, relative & absolute, absolute)
    elif criterion == "either":
        equations = np.where(active, relative | absolute, absolute)
    else:
        return dict(passed=False, available=False, reason="unsupported residual criterion")
    passed = bool(np.all(equations) and boundary <= policy["boundary_tolerance"]
                  and np.max(global_defect) <= policy["global_tolerance"]
                  and metrics.get("finite") and metrics.get("admissible"))
    return dict(passed=passed, available=True,
                equations=bool(np.all(equations)), boundary=bool(boundary <= policy["boundary_tolerance"]),
                global_balance=bool(np.max(global_defect) <= policy["global_tolerance"]),
                finite_admissible=bool(metrics.get("finite") and metrics.get("admissible")))


def audit_campaign(campaign, *, input_path=None, compare_live=True):
    issues, warnings, records = [], [], []

    def issue(name, detail, record=None):
        issues.append(dict(check=name, detail=detail, **({"record": record} if record else {})))

    protocol = campaign.get("protocol", {})
    provenance = campaign.get("provenance", {})
    timing_environment = campaign.get("timing_environment", {})
    environment_status = _environment_status(timing_environment)
    environment_clean = environment_status == "clean"
    if not environment_clean:
        issue("timing_environment", "Timing environment is " + str(environment_status) + "; these measurements cannot establish clean timing comparisons")
    if not campaign.get("completed"):
        issue("campaign_complete", "The campaign is not marked complete")
    if identity(protocol) != campaign.get("protocol_sha256"):
        issue("protocol_identity", "Saved protocol does not match its SHA-256")
    if identity(provenance.get("machine", {})) != provenance.get("machine_sha256"):
        issue("machine_identity", "Saved machine/package metadata does not match its SHA-256")
    for kind in ("source", "harness"):
        snapshot = provenance.get(kind, {})
        if not snapshot.get("files") or identity(snapshot.get("files", {})) != snapshot.get("sha256"):
            issue(kind + "_identity", "Missing or inconsistent saved file-map identity")
        if campaign.get(kind + "_unchanged") is not True:
            issue(kind + "_unchanged", "Runner did not verify an unchanged " + kind + " throughout the campaign")
    if compare_live:
        if source_snapshot()["sha256"] != provenance.get("source", {}).get("sha256"):
            issue("live_source_identity", "Current source differs from the recorded campaign snapshot")
        if harness_snapshot()["sha256"] != provenance.get("harness", {}).get("sha256"):
            issue("live_harness_identity", "Current numerical harness differs from the recorded campaign snapshot")

    manifest_path = SNAPSHOT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    if len(manifest) != 110:
        issue("legacy_manifest_count", f"Expected frozen 110-file manifest; found {len(manifest)}")
    if manifest_digest != provenance.get("legacy_manifest_sha256"):
        issue("legacy_manifest_identity", "Frozen manifest differs from recorded identity")
    bad_legacy = [name for name, expected in manifest.items()
                  if not (SNAPSHOT / name).resolve().is_relative_to(SNAPSHOT)
                  or not (SNAPSHOT / name).is_file()
                  or sha256((SNAPSHOT / name).read_bytes()).hexdigest() != expected]
    if bad_legacy:
        issue("legacy_source_integrity", bad_legacy)

    cases = {item.get("id"): item for item in campaign.get("cases", [])}
    if len(cases) != len(campaign.get("cases", [])) or set(cases) != set(protocol.get("cases", [])):
        issue("case_inventory", "Case definitions do not match the requested IDs exactly once")
    for case_id, case in cases.items():
        source = case.get("source", "")
        path = (ROOT / "src/nozzle_hyperbolic" / source).resolve()
        if not path.is_relative_to(ROOT / "src/nozzle_hyperbolic") or not path.is_file():
            issue("case_resource", f"Unavailable/unsafe resource: {source}", case_id)
            continue
        if sha256(path.read_bytes()).hexdigest() != case.get("source_sha256"):
            issue("case_resource_identity", "YAML bytes differ from the recorded definition", case_id)
        if yaml.safe_load(path.read_bytes()) != case.get("definition"):
            issue("case_definition", "Saved resolved definition differs from YAML", case_id)

    try:
        expected = expected_configurations(protocol)
    except (KeyError, ValueError, TypeError) as error:
        issue("configuration_protocol", str(error))
        expected = {}
    rows = campaign.get("results", [])
    observed = Counter(row.get("id") for row in rows)
    if set(observed) != set(expected) or any(count != 1 for count in observed.values()):
        issue("configuration_coverage", dict(missing=sorted(set(expected) - set(observed)),
                                             unexpected=sorted(set(observed) - set(expected)),
                                             duplicates={key: count for key, count in observed.items() if count != 1}))
    actual_counts = dict(Counter(row.get("status") for row in rows))
    if actual_counts != campaign.get("counts", {}):
        issue("status_counts", dict(recorded=campaign.get("counts"), actual=actual_counts))

    policy = protocol.get("residual", {})
    source_stable = campaign.get("source_unchanged") is True and campaign.get("harness_unchanged") is True and environment_clean
    for row in rows:
        key = row.get("id", "<missing-id>")
        case = cases.get(row.get("case_id"), {})
        definition = case.get("definition", {})
        requested = expected.get(key, {})
        for field in ("case_id", "cells", "reconstruction", "cap", "backend"):
            if row.get(field) != requested.get(field):
                issue("configuration_fields", field + " differs from ID/protocol", key)
        if "fluxes" in protocol and row.get("flux")!=requested.get("flux"):
            issue("configuration_flux", "Flux label differs from ID/protocol", key)
        if row.get("status") not in STATUSES:
            issue("status_schema", repr(row.get("status")), key)
        row_environment = row.get("timing_environment")
        row_environment_status = _environment_status(row_environment)
        row_environment_clean = row_environment_status == "clean"
        if not row_environment_clean:
            issue("record_timing_environment", "Timing block environment is " + row_environment_status, key)
        elif isinstance(row_environment, dict):
            for boundary in ("before", "after"):
                if boundary in row_environment and _environment_status(row_environment[boundary]) != "clean":
                    issue("record_timing_environment", "Clean block contradicts its " + boundary + " environment probe", key)
                    row_environment_clean = False
        if row.get("case_sha256") != case.get("source_sha256"):
            issue("record_case_identity", "Record and campaign case SHA-256 differ", key)
        if row.get("model") != case.get("model"):
            issue("record_model_identity", "Record physical-model label differs from its case", key)
        setup_failure = str(row.get("reason", "")).startswith("physical_case_setup_failed:")
        if not setup_failure:
            if row.get("machine_sha256") != provenance.get("machine_sha256"):
                issue("record_machine_identity", "Record machine identity differs", key)
            if row.get("source_sha256") != provenance.get("source", {}).get("sha256"):
                issue("record_source_identity", "Record source identity differs", key)
            settings = row.get("settings", {})
            profile = {name: deepcopy(protocol.get(name)) for name in ("scheme", "steady", "residual")}
            if "fluxes" in protocol:
                profile["scheme"]["flux"]=requested.get("flux")
            if requested.get("flux")=="roe" and "roe_options" in protocol:
                profile["roe_options"]=protocol["roe_options"]
            profile.update(cap=row.get("cap"), reconstruction=row.get("reconstruction"))
            if settings != profile or row.get("numerical_profile_sha256") != identity(profile):
                issue("record_profile_identity", "Record numerical settings/hash differ from protocol", key)
            metadata=row.get("metadata",{})
            if metadata and row.get("backend")=="current" and "fluxes" in protocol:
                if metadata.get("flux")!=requested.get("flux"):
                    issue("bound_flux_identity", "Bound operator flux differs from requested flux", key)
                if requested.get("flux")=="roe":
                    for name,value in protocol.get("roe_options",{}).items():
                        if (metadata.get("roe") or {}).get(name)!=value:
                            issue("bound_roe_identity", "Bound Roe policy differs: "+name, key)
                    if row.get("success"):
                        diagnostics=row.get("roe_diagnostics") or {}
                        if diagnostics.get("invalid_faces")!=0 or diagnostics.get("error"):
                            issue("final_roe_faces", "Successful Roe solve lacks a zero-invalid-face final audit", key)
        if row.get("backend") == "legacy" and row.get("metadata"):
            metadata = row["metadata"]
            if metadata.get("verified") is not True or metadata.get("manifest_sha256") != manifest_digest or metadata.get("manifest_files") != len(manifest):
                issue("record_legacy_identity", "Legacy binding did not identify the verified frozen source", key)
        samples = row.get("samples", [])
        timing = row.get("timing", {})
        expected_samples = requested.get("expected_samples", 0)
        warmup = row.get("warmup")
        completed_count = len(samples)
        if completed_count > expected_samples:
            issue("sample_count", "More timing samples than requested", key)
        if row.get("success") and (not warmup or not warmup.get("success") or completed_count != expected_samples or not all(item.get("success") for item in samples)):
            issue("sample_acceptance", "Success claimed without successful excluded solve and every requested sample", key)
        if warmup and not warmup.get("success") and samples:
            issue("warmup_exclusion", "Timed samples were collected after an unsuccessful excluded solve", key)
        if warmup and timing.get("expected_samples") != expected_samples:
            issue("expected_sample_count", "Record repetition target differs from mesh protocol", key)
        if completed_count < expected_samples and row.get("success") is not False:
            issue("incomplete_sample_status", "Incomplete timing collection was not identified as unsuccessful", key)
        if not row.get("success") and not row.get("reason"):
            issue("failure_reason", "An unsuccessful case has no explanation", key)
        times = [item.get("elapsed_s") for item in samples]
        if any(not _finite(value) or value < 0 for value in times):
            issue("timing_samples", "Negative/nonfinite timing sample", key)
        elif times:
            array = np.asarray(times)
            for name, value in (("mean_s", float(array.mean())), ("min_s", float(array.min())),
                                ("max_s", float(array.max())), ("std_s", float(array.std(ddof=1)) if len(array) > 1 else 0.0)):
                if not _close(timing.get(name), value):
                    issue("timing_statistics", name + " does not match retained samples", key)
            if timing.get("samples_s") != times:
                issue("timing_sample_identity", "Timing vector differs from sample records", key)
        elif any(timing.get(name) is not None for name in ("mean_s", "min_s", "max_s", "std_s")):
            issue("empty_timing_statistics", "Timing statistics exist without samples", key)

        monitor = _residual_acceptance(row.get("metrics", {}), policy)
        if row.get("success") and not monitor["passed"]:
            issue("representative_original_residual", monitor, key)
        if monitor["available"]:
            for saved, recomputed in (("all_equations_passed", "equations"), ("boundary_passed", "boundary"), ("global_passed", "global_balance")):
                if saved in row.get("metrics", {}) and bool(row["metrics"][saved]) != monitor[recomputed]:
                    issue("representative_monitor_flags", saved + " disagrees with saved original norms", key)
        if "history" in row and "work" in row:
            history = row["history"]
            measured = dict(accepted_steps=sum(item.get("phase") == "accepted" for item in history),
                            rejected_steps=sum(item.get("phase") == "rejected" for item in history),
                            newton_corrections=sum(item.get("inner_iterations", 0) for item in history),
                            rejected_newton_trials=sum(item.get("rejected_newton_trials", 0) for item in history),
                            max_inner_iterations=max((item.get("inner_iterations", 0) for item in history), default=0))
            if row["work"] != measured:
                issue("representative_work", "Saved work counters differ from the retained trajectory", key)
        sample_monitor_passes = []
        for number, sample in enumerate(samples, 1):
            assessment = _residual_acceptance(sample.get("metrics", {}), policy)
            sample_monitor_passes.append(bool(sample.get("success") and assessment["passed"]))
            if sample.get("success") and not assessment["passed"]:
                issue("sample_original_residual", dict(repeat=number, assessment=assessment), key)
            digest = sample.get("state_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                issue("sample_state_identity", f"Missing state hash for repeat {number}", key)
        failed_samples = {number for number, sample in enumerate(samples, 1) if not sample.get("success")}
        diagnostics = row.get("failure_diagnostics", [])
        diagnostic_numbers = [item.get("repeat") for item in diagnostics]
        if set(diagnostic_numbers) != failed_samples or len(diagnostic_numbers) != len(set(diagnostic_numbers)):
            issue("failed_sample_diagnostics", "Failed samples and full diagnostic records disagree", key)
        for diagnostic in diagnostics:
            if not all(field in diagnostic for field in ("state", "fields", "metrics", "history", "reason")):
                issue("failed_sample_evidence", "Failed sample lacks its full state/fields/metrics/history", key)

        representative = row.get("state", {}).get("full")
        if samples and representative is not None:
            selected = sorted(samples, key=lambda item: item["elapsed_s"])[len(samples) // 2]
            array = np.asarray(representative, dtype=float)
            if np.isfinite(array).all():
                digest = sha256(array.tobytes()).hexdigest()
                if digest != selected.get("state_sha256"):
                    issue("representative_identity", "Full state does not match the median-duration sample", key)
            elif selected.get("success"):
                issue("representative_finiteness", "A successful representative contains null/nonfinite state values", key)
            else:
                warnings.append(dict(record=key, check="failed_state_nonfinite", detail="Nonfinite failed state retained as JSON nulls; its original NaN bit pattern cannot be rehashed"))
        if warmup and not warmup.get("success") and (representative is None or "history" not in row):
            issue("failed_warmup_evidence", "Failed excluded solve lacks representative state/history", key)

        targets = definition.get("accuracy", {})
        accuracy_required = row.get("cells", 0) >= targets.get("recommended_finest_cells", math.inf)
        accuracy_assessed = row.get("accuracy_assessed") is True
        if "accuracy_assessed" in row and row["accuracy_assessed"] != accuracy_required:
            issue("accuracy_resolution", "Finest-resolution assessment flag differs from the case definition", key)
        if row.get("success") and accuracy_required and not accuracy_assessed:
            issue("accuracy_missing", "Successful finest-resolution result lacks its required accuracy assessment", key)
        branch = row.get("branch_passed", False)
        analytical = row.get("analytical_passed", False)
        if "branch_checks" in row and not row.get("assessment_error") and bool(all(row["branch_checks"].values())) != branch:
            issue("branch_consistency", "Branch flag differs from its component checks", key)
        expected_checks = {}
        fields = row.get("errors", {})
        conservative = row.get("conservative_errors", {})
        if not row.get("assessment_error") and (fields or conservative):
            for collection in (fields, conservative):
                for name, error in collection.items():
                    for norm in ("l1", "l2", "linf"):
                        value, scale = error.get(norm), error.get("scale")
                        if not _finite(value) or value < 0 or not _finite(scale) or scale <= 0 or not _close(error.get("normalized_" + norm), value / scale):
                            issue("error_normalization", name + ": " + norm, key)
                    if error.get("relative_active"):
                        if not _finite(error.get("relative_l1")) or error["relative_l1"] < 0:
                            issue("relative_error", name + " has an invalid active relative error", key)
                    elif error.get("relative_l1") is not None:
                        issue("relative_error", name + " has a relative error despite an inactive reference scale", key)
                    if _finite(error.get("l1")) and _finite(error.get("l2")) and _finite(error.get("linf")):
                        if error["l1"] > error["l2"] + 1e-11 * max(1, error["linf"]) or error["l2"] > error["linf"] + 1e-11 * max(1, error["linf"]):
                            issue("error_norm_ordering", name, key)
            normalized_values = [error.get("normalized_l1") for error in conservative.values()]
            relative_values = [error.get("relative_l1") if error.get("relative_active") else error.get("normalized_l1") for error in conservative.values()]
            for label, values in (("max_conservative_normalized_l1", normalized_values), ("max_conservative_relative_l1", relative_values)):
                if values and all(_finite(value) for value in values) and not _close(row.get("metrics", {}).get(label), max(values)):
                    issue("conservative_error_summary", label + " differs from component errors", key)
            if "normalized_field_l1" in targets:
                required = ("p", "rho", "velocity", "M", "T")
                expected_checks["normalized_field_l1"] = all(name in fields and _finite(fields[name].get("normalized_l1")) for name in required) and max(fields[name]["normalized_l1"] for name in required) <= targets["normalized_field_l1"]
            if "conservative_normalized_l1" in targets:
                values = [error.get("normalized_l1") for error in conservative.values()]
                expected_checks["conservative_normalized_l1"] = bool(values) and all(_finite(value) for value in values) and max(values) < targets["conservative_normalized_l1"]
            if "Cp_length_mean_absolute" in targets:
                value = fields.get("Cp", {}).get("l1")
                expected_checks["Cp_length_mean_absolute"] = _finite(value) and value <= targets["Cp_length_mean_absolute"]
            shock = row.get("shock", {})
            if shock.get("expected") is not None:
                value = shock.get("error_cells")
                expected_checks["shock_position"] = _finite(value) and value <= targets.get("shock_position_cells", 2.0)
            if row.get("accuracy_targets") != targets or row.get("accuracy_checks") != expected_checks:
                issue("accuracy_checks", dict(expected=expected_checks, recorded=row.get("accuracy_checks")), key)
            if bool(all(expected_checks.values())) != analytical:
                issue("analytical_acceptance", "Analytical flag differs from recomputed saved error checks", key)
            projected = row.get("backend") == "current" or row.get("model") == "euler_annular"
            expected_representation = "conservative finite-volume projection" if projected else "native coordinate point sampling"
            if row.get("reference_representation") != expected_representation:
                issue("reference_representation", "Reference comparison does not match backend storage convention", key)
        valid = bool(row.get("success") and monitor["passed"] and all(sample_monitor_passes)
                     and completed_count == expected_samples and branch
                     and (not accuracy_required or (accuracy_assessed and analytical))
                     and source_stable and row_environment_clean)
        if row.get("timing_valid") is not valid:
            issue("timing_validity", dict(expected=valid, recorded=row.get("timing_valid")), key)
        hashes = {sample.get("state_sha256") for sample in samples if sample.get("success")}
        if len(hashes) > 1:
            warnings.append(dict(record=key, check="repeat_state_variation", detail="Successful repeats are not bitwise identical; inspect retained residuals and physical root agreement"))
        records.append(dict(id=key, status=row.get("status"), expected_samples=expected_samples,
                            retained_samples=completed_count, original_residual_passed=monitor["passed"],
                            nonlinear_success=bool(row.get("success")), branch_passed=bool(branch),
                            accuracy_required=accuracy_required, accuracy_assessed=accuracy_assessed,
                            analytical_passed=bool(analytical), timing_environment_status=row_environment_status,
                            timing_valid=valid, repeated_state_bitwise_identical=len(hashes) <= 1))

    counts = dict(configurations_expected=len(expected), configurations_retained=len(rows),
                  excluded_solves_retained=sum("warmup" in row for row in rows),
                  timed_samples_retained=sum(len(row.get("samples", [])) for row in rows),
                  nonlinear_success=sum(row["nonlinear_success"] for row in records),
                  branch_passed=sum(row["branch_passed"] for row in records),
                  accuracy_required=sum(row["accuracy_required"] for row in records),
                  accuracy_assessed=sum(row["accuracy_assessed"] for row in records),
                  analytical_passed_at_required_resolution=sum(row["accuracy_assessed"] and row["analytical_passed"] for row in records),
                  timing_valid=sum(row["timing_valid"] for row in records), statuses=actual_counts)
    passed = not issues
    numerical_passed = bool(records) and all(row["nonlinear_success"] and row["branch_passed"] and (not row["accuracy_required"] or (row["accuracy_assessed"] and row["analytical_passed"])) for row in records)
    return dict(schema_version=1, scope="Saved-data identities, coverage, monitor thresholds, acceptance flags and diagnostic retention; no solver execution or fresh PDE residual evaluation",
                input=str(input_path) if input_path is not None else None, campaign_id=campaign.get("campaign_id"),
                protocol_sha256=campaign.get("protocol_sha256"), source_sha256=provenance.get("source", {}).get("sha256"),
                harness_sha256=provenance.get("harness", {}).get("sha256"), legacy_manifest_sha256=manifest_digest,
                compared_with_live_checkout=compare_live, timing_environment=timing_environment,
                clean_timing_environment=environment_clean, passed=passed,
                numerical_and_analytical_acceptance=numerical_passed, counts=counts,
                issues=issues, warnings=warnings, records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Explicit full campaign summary JSON")
    parser.add_argument("--output", type=Path, help="Audit JSON; default alongside input")
    parser.add_argument("--require-numerical-pass", action="store_true", help="Also return failure when any case misses numerical/accuracy acceptance")
    args = parser.parse_args(argv)
    input_path = args.input if args.input.is_absolute() else STUDY / args.input
    if not input_path.is_file() and args.input.is_file():
        input_path = args.input
    input_path = input_path.resolve()
    output = args.output if args.output is not None else input_path.with_name("audit.json")
    if not output.is_absolute():
        output = STUDY / output
    if output.resolve() == input_path:
        raise ValueError("Audit output must not overwrite its campaign input")
    campaign = json.loads(input_path.read_text(encoding="utf-8"))
    result = audit_campaign(campaign, input_path=input_path)
    result["input_sha256"] = sha256(input_path.read_bytes()).hexdigest()
    save(output, result)
    print("Evidence audit:", "PASSED" if result["passed"] else "FAILED",
          "| Numerical/analytical acceptance:", result["numerical_and_analytical_acceptance"],
          "| Counts:", result["counts"])
    if result["issues"]:
        print(json.dumps(result["issues"], indent=2))
    if not result["passed"] or args.require_numerical_pass and not result["numerical_and_analytical_acceptance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
