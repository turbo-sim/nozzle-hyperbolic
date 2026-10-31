"""Snapshot the implementation, environment and existing verification evidence.

Run from the repository root:
  poetry run python -m development.roberto.historical.euler_annular.scripts.write_manifest

This reads Git status without staging, runs no solver/tests, and never replaces
the historical review_manifest.json. Rerun after additional results or audits
finish; unavailable/partial evidence remains explicitly incomplete.
"""

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib


STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[5]
BASELINE = STUDY/"outputs"/"implementation_baseline"
CASE_FILE = re.compile(r"^.+?_(?:N)?[1-9][0-9]*\.json$")


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), REPOSITORY)).as_posix()


def fingerprint(path):
    return dict(path=relative(path), sha256=sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def source_snapshot():
    paths = set((REPOSITORY/"src").rglob("*.py"))
    paths.update((REPOSITORY/"tests").glob("test_*.py"))
    paths.update((STUDY/"scripts").glob("*.py"))
    paths.update((STUDY/"cases").glob("*.yaml"))
    paths.add(STUDY/"run_case.py")
    paths.update(REPOSITORY/name for name in (
        "pyproject.toml", "poetry.lock",
        "development/roberto/historical/euler_nozzle/verify_euler_nozzle_steady.py",
        "development/roberto/historical/euler_nozzle/verify_euler_nozzle_sod.py",
        "development/roberto/historical/BC_residual/scripts/source_adapter.py",
        "development/roberto/historical/BC_residual/scripts/boundary_system.py",
        "development/roberto/historical/BC_residual/scripts/coupled_step.py",
        "development/roberto/historical/BC_residual/scripts/compare_implicit.py",
    ))
    return {relative(path): sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def git_observation(root):
    def command(*args):
        process = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        return process.stdout.rstrip() if process.returncode == 0 else None

    status = command("status", "--short", "--untracked-files=all")
    return dict(head=command("rev-parse", "HEAD"), branch=command("branch", "--show-current"),
                dirty_paths=status.splitlines() if status else [],
                observation_only=True, staging_or_commits_performed=False)


def package_identity(module_name, distribution=None):
    try:
        module = import_module(module_name)
        return dict(available=True, distribution=distribution or module_name,
                    version=metadata.version(distribution or module_name),
                    module_version=getattr(module, "__version__", None),
                    import_path=relative(module.__file__))
    except Exception as error:
        return dict(available=False, exception_type=type(error).__name__, error=str(error))


def read_json(path):
    if not path.is_file():
        return dict(status="not_yet_available", path=relative(path)), None
    evidence = dict(status="available", **fingerprint(path))
    try:
        return evidence, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        evidence.update(status="unreadable_or_partial", error=str(error))
        return evidence, None


def pytest_evidence():
    logs = []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for path in sorted(BASELINE.glob("*test*.log")):
        text = ansi.sub("", path.read_text(encoding="utf-8", errors="replace"))
        summaries = [line.strip("= \r") for line in text.splitlines()
                     if re.search(r"\b[0-9]+ (?:passed|failed|skipped|errors?)\b", line)
                     and " in " in line]
        summary = summaries[-1] if summaries else None
        counts = ({label: int(count) for count, label in
                   re.findall(r"([0-9]+) (passed|failed|skipped|xfailed|xpassed|errors?|warnings?)\b", summary)}
                  if summary else {})
        logs.append(dict(**fingerprint(path), terminal_summary=summary,
                         terminal_summary_found=bool(summary), counts=counts,
                         has_failures=bool(counts.get("failed", 0)+counts.get("error", 0)
                                           +counts.get("errors", 0))))
    return dict(logs=logs, combined_total_not_inferred=True,
                note="Logs retain earlier failed attempts as well as later passing runs. Suites overlap; counts must not be added or attributed automatically to the final source snapshot.")


def parity_evidence(path):
    evidence, report = read_json(path)
    if report is not None:
        cases = report.get("cases", [])
        evidence.update(
            reference=report.get("reference"), configurations=len(cases),
            all_comparisons_bitwise_equal=bool(cases) and all(
                comparison.get("bitwise_equal", False) for case in cases
                for comparison in case.get("comparisons", {}).values()),
            configurations_checked=[{key: case.get(key) for key in
                                     ("variant", "reconstruction", "linear_solver", "local_jvp", "newton_tol", "count")}
                                    for case in cases],
        )
    return evidence


def legacy_evidence():
    steady, report = read_json(BASELINE/"legacy_steady"/"summary.json")
    if report is not None:
        cases = report.get("cases", [])
        steady.update(cases=[{key: run.get(key) for key in
                              ("case", "converged", "passed", "ssr", "normalized_L1",
                               "relative_boundary_mass_imbalance", "throat_M_exact", "throat_M_numerical")}
                             for run in cases],
                      all_native_roots_converged=all(run.get("converged", False) for run in cases),
                      all_analytical_checks_passed=all(run.get("passed", False) for run in cases),
                      failed_checks=[run["case"] for run in cases if not run.get("passed", False)],
                      note="The known sonic-throat/subsonic-exit analytical accuracy failure is retained; native convergence is a separate result.")
    sod, report = read_json(BASELINE/"legacy_sod"/"summary.json")
    if report is not None:
        snapshots = report.get("snapshots", [])
        sod.update(snapshots=snapshots,
                   all_snapshots_passed=bool(snapshots) and all(run.get("passed", False) for run in snapshots),
                   maximum_normalized_budget_error=max(
                       (abs(value) for run in snapshots
                        for value in run.get("normalized_conservation_error", {}).values()), default=None))
    boundary, report = read_json(BASELINE/"boundary_adapter_smoke.json")
    if report is not None:
        boundary.update(
            points=report.get("points"), case=report.get("case"),
            linear_relative_defect=report.get("linear_relative_defect"),
            old_boundary_independent=report.get("old_boundary_independent"),
            current_boundary_flux_dependent=report.get("current_boundary_flux_dependent"),
            runs=[{key: run.get(key) for key in ("sample_kind", "mode", "success", "accepted_updates", "message")}
                  for run in report.get("runs", [])],
            full_100_sample_timing_campaign_rerun=False,
        )
    return dict(steady=steady, sod=sod, retained_boundary_study_smoke=boundary)


def campaign_evidence(directory, current_sources):
    files = []
    for path in sorted(directory.glob("*.json")):
        if not CASE_FILE.fullmatch(path.name):
            continue
        evidence, report = read_json(path)
        if report is not None and {"case", "cells", "methods"} <= report.keys():
            saved_sources = report.get("source_hashes", {})
            evidence.update(
                case=report["case"], cells=report["cells"],
                settings=report.get("settings", {}),
                delta_factor=report.get("settings", {}).get("delta_factor"),
                flux_method=report.get("settings", {}).get("flux_method"),
                source_matches_current=bool(saved_sources) and all(
                    current_sources.get(name) == value for name, value in saved_sources.items()),
                methods={method: dict(
                    recorded_success=run.get("success"), message=run.get("message"),
                    analytical_target_met=run.get("analytical_target_met"),
                    spatial_rms=run.get("metrics", {}).get("spatial_rms"),
                    boundary_max=run.get("metrics", {}).get("boundary_max"),
                    conservation_max=run.get("metrics", {}).get("conservation_max"),
                    Cp_max_error=run.get("errors", {}).get("Cp_max"),
                    shock_error_cells=run.get("errors", {}).get("shock_error_cells"),
                ) for method, run in report["methods"].items()},
            )
        files.append(evidence)
    acceptance, result = read_json(directory/"acceptance.json")
    if result is not None:
        acceptance["recorded_result"] = result
    audit, result = read_json(directory/"audits.json")
    if result is not None:
        audit["recorded_result"] = {key: result.get(key) for key in (
            "audit_pass", "all_current_roots_met", "finest_grid_accuracy_met",
            "verification_acceptance", "audited_files", "failed_files", "source_unchanged_during_audit")}
        hashes = result.get("current_source_hashes", {})
        audit["audit_source_matches_current"] = bool(hashes) and all(
            current_sources.get(name.replace("\\", "/")) == value for name, value in hashes.items())
    return dict(directory=relative(directory), saved_cases=files,
                acceptance=acceptance, independent_saved_field_audit=audit,
                full_campaign_acceptance_available=acceptance["status"] == "available",
                independent_audit_available=audit["status"] == "available")


def narrow_width_diagnosis():
    """Retain failed narrow-width evidence without relabeling it as a new run."""
    path = STUDY/"outputs"/"diagnosis"/"shock35_N200_newton.json"
    evidence, report = read_json(path)
    moved_input = STUDY/"outputs"/"narrow_width"/"shock35_200.json"
    if report is not None:
        evidence.update(
            original_input_path=report.get("input"),
            original_input_sha256=report.get("input_sha256"),
            retained_input_path=relative(moved_input),
            retained_input_hash_matches=(sha256(moved_input.read_bytes()).hexdigest()
                                         == report.get("input_sha256")
                                         if moved_input.is_file() else None),
            source_unchanged_during_diagnosis=report.get("source_unchanged"),
            methods={name: dict(
                last_failed_dt=run.get("last_failed_dt"),
                options=run.get("options"),
                stages=[{key: stage.get(key) for key in (
                    "label", "residual_rms", "colored_dense_relative_inf",
                    "banded_dense_linear_relative_defect", "dense_linear_relative_defect",
                    "banded_dense_relative_direction", "correction_max_relative",
                    "raw_condition_number", "scaled_condition_number",
                    "dense_jvp_relative_difference")}
                        for stage in run.get("stages", [])],
            ) for name, run in report.get("methods", {}).items()},
            interpretation="At the failed shock35 N200 states with delta_factor=1e-6, colored and dense AD Jacobians agree near roundoff; banded directions agree with dense solves. The stalled MUSCL inner iterate is ill-conditioned and every admissible Armijo trial through 2^-16 increases the residual. This is negative robustness evidence for the narrow-width configuration, not evidence that a structural coloring or banded assembly error caused the failure. Finite-difference probes show a very narrow local nonlinear scale; this diagnosis alone does not identify every source of nonlinearity.",
        )
    return evidence


def eos_ast_audit():
    annular = REPOSITORY/"src/nozzle_hyperbolic/models/euler_annular"
    paths = [path for path in annular.rglob("*.py")
             if "analytical" not in path.relative_to(annular).parts]
    paths.extend(REPOSITORY/"src/nozzle_hyperbolic"/name for name in (
        "thermodynamics/adapters.py", "numerics/characteristic.py", "numerics/jacobians.py",
        "numerics/linear.py", "numerics/reconstruction.py", "solvers/finite_volume.py",
        "solvers/time_marching/implicit.py", "solvers/time_marching/cfl.py",
    ))
    forbidden = {"gamma", "cp", "cv"}
    findings = []
    for path in sorted(paths):
        if not path.is_file():
            findings.append(dict(path=relative(path), kind="missing_required_scan_file"))
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text, filename=str(path))):
            kind = None
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                kind = "forbidden_property_attribute"
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in forbidden:
                # This local is the dimensionless pressure-recovery coefficient
                # (p-p_in)/(p0-p_in), not thermodynamic heat capacity. Actual
                # .cp and ['cp'] property accesses remain forbidden everywhere.
                if not (node.id == "cp" and path == REPOSITORY/"src/nozzle_hyperbolic/models/euler_annular/monitoring.py"):
                    kind = "forbidden_property_name"
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                  and node.slice.value in forbidden):
                kind = "forbidden_property_subscript"
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module or "", *[alias.name for alias in node.names]])
                if any("analytical" in module.split(".") or module.startswith("development")
                       for module in modules):
                    kind = "forbidden_reference_or_development_import"
            if kind:
                findings.append(dict(path=relative(path), line=node.lineno, kind=kind,
                                     text=lines[node.lineno-1].strip()))
    fixtures = []
    for filename, name in (("test_euler_annular_fvm.py", "HPInterfaceOnly"),
                           ("test_euler_annular_physics.py", "RestrictedHPFluid")):
        path = REPOSITORY/"tests"/filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fixture = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        fixtures.append(dict(path=relative(path), class_name=name, line=fixture.lineno,
                             description=ast.get_docstring(fixture)))
    return dict(
        passed=not findings, scanned_files=[relative(path) for path in sorted(paths)], findings=findings,
        forbidden_accesses=sorted(forbidden), analytical_reference_directory_excluded=True,
        method="Python AST attribute/name/subscript access and import inspection; docstrings/comments do not count as EOS accesses",
        production_fixture_evidence=fixtures,
        evidence_scope="Restricted HP-only interface and pressure/enthalpy-shifted stiffened-gas fixtures exercise the general-EOS implementation. Passing fixture/interface tests do not establish a real-fluid annular flow validation campaign.",
        real_fluid_annular_flow_validation_claimed=False,
        dimensionless_monitor_identifier_exception="Local cp in annular/monitoring.py is pressure recovery; all thermodynamic cp attribute/subscript accesses remain scanned",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs/provenance/implementation_manifest.json")
    args = parser.parse_args()
    before = source_snapshot()
    git_before = git_observation(REPOSITORY)
    project = tomllib.loads((REPOSITORY/"pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPOSITORY/"poetry.lock").read_text(encoding="utf-8"))
    dependencies = project["tool"]["poetry"]["dependencies"]
    packages = {name: package_identity(name) for name in ("jax", "jaxlib", "jaxprop", "nurbspy", "numpy", "scipy")}
    geometry = package_identity("nurbspy.numpy", "nurbspy")
    selected = {"jax", "jaxlib", "jaxprop", "nurbspy", "numpy", "scipy", "equinox", "diffrax", "optimistix", "quadax", "pysolver-view", "pyyaml"}
    locked = {item["name"]: dict(version=item["version"], python_versions=item.get("python-versions"),
                                source=item.get("source"))
              for item in lock["package"] if item["name"] in selected}
    campaigns = {name: campaign_evidence(STUDY/"outputs"/name, before)
                 for name in ("grid_sequence_monitored_full", "grid_sequence", "grid_sequence_monitored", "verification", "nonuniform", "initial_cases", "work_controller",
                              "narrow_width", "regularization_width", "rusanov_diagnostic")}
    ast_audit = eos_ast_audit()
    evidence = dict(
        pytest=pytest_evidence(), legacy=legacy_evidence(),
        nozzle_extraction_parity=parity_evidence(BASELINE/"nozzle_extraction_parity.json"),
        nozzle_public_operator_parity=parity_evidence(BASELINE/"nozzle_metadata_parity.json"),
        narrow_width_failure_diagnosis=narrow_width_diagnosis(),
        campaigns=campaigns,
        wall_sources=read_json(STUDY/"outputs/wall_sources/audit.json")[0],
        wall_source_tests=read_json(STUDY/"outputs/wall_sources/production_tests.json")[0],
        residual_monitoring_tests=read_json(STUDY/"outputs/residual_monitoring/production_tests.json")[0],
        residual_monitoring_cases=read_json(STUDY/"outputs/residual_monitoring/validation_cases.json")[0],
        yaml_case_definitions=read_json(STUDY/"outputs/case_config/audit.json")[0],
        cp_accuracy_reassessment=read_json(STUDY/"outputs/grid_sequence/accuracy_reassessment.json")[0],
        note="Saved artifacts are evidence at their recorded checkpoints, not newly executed by this generator. Missing acceptance/audit files remain incomplete; retained accuracy/root failures are not waived.",
    )
    after = source_snapshot()
    local_nurbspy = REPOSITORY.parent.parent/"nurbspy"
    report = dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        repository=".", generator=relative(Path(__file__)),
        historical_review_manifest_replaced=False, git=git_before,
        historical_review_manifest=(fingerprint(STUDY/"old_notes/review_manifest.json")
                                    if (STUDY/"old_notes/review_manifest.json").is_file() else None),
        decision_documents=[fingerprint(STUDY/name) for name in (
            "old_notes/questionaire.md", "old_notes/IMPLEMENTATION_PLAN.md", "scripts/report_assets/implementation.html", "scripts/report_assets/theory.html", "scripts/report_assets/notes.html")
                            if (STUDY/name).is_file()],
        git_after=git_observation(REPOSITORY),
        source_fingerprints_before=before, source_fingerprints_after=after,
        source_unchanged_during_generation=before == after,
        project_files=[fingerprint(REPOSITORY/name) for name in ("pyproject.toml", "poetry.lock")],
        python=dict(executable=relative(sys.executable), version=sys.version,
                    implementation=sys.implementation.name),
        imported_packages=packages, geometry_backend=geometry,
        declared_dependencies={key: value for key, value in dependencies.items() if key == "python" or key in selected},
        locked_packages=locked, lock_metadata=lock.get("metadata", {}),
        geometry_backend_decision=dict(
            selected="NURBSpy core NumPy backend for fixed host preprocessing; sampled arrays feed the JAX FVM",
            rationale="NURBSpy 1.6's optional JAX extra requires the JAX 0.12 ecosystem. Installing core 1.6 supplies the required general NURBS geometry while retaining the verified JAX 0.6/JAXProp numerical stack and existing Quadax helpers. The latest JAX extra is not selected or claimed compatible with the retained stack.",
            geometry_automatic_differentiation_provided=False,
            numerical_fvm_uses_jax=True,
        ),
        latest_local_nurbspy=(dict(path=relative(local_nurbspy), **git_observation(local_nurbspy))
                              if local_nurbspy.is_dir() else dict(status="checkout_not_available")),
        accepted_scope=dict(
            model="quasi-1D axisymmetric general-EOS Euler with swirl",
            eos="Caller-supplied JAXProp states, sound speed and Gruneisen parameter; no numerical gamma/cp/cv closure",
            verification_backend="JAXProp FluidPerfectGas(Air), independent analytical references only",
            state_storage=["rho", "rho*v_m", "rho*ell", "rho*E"],
            nonlinear_variables=["p", "v_m", "ell", "h"], velocity_names=["v_m", "v_theta"],
            geometry="General NURBSpy midline/normal-height curves; radial/conical are special cases of one builder",
            verification_width="Constant normal b except straight-conical semi-divergence/cant-angle law",
            pressure_sweep="Fixed alpha_in; ell_in may change between operating points",
            boundaries="Coupled full residual, four algebraic rows per boundary, zero storage, no nested nonlinear BC solve",
            numerics="Shared regularized characteristic flux algebra, colored/banded Newton engine; constant and MUSCL reconstruction",
            wall_sources=dict(placeholders=["shear_stress_wall", "heat_flux_wall"],
                inputs="Fixed prescribed scalar or per-cell tau_w and q_w, default zero; no empirical correlations",
                forcing="Meridional drag, angular wall torque including radius, and heat flux in total energy; stationary shear has no total-energy sink"),
            verification_initialization="50/100/200/400 grid sequence; bounded primitive interpolation from successful stages, cold restart after a failed pair; nine-order relative target and separate 1e-8 safeguards; failures retained",
            monitoring=dict(
                production_interface="AnnularSolver.solve_steady_monitored (opt-in; legacy methods preserved)",
                definition="steady_spatial_fixed_scale_equation_v1",
                relative_target=1e-9, absolute_safeguard=1e-8,
                norms="volume-weighted per-equation RMS; also L1, maximum and worst cell",
                observables="YAML-configured Cp and actual numerical mass flux; optional accepted-window stationarity",
                primary_report_history="Full 72-solve grid_sequence_monitored_full rerun with actual initial and accepted equation histories",
                current_validation="68/72 meet all criteria; four N400 MUSCL stagnation failures retained; independent audits separate integrity from convergence",
                nozzle_migrated=False),
            characteristic_regularization_default=dict(
                delta_factor=0.1,
                selection="Prototype numerical default selected after controlled shock robustness comparisons; not a user-prescribed physical input",
                rationale="The original delta_factor=1e-6 campaign failed cold-start shock35 N200 solves for both reconstructions. Independent dense/colored/banded checks and controlled wider-width runs motivated selecting 0.1. The selected characteristic configuration must meet the same root and accuracy criteria; failures and diagnostic Rusanov runs remain separate evidence.",
                retained_negative_campaign="development/roberto/historical/euler_annular/outputs/narrow_width",
                selected_campaign="development/roberto/historical/euler_annular/outputs/grid_sequence_monitored_full",
                robustness_universal_guarantee=False,
            ),
            integration="Steady implicit focus with separate physical-time support",
            field_comparison_labels=["Analytical reference", "FVM (constant reconstruction)", "FVM (MUSCL reconstruction)"],
            nozzlex_comparison_required=False,
            deferred="Purely radial constant-width inflow cannot cross a smooth internal sonic point; additional transonic fixture remains unselected. Real-fluid application flow validation is deferred.",
        ),
        numerical_eos_ast_audit=ast_audit, evidence=evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True)+"\n", encoding="utf-8")
    print(json.dumps(dict(manifest=relative(args.output),
                          source_unchanged=report["source_unchanged_during_generation"],
                          eos_ast_audit_passed=ast_audit["passed"],
                          verification_acceptance=campaigns["grid_sequence_monitored_full"]["acceptance"],
                          verification_independent_audit=campaigns["grid_sequence_monitored_full"]["independent_saved_field_audit"]), indent=2))
    return 0 if before == after and ast_audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
