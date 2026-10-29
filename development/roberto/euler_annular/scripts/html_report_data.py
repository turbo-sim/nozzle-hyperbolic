"""Read-only, stdlib-only payload builder for the annular HTML results report.

Public API: ``build_report_data(study: Path) -> dict``. No numerical driver is
imported or run, no artifact is rewritten, and strict JSON serialization is
checked before returning. Authoritative case/method/grid literals are read from
the existing verification/report drivers without importing JAX or Matplotlib.
"""

import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .accuracy import ACCURACY_DEFINITION, assessed_errors, accuracy_target_met


FIELD_METADATA = (
    ("p", "Static pressure", "Pa"),
    ("rho", "Density", "kg/m3"),
    ("alpha_deg", "Flow angle", "deg"),
    ("M_m", "Meridional Mach", ""),
    ("M_theta", "Tangential Mach", ""),
    ("M", "Mach magnitude", ""),
)
METHOD_LABELS = {
    "constant": "FVM (constant reconstruction)",
    "muscl_van_leer": "FVM (MUSCL reconstruction)",
}
HISTORY_COLUMNS = (
    "step", "retry", "time", "time_new", "dt", "cfl", "inner_iterations",
    "inner_residual", "inner_status", "rejected_newton_trials", "admissible",
    "spatial_rms", "boundary_max", "conservation_max", "integrated_balance_scaled",
    "pseudo_time", "shock_proxy_m", "min_velocity_gradient", "inner_relative_update",
    "inlet_M_m", "outlet_M_m", "residual_only_accept", "next_cfl_factor",
    "inner_relative_update_nonfinite", "inner_residual_nonfinite",
    "phase", "attempt", "elapsed", "equation_l1", "equation_l2", "equation_linf",
    "equation_worst_cell", "equation_norm", "equation_relative", "relative_active",
    "equation_passed", "equation_max_ratio", "overall_max_ratio", "all_equations_passed",
    "boundary_ratio", "global_ratio", "finite", "monitors", "monitors_passed",
    "diagnostics_evaluated", "candidate_diagnostics", "candidate_monitor_values",
)
CASE_FILE = re.compile(r"^.+?_(?:N)?[1-9][0-9]*\.json$")


def resolve_verification_folder(study, folder=None):
    """Accept an absolute, repository-relative or study-relative data folder."""
    study = Path(study).resolve()
    if folder is None:
        return study/"outputs"/"grid_sequence_monitored_full"
    folder = Path(folder)
    if folder.is_absolute():
        return folder.resolve()
    candidate = study/folder
    return candidate.resolve() if candidate.is_dir() or not folder.is_dir() else folder.resolve()


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _root_pass(saved, run):
    """Original saved spatial, boundary and conservation root contract."""
    monitoring = run.get("monitoring")
    if monitoring and monitoring.get("definition") in ("steady_spatial_fixed_scale_equation_v1", "steady_spatial_fixed_scale_per_equation_v1"):
        metrics, policy = run["metrics"], monitoring["policy"]
        initial, current = monitoring["initial_equation_norms"], metrics["equation_norm"]
        equations = (len(initial)==len(current)==4 and
            all(_finite_number(value) and value <= policy["absolute_tolerance"]
                and (start <= policy["initial_floor"] or value <= start*policy["relative_tolerance"])
                for start,value in zip(initial,current)))
        return bool(run["success"] and metrics.get("admissible") and metrics.get("finite")
            and equations and metrics["boundary_max"] <= policy["boundary_tolerance"]
            and metrics["conservation_max"] <= policy["global_tolerance"]
            and all(_finite_number(item["value"]) and (not item["required"] or item["stable"])
                    for item in metrics.get("monitors",{}).values()))
    tolerance = saved["settings"]["tolerance"]
    return (run["success"] and run["metrics"].get("admissible", False)
            and all(_finite_number(run["metrics"][key]) and run["metrics"][key] <= tolerance
                    for key in ("spatial_rms", "boundary_max", "conservation_max")))


def _accuracy_pass(saved, run):
    return accuracy_target_met(saved, run)


def _history(records, columns=HISTORY_COLUMNS):
    """No trajectory sampling: retain accepted and rejected records separately."""
    keys = [key for key in columns if any(key in row for row in records)]
    accepted = [[row.get(key) for key in keys] for row in records if row.get("accepted", True)]
    rejected = [[row.get(key) for key in keys] for row in records if not row.get("accepted", True)]
    return dict(columns=keys, accepted=accepted, rejected=rejected,
                counts=dict(total=len(records), accepted=len(accepted), rejected=len(rejected),
                    initial=sum(row.get("phase")=="initial" for row in records),
                    accepted_updates=sum(row.get("accepted",True) and row.get("phase")!="initial" for row in records)),
                sampled=False)


class _Reader:
    def __init__(self, study):
        self.study = study.resolve()
        self.repository = self.study.parents[2]
        self.inputs = {}
        self.current_sources = {}
        self.nonfinite_inputs = []

    def relative(self, path):
        path = path.resolve()
        try:
            return path.relative_to(self.study).as_posix()
        except ValueError:
            return str(path)

    def read_text(self, path):
        raw = path.read_bytes()
        key = self.relative(path)
        signature = sha256(raw).hexdigest()
        if key in self.inputs and self.inputs[key]["sha256"] != signature:
            raise ValueError(f"Report input changed while loading: {key}")
        self.inputs[key] = dict(path=key, sha256=signature, bytes=len(raw))
        return raw.decode("utf-8")

    def json(self, path, *, optional=False):
        if optional and not path.is_file():
            return None
        try:
            data = json.loads(self.read_text(path))
        except (ValueError, OSError, UnicodeError) as error:
            raise ValueError(f"Cannot read report input {self.relative(path)}: {error}") from error
        # Report every nonfinite source location, including arrays omitted from
        # the compact payload, so trimming cannot conceal invalid source data.
        def inspect(value, location):
            if isinstance(value, float) and not math.isfinite(value):
                self.nonfinite_inputs.append(dict(file=self.relative(path), path=location,
                                                  kind="nan" if math.isnan(value) else "infinity"))
            elif isinstance(value, dict):
                for key, item in value.items():
                    inspect(item, f"{location}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    inspect(item, f"{location}[{index}]")
        inspect(data, "$")
        return data

    def literal(self, path, name):
        tree = ast.parse(self.read_text(path), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                    for target in node.targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"Expected an authoritative sequence literal: {name}")
                return tuple(value)
        raise ValueError(f"Missing authoritative literal {name} in {self.relative(path)}")

    def source_drift(self, recorded):
        differences, missing = [], []
        for name, original in recorded.items():
            normalized = name.replace("\\", "/")
            path = (self.repository/normalized).resolve()
            if not path.is_relative_to(self.repository) or not path.is_file():
                missing.append(normalized)
                continue
            if normalized not in self.current_sources:
                self.current_sources[normalized] = sha256(path.read_bytes()).hexdigest()
            current = self.current_sources[normalized]
            if current != original:
                differences.append(dict(path=normalized, recorded=original, current=current))
        return dict(recorded_fingerprints_available=bool(recorded),
                    matches_current=bool(recorded) and not differences and not missing,
                    differences=differences, missing=missing)

    def verify_unchanged(self):
        for name, signature in self.inputs.items():
            path = Path(name) if Path(name).is_absolute() else self.study/name
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != signature["sha256"]:
                raise ValueError(f"Report input changed while loading: {name}")
        for name, signature in self.current_sources.items():
            if sha256((self.repository/name).read_bytes()).hexdigest() != signature:
                raise ValueError(f"Numerical source changed while loading: {name}")


def _fields(fields, names, cells, *, required=True):
    fields = dict(fields)
    if "alpha_deg" in names and "alpha_deg" not in fields:
        fields["alpha_deg"] = [math.degrees(math.atan2(vt, vm))
                               for vm, vt in zip(fields["v_m"], fields["v_theta"])]
    if "M_theta" in names and "M_theta" not in fields:
        fields["M_theta"] = [vt/a if a is not None else
                               (vt*mm/vm if vm else math.copysign(mag, vt))
                               for vm, vt, mm, mag, a in zip(fields["v_m"], fields["v_theta"],
                               fields["M_m"], fields["M"], fields.get("a", [None]*cells))]
    result = {}
    for name in names:
        if name not in fields:
            if required:
                raise ValueError(f"Required saved field is missing: {name}")
            continue
        if len(fields[name]) != cells:
            raise ValueError(f"Saved field {name} has {len(fields[name])} values, expected {cells}")
        result[name] = fields[name]
    return result


def _audit(reader, path, saved):
    raw = reader.json(path, optional=True)
    if raw is None:
        return dict(available=False, path=reader.relative(path))
    methods = {}
    for method, run in raw.get("methods", {}).items():
        methods[method] = {key: run.get(key) for key in (
            "root_target_met", "accuracy_target_met", "admissible", "saved_arrays_reproduced",
            "effective_options", "actual_backend", "original_saved_state_unchanged",
            "recomputed_metrics", "boundary_mach", "boundary_regime_checks",
            "boundary_regime_tolerance", "integrated_balance", "transported_flux_spread",
        )}
    return dict(available=True, path=reader.relative(path), audit_pass=raw.get("audit_pass"),
                input_sha256=raw.get("input_sha256"),
                input_hash_matches=(raw.get("input_sha256") == reader.inputs[reader.relative(saved)]["sha256"]),
                root_tolerance=raw.get("root_tolerance"),
                reproduction_tolerance=raw.get("reproduction_tolerance"),
                no_newton_or_timestep_updates=raw.get("no_newton_or_timestep_updates"),
                sources={key: raw.get("sources", {}).get(key) for key in
                         ("matches_saved_snapshot", "changed", "missing_from_current_fingerprint")},
                audit_source_drift=reader.source_drift(raw.get("sources", {}).get("current", {})),
                versions=raw.get("versions"), methods=methods)


def _steady_mesh(reader, path, saved, methods):
    cells = saved["cells"]
    if len(saved["m"]) != cells or len(saved["m_faces"]) != cells+1:
        raise ValueError(f"Incomplete coordinate arrays: {reader.relative(path)}")
    if len(saved["r"]) != cells or len(saved["volumes"]) != cells:
        raise ValueError(f"Incomplete radius/volume arrays: {reader.relative(path)}")
    fields = [entry[0] for entry in FIELD_METADATA]
    audit = _audit(reader, path.with_name(path.stem+".audit.json"), path)
    numerical = {}
    for method in methods:
        run = saved["methods"][method]
        if (not isinstance(run["success"], bool) or len(run["state"]) != cells+2
                or any(len(row) != 4 for row in run["state"])):
            raise ValueError(f"Incomplete final state/status: {reader.relative(path)} {method}")
        if len(run["errors"]["conservative_L1"]) != 4:
            raise ValueError(f"Incomplete conservative error: {reader.relative(path)} {method}")
        numerical[method] = dict(
            fields=_fields(run["fields"], fields, cells), history=_history(run.get("history", [])),
            success=run["success"], message=run.get("message"), root_pass=_root_pass(saved, run),
            accuracy_pass=_accuracy_pass(saved, run),
            recorded_analytical_target_met=run.get("analytical_target_met"),
            metrics=run["metrics"], errors=assessed_errors(saved, run),
            effective_options=run.get("effective_options", run.get("options")),
            controller_audit=run.get("controller_audit"),
            monitoring=run.get("monitoring"),
            initialization=run.get("initialization"),
            audited_effective_options=audit.get("methods", {}).get(method, {}).get("effective_options"),
            flags=dict(admissible=run["metrics"].get("admissible"),
                       boundary_mach=run["metrics"].get("boundary_mach"),
                       boundary_constraints=run.get("boundary_constraints")),
        )
    return dict(
        cells=cells, path=reader.relative(path), m=saved["m"], m_faces=saved["m_faces"],
        r=saved["r"], volumes=saved["volumes"],
        reference=_fields(saved["reference"], fields, cells), methods=numerical,
        settings=saved["settings"],
        metadata={key: saved.get(key) for key in (
            "case", "backpressure", "inlet_regime", "outlet_regime", "alpha_in", "shock_m",
            "fluid", "reservoir", "reference_invariants", "versions")},
        source_drift=reader.source_drift(saved.get("source_hashes", {})), audit=audit,
    )


def _dataset_paths(directory):
    return sorted(path for path in directory.glob("*.json") if CASE_FILE.fullmatch(path.name))


def _transient(reader, directory, methods):
    results = []
    for path in _dataset_paths(directory):
        saved = reader.json(path)
        cells, numerical = saved["cells"], {}
        for method in methods:
            run = saved["methods"][method]
            budget = run["physical_budget"]
            accepted = budget["accepted_steps"]
            keys = [key for key in ("time", "dt", "content", "boundary_flux", "integrated_source",
                                     "step_defect", "step_defect_scaled")
                    if any(key in row for row in accepted)]
            compact_budget = {key: value for key, value in budget.items() if key != "accepted_steps"}
            compact_budget["steps"] = dict(columns=keys, rows=[[row.get(key) for key in keys]
                                                             for row in accepted], sampled=False)
            compact_budget["callback_checks"] = dict(
                count_matches=len(accepted) == run["checks"]["accepted_steps"],
                dt_sum_matches=math.isclose(sum(row["dt"] for row in accepted), run["physical_time"],
                                          rel_tol=1e-11, abs_tol=1e-14),
            )
            numerical[method] = {key: run.get(key) for key in (
                "checks", "message", "physical_time", "effective_options", "conservative_L1",
                "field_L1", "density_min", "density_max", "centroid")}
            numerical[method].update(fields=run["fields"], history=_history(run.get("history", [])),
                                     budget=compact_budget)
        results.append(dict(
            case=saved["case"], cells=cells, path=reader.relative(path), m=saved["m"],
            m_faces=saved["m_faces"], volumes=saved["volumes"], settings=saved["settings"],
            reference=saved["reference"]["fields"],
            initial_reference=saved["initial_reference"]["fields"], methods=numerical,
            versions=saved.get("versions"), source_drift=reader.source_drift(saved.get("source_hashes", {})),
        ))
    results.sort(key=lambda item: item["cells"])
    return dict(runs=results, acceptance=reader.json(directory/"acceptance.json", optional=True))


def _shock_audits(reader, directory):
    path = directory/"shock_audits.json"
    saved = reader.json(path, optional=True)
    if saved is None:
        return dict(available=False, path=reader.relative(path))
    sources = []
    for source in saved["sources"]:
        sources.append(dict(
            case=source["case"], cells=source["cells"],
            file=reader.relative(Path(source["file"])), json_sha256=source.get("json_sha256"),
            recorded_source_hash_differences=source.get("recorded_source_hash_differences"),
            methods=[{key: value for key, value in run.items()
                      if key != "raw_state_including_boundary_traces"} for run in source["methods"]],
        ))
    return dict(available=True, path=reader.relative(path), settings=saved["settings"],
                versions=saved.get("audit_versions"), sources=sources,
                audit_source_drift=reader.source_drift(saved.get("current_source_hashes", {})))


def _diagnostics(reader, outputs):
    negative, rusanov = [], []
    for directory, target in ((outputs/"narrow_width", negative), (outputs/"rusanov_diagnostic", rusanov)):
        for path in _dataset_paths(directory):
            saved = reader.json(path)
            for method, run in saved["methods"].items():
                target.append(dict(case=saved["case"], cells=saved["cells"], method=method,
                                   path=reader.relative(path), success=run["success"],
                                   root_pass=_root_pass(saved, run), accuracy_pass=_accuracy_pass(saved, run),
                                   message=run.get("message"), settings=saved["settings"],
                                   metrics=run["metrics"], errors=assessed_errors(saved, run)))
    path = outputs/"diagnosis"/"shock35_N200_newton.json"
    diagnosis = reader.json(path, optional=True)
    # The detailed trial values are useful; large copied source maps are not
    # needed twice in the embedded HTML payload.
    if diagnosis is not None:
        diagnosis = {key: value for key, value in diagnosis.items()
                     if key not in ("source_before", "source_after")}
    return dict(narrow_width=dict(entries=negative, failed=[row for row in negative if not row["success"]]),
                rusanov_diagnostic=dict(entries=rusanov,
                    root_passes=sum(row["root_pass"] for row in rusanov),
                    accuracy_passes=sum(row["accuracy_pass"] for row in rusanov)),
                narrow_width_newton=diagnosis)


def _monitoring_validation(reader, outputs, primary=None):
    """Summarize the selected new-policy campaign without duplicate histories."""
    rows=[]
    directory=primary if primary is not None else outputs/"grid_sequence_monitored"
    if not directory.is_dir():
        directory=outputs/"monitoring_smoke"
    for path in _dataset_paths(directory):
        saved=reader.json(path)
        for method,run in saved["methods"].items():
            meta=run.get("monitoring")
            if not meta:
                continue
            metrics=run["metrics"]
            active=[value for value,enabled in zip(metrics["equation_relative"],metrics["relative_active"]) if enabled]
            rows.append(dict(case=saved["case"],cells=saved["cells"],method=method,
                path=reader.relative(path),root_pass=_root_pass(saved,run),
                max_absolute=max(metrics["equation_norm"]),
                max_relative=max(active) if active else None,
                accepted_updates=meta["accepted_updates"],reason=meta["reason"],
                policy=meta["policy"],monitors=metrics.get("monitors",{}),
                source_drift=reader.source_drift(saved.get("source_hashes",{}))))
    return dict(available=bool(rows),runs=rows,
        directory=reader.relative(directory),
        note="Summary of the selected campaign; failed convergence targets are retained independently of analytical accuracy.")


def _sanitize(value, replacements, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        replacements.append(dict(path=path, kind="nan" if math.isnan(value) else "infinity"))
        return None
    if isinstance(value, dict):
        return {key: _sanitize(item, replacements, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_sanitize(item, replacements, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"Unsupported HTML payload value at {path}: {type(value).__name__}")


def build_report_data(study: Path, *, verification_folder: Path | None = None) -> dict:
    """Build a self-contained strict-JSON payload, preserving every recorded run.

    Paths in provenance and plot records are relative to ``study``. Histories
    are columnar arrays with no sampling; rejected timesteps remain available.
    Root flags retain the saved original stopping definition. Accuracy uses the
    current Cp mean-error criterion; original accuracy flags remain recorded.
    Independent audits and source drift are separate. No test or solver runs.
    """
    reader = _Reader(Path(study))
    reader.read_text(Path(__file__))
    reader.read_text(Path(__file__).with_name("accuracy.py"))
    study, outputs = reader.study, reader.study/"outputs"
    primary = resolve_verification_folder(study, verification_folder)
    verification = study/"scripts"/"verify_annular.py"
    cases = reader.literal(verification, "DEFAULT_CASES")
    methods = reader.literal(verification, "RECONSTRUCTIONS")
    meshes = reader.literal(verification, "DEFAULT_MESHES")
    results, missing = [], []
    for case in cases:
        records = []
        for cells in meshes:
            path = primary/f"{case}_{cells}.json"
            if not path.is_file():
                missing.append(reader.relative(path))
                continue
            saved = reader.json(path)
            if saved["case"] != case or saved["cells"] != cells:
                raise ValueError(f"Primary case/cell metadata mismatch: {reader.relative(path)}")
            try:
                records.append(_steady_mesh(reader, path, saved, methods))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Incomplete primary result {reader.relative(path)}: {error}") from error
        results.append(dict(id=case, label=case.replace("_", " ").capitalize(), meshes=records,
                            is_shocked=any(row["metadata"]["shock_m"] is not None for row in records),
                            grid_errors=[dict(cells=row["cells"], **{
                                method: dict(**row["methods"][method]["errors"],
                                             root_pass=row["methods"][method]["root_pass"],
                                             accuracy_pass=row["methods"][method]["accuracy_pass"])
                                for method in methods}) for row in records]))
    if missing:
        raise ValueError("HTML report requires the complete canonical case/grid/method matrix; missing:\n  "
                         + "\n  ".join(missing))
    entries = [(case["id"], row, method, row["methods"][method])
               for case in results for row in case["meshes"] for method in methods]
    finest = [entry for entry in entries if entry[1]["cells"] == max(meshes)]
    audited = [entry[1]["audit"]["methods"][entry[2]] for entry in entries
               if entry[1]["audit"]["available"] and entry[2] in entry[1]["audit"].get("methods", {})]
    payload = dict(
        schema_version=1,
        fields=[dict(key=key, label=label, unit=unit) for key, label, unit in FIELD_METADATA],
        methods=[dict(id=method, label=METHOD_LABELS.get(method, method)) for method in methods],
        reference_label="Analytical reference", meshes=list(meshes),
        summary=dict(
            cases=len(results), primary_files=sum(len(case["meshes"]) for case in results),
            method_entries=len(entries), root_passes=sum(run["root_pass"] for _, _, _, run in entries),
            finest_method_entries=len(finest),
            finest_accuracy_passes=sum(run["accuracy_pass"] for _, _, _, run in finest),
            finest_accuracy_failures=[dict(case=case, method=method)
                                      for case, _, method, run in finest if not run["accuracy_pass"]],
            audited_method_entries=len(audited),
            audited_root_passes=sum(bool(row["root_target_met"]) for row in audited),
            audited_array_reproduction_passes=sum(bool(row["saved_arrays_reproduced"]) for row in audited),
            root_tolerances=sorted({row["settings"]["tolerance"] for _, row, _, _ in entries}),
            relative_tolerances=sorted({run["monitoring"]["policy"]["relative_tolerance"]
                                       for _, _, _, run in entries if run.get("monitoring")}),
            convergence_failures=[dict(case=case,cells=row["cells"],method=method,
                                      reason=run.get("monitoring",{}).get("reason",run.get("message")))
                                  for case,row,method,run in entries if not run["root_pass"]],
            delta_factors=sorted({row["settings"].get("delta_factor") for _, row, _, _ in entries}, key=str),
        ),
        cases=results,
        controller_sweep=reader.json(outputs/"controller_sweep"/"summary.json", optional=True),
        implementation=reader.json(outputs/"provenance"/"implementation_manifest.json", optional=True),
        campaign=reader.json(outputs/"cfl_recommended"/"campaign.json", optional=True),
        controller_audit=reader.json(outputs/"cfl_recommended"/"controller_audit.json", optional=True),
        wall_sources=reader.json(outputs/"wall_sources"/"audit.json", optional=True),
        audits=reader.json(primary/"audits.json", optional=True),
        acceptance=reader.json(primary/"acceptance.json", optional=True),
        accuracy_reassessment=reader.json(primary/"accuracy_reassessment.json", optional=True),
        accuracy_definition=dict(identifier=ACCURACY_DEFINITION,
            Cp_mean_absolute_limit=0.01, Cp_weighting="physical meridional cell width / domain length",
            conservative_L1_max=1e-3, historical_result_files_unchanged=True),
        shocks=_shock_audits(reader, primary),
        nonuniform=[_steady_mesh(reader, path, reader.json(path), methods)
                    for path in _dataset_paths(outputs/"nonuniform")],
        nonuniform_audits=reader.json(outputs/"nonuniform"/"audits.json", optional=True),
        transient=_transient(reader, outputs/"transient", methods),
        diagnostics=_diagnostics(reader, outputs),
        monitoring_validation=_monitoring_validation(reader,outputs,primary),
    )
    payload["audit_source_drift"] = reader.source_drift(
        (payload["audits"] or {}).get("current_source_hashes", {}))
    payload["nonuniform_audit_source_drift"] = reader.source_drift(
        (payload["nonuniform_audits"] or {}).get("current_source_hashes", {}))
    reader.verify_unchanged()
    payload["provenance"] = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        loader="scripts/html_report_data.py", loader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_sources_or_saved_json_modified=False, solver_or_tests_run=False,
        all_read_inputs_unchanged_during_load=True, current_sources_unchanged_during_load=True,
        authoritative_configuration_paths=[reader.relative(verification)],
        primary_directory=reader.relative(primary),
        input_files=list(reader.inputs.values()), current_source_sha256=reader.current_sources,
        primary_file_count=payload["summary"]["primary_files"],
        rendered_method_entry_count=payload["summary"]["method_entries"],
        primary_source_drift=[dict(case=case["id"], cells=row["cells"], **row["source_drift"])
                              for case in results for row in case["meshes"]],
        nonfinite_source_values=dict(count=len(reader.nonfinite_inputs), locations=reader.nonfinite_inputs),
        trimming="Full final states/face arrays duplicate plotted fields and remain in the immutable source JSONs. Accepted and rejected histories are unsampled columnar arrays retaining original spatial/BC/balance values and inner statuses. Shock audits omit only the duplicate full raw-state array.",
    )
    replacements = []
    payload = _sanitize(payload, replacements)
    payload["provenance"]["nonfinite_payload_replacements"] = dict(count=len(replacements), locations=replacements)
    json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return payload
