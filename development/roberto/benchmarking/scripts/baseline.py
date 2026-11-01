"""Export an immutable, compact timing baseline and a saved-data Markdown report.

This module uses only the standard library. It never imports a physical model,
JAX, or a numerical driver, and never advances a flow solution. Relative CLI
paths are resolved beneath ``development/roberto/benchmarking``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
_STRICT_FLAGS = ("finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed")
_RECORD_KEYS = (
    "id", "case_id", "model", "label", "backend", "reconstruction", "cells", "cap",
    "settings", "numerical_profile_sha256", "case_sha256", "machine_sha256", "source_sha256",
    "status", "success", "reason", "timing_valid", "errors", "conservative_errors", "metrics",
    "analytical_passed", "accuracy_assessed", "accuracy_checks", "accuracy_targets",
    "branch_passed", "branch_checks", "shock", "reference_representation", "known_limit",
    "assessment_error", "work", "warmup", "timing", "metadata", "timing_environment",
)
_SAMPLE_KEYS = (
    "elapsed_s", "success", "reason", "work", "metrics", "state_sha256",
    "analytical_passed", "accuracy_assessed", "branch_passed", "accuracy_checks", "branch_checks",
)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def timing_environment_status(campaign):
    """Only an explicit clean environment permits comparative timing claims."""
    environment = campaign.get("timing_environment")
    status = environment.get("status") if isinstance(environment, dict) else environment
    if isinstance(status, str):
        status = status.strip().lower()
    return status if isinstance(status, str) and status in ("clean", "contended", "unverified") else "unverified"


def read_campaign(path):
    """Read the standalone summary or its explicitly linked scenario records.

    Reading a baseline is deliberately independent of the production source.
    Linked records must remain inside the summary directory; missing evidence
    causes an error rather than quietly exporting an incomplete baseline.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    campaign = json.loads(raw)
    if not isinstance(campaign, dict) or not isinstance(campaign.get("results"), list):
        raise ValueError("Expected a campaign JSON object containing a results list")
    evidence = {path.name: sha256(raw).hexdigest()}
    records = []
    for row in campaign["results"]:
        if isinstance(row, str):
            row = {"file": row}
        if not isinstance(row, dict):
            raise ValueError("Each campaign result must be an object or a JSON filename")
        filename = row.get("file")
        if isinstance(filename, str):
            linked = (path.parent / filename).resolve()
            if not linked.is_relative_to(path.parent) or linked.suffix.lower() != ".json":
                raise ValueError(f"Result reference escapes the campaign directory: {filename}")
            blob = linked.read_bytes()
            detail = json.loads(blob)
            if not isinstance(detail, dict):
                raise ValueError(f"Referenced result must be a JSON object: {filename}")
            evidence[linked.relative_to(path.parent).as_posix()] = sha256(blob).hexdigest()
            row = {**detail, **row}
        records.append(row)
    campaign["results"] = records
    return campaign, {
        "sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
        "evidence_sha256": _identity(evidence),
        "evidence_files": len(evidence),
    }


def _sample_valid(sample):
    if not isinstance(sample, dict) or sample.get("success") is not True:
        return False
    metrics = sample.get("metrics", {})
    if not all(metrics.get(key) is True for key in _STRICT_FLAGS):
        return False
    if sample.get("branch_passed") is False:
        return False
    if sample.get("accuracy_assessed") is True and sample.get("analytical_passed") is not True:
        return False
    return True


def timing_valid(record):
    """Conservative eligibility for a time-to-solution comparison.

    Numerical failure timings remain in the baseline, but never enter a ratio.
    The campaign runner owns the residual audit; require its separate final
    flags as well as all recorded repetitions and the excluded warmup to pass.
    """
    if any(record.get(key) is not True for key in ("success", "branch_passed", "timing_valid")):
        return False
    if timing_environment_status(record) != "clean":
        return False
    environment = record.get("timing_environment")
    if isinstance(environment, dict) and any(
        timing_environment_status({"timing_environment": environment[name]}) != "clean"
        for name in ("before", "after") if name in environment
    ):
        return False
    if record.get("status") != "passed":
        return False
    if record.get("accuracy_assessed") is True and record.get("analytical_passed") is not True:
        return False
    if not all(record.get("metrics", {}).get(key) is True for key in _STRICT_FLAGS):
        return False
    if record.get("warmup", {}).get("success") is not True:
        return False
    timing = record.get("timing", {})
    times = timing.get("samples_s", [])
    expected = timing.get("expected_samples")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        return False
    if len(times) != expected or any(_number(value) is None or value <= 0 for value in times):
        return False
    samples = record.get("samples", [])
    if len(samples) != expected or not all(_sample_valid(sample) for sample in samples):
        return False
    mean = _number(timing.get("mean_s"))
    return mean is not None and mean > 0


def _configuration(record):
    return tuple(record.get(key) for key in ("case_id", "reconstruction", "cells", "cap"))


def matched_pairs(campaign):
    """Return only finest-grid pairs with complete success and matched inputs."""
    rows = campaign.get("results", [])
    finest = defaultdict(int)
    for row in rows:
        cells = row.get("cells")
        if isinstance(cells, int):
            finest[row.get("case_id")] = max(finest[row.get("case_id")], cells)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("cells") == finest[row.get("case_id")]:
            grouped[_configuration(row)][row.get("backend")].append(row)
    pairs = []
    unchanged = (campaign.get("source_unchanged") is True
                 and campaign.get("harness_unchanged") is True
                 and timing_environment_status(campaign) == "clean")
    for key, group in sorted(grouped.items()):
        if not unchanged or len(group["current"]) != 1 or len(group["legacy"]) != 1:
            continue
        current, legacy = group["current"][0], group["legacy"][0]
        if not timing_valid(current) or not timing_valid(legacy):
            continue
        identities = ("case_sha256", "numerical_profile_sha256", "machine_sha256", "source_sha256")
        if not all(current.get(name) and current.get(name) == legacy.get(name) for name in identities):
            continue
        if current.get("settings") != legacy.get("settings"):
            continue
        ratio = current["timing"]["mean_s"] / legacy["timing"]["mean_s"]
        pairs.append({"configuration": key, "current": current, "legacy": legacy, "ratio": ratio})
    return pairs


def validity_counts(campaign):
    rows = campaign.get("results", [])
    environment = timing_environment_status(campaign)
    provenance_clean = campaign.get("source_unchanged") is True and campaign.get("harness_unchanged") is True
    return {
        "configurations": len(rows),
        "by_status": dict(sorted(Counter(row.get("status", "unclassified") for row in rows).items())),
        "root_converged": sum(row.get("success") is True for row in rows),
        "branch_passed": sum(row.get("branch_passed") is True for row in rows),
        "accuracy_assessed": sum(row.get("accuracy_assessed") is True for row in rows),
        "accuracy_passed_when_assessed": sum(row.get("accuracy_assessed") is True and row.get("analytical_passed") is True for row in rows),
        "timing_eligible": sum(timing_valid(row) for row in rows) if environment == "clean" and provenance_clean else 0,
        "timing_environment_status": environment,
        "measured_samples": sum(len(row.get("timing", {}).get("samples_s", [])) for row in rows),
        "failed_measured_samples": sum(sample.get("success") is not True for row in rows for sample in row.get("samples", [])),
        "finest_valid_native_current_pairs": len(matched_pairs(campaign)),
    }


def build_baseline(campaign, *, source_summary):
    """Keep summary-compatible identities and measurements without full fields."""
    if campaign.get("completed") is not True:
        raise ValueError("Only a completed campaign can establish an immutable baseline")
    if timing_environment_status(campaign) != "clean":
        raise ValueError("Only an explicitly clean timing environment can establish an immutable baseline; missing status is unverified")
    if any(timing_environment_status(row) != "clean" for row in campaign.get("results", [])):
        raise ValueError("Every configuration must retain an explicitly clean timing block environment before baseline export")
    if campaign.get("source_unchanged") is not True or campaign.get("harness_unchanged") is not True:
        raise ValueError("Source or numerical harness changed during the campaign; baseline export refused")
    provenance = dict(campaign.get("provenance", {}))
    for name in ("source", "harness"):
        provenance[name] = _pick(provenance.get(name, {}), ("sha256",))
        if not provenance[name].get("sha256"):
            raise ValueError(f"Missing {name} aggregate identity")
    if not provenance.get("machine_sha256") or not provenance.get("legacy_manifest_sha256"):
        raise ValueError("Machine and frozen legacy manifest identities are required")
    rows = []
    for row in campaign.get("results", []):
        compact = _pick(row, _RECORD_KEYS)
        compact["samples"] = [_pick(sample, _SAMPLE_KEYS) for sample in row.get("samples", [])]
        compact["failure_diagnostic_count"] = len(row.get("failure_diagnostics", []))
        rows.append(compact)
    result = _pick(campaign, (
        "schema_version", "campaign_id", "created_at", "completed", "protocol", "protocol_sha256",
        "elapsed_s", "source_unchanged", "harness_unchanged", "timing_environment", "counts",
    ))
    result.update(
        baseline_format_version=1,
        kind="immutable_compact_benchmark_baseline",
        source_summary=source_summary,
        provenance=provenance,
        cases=[_pick(case, ("id", "label", "model", "description", "source", "source_sha256", "known_limit", "definition"))
               for case in campaign.get("cases", [])],
        results=rows,
        validity_counts=validity_counts(campaign),
        omissions="Full numerical fields, analytical curves, histories, failed-sample snapshots and per-file source maps remain in the original campaign evidence.",
    )
    return result


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _sci(value):
    value = _number(value)
    return "—" if value is None else f"{value:.2e}"


def _max_error(row, collection, fields=None):
    if not row:
        return None
    errors = row.get(collection, {})
    selected = errors.values() if fields is None else (errors.get(name, {}) for name in fields)
    values = [_number(error.get("normalized_l1")) for error in selected if isinstance(error, dict)]
    return max((value for value in values if value is not None), default=None)


def _time_text(row):
    if not row:
        return "missing"
    timing = row.get("timing", {})
    mean, lower, upper = (_number(timing.get(key)) for key in ("mean_s", "min_s", "max_s"))
    count = len(timing.get("samples_s", []))
    if mean is None:
        return f"— (n={count})"
    interval = f"{lower:.3f}–{upper:.3f}" if lower is not None and upper is not None else "—"
    return f"{mean:.3f} [{interval}] (n={count})"


def _flag(value):
    return "pass" if value is True else "FAIL" if value is False else "unknown"


def _status_text(row):
    if not row:
        return "missing"
    accuracy = _flag(row.get("analytical_passed")) if row.get("accuracy_assessed") is True else "unassessed"
    return f"{_escape(row.get('status', 'unclassified'))}; R {_flag(row.get('success'))}, A {accuracy}, B {_flag(row.get('branch_passed'))}"


def _link(path, directory):
    return Path(os.path.relpath(Path(path).resolve(), Path(directory).resolve())).as_posix().replace(" ", "%20")


def render_markdown(campaign, *, summary_path, baseline_path, markdown_path):
    """Generate the human-readable campaign assessment from saved evidence."""
    protocol = campaign.get("protocol", {})
    provenance = campaign.get("provenance", {})
    rows = campaign.get("results", [])
    counts = validity_counts(campaign)
    pairs = matched_pairs(campaign)
    pair_map = {tuple(pair["configuration"]): pair for pair in pairs}
    case_meta = {case["id"]: case for case in campaign.get("cases", [])}
    case_ids = protocol.get("cases", list(case_meta))
    if not isinstance(case_ids, list):
        case_ids = list(case_meta)
    reconstructions = protocol.get("reconstructions", sorted({row.get("reconstruction") for row in rows}))
    caps = protocol.get("newton_caps", sorted({row.get("cap") for row in rows if row.get("cap") is not None}))
    grids = protocol.get("grids", [])
    finest = max(grids, default=max((row.get("cells", 0) for row in rows), default=0))
    indexed = defaultdict(list)
    for row in rows:
        indexed[(*_configuration(row), row.get("backend"))].append(row)

    def unique(case_id, reconstruction, cap, backend):
        values = indexed[(case_id, reconstruction, finest, cap, backend)]
        return values[0] if len(values) == 1 else None

    directory = Path(markdown_path).parent
    lines = [
        "# Steady verification and timing benchmark",
        "",
        f"Campaign `{_escape(campaign.get('campaign_id', 'unknown'))}`, recorded {_escape(campaign.get('created_at', 'unknown'))}. "
        f"This report uses saved results for {len(case_ids)} physical cases; it runs no simulations.",
        "",
        f"[Interactive report](REPORT.html) · [Complete campaign evidence]({_link(summary_path, directory)}) · "
        f"[Immutable compact baseline]({_link(baseline_path, directory)})",
        "",
        f"Recorded {counts['configurations']} configurations and {counts['measured_samples']} measured complete solves. "
        f"{counts['root_converged']} configurations converged to the strict spatial/BC/global criteria; "
        f"{counts['accuracy_passed_when_assessed']}/{counts['accuracy_assessed']} passed assessed analytical accuracy, "
        f"{counts['branch_passed']} passed the physical branch audit, and {counts['timing_eligible']} qualify for timing comparison.",
        "",
        f"Timing environment: **{counts['timing_environment_status']}**. "
        "Comparative timing and immutable baseline export require an explicit clean status; a missing status is unverified.",
        "",
        "| Recorded outcome | Configurations |",
        "|---|---:|",
    ]
    lines.extend(f"| {_escape(status)} | {number} |" for status, number in counts["by_status"].items())
    lines += [
        "",
        "## Controlled protocol",
        "",
        f"Meshes: {', '.join(str(grid) for grid in grids)} cells. Reconstruction: {', '.join(_escape(recon) for recon in reconstructions)}. "
        f"Newton caps: {', '.join(str(cap) for cap in caps)}. Finest-grid repeats: {protocol.get('repeats', 'unknown')}; "
        f"refinement repeats: {protocol.get('refinement_repeats', 'unknown')}. One excluded complete warmup precedes each configuration. "
        "A failed warmup skips timed repetitions and remains a recorded failure.",
        "",
        _escape(protocol.get("timing", "Timing boundaries are recorded in the campaign protocol.")) + ".",
        "",
        _escape(protocol.get("initialization", "Initialization is recorded in the campaign protocol.")) + ".",
        "",
        "The current backend calls the production public steady driver with reusable operator/sampler bindings. "
        "The frozen native kernels use a common strict outer convergence controller, rather than their historical update-only termination. "
        "Matching CFL/Newton settings does not require the two discretizations to follow identical Newton or pseudo-time trajectories. "
        "Native nozzle nodal coordinates and boundary elimination differ from the current finite-volume cells and coupled traces.",
        "",
        "```json",
        json.dumps({key: protocol.get(key) for key in ("scheme", "steady", "residual")}, indent=2, allow_nan=False),
        "```",
        "",
        "## Finest-grid timings",
        "",
        f"All requested cases are listed at {finest} cells. Times are mean [minimum–maximum] seconds, with measured sample count. "
        "R = strict numerical convergence, A = assessed analytical accuracy, B = intended physical branch. "
        "Unsuccessful or wrong-branch results retain elapsed time but receive no comparative ratio.",
        "",
    ]
    for cap in caps:
        if len(caps) > 1:
            lines += [f"### Newton cap {cap}", ""]
        lines += [
            "| Physical case | Reconstruction | Old native seconds | Current seconds | Native R/A/B | Current R/A/B | Current/native |",
            "|---|---|---:|---:|---|---|---:|",
        ]
        for case_id in case_ids:
            for reconstruction in reconstructions:
                native = unique(case_id, reconstruction, cap, "legacy")
                current = unique(case_id, reconstruction, cap, "current")
                pair = pair_map.get((case_id, reconstruction, finest, cap))
                ratio = f"{pair['ratio']:.3f}×" if pair else "—"
                lines.append(f"| `{_escape(case_id)}` | {_escape(reconstruction)} | {_time_text(native)} | {_time_text(current)} | {_status_text(native)} | {_status_text(current)} | {ratio} |")
        lines.append("")
    if pairs:
        native_sum = sum(pair["legacy"]["timing"]["mean_s"] for pair in pairs)
        current_sum = sum(pair["current"]["timing"]["mean_s"] for pair in pairs)
        geometric_ratio = math.exp(sum(math.log(pair["ratio"]) for pair in pairs) / len(pairs))
        lines += [
            f"Across the {len(pairs)} fully valid, matched finest-grid pairs only, the sums of per-configuration mean times are "
            f"{native_sum:.3f} s native and {current_sum:.3f} s current (current/native {current_sum/native_sum:.3f}×). "
            f"The geometric mean of the pairwise current/native ratios is {geometric_ratio:.3f}×. "
            "These aggregates exclude every failed, inaccurate, unassessed-identity or wrong-branch pair; they are not an all-case performance claim.",
            "",
        ]
    else:
        lines += ["No finest-grid pair met all success, accuracy, branch, sample-count, identity and clean-environment checks; no aggregate timing ratio is reported.", ""]
    lines += [
        "## Analytical error overview",
        "",
        "Entries below are old native / current. Field error is the maximum fixed-scale normalized $L_1$ error over "
        r"$p,\rho,u$ (or $v_m$), $M,T$; conservative error is the maximum fixed-scale normalized conservative $L_1$ error. "
        "The annular $C_p$ error uses length weighting. Shock error is in finest-grid cell widths. "
        "Current cells and native annular cells are compared with an independent conservative finite-volume reference projection; "
        "the native nodal nozzle uses analytical point samples at its own coordinates. These are explicitly different error representations.",
        "",
        "Annular smooth-case acceptance retains the historical conservative fixed-scale $L_1<10^{-3}$ and mean "
        r"$|\Delta C_p|\leq0.01$ criteria. Annular shock cases retain the historical shock-position limit of two cells; "
        "all field errors remain visible. Nozzle acceptance uses the versioned case's normalized-field and shock targets. "
        "Coarser-grid errors are reported in the interactive report without claiming the finest-grid acceptance threshold was assessed there.",
        "",
        "| Physical case | Reconstruction / cap | Field $L_1$ native / current | Conservative $L_1$ native / current | $C_p$ mean absolute native / current | Shock cells native / current |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case_id in case_ids:
        for reconstruction in reconstructions:
            for cap in caps:
                native = unique(case_id, reconstruction, cap, "legacy")
                current = unique(case_id, reconstruction, cap, "current")
                values = []
                for collection, fields in (("errors", ("p", "rho", "velocity", "M", "T")), ("conservative_errors", None)):
                    values.append(" / ".join(_sci(_max_error(row, collection, fields)) for row in (native, current)))
                values.append(" / ".join(_sci((row or {}).get("errors", {}).get("Cp", {}).get("l1")) for row in (native, current)))
                values.append(" / ".join(_sci((row or {}).get("shock", {}).get("error_cells")) for row in (native, current)))
                lines.append(f"| `{_escape(case_id)}` | {_escape(reconstruction)} / {cap} | " + " | ".join(values) + " |")
    lines += ["", "## Failures and interpretation limits", ""]
    invalid = [row for row in rows if not timing_valid(row)]
    if invalid:
        lines += ["Every unsuccessful configuration is retained in the campaign. The following entries are excluded from comparative timing:", ""]
        for row in invalid:
            explanation = row.get("assessment_error") or row.get("reason") or row.get("status", "unclassified")
            failed_checks = [name for family in ("accuracy_checks", "branch_checks") for name, passed in row.get(family, {}).items() if passed is False]
            suffix = f"; failed checks: {', '.join(failed_checks)}" if failed_checks else ""
            lines.append(f"- `{_escape(row.get('id', _configuration(row)))}`: {_escape(row.get('status', 'unclassified'))}; {_escape(explanation)}{_escape(suffix)}.")
        lines.append("")
    else:
        lines += ["All recorded configurations passed the numerical, assessed-accuracy, branch and repeat-completeness checks.", ""]
    for case_id in case_ids:
        limit = case_meta.get(case_id, {}).get("known_limit")
        if limit:
            lines.append(f"- `{_escape(case_id)}`: {_escape(limit)}")
    lines += [
        "",
        "Strict residual convergence establishes a discrete root, not analytical accuracy or uniqueness. "
        "Small residuals can coexist with sonic-limit accuracy or branch-selection discrepancies. "
        "Independent reference errors, branch audits and shock positions remain separate acceptance signals. "
        "The HTML report contains each representative trajectory, CFL history, field comparison, sample distribution and grid-refinement study.",
        "",
        "## Reproducibility and future comparisons",
        "",
        f"- Git head: `{provenance.get('git_head', 'unknown')}`.",
        f"- Production source SHA-256: `{provenance.get('source', {}).get('sha256', 'unknown')}`; unchanged: `{campaign.get('source_unchanged')}`.",
        f"- Numerical harness SHA-256: `{provenance.get('harness', {}).get('sha256', 'unknown')}`; unchanged: `{campaign.get('harness_unchanged')}`.",
        f"- Protocol SHA-256: `{campaign.get('protocol_sha256', 'unknown')}`.",
        f"- Machine/environment SHA-256: `{provenance.get('machine_sha256', 'unknown')}`.",
        f"- Frozen pre-integration manifest SHA-256: `{provenance.get('legacy_manifest_sha256', 'unknown')}`.",
        "",
        "The compact baseline preserves physical case hashes, effective numerical settings, machine/package/thread identity, "
        "all timing samples and summary errors/flags. Full fields, histories and failed-sample snapshots remain in the complete evidence. "
        "The exporter refuses to overwrite a different baseline; choose a new versioned filename for a new campaign.",
        "",
        "Regenerate the HTML after a future campaign with the saved baseline:",
        "",
        "```powershell",
        "poetry run python -B -m development.roberto.benchmarking.scripts.report `",
        "  --input development/roberto/benchmarking/outputs/<new_campaign>/summary.json `",
        f"  --baseline development/roberto/benchmarking/{Path(os.path.relpath(baseline_path, STUDY)).as_posix()}",
        "```",
        "",
        "A regression comparison is valid only when physical case definitions, mesh, reconstruction, solver settings, "
        "residual policy, precision and machine environment match and both configurations pass. "
        "Source hashes identify the implementations being compared; they are expected to change when testing a source revision.",
        "",
    ]
    return "\n".join(lines)


def write_immutable(path, value):
    """Create a baseline once; byte-identical regeneration is idempotent."""
    path = Path(path)
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
        return True
    except FileExistsError:
        if path.read_bytes() != raw:
            raise FileExistsError(f"Immutable baseline already exists with different content: {path}; choose a new filename") from None
        return False


def _study_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (STUDY / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Saved campaign summary; relative to benchmarking/")
    parser.add_argument("--output", type=Path, default=Path("baselines/initial_2026-10-31.json"), help="New immutable baseline JSON; relative to benchmarking/")
    parser.add_argument("--markdown", type=Path, default=Path("BENCHMARK_REPORT.md"), help="Markdown assessment; relative to benchmarking/")
    args = parser.parse_args(argv)
    input_path, output_path, markdown_path = map(_study_path, (args.input, args.output, args.markdown))
    for path in (output_path, markdown_path):
        if not path.is_relative_to(STUDY):
            parser.error("Baseline and Markdown outputs must remain beneath benchmarking/")
    if output_path.suffix.lower() != ".json" or markdown_path.suffix.lower() != ".md":
        parser.error("Use .json for the immutable baseline and .md for the Markdown assessment")
    if len({input_path, output_path, markdown_path}) != 3:
        parser.error("Input, baseline output and Markdown output must be distinct files")
    campaign, evidence = read_campaign(input_path)
    evidence["path"] = input_path.relative_to(STUDY).as_posix() if input_path.is_relative_to(STUDY) else input_path.name
    baseline = build_baseline(campaign, source_summary=evidence)
    markdown = render_markdown(campaign, summary_path=input_path, baseline_path=output_path, markdown_path=markdown_path)
    created = write_immutable(output_path, baseline)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"{'Created' if created else 'Verified unchanged'} immutable baseline: {output_path}")
    print(f"Wrote saved-data assessment: {markdown_path}")
    print(json.dumps(baseline["validity_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
