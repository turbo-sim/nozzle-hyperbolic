"""Summarize paper scalar and mesh comparisons from immutable saved results."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def latest(records, case_id, flux, cells):
    rows = [
        (p, r)
        for p, r in records
        if r.get("case_id") == case_id
        and r.get("flux") == flux
        and r.get("cells") == cells
    ]
    return max(rows, key=lambda item: item[0].stat().st_mtime) if rows else None


def observables(path, row):
    if not path.with_name("solution.npz").exists():
        return dict(verified=False, reason=row.get("error", row.get("reason")))
    with np.load(path.with_name("solution.npz")) as saved:
        x, area, pressure = saved["x_m"], saved["area_m2"], saved["p"]
        throat_x = x[int(np.argmin(area))]
        mdot = (
            row.get("verification", {}).get("mass_flow", {}).get("numerical_inlet_kg_s")
        )
        endpoint_pressure = float(saved["state"][-1, 0])
        result = dict(
            verified=bool(row.get("verification", {}).get("verified")),
            absolute_targets_pass=row.get("verification", {}).get(
                "absolute_targets_pass"
            ),
            intended_regime_pass=row.get("verification", {}).get(
                "intended_regime_pass"
            ),
            interior_mach_max=float(np.max(saved["M"])),
            p_throat_Pa=float(np.interp(throat_x, x, pressure)),
            p_out_Pa=endpoint_pressure,
            p_min_Pa=float(pressure.min()),
            mass_flow_kg_s=mdot,
            source_result=str(path.relative_to(ROOT)),
            throat_sampling="nearest physical-cell area minimum; no exact sonic interpolation inferred",
        )
        if "s" in saved and row.get("reservoir"):
            # Boundary trace entropy is evaluated by reservoir/inlet constraints;
            # reported exit estimate uses the final physical cell's EOS entropy.
            result["entropy_rise_J_kgK"] = float(
                saved["s"][-1] - row["reservoir"]["s0_J_kgK"]
            )
        down = np.flatnonzero(x[:-1] > throat_x)
        if len(down):
            shock = down[int(np.argmax(np.diff(pressure)[down]))]
            result["shock_location_gradient_m"] = float(0.5 * (x[shock] + x[shock + 1]))
        return result


def main():
    definitions = read(ROOT / "case_definitions.json")
    records = [(p, read(p)) for p in (ROOT / "outputs").glob("**/result.json")]
    summary = dict(
        table1=[],
        table3=[],
        digitized_pressure_comparisons=[],
        mesh_studies=[],
        interpretation=[
            "Numerical verification and agreement with published data are separate judgments.",
            "Table 1 shock entropy is sampled at the last physical cell relative to reservoir entropy; numerical boundary mass flux defines mass flow.",
            "Table 3 labels kg/s in the paper, but archived CFD values use half-channel flow per unit depth; both dimensional conventions are retained.",
            "Mesh percentages are absolute differences from the current finest verified state, not errors against experiment or proof of mesh independence.",
            "Single compile-inclusive solve observations do not establish a performance speedup.",
        ],
    )
    cases = {c["id"]: c for c in definitions["cases"]}
    for index, case_id in enumerate(("G_shock_25", "G_shock_50", "G_shock_75")):
        case = cases[case_id]
        exact = definitions["paper_tables"]["table1"]["entropy_exact_J_kgK"][index]
        for flux in case["fluxes"]:
            selected = latest(records, case_id, flux, case["cells"])
            if not selected:
                continue
            path, row = selected
            values = observables(path, row)
            entropy = values.get("entropy_rise_J_kgK")
            reference = next(
                (
                    s
                    for s in row.get("comparison", {}).get("series", [])
                    if s.get("kind") == "analytical"
                ),
                None,
            )
            contour_entropy = None
            if reference and reference.get("T") and reference.get("p_Pa"):
                contour_entropy = float(
                    1004.5 * np.log(reference["T"][-1] / case["T0_K"])
                    - 287 * np.log(reference["p_Pa"][-1] / case["p0_Pa"])
                )
            summary["table1"].append(
                dict(
                    case_id=case_id,
                    flux=flux,
                    **values,
                    paper_entropy_J_kgK=exact,
                    selected_contour_exact_entropy_J_kgK=contour_entropy,
                    entropy_difference_from_selected_contour_percent=(
                        None
                        if entropy is None or not contour_entropy
                        else 100 * (entropy / contour_entropy - 1)
                    ),
                    entropy_difference_from_paper_percent=(
                        None if entropy is None else 100 * (entropy / exact - 1)
                    ),
                    target_shock_x_m=row["case"].get("target_shock_x_m"),
                )
            )
    for case_id in ("G_adapted",):
        case = cases[case_id]
        for flux in case["fluxes"]:
            selected = latest(records, case_id, flux, case["cells"])
            if selected:
                path, row = selected
                values = observables(path, row)
                summary["table1"].append(
                    dict(
                        case_id=case_id,
                        flux=flux,
                        **values,
                        paper_exact_mass_flow_kg_s=22.48,
                        mass_flow_difference_from_paper_percent=(
                            100 * (values["mass_flow_kg_s"] / 22.48 - 1)
                            if values.get("mass_flow_kg_s") is not None
                            else None
                        ),
                    )
                )
    for case_id in ("L1", "L5"):
        case = cases[case_id]
        width = definitions["geometries"][case["geometry"]]["width_m"]
        paper = definitions["paper_tables"]["table3"][case_id]
        for flux in case["fluxes"]:
            selected = latest(records, case_id, flux, case["cells"])
            if selected:
                path, row = selected
                values = observables(path, row)
                full = values.get("mass_flow_kg_s")
                per_depth = None if full is None else full / (2 * width)
                historical = paper["paper_HEM_" + flux]
                summary["table3"].append(
                    dict(
                        case_id=case_id,
                        flux=flux,
                        **values,
                        full_channel_mass_flow_kg_s=full,
                        half_channel_mass_flow_kg_s_m=per_depth,
                        paper_HEM_reported=historical,
                        paper_CFD_mixture_reported=paper["CFD_mixture"],
                        difference_from_paper_HEM_percent=(
                            None
                            if per_depth is None
                            else 100 * (per_depth / historical - 1)
                        ),
                        difference_from_paper_CFD_percent=(
                            None
                            if per_depth is None
                            else 100 * (per_depth / paper["CFD_mixture"] - 1)
                        ),
                    )
                )
    for study in definitions["mesh_studies"]:
        points = []
        for cells in study["cells"]:
            selected = latest(records, study["case_id"], study["flux"], cells)
            if selected:
                path, row = selected
                points.append(dict(cells=cells, **observables(path, row)))
        baseline = next(
            (p for p in points if p["cells"] == max(study["cells"]) and p["verified"]),
            None,
        )
        for point in points:
            for key in ("p_throat_Pa", "p_out_Pa", "p_min_Pa", "mass_flow_kg_s"):
                value, reference = point.get(key), (
                    None if baseline is None else baseline.get(key)
                )
                point[key + "_difference_percent"] = (
                    None
                    if value is None or reference in (None, 0)
                    else 100 * abs(value / reference - 1)
                )
        summary["mesh_studies"].append(
            dict(
                case_id=study["case_id"],
                flux=study["flux"],
                points=points,
                requested_cells=study["cells"],
                finest_verified=baseline is not None,
                note=study.get("notes"),
                observables=study["observables"],
            )
        )
    digitized_path = ROOT / "outputs/paper_digitization.json"
    if digitized_path.exists():
        digitized = read(digitized_path)
        if digitized.get("validated"):
            for case_id, series in digitized["cases"].items():
                case = cases[case_id]
                for flux in case["fluxes"]:
                    selected = latest(records, case_id, flux, case["cells"])
                    if not selected:
                        continue
                    path, row = selected
                    if not path.with_name("solution.npz").exists():
                        continue
                    with np.load(path.with_name("solution.npz")) as saved:
                        x, p = saved["x_m"], saved["p"]
                        for reference in series:
                            rx, rp = np.asarray(reference["x_m"]), np.asarray(
                                reference["p_Pa"]
                            )
                            valid = (
                                np.isfinite(rx)
                                & np.isfinite(rp)
                                & (rx >= x[0])
                                & (rx <= x[-1])
                            )
                            if not valid.any():
                                continue
                            difference = np.interp(rx[valid], x, p) - rp[valid]
                            rms = float(np.sqrt(np.mean(difference**2)))
                            summary["digitized_pressure_comparisons"].append(
                                dict(
                                    case_id=case_id,
                                    flux=flux,
                                    reference_label=reference["label"],
                                    reference_kind=reference["kind"],
                                    samples=int(valid.sum()),
                                    pressure_rmse_bar=rms / 1e5,
                                    pressure_max_error_bar=float(
                                        np.max(np.abs(difference))
                                    )
                                    / 1e5,
                                    pressure_rmse_percent_p0=100 * rms / case["p0_Pa"],
                                    digitization_pressure_uncertainty_bar_max=float(
                                        np.max(reference["pressure_uncertainty_Pa"])
                                    )
                                    / 1e5,
                                    x_uncertainty_mm_max=float(
                                        np.max(reference["x_uncertainty_m"])
                                    )
                                    * 1e3,
                                    current_solution_verified=bool(
                                        row.get("verification", {}).get("verified")
                                    ),
                                    source_result=str(path.relative_to(ROOT)),
                                    note="Only visible digitized samples scored; interpolate the dense current field at those points, never fill gaps in historical curves. Graph-reading bounds are not statistical/experimental uncertainties.",
                                )
                            )
    primary = []
    for case in definitions["cases"]:
        for flux in case["fluxes"]:
            selected = latest(records, case["id"], flux, case["cells"])
            if selected:
                path, row = selected
                primary.append(
                    dict(
                        case_id=case["id"],
                        flux=flux,
                        cells=case["cells"],
                        verified=bool(row.get("verification", {}).get("verified")),
                        reason=row.get("error", row.get("reason")),
                        source_result=str(path.relative_to(ROOT)),
                        comparison=row.get("comparison", {}).get("errors", []),
                    )
                )
            else:
                primary.append(
                    dict(
                        case_id=case["id"],
                        flux=flux,
                        cells=case["cells"],
                        verified=False,
                        reason="not_attempted",
                    )
                )
    summary["primary_coverage"] = primary
    summary["counts"] = dict(
        primary_requested=len(primary),
        primary_attempted=sum(p["reason"] != "not_attempted" for p in primary),
        primary_verified=sum(p["verified"] for p in primary),
        retained_attempts=len(records),
        mesh_requested=sum(len(s["cells"]) for s in definitions["mesh_studies"]),
        mesh_attempted=sum(len(s["points"]) for s in summary["mesh_studies"]),
        mesh_verified=sum(
            p["verified"] for s in summary["mesh_studies"] for p in s["points"]
        ),
    )
    target = ROOT / "outputs/summary.json"
    target.write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    counts = summary["counts"]
    lines = [
        "# Turbo Expo 2025 reproduction results",
        "",
        "Saved results from the current `roberto-branch` HEM and implicit backward Euler. "
        "See [REPORT.html](REPORT.html) for fields, original-equation checks, every retained attempt, "
        "reference comparisons and the embedded evidence archive.",
        "",
        f"Primary configurations: **{counts['primary_verified']}/{counts['primary_requested']} verified**, "
        f"{counts['primary_attempted']} attempted. Appendix mesh points: "
        f"**{counts['mesh_verified']}/{counts['mesh_requested']} verified**, {counts['mesh_attempted']} attempted. "
        f"The campaign retains {counts['retained_attempts']} attempts, including pilots and superseded settings.",
        "",
        "Verification requires the original spatial equations, boundary equations, global flux balances, "
        "EOS admissibility and physical branch. It is separate from published-data agreement.",
        "",
        "| Case | Paper cells | Method | Latest numerical outcome |",
        "|---|---:|---|---|",
    ]
    for item in primary:
        outcome = (
            "Verified steady root"
            if item["verified"]
            else str(item["reason"]).replace("|", "/")
        )
        lines.append(
            f"| {item['case_id']} | {item['cells']} | {item['flux']} | {outcome} |"
        )
    branch_failures = [
        (study["case_id"], point)
        for study in summary["mesh_studies"]
        for point in study["points"]
        if point.get("absolute_targets_pass")
        and point.get("intended_regime_pass") is False
    ]
    if branch_failures:
        lines += [
            "",
            "## Coarse-mesh flow regimes",
            "",
            "These meshes satisfy the original steady equations but do not recover the requested shock regime. "
            "They remain recorded as converged discrete roots with failed physical-branch verification. "
            "They are excluded from a claim that every mesh reproduces the paper's regime.",
            "",
            "| Case | Cells | Maximum interior Mach | Original equation targets | Intended shock regime |",
            "|---|---:|---:|---|---|",
        ]
        for case_id, point in branch_failures:
            lines.append(
                f"| {case_id} | {point['cells']} | {point['interior_mach_max']:.6g} | Pass | Absent |"
            )
    lines += [
        "",
        "## Published scalars",
        "",
        "The selected ideal contour matches the published critical mass flow and outlet/throat area ratio. "
        "Its exact shock-entropy values differ from Table 1; numerical error against the selected contour "
        "and disagreement with the historical paper scalar are therefore reported separately.",
        "",
        "| Case | Method | Current entropy rise [J/(kg K)] | Selected-contour exact | Paper | Difference from contour [%] |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return (
            f"{value:.6g}"
            if value is not None and np.isfinite(value)
            else "Unavailable"
        )

    for item in summary["table1"]:
        if "paper_entropy_J_kgK" in item:
            lines.append(
                f"| {item['case_id']} | {item['flux']} | {fmt(item.get('entropy_rise_J_kgK'))} | {fmt(item.get('selected_contour_exact_entropy_J_kgK'))} | {fmt(item['paper_entropy_J_kgK'])} | {fmt(item.get('entropy_difference_from_selected_contour_percent'))} |"
            )
    lines += [
        "",
        "Lettieri's historical Table 3 values use half-channel flow per unit depth, despite "
        "the paper's kg/s label. Current full-channel flow and the comparable per-depth quantity are both shown.",
        "",
        "| Case | Method | Full channel [kg/s] | Half channel per depth [kg/(s m)] | Paper HEM per depth | Difference [%] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summary["table3"]:
        lines.append(
            f"| {item['case_id']} | {item['flux']} | {fmt(item.get('full_channel_mass_flow_kg_s'))} | {fmt(item.get('half_channel_mass_flow_kg_s_m'))} | {fmt(item['paper_HEM_reported'])} | {fmt(item.get('difference_from_paper_HEM_percent'))} |"
        )
    if summary["digitized_pressure_comparisons"]:
        lines += [
            "",
            "## Visible Petruccelli paper curves",
            "",
            "Figure 9's embedded raster plots supply independently reviewed, partial pressure-curve samples. "
            "They are approximate graph digitizations, with stroke/calibration uncertainty and visual overlays "
            "in the HTML evidence. Upstream overlaps, sharp jump interiors and ambiguous/merged tails are omitted. "
            "Only observed points are scored; these metrics do not describe complete historical profiles.",
            "",
            "| Case | Current method | Visible reference | Samples | Pressure RMSE [bar] | RMSE / reservoir pressure [%] |",
            "|---|---|---|---:|---:|---:|",
        ]
        for item in summary["digitized_pressure_comparisons"]:
            lines.append(
                f"| {item['case_id']} | {item['flux']} | {item['reference_label']} | {item['samples']} | {fmt(item['pressure_rmse_bar'])} | {fmt(item['pressure_rmse_percent_p0'])} |"
            )
    lines += [
        "",
        "Historical solver fields were not committed, and P1–P4 reference CFD arrays were not recovered. "
        "Spinelli's coordinate mapping remains uncertain. These gaps preclude a complete historical "
        "field-regression claim. All recoverable measured/CFD reference series and current discrepancies "
        "are shown in the HTML; no missing curve or uncertainty is fabricated.",
        "",
        "Current source epochs and dirty implementation snapshots are retained with each attempt. "
        "The stable polynomial-volume integration correction was checked independently: the original "
        "spatial residual was bitwise unchanged and the corrected-volume residual norms still passed. "
        "Reported solve times are single compilation-inclusive observations; they establish no speedup.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {target}; {len(records)} retained attempts")


if __name__ == "__main__":
    main()
