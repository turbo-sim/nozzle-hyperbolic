"""Read-only, fitted one-sided normal-shock audits of saved annular solutions.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.audit_shocks
  poetry run python -m development.roberto.euler_annular.scripts.audit_shocks --folder development/roberto/euler_annular/outputs/work_controller

No solver is constructed or advanced. Linear fits of W=(p,v_m,ell,h) outside
the captured shock are extrapolated to ONE estimated station and radius before
EOS evaluation. Finite fitting/discretization errors are reported; planar jump
agreement to roundoff is neither expected nor used as an acceptance criterion.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path

import jaxprop as jxp
import numpy as np

from development.roberto.euler_annular.scripts.verify_annular import (
    RECONSTRUCTIONS, STUDY, fingerprints, make_case,
)


VARIABLES = ("p", "v_m", "ell", "h")


def _safe_json(value):
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fit_side(m, W, radii, station, requested_indices):
    """Linear W fit in a scaled station-centred coordinate, without clipping."""
    indices = requested_indices[(requested_indices >= 0) & (requested_indices < len(m))]
    result = dict(requested_indices=requested_indices, indices=indices,
                  m=m[indices], radius_at_cells=radii[indices], raw_W=W[indices],
                  variable_order=VARIABLES)
    if len(indices) != len(requested_indices):
        result["error"] = "The fixed stencil extends beyond the physical cells"
        return result
    if len(indices) < 3:
        result["error"] = "A fitted diagnostic requires at least three cells per side"
        return result
    span = float(np.ptp(m[indices]))
    if not np.isfinite(span) or span <= 0 or not np.isfinite(W[indices]).all():
        result["error"] = "Nonfinite window states or nonpositive physical fitting span"
        return result
    coordinate = (m[indices]-station)/span
    matrix = np.column_stack((np.ones(len(indices)), coordinate))
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, W[indices], rcond=None)
    if rank != 2:
        result["error"] = "The one-sided fitting matrix is rank deficient"
        return result
    residual = matrix@coefficients-W[indices]
    result.update(
        fitted_W=coefficients[0], coefficients=coefficients,
        coefficient_coordinate="t=(m-estimated_station)/window_span; W=coefficient[0]+t*coefficient[1]",
        span=span, singular_values=singular_values,
        rms_W=np.sqrt(np.mean(residual**2, axis=0)),
        max_abs_fit_residual_W=np.max(np.abs(residual), axis=0),
        nearest_sample_distance=float(np.min(np.abs(m[indices]-station))),
        extrapolation_distance_over_span=float(np.min(np.abs(m[indices]-station))/span))
    return result


def hp_state(fluid, W, radius):
    """Physical state at the common station, with no perfect-gas EOS substitution."""
    W = np.asarray(W, dtype=float)
    if not np.isfinite(W).all() or W[0] <= 0 or W[1] <= 0 or radius <= 0:
        raise ValueError("Fitted state requires finite W, positive pressure/meridional velocity/radius")
    state = fluid.get_state(jxp.HmassP_INPUTS, W[3], W[0])
    rho, sound, entropy, temperature = (float(state[key]) for key in ("rho", "a", "s", "T"))
    if not np.isfinite((rho, sound, entropy, temperature)).all() or min(rho, sound, temperature) <= 0:
        raise ValueError("The fitted HP state is outside the physical EOS domain")
    v_theta = W[2]/radius
    return dict(W=W, p=float(W[0]), v_m=float(W[1]), ell=float(W[2]), h=float(W[3]),
                r=float(radius), rho=rho, a=sound, s=entropy, T=temperature,
                v_theta=float(v_theta), M_m=float(W[1]/sound),
                h0=float(W[3]+.5*(W[1]**2+v_theta**2)))


def audit_method(saved, method, case, fluid, inner_offset, outer_offset):
    run = saved["methods"][method]
    m = np.asarray(saved["m"], dtype=float)
    radii = np.asarray(saved["r"], dtype=float)
    full = np.asarray(run["state"], dtype=float)
    audit = dict(method=method, root_success=bool(run["success"]),
                 raw_state_including_boundary_traces=full, flags=[])
    if full.shape != (len(m)+2, 4) or len(m) < 3 or not np.isfinite(full).all():
        audit.update(valid_extrapolation=False, flags=["Invalid saved state shape or nonfinite state"])
        return audit
    W = full[1:-1]
    gradient = np.diff(W[:,1])/np.diff(m)
    index = int(np.argmin(gradient))
    station = float((m[index]+m[index+1])/2)
    metrics = case.geometry.evaluate(station)
    radius, area = float(metrics.r), float(metrics.A)
    geometry_r = np.asarray(case.geometry.evaluate(m).r)
    faces = np.asarray(saved["m_faces"], dtype=float)
    if (not np.allclose(geometry_r, radii, rtol=2e-12, atol=2e-13)
            or not np.isclose(faces[-1], case.geometry.length, rtol=2e-12, atol=2e-13)):
        audit["flags"].append("Saved geometry differs from the reconstructed case")
    if gradient[index] >= 0:
        audit["flags"].append("The saved field has no negative meridional-velocity gradient")
    offsets = np.arange(inner_offset, outer_offset+1)
    upstream = fit_side(m, W, radii, station, index-offsets[::-1])
    downstream = fit_side(m, W, radii, station, index+1+offsets)
    local_width = float(faces[index+1]-faces[index])
    audit.update(
        interface_left_cell=index, estimated_station=station,
        strongest_velocity_gradient=float(gradient[index]),
        known_analytical_shock_station=float(saved["shock_m"]),
        location_error_m=station-float(saved["shock_m"]),
        location_error_local_cells=(station-float(saved["shock_m"]))/local_width,
        common_radius=radius, common_area=area,
        upstream_window=upstream, downstream_window=downstream)
    # The exterior fits must not hide extrema inside the excluded shock smear.
    # Compare with the saved conserved-reference projection at those cells,
    # without turning this descriptive local error into a new acceptance rule.
    core = np.arange(max(0, index-inner_offset+1), min(len(m), index+inner_offset+1))
    if len(core):
        p_reference = np.asarray(saved["reference"]["p"])[core]
        vm_reference = np.asarray(saved["reference"]["v_m"])[core]
        rho_numerical = np.asarray(run["fields"]["rho"])[core]
        rho_reference = np.asarray(saved["reference"]["rho"])[core]
        audit["excluded_shock_region"] = dict(
            indices=core, m=m[core], raw_W=W[core],
            reference_W=np.asarray(saved["reference_W"])[core],
            pressure_max=float(np.max(W[core,0])),
            reference_pressure_max=float(np.max(p_reference)),
            pressure_peak_over_reference_window_max=float(np.max(W[core,0])/np.max(p_reference)-1),
            maximum_positive_pressure_error_Pa=float(np.max(W[core,0]-p_reference)),
            density_max=float(np.max(rho_numerical)),
            reference_density_max=float(np.max(rho_reference)),
            density_peak_over_reference_window_max=float(np.max(rho_numerical)/np.max(rho_reference)-1),
            v_m_min=float(np.min(W[core,1])),
            reference_v_m_min=float(np.min(vm_reference)),
            v_m_min_relative_to_reference_window_min=float(np.min(W[core,1])/np.min(vm_reference)-1))
    for label, window in (("upstream", upstream), ("downstream", downstream)):
        if "error" in window:
            audit["flags"].append(f"{label}: {window['error']}")
    if audit["flags"]:
        audit["valid_extrapolation"] = False
        return audit
    try:
        before = hp_state(fluid, upstream["fitted_W"], radius)
        after = hp_state(fluid, downstream["fitted_W"], radius)
    except (ValueError, RuntimeError) as error:
        audit.update(valid_extrapolation=False)
        audit["flags"].append(f"Invalid HP extrapolation: {type(error).__name__}: {error}")
        return audit
    audit.update(valid_extrapolation=True, upstream=before, downstream=after)
    for label, window, expected_supersonic in (("upstream", upstream, True),
                                               ("downstream", downstream, False)):
        sample = fluid.get_state(jxp.HmassP_INPUTS, window["raw_W"][:,3], window["raw_W"][:,0])
        mach = window["raw_W"][:,1]/np.asarray(sample["a"])
        window["M_m_at_native_cells"] = mach
        if not np.isfinite(mach).all() or np.any(mach <= 1 if expected_supersonic else mach >= 1):
            audit["flags"].append(f"{label} window contains the opposite meridional branch; possible shock-smear contamination")
    checks = dict(upstream_meridionally_supersonic=before["M_m"] > 1,
                  downstream_meridionally_subsonic=0 < after["M_m"] < 1,
                  pressure_increases=after["p"] > before["p"],
                  density_increases=after["rho"] > before["rho"],
                  meridional_velocity_decreases=after["v_m"] < before["v_m"],
                  entropy_increases=after["s"] > before["s"])
    audit["physical_checks"] = checks
    audit["flags"].extend(key for key, passed in checks.items() if not passed)
    mass1, mass2 = before["rho"]*before["v_m"], after["rho"]*after["v_m"]
    momentum1 = before["p"]+before["rho"]*before["v_m"]**2
    momentum2 = after["p"]+after["rho"]*after["v_m"]**2
    energy_scale = max(abs(before["h0"]), abs(after["h0"]), before["a"]**2)
    angular_scale = abs(before["ell"]) if before["ell"] != 0 else radius*before["a"]
    audit["planar_jump"] = dict(
        mass_flux_upstream=mass1, mass_flux_downstream=mass2,
        mass_defect=(mass2-mass1)/max(abs(mass1), abs(mass2)),
        normal_momentum_upstream=momentum1, normal_momentum_downstream=momentum2,
        normal_momentum_defect=(momentum2-momentum1)/max(abs(momentum1), abs(momentum2)),
        h0_defect=(after["h0"]-before["h0"])/energy_scale,
        h0_normalization=energy_scale,
        ell_difference=after["ell"]-before["ell"],
        ell_defect=(after["ell"]-before["ell"])/angular_scale,
        ell_normalization=angular_scale,
        v_theta_difference=after["v_theta"]-before["v_theta"],
        v_theta_defect_over_upstream_sound=(after["v_theta"]-before["v_theta"])/before["a"],
        entropy_rise=after["s"]-before["s"],
        entropy_rise_over_R=(after["s"]-before["s"])/case.reference.R)
    if checks["upstream_meridionally_supersonic"]:
        # Build the oracle from FITTED upstream strength, not the exact branch
        # at the known reference location. Normal-shock algebra remains solely
        # in the independent analytical module used by verification fixtures.
        try:
            fitted_reference = replace(
                case.reference, mdot=mass1*area, h0=before["h0"],
                K=before["p"]/before["rho"]**case.reference.gamma,
                ell=before["ell"], default_branch="sup")
            oracle = fitted_reference.normal_shock(station)
        except (ValueError, RuntimeError) as error:
            audit["flags"].append(f"Fitted-upstream normal-shock oracle failed: {type(error).__name__}: {error}")
            return audit
        expected = oracle["downstream"]
        actual_ratios = dict(p=after["p"]/before["p"], rho=after["rho"]/before["rho"],
                             T=after["T"]/before["T"], v_m=after["v_m"]/before["v_m"])
        expected_ratios = dict(p=oracle["pressure_ratio"], rho=oracle["density_ratio"],
                               T=float(expected["T"])/before["T"],
                               v_m=float(expected["v_m"])/before["v_m"])
        audit["normal_shock_reference_from_fitted_upstream"] = dict(
            gamma=case.reference.gamma, upstream_M_m=before["M_m"],
            expected_downstream_W=expected["W"], expected_downstream_M_m=float(expected["M_m"]),
            observed_downstream_M_m=after["M_m"],
            downstream_M_m_relative_defect=(after["M_m"]-float(expected["M_m"]))/float(expected["M_m"]),
            observed_ratios=actual_ratios, expected_ratios=expected_ratios,
            ratio_relative_defects={key: actual_ratios[key]/expected_ratios[key]-1
                                    for key in actual_ratios},
            expected_entropy_rise=oracle["entropy_rise"])
    else:
        audit["flags"].append("No compressive normal-shock oracle for the fitted upstream branch")
    return audit


def markdown_report(report):
    lines = ["# Fitted annular normal-shock diagnostics", "",
             "This read-only audit uses saved numerical states and advances no timestep. "
             "Roots, shock-location targets and jump diagnostics are separate results.", "",
             "The estimated station is the midpoint across the strongest negative cell-centre "
             "meridional-velocity gradient. Linear fits extrapolate $(p,v_m,\\ell,h)$ from "
             "one-sided windows to that same station. Both HP states use its same radius.", "",
             f"Requested offsets from the adjacent upstream/downstream cells are "
             f"{report['settings']['inner_offset']} through {report['settings']['outer_offset']} inclusive. "
             "Actual indices, coordinates, raw states, fit residuals and extrapolation distances "
             "are retained in `shock_audits.json`. A missing fixed stencil is flagged.", "",
             "The normal-shock ratios use the **fitted upstream meridional Mach**, through the "
             "independent perfect-gas analytical module. The known analytical shock location "
             "does not supply the fitted strength. No tight jump tolerance is invented: "
             "finite-window fitting and a smeared numerical shock can produce appreciable "
             "defects, particularly on coarse meshes. Compare each quantity under refinement.", "",
             "The script exit status flags unusable stencils, inadmissible extrapolations or "
             "physical branch errors only. A zero status does not impose an accuracy threshold "
             "on the reported finite-window jump defects. A valid HP fit is not proof that "
             "its stencil lies outside every smeared shock cell: diffuse first-order fluxes "
             "can require wider offsets even when both windows have the expected Mach signs. "
             "Use the recorded raw windows and offset sensitivity to assess contamination.", "",
             "| Case | Cells | Reconstruction | Root | Valid fit | $M_{m1}$ | $M_{m2}$ | "
             "$\\Delta s$ [J/(kg K)] | Mass defect | Momentum defect | $h_0$ defect | "
             "$\\ell$ defect | Pressure-ratio defect | Density-ratio defect |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source in report["sources"]:
        for row in source["methods"]:
            jump = row.get("planar_jump", {})
            oracle = row.get("normal_shock_reference_from_fitted_upstream", {})
            ratios = oracle.get("ratio_relative_defects", {})
            def number(value):
                return "n/a" if value is None else f"{value:.5g}"
            values = [row.get("upstream", {}).get("M_m"), row.get("downstream", {}).get("M_m"),
                      jump.get("entropy_rise"), jump.get("mass_defect"),
                      jump.get("normal_momentum_defect"), jump.get("h0_defect"),
                      jump.get("ell_defect"), ratios.get("p"), ratios.get("rho")]
            lines.append(f"| {source['case']} | {source['cells']} | {row['method']} | "
                         f"{row['root_success']} | {row['valid_extrapolation']} | "
                         + " | ".join(number(value) for value in values) + " |")
    lines.extend(["", "Mass and normal-momentum defects are downstream minus upstream, divided "
                  "by the larger absolute side flux. The $h_0$ scale is the largest of the "
                  "two absolute total enthalpies and upstream $a^2$. Angular continuity "
                  "uses upstream $|\\ell|$, or $ra$ for a zero-swirl reference. Tangential "
                  "velocity is also compared directly and scaled by upstream sound speed.", "",
                  "## Extrema inside the excluded shock region", "",
                  "These raw-cell diagnostics retain pressure/density peaks and velocity minima "
                  "that the one-sided fits exclude. Reference values use the saved conserved "
                  "cell projection over the same local indices; no new peak tolerance is imposed.", "",
                  "| Case | Cells | Reconstruction | Pressure peak [Pa] | Reference local max [Pa] | "
                  "Pressure peak relative excess | Density peak relative excess | Velocity minimum relative defect |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"])
    for source in report["sources"]:
        for row in source["methods"]:
            core = row.get("excluded_shock_region")
            if core is None:
                continue
            values = [core["pressure_max"], core["reference_pressure_max"],
                      core["pressure_peak_over_reference_window_max"],
                      core["density_peak_over_reference_window_max"],
                      core["v_m_min_relative_to_reference_window_min"]]
            lines.append(f"| {source['case']} | {source['cells']} | {row['method']} | "
                         + " | ".join(f"{value:.6g}" for value in values) + " |")
    lines.extend(["",
                  "## Flags and fitting windows", ""])
    for source in report["sources"]:
        lines.extend([f"### {source['case']}, {source['cells']} cells", "",
                      f"Source: `{source['file']}`; SHA-256 `{source['json_sha256']}`.", ""])
        for row in source["methods"]:
            flags = "; ".join(row["flags"]) if row["flags"] else "No physical/stencil flags"
            lines.append(f"- **{row['method']}**: {flags}.")
            if "estimated_station" in row:
                lines.append(f"  Estimated station {row['estimated_station']:.8g} m; "
                             f"known station {row['known_analytical_shock_station']:.8g} m; "
                             f"signed error {row['location_error_local_cells']:.5g} local cells. "
                             f"Common radius {row['common_radius']:.8g} m.")
                for side in ("upstream", "downstream"):
                    window = row[f"{side}_window"]
                    indices = np.asarray(window["indices"]).tolist()
                    lines.append(f"  {side.capitalize()} physical-cell indices: `{indices}`.")
        changed = source["recorded_source_hash_differences"]
        lines.extend(["", f"Recorded/current source-hash differences: {len(changed)}; "
                      "the exact paths and both version records are retained in JSON.", ""])
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", "--input-folder", type=Path,
                        default=STUDY/"outputs"/"verification")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inner-offset", type=int, default=3)
    parser.add_argument("--outer-offset", type=int, default=8)
    args = parser.parse_args()
    if args.inner_offset < 0 or args.outer_offset-args.inner_offset+1 < 3:
        parser.error("Offsets must be nonnegative and select at least three cells per side")
    destination = args.folder if args.output is None else args.output
    paths = sorted(path for path in args.folder.glob("shock*.json") if path.name != "shock_audits.json")
    if not paths:
        parser.error(f"No saved shock-case JSON files in {args.folder}")
    current_hashes = fingerprints()
    fluid = jxp.FluidPerfectGas(name="Air")
    report = dict(
        settings=dict(inner_offset=args.inner_offset, outer_offset=args.outer_offset,
                      fit="linear W extrapolation to one station/radius", jump_tolerances=None),
        audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        current_source_hashes=current_hashes,
        audit_versions={name: metadata.version(name) for name in ("jax", "jaxprop", "nurbspy", "numpy", "scipy")},
        fluid="jaxprop.FluidPerfectGas(Air)", sources=[])
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("shock_m") is None or "methods" not in saved:
            continue
        case = make_case(saved["case"], fluid)
        recorded_hashes = saved.get("source_hashes", {})
        source = dict(
            case=saved["case"], cells=saved["cells"], file=str(path.resolve()),
            json_sha256=sha256(path.read_bytes()).hexdigest(),
            recorded_versions=saved.get("versions", {}), recorded_source_hashes=recorded_hashes,
            recorded_source_hash_differences={key: dict(recorded=value, current=current_hashes.get(key))
                                              for key, value in recorded_hashes.items()
                                              if current_hashes.get(key) != value},
            methods=[audit_method(saved, method, case, fluid, args.inner_offset, args.outer_offset)
                     for method in RECONSTRUCTIONS if method in saved["methods"]])
        report["sources"].append(source)
    report["sources"].sort(key=lambda item: (item["case"], item["cells"]))
    if not report["sources"]:
        parser.error("No supported saved shocked solutions were found")
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"shock_audits.json").write_text(
        json.dumps(_safe_json(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (destination/"SHOCK_AUDITS.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Audited {len(report['sources'])} saved cases; outputs in {destination}", flush=True)
    return 0 if all(row["valid_extrapolation"] and not row["flags"]
                    for source in report["sources"] for row in source["methods"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
