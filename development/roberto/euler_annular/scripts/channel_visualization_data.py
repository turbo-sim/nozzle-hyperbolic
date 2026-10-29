"""Cache actual NURBS channels and illustrative quasi-1D flow paths for HTML.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.channel_visualization_data

Public API: ``build_channel_data(study: Path) -> dict``. This preparation imports
the authoritative case factory, but never builds or advances a numerical solver.
Saved finest-grid MUSCL velocities supply a kinematic angular phase. Geometry
and saved states remain read-only; the optional JSON cache keeps ordinary HTML
rebuilding independent of JAX/NURBSpy imports.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import re

import jaxprop as jxp
import numpy as np
from scipy.integrate import cumulative_trapezoid

from nozzle_hyperbolic.geometry.annular import conical_channel

from .verify_annular import DEFAULT_CASES, REPOSITORY, STUDY, make_case
from .html_report_data import resolve_verification_folder


METHOD = "muscl_van_leer"
ARROW_FRACTIONS = (0.22, 0.50, 0.78)
GEOMETRY_SOURCE = REPOSITORY / "src/nozzle_hyperbolic/geometry/annular.py"
CASE_SOURCE = Path(__file__).with_name("verify_annular.py")


def _read_saved(path, study):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=path.relative_to(study).as_posix(),
                                sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _finest_path(directory, case):
    pattern = re.compile(rf"^{re.escape(case)}_(\d+)\.json$")
    choices = [(int(match.group(1)), path) for path in directory.glob(f"{case}_*.json")
               if (match := pattern.fullmatch(path.name))]
    if not choices:
        raise FileNotFoundError(f"No saved numerical result for {case!r} in {directory}")
    return max(choices, key=lambda item: item[0])[1]


def _controls(geometry):
    def common(curve):
        return dict(weights=np.asarray(curve.W), degree=int(curve.p),
                    knots=np.asarray(curve.U))

    midline, height = geometry.midline, geometry.height
    # A scalar height function stores ordinate controls only. Greville positions
    # are an explicitly labeled display convention, not physical midline points.
    knots, degree = np.asarray(height.U), int(height.p)
    greville = np.asarray([np.mean(knots[index+1:index+degree+1])
                           for index in range(height.P.shape[1])])
    return dict(
        midline=dict(x=np.asarray(midline.P)[0], r=np.asarray(midline.P)[1],
                     **common(midline)),
        height=dict(sigma=greville, m=greville*geometry.length,
                    b=np.asarray(height.P)[0], **common(height),
                    sigma_placement="Greville abscissae of the scalar height basis",
                    physical_midline_points=False))


def _arrows(m, coordinates):
    """Three arrows from actual forward path chords, including normal rotation."""
    arrows = []
    for fraction in ARROW_FRACTIONS:
        index = int(np.clip(np.searchsorted(m, fraction*m[-1]), 1, len(m)-2))
        tail, head = coordinates[index-1], coordinates[index+1]
        chord = head-tail
        length = float(np.linalg.norm(chord))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("A channel arrow has no finite forward path direction")
        arrows.append(dict(m=float(m[index]), m_tail=float(m[index-1]),
                           m_head=float(m[index+1]), tail=tail, head=head,
                           position=head, direction=chord/length, chord_length=length))
    return arrows


def _finite_or_none(values):
    """Keep invalid flow evidence JSON-safe without repairing numerical values."""
    return [float(value) if np.isfinite(value) else None for value in values]


def _flow(saved, run, geometry, m, metrics):
    centers = np.asarray(saved["m"], dtype=float)
    faces = np.asarray(saved["m_faces"], dtype=float)
    state = np.asarray(run["state"], dtype=float)
    if state.shape != (len(centers)+2, 4):
        raise ValueError("Saved MUSCL state must contain both boundary trace blocks")
    source_m = np.r_[faces[0], centers, faces[-1]]
    if (not np.isfinite(source_m).all() or not np.all(np.diff(source_m) > 0)
            or not np.isclose(source_m[0], 0., rtol=0, atol=1e-13)
            or not np.isclose(source_m[-1], geometry.length, rtol=1e-11, atol=1e-13)):
        raise ValueError("Saved MUSCL coordinates differ from the physical channel domain")
    v_m_source, ell_source = state[:, 1], state[:, 2]
    v_m = np.interp(m, source_m, v_m_source)
    ell = np.interp(m, source_m, ell_source)
    v_theta = ell/metrics.r
    flags = []
    if not np.isfinite(v_m_source).all() or not np.isfinite(ell_source).all():
        flags.append("Nonfinite saved velocity/angular momentum; angular paths are unavailable")
    if np.any(v_m_source <= 0):
        flags.append("Nonpositive saved meridional velocity; no angular division or flow arrows were attempted")
    for key, expected in (("v_m", v_m_source[1:-1]), ("ell", ell_source[1:-1])):
        recorded = run.get("fields", {}).get(key)
        if recorded is not None and not np.allclose(recorded, expected, rtol=1e-11, atol=1e-12):
            flags.append(f"Saved physical {key} field differs from the saved primitive state")
    result = dict(
        valid=False, flags=flags, m=m, v_m=_finite_or_none(v_m),
        ell=_finite_or_none(ell), v_theta=_finite_or_none(v_theta),
        theta=None, dtheta_dm=None,
        theta_rule="dtheta/dm = v_theta/(r*v_m) = ell/(r^2*v_m)",
        interpolation="piecewise linear saved primitive v_m and ell, including physical-face traces",
        integration="cumulative trapezoidal quadrature on the union of saved state and rendering positions",
        source_points=dict(m=source_m, v_m=_finite_or_none(v_m_source),
                           ell=_finite_or_none(ell_source)))
    if flags:
        return result
    integration_m = np.unique(np.r_[m, source_m])
    integration_r = np.asarray(geometry.evaluate(integration_m).r)
    integration_v_m = np.interp(integration_m, source_m, v_m_source)
    integration_ell = np.interp(integration_m, source_m, ell_source)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        derivative = integration_ell/(integration_r**2*integration_v_m)
        theta = cumulative_trapezoid(derivative, integration_m, initial=0.)
    if not np.isfinite(derivative).all() or not np.isfinite(theta).all():
        flags.append("Angular-phase quadrature is nonfinite; paths were not substituted with geometric flow lines")
        return result
    result.update(valid=True, theta=np.interp(m, integration_m, theta),
                  dtheta_dm=ell/(np.asarray(metrics.r)**2*v_m),
                  integration_points=len(integration_m),
                  total_angle_rad=float(theta[-1]),
                  total_turns=float(theta[-1]/(2*np.pi)))
    return result


def audit_channel_record(record):
    """Check metric, offset, revolution and forward-arrow identities explicitly."""
    geometry = record["geometry"]
    m, x, radius, height, x_prime, r_prime, area = (
        np.asarray(geometry[key], dtype=float) for key in
        ("m", "x", "r", "b", "x_prime", "r_prime", "A"))
    area_error = float(np.max(np.abs(area-2*np.pi*radius*height)/np.maximum(np.abs(area), 1e-300)))
    tangent_error = float(np.max(np.abs(x_prime*x_prime+r_prime*r_prime-1)))
    offset_error = 0.
    wall_radii = []
    for label, eta in (("minus", -.5), ("plus", .5)):
        wall = record["walls"][label]
        actual = np.column_stack((wall["x"], wall["r"]))
        expected = np.column_stack((x-eta*height*r_prime, radius+eta*height*x_prime))
        offset_error = max(offset_error, float(np.max(np.abs(actual-expected))))
        wall_radii.extend(wall["r"])
    arrow_norm_error, radius_error = 0., 0.
    arrows_forward = True
    for path in record["cross_gap_paths"]+record["paths_3d"]:
        for arrow in path["arrows"]:
            direction = np.asarray(arrow["direction"])
            chord = np.asarray(arrow["head"])-np.asarray(arrow["tail"])
            arrow_norm_error = max(arrow_norm_error, abs(float(np.linalg.norm(direction))-1))
            arrows_forward &= (arrow["m_head"] > arrow["m_tail"] and float(direction@chord) > 0)
    for path in record["paths_3d"]:
        radius_error = max(radius_error, float(np.max(np.abs(
            np.hypot(path["y"], path["z"])-np.asarray(path["r"])))))
    finite_geometry = bool(np.isfinite(np.column_stack((m, x, radius, height, x_prime, r_prime, area))).all())
    checks = dict(finite_geometry=finite_geometry, positive_radius=bool(np.all(radius > 0)),
                  positive_height=bool(np.all(height > 0)),
                  positive_wall_radii=bool(np.all(np.asarray(wall_radii) > 0)),
                  increasing_m=bool(np.all(np.diff(m) > 0)), unit_tangent=tangent_error < 1e-10,
                  area_identity=area_error < 1e-12, wall_offset_identity=offset_error < 1e-12,
                  finite_unit_arrows=bool(np.isfinite(arrow_norm_error) and arrow_norm_error < 1e-12),
                  arrows_follow_increasing_m=bool(arrows_forward),
                  revolution_radius_identity=radius_error < 1e-12,
                  angular_flow_available=record["flow"]["valid"])
    return dict(passed=all(checks.values()), checks=checks,
                max_unit_tangent_error=tangent_error, max_area_relative_error=area_error,
                max_wall_offset_error_m=offset_error, max_arrow_unit_norm_error=arrow_norm_error,
                max_revolution_radius_error_m=radius_error,
                minimum_wall_radius_m=float(min(wall_radii)),
                arrows_per_path=len(ARROW_FRACTIONS),
                phase_integration_is_visualization_only=True)


def _record(identifier, kind, saved, source, geometry, points):
    run = saved["methods"][METHOD]
    m = np.linspace(0., geometry.length, points)
    metrics = geometry.evaluate(m)
    normal_x, normal_r = -metrics.r_prime, metrics.x_prime
    walls = {label: dict(x=metrics.x+eta*metrics.b*normal_x,
                         r=metrics.r+eta*metrics.b*normal_r)
             for label, eta in (("minus", -.5), ("plus", .5))}
    flow = _flow(saved, run, geometry, m, metrics)
    cross_gap = []
    for eta in (-.3, 0., .3):
        coordinates = np.column_stack((metrics.x+eta*metrics.b*normal_x,
                                       metrics.r+eta*metrics.b*normal_r))
        cross_gap.append(dict(eta=eta, x=coordinates[:, 0], r=coordinates[:, 1],
                              arrows=_arrows(m, coordinates) if flow["valid"] else [],
                              interpretation="constant normal-height-fraction guide; cross-gap velocity is not resolved"))
    paths = []
    if flow["valid"]:
        for theta0 in np.arange(6)*2*np.pi/6:
            theta = flow["theta"]+theta0
            coordinates = np.column_stack((metrics.x, metrics.r*np.cos(theta), metrics.r*np.sin(theta)))
            paths.append(dict(eta=0., theta0=float(theta0), theta=theta, r=metrics.r,
                              x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                              arrows=_arrows(m, coordinates)))
    success = run.get("success", run.get("checks", {}).get("success", False))
    record = dict(
        id=identifier, kind=kind, case=saved["case"], cells=int(saved["cells"]),
        method=METHOD, source=source, source_success=bool(success),
        source_is_final_time_snapshot=(kind == "transient"),
        geometry=dict(m=m, x=metrics.x, r=metrics.r, b=metrics.b,
                      x_prime=metrics.x_prime, r_prime=metrics.r_prime,
                      A=metrics.A, normal_x=normal_x, normal_r=normal_r,
                      length=float(geometry.length)),
        walls=walls, control_points=_controls(geometry), flow=flow,
        cross_gap_paths=cross_gap, paths_3d=paths,
        labels=dict(x="Axial coordinate x [m]", r="Radius r [m]",
                    y="Cartesian transverse coordinate y [m]", z="Cartesian transverse coordinate z [m]",
                    m="Meridional arclength m [m]", b="Normal height b [m]", theta="Azimuth theta [rad]"),
        interpretation=("Representative instantaneous velocity-field guides from the final physical-time snapshot; not time-integrated particle trajectories. "
                        if kind == "transient" else "Representative steady quasi-1D kinematic paths. ")
                       + "The model has no resolved cross-gap or circumferential velocity field. Constant-offset meridional guides are illustrative; "
                         "normal rotation and width variation alter their tangents. Six 3D midgap seeds reuse the centerline angular phase. "
                         "Arrows use the actual increasing-m plotted path chords; no geometric surface point is presented as a NURBS control point.")
    if "r" in saved:
        reconstructed = np.asarray(geometry.evaluate(np.asarray(saved["m"])).r)
        error = float(np.max(np.abs(reconstructed-np.asarray(saved["r"]))))
        record["saved_center_radius_max_error_m"] = error
        if error > 1e-11:
            raise ValueError(f"Saved radii do not match the reconstructed NURBS geometry for {identifier}")
    record["audit"] = audit_channel_record(record)
    return record


def build_channel_data(study: Path, *, points=161, verification_folder: Path | None = None) -> dict:
    """Return canonical channels plus available graded-mesh/transient channels."""
    if not isinstance(points, int) or isinstance(points, bool) or points < 9:
        raise ValueError("Use at least nine rendering positions")
    study = Path(study).resolve()
    primary = resolve_verification_folder(study, verification_folder)
    watched = [Path(__file__), CASE_SOURCE, GEOMETRY_SOURCE]
    source_hashes = {path.relative_to(REPOSITORY).as_posix(): sha256(path.read_bytes()).hexdigest()
                     for path in watched}
    fluid = jxp.FluidPerfectGas(name="Air")  # Only the existing case/reference factory needs this backend.
    cases, geometries, inputs = {}, {}, {}
    for name in DEFAULT_CASES:
        path = _finest_path(primary, name)
        saved, source = _read_saved(path, study)
        if saved["case"] != name:
            raise ValueError(f"Saved case identifier differs from filename: {path}")
        geometry = make_case(name, fluid).geometry
        geometries[name] = geometry
        cases[name] = _record(name, "canonical", saved, source, geometry, points)
        inputs[source["path"]] = source
    nonuniform_directory = study/"outputs"/"nonuniform"
    if nonuniform_directory.is_dir():
        path = _finest_path(nonuniform_directory, "radial_outflow")
        saved, source = _read_saved(path, study)
        cases["nonuniform"] = _record("nonuniform", "nonuniform", saved, source,
                                       geometries["radial_outflow"], points)
        inputs[source["path"]] = source
    transient_directory = study/"outputs"/"transient"
    if transient_directory.is_dir():
        path = _finest_path(transient_directory, "contact")
        saved, source = _read_saved(path, study)
        settings = saved["settings"]
        geometry = conical_channel(settings["length"], settings["radius"], settings["width"], 0.)
        cases["transient"] = _record("transient", "transient", saved, source, geometry, points)
        inputs[source["path"]] = source
    for source in inputs.values():
        if sha256((study/source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Saved channel input changed while loading: {source['path']}")
    if any(sha256(path.read_bytes()).hexdigest() != source_hashes[path.relative_to(REPOSITORY).as_posix()]
           for path in watched):
        raise ValueError("A geometry preparation source changed while loading")
    result = dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        conventions=dict(axis_of_revolution="x", cartesian="(x, r*cos(theta), r*sin(theta))",
                         normal="(-r_prime, x_prime)", eta="normal displacement divided by b; walls eta=-0.5,+0.5",
                         forward_direction="increasing meridional arclength m; radial inflow has decreasing r",
                         angle="radians; positive v_theta advances theta",
                         control_points="Actual normalized NURBSpy P/W/p/U; height control abscissae use labeled Greville placement",
                         rendering_only=True, numerical_solve_performed=False),
        cases=cases, audit_pass=all(record["audit"]["passed"] for record in cases.values()),
        provenance=dict(input_files=list(inputs.values()), source_hashes=source_hashes,
                        versions={name: metadata.version(name) for name in ("nurbspy", "jaxprop", "numpy", "scipy")},
                        source_and_inputs_unchanged_during_build=True))
    return json.loads(json.dumps(result, allow_nan=False,
                                 default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--points", type=int, default=161)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.study/"outputs"/"html_report"/"channel_geometry.json"
    result = build_channel_data(args.study, points=args.points)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, separators=(",", ":"))+"\n", encoding="utf-8")
    print(f"Saved {len(result['cases'])} channel visualizations to {output}")
    for key, record in result["cases"].items():
        print(f"{key}: audit={record['audit']['passed']} phase={record['flow'].get('total_angle_rad')} rad "
              f"3D paths={len(record['paths_3d'])} flags={record['flow']['flags']}")
    return 0 if result["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
