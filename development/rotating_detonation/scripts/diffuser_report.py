"""Offline report for angle-only forcing of the annular transition duct.

This presentation-only module reads outputs/diffuser and never imports a solver.
Its embedded archive retains the plot fields/settings/audits; the larger raw
primitive and accepted-step records remain in the fingerprinted numerical files.
Run with ``python -m development.rotating_detonation.scripts.diffuser_report``.
"""

from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
DATA = STUDY / "outputs/diffuser"
VENDOR = STUDY / "scripts/vendor"
COLORS = ["#7e03a8", "#cc4778", "#f89540", "#4b2e83", "#f0c345"]
LABELS = {"rk3": "SSPRK3", "implicit": "Backward Euler"}
OMITTED_TRANSIENT_KEYS = ("states", "history", "metrics", "rejections")
FIELDS = [
    ("p", "Static pressure", "bar", 1e-5),
    ("M", "Total Mach number", "M", 1),
    ("M_m", "Meridional Mach number", "Mₘ", 1),
    ("T", "Static temperature", "K", 1),
    ("v_m", "Meridional velocity", "m/s", 1),
    ("alpha_deg", "Flow angle", "deg", 1),
    ("mdot", "Mass flow", "kg/s", 1),
]


def load(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=5):
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, int, np.number)):
        return f"{value:.{digits}g}" if np.isfinite(value) else esc(value)
    return esc(value)


def maximum(value):
    try:
        a = np.asarray(value, dtype=float)
        return float(np.max(np.abs(a))) if a.size else None
    except (TypeError, ValueError):
        return None


def details(title, value):
    return f"<details><summary>{esc(title)}</summary><pre>{esc(json.dumps(value, indent=2))}</pre></details>"


def table(headers, rows):
    if not rows:
        return '<p class="muted">No records are available for this comparison.</p>'
    return (
        '<div class="tablewrap"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def field(record, name):
    fields = record.get("fields", {})
    alias = {"v_m": "vm", "v_theta": "vtheta", "M_m": "Mm"}
    value = fields.get(name, fields.get(alias.get(name, "")))
    return np.asarray(value, dtype=float) if value is not None else np.array([])


def steady_grids(value):
    return value.get("grids", value.get("representative", []))


def assessment(data, selected_id):
    value = data.get("periodicity", {})
    if selected_id in value and isinstance(value[selected_id], dict):
        return value[selected_id]
    return value if isinstance(value, dict) else {}


def period_pass(value):
    return value.get("verified", value.get("periodic_verified", value.get("passed", False))) is True


def line(x, y, name, color, dash="solid", **extra):
    return dict(x=np.asarray(x).tolist(), y=np.asarray(y).tolist(), name=name,
                mode="lines", line=dict(color=color, width=2.4, dash=dash), **extra)


def file_fingerprint(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def archive_snapshots(record, frames):
    """Project long spatial snapshot series while retaining all boundary times."""
    if frames < 1:
        raise ValueError("Snapshot projection requires a positive frames_per_cycle")
    compact = {key: value for key, value in record.items() if key not in OMITTED_TRANSIENT_KEYS}
    count = len(record.get("times", []))
    if count <= 28*frames+1:
        return compact
    indices = sorted(set(range(min(3*frames+1, count)))
                     | set(range(max(0, count-(25*frames+1)), count)))

    def selected(series, key):
        if len(series) != count:
            raise ValueError(f"Cannot project {key}: {len(series)} values for {count} snapshot times")
        return [series[i] for i in indices]

    compact["times"] = selected(record["times"], "times")
    for key in ("fields", "boundary_fields"):
        if key in record:
            compact[key] = {name: selected(series, f"{key}.{name}") for name, series in record[key].items()}
    if "contents" in record:
        compact["contents"] = selected(record["contents"], "contents")
    compact["boundary_history"] = dict(times=record["times"], fields=record.get("boundary_fields", {}))
    compact["snapshot_selection"] = dict(raw_indices=indices, original_count=count,
        retained_count=len(indices), frames_per_cycle=frames, startup_cycles=3, final_cycles=25,
        reason="Long settling record: retain the first three and final twenty-five forcing cycles of spatial snapshots. Full sampled boundary history is embedded separately; all raw spatial snapshots remain in the fingerprinted numerical artifacts.")
    return compact


def compact_evidence(data, geometry, visuals, audit, *, data_dir=DATA, digest=None):
    """Prune bulk step evidence and project long snapshot series in the download.

    Container copies and index selection preserve retained numerical values
    exactly. The loaded campaign and on-disk evidence are never modified.
    """
    data_dir = Path(data_dir)
    compact = dict(data)
    compact["transient"] = {name: archive_snapshots(record,
        int(record.get("config", data.get("config", {})).get("forcing", {}).get("frames_per_cycle", 24)))
        for name, record in data.get("transient", {}).items()}

    def artifact(path, omitted):
        try:
            relative = path.resolve().relative_to(STUDY).as_posix()
        except ValueError:
            relative = str(path.resolve())
        return dict(path=relative, available=path.is_file(), omitted_keys=omitted,
                    **(file_fingerprint(path) if path.is_file() else {}))

    omissions = {name: [key for key in OMITTED_TRANSIENT_KEYS if key in record]
                 for name, record in data.get("transient", {}).items()}
    manifest = dict(
        raw_results=artifact(data_dir/"results.json", [f"transient.{name}.{key}"
            for name, keys in omissions.items() for key in keys]),
        raw_runs={name: artifact(data_dir/"runs"/f"{name}.json", keys) for name, keys in omissions.items()},
        snapshot_projections={name: record["snapshot_selection"] for name, record in compact["transient"].items() if "snapshot_selection" in record},
        omission_scope="Transient states/history/metrics/rejections are omitted from this embedded archive. Records longer than twenty-eight output cycles retain spatial snapshots from their first three and final twenty-five cycles, with explicit original indices and complete sampled boundary history. Full raw results and individual run JSON files are unchanged.",
        retained_content="Exact configuration, selected spatial snapshots with aligned times/fields/boundary_fields/contents, complete sampled boundary histories, all cycle metrics/summaries, steady states, comparisons, temporal checks, provenance, geometry and audits. No floating-point values are rounded.",
    )
    return dict(results=compact, geometry=geometry, visuals=visuals, audit=audit,
                results_sha256=digest, raw_artifact_manifest=manifest)


def make_report(data, geometry=None, visuals=None, audit=None, *, data_dir=DATA, digest=None):
    """Return HTML from retained evidence; missing analyses are labeled pending.

    Supported comparison form: x, steady:{field:[]}, mean:{field:[]},
    envelope:{field:{min:[],max:[]}}, window_start_s, window_end_s,
    complete_cycles, verified_periodic, summary, temporal_mean_sensitivity.
    The compact fields:{field:{steady:[],mean:[],min:[],max:[]}} form also works.
    """
    cfg = data.get("config", {})
    geometry = geometry or data.get("geometry", {})
    visuals = visuals or {}
    audit = audit or {}
    transient = data.get("transient", {})
    selected_id = data.get("selected_id", next(iter(transient), ""))
    chosen = transient.get(selected_id, {})
    phase_count = int(chosen.get("config", cfg).get("forcing", {}).get("frames_per_cycle", 24))
    slider_record = archive_snapshots(chosen, phase_count)
    slider_selection = slider_record.get("snapshot_selection")
    slider_indices = slider_selection["raw_indices"] if slider_selection else list(range(len(chosen.get("times", []))))
    slider_gaps = [dict(after_frame=i, raw_after=left, raw_before=right,
        from_time_s=chosen["times"][left], to_time_s=chosen["times"][right])
        for i, (left, right) in enumerate(zip(slider_indices[:-1], slider_indices[1:])) if right-left > 1]
    periodic = assessment(data, selected_id)
    verified = period_pass(periodic)
    grids = {name: steady_grids(v) for name, v in data.get("steady", {}).items()}
    grids = {name: v for name, v in grids.items() if v}
    base_name = "implicit" if "implicit" in grids else next(iter(grids), "")
    baseline = grids[base_name][-1] if base_name else {}
    plots = []
    math_manifest = load(VENDOR/"mathjax-render-manifest.json")
    if math_manifest is None:
        raise FileNotFoundError("Prerendered MathJax assets are missing; use scripts/vendor/render_math.mjs")
    used_math = set()

    def equation(key):
        formula = math_manifest["formulas"][key]
        path = VENDOR/formula["svg_file"]
        svg = path.read_text(encoding="utf-8")
        if sha256(svg.encode()).hexdigest() != formula["sha256"]:
            raise ValueError(f"Prerendered MathJax SVG digest mismatch: {key}")
        used_math.add(key)
        return f'<div class="equation" data-math-key="{key}" data-latex="{esc(formula["tex"])}">{svg}</div>'

    def plot(title, traces, xlabel, ylabel, **layout):
        if not traces:
            return ""
        ident = f"plot-{len(plots)}"
        base = dict(title=dict(text=title, x=.04), xaxis=dict(title=xlabel),
                    yaxis=dict(title=ylabel))
        base.update(layout)
        plots.append(dict(id=ident, data=traces, layout=base))
        return f'<div id="{ident}" class="plot" role="img" aria-label="{esc(title)}"></div>'

    def asset(filename, caption, ident):
        path = data_dir / Path(filename).name
        if not path.exists():
            return f'<p class="notice">{esc(caption)}: the asset has not been generated.</p>'
        if visuals.get("results_sha256") and digest and visuals["results_sha256"] != digest:
            return '<p class="notice">Visual evidence belongs to a different results digest and is omitted.</p>'
        mime = "image/gif" if path.suffix.lower() == ".gif" else "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return (f'<figure><img id="{ident}" src="data:{mime};base64,{encoded}" alt="{esc(caption)}">'
                f'<figcaption>{esc(caption)}</figcaption><button data-save-image="{ident}" '
                f'data-filename="{esc(path.name)}">Download image</button></figure>')

    def case_label(name, value):
        method = value.get("method", name)
        role = {"primary": "primary run", "time_refinement": "time-step check"}.get(value.get("analysis_role"))
        return f"{LABELS.get(method, method)} · CFL {fmt(value.get('cfl'))}" + (f" · {role}" if role else "")

    forcing = cfg.get("forcing", {})
    frequency = forcing.get("frequency_Hz")
    end_time = chosen.get("times", [None])[-1] if chosen.get("times") else None
    pressure_bar = cfg.get("outlet", {}).get("pressure_Pa", 0)*1e-5
    inlet = cfg.get("inlet", {})
    nominal_angle = inlet.get("angle_deg", 0)
    angle_amplitude = forcing.get("angle_amplitude_deg", 0)
    angle_range = f"{fmt(nominal_angle-angle_amplitude)}–{fmt(nominal_angle+angle_amplitude)}°"
    gcfg = cfg.get("geometry", {})
    radius = gcfg.get("tip_radius_m")
    upstream_length = gcfg.get("constant_length_m")
    expansion_length = gcfg.get("expansion_length_m")
    tail_length = gcfg.get("extension_length_m", 0)
    entrance_blend = gcfg.get("junction_blend_m", 0)
    exit_blend = gcfg.get("exit_blend_m", entrance_blend)
    hub_in = gcfg.get("inlet_hub_tip")
    hub_out = gcfg.get("outlet_hub_tip")
    area_ratio = gcfg.get("outlet_area_ratio")
    if area_ratio is not None and hub_in is not None:
        hub_out = float(np.sqrt(1-area_ratio*(1-hub_in*hub_in)))
    if "A" in geometry and len(geometry["A"]):
        area_ratio = float(geometry["A"][-1]/geometry["A"][0])
    full_cells = len(chosen.get("x", baseline.get("x", [])))
    grid_counts = [r.get("cells") for r in grids.get(base_name, [])]
    grid_chain = " → ".join(str(n) for n in grid_counts) if grid_counts else "the configured sequence"
    statuses = [r.get("success", False) for v in grids.values() for r in v]
    statuses += [v.get("success", False) for v in transient.values()]
    status = "All retained solves completed" if statuses and all(statuses) else "Finite transient results retained; stopping diagnostics are shown"
    title = '<div class="eyebrow">NOZZLE-HYPERBOLIC / ROTATING-DETONATION STUDY</div><h1>A swirling flow.<br>A meridional diffuser.</h1>'
    title += '<p class="lead">An annular transition duct driven only by its inlet angle. Reservoir conditions and exit pressure stay fixed; Mach number and mass flow respond to the forcing.</p>'
    title += '<div class="cards">' + "".join(
        f"<div><b>{esc(a)}</b><span>{esc(b)}</span></div>" for a, b in [
            (angle_range, "Prescribed inlet angle range"),
            (fmt(pressure_bar) + " bar", "Constant exit static pressure"),
            ((fmt(end_time * 1000) + " ms") if end_time is not None else "Pending", "Retained physical end time"),
            ("Verified" if verified else "Not established", "Periodic state of selected run")]) + '</div>'
    title += f'<p class="status">{esc(status)}. Solver convergence, flow regime and periodicity are assessed separately.</p>'

    # Geometry is always present as a report chapter, even when evidence is incomplete.
    geom = f'<h2>The duct geometry</h2><p>The cylindrical tip has radius {fmt(radius)} m. A {fmt(upstream_length)} m constant passage starts at hub/tip ratio {fmt(hub_in)}; a {fmt(expansion_length)} m expansion lowers that ratio to {fmt(hub_out,8)}. A {fmt(tail_length)} m straight, constant-area tail provides an axial outlet. The area ratio is {fmt(area_ratio,8)}. The hub slope blends over {fmt(entrance_blend)} m at the expansion entrance and {fmt(exit_blend)} m at its exit, with a linear central retreat.</p>'
    geom += equation("area_ratio")
    required = ("tip_x", "tip_r", "hub_x", "hub_r", "x", "r")
    if all(k in geometry for k in required):
        geom += plot("Meridional section · the physical walls and midline", [
            line(geometry["tip_x"], geometry["tip_r"], "Tip", COLORS[2]),
            line(geometry["hub_x"], geometry["hub_r"], "Hub", COLORS[0]),
            line(geometry["x"], geometry["r"], "Midline", COLORS[1], "dash")],
            "Axial coordinate x [m]", "Radius [m]")
    else:
        geom += '<p class="notice">Sampled geometry evidence is pending.</p>'
    if all(k in geometry for k in ("m", "b", "A")):
        geom += plot("Normal height and flow area", [
            line(geometry["m"], np.asarray(geometry["b"]) * 1000, "Height", COLORS[0]),
            line(geometry["m"], np.asarray(geometry["A"]) * 1e4, "Area", COLORS[2], yaxis="y2")],
            "Midline arclength m [m]", "Normal height [mm]",
            yaxis2=dict(title="Area [cm²]", overlaying="y", side="right"))
    geom += equation("normal_geometry") + '<p>The height is normal to the midline, so it differs from the radial wall gap where the duct slopes. The chosen height keeps the physical offset tip cylindrical. The inlet and final outlet are axial. A quasi-1D rendering extends each cell value uniformly through that normal height; it does not resolve a transverse profile.</p>'
    geom += asset(visuals.get("steady_file", "steady_contours.png"),
                  "Steady solution rendered along the actual duct using the plasma colormap.", "steady-contours")

    bc = f'<h2>Angle forcing with fixed reservoir and exit pressure</h2><p>The inlet holds stagnation temperature at {fmt(inlet.get("stagnation_temperature_K"))} K and stagnation pressure at {fmt(inlet.get("stagnation_pressure_Pa",0)*1e-5)} bar. Its angle has mean {fmt(nominal_angle)}° and amplitude {fmt(angle_amplitude)}° at {fmt(frequency)} Hz. No Mach-number waveform is prescribed. Mach number and throughput respond to the duct and its constant exit static pressure of {pressure_bar:.8f} bar.</p>'
    bc += equation("angle_forcing") + equation("reservoir_inlet")
    bc += '<p>At a positive, subsonic-meridional inlet, these three physical conditions are completed by the outgoing acoustic relation from the interior. The relevant characteristic speed uses meridional velocity.</p>'
    bc += equation("normal_characteristics") + equation("pressure_outlet")
    bc += '<p>The subsonic pressure outlet combines the prescribed pressure with outgoing acoustic, entropy and angular-momentum information. Its solved physical trace supplies the outlet flux. Constant pressure is a downstream load that can reflect acoustic disturbances; it is not a nonreflecting turbine or plenum model.</p>'
    bc += '<p>Compressible storage permits instantaneous inlet and outlet mass flow to differ. Equality of their averages is expected after a periodic state has been established; a settling transient can have a changing duct inventory.</p>' + equation("mass_inventory")
    bc += details("Effective outlet and inlet configuration", {k: cfg.get(k) for k in ("inlet", "outlet", "forcing")})

    steady = f'<h2>The steady diffuser branch</h2><p>HLL flux and van Leer MUSCL reconstruction are used on {esc(grid_chain)} cells. Grid sequencing transfers a converged coarse field to the next grid; every grid must satisfy its original spatial equations. The transient uses one {full_cells}-cell grid over the full duct and tail.</p>'
    for key, label, unit, scale in FIELDS:
        traces = []
        for i, (name, records) in enumerate(grids.items()):
            r = records[-1]
            value = field(r, key)
            if value.size and len(r.get("x", [])) == value.size:
                traces.append(line(r["x"], value * scale, LABELS.get(name, name), COLORS[i % len(COLORS)]))
        steady += plot(label, traces, "Axial coordinate x [m]", unit)
    rows = []
    for name, records in grids.items():
        for r in records:
            mt = r.get("metrics", {})
            rows.append([esc(LABELS.get(name, name)), fmt(r.get("cells")), fmt(r.get("success")),
                         esc(r.get("reason", "")), fmt(maximum(mt.get("equation_norm"))),
                         fmt(mt.get("boundary_max")), fmt(maximum(mt.get("global_defect"))), fmt(r.get("elapsed_s"))])
    steady += table(["Method", "Cells", "Solver pass", "Reason", "Equation RMS max", "BC max", "Global defect max", "Elapsed [s]"], rows)
    traces = []
    for i, (name, records) in enumerate(grids.items()):
        accepted = [v for v in records[-1].get("history", []) if v.get("accepted", True)]
        valid = [v for v in accepted if maximum(v.get("equation_norm")) is not None]
        if valid:
            traces.append(line([v.get("step", j) for j, v in enumerate(valid)],
                               [maximum(v["equation_norm"]) for v in valid], LABELS.get(name, name), COLORS[i % len(COLORS)]))
    steady += plot("Original spatial residual · final grid", traces, "Accepted step", "Maximum fixed-scale equation RMS", yaxis=dict(title="Maximum fixed-scale equation RMS", type="log"))
    steady += '<h3>Physical branch checks</h3><p>The design branch is meridionally subsonic and loses throughflow velocity along the expansion. The transient audit checks 0 &lt; Mₘ &lt; 1 at every cell and physical boundary trace, and net meridional deceleration from the expansion entry to exit. Inlet total Mach is measured. Instantaneous total-speed ratios are reported separately because waves reach the two planes at different phases.</p>'
    steady += details("Saved steady summaries and independent checks", {name: {"summary": rec[-1].get("summary", {}), "audit": rec[-1].get("audit", {})} for name, rec in grids.items()})
    reference = data.get("reference", {})
    spatial_rows = []
    for key, label, unit, scale in FIELDS:
        actual = field(baseline, key)
        exact = np.asarray(reference.get("fields", {}).get(key, []))
        if actual.size and exact.shape == actual.shape:
            error = actual-exact
            spatial_rows.append([esc(label), esc(unit), fmt(np.sqrt(np.mean(error*error))*scale),
                fmt(actual[-1]*scale), fmt(exact[-1]*scale),
                fmt(100*error[-1]/exact[-1] if abs(exact[-1]) > 1e-15 else None)])
    if spatial_rows:
        steady += '<h3>Spatial accuracy against the smooth steady reference</h3><p>The independent perfect-gas reference conserves entropy, stagnation enthalpy, angular momentum and mass flow on the meridionally subsonic branch. This comparison measures spatial accuracy separately from residual convergence.</p>'
        steady += table(["Field", "Unit", "Spatial RMS error", "Computed last cell", "Reference last cell", "Last-cell error [%]"], spatial_rows)
    if audit:
        steady += details("Independent saved-state audit", audit)

    forced = '<h2>The retained transient response</h2>'
    primary = transient.get(data.get("startup_id", selected_id), chosen)
    primary_times = primary.get("times", [])
    primary_cycles = (primary_times[-1]-primary_times[0])*frequency if primary_times and frequency else None
    forced += f'<p>The angle forcing frequency is {fmt(frequency)} Hz. The primary run spans <b>{fmt(primary_cycles)} forcing cycles</b> from the converged steady state and reaches {fmt(primary_times[-1] if primary_times else None)} s. The shorter time-step check restarts from a retained state to compare the same final physical interval. Completing the requested time limit is separate from reaching a purely periodic response.</p>'
    forced += f'<p class="notice"><b>Selected trajectory: {esc(selected_id or "pending")}.</b> Periodic state: {"verified by the retained criterion" if verified else "not established by the retained evidence"}. An end time by itself is not evidence of a periodic state.</p>'
    forced += table(["Run", "Start time [ms]", "End time [ms]", "Duration [cycles]", "Max meridional Mach", "Stopping reason"], [
        [esc(case_label(name, r)), fmt(r["times"][0]*1e3 if r.get("times") else None), fmt(r["times"][-1]*1e3 if r.get("times") else None),
         fmt((r["times"][-1]-r["times"][0])*frequency if r.get("times") and frequency else None),
         fmt(r.get("summary", {}).get("domain_M_m_max")), esc(r.get("reason", ""))]
        for name, r in transient.items()])
    collection_rows = []
    for name, record in transient.items():
        summary = record.get("summary", {})
        collection_rows.append([esc(case_label(name, record)), fmt(summary.get("periodic_candidate_cycle")),
            fmt(summary.get("collection_start_cycle")), fmt(summary.get("collection_end_cycle")),
            fmt(summary.get("collection_cycles_completed", summary.get("collection_cycles"))), fmt(summary.get("periodic_verified"))])
    forced += table(["Run / purpose", "Periodic candidate cycle", "Collection start cycle", "Collection end cycle", "Collected cycles", "Verified"], collection_rows)
    if periodic:
        forced += '<p>' + esc(periodic.get("criterion", "See the retained assessment for the exact phase-aligned, long-lag and inventory criteria and their normalization.")) + '</p>'
        forced += equation("periodic_rms")
        cycles = periodic.get("cycles", [])
        if cycles:
            for title_text, quantity, candidates in [
                ("Full-field periodicity · adjacent and long-lag cycles", "Maximum normalized RMS", [("maximum", "Adjacent cycle"), ("lag_maximum", "Long-lag cycle")]),
                ("Local periodicity · strongest retained cell change", "Maximum normalized pointwise change", [("local_maximum", "Adjacent cycle"), ("lag_local_maximum", "Long-lag cycle")]),
                ("Conserved-inventory repeatability", "Maximum normalized inventory drift", [("inventory_maximum", "Inventory drift")]),
            ]:
                traces = []
                for i, (key, label) in enumerate(candidates):
                    rows = [row for row in cycles if row.get(key) is not None and np.isfinite(row[key])]
                    if rows:
                        traces.append(line([row["cycle"] for row in rows], [row[key] for row in rows], label, COLORS[i]))
                forced += plot(title_text, traces, "Completed forcing cycle", quantity, yaxis=dict(title=quantity, type="log"))
        forced += details("Periodic-state assessment", periodic)
    for key, label in (("inlet_M", "Inlet absolute Mach"), ("inlet_M_m", "Inlet meridional Mach"), ("outlet_M_m", "Outlet meridional Mach"), ("inlet_mdot", "Inlet mass flow [kg/s]"), ("outlet_mdot", "Outlet mass flow [kg/s]")):
        traces = []
        for i, (name, r) in enumerate(transient.items()):
            y = r.get("boundary_fields", {}).get(key)
            if y is not None and len(y) == len(r.get("times", [])):
                traces.append(line(np.asarray(r["times"]) * 1000, y, case_label(name, r), COLORS[i % len(COLORS)], "dash" if r.get("method") == "implicit" else "solid"))
        forced += plot(label + " · solved physical trace", traces, "Physical time [ms]", label)
    if chosen.get("times") and chosen.get("fields"):
        available = [(k, label, unit, scale) for k, label, unit, scale in FIELDS if field(chosen, k).ndim == 2]
        if available:
            forced += '<h3>Inspect an instantaneous profile</h3>'
            if slider_selection:
                gap_text = '; '.join(f'{fmt(gap["from_time_s"]*frequency)}–{fmt(gap["to_time_s"]*frequency)} cycles' for gap in slider_gaps) if frequency else "the middle settling interval"
                forced += f'<p class="notice">The spatial slider retains the first three and last twenty-five cycles. Its gap is explicit: {gap_text}. Actual time and original snapshot index are shown. The boundary and modulation curves above retain the complete primary history.</p>'
            forced += '<div class="controls"><label>Field <select id="frame-field">' + "".join(f'<option value="{esc(k)}">{esc(label)} [{esc(unit)}]</option>' for k, label, unit, _ in available) + '</select></label><label>Time <input id="frame" type="range" min="0" max="' + str(len(slider_record["times"]) - 1) + '" value="0"><output id="frame-time"></output><output id="frame-raw-index"></output></label></div><div id="animation" class="plot"></div>'
    startup_window = visuals.get("startup_window_s")
    late_window = visuals.get("late_window_s")
    startup_caption = "Startup response. Time labels are physical; playback is slowed."
    if startup_window:
        startup_caption += f" Window: {fmt(startup_window[0])}–{fmt(startup_window[1])} s."
    forced += asset(visuals.get("startup_file", "startup.gif"), startup_caption, "startup-gif")
    late_file = visuals.get("late_file", "periodic.gif" if verified else "late_transient.gif")
    late_caption = "Verified periodic response." if verified else "Late valid transient, not periodic."
    if not verified and chosen.get("summary", {}).get("violation"):
        late_caption += " Frames stop before the recorded meridional-regime violation."
    if late_window:
        late_caption += f" Animation window: {fmt(late_window[0])}–{fmt(late_window[1])} s."
    collection_window = visuals.get("analysis_window_s", visuals.get("collection_window_s"))
    if late_window and collection_window and frequency:
        shown = (late_window[1]-late_window[0])*frequency
        collected = (collection_window[1]-collection_window[0])*frequency
        late_caption += f" The GIF shows {fmt(shown)} representative cycles out of the {fmt(collected)}-cycle " + ("verified collection" if verified else "finite window") + " used for the mean."
    forced += asset(late_file, late_caption, "periodic-gif")
    forced += '<p>Both animations use a fixed plasma scale when the visual metadata confirms a shared range. Each frame is a quasi-1D field extended over the physical duct, with no resolved radial or azimuthal variation.</p>'
    forced += details("All-time regime and boundary diagnostics", {name: r.get("summary", {}) for name, r in transient.items()})
    reflections = data.get("reflections", {})
    if reflections.get("probes"):
        forced += ('<h3>Acoustic components in the straight tail</h3>' if verified else '<h3>Finite-window harmonic diagnostic in the straight tail</h3>') + equation("acoustic_split")
        forced += '<p>The retained perturbations are split using the local mean density and sound speed. Fourier amplitudes and phases refer to the forcing frequency and the stated ' + ('periodic collection.' if verified else 'finite transient window; slow modulation can contribute to this diagnostic.') + ' This local linear diagnostic does not establish a universal duct or boundary reflection coefficient.</p>'
        probes = reflections["probes"]
        if isinstance(probes, dict):
            probes = [dict(id=name, **value) for name, value in probes.items()]
        forced += table(["Probe x [m]", "Outgoing amplitude [Pa]", "Returning amplitude [Pa]", "Amplitude ratio", "Phase difference [deg]"], [
            [fmt(v.get("x_m", v.get("x"))), fmt(v.get("outgoing_amplitude_Pa")), fmt(v.get("reflected_amplitude_Pa")),
             fmt(v.get("amplitude_ratio")), fmt(v.get("phase_difference_deg"))] for v in probes])
        forced += details("Retained acoustic-probe analysis", reflections)

    comparison = data.get("comparison", {})
    # Accept both a direct selected-case analysis and a map of analyses by case ID.
    if selected_id in comparison and isinstance(comparison[selected_id], dict):
        comparison = comparison[selected_id]
    cverified = comparison.get("verified_periodic", verified)
    mean_page = '<h2>Steady flow and the retained time average</h2><p>The instantaneous diagnostic fields are averaged directly; for example, the mean Mach number is not recomputed from a mean primitive state. The averaging interval and its periodicity status are stated explicitly.</p>'
    mean_page += equation("time_average")
    if not cverified:
        mean_page += '<p class="notice">Periodicity has not been established. The curves below compare the steady solution with a finite transient-window mean. They show the retained trends and cannot establish the cyclic steady-state effect.</p>'
    if comparison:
        mean_page += table(["Window start [s]", "Window end [s]", "Complete cycles", "Verified periodic"], [[fmt(comparison.get("window_start_s")), fmt(comparison.get("window_end_s")), fmt(comparison.get("complete_cycles")), fmt(cverified)]])
        mean_page += '<p>' + esc(comparison.get("quadrature", "The saved comparison defines the integration rule and selected phase samples.")) + '</p>'
        if cverified:
            mean_page += f'<p>The reported mean spans <b>{fmt(comparison.get("complete_cycles"))} complete, independently verified collection cycles</b>. The animation can show a shorter representative subset; it does not define the averaging window.</p>'
        else:
            start_cycle = comparison.get("window_start_s", 0)*frequency if frequency else None
            end_cycle = comparison.get("window_end_s", 0)*frequency if frequency else None
            mean_page += f'<p>The reported mean uses the final <b>{fmt(comparison.get("complete_cycles"))} complete forcing cycles, {fmt(start_cycle)}–{fmt(end_cycle)}</b>. It is a finite-time mean, not a periodic-state mean. The shorter GIF represents this interval without defining the averaging window.</p>'
        x = comparison.get("x", baseline.get("x", []))
        summary = []
        for key, label, unit, scale in FIELDS:
            packed = comparison.get("fields", {}).get(key, {})
            ss = packed.get("steady", comparison.get("steady", baseline.get("fields", {})).get(key))
            av = packed.get("mean", comparison.get("mean", {}).get(key))
            env = comparison.get("envelope", {}).get(key, {})
            low, high = packed.get("min", env.get("min")), packed.get("max", env.get("max"))
            if ss is None or av is None or len(ss) != len(x) or len(av) != len(x):
                continue
            ss, av = np.asarray(ss), np.asarray(av)
            traces = []
            if low is not None and high is not None:
                traces.extend([line(x, np.asarray(low) * scale, "Window minimum", "rgba(126,3,168,0)", showlegend=False),
                               line(x, np.asarray(high) * scale, "Window envelope", "rgba(126,3,168,0)", fill="tonexty", fillcolor="rgba(126,3,168,.13)")])
            traces.extend([line(x, ss * scale, "Steady baseline", COLORS[0], "dash"), line(x, av * scale, "Periodic mean" if cverified else "Finite transient mean", COLORS[2])])
            mean_page += plot(label + " · steady and " + ("periodic mean" if cverified else "finite-window mean"), traces, "Axial coordinate x [m]", unit)
            delta = av - ss
            percent = 100 * delta[-1] / ss[-1] if abs(ss[-1]) > 1e-15 else None
            summary.append([esc(label), fmt(ss[-1] * scale), fmt(av[-1] * scale), fmt(delta[-1] * scale), fmt(percent), fmt(np.sqrt(np.mean(delta * delta)) * scale)])
        mean_page += table(["Field", "Steady last cell", "Mean last cell", "Last-cell change", "Last-cell change [%]", "Spatial RMS change"], summary)
        sensitivity = comparison.get("window_sensitivity", {})
        sensitivity_rows = []
        for name, window in sensitivity.items():
            for key, label, unit in (("M_m", "Meridional Mach", "1"), ("v_m", "Meridional velocity", "m/s"), ("mdot", "Mass flow", "kg/s")):
                values = window.get("fields", {}).get(key)
                if values:
                    sensitivity_rows.append([fmt(window.get("cycles")), esc(label), esc(unit),
                        fmt(values.get("mean_last_cell")), fmt(values.get("last_cell_difference_from_ten_cycle_mean")),
                        fmt(values.get("spatial_RMS_difference_from_ten_cycle_mean"))])
        if sensitivity_rows:
            mean_page += '<h3>Sensitivity to the finite averaging window</h3><p>The following longer windows end at the same final time. Their differences from the ten-cycle mean measure sensitivity to the remaining modulation and evolving transient. A finite-window change is not a cyclic steady-state shift.</p>'
            mean_page += table(["Window [cycles]", "Field", "Unit", "Mean last cell", "Change from ten-cycle mean", "Spatial RMS change"], sensitivity_rows)
            mean_page += details("Retained averaging-window sensitivity", sensitivity)
        for key in ("summary", "temporal_mean_sensitivity", "temporal_errors", "quadrature_sensitivity", "matching_conditions"):
            if key in comparison:
                mean_page += details(key.replace("_", " ").capitalize(), comparison[key])
    else:
        mean_page += '<p class="notice">A time-average comparison has not been generated.</p>'
    mean_page += '<p>Both calculations retain the same reservoir conditions and constant exit pressure. A finite-window shift can include startup and slowly evolving inventory. Time-step and averaging-quadrature sensitivity must be checked before interpreting a persistent mean-flow effect.</p>'
    mean_page += f'<p>The steady analytical comparison also gives a scale for spatial discretization error. A small mean-flow shift on one {full_cells}-cell mesh is not a demonstrated physical unsteady effect when it is comparable with that spatial bias. Comparing both flows on the same mesh can cancel some error, but transient grid refinement is still needed to quantify the remainder.</p>'

    numerics = f'<h2>SSPRK3 time-step refinement</h2><p>The physical transient uses third-order SSPRK3 with HLL flux and van Leer MUSCL reconstruction on {full_cells} cells. The primary run covers the requested horizon from steady initial conditions. The shorter check reduces CFL from a saved state on the same mesh and physical boundary conditions. The final finite interval is compared at identical absolute times; it is not a second full startup campaign or a periodicity claim.</p>'
    refinement_record = transient.get(data.get("comparison_reference_id"), {})
    refinement_times = refinement_record.get("times", [])
    if refinement_times and primary_times and refinement_times[0] > primary_times[0]+1e-12:
        inherited = (refinement_times[0]-primary_times[0])*frequency if frequency else None
        checked = (refinement_times[-1]-refinement_times[0])*frequency if frequency else None
        numerics += f'<p class="notice">The finer-CFL restart inherits the primary trajectory through cycle {fmt(inherited)}. It measures the additional time-step difference over the following {fmt(checked)} cycles. It does <b>not independently bound accumulated temporal error in the first {fmt(inherited)} cycles</b> or verify the full primary horizon.</p>'
    temporal = data.get("temporal_comparison", {})
    if temporal.get("criterion"):
        numerics += '<p>' + esc(temporal["criterion"]) + '</p>'
    error_rows = []
    window_rows = []
    phase_averaged = False
    matched_transient = False
    for name, value in temporal.get("comparisons", {}).items():
        windows = value.get("windows", {})
        if not windows and value.get("fields"):
            windows = {"retained comparison": value}
        for window_name, window in windows.items():
            errors = window.get("fields", {})
            window_label = window_name
            if window_name == "matched_transient":
                matched_transient = True
                window_label = "Matched finite transient"
                now = [window.get("window_start_s"), window.get("window_end_s")]
                window_rows.append([esc(name), fmt(now[0]), fmt(now[-1]), fmt(now[0]), fmt(now[-1]), fmt(window.get("samples"))])
            if window.get("periodic_state_claimed"):
                phase_averaged = True
                window_label = f'{window.get("cycles", "Retained")}-cycle phase averages'
                now, ref = window.get("record_window_s", []), window.get("reference_window_s", [])
                window_rows.append([esc(name), fmt(now[0] if now else None), fmt(now[-1] if now else None),
                    fmt(ref[0] if ref else None), fmt(ref[-1] if ref else None), fmt(window.get("phase_samples"))])
            error_rows.append([esc(name), esc(window_label)] + [fmt(errors.get(key, {}).get("time_spatial_RMS_difference",
                errors.get(key, {}).get("phase_spatial_RMS_difference", errors.get(key, {}).get("RMS"))))
                for key in ("p", "M", "M_m", "T", "alpha_deg")])
    if error_rows:
        numerics += ('<h3>Matched transient signal differences</h3>' if matched_transient else '<h3>Retained matched-phase signal differences</h3>') + '<p>RMS across retained times and all cells, relative to ' + esc(temporal.get("reference_id", "the selected reference")) + '. '
        numerics += ('Both records are sampled at the same absolute times in the finite analysis interval. No phase averaging or periodic-state claim is made.' if matched_transient else 'Each waveform is phase-averaged across its independently verified collection. The primary and refinement windows can differ in absolute time; they share the same forcing phase.' if phase_averaged else 'Window identifiers refer to the exact retained analysis, including its periodicity status.') + '</p>'
        numerics += table(["Run", "Window", "Pressure [Pa]", "Total Mach", "Meridional Mach", "Temperature [K]", "Angle [deg]"], error_rows)
        if window_rows:
            numerics += table(["Run", "Record start [s]", "Record end [s]", "Reference start [s]", "Reference end [s]", "Saved samples / phases"], window_rows)
    error_checks = []
    for name, value in data.get("temporal_errors", {}).items():
        values = [v["normalized_RMS"] for v in value.values() if isinstance(v, dict) and "normalized_RMS" in v]
        if values and name != data.get("comparison_reference_id", selected_id):
            error_checks.append((name, max(values), bool(value.get("passed"))))
    if error_checks:
        numerics += '<h3>What the refinement checks support</h3><p>The following values are the largest retained normalized RMS differences, using the fixed scales and exact windows recorded with each check.</p>'
        numerics += '<p>' + '; '.join(esc(case_label(name, transient.get(name, {}))) + ' gives ' + fmt(value) + (' and passes its retained accuracy check' if passed else ' and fails its retained accuracy check') for name, value, passed in error_checks) + '.</p>'
        numerics += '<p>The selected SSPRK3 trajectory provides the reported mean; ' + esc(data.get("comparison_reference_id", selected_id)) + ' provides the time-step reference on the matched interval. Time-step agreement does not establish periodicity or remove spatial discretization error.</p>'
    timing_rows = []
    regime_rows = []
    for name, r in transient.items():
        history = r.get("history", [])
        accepted = [h for h in history if h.get("accepted", True)]
        dts = [h["dt"] for h in accepted if h.get("dt", 0) > 0]
        timing_rows.append([esc(name), esc(case_label(name, r)), fmt(r.get("success")), esc(r.get("reason", "")), fmt(r.get("elapsed_s", r.get("summary", {}).get("elapsed_s"))), fmt(len(accepted)), fmt(len(r.get("rejections", []))), fmt(min(dts) * 1e6 if dts else None), fmt(max(dts) * 1e6 if dts else None)])
        summary = r.get("summary", {})
        deceleration = all(v.get("expansion_meridional_ratio", float("inf")) < 1 for v in accepted) if accepted else None
        regime_rows.append([esc(name)] + [fmt(summary.get(key)) for key in ("inlet_M_min", "inlet_M_max", "inlet_M_m_max", "domain_M_m_max")] + [fmt(deceleration), fmt(summary.get("maximum_expansion_speed_ratio"))])
    numerics += table(["Run", "Method / CFL", "Requested run completed", "Stopping reason", "Elapsed [s]", "Accepted steps", "Rejected trials", "Min Δt [μs]", "Max Δt [μs]"], timing_rows)
    numerics += '<p>A stopped case is distinct from a failed nonlinear time step. The finite-volume updates can converge, satisfy their boundary equations and conserve the accumulated budgets while the evolving flow reaches the prescribed meridional-regime limit. The independent numerical and physical checks are retained separately.</p>'
    verification_rows = []
    for name, check in audit.get("transient", {}).items():
        verification_rows.append([esc(name), fmt(check.get("boundary_max")),
            fmt(check.get("outlet_pressure", {}).get("maximum_absolute_error_Pa")),
            fmt(check.get("conservation", {}).get("maximum_absolute_defect")),
            fmt(check.get("numerical_checks_passed")), fmt(check.get("requested_meridional_regime_passed"))])
    if verification_rows:
        numerics += table(["Run", "Scaled BC residual max", "Exit pressure error [Pa]", "Conservation defect max", "Numerical checks", "Meridional regime over full record"], verification_rows)
    numerics += '<p>These are validation-run costs for intervals of different length, not a repeated statistical benchmark. Shared-machine load and setup/JIT costs limit timing comparisons; the retained provenance records the execution protocol. No speed ranking is inferred from the primary campaign and its shorter refinement check.</p>'
    numerics += table(["Run", "Min inlet M", "Max inlet M", "Max inlet Mₘ", "Max domain Mₘ", "Meridional deceleration", "Max expansion speed ratio"], regime_rows)
    numerics += '<p>Regime summaries should include all accepted time steps and physical traces. The interactive plots show output samples; they cannot alone prove that no excursion occurred between frames.</p>'
    for key in ("time_step_refinement", "temporal_comparison", "temporal_errors", "timings", "recommendation"):
        if key in data:
            numerics += details(key.replace("_", " ").capitalize(), data[key])
    numerics += '<h3>Model and provenance</h3><p>This is inviscid, adiabatic quasi-1D annular Euler flow with swirl and a perfect-gas Air backend. It models a transition duct driven by a representative periodic load; it does not resolve detonation, chemistry, azimuthal waves, turbulence, or turbine impedance. The new boundary implementation and all outputs stay in this study folder. Concurrent source edits are recorded rather than described as this study’s changes.</p>'
    if frequency:
        numerics += f'<p>The chosen {fmt(frequency)} Hz forcing remains a representative wave-passing assumption, not a combustor-frequency prediction.</p>'
    numerics += '<button id="download-data">Download embedded numerical evidence (JSON.gz)</button>'
    numerics += details("Configuration", cfg) + details("Provenance", data.get("provenance", {}))
    numerics += '<p>Equations are prerendered from LaTeX with MathJax 3.2.2 as self-contained SVG paths. No math renderer, external font or network connection is needed when opening this report.</p>'
    numerics += '<details><summary>MathJax license (Apache-2.0)</summary><pre>' + esc((VENDOR/"MATHJAX_LICENSE.txt").read_text(encoding="utf-8")) + '</pre></details>'
    numerics += '<p class="muted">This report contains the angle-only, constant-exit-pressure case. Regenerating this presentation from its saved outputs does not execute the solver.</p>'
    bundle = compact_evidence(data, geometry, visuals, audit, data_dir=data_dir, digest=digest)
    bundle["math_rendering"] = dict(math_manifest, used_formulas=sorted(used_math))
    numerics += '<p>Plot data, settings and audits are embedded; full accepted-step and primitive evidence is retained in <code>outputs/diffuser/runs</code>. For long settling records the archive and spatial slider retain snapshots from the first three and last twenty-five cycles, with exact original indices and aligned times, fields, boundary values and inventories. Full sampled boundary histories and every cycle metric remain embedded. The manifest records the selection, raw artifact paths, SHA256 hashes and omitted bulk keys. Retained numerical values are unchanged, and the slider labels its time gap. The HTML needs no raw files to display its plots, animations or downloads.</p>'
    numerics += details("Raw numerical artifact manifest", bundle["raw_artifact_manifest"])

    chapters = [("geometry", "Geometry", geom), ("boundaries", "Boundary conditions", bc),
                ("steady", "Steady diffuser", steady), ("forced", "Forced response", forced),
                ("mean", "Steady vs mean", mean_page), ("numerics", "Accuracy & evidence", numerics)]
    vendor = STUDY / "scripts/vendor/plotly-2.35.2.min.js.gz"
    javascript = gzip.decompress(vendor.read_bytes()).decode("utf-8")
    archive = base64.b64encode(gzip.compress(json.dumps(bundle, separators=(",", ":")).encode(), mtime=0)).decode()
    # Pack the duplicated slider matrices as exact IEEE754 doubles. The full
    # downloadable archive above keeps every field series as ordinary JSON.
    frame_fields = {}
    for key, _, _, _ in FIELDS:
        values = np.asarray(slider_record.get("fields", {}).get(key, []), dtype="<f8")
        frame_fields[key] = dict(encoding="float64-le/base64", shape=list(values.shape),
            data=base64.b64encode(values.tobytes()).decode()) if values.size else []
    frame_data = dict(times=slider_record.get("times", []), x=chosen.get("x", []),
                      fields=frame_fields,
                      raw_indices=slider_indices, original_count=len(chosen.get("times", [])),
                      snapshot_selection=slider_selection, gaps=slider_gaps,
                      descriptors={key: dict(label=label, unit=unit, scale=scale) for key, label, unit, scale in FIELDS})
    specs = json.dumps(plots, separators=(",", ":")).replace("</", "<\\/")
    frames = json.dumps(frame_data, separators=(",", ":")).replace("</", "<\\/")
    doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Annular transition duct · angle forcing</title><style>' + CSS + '</style></head><body><header>' + title + '</header><main><nav aria-label="Report chapters">'
    doc += "".join(f'<button data-tab="{key}" class="{"active" if i == 0 else ""}">{label}</button>' for i, (key, label, _) in enumerate(chapters))
    doc += '<button id="theme" aria-label="Toggle color theme">◐ Theme</button></nav>'
    doc += "".join(f'<section id="{key}" class="{"active" if i == 0 else ""}">{content}</section>' for i, (key, _, content) in enumerate(chapters))
    doc += '<footer>Standalone numerical report · all plots, images and downloadable evidence are embedded. No local server or network connection is required.</footer></main><script>' + javascript + '</script><script>'
    doc += f'const plots={specs};const frames={frames};const archive="{archive}";' + SCRIPT + '</script></body></html>'
    return doc


CSS = """
:root{--bg:#f5f3f8;--card:#fff;--ink:#272034;--muted:#71677d;--line:#e5deec;--accent:#7e03a8;--note:#f3eaf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,Segoe UI,sans-serif}
body.dark{--bg:#14101c;--card:#211a2d;--ink:#f3edf8;--muted:#baadc7;--line:#45374f;--note:#2f203b}
header,main{max-width:1210px;margin:auto;padding:38px}header{padding-top:65px;padding-bottom:8px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:.18em;color:#b12a90}
h1{font-size:clamp(39px,6vw,77px);letter-spacing:-.065em;font-weight:730;line-height:1.03;margin:24px 0}h2{font-size:clamp(26px,4vw,35px);line-height:1.2;letter-spacing:-.04em;margin:36px 0 20px}h3{margin-top:34px;letter-spacing:-.02em}
p{max-width:1050px}.lead{font-size:20px;line-height:1.6;color:var(--muted);max-width:870px}.muted,footer{color:var(--muted)}.status{font-size:13px;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0 18px}.cards>div{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:20px}.cards b,.cards span{display:block}.cards b{font-size:21px;line-height:1.25;overflow-wrap:anywhere}.cards span{color:var(--muted);font-size:12px;margin-top:9px}
nav{display:flex;gap:7px;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);padding:14px 0;z-index:20;border-bottom:1px solid var(--line)}button,select{font:inherit;font-size:13px;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:22px;padding:10px 15px;cursor:pointer}button.active{background:var(--accent);color:#fff;border-color:var(--accent)}button:focus-visible,select:focus-visible{outline:3px solid #f89540;outline-offset:2px}#theme{margin-left:auto}
section{display:none}section.active{display:block}.plot{height:435px;background:var(--card);border:1px solid var(--line);border-radius:15px;overflow:hidden;margin:24px 0}.tablewrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:13px;margin:25px 0}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:12px 15px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:650}tbody tr:last-child td{border:0}
.notice{padding:17px 20px;background:var(--note);border-left:3px solid var(--accent);border-radius:0 10px 10px 0}.equation{padding:23px;background:var(--card);border:1px solid var(--line);border-radius:12px;text-align:center;font-size:18px;overflow:auto}.equation svg{max-width:100%;height:auto;vertical-align:middle}
figure{margin:26px 0;background:var(--card);padding:15px;border:1px solid var(--line);border-radius:15px}figure img{display:block;width:100%;height:auto;border-radius:9px}figcaption{color:var(--muted);font-size:13px;margin:12px 0}details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:16px 0}summary{cursor:pointer;font-size:14px;font-weight:650}pre{max-height:560px;overflow:auto;font-size:12px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}.controls{display:flex;gap:22px;align-items:center;padding:17px;background:var(--card);border-radius:12px;flex-wrap:wrap}.controls label{font-size:13px}.controls input{vertical-align:middle;width:220px}.controls output{margin-left:12px}footer{font-size:12px;border-top:1px solid var(--line);padding:30px 0;margin-top:42px}
@media(max-width:760px){header,main{padding:22px}header{padding-top:40px}.cards{grid-template-columns:repeat(2,1fr)}.cards b{font-size:17px}.plot{height:365px}#theme{margin-left:0}.equation{font-size:16px}.controls{display:block}.controls label{display:block;margin:9px 0}}
"""

SCRIPT = """
for(const [key,packed] of Object.entries(frames.fields)){if(packed?.encoding==='float64-le/base64'){const bytes=Uint8Array.from(atob(packed.data),c=>c.charCodeAt(0));const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);const [rows,cols]=packed.shape;frames.fields[key]=Array.from({length:rows},(_,i)=>Array.from({length:cols},(_,j)=>view.getFloat64(8*(i*cols+j),true)))}}
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function layout(base){const dark=document.body.classList.contains('dark');return {...base,paper_bgcolor:dark?'#211a2d':'#fff',plot_bgcolor:dark?'#211a2d':'#fff',font:{family:'system-ui',color:dark?'#f3edf8':'#272034'},margin:{t:80,b:65,l:75,r:85},legend:{orientation:'h',y:1.08}}}
function animate(){const slider=document.getElementById('frame');if(!slider)return;const i=+slider.value;const key=document.getElementById('frame-field').value;const info=frames.descriptors[key];const row=frames.fields[key]?.[i];if(!row)return;document.getElementById('frame-time').textContent=(frames.times[i]*1000).toFixed(4)+' ms';const indexLabel=document.getElementById('frame-raw-index');if(indexLabel)indexLabel.textContent=' · raw snapshot '+frames.raw_indices[i]+'/'+(frames.original_count-1)+(i>0&&frames.raw_indices[i]-frames.raw_indices[i-1]>1?' · after the omitted settling interval':'');let lo=Infinity,hi=-Infinity;for(const sample of frames.fields[key])for(const v of sample){lo=Math.min(lo,v);hi=Math.max(hi,v)}const pad=.04*(hi-lo||1);Plotly.react('animation',[{x:frames.x,y:row.map(v=>v*info.scale),mode:'lines',name:info.label,line:{color:'#cc4778',width:3}}],layout({title:{text:info.label+' · selected transient',x:.04},xaxis:{title:'Axial coordinate x [m]'},yaxis:{title:info.unit,range:[(lo-pad)*info.scale,(hi+pad)*info.scale]}}),config)}
function render(){for(const p of plots){const node=document.getElementById(p.id);if(node?.offsetParent!==null)Plotly.react(p.id,p.data,layout(p.layout),config)}if(document.getElementById('animation')?.offsetParent!==null)animate()}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('section').forEach(s=>s.classList.toggle('active',s.id===b.dataset.tab));document.querySelectorAll('[data-tab]').forEach(t=>t.classList.toggle('active',t===b));render()});
document.querySelectorAll('[data-save-image]').forEach(b=>b.onclick=()=>{const a=document.createElement('a');a.href=document.getElementById(b.dataset.saveImage).src;a.download=b.dataset.filename;a.click()});
document.getElementById('theme').onclick=()=>{document.body.classList.toggle('dark');render()};
document.getElementById('frame')?.addEventListener('input',animate);document.getElementById('frame-field')?.addEventListener('change',animate);
document.getElementById('download-data').onclick=()=>{const bytes=Uint8Array.from(atob(archive),c=>c.charCodeAt(0));const url=URL.createObjectURL(new Blob([bytes],{type:'application/gzip'}));const a=document.createElement('a');a.href=url;a.download='annular_diffuser_evidence.json.gz';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
"""


def build(data_dir=DATA, output=None):
    data_dir = Path(data_dir)
    path = data_dir / "results.json"
    data = load(path)
    if data is None:
        raise FileNotFoundError(f"No diffuser results: {path}")
    output = Path(output) if output else STUDY / "REPORT.html"
    text = make_report(data, load(data_dir / "geometry.json", {}),
                       load(data_dir / "visuals.json", {}), load(data_dir / "audit.json", {}),
                       data_dir=data_dir, digest=sha256(path.read_bytes()).hexdigest())
    output.write_text(text, encoding="utf-8")
    print(f"Offline diffuser report: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.data_dir, args.output)


if __name__ == "__main__":
    main()
