"""Build a single offline HTML report from saved evidence; no solver imports."""

import gzip
import base64
from hashlib import sha256
import html
import json
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
COLORS = ["#6a00a8", "#b12a90", "#e16462", "#fca636"]
LABELS = {"euler": "Explicit Euler", "rk3": "SSPRK3", "implicit": "Backward Euler"}


def build():
    d = json.loads((STUDY / "outputs/results.json").read_text())
    g = json.loads((STUDY / "outputs/geometry.json").read_text())
    results_digest = sha256((STUDY / "outputs/results.json").read_bytes()).hexdigest()
    cfg = d["config"]
    plots = []

    def plot(title, traces, xlabel, ylabel, **layout):
        axis_type = layout.pop("yaxis_type", "linear")
        ident = f"plot{len(plots)}"
        plots.append(
            dict(
                id=ident,
                data=traces,
                layout=dict(
                    title=title,
                    xaxis=dict(title=xlabel),
                    yaxis=dict(title=ylabel, type=axis_type),
                    **layout,
                ),
            )
        )
        return f'<div id="{ident}" class="plot"></div>'

    def line(x, y, name, color=None, **kw):
        return dict(
            x=np.asarray(x).tolist(),
            y=np.asarray(y).tolist(),
            name=name,
            mode="lines",
            line=dict(color=color or COLORS[0], width=2.5),
            **kw,
        )

    def table(headers, rows):
        return (
            '<div class="tablewrap"><table><thead><tr>'
            + "".join(f"<th>{h}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows
            )
            + "</tbody></table></div>"
        )

    def fmt(x):
        return f"{x:.3g}"

    def asset(relative, ident, caption):
        path = STUDY / relative
        mime = "image/gif" if path.suffix == ".gif" else "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f'<figure><img id="{ident}" class="artifact" src="data:{mime};base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption><button data-save-image="{ident}" data-filename="{path.name}">Download {path.name}</button></figure>'

    rows = []
    for kind in ("steady", "transient"):
        for name, v in d[kind].items():
            samples = v["samples"]
            tt = [s["elapsed_s"] for s in samples]
            rows.append(
                [
                    kind,
                    LABELS[name],
                    f"{sum(s['success'] for s in samples)} / {len(samples)}",
                    fmt(np.mean(tt)),
                    fmt(min(tt)),
                    fmt(max(tt)),
                    fmt(
                        sum(s["elapsed_s"] for s in v["warmup"])
                        if kind == "steady"
                        else v["warmup"]["elapsed_s"]
                    ),
                ]
            )
    allok = all(
        s["success"]
        for kind in ("steady", "transient")
        for v in d[kind].values()
        for s in v["samples"]
    )
    intro = f"""<div class="eyebrow">NOZZLE-HYPERBOLIC · NUMERICAL STUDY</div><h1>Annular transition duct</h1><p class="lead">Swirling supersonic flow, a retreating hub, and a periodically driven inlet inspired by a rotating detonation engine.</p><div class="cards"><div><b>0.8 &rarr; 0.7</b><span>hub / tip radius</span></div><div><b>20 bar · 1000 K</b><span>inlet stagnation targets</span></div><div><b>{cfg['forcing']['frequency_Hz']/1000:g} kHz</b><span>sinusoidal external forcing</span></div><div><b>{'PASS' if allok else 'CHECK FAILURES'}</b><span>all measured solver statuses</span></div></div>"""
    tiperr = float(
        np.max(np.abs(np.array(g["tip_r"]) - cfg["geometry"]["tip_radius_m"]))
    )
    geometry = "<h2>Geometry through midline and normal height</h2><p>The tip remains cylindrical. The first 0.10 m is constant, followed by a 0.20 m expansion. A 0.02 m junction blends the midline slope; the remaining hub is straight. The production geometry requires a continuous midline tangent, so the sharp corner is rounded explicitly. Lengths refer to axial midline coordinates; the normal inlet/outlet cross sections have slightly offset wall endpoints.</p>"
    geometry += plot(
        "Meridional section",
        [
            line(g["tip_x"], g["tip_r"], "Tip", COLORS[2]),
            line(g["hub_x"], g["hub_r"], "Hub", COLORS[0]),
            line(g["x"], g["r"], "Midline", COLORS[1]),
        ],
        "Axial coordinate x [m]",
        "Radius [m]",
    )
    geometry += plot(
        "Normal height and annular area",
        [
            line(g["m"], np.array(g["b"]) * 1000, "Normal height [mm]"),
            dict(
                **line(g["m"], np.array(g["A"]) * 1e4, "Area [cm²]", COLORS[2]),
                yaxis="y2",
            ),
        ],
        "Midline arclength m [m]",
        "Normal height [mm]",
        yaxis2=dict(title="Area [cm²]", overlaying="y", side="right"),
    )
    geometry += f'<p>The actual offset walls are x<sub>tip,hub</sub> = x ± br&prime;/2 and r<sub>tip,hub</sub> = r ± bx&prime;/2. Hence choose b = 2(R<sub>tip</sub> - r)/x&prime;, rather than treating normal height as radial gap. The height curve uses normalized midline arclength. Sampled maximum tip-radius error: <b>{tiperr:.3g} m</b>; endpoint hub/tip ratios: <b>{g["hub_r"][0]/g["tip_r"][0]:.8f}, {g["hub_r"][-1]/g["tip_r"][-1]:.8f}</b>. The area used by the solver is 2&pi;rb, normal to the midline.</p>'
    steady = "<h2>Steady solution · 50 &rarr; 100 &rarr; 200 cells</h2><p>HLL flux with van Leer MUSCL reconstruction. Every grid solves its original spatial equations; successful coarse solutions initialize the next grid by componentwise interpolation. This is grid sequencing, rather than a multigrid correction scheme.</p>"
    for field, title, unit, scale in [
        ("p", "Static pressure", "bar", 1e-5),
        ("M", "Total Mach number", "–", 1),
        ("alpha_deg", "Flow angle", "deg", 1),
        ("mdot", "Cell mass flow", "kg/s", 1),
    ]:
        steady += plot(
            title,
            [
                line(
                    v["representative"][-1]["x"],
                    np.array(v["representative"][-1]["fields"][field]) * scale,
                    LABELS[name],
                    COLORS[i],
                )
                for i, (name, v) in enumerate(d["steady"].items())
            ],
            "Axial coordinate x [m]",
            unit,
        )
    sr = []
    res = []
    for i, (name, v) in enumerate(d["steady"].items()):
        for sol in v["representative"]:
            mt = sol["metrics"]
            sr.append(
                [
                    LABELS[name],
                    sol["cells"],
                    str(sol["success"]),
                    sol["reason"],
                    fmt(max(mt["equation_norm"])),
                    fmt(mt["boundary_max"]),
                    fmt(max(mt["global_defect"])),
                ]
            )
        hist = v["representative"][-1]["history"]
        h = [r for r in hist if r["accepted"]]
        res.append(
            line(
                [r["step"] for r in h],
                [max(r["equation_norm"]) for r in h],
                LABELS[name],
                COLORS[i],
            )
        )
    steady += plot(
        "Final-grid residual · last measured trajectory",
        res,
        "Accepted step",
        "Maximum equation RMS",
        yaxis_type="log",
    )
    steady += table(
        [
            "Method",
            "Cells",
            "Converged",
            "Reason",
            "Equation RMS max",
            "BC max",
            "Global defect max",
        ],
        sr,
    )
    audit_path = STUDY / "outputs/audit.json"
    if (
        audit_path.exists()
        and json.loads(audit_path.read_text()).get("results_sha256") == results_digest
    ):
        audit = json.loads(audit_path.read_text())
        steady += "<h3>Independent saved-state audit and analytical comparison</h3><p>The original spatial residual is recomputed from the saved states in a separate process. A smooth-flow analytical reference independently conserves mass flow, angular momentum, stagnation enthalpy and entropy on the supersonic branch; it is used only for verification. Analytic differences measure spatial/model discretization error separately from root convergence.</p>"
        steady += table(
            [
                "Method",
                "Cells",
                "Independent residual pass",
                "RMS pressure error [Pa]",
                "RMS Mach error",
                "RMS angle error [deg]",
            ],
            [
                [
                    r["method"],
                    r["cells"],
                    str(r["passed"]),
                    fmt(r["analytic_RMS_errors"]["p"]),
                    fmt(r["analytic_RMS_errors"]["M"]),
                    fmt(r["analytic_RMS_errors"]["alpha_deg"]),
                ]
                for r in audit["steady"]
            ],
        )
        grids = [r for r in audit["steady"] if r["method"] == "implicit"]
        steady += plot(
            "Spatial refinement against smooth-flow reference",
            [
                line(
                    [r["cells"] for r in grids],
                    [
                        r["analytic_RMS_errors"]["p"]
                        / cfg["inlet"]["stagnation_pressure_Pa"]
                        for r in grids
                    ],
                    "RMS pressure / inlet p0",
                    COLORS[0],
                ),
                line(
                    [r["cells"] for r in grids],
                    [r["analytic_RMS_errors"]["M"] for r in grids],
                    "RMS Mach error",
                    COLORS[2],
                ),
            ],
            "Physical cells",
            "RMS error",
            yaxis_type="log",
        )
        exact = next(
            (
                r
                for r in audit["steady"]
                if r["method"] == "implicit" and r["cells"] == 200
            ),
            audit["steady"][-1],
        )
        steady += plot(
            "Smooth-flow analytical reference versus 200-cell solution",
            [
                line(
                    exact["x"],
                    np.array(exact["analytic"]["p"]) / 1e5,
                    "Analytical",
                    COLORS[3],
                )
            ]
            + [
                line(
                    v["representative"][-1]["x"],
                    np.array(v["representative"][-1]["fields"]["p"]) / 1e5,
                    LABELS[name],
                    COLORS[i],
                )
                for i, (name, v) in enumerate(d["steady"].items())
            ],
            "Axial coordinate x [m]",
            "Pressure [bar]",
        )
    reference = next(
        (
            v["representative"][-1]
            for name, v in d["steady"].items()
            if name == "implicit"
        ),
        next(iter(d["steady"].values()))["representative"][-1],
    )
    rootrows = []
    for name, v in d["steady"].items():
        f = v["representative"][-1]["fields"]
        ref = reference["fields"]
        rootrows.append(
            [
                LABELS[name],
                fmt(
                    np.max(np.abs(np.array(f["p"]) - ref["p"]))
                    / cfg["inlet"]["stagnation_pressure_Pa"]
                ),
                fmt(np.max(np.abs(np.array(f["M"]) - ref["M"]))),
                fmt(np.ptp(f["mdot"]) / np.mean(f["mdot"])),
            ]
        )
    steady += table(
        [
            "Method",
            "Max delta p / p0 vs implicit",
            "Max delta M vs implicit",
            "Cell mass-flow range / mean",
        ],
        rootrows,
    )
    steady += "<p>Cell-centred &rho;vA is a reconstruction diagnostic, not the conservative face flux. Convergence uses the original integrated residual divided by cell volume and fixed conservative scales, multiplied by reference time, then a volume-weighted equation RMS. Each equation must pass relative 10<sup>-6</sup> OR absolute 10<sup>-9</sup>; boundary and global defects independently require 10<sup>-9</sup>. Explicit steady CFL is capped at 0.4; backward Euler grows from 0.2 to at most 10<sup>6</sup>.</p>"
    transient = "<h2>Forced transient · fixed 200-cell grid</h2><p>The converged 200-cell steady solution initializes every transient. External M = 1.2 + 0.3 sin(2&pi;ft) and &alpha; = 30° + 10° sin(2&pi;ft), in phase. Stagnation targets remain 20 bar and 1000 K. Eight periods are simulated; this finite run is not a demonstrated periodic limit cycle.</p>"
    errrows = []
    for field, title, unit, scale in [
        ("p", "Outlet-cell pressure", "bar", 1e-5),
        ("M", "Outlet-cell total Mach number", "–", 1),
        ("alpha_deg", "Outlet-cell flow angle", "deg", 1),
    ]:
        traces = [
            line(
                np.array(v["times"]) * 1e3,
                np.array(v["fields"][field])[:, -1] * scale,
                LABELS[name],
                COLORS[i],
            )
            for i, (name, v) in enumerate(d["transient"].items())
        ]
        if traces:
            transient += plot(title, traces, "Time [ms]", unit)
    if d["transient"]:
        first = next(iter(d["transient"].values()))
        t = np.array(first["times"])
        f = cfg["forcing"]["frequency_Hz"]
        gamma = 1.4016904909149628
        R = 287.04739368889705
        cp = gamma * R / (gamma - 1)
        traces = [
            line(
                t * 1e3,
                1.2 + 0.3 * np.sin(2 * np.pi * f * t),
                "External Mach target",
                COLORS[3],
            )
        ]
        p0tr = [line(t * 1e3, np.full(len(t), 20.0), "External p0 target", COLORS[3])]
        angletr = [
            line(
                t * 1e3,
                30 + 10 * np.sin(2 * np.pi * f * t),
                "External angle target",
                COLORS[3],
            )
        ]
        temptr = [
            line(t * 1e3, np.full(len(t), 1000.0), "External T0 target", COLORS[3])
        ]
        for i, (name, v) in enumerate(d["transient"].items()):
            z = np.array(v.get("inlet_interior_state", np.array(v["states"])[:, 1, :]))
            T = z[:, 3] / cp
            a = np.sqrt(gamma * R * T)
            speed = np.hypot(z[:, 1], z[:, 2] / g["r"][0])
            M = speed / a
            traces.append(
                line(
                    np.array(v["times"]) * 1e3,
                    M,
                    LABELS[name] + " duct-side state",
                    COLORS[i],
                )
            )
            p0 = z[:, 0] * (1 + (gamma - 1) * M * M / 2) ** (gamma / (gamma - 1))
            p0tr.append(
                line(np.array(v["times"]) * 1e3, p0 / 1e5, LABELS[name], COLORS[i])
            )
            angletr.append(
                line(
                    np.array(v["times"]) * 1e3,
                    np.rad2deg(np.arctan2(z[:, 2] / g["r"][0], z[:, 1])),
                    LABELS[name],
                    COLORS[i],
                )
            )
            temptr.append(
                line(
                    np.array(v["times"]) * 1e3,
                    T * (1 + (gamma - 1) * M * M / 2),
                    LABELS[name],
                    COLORS[i],
                )
            )
        transient += plot(
            "External target versus duct-side inlet Mach",
            traces,
            "Time [ms]",
            "Total Mach number",
        )
        transient += plot(
            "External target versus duct-side inlet angle",
            angletr,
            "Time [ms]",
            "Flow angle [deg]",
        )
        transient += plot(
            "Duct-side inlet stagnation temperature", temptr, "Time [ms]", "T0 [K]"
        )
        transient += plot(
            "Duct-side inlet stagnation pressure", p0tr, "Time [ms]", "p0 [bar]"
        )
        transient += (
            '<h3>Explore the pressure field</h3><div class="slider"><label>Output time <input id="frame" type="range" min="0" max="'
            + str(len(first["times"]) - 1)
            + '" value="0"><output id="frame-time"></output></label></div><div id="animation" class="plot"></div>'
        )
        for name, v in d["transient"].items():
            ref = d.get("reference", {})
            errors = []
            if ref.get("success") and len(ref["times"]) == len(v["times"]):
                for field, scale in [
                    ("p", cfg["inlet"]["stagnation_pressure_Pa"]),
                    ("M", 1),
                    ("alpha_deg", 1),
                ]:
                    diff = (np.array(v["fields"][field]) - ref["fields"][field]) / scale
                    errors.append(
                        fmt(
                            np.sqrt(
                                np.mean(
                                    diff[-cfg["forcing"]["frames_per_cycle"] :] ** 2
                                )
                            )
                        )
                    )
            else:
                errors = ["unavailable"] * 3
            budget = max(
                max(abs(np.array(m["conservation_budget"]))) for m in v["metrics"]
            )
            errrows.append(
                [
                    LABELS[name],
                    v["cfl"],
                    *errors,
                    fmt(budget),
                    max(m["boundary_max"] for m in v["metrics"]),
                ]
            )
        transient += table(
            [
                "Method",
                "CFL",
                "Last-cycle RMS delta p/p0",
                "RMS delta M",
                "RMS delta angle [deg]",
                "Max normalized budget defect",
                "Max BC defect",
            ],
            errrows,
        )
        transient += "<p>The temporal comparison uses SSPRK3 at CFL 0.2 on the same 200-cell mesh; it estimates temporal sensitivity only, and is not an exact physical reference. Method differences include both temporal damping and timestep size. Backward Euler CFL 2 is five times the explicit CFL; stability does not establish time accuracy.</p>"
    visuals_path = STUDY / "outputs/visuals.json"
    if (
        visuals_path.exists()
        and json.loads(visuals_path.read_text()).get("results_sha256") == results_digest
    ):
        visuals = json.loads(visuals_path.read_text())
        steady += "<h3>Plasma contours in the actual meridional duct section</h3><p>These contours extend the quasi-one-dimensional cell solution uniformly across the channel height. The outlines show the actual hub and tip. They visualize the computed streamwise solution and do not imply resolved transverse gradients.</p>"
        steady += asset(
            visuals["steady_file"],
            "steady-contours",
            "Steady Mach number, static pressure and flow angle in the annular transition duct.",
        )
        transient += "<h3>Startup and periodic-state animations</h3><p>The benchmark integrates eight periods, ending at 2.286 ms. Periodicity is checked independently by comparing all four primitive variables across every physical cell at 24 aligned phases. The largest normalized RMS must be below 0.001 for two consecutive cycle comparisons. If needed, a separate RK3 extension continues for whole periods; the original timing window remains unchanged.</p>"
        periodic = json.loads((STUDY / "outputs/periodicity.json").read_text())
        transient += table(
            [
                "Method",
                "Periodic at its final time",
                "Last max cycle difference",
                "End time [ms]",
            ],
            [
                [
                    name,
                    str(value["verified"]),
                    fmt(value["cycles"][-1]["maximum"]),
                    fmt(value["end_time_s"] * 1000),
                ]
                for name, value in periodic.items()
            ],
        )
        transient += plot(
            "Cycle-to-cycle repeatability",
            [
                line(
                    [r["cycle"] for r in value["cycles"]],
                    [r["maximum"] for r in value["cycles"]],
                    name,
                    COLORS[i % len(COLORS)],
                )
                for i, (name, value) in enumerate(periodic.items())
            ],
            "Forcing cycle",
            "Max normalized primitive RMS difference",
            yaxis_type="log",
        )
        transient += f"<p>The animation trajectory ends after <b>{visuals['cycles']} cycles ({visuals['end_time_s']*1000:.3f} ms)</b>. Its final two cycle comparisons satisfy the periodicity test: <b>{visuals['periodic_verified']}</b>. This demonstrates repeatability at the stated threshold, rather than exact asymptotic periodicity. The startup GIF shows the first three periods; the late GIF shows the final two. Both share fixed plasma color limits. Playback is slowed; time labels show physical time.</p>"
        transient += asset(
            visuals["startup_file"],
            "startup-animation",
            "Startup from the converged steady solution: first three forcing cycles. The loop deliberately restarts at t=0.",
        )
        transient += asset(
            visuals["late_file"],
            "periodic-animation",
            "Final two forcing cycles; periodicity status is given above. Quasi-1D Mach and pressure rendered inside the duct geometry.",
        )
    mean_page = "<h2>Steady flow versus the periodic time average</h2>"
    mean_path = STUDY / "outputs/mean_comparison.json"
    if (
        mean_path.exists()
        and json.loads(mean_path.read_text()).get("results_sha256") == results_digest
    ):
        averaged = json.loads(mean_path.read_text())
        mean_page += f"<p>The reference is the converged steady solution at Mach 1.2 and angle 30 degrees. The unsteady mean uses the final <b>two complete RK3 cycles</b>, from <b>{averaged['window_start_s']*1000:.3f} to {averaged['window_end_s']*1000:.3f} ms</b>. Periodicity verified at the stated threshold: <b>{averaged['verified_periodic']}</b>.</p>"
        mean_page += (
            "<p>"
            + averaged["quadrature"]
            + " Shading below shows the instantaneous range over the averaging window.</p>"
        )
        for field, title, unit, scale in [
            ("p", "Static pressure", "bar", 1e-5),
            ("M", "Total Mach number", "-", 1),
            ("T", "Static temperature", "K", 1),
            ("alpha_deg", "Flow angle", "deg", 1),
            ("mdot", "Mass flow", "kg/s", 1),
        ]:
            x = averaged["x"]
            band = averaged["envelope"][field]
            low = line(
                x, np.array(band["min"]) * scale, "Instantaneous minimum", COLORS[2]
            )
            low["showlegend"] = False
            low["line"]["width"] = 0
            high = line(
                x,
                np.array(band["max"]) * scale,
                "Periodic oscillation range",
                COLORS[2],
            )
            high.update(fill="tonexty", fillcolor="rgba(225,100,98,0.16)")
            high["line"]["width"] = 0
            mean_page += plot(
                title + ": steady and periodic mean",
                [
                    low,
                    high,
                    line(
                        x,
                        np.array(averaged["steady"][field]) * scale,
                        "Steady baseline",
                        COLORS[0],
                    ),
                    line(
                        x,
                        np.array(averaged["mean"][field]) * scale,
                        "Periodic time mean",
                        COLORS[2],
                    ),
                ],
                "Axial coordinate x [m]",
                unit,
            )
        mean_rows = []
        for field, label, scale in [
            ("p", "Pressure [bar]", 1e-5),
            ("M", "Mach number", 1),
            ("T", "Temperature [K]", 1),
            ("alpha_deg", "Angle [deg]", 1),
            ("mdot", "Mass flow [kg/s]", 1),
        ]:
            r = averaged["summary"][field]
            mean_rows.append(
                [
                    label,
                    fmt(r["steady_outlet"] * scale),
                    fmt(r["mean_outlet"] * scale),
                    fmt(r["outlet_difference"] * scale),
                    fmt(r["maximum_absolute_difference"] * scale),
                    (
                        fmt(r.get("outlet_percent", float("nan")))
                        if "outlet_percent" in r
                        else "n/a"
                    ),
                ]
            )
        mean_page += table(
            [
                "Quantity",
                "Steady outlet cell",
                "Periodic mean outlet cell",
                "Outlet difference",
                "Max local absolute difference",
                "Outlet difference [%]",
            ],
            mean_rows,
        )
        p = averaged["summary"]["p"]
        mass = averaged["summary"]["mdot"]
        mach = averaged["summary"]["M"]
        mean_page += f"<p>For this 200-cell model, the periodic mean outlet pressure differs from the steady baseline by <b>{p['outlet_percent']:+.3f}%</b>, mean outlet mass flow by <b>{mass['outlet_percent']:+.3f}%</b>, and mean outlet Mach by <b>{mach['outlet_difference']:+.4f}</b>. The maximum local pressure difference is <b>{p['maximum_percent']:.3f}%</b>.</p>"
        sensitivities = averaged["temporal_mean_sensitivity"]
        if sensitivities:
            mean_page += "<h3>Temporal sensitivity of the mean</h3>" + table(
                ["Quantity", "RMS change on halving CFL", "Max local change"],
                [
                    [
                        field,
                        fmt(value["RMS_difference"]),
                        fmt(value["maximum_absolute_difference"]),
                    ]
                    for field, value in sensitivities.items()
                ],
            )
            mean_page += table(
                ["Quantity", "RMS change: 12 to 24 phases per cycle"],
                [
                    [
                        field,
                        fmt(
                            averaged["summary"][field][
                                "averaging_quadrature_RMS_change"
                            ]
                        ),
                    ]
                    for field in ("p", "M", "alpha_deg")
                ],
            )
            mean_page += (
                "<p>Units are Pa for pressure, dimensionless for Mach, and degrees for angle. "
                + averaged["temporal_sensitivity_window"]
                + " These differences estimate timestep sensitivity; unsteady spatial convergence has not been established.</p>"
            )
        if sensitivities and averaged["verified_periodic"]:
            uncertainty = max(
                sensitivities["p"]["RMS_difference"],
                p["averaging_quadrature_RMS_change"],
            )
            evidence = (
                "exceeds five times the tested timestep and averaging-quadrature sensitivities"
                if p["RMS_difference"] > 5 * uncertainty
                else "is comparable to the tested numerical sensitivities and needs further refinement for a firm interpretation"
            )
            mean_page += f"<p>The spatial RMS pressure shift in the mean is {p['RMS_difference']:.3g} Pa; it <b>{evidence}</b>. The timestep comparison uses independently verified periodic windows at both CFL values.</p>"
        inlet_steady = averaged["steady"]["mdot"][0]
        inlet_mean = averaged["mean"]["mdot"][0]
        mean_page += f"<h3>Interpretation</h3><p>The mean inlet-cell mass-flow diagnostic changes by <b>{100*(inlet_mean-inlet_steady)/inlet_steady:+.3f}%</b>. The boundary conversion and Euler fluxes are nonlinear: mean Mach and mean angle equal their steady settings, while the mean mass, momentum and energy fluxes need not equal steady fluxes evaluated at those settings. The measured differences therefore include finite-amplitude boundary averaging, wave transport and numerical dissipation; they should not be attributed solely to storage terms. The quasi-1D contours represent the same computed fields used in this comparison.</p>"
    else:
        mean_page += "<p>Generate verified-cycle averages with the visuals script after the transient campaign.</p>"
    timing = "<h2>Measured time to solution</h2>" + table(
        [
            "Problem",
            "Method",
            "Successes",
            "Mean [s]",
            "Min [s]",
            "Max [s]",
            "Excluded solve [s]",
        ],
        rows,
    )
    step_rows = []
    for name, v in d["transient"].items():
        accepted = [r for r in v["history"] if r["accepted"]]
        step_rows.append(
            [
                LABELS[name],
                v["cfl"],
                v["samples"][-1]["accepted_steps"],
                v["samples"][-1]["rejected_steps"],
                fmt(min(r["dt"] for r in accepted) * 1e6),
                fmt(max(r["dt"] for r in accepted) * 1e6),
                fmt(max(r["cfl"] for r in accepted)),
            ]
        )
    if step_rows:
        timing += table(
            [
                "Transient method",
                "Nominal CFL",
                "Accepted steps",
                "Rejected trials",
                "Min dt [microseconds]",
                "Max dt [microseconds]",
                "Max actual CFL",
            ],
            step_rows,
        )
    timing += (
        "<p>"
        + html.escape(d["timing_protocol"])
        + "</p><p>Each method has its own complete excluded solve, followed by "
        + str(len(next(iter(d["steady"].values()))["samples"]))
        + " measured solves. Fields, controller state and initial conditions reset for every solve. Timings include JAX synchronization through host monitoring. Flow-step kernels are reused; the unmodified driver creates monitoring reducers per solve, and that setup remains included. Methods run sequentially, reversing order on alternate repetitions; system-load bias remains possible. Excluded times include first compilation and should not be used as warmed solver costs.</p>"
    )
    recommendation = "<h3>Recommendation</h3>"
    steady_agreement = all(
        np.max(
            np.abs(
                np.array(v["representative"][-1]["fields"]["p"])
                - reference["fields"]["p"]
            )
        )
        / cfg["inlet"]["stagnation_pressure_Pa"]
        < 1e-6
        and np.max(
            np.abs(
                np.array(v["representative"][-1]["fields"]["M"])
                - reference["fields"]["M"]
            )
        )
        < 1e-5
        for v in d["steady"].values()
    )
    if allok and steady_agreement:
        fastest = min(
            d["steady"],
            key=lambda n: np.mean([s["elapsed_s"] for s in d["steady"][n]["samples"]]),
        )
        recommendation += f"<p>For the steady root, use <b>{LABELS[fastest]}</b> with 50 &rarr; 100 &rarr; 200 grid sequencing: it has the smallest measured mean among the converged methods here. For time-dependent calculations, use <b>SSPRK3 at CFL 0.4</b> as the starting choice, then halve CFL to assess temporal convergence. Explicit Euler and backward Euler are first order in time; compare their measured phase and amplitude errors with the required signal accuracy before choosing them for their cost. There is no implicit RK3 method in the current package.</p>"
    else:
        recommendation += "<p>Some measured cases failed or the final steady fields disagree beyond the comparison threshold. Retain the diagnostics and do not infer a time-to-solution speedup without matching roots. Use the successful cases and the temporal-refinement comparison to guide the next numerical investigation.</p>"
    timing += recommendation
    theory = """<h2>Boundary conditions and interpretation</h2><p>This is an inviscid, adiabatic, quasi-one-dimensional annular Euler model with angular momentum and a calorically perfect Air backend. It represents transition-duct response to an imposed periodic signal. It does not resolve azimuthal detonations, chemistry, turbulent losses or a combustor.</p><h3>Why the fluctuating inlet needs characteristic matching</h3><p>The relevant sonic quantity is the meridional Mach number M cos &alpha;, rather than total Mach M. The baseline value is 1.039; the forcing crosses unity. Supersonic inflow has four incoming characteristics, so the complete external primitive state is prescribed. Subsonic inflow has three incoming characteristics and one outgoing characteristic. Imposing both stagnation quantities, Mach and angle there would overconstrain the Euler boundary.</p><p>The study-local model prescribes all four quantities in an external Riemann state. The production HLL flux couples that external state to the MUSCL-reconstructed duct-side state at the inlet. Its two-sided wave-speed bounds account for information entering and leaving the duct during subsonic operation, without imposing four conditions on the physical interface trace. The algebraic endpoint therefore represents the external state, rather than the physical interface state in subsonic flow. The report compares targets with the reconstructed duct-side inlet state; neither is an exact resolved Riemann trace. Flow reversal is audited as failure. A preliminary hard-switch characteristic inlet stalled near the sonic transition at 0.202 ms; its failed diagnostics are preserved in outputs/failed_characteristic_pilot/ and excluded from final timings.</p><h3>Low-pressure automatic outflow</h3><p>The automatic production acoustic outlet uses a 0.10 bar ambient pressure only while information can enter from downstream. Sonic and supersonic outflow use outgoing-characteristic logic. Thus ambient pressure is not forced onto a supersonic exit, and the internal exit pressure can differ from ambient. Here the approximately 3 bar internal exit pressure is above the 0.10 bar ambient; external underexpanded-jet adjustment lies outside this duct model. The expansion of a supersonic stream generally accelerates it; calling the geometry a diffusion channel does not imply aerodynamic diffusion.</p><h3>Frequency choice</h3><p>At a mean radius of 0.09 m, a single wave moving at approximately 2 km/s gives f = D / (2&pi;r) ˜ 3.54 kHz. We select 3.5 kHz as a plausible representative value, not a prediction for a particular combustor. Multiple waves multiply the wave-passing frequency. <a href="https://www.dlr.de/en/ra/about-us/departments/rocket-propulsion-technology/combustion-dynamics/rotating-detonation-engines">DLR</a> describes kilometre-per-second detonation speeds and kilohertz wave circulation; <a href="https://www.nasa.gov/wp-content/uploads/2024/01/2024sbirphaseisolicitationfinal.pdf?emrc=685eba6fd9c6c">NASA’s 2024 SBIR solicitation</a> notes that multiple co-rotating waves can produce composite frequencies of 10 kHz or more.</p>"""
    theory += (
        "<h3>Reproduce and inspect</h3><pre>poetry run python -B -m development.rotating_detonation.run_case\npoetry run python -B -m development.rotating_detonation.run_case --report-only</pre><p>The existing .venv/Scripts/python.exe can replace “poetry run python” if Poetry is absent from PATH. All files are written inside development/rotating_detonation/. Production source hashes before and after agree: <b>"
        + str(d.get("source_unchanged", "campaign not yet finished"))
        + '</b>. Pre-existing repository edits are retained.</p><button id="download">Download embedded campaign data (JSON.gz)</button><details><summary>Configuration and environment</summary><pre>'
        + html.escape(
            json.dumps(dict(config=cfg, environment=d["environment"]), indent=2)
        )
        + "</pre></details>"
    )
    if not d.get("source_unchanged", True):
        provenance = (
            json.loads((STUDY / "outputs/audit.json").read_text())
            if (STUDY / "outputs/audit.json").exists()
            else {}
        )
        theory += (
            "<h3>Concurrent source changes detected</h3><p>The before/after source manifests differ because files on disk changed during the long benchmark. This study writes only inside its own directory. The changed paths concern Osher selectors/configuration and the shared operator; the benchmark process had already imported its modules. Retained HLL steady states are independently checked against the current spatial equations, and selected supersonic, subsonic and late-time output intervals are replayed under the current checkout with every integrator. HLL compatibility audit passed: <b>"
            + str(provenance.get("current_hll_compatibility_passed", "pending"))
            + "</b>. The original before/after hashes and changed-path list remain in the evidence. A claim that the entire checkout stayed unchanged would be incorrect.</p>"
        )
    vendor = (
        ROOT
        / "development/roberto/benchmarking/scripts/report_assets/vendor/plotly-2.35.2.min.js.gz"
    )
    js = gzip.decompress(vendor.read_bytes()).decode()
    compact = {
        "transient": {
            name: {"times": v["times"], "x": v["x"], "fields": {"p": v["fields"]["p"]}}
            for name, v in d["transient"].items()
        }
    }
    payload = json.dumps(compact, separators=(",", ":")).replace("</", "<\\/")
    embedded = dict(d)
    embedded["report_analysis"] = {}
    for name in ("audit", "periodicity", "mean_comparison", "visuals"):
        path = STUDY / "outputs" / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if name == "periodicity" or data.get("results_sha256") == results_digest:
                embedded["report_analysis"][name] = data
    archive = base64.b64encode(
        gzip.compress(json.dumps(embedded, separators=(",", ":")).encode(), mtime=0)
    ).decode()
    specs = json.dumps(plots, separators=(",", ":")).replace("</", "<\\/")
    doc = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Annular transition duct · Rotating detonation study</title><style>
:root{--bg:#f5f3f8;--card:#fff;--ink:#252036;--muted:#6e677c;--accent:#6a00a8;--line:#e4dfeb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,sans-serif}body.dark{--bg:#14121d;--card:#211c2d;--ink:#f1edf6;--muted:#b7adc5;--line:#44394f}header,main{max-width:1200px;margin:auto;padding:36px}header{padding-bottom:16px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.18em;color:#b12a90}h1{font-size:clamp(36px,5vw,66px);letter-spacing:-.055em;line-height:1.1;margin:16px 0}h2{font-size:30px;letter-spacing:-.035em}h3{margin-top:34px}.lead{font-size:20px;color:var(--muted);max-width:850px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}.cards div{background:var(--card);padding:22px;border:1px solid var(--line);border-radius:15px}.cards b,.cards span{display:block}.cards b{font-size:22px}.cards span{font-size:13px;color:var(--muted)}nav{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);padding:16px 0;z-index:10;border-bottom:1px solid var(--line)}button{background:var(--card);color:var(--ink);border:1px solid var(--line);padding:10px 17px;border-radius:24px;cursor:pointer;font:inherit;font-size:14px}button.active{background:#6a00a8;color:white}section{display:none}section.active{display:block}.plot{height:420px;background:var(--card);border:1px solid var(--line);border-radius:15px;margin:24px 0;overflow:hidden}.artifact{width:100%;height:auto;display:block;border-radius:12px}figure{margin:28px 0;background:var(--card);padding:16px;border:1px solid var(--line);border-radius:15px}figcaption{font-size:13px;color:var(--muted);margin:12px 0}.tablewrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:13px 16px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}pre{background:var(--card);border:1px solid var(--line);padding:20px;overflow:auto;border-radius:12px;font-size:13px}a{color:#c65ba8}.slider{background:var(--card);padding:18px;border-radius:12px}input{width:55%;vertical-align:middle}footer{color:var(--muted);font-size:12px;padding:25px 0}@media(max-width:720px){header,main{padding:20px}.cards{grid-template-columns:repeat(2,1fr)}.cards b{font-size:17px}.plot{height:360px}}</style></head><body>"""
    doc += (
        "<header>"
        + intro
        + "</header><main><nav>"
        + "".join(
            f'<button data-tab="{key}" class="'
            + ("active" if key == "geometry" else "")
            + '">'
            + label
            + "</button>"
            for key, label in [
                ("geometry", "Geometry"),
                ("steady", "Steady flow"),
                ("transient", "Forced response"),
                ("mean", "Steady vs mean"),
                ("timing", "Timings"),
                ("theory", "Model & reproduction"),
            ]
        )
        + '<button id="theme">Theme</button></nav>'
    )
    for key, content in [
        ("geometry", geometry),
        ("steady", steady),
        ("transient", transient),
        ("mean", mean_page),
        ("timing", timing),
        ("theory", theory),
    ]:
        doc += (
            f'<section id="{key}" class="'
            + ("active" if key == "geometry" else "")
            + '">'
            + content
            + "</section>"
        )
    doc += (
        "<footer>Generated from retained numerical evidence. All plots, scripts and downloadable data are embedded; no server or internet connection required.</footer></main><script>"
        + js
        + "</script><script>const campaign="
        + payload
        + ";const archive="
        ""
        + '"'
        + archive
        + '"'
        + """;const plots="""
        + specs
        + """;
const config={responsive:true,displaylogo:false};
function layout(base){const dark=document.body.classList.contains('dark');return {...base,paper_bgcolor:dark?'#211c2d':'#fff',plot_bgcolor:dark?'#211c2d':'#fff',font:{family:'system-ui',color:dark?'#f1edf6':'#252036'},margin:{t:60,b:60,l:75,r:80},legend:{orientation:'h',y:1.15}}}
function render(){for(const p of plots){if(document.getElementById(p.id).offsetParent!==null)Plotly.react(p.id,p.data,layout(p.layout),config)}if(document.getElementById('animation') && document.getElementById('animation').offsetParent!==null)animate()}
function animate(){const i=+document.getElementById('frame').value;const traces=Object.entries(campaign.transient).map(([name,v],k)=>({x:v.x,y:v.fields.p[i].map(p=>p/1e5),name:({euler:'Explicit Euler',rk3:'SSPRK3',implicit:'Backward Euler'})[name],mode:'lines',line:{color:['#6a00a8','#b12a90','#e16462'][k]}}));const t=Object.values(campaign.transient)[0].times[i];document.getElementById('frame-time').textContent=(t*1000).toFixed(3)+' ms';Plotly.react('animation',traces,layout({title:'Instantaneous pressure',xaxis:{title:'Axial coordinate x [m]'},yaxis:{title:'Pressure [bar]'}}),config)}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('section').forEach(s=>s.classList.toggle('active',s.id===b.dataset.tab));document.querySelectorAll('[data-tab]').forEach(t=>t.classList.toggle('active',t===b));render()});
document.querySelectorAll('[data-save-image]').forEach(b=>b.onclick=()=>{const a=document.createElement('a');a.href=document.getElementById(b.dataset.saveImage).src;a.download=b.dataset.filename;a.click()});
document.getElementById('theme').onclick=()=>{document.body.classList.toggle('dark');render()};document.getElementById('frame')?.addEventListener('input',animate);
document.getElementById('download').onclick=()=>{const bytes=Uint8Array.from(atob(archive),c=>c.charCodeAt(0));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([bytes],{type:'application/gzip'}));a.download='rotating_detonation_results.json.gz';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></body></html>"""
    )
    (STUDY / "REPORT.html").write_text(doc, encoding="utf-8")
    print("Report:", STUDY / "REPORT.html", flush=True)


if __name__ == "__main__":
    build()
