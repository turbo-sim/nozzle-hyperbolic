"""Conservation, branch, and independent smooth-primitive checks.

This audit uses returned states rather than nonlinear-solver success flags.
Failures are returned in ``failed_checks`` so the driver can save the complete
evidence before raising. Entropy and tabulated acoustic inconsistencies are
reported separately from conservation of the density-defined Euler system.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


THRESHOLDS = {
    "mass_flow_relative": 1e-9,
    "total_enthalpy_relative": 1e-12,
    "rankine_hugoniot_relative": 1e-9,
    "throat_mach_absolute": 1e-8,
    "backpressure_relative": 1e-8,
    "smooth_primitive_relative_to_h0": 1e-7,
    "branch_mach_allowance": 1e-8,
}


def _maximum(values):
    return float(np.max(np.asarray(values)))


def _entropy_metrics(entropy):
    entropy = np.asarray(entropy, dtype=float)
    reference = float(entropy[0])
    return {
        "reference_J_kgK": reference,
        "end_minus_start_J_kgK": float(entropy[-1] - reference),
        "max_absolute_drift_J_kgK": _maximum(np.abs(entropy - reference)),
        "range_J_kgK": float(np.ptp(entropy)),
    }


def _profile_metrics(profile, mdot, h0, settings):
    q = {key: np.asarray(value) for key, value in profile.items()}
    finite_fields = ("p", "h", "rho", "u", "T", "s", "A", "x", "a_euler",
                     "a_table", "M_euler", "M_table", "inverse_a2", "Q", "phase_flag")
    finite = all(bool(np.isfinite(q[key]).all()) for key in finite_fields)
    positive = all(bool((q[key] > 0).all()) for key in
                   ("p", "rho", "T", "A", "a_euler", "a_table", "inverse_a2"))
    bounds = bool(
        (q["h"] >= settings["h_min"]).all()
        and (q["h"] <= settings["h_max"]).all()
        and (q["p"] >= settings["p_min"]).all()
        and (q["p"] <= settings["p_max"]).all()
    )
    # The phase flag is itself interpolated. Requiring its value near one keeps
    # quality smeared into an adjacent single-phase cell out of this count.
    interior = ((q["Q"] > 1e-3) & (q["Q"] < 1 - 1e-3)
                & (q["phase_flag"] >= 0.99))
    return {
        "samples": int(len(q["x"])),
        "finite_states": finite,
        "positive_states_and_hyperbolic": positive,
        "within_table_bounds": bounds,
        "mass_flow_relative_max": _maximum(np.abs(q["rho"] * q["u"] * q["A"] / mdot - 1)),
        "total_enthalpy_relative_max": _maximum(np.abs(q["h"] + q["u"]**2 / 2 - h0)) / max(abs(h0), 1.0),
        "table_sound_speed_relative_mismatch_max": _maximum(np.abs(q["a_table"] / q["a_euler"] - 1)),
        "quality_mass_raw_min": float(np.min(q["Q"])),
        "quality_mass_raw_max": _maximum(q["Q"]),
        "phase_flag_raw_min": float(np.min(q["phase_flag"])),
        "phase_flag_raw_max": _maximum(q["phase_flag"]),
        "interior_two_phase_sample_fraction": float(np.mean(interior)),
        "interior_two_phase_sample_count": int(np.sum(interior)),
        "pressure_min_Pa": float(np.min(q["p"])),
        "pressure_max_Pa": _maximum(q["p"]),
        "enthalpy_min_J_kg": float(np.min(q["h"])),
        "enthalpy_max_J_kg": _maximum(q["h"]),
    }


def _smooth_primitive(eos, path, pressure_min, pressure_max, *, intervals=256, order=8):
    """Audit dense ODE output with an independent Gauss integral in log(p).

    Every local enthalpy increment is compared with integral p/rho d(log p).
    The accumulated absolute defect prevents cancellation between intervals.
    Only pressure ranges present in the delivered profiles are included.
    """
    nodes, weights = leggauss(order)
    edges = np.linspace(np.log(pressure_min), np.log(pressure_max), intervals + 1)
    centres = (edges[1:] + edges[:-1]) / 2
    half_widths = np.diff(edges) / 2
    logp = centres[:, None] + half_widths[:, None] * nodes
    p = np.exp(logp)
    # Evaluations in one batch use the same bicubic backend as the reference;
    # the quadrature nodes and algorithm are independent of the ODE integrator.
    h = path.enthalpy(p.ravel()).reshape(p.shape)
    density = eos.state(h, p)["rho"]
    integral = half_widths * np.sum(weights * p / density, axis=1)
    h_edges = path.enthalpy(np.exp(edges))
    defects = np.diff(h_edges) - integral
    scale = max(abs(float(path.h0)), 1.0)
    return {
        "pressure_min_Pa": float(pressure_min),
        "pressure_max_Pa": float(pressure_max),
        "log_pressure_intervals": intervals,
        "gauss_order": order,
        "enthalpy_increment_J_kg": float(h_edges[-1] - h_edges[0]),
        "quadrature_integral_J_kg": float(np.sum(integral)),
        "signed_accumulated_defect_J_kg": float(np.sum(defects)),
        "absolute_accumulated_defect_J_kg": float(np.sum(np.abs(defects))),
        "max_local_defect_J_kg": _maximum(np.abs(defects)),
        "absolute_accumulated_defect_relative_to_h0": float(np.sum(np.abs(defects)) / scale),
    }


def audit_case(eos, upstream, nozzle, mdot, adapted_profile, shock_profile, shock, backpressure):
    """Return JSON-serializable metrics and explicit pass/fail checks.

    The caller must persist this dictionary and reject ``passed == False``.
    Diagnostic entropy drift, raw quality bounds, and acoustic mismatch have
    no hidden acceptance tolerance: they expose the 100x100 interpolation floor.
    """
    h0 = float(upstream.h0)
    profiles = {"adapted": adapted_profile, "internal_shock": shock_profile}
    metrics = {name: _profile_metrics(profile, mdot, h0, eos.settings)
               for name, profile in profiles.items()}
    checks = {}
    for name, result in metrics.items():
        for field in ("finite_states", "positive_states_and_hyperbolic", "within_table_bounds"):
            checks[f"{name}_{field}"] = result[field]
        checks[f"{name}_mass_flow"] = bool(result["mass_flow_relative_max"] < THRESHOLDS["mass_flow_relative"])
        checks[f"{name}_total_enthalpy"] = bool(result["total_enthalpy_relative_max"] < THRESHOLDS["total_enthalpy_relative"])

    star = upstream.state(upstream.p_star)
    throat_mach_error = abs(float(star["u"] / star["a_euler"] - 1))
    checks["sonic_throat"] = bool(throat_mach_error < THRESHOLDS["throat_mach_absolute"])
    checks["choked_mass_flow"] = bool(abs(star["rho"] * star["u"] * nozzle.At / mdot - 1) < THRESHOLDS["mass_flow_relative"])

    allowance = THRESHOLDS["branch_mach_allowance"]
    for name, profile in profiles.items():
        x = np.asarray(profile["x"])
        mach = np.asarray(profile["M_euler"])
        side = np.asarray(profile["side"])
        checks[f"{name}_subsonic_convergent_branch"] = bool((mach[x < 0] <= 1 + allowance).all())
        if name == "adapted":
            checks["adapted_supersonic_divergent_branch"] = bool((mach[x > 0] >= 1 - allowance).all())
        else:
            preshock = (x > 0) & np.isin(side, ("upstream", "shock_pre"))
            postshock = np.isin(side, ("shock_post", "downstream"))
            checks["shock_supersonic_upstream_branch"] = bool((mach[preshock] >= 1 - allowance).all())
            checks["shock_subsonic_downstream_branch"] = bool((mach[postshock] < 1 + allowance).all())

    pre, post = shock["pre"], shock["post"]
    mass1, mass2 = pre["rho"] * pre["u"], post["rho"] * post["u"]
    momentum1 = pre["p"] + pre["rho"] * pre["u"]**2
    momentum2 = post["p"] + post["rho"] * post["u"]**2
    total1 = pre["h"] + pre["u"]**2 / 2
    total2 = post["h"] + post["u"]**2 / 2
    rh = {
        "mass_relative": float(abs(mass2 - mass1) / max(abs(mass1), 1.0)),
        "momentum_relative": float(abs(momentum2 - momentum1) / max(abs(momentum1), 1.0)),
        "total_enthalpy_relative": float(abs(total2 - total1) / max(abs(total1), 1.0)),
        "pressure_ratio": float(post["p"] / pre["p"]),
        "density_ratio": float(post["rho"] / pre["rho"]),
        "entropy_increase_J_kgK": float(post["s"] - pre["s"]),
        "upstream_M_euler": float(pre["u"] / pre["a_euler"]),
        "downstream_M_euler": float(post["u"] / post["a_euler"]),
        "x_shock_m": float(shock["x"]),
    }
    for field in ("mass_relative", "momentum_relative", "total_enthalpy_relative"):
        checks[f"rankine_hugoniot_{field}"] = bool(rh[field] < THRESHOLDS["rankine_hugoniot_relative"])
    checks["shock_compressive_pressure"] = bool(rh["pressure_ratio"] > 1)
    checks["shock_compressive_density"] = bool(rh["density_ratio"] > 1)
    checks["shock_positive_entropy_increase"] = bool(rh["entropy_increase_J_kgK"] > 0)
    checks["shock_lax_mach"] = bool(rh["upstream_M_euler"] > 1 and rh["downstream_M_euler"] < 1)
    checks["shock_inside_divergent_section"] = bool(0 < shock["x"] < nozzle.x_out)

    backpressure_relative = float(abs(shock["exit"]["p"] / backpressure - 1))
    sampled_backpressure_relative = float(abs(shock_profile["p"][-1] / backpressure - 1))
    checks["backpressure_root"] = bool(backpressure_relative < THRESHOLDS["backpressure_relative"])
    checks["sampled_exit_backpressure"] = bool(sampled_backpressure_relative < THRESHOLDS["backpressure_relative"])

    shock_sides = np.asarray(shock_profile["side"])
    mask_up = np.isin(shock_sides, ("upstream", "shock_pre"))
    mask_down = np.isin(shock_sides, ("shock_post", "downstream"))
    entropy = {
        "adapted_smooth_path": _entropy_metrics(adapted_profile["s"]),
        "shock_upstream_smooth_path": _entropy_metrics(np.asarray(shock_profile["s"])[mask_up]),
        "shock_downstream_smooth_path": _entropy_metrics(np.asarray(shock_profile["s"])[mask_down]),
    }
    primitive = {}
    for name, path, pressures in (
        ("adapted_smooth_path", upstream, np.asarray(adapted_profile["p"])),
        ("shock_upstream_smooth_path", upstream, np.asarray(shock_profile["p"])[mask_up]),
        ("shock_downstream_smooth_path", shock["downstream"], np.asarray(shock_profile["p"])[mask_down]),
    ):
        result = _smooth_primitive(eos, path, float(np.min(pressures)), float(np.max(pressures)))
        primitive[name] = result
        checks[f"{name}_independent_quadrature"] = bool(
            result["absolute_accumulated_defect_relative_to_h0"]
            < THRESHOLDS["smooth_primitive_relative_to_h0"]
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "profiles": metrics,
        "rankine_hugoniot": rh,
        "throat_M_euler_absolute_error": throat_mach_error,
        "shock_backpressure_relative_error": backpressure_relative,
        "sampled_shock_exit_backpressure_relative_error": sampled_backpressure_relative,
        "smooth_entropy_diagnostics": entropy,
        "independent_smooth_primitive_quadrature": primitive,
        "diagnostic_definitions": {
            "acoustic_mismatch": "max abs(a_table/a_euler - 1); no acceptance threshold",
            "interior_two_phase": "raw 1e-3 < quality_mass < 1-1e-3 and phase_flag >= 0.99; sample count, not a length fraction",
            "primitive_defect": "sum of absolute local defects in delta(h)-integral(p/rho d(log p)), divided by abs(h0)",
            "entropy": "interpolated JAXProp entropy along each mechanical smooth segment; no zero-drift assertion",
        },
    }
