"""Reproduce general-EOS Roe algebra/property audits, without flow solves.

Run from the repository root:
    poetry run python -B -m development.roe_method.scripts.audit_roe

Endpoint properties are explicitly evaluated with HmassP_INPUTS. PT calls only
select physical enthalpies and the fixed reference state. All algebra under
audit comes from the production model helper; this script supplies independent
conservative maps and checks, not an alternative Roe implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.euler_roe import euler_roe_data, roe_metadata
from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SEED, ENDPOINTS, PAIRS = 637, 150, 3000
SHIFT, RADIUS = -7.3e5, 0.17
EPS = np.finfo(np.float64).eps


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed as a distribution"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _fingerprints():
    paths = [Path(__file__), ROOT / "pyproject.toml", ROOT / "poetry.lock"]
    paths += [ROOT / "src/nozzle_hyperbolic" / name for name in (
        "models/euler_roe.py", "models/euler_nozzle/model.py",
        "models/euler_annular/model.py", "thermodynamics/adapters.py",
        "numerics/fluxes.py", "numerics/characteristic.py", "numerics/roe.py",
        "solvers/fvm/config.py", "solvers/fvm/operator.py", "solvers/fvm/contracts.py",
    )]
    jaxprop_root = Path(jxp.__file__).resolve().parent
    paths += [jaxprop_root / name for name in (
        "coolprop/jax_wrapper.py", "coolprop/core_calculations.py",
        "coolprop/fluid_properties.py", "helpers_props.py",
    )]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def _thermo(fluid, states):
    """Host PH evaluation; returns numerical properties and preserves failures."""
    result, failures = [], []
    for index, state in enumerate(states):
        try:
            value = fluid.get_state(jxp.HmassP_INPUTS, float(state[-1]), float(state[0]))
            result.append([float(value[key]) for key in ("rho", "a", "G")])
        except Exception as error:
            result.append([float("nan")] * 3)
            failures.append(dict(index=index, state=np.asarray(state).tolist(),
                                 error=repr(error)))
    values = np.asarray(result)
    return {key: values[:, i] for i, key in enumerate(("rho", "a", "G"))}, failures


def _qf(state, thermo, radius):
    """Independent homogeneous Euler maps in the common transverse face frame."""
    p, u, h, rho = state[..., 0], state[..., 1], state[..., -1], thermo["rho"]
    v = 0.0 if radius is None else state[..., 2] / radius
    total_h = h + 0.5 * (u*u + v*v)
    q, flux = [rho, rho*u], [rho*u, rho*u*u+p]
    if radius is not None:
        q.append(rho*v)
        flux.append(rho*u*v)
    return np.stack((*q, rho*total_h-p), -1), np.stack((*flux, rho*u*total_h), -1)


def _summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(min=float(finite.min()) if finite.size else None,
                max=float(finite.max()) if finite.size else None,
                nonfinite=int(values.size-finite.size))


def _json_safe(value):
    """Preserve failed/nonfinite records in standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _audit_pair_kind(kind, left, right, tl, tr, reference):
    radius = None if kind == "nozzle" else RADIUS
    evaluate = jax.jit(lambda l, r, a, b: euler_roe_data(
        l, r, a, b, reference, 0.1, radius=radius))
    data = jax.device_get(evaluate(left, right, tl, tr))
    rho, pressure, sound, _ = reference
    qscale = rho*np.asarray([1., sound] + ([] if radius is None else [sound]) + [sound**2])
    fscale = sound*qscale
    ql, fl = _qf(left, tl, radius)
    qr, fr = _qf(right, tr, radius)
    defect = (np.einsum("...ij,...j->...i", data["matrix"], qr-ql)-(fr-fl))/fscale
    # Mixed endpoint-scaled normalization remains meaningful for zero jumps.
    mixed_defect = np.max(np.abs(defect)/(1+np.abs(fl/fscale)+np.abs(fr/fscale)), -1)
    n = left.shape[-1]
    ps = data["projectors"]
    normalization = qscale[None, :]/qscale[:, None]
    completeness = np.max(np.abs((ps.sum(-3)-np.eye(n))*normalization), axis=(-2, -1))
    products = np.stack([ps[:, i]@ps[:, j]-(ps[:, i] if i == j else 0)
                         for i in range(3) for j in range(3)], 1)
    projector_defect = np.max(np.abs(products*normalization), axis=(-3, -2, -1))
    eigendefect = np.max(np.abs((data["matrix"][:, None]@ps
        -data["eigenvalues"][..., None, None]*ps)*normalization/sound), axis=(-3, -2, -1))
    shifted_left, shifted_right = left.copy(), right.copy()
    shifted_left[..., -1] += SHIFT
    shifted_right[..., -1] += SHIFT
    shifted = jax.device_get(jax.jit(lambda l, r, a, b: euler_roe_data(
        l, r, a, b, (*reference[:3], reference[3]+SHIFT), 0.1, radius=radius))(
        shifted_left, shifted_right, tl, tr))
    transform, inverse = np.eye(n), np.eye(n)
    transform[-1, 0], inverse[-1, 0] = SHIFT, -SHIFT
    shift_matrix = np.max(np.abs((shifted["matrix"]-transform@data["matrix"]@inverse)
                                *normalization/sound), axis=(-2, -1))
    shift_projector = np.max(np.abs((shifted["projectors"]-transform@ps@inverse)
                                   *normalization), axis=(-3, -2, -1))
    shift_speed = np.max(np.abs(shifted["eigenvalues"]-data["eigenvalues"])/sound, -1)
    okay = (data["valid"] & shifted["valid"] & (mixed_defect < 2e-11)
            & (completeness < 2e-11) & (projector_defect < 2e-11)
            & (eigendefect < 2e-11) & (shift_matrix < 2e-10)
            & (shift_projector < 2e-10) & (shift_speed < 2e-11))
    records = [dict(index=i, valid=bool(data["valid"][i]), passed=bool(okay[i]),
        pressure_secant_defect=float(data["pressure_secant_defect"][i]),
        sound_speed_squared=float(data["sound_speed_squared"][i]),
        mixed_matrix_jump_defect=float(mixed_defect[i]),
        shift_matrix_defect=float(shift_matrix[i])) for i in range(len(left))]
    return dict(kind=kind, pair_count=len(left), passed=int(okay.sum()),
        failed=int((~okay).sum()), invalid=int((~data["valid"]).sum()),
        nonpositive_roe_sound_speed=int((data["sound_speed_squared"] <= 0).sum()),
        sound_speed_squared=_summary(data["sound_speed_squared"]),
        absolute_pressure_secant_defect=_summary(np.abs(data["pressure_secant_defect"])),
        absolute_matrix_jump_defect_normalized_by_reference_flux=_summary(np.abs(defect)),
        mixed_matrix_jump_defect=_summary(mixed_defect),
        projector_completeness_defect=_summary(completeness),
        projector_product_defect=_summary(projector_defect),
        eigendecomposition_defect=_summary(eigendefect),
        energy_shift_matrix_defect=_summary(shift_matrix),
        energy_shift_projector_defect=_summary(shift_projector),
        energy_shift_speed_defect=_summary(shift_speed),
        failures=[dict(records[i], left=left[i].tolist(), right=right[i].tolist())
                  for i in np.flatnonzero(~okay)], records=records)


def _host_jvp(reference):
    fluid = as_jax_fluid(jxp.Fluid("CO2", backend="HEOS"))
    rho, pressure, sound, enthalpy = reference
    results = []
    for radius in (None, RADIUS):
        state = jnp.asarray([pressure, 130., enthalpy] if radius is None
                            else [pressure, 130., 3., enthalpy])
        def thermo(w):
            return fluid.get_state(jxp.HmassP_INPUTS, w[-1], w[0])
        def maps(w):
            rho = thermo(w)["rho"]
            p, u, h = w[0], w[1], w[-1]
            v = 0. if radius is None else w[2]/radius
            H = h+0.5*(u*u+v*v)
            q, f = [rho, rho*u], [rho*u, rho*u*u+p]
            if radius is not None:
                q.append(rho*v); f.append(rho*u*v)
            return jnp.stack((*q, rho*H-p)), jnp.stack((*f, rho*u*H))
        c, d = jax.jacfwd(maps)(state)
        expected = d@jnp.linalg.inv(c)
        t = thermo(state)
        actual = euler_roe_data(state, state, t, t, reference, .1, radius=radius)
        qscale = rho*jnp.asarray([1., sound]+([] if radius is None else [sound])+[sound**2])
        normalize = lambda a: a*qscale[None, :]/qscale[:, None]/sound
        mismatch = normalize(actual["matrix"]-expected)
        partials = jax.jacfwd(lambda hp: fluid.get_state(jxp.HmassP_INPUTS, hp[0], hp[1])["rho"])(
            jnp.asarray([enthalpy, pressure]))
        analytic = jnp.asarray([-t["rho"]*t["G"]/t["a"]**2, (1+t["G"])/t["a"]**2])
        results.append(dict(kind="nozzle" if radius is None else "annular", state=np.asarray(state).tolist(),
            valid=bool(actual["valid"]), finite=bool(np.isfinite(mismatch).all()),
            max_absolute_normalized_matrix_mismatch=float(jnp.max(jnp.abs(mismatch))),
            relative_matrix_frobenius_mismatch=float(jnp.linalg.norm(mismatch)/jnp.linalg.norm(normalize(expected))),
            density_partials_order=["drho_dh_at_p", "drho_dp_at_h"],
            density_partials_host_jvp=np.asarray(partials).tolist(),
            density_partials_from_a_G=np.asarray(analytic).tolist(),
            density_partials_relative_mismatch=np.asarray((partials-analytic)/analytic).tolist()))
    return results


def _tiny_jumps(fluid, reference):
    records = []
    for pressure, temperature in ((1e6, 320.), (5e6, 320.), (8e6, 320.),
                                  (8e6, 305.), (1e7, 320.), (1e7, 400.)):
        h = float(fluid.get_state(jxp.PT_INPUTS, pressure, temperature)["h"])
        for amplitude in (0., 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12):
            dp, dh = np.asarray([.37*pressure, -.23*reference[2]**2])*amplitude
            left = np.asarray([[pressure-.5*dp, 130., h-.5*dh]])
            right = np.asarray([[pressure+.5*dp, 130., h+.5*dh]])
            tl, el = _thermo(fluid, left); tr, er = _thermo(fluid, right)
            data = jax.device_get(euler_roe_data(left, right, tl, tr, reference, .1))
            ql, fl = _qf(left, tl, None); qr, fr = _qf(right, tr, None)
            scale = reference[0]*np.asarray([reference[2], reference[2]**2, reference[2]**3])
            defect = (data["matrix"]@(qr-ql)[..., None])[..., 0]-(fr-fl)
            records.append(dict(pressure=pressure, temperature=temperature, amplitude=amplitude,
                valid=bool(data["valid"][0]), property_failures=el+er,
                pressure_secant_defect=float(data["pressure_secant_defect"][0]),
                pressure_defect_in_machine_eps=float(data["pressure_secant_defect"][0]/EPS),
                max_reference_flux_normalized_matrix_defect=float(np.max(np.abs(defect/scale)))))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs/roe_audit.json")
    args = parser.parse_args()
    started = time.perf_counter()
    fingerprints = _fingerprints()
    fluid, rng = jxp.Fluid("CO2", backend="HEOS"), np.random.default_rng(SEED)
    ref = fluid.get_state(jxp.PT_INPUTS, 8e6, 320.)
    reference = tuple(float(ref[key]) for key in ("rho", "p", "a", "h"))
    states, property_failures = [], []
    for index in range(ENDPOINTS):
        p = float(np.exp(rng.uniform(np.log(1e6), np.log(20e6))))
        temperature = float(rng.uniform(310., 500.))
        try:
            h = float(fluid.get_state(jxp.PT_INPUTS, p, temperature)["h"])
            ph = fluid.get_state(jxp.HmassP_INPUTS, h, p)
            states.append(dict(pressure=p, temperature=temperature, enthalpy=h,
                               **{key: float(ph[key]) for key in ("rho", "a", "G")}))
        except Exception as error:
            property_failures.append(dict(index=index, pressure=p, temperature=temperature, error=repr(error)))
            states.append(dict(pressure=p, temperature=temperature, enthalpy=float("nan"),
                               rho=float("nan"), a=float("nan"), G=float("nan")))
    pairs = []
    for _ in range(PAIRS):
        il, ir = rng.integers(0, ENDPOINTS, 2)
        ul, ur = rng.uniform(-100., 700., 2)
        vl, vr = rng.uniform(-300., 300., 2)
        pairs.append(dict(left=int(il), right=int(ir), u_left=ul, u_right=ur, v_left=vl, v_right=vr))
    audits = []
    for kind in ("nozzle", "annular"):
        left, right = [], []
        for pair in pairs:
            for target, side in ((left, "left"), (right, "right")):
                state = states[pair[side]]
                row = [state["pressure"], pair[f"u_{side}"]]
                if kind == "annular":
                    row.append(RADIUS*pair[f"v_{side}"])
                target.append([*row, state["enthalpy"]])
        left, right = np.asarray(left), np.asarray(right)
        tl = {key: np.asarray([states[pair["left"]][key] for pair in pairs]) for key in ("rho", "a", "G")}
        tr = {key: np.asarray([states[pair["right"]][key] for pair in pairs]) for key in ("rho", "a", "G")}
        audits.append(_audit_pair_kind(kind, left, right, tl, tr, reference))
    tiny = _tiny_jumps(fluid, reference)
    host_jvp = _host_jvp(reference)
    fingerprints_after = _fingerprints()
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Production Roe matrix/property audit; no flow solves or timings",
        elapsed_seconds=time.perf_counter()-started,
        source=dict(git_head=_git("rev-parse", "HEAD"), branch=_git("branch", "--show-current"),
                    fingerprints_sha256=fingerprints,
                    files_changed_during_audit=[p for p in fingerprints if fingerprints[p] != fingerprints_after.get(p)]),
        environment=dict(python=sys.version, platform=platform.platform(), jax_enable_x64=True,
            jax_backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
            versions={name:_version(name) for name in ("jax", "jaxlib", "jaxprop", "CoolProp", "numpy", "nozzle-hyperbolic")}),
        settings=dict(seed=SEED, endpoint_count=ENDPOINTS, pair_count_per_model=PAIRS,
            pressure_sampling="exp(uniform(log(1e6),log(20e6))) Pa", temperature_range_kelvin=[310.,500.],
            normal_velocity_range=[-100.,700.], transverse_velocity_range=[-300.,300.],
            fluid="CO2", backend="HEOS", numerical_input_pair="HmassP_INPUTS: enthalpy first, pressure second",
            reference_pt=[8e6,320.], reference_rho_p_a_h=reference, fixed_annular_face_radius=RADIUS,
            energy_reference_shift=SHIFT, delta_factor=.1, roe_conventions=roe_metadata(reference),
            matrix_audit="mixed maxabs(A*dU-dF)/(reference_F_scale+abs(F_L)+abs(F_R)) <2e-11",
            source_property_call_policy="PT only for physical setup/enthalpy selection; all endpoint data reevaluated PH"),
        property_failures=property_failures, endpoints=states, pair_draws=pairs, pair_audits=audits,
        host_jvp_audits=host_jvp, tiny_jump_audits=tiny,
        summary=dict(pair_checks_passed=sum(a["passed"] for a in audits),
            pair_checks_failed=sum(a["failed"] for a in audits),
            invalid_roe_pairs=sum(a["invalid"] for a in audits),
            tiny_jump_valid=sum(r["valid"] for r in tiny), tiny_jump_total=len(tiny),
            property_failure_count=len(property_failures),
            host_jvp_relative_mismatch_max=max(r["relative_matrix_frobenius_mismatch"] for r in host_jvp)),
        limitations=["State-pair algebra audits do not establish real-gas flow convergence, positivity, or a nonlinear entropy theorem.",
            "The same smooth single-component CO2 HEOS branch is used on both sides; no phase-switch or two-fluid verification.",
            "JAXProp's configured host bridge differentiates properties by one-sided finite differences with step1e-6*(abs(input)+1).",
            "Near-critical PH inversion noise can exceed machine epsilon; small-jump identity uses the production mixed audit tolerance.",
            "Independently interpolated bicubic rho/a/G data have not been verified here."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), **result["summary"],
                          elapsed_seconds=result["elapsed_seconds"]), indent=2))
    if property_failures or result["summary"]["pair_checks_failed"] or not all(r["valid"] for r in tiny):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
