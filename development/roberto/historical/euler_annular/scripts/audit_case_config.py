"""Audit YAML inputs against retained physical verification data and bad inputs.

This is a study audit, not a production pytest test. It performs no solves.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jaxprop as jxp
import numpy as np

from .case_config import (DEFAULT_CONFIG, STUDY, available_cases, build_case,
                          load_case_config, validate_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, default=STUDY / "outputs" / "grid_sequence")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs" / "case_config" / "audit.json")
    args = parser.parse_args()
    fluid = jxp.FluidPerfectGas(name="Air")
    checks = []
    cases = available_cases()
    expected_names = ("radial_outflow", "radial_inflow", "sonic_exit", "swirl_diffusion",
                      "conical_subsonic", "curved_subsonic", "supersonic_expansion", "shock35", "shock70",
                      "radial_outflow_m005", "radial_outflow_m020", "radial_outflow_swirl")
    assert cases == expected_names
    for name in cases:
        case = build_case(name, fluid)
        path = args.saved / f"{name}_50.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            mesh = case.geometry.mesh(50)
            differences = {}
            for key, actual in (("m", mesh.m_centers), ("m_faces", mesh.m_faces),
                                ("r", mesh.r_center), ("volumes", mesh.volumes)):
                recorded = np.asarray(saved[key])
                differences[key] = float(np.max(np.abs(actual-recorded)) / max(np.max(np.abs(recorded)), 1e-30))
                np.testing.assert_allclose(actual, recorded, rtol=2e-14, atol=1e-16)
            for key, recorded in saved["reference_invariants"].items():
                actual = float(getattr(case.reference, key))
                differences[key] = abs(actual-recorded) / max(abs(recorded), 1.0)
                np.testing.assert_allclose(actual, recorded, rtol=2e-14, atol=1e-15)
            backpressure = float(case.reference.profile(case.geometry.length, shock_m=case.shock_m)["p"])
            np.testing.assert_allclose(backpressure, saved["backpressure"], rtol=2e-14)
            assert case.alpha == saved["alpha_in"]
            assert case.inlet_regime == saved["inlet_regime"] and case.outlet_regime == saved["outlet_regime"]
            assert case.shock_m == saved["shock_m"]
            checks.append(dict(case=name, saved_file=str(path), physical_data_match=True,
                               largest_relative_difference=max(differences.values())))
        else:
            expected_mach, expected_alpha = {
                "radial_outflow_m005": (.05, 0.),
                "radial_outflow_m020": (.2, 0.),
                "radial_outflow_swirl": (.1, np.arctan(1.5)),
            }[name]
            inlet = case.reference.profile(0.0)
            np.testing.assert_allclose(inlet["M_m"], expected_mach, rtol=1e-13)
            assert case.alpha == expected_alpha
            checks.append(dict(case=name, inlet_mach=float(inlet["M_m"]), physical_inputs_match=True))
    base = load_case_config()["config"]
    bad_inputs = (
        ("unknown field", lambda c: c["run"].update(cfl_growht=1.2)),
        ("unknown monitor", lambda c: c["monitors"][0].update(quantity="made_up")),
        ("duplicate monitor", lambda c: c["monitors"].append(deepcopy(c["monitors"][0]))),
        ("negative residual tolerance", lambda c: c["residuals"].update(relative_tolerance=-1.)),
        ("invalid CFL growth", lambda c: c["run"].update(cfl_growth=1.)),
        ("bool wall flux", lambda c: c["walls"].update(heat_flux=True)),
        ("one-cell mesh", lambda c: c["mesh"].update(cells=1)),
    )
    for label, mutate in bad_inputs:
        value = deepcopy(base)
        mutate(value)
        try:
            validate_config(value)
        except (ValueError, TypeError):
            checks.append(dict(check=label, rejected=True))
        else:
            raise AssertionError(f"Bad config accepted: {label}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.parent / "invalid_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    duplicate = folder / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    cycle = folder / "cycle.yaml"
    cycle.write_text("schema_version: 1\nextends: cycle.yaml\n", encoding="utf-8")
    for label, path in (("duplicate YAML key", duplicate), ("cyclic inheritance", cycle)):
        try:
            available_cases(path)
        except ValueError:
            checks.append(dict(check=label, rejected=True))
        else:
            raise AssertionError(f"Bad file accepted: {label}")
    args.output.write_text(json.dumps(dict(passed=True, cases=len(cases), checks=checks), indent=2) + "\n", encoding="utf-8")
    print(f"YAML inputs: {len(cases)} physical cases and {len(checks)-len(cases)} invalid-input checks passed")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
