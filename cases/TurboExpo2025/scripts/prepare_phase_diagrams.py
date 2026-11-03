"""Save reproducible equilibrium saturation context without advancing CFD.

Run from the repository root::

    poetry run python -B -m cases.TurboExpo2025.scripts.prepare_phase_diagrams

The curves are native HEOS equilibrium data, independent of the bicubic tables
used by the finite-volume calculation. This script supplies thermodynamic
context for Figures 5 and 7; it does not digitize a published curve.
"""

import json
from pathlib import Path

import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.solvers.convergence_output import strict_json

ROOT = Path(__file__).resolve().parents[1]
FLUIDS = ("CO2", "MDM")
POINTS = 120


def prepare():
    cases = json.loads((ROOT / "case_definitions.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    results = []
    for name in FLUIDS:
        fluid = jxp.Fluid(name, backend="HEOS")
        minimum = float(fluid.triple_point_liquid["T"]) + 0.1
        maximum = float(fluid.critical_point["T"]) - 0.2
        temperatures = np.linspace(minimum, maximum, POINTS)
        item = dict(
            fluid=name,
            backend="JAXProp HEOS host equilibrium",
            temperature_limits_K=[minimum, maximum],
            points=POINTS,
        )
        for label, quality in (("saturation_liquid", 0.0), ("saturation_vapor", 1.0)):
            states = [
                fluid.get_state(jxp.QT_INPUTS, quality, float(T)) for T in temperatures
            ]
            entropy = np.asarray([float(state["s"]) for state in states])
            if not np.isfinite(entropy).all():
                raise ValueError(f"Nonfinite {name} {label} entropy")
            item[label] = dict(T=temperatures, s=entropy)
        critical = {key: float(fluid.critical_point[key]) for key in ("T", "s")}
        if not all(np.isfinite(value) for value in critical.values()):
            raise ValueError(f"Nonfinite {name} critical state")
        item["critical"] = critical
        item["reservoirs"] = []
        for case in cases:
            if case["fluid"] != name:
                continue
            state = fluid.get_state(jxp.PT_INPUTS, case["p0_Pa"], case["T0_K"])
            row = dict(
                case_id=case["id"],
                p0=case["p0_Pa"],
                T0=case["T0_K"],
                s0=float(state["s"]),
                h0=float(state["h"]),
            )
            if not all(
                np.isfinite(value) for key, value in row.items() if key != "case_id"
            ):
                raise ValueError(f"Nonfinite reservoir for {case['id']}")
            item["reservoirs"].append(row)
        results.append(item)
    return dict(
        schema_version=1,
        fluids=results,
        audit=dict(
            all_values_finite=True,
            saturation="native HEOS QT equilibrium,120 equally spaced temperatures per branch",
            units=dict(T="K", s="J/(kg K)", h0="J/kg", p0="Pa"),
        ),
    )


def main():
    target = ROOT / "outputs/phase_diagrams.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = prepare()
    target.write_text(strict_json(result), encoding="utf-8")
    print(f"Saved {target}")
    for fluid in result["fluids"]:
        print(
            f"{fluid['fluid']}: {POINTS} samples per phase; "
            f"{len(fluid['reservoirs'])} reservoir records; all finite"
        )


if __name__ == "__main__":
    main()
