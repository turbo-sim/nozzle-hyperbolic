"""Print case feasibility; all flow properties use the 100x100 bicubic tables."""
from pathlib import Path
import json

from .thermodynamics import BicubicCO2
from .reference import Nozzle, SmoothPath, normal_shock, downstream_exit


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    settings = json.loads((root / "cases.json").read_text())
    for case in settings["cases"]:
        nozzle = Nozzle(**(settings["geometry"] | case.get("geometry_override", {})))
        eos = BicubicCO2(case["table"], output)
        reservoir = eos.reservoir(case["p0"], case["T0"])
        path = SmoothPath(eos, reservoir["h"], reservoir["p"], reservoir["h"],
                          eos.settings["p_min"] * 1.001)
        star = path.choke()
        mdot = nozzle.At * star["G"]
        adapted = path.state(path.pressure_for_flux(mdot / nozzle.area(nozzle.x_out), "sup"))
        print(case["id"], flush=True)
        for label, q in (("reservoir", reservoir), ("throat", star), ("adapted exit", adapted)):
            print(label, {k: q[k] for k in ("p", "T", "h", "rho", "Q", "Z_table", "a_euler", "a_table", "s")}, flush=True)
        for fraction in (0.15, 0.35, 0.6, 0.8, 0.95):
            pre = path.state(path.pressure_for_flux(mdot / nozzle.area(fraction * nozzle.x_out), "sup"))
            post = normal_shock(eos, pre, reservoir["h"])
            _, exit_state = downstream_exit(eos, post, reservoir["h"], mdot, nozzle.area(nozzle.x_out))
            print("shock", fraction, "p_back", exit_state["p"], "M", pre["M_euler"], post["M_euler"],
                  "Q", pre["Q"], post["Q"], "ds", post["s"] - pre["s"], flush=True)


if __name__ == "__main__":
    main()
