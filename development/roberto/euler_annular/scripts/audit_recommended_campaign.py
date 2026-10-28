"""Audit fixed-growth controller histories and compare the archived campaign.

Complements audit_results, which independently reconstructs the PDE operators.
This audit replays bookkeeping only; it performs no timestep or Newton solves.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .recommended_controller import SETTINGS
from .verify_annular import DEFAULT_CASES, RECONSTRUCTIONS, STUDY, fingerprints, write_json


def audit_file(path, baseline):
    data = json.loads(path.read_text(encoding="utf-8"))
    previous = json.loads(baseline.read_text(encoding="utf-8"))
    settings = data["settings"]
    assert all(settings[key] == value for key, value in SETTINGS.items())
    comparisons = {}
    for method in RECONSTRUCTIONS:
        run, old = data["methods"][method], previous["methods"][method]
        assert np.array_equal(run["initial_state"], old["initial_state"])
        cfl, accepted, retry, pseudo_time = settings["cfl"], 0, 0, 0.
        for row in run["history"]:
            assert row["step"] == accepted+1 and row["retry"] == retry
            assert np.isclose(row["cfl"], cfl, rtol=1e-12, atol=0)
            assert 0 <= row["inner_iterations"] <= 8
            assert not row["residual_only_accept"]
            if row["accepted"]:
                assert row["inner_status"] == 0 and row["admissible"]
                assert row["inner_residual"] <= 1e-10
                assert row["inner_relative_update"] <= 1e-10
                components = np.asarray(row["integrated_balance_scaled"])
                assert components.shape == (4,) and np.isfinite(components).all()
                assert np.isclose(np.max(abs(components)), row["conservation_max"],
                                  rtol=1e-9, atol=1e-14)
                pseudo_time += row["dt"]
                assert np.isclose(pseudo_time, row["pseudo_time"], rtol=1e-12, atol=1e-15)
                cfl = min(1e6, cfl*1.2)
                accepted += 1
                retry = 0
            else:
                cfl *= .5
                retry += 1
        assert accepted == run["controller_audit"]["accepted"]
        assert len(run["history"]) == run["controller_audit"]["attempts"]
        # A direct pressure maximum in Pa is retained without an arbitrary mask.
        state_difference = np.asarray(run["state"])-np.asarray(old["state"])
        flux_scales = np.asarray(old["errors"]["integrated_balance_scales"])
        sound = flux_scales[1]/flux_scales[0]
        pressure = (old["initial_state"][0][0] if data["inlet_regime"] == "supersonic"
                    else data["reservoir"]["p0"])
        primitive_scales = np.array([pressure, sound, flux_scales[2]/flux_scales[0], sound**2])
        scaled_difference = float(np.max(abs(state_difference)/primitive_scales))
        comparisons[method] = dict(
            root=run["success"], accuracy=run["analytical_target_met"],
            old_accepted=sum(r["accepted"] for r in old["history"]),
            accepted=accepted, rejected=len(run["history"])-accepted,
            old_rejected=sum(not r["accepted"] for r in old["history"]),
            newton_corrections=sum(r["inner_iterations"] for r in run["history"]),
            backtracks=sum(r["rejected_newton_trials"] for r in run["history"]),
            maximum_pressure_difference_Pa=float(np.max(abs(state_difference[:, 0]))),
            maximum_scaled_state_difference=scaled_difference,
            matched_archived_discrete_state=scaled_difference < 1e-5,
            old_Cp_error=old["errors"]["Cp_max"], Cp_error=run["errors"]["Cp_max"],
            final_original_defect=max(run["metrics"][key] for key in
                ("spatial_rms", "boundary_max", "conservation_max")),
            initial_dt=run["controller_audit"]["initial_dt"],
        )
    return dict(case=data["case"], cells=data["cells"], methods=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=STUDY/"outputs"/"verification")
    args = parser.parse_args()
    root = STUDY/"outputs"/"cfl_recommended"
    before = json.loads((root/"source_before.json").read_text(encoding="utf-8"))
    unchanged = before == fingerprints()
    assert unchanged, "Production numerical source changed during campaign"
    records = [audit_file(args.directory/f"{case}_{cells}.json",
                         STUDY/"outputs"/"before_cfl_recommended"/"verification"/f"{case}_{cells}.json")
               for case in DEFAULT_CASES for cells in (50, 100, 200, 400)]
    report = dict(passed=True, source_unchanged=unchanged, settings=SETTINGS,
                  case_mesh_files=len(records), numerical_solutions=2*len(records), records=records)
    write_json(root/"controller_audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
