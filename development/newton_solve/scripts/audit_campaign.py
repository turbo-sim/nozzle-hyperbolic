"""Audit saved coverage, immutable inputs and paired, independent initialization."""
import argparse
import itertools
import numpy as np

from .common import OUTPUT, identity, load, profile, require_source, run_id, save, record_paths


def audit_campaign(*, allow_incomplete=False):
    settings = profile()
    provenance = load(OUTPUT/"provenance.json")
    require_source(provenance)
    paths = record_paths(OUTPUT/"runs")
    expected = {run_id(*config) for config in itertools.product(
        settings["cases"], settings["outlets"], settings["fluxes"], settings["reconstructions"],
        settings["starts"], settings["strategies"])}
    keys, checks = [], {}
    details, pairs = [], {}
    for path in paths:
        row = load(path)
        label = row["id"]
        keys.append(label)
        entry = dict(id=label)
        entry["source_identity"] = all(row.get(key) == provenance[key] for key in
                                        ("source_sha256", "settings_sha256", "client_sha256"))
        entry["target_grid"] = row["cells"] == settings["cells"]
        initial = row.get("initial_state")
        entry["initial_hash"] = (initial is None and row.get("init_hash") is None) or identity(initial) == row.get("init_hash")
        entry["independent_root_flag"] = row["verified"] == row["audit"]["passed"]
        if initial is not None:
            full = np.asarray(initial)
            entry["full_state_shape"] = full.shape == (settings["cells"]+2, 3)
            init = row["initialization"]
            entry["no_hidden_target_continuation"] = (init["analytical_initial_field"] is False
                                                      and init["target_grid_sequence"] is False
                                                      and init["target_implicit_prefix"] is False)
            if row["start"] == "stagnant":
                entry["true_uniform_rest"] = bool(np.all(full[:, 1] == 0.0)
                                                    and np.all(full[:, 0] == full[0, 0])
                                                    and np.all(full[:, -1] == full[0, -1]))
            else:
                seed = load(OUTPUT/"seeds"/(init["seed_id"]+".json"))
                entry["exact_numerical_adapted_seed"] = bool(seed["verified"] and seed["seed_sha256"] == init["seed_sha256"]
                                                               and np.array_equal(full, np.asarray(seed["state"])))
            if row["strategy"] == "direct_newton":
                direct = row.get("direct", {})
                accepted = sum(item.get("phase") == "accepted" for item in row["history"])
                entry["accepted_newton_limit"] = accepted <= settings["direct"]["max_accepted_steps"]
                entry["accepted_step_accounting"] = accepted == row["work"]["accepted_steps"]
                entry["custom_pysolver_backend"] = row.get("backend", {}).get("library") == "pysolver"
                entry["original_steady_equations"] = (direct.get("original_steady_residual") is not None
                                                        and direct.get("original_integrated_spatial_residual") is not None
                                                        and direct.get("original_boundary_residual") is not None)
                entry["bit_identical_actual_direct_start"] = (direct.get("initial_state_sha256") is not None
                                                               and direct["initial_state_sha256"] == direct.get("actual_initial_state_sha256"))
        details.append(entry)
        pair = tuple(row[key] for key in ("case_id", "outlet", "flux", "reconstruction", "start"))
        pairs.setdefault(pair, []).append(dict(init_hash=row.get("init_hash"), has_state=initial is not None))
    checks.update(unique_runs=len(keys) == len(set(keys)), no_unrequested_runs=set(keys) <= expected,
                  coverage=allow_incomplete or set(keys) == expected)
    paired = []
    for key, entries in pairs.items():
        if len(entries) == len(settings["strategies"]) and all(row["has_state"] for row in entries):
            passed = len({row["init_hash"] for row in entries}) == 1
            paired.append(dict(config=key, passed=passed))
    checks["all_record_checks"] = all(all(value for key, value in item.items() if key != "id") for item in details)
    checks["identical_strategy_starts"] = all(pair["passed"] for pair in paired)
    result = dict(passed=all(checks.values()), checks=checks, expected_runs=len(expected), completed_runs=len(keys),
                  missing_runs=sorted(expected-set(keys)), paired_initializations=paired, records=details)
    save(OUTPUT/"audit_campaign.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = audit_campaign(allow_incomplete=args.allow_incomplete)
    print({key: value for key, value in result.items() if key not in ("records", "paired_initializations", "missing_runs")})
    if not result["passed"]:
        raise SystemExit("Saved-evidence audit failed; see outputs/audit_campaign.json")


if __name__ == "__main__":
    main()
