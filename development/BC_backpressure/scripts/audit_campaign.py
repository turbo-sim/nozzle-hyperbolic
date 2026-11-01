"""Check saved execution identities and optionally re-evaluate original equations."""
import argparse
from hashlib import sha256
import itertools
from types import SimpleNamespace

import numpy as np

from .common import OUTPUT, clean, identity, load, profile, save, configure_compilation_cache, require_source


def audit(fresh=False):
    campaign = load(OUTPUT / "campaign.json")
    settings, provenance = campaign["settings"], campaign["provenance"]
    if fresh:
        require_source(provenance)
    cache = configure_compilation_cache() if fresh else None
    expected = set(itertools.product(settings["cases"], settings["outlets"], settings["fluxes"],
                                     settings["reconstructions"], settings["starts"], settings["grids"]))
    actual = [(row["case_id"], row["outlet"], row["flux"], row["reconstruction"], row["start"], row["cells"])
              for row in campaign["results"]]
    checks = dict(expected_matrix=set(actual) == expected, unique_records=len(actual) == len(set(actual)),
                  settings_identity=identity(settings) == provenance["settings_sha256"],
                  current_settings_match=identity(profile()) == provenance["settings_sha256"],
                  source_identity=identity(provenance["source_files"]) == provenance["source_sha256"])
    checks["source_snapshot_bytes"] = all(
        (OUTPUT/provenance["snapshot"]/name).exists()
        and sha256((OUTPUT/provenance["snapshot"]/name).read_bytes()).hexdigest() == digest
        for name, digest in {**provenance["source_files"], **provenance["study_files"]}.items())
    import jax
    jax.config.update("jax_enable_x64", True)
    from .problem import target_context, family_of
    contexts = {name: target_context(name) for name in settings["cases"]}
    case_metadata = {row["id"]: row for row in campaign["cases"]}
    table_hashes = {}
    for name, context in contexts.items():
        if context.real_gas:
            table_hashes[name] = sha256(context.physical.eos.table_file.read_bytes()).hexdigest()
        else:
            table_hashes[name] = None
    checks["case_table_identities"] = all(
        case_metadata[name]["table_sha256"] == context.table_sha256 == table_hashes[name]
        for name, context in contexts.items())
    seeds = {row["id"]: row for row in campaign["seeds"]}
    expected_seeds = set(itertools.product({family_of(name) for name in settings["cases"]},
                                           settings["fluxes"], settings["reconstructions"]))
    checks["expected_seed_matrix"] = {(row["family"], row["flux"], row["reconstruction"])
                                     for row in campaign["seeds"]} == expected_seeds
    seed_checks = []
    for seed in campaign["seeds"]:
        fingerprint = identity({key: seed[key] for key in
                                ("state", "points", "scheme", "source_sha256", "settings_sha256")})
        local = dict(actual_seed_state_identity=fingerprint == seed["seed_sha256"],
                     source_identity=seed["source_sha256"] == provenance["source_sha256"],
                     settings_identity=seed["settings_sha256"] == provenance["settings_sha256"],
                     numerical_solve_evidence=bool((seed["history"] or not seed["success"])
                                                  and seed["start"] == "numerical_adapted_preparation"),
                     verified_matches_audit=seed["verified"] == bool(seed["success"] and seed["audit"]["passed"]))
        table = table_hashes.get(seed["family"]+"_shock")
        local["table_identity"] = seed["table_sha256"] == table
        local["declared_scheme"] = all(seed["scheme"][key] == value for key, value in settings["scheme"].items())
        local["declared_scheme"] = local["declared_scheme"] and all(seed["scheme"][key] == seed[key] for key in ("flux", "reconstruction"))
        seed_checks.append(dict(id=seed["id"], passed=all(local.values()), checks=local))
    checks["all_seed_records_consistent"] = all(row["passed"] for row in seed_checks)
    records_by_id = {row["id"]: row for row in campaign["results"]}
    rows = []
    for row in campaign["results"]:
        local = dict(source_identity=row["source_sha256"] == provenance["source_sha256"],
                     settings_identity=row["settings_sha256"] == provenance["settings_sha256"],
                     verified_matches_audit=row["verified"] == bool(row["success"] and row["audit"]["passed"]),
                     table_identity=row.get("table_sha256") == table_hashes[row["case_id"]])
        if row["state"] is not None:
            local["declared_scheme"] = all(row["scheme"][key] == value for key, value in settings["scheme"].items())
            local["declared_scheme"] = local["declared_scheme"] and all(row["scheme"][key] == row[key] for key in ("flux", "reconstruction"))
            local["initial_state_identity"] = identity(row["initial_state"]) == row["initialization"]["initial_state_sha256"]
            initial = np.asarray(row["initial_state"])
            points = np.asarray(row["points"])
            source_id = row["initialization"]["source_run_id"]
            if source_id is not None:
                previous = records_by_id.get(source_id)
                valid = previous is not None and previous["success"] and previous["cells"] == row["initialization"]["source_cells"]
                valid = valid and previous["cells"] < row["cells"] and all(previous[key] == row[key]
                            for key in ("case_id", "outlet", "flux", "reconstruction", "start"))
                if valid:
                    expected_initial = np.column_stack([np.interp(points, previous["points"], np.asarray(previous["state"])[:, j])
                                                        for j in range(initial.shape[1])])
                    valid = np.array_equal(initial, expected_initial)
                local["exact_successful_numerical_prolongation"] = bool(valid)
            if row["start"] == "stagnant" and source_id is None:
                local["stagnant_velocities_zero"] = bool(np.all(initial[:, 1:-1] == 0.0))
                model = contexts[row["case_id"]].model(row["cells"], row["outlet"], start="stagnant")
                local["exact_uniform_reservoir_state"] = bool(np.array_equal(initial, np.asarray(model.initial_state())))
            if row["start"] == "adapted":
                seed = seeds.get(row["initialization"]["seed_id"])
                local["adapted_seed_numerical_and_verified"] = bool(seed is not None and seed["verified"]
                    and seed["seed_sha256"] == row["initialization"]["seed_sha256"])
                if source_id is None and seed is not None:
                    expected_initial = np.column_stack([np.interp(points, seed["points"], np.asarray(seed["state"])[:, j])
                                                        for j in range(initial.shape[1])])
                    local["exact_numerical_adapted_seed_array"] = bool(np.array_equal(initial, expected_initial))
            if source_id is None:
                peer_outlet = "finite_wave" if row["outlet"] == "acoustic" else "acoustic"
                peers = [other for other in campaign["results"] if other["outlet"] == peer_outlet
                         and other["state"] is not None and other["initialization"]["source_run_id"] is None
                         and all(other[key] == row[key] for key in ("case_id", "flux", "reconstruction", "start", "cells"))]
                if peers:
                    local["identical_physical_start_across_outlet_laws"] = all(
                        np.array_equal(initial, np.asarray(other["initial_state"])) for other in peers)
            if fresh:
                from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, ResidualPolicy
                from .audit import evaluate_solution
                context = target_context(row["case_id"])
                model = context.model(row["cells"], row["outlet"], start="stagnant")
                operator = Operator(model, Scheme(**row["scheme"]))
                solution = SimpleNamespace(state=np.asarray(row["state"]), success=row["success"],
                                           history=row["history"], operator=operator)
                result = evaluate_solution(context, solution, ResidualPolicy(**settings["residual"]))
                local["fresh_audit_matches"] = bool(result["audit"]["passed"] == row["audit"]["passed"])
                local["fresh_target_matches"] = result["target_correct"] == row.get("target_correct")
        rows.append(dict(id=row["id"], passed=all(local.values()), checks=local,
                         failed_checks=[name for name, passed in local.items() if not passed]))
    checks["all_saved_records_consistent"] = all(row["passed"] for row in rows)
    result = dict(passed=all(checks.values()), checks=checks,
                  failed_checks=[name for name, passed in checks.items() if not passed],
                  summary=campaign["summary"], fresh_equations_evaluated=fresh,
                  compilation_cache=cache, records=rows, seeds=seed_checks,
                  note="Evidence consistency is separate from solver/target success; unsuccessful rows remain unsuccessful.")
    save(OUTPUT / "campaign_audit.json", result)
    print(clean({key: value for key, value in result.items() if key != "records"}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if not audit(args.fresh)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
