"""Run the CO2 accuracy/robustness campaign with production grid sequencing."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .common import STUDY, REPO, BASELINE, OUTPUT, identity, provenance, save, source_manifest
from .profiles import register, validate_existing


def read_fields(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: [r[key] if key == "side" else float(r[key]) for r in rows] for key in rows[0]}


def case_metadata():
    prepared = OUTPUT / "reference_summary.json"
    summary = json.loads((prepared if prepared.exists() else BASELINE / "outputs" / "summary.json").read_text())
    reference_folder = OUTPUT if prepared.exists() else BASELINE / "outputs"
    records = []
    for physical in summary["cases"]:
        for regime in ("adapted", "shock"):
            prefix = "reference_" if prepared.exists() else ""
            fields = read_fields(reference_folder / f"{prefix}{physical['id']}_{regime}.csv")
            records.append(dict(
                id=physical["id"] + "_" + regime, fluid_case=physical["id"], regime=regime,
                label=physical["label"] + (" — adapted" if regime == "adapted" else " — internal shock"),
                p0=physical["settings"]["p0"], T0=physical["settings"]["T0"],
                backpressure=physical["adapted_exit"]["p"] if regime == "adapted" else physical["backpressure"],
                geometry=physical["geometry"], mdot=physical["mdot"],
                shock_position=physical["shock"]["x"] if regime == "shock" else None,
                table=physical["settings"]["table"], table_sha256=physical["table_sha256"],
                reference=fields, baseline_audit=physical["audit"],
            ))
    return records


def assemble(metadata, profile):
    results, chains = [], []
    for fluid_case, regime, flux, reconstruction in product(
            profile["fluid_cases"], profile["regimes"], profile["fluxes"], profile["reconstructions"]):
        case_id = fluid_case + "_" + regime
        chain = "__".join((case_id, flux, reconstruction))
        folder = OUTPUT / "runs" / chain
        chain_file = folder / "chain.json"
        chain_record = json.loads(chain_file.read_text()) if chain_file.exists() else dict(chain=chain, completed=False)
        chains.append(chain_record)
        for cells in profile["grids"]:
            path = folder / f"N{cells}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
            else:
                results.append(dict(id=f"{chain}__N{cells}", case_id=case_id, flux=flux,
                                    reconstruction=reconstruction, cells=cells,
                                    success=False, verified=False, reason="not_completed",
                                    fields={}, errors={}, history=[], audit={},
                                    initialization=dict(method="not completed")))
    metadata = dict(metadata)
    metadata["source_after"] = source_manifest()
    metadata["src_unchanged"] = metadata["source_before"] == metadata["source_after"]
    diagnostic_trials = []
    for path in sorted((OUTPUT / "diagnostics").glob("*/runs/*/N*.json")):
        row = json.loads(path.read_text())
        row["diagnostic_trial"] = path.parents[2].name
        diagnostic_trials.append(row)
    metadata["diagnostic_trials"] = diagnostic_trials
    registry = OUTPUT / "profile_registry.json"
    if registry.exists():
        metadata["profile_registry"] = json.loads(registry.read_text())
    snapshot = OUTPUT / "source_snapshot_manifest.json"
    if snapshot.exists():
        metadata["source_snapshot"] = json.loads(snapshot.read_text())
    campaign = dict(metadata=metadata, settings=profile, cases=case_metadata(), results=results, chains=chains)
    campaign["summary"] = dict(
        expected_stages=len(results), completed_stages=sum(r["reason"] != "not_completed" for r in results),
        native_converged=sum(bool(r["success"]) for r in results),
        independently_verified=sum(bool(r.get("verified")) for r in results),
        chains_completed=sum(bool(c.get("completed")) for c in chains),
        expected_chains=len(chains),
    )
    save(OUTPUT / "campaign.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluid-case", nargs="+", choices=("supercritical", "lettieri_equilibrium"))
    parser.add_argument("--regime", nargs="+", choices=("adapted", "shock"))
    parser.add_argument("--flux", nargs="+", choices=("rusanov", "hll", "roe"))
    parser.add_argument("--reconstruction", nargs="+", choices=("constant", "muscl_van_leer"))
    parser.add_argument("--grids", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    profile = json.loads((STUDY / "settings.json").read_text())
    OUTPUT.mkdir(exist_ok=True)
    if not args.assemble_only:
        from .prepare_reference import prepare_reference
        prepare_reference()
    metadata_file = OUTPUT / "provenance.json"
    if args.resume or args.assemble_only:
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else provenance()
        if metadata["source_before"] != source_manifest():
            raise RuntimeError("Source changed since campaign start; keep runs separate")
    else:
        existing = list((OUTPUT / "runs").glob("*/N*.json"))
        if existing:
            raise RuntimeError("Existing campaign stages found; use --resume to preserve/reuse them")
        metadata = provenance()
    save(metadata_file, metadata)
    if metadata.get("profile_sha256", identity(profile)) != identity(profile):
        validate_existing(profile)
    else:
        register(profile)
    metadata["profile_sha256"] = identity(profile)
    reference_metadata = json.loads((OUTPUT / "reference_summary.json").read_text())["metadata"]
    metadata["reference_fingerprint"] = reference_metadata["reference_fingerprint"]
    save(metadata_file, metadata)
    if not args.assemble_only:
        for fluid_case, regime, flux, reconstruction in product(
                args.fluid_case or profile["fluid_cases"], args.regime or profile["regimes"],
                args.flux or profile["fluxes"], args.reconstruction or profile["reconstructions"]):
            chain = "__".join((fluid_case + "_" + regime, flux, reconstruction))
            command = [sys.executable, "-B", "-u", "-m",
                       "development.real_gas.fvm_solutions.scripts.worker",
                       "--fluid-case", fluid_case, "--regime", regime,
                       "--flux", flux, "--reconstruction", reconstruction]
            if args.grids:
                command += ["--grids", *map(str, args.grids)]
            if args.resume:
                command.append("--resume")
            log = OUTPUT / "logs" / f"{chain}.log"
            log.parent.mkdir(exist_ok=True)
            started = perf_counter()
            print(f"START {chain}", flush=True)
            with log.open("a" if args.resume else "w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            print(f"END {chain}: exit={result.returncode} wall={perf_counter()-started:.1f}s log={log}", flush=True)
            assemble(metadata, profile)
            if metadata["source_before"] != source_manifest():
                raise RuntimeError("Source changed during campaign; completed evidence retained without mixing versions")
    campaign = assemble(metadata, profile)
    print(json.dumps(campaign["summary"]), flush=True)
    if not campaign["metadata"]["src_unchanged"]:
        raise RuntimeError("Source manifest changed during campaign")
    if not args.no_report:
        from .report import main as report_main
        # The generator's CLI must not parse the campaign driver's arguments.
        original = sys.argv
        try:
            sys.argv = ["report"]
            report_main()
        finally:
            sys.argv = original


if __name__ == "__main__":
    main()
