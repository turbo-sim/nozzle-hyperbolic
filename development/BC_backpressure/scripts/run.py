"""Prepare numerical seeds, run the requested matrix, or assemble saved evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from pathlib import Path
import subprocess
import sys

from .common import OUTPUT, REPO, clean, load, profile, pin_source, save


def assemble():
    from .problem import target_context
    settings = profile()
    records = [load(path) for path in sorted((OUTPUT/"runs").glob("*/N*.json"))]
    seeds = [load(path) for path in sorted((OUTPUT/"seeds").glob("*.json"))]
    cases = [target_context(case).metadata() for case in settings["cases"]]
    provenance_path = OUTPUT / "provenance.json"
    expected = len(settings["cases"])*len(settings["outlets"])*len(settings["fluxes"])*len(settings["reconstructions"])*len(settings["starts"])*len(settings["grids"])
    campaign = dict(schema_version=1, settings=settings,
                    provenance=load(provenance_path) if provenance_path.exists() else None,
                    cases=cases, results=records, seeds=seeds,
                    summary=dict(expected_stages=expected, completed_stages=len(records),
                                 native_converged=sum(row["success"] for row in records),
                                 independently_verified=sum(row["verified"] for row in records),
                                 target_correct=sum(bool(row.get("target_correct")) for row in records),
                                 expected_final_mesh_runs=expected//len(settings["grids"]),
                                 final_mesh_completed=sum(row["cells"] == max(settings["grids"]) for row in records)))
    save(OUTPUT / "campaign.json", campaign)
    print(clean(campaign["summary"]), flush=True)
    return campaign


def launch(arguments, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as stream:
        command = [sys.executable, "-B", "-u", "-m", "development.BC_backpressure.scripts.worker", *arguments]
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    print(f"{'OK' if result.returncode == 0 else 'ERROR'} {' '.join(arguments)} -> {logfile.relative_to(REPO)}", flush=True)
    return result.returncode


def main():
    settings = profile()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", choices=settings["cases"])
    parser.add_argument("--outlet", nargs="+", choices=settings["outlets"])
    parser.add_argument("--flux", nargs="+", choices=settings["fluxes"])
    parser.add_argument("--reconstruction", nargs="+", choices=settings["reconstructions"])
    parser.add_argument("--start", nargs="+", choices=settings["starts"])
    parser.add_argument("--grids", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=settings["execution"]["max_parallel_chains"])
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seeds-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    if args.assemble_only:
        assemble()
        if not args.no_report:
            from .report import build_report
            build_report()
        return
    if args.jobs < 1 or args.jobs > settings["execution"]["max_parallel_chains"]:
        parser.error("jobs must be within the declared parallel-chain limit")
    pin_source()
    if args.prepare_only:
        return
    from .common import chain_id, seed_id
    from .problem import family_of
    cases, outlets = args.case or settings["cases"], args.outlet or settings["outlets"]
    fluxes, reconstructions = args.flux or settings["fluxes"], args.reconstruction or settings["reconstructions"]
    starts = [start for start in settings["starts"] if start in (args.start or settings["starts"])]
    errors = []
    if "adapted" in starts or args.seeds_only:
        tasks = []
        for family, flux, reconstruction in itertools.product(sorted({family_of(case) for case in cases}), fluxes, reconstructions):
            key = seed_id(family, flux, reconstruction)
            tasks.append((["--seed-family", family, "--flux", flux, "--reconstruction", reconstruction], OUTPUT/"logs"/(key+".log")))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(launch, *task) for task in tasks]
            errors.extend(future.result() for future in as_completed(futures))
    if not args.seeds_only:
        tasks = []
        # Stagnant chains precede adapted chains so identical target kernels can
        # reuse compiled code instead of compiling both starts concurrently.
        for start, case, outlet, flux, reconstruction in itertools.product(starts, cases, outlets, fluxes, reconstructions):
            key = chain_id(case, outlet, flux, reconstruction, start)
            arguments = ["--case", case, "--outlet", outlet, "--flux", flux,
                         "--reconstruction", reconstruction, "--start", start]
            if args.grids:
                arguments += ["--grids", *map(str, args.grids)]
            tasks.append((arguments, OUTPUT/"logs"/(key+".log")))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(launch, *task) for task in tasks]
            errors.extend(future.result() for future in as_completed(futures))
    assemble()
    if not args.no_report:
        from .report import build_report
        build_report()
    if any(errors):
        raise SystemExit("Some workers raised errors; inspect retained logs and chain records")


if __name__ == "__main__":
    main()
