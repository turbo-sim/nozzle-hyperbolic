"""Resume isolated target solves or assemble/report retained numerical evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from hashlib import sha256
import subprocess
import sys

from .common import OUTPUT, REPO, load, profile, pin_source, save, run_id, seed_id, record_paths


def record_index(path):
    """Reduce one raw record at a time; full trial vectors stay lossless on disk."""
    row = load(path)
    keys = ("id", "case_id", "outlet", "flux", "reconstruction", "start", "strategy", "cells",
            "success", "native_success", "verified", "target_correct", "reason", "elapsed_s", "compile_s", "solve_s",
            "source_sha256", "settings_sha256", "client_sha256", "init_hash", "seed_sha256", "variant")
    result = {key: row[key] for key in keys if key in row}
    result["file"] = path.relative_to(OUTPUT).as_posix()
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    result["file_sha256"] = digest.hexdigest()
    result["stored_bytes"] = path.stat().st_size
    return result


def assemble():
    """Presentation assembly reads saved evidence and never imports a flow solver."""
    settings = profile()
    records = [record_index(path) for path in record_paths(OUTPUT/"runs")]
    seeds = [record_index(path) for path in record_paths(OUTPUT/"seeds")]
    stages = [record_index(path) for folder in sorted((OUTPUT/"seed_stages").glob("*")) for path in record_paths(folder)]
    variants = [record_index(path) for folder in sorted((OUTPUT/"variants").glob("*/runs")) for path in record_paths(folder)]
    expected = 1
    for name in ("cases", "outlets", "fluxes", "reconstructions", "starts", "strategies"):
        expected *= len(settings[name])
    summary = dict(expected_runs=expected, completed_runs=len(records),
                   native_converged=sum(bool(row["native_success"]) for row in records),
                   independently_verified=sum(bool(row["verified"]) for row in records),
                   target_correct=sum(bool(row.get("target_correct")) for row in records),
                   by_strategy={strategy: dict(completed=sum(row["strategy"] == strategy for row in records),
                       native_converged=sum(row["strategy"] == strategy and row["native_success"] for row in records),
                       verified=sum(row["strategy"] == strategy and row["verified"] for row in records),
                       target_correct=sum(row["strategy"] == strategy and bool(row.get("target_correct")) for row in records))
                       for strategy in settings["strategies"]},
                   expected_seeds=len(settings["fluxes"])*len(settings["reconstructions"]),
                   completed_seeds=len(seeds), verified_seeds=sum(row["verified"] for row in seeds))
    campaign = dict(schema_version=1, settings=settings, provenance=load(OUTPUT/"provenance.json") if (OUTPUT/"provenance.json").exists() else None,
                    cases=load(OUTPUT/"cases.json") if (OUTPUT/"cases.json").exists() else [],
                    results=records, seeds=seeds, seed_stages=stages, variants=variants, summary=summary)
    save(OUTPUT/"campaign.json", campaign)
    print(summary, flush=True)
    return campaign


def launch(arguments, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-B", "-u", "-m", "development.newton_solve.scripts.worker", *arguments],
                                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    print(f"{'OK' if result.returncode == 0 else 'ERROR'} {' '.join(arguments)} -> {logfile.relative_to(REPO)}", flush=True)
    return result.returncode


def main():
    settings = profile()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "outlet", "flux", "reconstruction", "start", "strategy"):
        plural = {"case":"cases", "outlet":"outlets", "flux":"fluxes", "reconstruction":"reconstructions", "start":"starts", "strategy":"strategies"}[name]
        parser.add_argument("--"+name, nargs="+", choices=settings[plural])
    parser.add_argument("--jobs", type=int, default=settings["execution"]["max_parallel_workers"])
    parser.add_argument("--resume", action="store_true", help="Explicitly request the default behavior: preserve and skip every saved attempt, including failures")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seeds-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--first-step-fraction", type=float, choices=(0.1, 0.25, 1.0),
                        help="Optional explicit supplemental direct-Newton variant, saved outside baseline results")
    args = parser.parse_args()
    if args.assemble_only:
        assemble()
        if not args.no_report:
            from .report import build_report
            build_report()
        return
    if not 1 <= args.jobs <= settings["execution"]["max_parallel_workers"]:
        parser.error("jobs must remain within the declared independent worker limit")
    pin_source()
    if not (OUTPUT/"cases.json").exists():
        from .problem import target_context
        save(OUTPUT/"cases.json", [target_context(case).metadata() for case in settings["cases"]])
    if args.prepare_only:
        return
    selected = {name: getattr(args, name) or settings[plural] for name, plural in
                (("case", "cases"), ("outlet", "outlets"), ("flux", "fluxes"),
                 ("reconstruction", "reconstructions"), ("start", "starts"), ("strategy", "strategies"))}
    if args.first_step_fraction is not None and selected["strategy"] != ["direct_newton"]:
        parser.error("Select --strategy direct_newton for an explicit initial-step variant")
    errors = []
    if "adapted" in selected["start"] or args.seeds_only:
        tasks = []
        for flux, reconstruction in itertools.product(selected["flux"], selected["reconstruction"]):
            key = seed_id(flux, reconstruction)
            tasks.append((["--seed", "--flux", flux, "--reconstruction", reconstruction], OUTPUT/"logs"/("seed__"+key+".log")))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            errors += [future.result() for future in as_completed([pool.submit(launch, *task) for task in tasks])]
    if not args.seeds_only:
        tasks = []
        for start, case, outlet, flux, reconstruction, strategy in itertools.product(
                selected["start"], selected["case"], selected["outlet"], selected["flux"], selected["reconstruction"], selected["strategy"]):
            key = run_id(case, outlet, flux, reconstruction, start, strategy)
            arguments = ["--case", case, "--outlet", outlet, "--flux", flux, "--reconstruction", reconstruction,
                         "--start", start, "--strategy", strategy]
            if args.first_step_fraction is not None:
                arguments += ["--first-step-fraction", str(args.first_step_fraction)]
                key += "__fraction_"+str(args.first_step_fraction).replace(".", "p")
            tasks.append((arguments, OUTPUT/"logs"/(key+".log")))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            errors += [future.result() for future in as_completed([pool.submit(launch, *task) for task in tasks])]
    assemble()
    if not args.no_report:
        from .report import build_report
        build_report()
    if any(errors):
        raise SystemExit("Workers raised errors; retained logs identify each unsuccessful launch")


if __name__ == "__main__":
    main()
