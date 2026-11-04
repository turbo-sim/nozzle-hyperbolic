"""Independent first-process and persistent-cache startup probes.

Uses a separate cache for each method, so no method receives executables from
another method. These diagnostic probes are separate from warm benchmarks.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .common import ROOT, STUDY, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs/startup_fmg_20261104"
    )
    parser.add_argument("--case", default="shock_30")
    parser.add_argument("--reconstruction", default="muscl_van_leer")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sequence", "fas_fmg"],
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "startup.json").exists():
        raise FileExistsError("Use a new startup probe directory")
    records = []
    for method in args.methods:
        cache = output / "caches" / method
        if cache.exists():
            raise FileExistsError("A first-use probe requires a new cache directory")
        for phase in ("empty_cache", "populated_cache"):
            target = output / method / phase
            env = os.environ.copy()
            env["JAX_COMPILATION_CACHE_DIR"] = str(cache)
            env["NOZZLE_MULTIGRID_ANCESTOR_PIDS"] = str(os.getpid())
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "development.multigrid.scripts.run",
                "--output",
                str(target),
                "--cases",
                args.case,
                "--reconstructions",
                args.reconstruction,
                "--methods",
                method,
                "--repeats",
                "0",
                "--no-report",
            ]
            started = perf_counter()
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            wall = perf_counter() - started
            summary_path = target / "summary.json"
            record = {
                "method_id": method,
                "phase": phase,
                "case_id": "euler_nozzle/" + args.case,
                "reconstruction": args.reconstruction,
                "process_wall_s": wall,
                "exit_code": result.returncode,
                "cache_directory": str(cache),
                "scope": "Separate process including imports, setup, first complete solve, assessment and serialization",
            }
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                group = summary["groups"][0]
                row = group["methods"][0]
                record.update(
                    shared_setup_s=group["shared_setup_s"],
                    method_setup_s=row["setup_s"],
                    shared_compilation=group["shared_compilation"],
                    setup_compilation=row["setup_compilation"],
                    first_solve=row["warmup"],
                    environment_verified=group.get("timing_environment_verified"),
                )
            records.append(record)
            save(output / "startup.json", {"records": records, "complete": False})
            if result.returncode:
                return result.returncode
    save(output / "startup.json", {"records": records, "complete": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
