"""Annular 50/100/200/400 grid sequencing with unchanged physical equations.

The multigrid study's bounded primitive interpolation is used only to initialize
the next mesh. Each saved mesh is independently converged to the same original
root target, because every mesh contributes to the reported accuracy curves.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
import jaxprop as jxp
import numpy as np

from .verify_annular import (CASES, DEFAULT_CASES, DEFAULT_MESHES, RECONSTRUCTIONS,
                            STUDY, fingerprints, make_case, report, verify, write_json)
from .case_config import DEFAULT_CONFIG, build_case, load_case_config


def transfer(previous, new_points):
    """Bounded linear interpolation of p, v_m, ell, h including physical traces."""
    old_points = np.r_[previous["m_faces"][0], previous["m"], previous["m_faces"][-1]]
    return {method: np.column_stack([
                np.interp(new_points, old_points, np.asarray(run["state"])[:, column])
                for column in range(4)])
            for method, run in previous["methods"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--cells", nargs="+", type=int, default=list(DEFAULT_MESHES))
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controller", choices=("equation_monitored", "cfl_recommended"), default="equation_monitored")
    parser.add_argument("--output", type=Path, default=STUDY/"outputs/grid_sequence_monitored_full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.cells != sorted(set(args.cells)) or min(args.cells)<2:
        parser.error("Use a strictly increasing sequence of physical cell counts")
    args.output.mkdir(parents=True, exist_ok=True)
    source_before = fingerprints()
    results = []
    if args.aggregate_only:
        for name in args.cases:
            for count in args.cells:
                results.append(json.loads((args.output/f"{name}_{count}.json").read_text(encoding="utf-8")))
    else:
        fluid = jxp.FluidPerfectGas(name="Air")
        for name in args.cases:
            case = build_case(name, fluid, config_path=args.config)
            if args.controller == "equation_monitored" and case.config["residuals"]["absolute_tolerance"] != args.tolerance:
                raise ValueError("--tolerance must match the YAML absolute residual safeguard")
            previous, previous_path, failed_path = None, None, None
            for count in args.cells:
                path = args.output/f"{name}_{count}.json"
                mesh = case.geometry.mesh(count)
                points = np.r_[mesh.m_faces[0], mesh.m_centers, mesh.m_faces[-1]]
                seeds = transfer(previous, points) if previous is not None else None
                metadata = dict(strategy="uniform cold physical guess" if previous is None else
                                 "bounded linear primitive interpolation",
                    hierarchy=args.cells, level=args.cells.index(count), target_tolerance=args.tolerance,
                    interpolation_variables=["p","v_m","ell","h"],
                    physical_time_remap=False, defect_forcing=False, cfl_reset_each_mesh=True,
                    parent_file=previous_path.name if previous_path else None,
                    parent_sha256=sha256(previous_path.read_bytes()).hexdigest() if previous_path else None)
                if args.controller == "equation_monitored":
                    metadata["residual_normalization"] = "fresh initial equation norms on each mesh"
                    metadata["relative_tolerance"] = case.config["residuals"]["relative_tolerance"]
                if failed_path is not None:
                    metadata.update(strategy="uniform cold restart after failed coarser stage",
                        failed_coarser_file=failed_path.name,
                        failed_coarser_sha256=sha256(failed_path.read_bytes()).hexdigest())
                if args.resume and path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if data["case"] != name or data["cells"] != count:
                        raise ValueError(f"Saved case identity differs: {path}")
                    if data["source_hashes"] != source_before or data.get("initialization") != metadata:
                        raise ValueError(f"Saved source/transfer metadata differs: {path}")
                    for key, value in dict(tolerance=args.tolerance,max_steps=args.max_steps,cfl=.2,
                                           controller=args.controller,delta_factor=.1,
                                           flux_method="characteristic").items():
                        if data["settings"].get(key) != value:
                            raise ValueError(f"Saved {key} differs: {path}")
                    if args.controller == "equation_monitored" and (
                            data.get("case_config") != case.config
                            or data.get("case_config_provenance") != case.config_provenance):
                        raise ValueError(f"Saved YAML configuration differs: {path}")
                    if seeds is not None:
                        for method in RECONSTRUCTIONS:
                            if not np.array_equal(seeds[method], data["methods"][method]["initial_state"]):
                                raise ValueError(f"Saved interpolation differs: {path} {method}")
                    print(f"Reusing completed grid stage {name} N={count}; no extra solve",flush=True)
                else:
                    data = verify(case, fluid, count, args.output, args.max_steps,
                                  args.tolerance, .2, mesh=mesh, controller=args.controller,
                                  initial_states=seeds, initialization_metadata=metadata)
                results.append(data)
                report(results,args.output)
                if not all(run["success"] for run in data["methods"].values()):
                    print(f"Unconverged stage retained at {path}; next mesh restarts cold", flush=True)
                    previous, previous_path, failed_path = None, None, path
                else:
                    previous, previous_path, failed_path = data, path, None
                jax.clear_caches()
    if fingerprints() != source_before:
        raise RuntimeError("Annular numerical source changed during the campaign")
    for data in results:
        if data["source_hashes"] != source_before:
            raise ValueError(f"Saved source differs for {data['case']} N={data['cells']}")
        if data["case"] not in args.cases or data["cells"] not in args.cells:
            raise ValueError("Aggregated data contains an unexpected case or mesh")
        if (data["initialization"]["hierarchy"] != args.cells
                or data["settings"]["tolerance"] != args.tolerance
                or data["settings"]["max_steps"] != args.max_steps):
            raise ValueError("Aggregated data settings differ from the requested campaign")
    report(results,args.output)
    roots = all(run["success"] for data in results for run in data["methods"].values())
    finest = [data for data in results if data["cells"]==max(args.cells)]
    accuracy = all(run["analytical_target_met"] for data in finest for run in data["methods"].values())
    write_json(args.output/"acceptance.json",dict(all_roots=roots,finest_grid_accuracy=accuracy,
        success=roots and accuracy,case_mesh_files=len(results),numerical_solutions=2*len(results),
        grid_sequence=args.cells,each_level_tolerance=args.tolerance,source_unchanged=True,
        absolute_safeguard=args.tolerance,
        relative_tolerance=(results[0]["settings"]["residual_policy"]["relative_tolerance"]
                            if args.controller=="equation_monitored" else None),
        failed_solutions=[dict(case=data["case"],cells=data["cells"],method=method,
                              reason=run.get("monitoring", {}).get("reason", run["message"]))
                          for data in results for method,run in data["methods"].items() if not run["success"]],
        strategy="bounded primitive initialization from successful coarser stages; cold restart after any failed stage"))
    print(f"Grid sequence: {2*len(results)} solves, all_roots={roots}, finest_accuracy={accuracy}",flush=True)
    return 0 if roots and accuracy else 1


if __name__=="__main__":
    raise SystemExit(main())
