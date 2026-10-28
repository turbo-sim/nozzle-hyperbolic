"""Verify the same annular model on a prescribed graded arclength mesh.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.verify_nonuniform

The existing verification driver owns case setup, independent cell-integrated
references, both cold solves, field/error/conservation audits and three-curve
plots. This wrapper changes only the face distribution and records its recipe.
"""

import argparse
from hashlib import sha256
from pathlib import Path

import jaxprop as jxp
import numpy as np

from .verify_annular import REPOSITORY, STUDY, make_case, report, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--mesh-exponent", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--controller", choices=("native", "cfl_recommended"), default="native")
    parser.add_argument("--output", type=Path, default=STUDY/"outputs"/"nonuniform")
    parser.add_argument("--root-only", action="store_true",
                        help="Require the original root targets; still report any analytical target failure")
    args = parser.parse_args()
    if args.cfl is None:
        args.cfl = 0.2 if args.controller == "cfl_recommended" else 20.0
    if args.cells < 2 or not np.isfinite(args.mesh_exponent) or args.mesh_exponent <= 0:
        parser.error("Use at least two physical cells and a finite positive mesh exponent")
    args.output.mkdir(parents=True, exist_ok=True)
    fluid = jxp.FluidPerfectGas(name="Air")
    case = make_case("radial_outflow", fluid)
    faces = case.geometry.length*np.linspace(0.0, 1.0, args.cells+1)**args.mesh_exponent
    mesh = case.geometry.mesh(m_faces=faces)
    result = verify(case, fluid, args.cells, args.output, args.max_steps,
                    args.tolerance, args.cfl, mesh=mesh, controller=args.controller)
    result["settings"].update(
        mesh_definition="m_j=L*(j/N)**mesh_exponent", mesh_exponent=args.mesh_exponent)
    result["verification_driver"] = dict(
        module="development.roberto.euler_annular.scripts.verify_nonuniform",
        sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    adapter = REPOSITORY/"src/nozzle_hyperbolic/thermodynamics/adapters.py"
    if adapter.is_file():
        result["source_hashes"][str(adapter.relative_to(REPOSITORY))] = sha256(adapter.read_bytes()).hexdigest()
    write_json(args.output/f"{case.name}_{args.cells}.json", result)
    report([result], args.output)
    roots = all(run["success"] for run in result["methods"].values())
    accuracy = all(run["analytical_target_met"] for run in result["methods"].values())
    acceptance = dict(
        case=case.name, cells=args.cells, mesh_exponent=args.mesh_exponent,
        root_only=args.root_only, all_roots=roots, analytical_targets_met=accuracy,
        verification_acceptance=roots and accuracy,
        success=roots and (args.root_only or accuracy))
    write_json(args.output/"acceptance.json", acceptance)
    print(f"Nonuniform audit: roots={roots}, analytical targets={accuracy}", flush=True)
    return 0 if acceptance["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
