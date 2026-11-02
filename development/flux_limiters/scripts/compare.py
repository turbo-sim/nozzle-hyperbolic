"""Cold-start limiter comparison; rebuild plots/report with --report-only.

Run from the repository root with the project's Poetry environment. This is a
numerical accuracy/convergence study, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
ANDREA_SHA = "bcfe6ae1ec8abd71528e57b218ac8291dc9461fa"
METHODS = ("van_leer", "minmod", "superbee", "van_albada")


def serial(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial), encoding="utf-8")


def fingerprint():
    paths = [
        "src/nozzle_hyperbolic/numerics/reconstruction.py",
        "src/nozzle_hyperbolic/solvers/fvm/operator.py",
        "src/nozzle_hyperbolic/solvers/fvm/config.py",
        "src/nozzle_hyperbolic/solvers/fvm/drivers.py",
        "src/nozzle_hyperbolic/models/euler_nozzle/model.py",
        "src/nozzle_hyperbolic/solvers/time_marching/implicit.py",
    ]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def run(args, out):
    if any(out.glob("*.json")):
        raise FileExistsError("Retain prior attempt and provenance; choose a fresh --output directory")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, solve_single_grid
    from nozzle_hyperbolic.verification import build_case, load_case

    definition = load_case("euler_nozzle/shock_35")
    settings = SteadySettings(max_steps=args.max_steps)
    metadata = dict(
        purpose="convergence and accuracy; elapsed times are compile-inclusive and not a performance comparison",
        case=definition.to_dict(), case_sha256=definition.source_sha256,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=fingerprint(), andrea_reference=ANDREA_SHA,
        python=sys.version, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "jaxprop", "numpy")},
        backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
        settings=asdict(settings), initial_state="model-owned cold state independently for every method and grid",
    )
    dump(out / "metadata.json", metadata)
    for cells in args.cells:
        problem = build_case(definition, cells=cells)
        reference = problem.cell_reference()
        dense_x = np.linspace(problem.mesh.faces[0], problem.mesh.faces[-1], 1501)
        dense_reference = problem.reference_fields(dense_x)
        for limiter in args.methods:
            path = out / f"{limiter}_{cells}.json"
            if path.exists():
                raise FileExistsError(f"Retain prior attempt; choose a fresh --output directory: {path}")
            scheme = Scheme(reconstruction="muscl_" + limiter, flux="stadtke", newton_backtracking=True)
            record = dict(cells=cells, limiter=limiter, scheme=asdict(scheme), success=False)
            try:
                solution = solve_single_grid(problem.model, scheme, settings, initial_state=problem.initial_state)
                fields = problem.numerical_fields(solution.state)
                q = np.asarray(problem.model.cell_conserved(jnp.asarray(solution.state[1:-1])))
                weights = np.asarray(problem.mesh.volumes) / np.sum(problem.mesh.volumes)
                q_l1 = np.sum(weights[:, None] * np.abs(q - reference["q"]), axis=0) / problem.model.conservative_scales
                p_l1 = float(np.sum(weights * np.abs(fields["p"] - reference["fields"]["p"])) / problem.model.p0)
                # Locate the strongest pressure recovery downstream of the throat.
                centers = np.asarray(problem.mesh.centers)
                midpoints = .5 * (centers[:-1] + centers[1:])
                gradients = np.diff(fields["p"]) / np.diff(centers)
                shock_index = int(np.argmax(np.where(midpoints > .5 * problem.geometry.length, gradients, -np.inf)))
                shock = float(midpoints[shock_index])
                record.update(
                    success=bool(solution.success), reason=solution.reason, metrics=solution.metrics,
                    history=solution.history, elapsed_compile_inclusive_s=solution.elapsed_s,
                    state=solution.state, coordinate=centers, pressure=fields["p"], mach=fields["M"] if "M" in fields else fields["u"] / fields["a"],
                    reference_pressure=reference["fields"]["p"],
                    analytical_coordinate=dense_x, analytical_pressure=dense_reference["p"],
                    conservative_normalized_l1=q_l1, pressure_normalized_l1=p_l1,
                    shock_location_m=shock, reference_shock_location_m=problem.shock_position,
                    shock_error_m=abs(shock - problem.shock_position),
                    accepted_steps=sum(row.get("phase") == "accepted" for row in solution.history),
                    rejected_steps=sum(row.get("phase") == "rejected" for row in solution.history),
                )
                print(f"{limiter:11} N={cells}: {solution.reason}; pressure L1/p0={p_l1:.5e}", flush=True)
            except Exception as exc:
                record.update(reason="exception", error=repr(exc), traceback=traceback.format_exc())
                print(f"{limiter:11} N={cells}: {exc!r}", flush=True)
            dump(path, record)


def report(out):
    """Delegate presentation to the saved-data Plotly renderer."""
    from development.flux_limiters.scripts.report import build_report
    build_report(out)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs" / "comparison")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run(args, args.output)
    report(args.output)


if __name__ == "__main__":
    main()
