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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = [json.loads(path.read_text(encoding="utf-8")) for path in out.glob("*.json") if path.name != "metadata.json"]
    fields = [row for row in records if "pressure" in row]
    if not fields:
        raise RuntimeError("No final fields available; inspect preserved exception records")
    cells = sorted(set(row["cells"] for row in fields))
    colors = dict(zip(METHODS, plt.cm.plasma(np.linspace(.2, .8, len(METHODS)))))
    styles = dict(zip(METHODS, ["-", "--", "-.", ":"]))
    fig, axes = plt.subplots(2, len(cells), figsize=(6 * len(cells), 8), squeeze=False)
    for column, count in enumerate(cells):
        rows = [row for row in fields if row["cells"] == count]
        reference = rows[0]
        axes[0, column].plot(reference["analytical_coordinate"], np.asarray(reference["analytical_pressure"]) / 1e5, "k-", lw=1.5, label="analytical point profile")
        for row in rows:
            label = row["limiter"] + (" (failed)" if not row["success"] else "")
            color, style = colors[row["limiter"]], styles[row["limiter"]]
            axes[0, column].plot(row["coordinate"], np.asarray(row["pressure"]) / 1e5, style, color=color, label=label)
            accepted = [item for item in row["history"] if item.get("accepted") and "equation_norm" in item]
            axes[1, column].semilogy([item["step"] for item in accepted], [max(item["equation_norm"]) for item in accepted], style, color=color, label=label)
        axes[0, column].set(xlabel="Axial coordinate [m]", ylabel="Pressure [bar]", title=f"Shock at 35% of diverging section; N={count}")
        axes[1, column].set(xlabel="Accepted updates [-]", ylabel="Maximum per-equation fixed-scale L2 [-]")
        for axis in axes[:, column]:
            axis.grid(True, alpha=.25)
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    columns = ("Limiter", "Cells", "Status", "Accepted / rejected", "Pressure L1/p0", "Max conserved L1/scale", "Shock error [m]", "Max global defect")
    table = []
    for row in sorted(records, key=lambda row: (row["cells"], row["limiter"])):
        table.append([row["limiter"], str(row["cells"]), row["reason"],
                      f'{row.get("accepted_steps", 0)} / {row.get("rejected_steps", 0)}',
                      f'{row.get("pressure_normalized_l1", float("nan")):.6g}',
                      f'{max(row.get("conservative_normalized_l1", [float("nan")])):.6g}',
                      f'{row.get("shock_error_m", float("nan")):.6g}',
                      f'{max(row.get("metrics", {}).get("global_defect", [float("nan")])):.6g}'])
    markdown = ["# Limiter comparison results", "", "Cold starts; shared perfect-gas nozzle shock_35 case. Full settings, histories, failures and source hashes are in the JSON records.", "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    markdown.extend("| " + " | ".join(row) + " |" for row in table)
    markdown.extend(["", "Field errors use analytical conserved cell averages and fixed physical scales; shock position uses the strongest downstream pressure gradient. Failure rows remain visible and do not establish converged discretization errors. Timings are compile-inclusive and concurrent-work contaminated, so no speed claim is made.", "", "![Profiles and convergence](comparison.png)"])
    (out / "RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    relative = Path(os.path.relpath(out.resolve(), STUDY)).as_posix()
    (STUDY / "RESULTS.md").write_text(
        ("\n".join(markdown) + "\n").replace("(comparison.png)", f"({relative}/comparison.png)"),
        encoding="utf-8",
    )
    image = base64.b64encode((out / "comparison.png").read_bytes()).decode("ascii")
    headers = "".join(f"<th>{html.escape(value)}</th>" for value in columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>" for row in table)
    data = base64.b64encode(json.dumps(records).encode()).decode()
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Limiter comparison</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1400px;color:#19212d}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:left}}img{{max-width:100%}}nav a{{margin-right:1rem}}p{{max-width:80ch}}</style>
<h1>Canonical nozzle limiter comparison</h1><nav><a href="#results">Results</a><a href="#profiles">Profiles and convergence</a><a href="#method">Method</a></nav>
<section id="results"><h2>Results</h2><table><tr>{headers}</tr>{rows}</table></section>
<section id="profiles"><h2>Profiles and convergence</h2><img alt="Pressure profiles and per-equation convergence" src="data:image/png;base64,{image}"></section>
<section id="method"><h2>Method and limits</h2><p>All methods use the same cold physical state, corrected Stadtke flux, safeguarded backward Euler, fixed residual policy and zero additional filter. The reference is the packaged perfect-gas shock_35 analytical solution. Errors compare conserved cell projections, not a hidden analytical initialization. Refinement results apply to this case only. All failures remain in the table and downloads.</p><p>Minmod is least compressive; Superbee selects the largest signed magnitude correctly for either sign. The selected Van Albada has an explicit extrema gate and safe inactive arithmetic. These piecewise smooth limiters do not guarantee Newton robustness or positivity for a coupled fluid system.</p><p>JSON records retain per-equation norms, budgets, boundary defects and final fields. Compile-inclusive elapsed times are not eligible for a speed comparison.</p><a download="limiter_results.json" href="data:application/json;base64,{data}">Download all result records</a></section></html>'''
    (STUDY / "REPORT.html").write_text(document, encoding="utf-8")


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
