"""Report the matched implicit-boundary benchmark and ten-run timing statistics."""

import argparse
import json
import os
from pathlib import Path

from development_new.boundary_conditions.scripts.plot_implicit import LABELS, plot


def write_report(output):
    data = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    plot(output)
    root = Path(__file__).resolve().parents[1]
    relative = Path(os.path.relpath(output.resolve(), root)).as_posix()
    measured = [r for c in data["cases"] for r in c["measured_runs"]]
    excluded = [r for c in data["cases"] for r in c["excluded_runs"]]
    text = f"""# Matched implicit-boundary benchmark

{sum(r['success'] for r in measured)} of {len(measured)} measured solves reached the
original steady and boundary targets on the {data['settings']['points']}-point mesh.
There are exactly two methods across five backpressures. Each method/case has
**one excluded full compilation run, then {data['settings']['runs']} measured runs**.
The {len(excluded)} excluded runs do not enter any timing statistic.

Both methods now use the **same source-native JAX Newton loop algorithm,
colored block Jacobian assembly, and block-banded Thomas solver code**.
The old PySolverView dense inner solver has been removed. The inspected source
path does not use Optimistix; introducing it would add a different solver rather
than matching the current source implementation.

The current source timestep kernel is called directly for internal BCs. The
experimental kernel reuses the source residual/assembly/linear-solver code with
six additional ghost unknowns and algebraic boundary rows. Both use the same
relative-update stopping formula, maximum of eight inner Newton iterations,
CFL/SER and increment/retry controls. Ghost rows have zero storage. Exact
reservoir equations replace the finite three-update inlet inversion; no nested
inlet root solve is called by the coupled residual or shared monitor.

## Timing: minimum, maximum and mean

![Ten-run warm full-solve timing distributions]({relative}/timings.png)

[Vector PDF]({relative}/timings.pdf).

Markers show means, whiskers span minimum to maximum, and faint points show
every measured run. All starts are cold physical starts; warm timing means
compiled kernels are reused, not that a solution is warm-started. Both methods
use the identical initial full field, reset controller state on every run,
and alternate execution order between repeats. No method is given a previous
solution. First-run histories/statuses are retained separately.

| Case | Method | Measured runs | Minimum [s] | Maximum [s] | Mean [s] | All roots verified? |
|---|---|---:|---:|---:|---:|---|
"""
    for case in data["cases"]:
        for mode in data["methods"]:
            stats = case["timings"][mode]
            text += (
                f"| {case['name']} | {LABELS[mode]} | {stats['count']} | "
                f"{stats['min_s']:.4f} | {stats['max_s']:.4f} | {stats['mean_s']:.4f} | "
                f"{'Yes' if stats['all_success'] else 'No'} |\n"
            )
    text += "\n| Case | Coupled/source mean-time ratio |\n|---|---:|\n"
    for case in data["cases"]:
        ratio = (
            case["timings"]["coupled"]["mean_s"]
            / case["timings"]["production"]["mean_s"]
        )
        text += f"| {case['name']} | {ratio:.3f} |\n"
    text += f"""
Timing includes synchronized timestep work, both methods' identical monitor,
controller and rejected trials. Geometry/reference construction, kernel setup,
final audits, file writing and plotting are outside solve time. The first
complete solve of each method/case is discarded even if some kernels were
already compiled by prior initialization. Kernels are built once per case and
are not recreated between measured runs. Device results are materialized before
stopping the timer. All measured solves are sequential, with one BLAS thread.
These are local-machine observations, not portable performance guarantees.

## Convergence: five backpressures

![Five-panel convergence history comparison]({relative}/convergence_histories.png)

[Vector PDF]({relative}/convergence_histories.pdf).

One actual median-duration measured run per method/case supplies each plotted
trajectory; residual histories are not averaged or selected by best speed.
Each subplot fixes backpressure and contains one line per boundary treatment.
The common residual is the maximum of the fixed-scale interior spatial RMS
and dimensionless boundary RMS; the dotted target is $10^{{-8}}$. Both methods
use the same JIT monitor on their actual ghost-inclusive state. No internal
inlet inversion is evaluated merely to monitor the coupled solution.

## Final pressure: five backpressures

![Five-panel final pressure comparison]({relative}/pressure_distributions.png)

[Vector PDF]({relative}/pressure_distributions.pdf).

The same representative measured runs supply actual pressure fields, excluding
ghosts. Dotted analytic curves are used for verification only, never as seeds.
The sonic-limit discrepancy is preserved in the analytic checks.

| Case | Method | Accepted updates | Rejected trials | Final RMS | Analytic check |
|---|---|---:|---:|---:|---|
"""
    for case in data["cases"]:
        for run in case["runs"]:
            text += (
                f"| {case['name']} | {LABELS[run['mode']]} | {run['accepted_updates']} | "
                f"{run['rejected_trials']} | ${run['history'][-1]['residual_rms']:.3g}$ | "
                f"{'Pass' if run['diagnostics']['analytical_pass'] else 'Fail'} |\n"
            )
    text += "\n| Case | Maximum pressure difference [Pa] | Scaled field RMS difference |\n|---|---:|---:|\n"
    for case in data["cases"]:
        text += f"| {case['name']} | ${case['maximum_pressure_difference_Pa']:.3g}$ | ${case['scaled_state_difference_rms']:.3g}$ |\n"
    text += r"""
## Interpretation and limits

The former order-of-magnitude gap mixed compiled banded source solves with a
Python/custom dense solver, so it could not establish a boundary-formulation
cost. This benchmark removes that mismatch. Remaining differences include the
six additional algebraic variables, two extra cell rows/superblock grouping,
boundary equations/EOS evaluations, and any resulting Newton/retry work.
They are consequences of the boundary treatment within a matched implementation.

The relative-update stopping formula is applied to all unknowns: interior
variables in source, and interior plus ghosts in the coupled system. Iterates
and factorization roundoff therefore need not be bitwise identical. Both use
the source colored JVP and block-Thomas algorithms, including within-superblock
pivoting rather than a different globally pivoted dense method. Independent
dense AD/solve tests verify the enlarged stencil and linear corrections.

Eight inner updates are a cap, not a proof that every transient nonlinear
system is solved exactly. Native source behavior is retained rather than adding
an experimental least-squares/root tolerance. Final convergence is independently
audited against original spatial equations, boundary constraints, mass flow
and analytic profiles. The comparison uses a shared fixed-scale steady stop,
so standalone driver SSR/plateau counts can differ (56 versus 58 for subsonic
flow in the baseline study).

The coupled inlet solves exact reservoir relations instead of three finite
pressure updates and does not replicate inlet enthalpy clipping outside the
tested admissible branch. This limitation is unchanged. No general EOS/grid/
initialization robustness theorem follows from these five cases.

## Data and source integrity

All `src` Python hashes match before and after the benchmark. No source file
was edited or monkeypatched. The read-only adapter rebinds private source
closures in new experimental function objects. Only residual/boundary equations
and cell-count bindings change; assembly and linear-solver code are shared.
The current source snapshot and tests guard against silently stale assumptions.
"""
    text += (
        f"\n[All {len(measured)} timing samples]({relative}/timings.csv), "
        f"[full measured and excluded histories, fields and source hashes]({relative}/summary.json). "
        "See [README.md](README.md) and [BC_IMPLEMENTATION.md](BC_IMPLEMENTATION.md).\n"
    )
    name = (
        "IMPLICIT_REPORT.md"
        if output.name == "steady"
        else f"{output.name.upper()}_REPORT.md"
    )
    (root / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "steady",
    )
    write_report(parser.parse_args().output)
