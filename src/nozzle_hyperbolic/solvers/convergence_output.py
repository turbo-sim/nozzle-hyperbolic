"""Portable solver summaries, console histories and strict JSON export."""

import math
import sys
import json
import textwrap

import numpy as np


def strict_json(value):
    """Serialize a result mapping and classify nonfinite failure diagnostics.

    Nonfinite values become JSON null, with their original location and kind
    retained in ``nonfinite_values``. The signed numerical data are untouched;
    this is an export convention, not a monitoring floor or convergence rule.
    NumPy/JAX arrays and NumPy scalars are supported without importing JAX here.
    """
    if not isinstance(value, dict):
        raise TypeError("A convergence result must be a mapping")
    nonfinite = []

    def convert(item, location):
        if isinstance(item, np.ndarray) or (
            hasattr(item, "__array__") and not np.isscalar(item)
        ):
            return convert(np.asarray(item).tolist(), location)
        if isinstance(item, np.generic):
            return convert(item.item(), location)
        if isinstance(item, dict):
            return {
                str(key): convert(entry, f"{location}.{key}")
                for key, entry in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                convert(entry, f"{location}[{index}]")
                for index, entry in enumerate(item)
            ]
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite.append(
                dict(
                    path=location,
                    classification=(
                        "nan"
                        if math.isnan(item)
                        else "positive_infinity" if item > 0 else "negative_infinity"
                    ),
                )
            )
            return None
        return item

    result = convert(value, "$")
    previous = result.get("nonfinite_values", [])
    if not isinstance(previous, list):
        raise ValueError("nonfinite_values must be a list when present")
    result["nonfinite_values"] = [*previous, *nonfinite]
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


def solution_summary(solution):
    """Return consistent outcome, work counters and final physical diagnostics.

    Parameters
    ----------
    solution : Solution
        Steady result or transient snapshot with its cumulative attempt history.

    Returns
    -------
    dict
        Success/reason, reached time, accepted/rejected step counts, inner work
        including rejected attempts, and separate spatial, boundary, constraint,
        global-balance and transient content-budget diagnostics. Missing metrics
        remain None; nonfinite diagnostics are preserved for ``strict_json``.

    Notes
    -----
    Initial history rows do not count as accepted updates. Inner iterations
    retain the selected integrator's meaning: implicit flow Newton corrections
    or explicit stage storage/boundary corrections. Reduced implicit counts do
    not include nested boundary work. A transient global spatial imbalance is
    not its content-change budget or a steady convergence requirement.
    """
    rows = [row for row in solution.history if row.get("phase") != "initial"]
    accepted = [row for row in rows if row.get("accepted", False)]
    rejected = [row for row in rows if not row.get("accepted", False)]
    metrics = solution.metrics

    def maximum(name):
        if name not in metrics:
            return None
        values = np.asarray(metrics[name])
        return float(np.max(np.abs(values))) if values.size else 0.0

    return dict(
        success=bool(solution.success),
        reason=solution.reason,
        time_s=float(solution.time),
        elapsed_s=float(solution.elapsed_s),
        accepted_steps=len(accepted),
        rejected_steps=len(rejected),
        inner_iterations=sum(row.get("inner_iterations", 0) for row in rows),
        accepted_inner_iterations=sum(
            row.get("inner_iterations", 0) for row in accepted
        ),
        rejected_inner_iterations=sum(
            row.get("inner_iterations", 0) for row in rejected
        ),
        rejected_newton_trials=sum(
            row.get("rejected_newton_trials", 0) for row in rows
        ),
        final_actual_cfl=accepted[-1].get("cfl") if accepted else None,
        equation_norm=metrics.get("equation_norm"),
        equation_relative=metrics.get("equation_relative"),
        boundary_max=maximum("boundary_max"),
        constraint_max=maximum("constraint_max"),
        global_balance_max=maximum("global_defect"),
        conservation_budget_max=maximum("conservation_budget"),
    )


class ConvergenceTable:
    """At most 100 columns; equation reductions and separate absolute checks.

    R is the largest initial-relative norm among active equations, not a
    tolerance ratio. Integral monitors remain in saved diagnostics, not here.
    Equations are printed in groups of three so cell constraints remain visible.
    G is the maximum absolute entry in the solver's ``global_defect`` vector.
    Older callers may supply the scalar ``conservation_max`` instead.
    C is the independently scaled maximum cell-constraint defect.

    Parameters
    ----------
    equation_names : iterable of str
        Transported equation labels used by steady reporting.
    monitor_names : iterable of str, optional
        Retained caller metadata; engineering monitors stay in saved diagnostics.
    stream : file-like, optional
        Output stream, defaulting to current standard output.
    every : int, optional
        Print every nth accepted step, default one. Rejections always print.
    mode : {"steady", "transient"}, optional
        Steady mode prints spatial reductions and algebraic/global defects.
        Transient mode prints physical time, inner residual and integrated
        content-budget error, without presenting spatial balance as convergence.
    """

    def __init__(
        self, equation_names, monitor_names=(), *, stream=None, every=1, mode="steady"
    ):
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("Print interval must be a positive integer")
        if mode not in ("steady", "transient"):
            raise ValueError("Reporting mode must be steady or transient")
        self.mode = mode
        self.equations = tuple(equation_names)
        self.monitors = tuple(monitor_names)
        self.stream = sys.stdout if stream is None else stream
        self.every = every
        self.started = False
        self.last_step = None

    @staticmethod
    def _number(value):
        try:
            number = float(value)
            if not math.isfinite(number):
                return "nan" if math.isnan(number) else "+inf" if number > 0 else "-inf"
            for precision in (2, 1, 0):
                formatted = f"{number:.{precision}e}"
                if len(formatted) <= 8:
                    return formatted
        except (ValueError, TypeError, OverflowError):
            return "--"
        return "overflow"

    @staticmethod
    def _count(value, width):
        if value is None:
            return "--"
        try:
            number = int(value)
            text = str(number)
            if len(text) <= width:
                return text
            text = f"{number:.0e}".replace("e+0", "e").replace("e+", "e")
            return text if len(text) <= width else "*" * width
        except (ValueError, TypeError, OverflowError):
            return "--"

    def __call__(self, row, *, force=False):
        step = int(row.get("step", 0))
        initial = row.get("phase") == "initial"
        accepted = row.get("accepted", True)
        if not force and accepted and not initial and step % self.every:
            return
        if self.mode == "transient":
            if not self.started:
                print(
                    "Inner: stage/implicit residual; Budget: normalized content-change error.",
                    file=self.stream,
                )
                print(
                    "   Step       t [s]      CFL       dt  Nwt    Inner   Budget Event",
                    file=self.stream,
                )
            budget = row.get("budget")
            budget = None if budget is None else np.max(np.abs(np.asarray(budget)))
            values = [
                self._count(step, 7),
                self._number(row.get("time_s")),
                self._number(row.get("cfl")),
                self._number(row.get("dt")),
                self._count(row.get("inner_iterations"), 4),
                self._number(row.get("inner_residual")),
                self._number(budget),
                "initial" if initial else "accept" if accepted else "retry",
            ]
            widths = [7, 11, 8, 8, 4, 8, 8, 7]
            print(
                " ".join(value.rjust(width) for value, width in zip(values, widths)),
                file=self.stream,
                flush=True,
            )
            self.started, self.last_step = True, step
            return
        if not self.started:
            descriptions = [
                "; ".join(f"R{i+1}={name}" for i, name in enumerate(self.equations)),
                "R=max active Rk; Rk=norm/initial norm; --=initially satisfied (absolute check).",
                "BC, C, G: boundary, cell-constraint and global-balance defects; Nwt: inner corrections.",
            ]
            for description in descriptions:
                print(textwrap.fill(description, width=100), file=self.stream)
        relative = row.get("equation_relative", [None] * len(self.equations))
        active = row.get("relative_active", [True] * len(self.equations))
        selected = [value for value, flag in zip(relative, active) if flag]
        try:
            overall = (
                (
                    max(selected)
                    if all(math.isfinite(float(v)) for v in selected)
                    else float("nan")
                )
                if selected
                else None
            )
        except (TypeError, ValueError, OverflowError):
            overall = None
        conservation = row.get("conservation_max")
        if "global_defect" in row:
            defects = np.asarray(row["global_defect"])
            conservation = np.max(np.abs(defects)) if defects.size else None
        for start in range(0, max(1, len(self.equations)), 3):
            end = min(start + 3, len(self.equations))
            headers = [
                "Iter",
                "CFL",
                "dt",
                "Nwt",
                "R",
                *[f"R{i+1}" for i in range(start, end)],
                "BC",
                "C",
                "G",
                "Event",
            ]
            widths = [6, 8, 8, 3, 8, *([8] * (end - start)), 8, 8, 8, 7]
            if not self.started or len(self.equations) > 3:
                print(
                    " ".join(
                        label.rjust(width) for label, width in zip(headers, widths)
                    ),
                    file=self.stream,
                )
                print(" ".join("-" * width for width in widths), file=self.stream)
            values = [
                self._count(step, 6),
                self._number(row.get("cfl")),
                self._number(row.get("dt")),
                self._count(row.get("inner_iterations"), 3),
                self._number(overall),
                *[
                    self._number(relative[i]) if active[i] else "--"
                    for i in range(start, end)
                ],
                self._number(row.get("boundary_max")),
                self._number(row.get("constraint_max")),
                self._number(conservation),
                "initial" if initial else "accept" if accepted else "retry",
            ]
            print(
                " ".join(value.rjust(width) for value, width in zip(values, widths)),
                file=self.stream,
                flush=True,
            )
        self.started = True
        self.last_step = step
