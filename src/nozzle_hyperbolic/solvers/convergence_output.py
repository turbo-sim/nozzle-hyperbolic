"""Plain-text convergence table reusable by model adapters and case runners."""

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
        if isinstance(item, np.ndarray) or (hasattr(item, "__array__") and not np.isscalar(item)):
            return convert(np.asarray(item).tolist(), location)
        if isinstance(item, np.generic):
            return convert(item.item(), location)
        if isinstance(item, dict):
            return {str(key): convert(entry, f"{location}.{key}") for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(entry, f"{location}[{index}]") for index, entry in enumerate(item)]
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite.append(dict(path=location, classification="nan" if math.isnan(item) else
                                  "positive_infinity" if item > 0 else "negative_infinity"))
            return None
        return item

    result = convert(value, "$")
    previous = result.get("nonfinite_values", [])
    if not isinstance(previous, list):
        raise ValueError("nonfinite_values must be a list when present")
    result["nonfinite_values"] = [*previous, *nonfinite]
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


class ConvergenceTable:
    """At most 100 columns; equation reductions and separate absolute checks.

    R is the largest initial-relative norm among active equations, not a
    tolerance ratio. Integral monitors remain in saved diagnostics, not here.
    More than four equations are printed in labelled groups of four.
    """

    def __init__(self, equation_names, monitor_names=(), *, stream=None, every=1):
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("Print interval must be a positive integer")
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
        if not self.started:
            descriptions = ["; ".join(f"R{i+1}={name}" for i, name in enumerate(self.equations)),
                "R=max active Rk; Rk=norm/initial norm; --=initially satisfied (absolute check).",
                "BC, G: absolute boundary and global-balance defects; Nwt: Newton corrections."]
            for description in descriptions:
                print(textwrap.fill(description, width=100), file=self.stream)
        relative = row.get("equation_relative", [None] * len(self.equations))
        active = row.get("relative_active", [True] * len(self.equations))
        selected = [value for value, flag in zip(relative, active) if flag]
        try:
            overall = (max(selected) if all(math.isfinite(float(v)) for v in selected)
                       else float("nan")) if selected else None
        except (TypeError, ValueError, OverflowError):
            overall = None
        for start in range(0, max(1, len(self.equations)), 4):
            end = min(start+4, len(self.equations))
            headers = ["Iter", "CFL", "dt", "Nwt", "R",
                       *[f"R{i+1}" for i in range(start, end)], "BC", "G", "Event"]
            widths = [6, 8, 8, 3, 8, *([8]*(end-start)), 8, 8, 7]
            if not self.started or len(self.equations) > 4:
                print(" ".join(label.rjust(width) for label, width in zip(headers, widths)), file=self.stream)
                print(" ".join("-"*width for width in widths), file=self.stream)
            values = [self._count(step, 6), self._number(row.get("cfl")), self._number(row.get("dt")),
                      self._count(row.get("inner_iterations"), 3), self._number(overall),
                      *[self._number(relative[i]) if active[i] else "--" for i in range(start, end)],
                      self._number(row.get("boundary_max")), self._number(row.get("conservation_max")),
                      "initial" if initial else "accept" if accepted else "retry"]
            print(" ".join(value.rjust(width) for value, width in zip(values, widths)),
                  file=self.stream, flush=True)
        self.started = True
        self.last_step = step
