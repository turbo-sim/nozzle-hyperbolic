"""Scalar engineering monitors, independent of a particular flow solver.

The model adapter evaluates quantities. This module owns named configuration,
accepted-state windows and optional stationarity checks, not physical formulas.
"""

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MonitorSpec:
    name: str
    quantity: str
    reduction: str
    parameters: dict = field(default_factory=dict)
    convergence: bool = False
    window: int = 20
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 0.0
    reference: float = 1.0

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.name, self.quantity, self.reduction)):
            raise ValueError("Monitor name, quantity and reduction must be nonempty strings")
        if not isinstance(self.parameters, dict):
            raise ValueError("Monitor parameters must be a mapping")
        if not isinstance(self.convergence, bool):
            raise ValueError("Monitor convergence must be a boolean")
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            raise ValueError("Monitor window must be an integer of at least two samples")
        for name in ("absolute_tolerance", "relative_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Monitor {name} must be finite and nonnegative")
        if self.absolute_tolerance == self.relative_tolerance == 0:
            raise ValueError("At least one monitor tolerance must be positive")
        if not math.isfinite(self.reference) or self.reference <= 0:
            raise ValueError("Monitor reference must be finite and positive")


class MonitorTracker:
    """Peak-to-peak stationarity over a complete window of accepted samples.

    Threshold = absolute_tolerance + relative_tolerance * fixed reference.
    Reference is specified at setup, never the current value (which can cross
    zero). Optional checks supplement equation/BC/global convergence.
    """

    def __init__(self, specs=()):
        self.specs = tuple(specs)
        if len({spec.name for spec in self.specs}) != len(self.specs):
            raise ValueError("Monitor names must be unique")
        self.windows = {spec.name: deque(maxlen=spec.window) for spec in self.specs}
        self.diagnostics = {}
        self.passed = not self.specs

    def update(self, values):
        if set(values) != set(self.windows):
            raise ValueError("Monitor values must match all configured monitor names")
        diagnostics = {}
        for spec in self.specs:
            value = float(values[spec.name])
            samples = self.windows[spec.name]
            samples.append(value)
            finite = all(math.isfinite(item) for item in samples)
            span = max(samples) - min(samples) if finite else None
            threshold = spec.absolute_tolerance + spec.relative_tolerance * spec.reference
            stable = bool(finite and len(samples) == spec.window and span <= threshold)
            diagnostics[spec.name] = dict(value=value, range=span, threshold=threshold,
                stable=stable, samples=len(samples), required=spec.convergence)
        self.diagnostics = diagnostics
        self.passed = all(math.isfinite(row["value"]) and (not row["required"] or row["stable"])
                          for row in diagnostics.values())
        return diagnostics
