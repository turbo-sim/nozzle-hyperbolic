"""Physical annular observables; numerical convergence tracking stays shared."""

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np


class AnnularQuantities:
    """Resolve named pressure coefficient and actual numerical mass-flow values.

    The callback accepts ``(full_state, weighted_numerical_face_flux)`` and
    returns values keyed by monitor name. It owns no solver or time driver.
    Pressure coefficient uses the instantaneous inlet trace and a fixed total
    pressure. Supersonic inlet data must provide that total pressure explicitly.
    """

    def __init__(self, model, monitors=(), *, pressure_total=None):
        self.model = model
        self.monitors = tuple(monitors)
        if any(
            not all(
                hasattr(spec, key)
                for key in ("name", "quantity", "reduction", "parameters")
            )
            for spec in self.monitors
        ):
            raise TypeError(
                "monitors must contain named physical quantity specifications"
            )
        if len({spec.name for spec in self.monitors}) != len(self.monitors):
            raise ValueError("Monitor names must be unique")
        self.resolved = []
        for spec in self.monitors:
            quantity = {"cp": "pressure_coefficient", "mdot": "mass_flow"}.get(
                spec.quantity, spec.quantity
            )
            if quantity == "pressure_coefficient":
                if spec.reduction not in (
                    "inlet",
                    "outlet",
                    "length_mean",
                    "length_integral",
                ):
                    raise ValueError(f"Unsupported Cp reduction {spec.reduction!r}")
                p0 = spec.parameters.get("pressure_total", pressure_total)
                if p0 is None:
                    p0 = model.inlet_data.get("p0")
                if p0 is None or not np.isfinite(p0) or p0 <= 0:
                    raise ValueError(
                        "Cp monitoring needs a finite positive pressure_total"
                    )
            elif quantity == "mass_flow":
                if spec.reduction not in ("inlet", "outlet", "mean"):
                    raise ValueError(
                        f"Unsupported mass-flow reduction {spec.reduction!r}"
                    )
                p0 = None
            else:
                raise ValueError(f"Unknown annular monitor quantity {spec.quantity!r}")
            self.resolved.append((quantity, spec.reduction, p0))
        self.metadata = [
            dict(**asdict(spec), resolved_quantity=quantity, resolved_pressure_total=p0)
            for spec, (quantity, _, p0) in zip(self.monitors, self.resolved)
        ]
        self._widths = jnp.asarray(model.mesh.widths)
        self._length = float(np.sum(model.mesh.widths))

    def __call__(self, full, face_flux):
        values = {}
        for spec, (quantity, reduction, p0) in zip(self.monitors, self.resolved):
            if quantity == "pressure_coefficient":
                pin = full[0, 0]
                denominator = p0 - pin
                safe = jnp.where(denominator > 0, denominator, 1.0)
                cp = (full[1:-1, 0] - pin) / safe
                if reduction == "inlet":
                    value = jnp.asarray(0.0)
                elif reduction == "outlet":
                    value = (full[-1, 0] - pin) / safe
                elif reduction == "length_mean":
                    value = jnp.sum(self._widths * cp) / self._length
                else:
                    value = jnp.sum(self._widths * cp)
                value = jnp.where(denominator > 0, value, jnp.nan)
            elif reduction == "inlet":
                value = face_flux[0, 0]
            elif reduction == "outlet":
                value = face_flux[-1, 0]
            else:
                value = (
                    jnp.sum(self._widths * (face_flux[:-1, 0] + face_flux[1:, 0]) / 2)
                    / self._length
                )
            values[spec.name] = value
        return values
