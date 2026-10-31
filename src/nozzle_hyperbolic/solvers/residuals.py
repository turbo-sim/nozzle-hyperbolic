"""Model-independent norms of signed, integrated steady conservation rows.

These reductions never enter the residual differentiated by Newton. Reference
scales and weights are fixed at setup; neither CFL nor evolving state is used.
"""

from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np


DEFINITION = "steady_spatial_fixed_scale_equation_v1"


@dataclass(frozen=True)
class ResidualPolicy:
    norm: str = "l2"
    weighting: str = "volume"
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-9
    boundary_tolerance: float = 1e-9
    global_tolerance: float = 1e-9
    initial_floor: float = 1e-14
    criterion: str = "either"

    def __post_init__(self):
        if self.norm not in ("l1", "l2"):
            raise ValueError("Residual norm must be l1 or l2")
        if self.weighting not in ("volume", "cell"):
            raise ValueError("Residual weighting must be volume or cell")
        if self.criterion not in ("either", "both", "relative", "absolute"):
            raise ValueError("criterion must be either, both, relative or absolute")
        for name in (
            "relative_tolerance",
            "absolute_tolerance",
            "boundary_tolerance",
            "global_tolerance",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.initial_floor) or self.initial_floor < 0:
            raise ValueError("initial_floor must be finite and nonnegative")


def validate_residual_setup(volumes, conservative_scales, reference_time):
    """Host-side setup validation, called before tracing a reduction kernel."""
    for name, values in (
        ("volumes", volumes),
        ("conservative_scales", conservative_scales),
    ):
        array = np.asarray(values, dtype=float)
        if (
            array.ndim != 1
            or not array.size
            or not np.all(np.isfinite(array) & (array > 0))
        ):
            raise ValueError(f"{name} must be a nonempty positive finite vector")
    if not np.isfinite(reference_time) or reference_time <= 0:
        raise ValueError("reference_time must be finite and positive")


def reduce_residuals(
    integrated_residual,
    volumes,
    conservative_scales,
    reference_time,
    *,
    weighting="volume",
):
    """Return per-equation mean absolute, RMS, maximum and worst-cell index.

    Inputs are (N,M) signed integrated rows, (N,) physical volumes and (M,)
    conservative density scales. ``reference_time`` is L_ref/U_ref. Validate
    physical setup with :func:`validate_residual_setup` outside JIT.
    """
    residual = jnp.asarray(integrated_residual)
    volumes, scales = jnp.asarray(volumes), jnp.asarray(conservative_scales)
    if (
        residual.ndim != 2
        or volumes.shape != (residual.shape[0],)
        or scales.shape != (residual.shape[1],)
    ):
        raise ValueError(
            "Residual, volumes and scales must have shapes (N,M), (N,), (M,)"
        )
    if weighting == "volume":
        weights = volumes / jnp.sum(volumes)
    elif weighting == "cell":
        weights = jnp.full(
            volumes.shape, 1.0 / residual.shape[0], dtype=jnp.result_type(residual, 1.0)
        )
    else:
        raise ValueError("Residual weighting must be volume or cell")
    scaled = residual * reference_time / (volumes[:, None] * scales)
    absolute = jnp.abs(scaled)
    # Divide by each equation's maximum before squaring to avoid overflow for
    # large but finite residuals. Exact zero equations stay exactly zero.
    largest = jnp.max(absolute, axis=0)
    safe = jnp.where(largest > 0, largest, 1.0)
    rms = largest * jnp.sqrt(jnp.sum(weights[:, None] * (scaled / safe) ** 2, axis=0))
    return dict(
        l1=jnp.sum(weights[:, None] * absolute, axis=0),
        l2=rms,
        linf=largest,
        worst_cell=jnp.argmax(absolute, axis=0),
        finite=jnp.all(jnp.isfinite(scaled)),
    )


def assess_residuals(norms, initial_norms, policy):
    """Assess every equation using the explicitly selected stopping criterion.

    Equations initially at/below ``initial_floor`` use the absolute target only.
    This exception is explicit in ``relative_active``, including relative-only
    mode. Norm scales and initial norms remain fixed throughout a solve.
    """
    current, initial = jnp.asarray(norms), jnp.asarray(initial_norms)
    if current.ndim != 1 or current.shape != initial.shape:
        raise ValueError("Current and initial equation norms must be matching vectors")
    active = initial > policy.initial_floor
    safe_initial = jnp.where(initial > 0, initial, 1.0)
    relative = jnp.where(initial > 0, current / safe_initial, 0.0)
    absolute_ratio = current / policy.absolute_tolerance
    relative_ratio = relative / policy.relative_tolerance
    absolute_ok = current <= policy.absolute_tolerance
    relative_ok = active & (relative <= policy.relative_tolerance)
    if policy.criterion == "either":
        ratio = jnp.where(
            active, jnp.minimum(absolute_ratio, relative_ratio), absolute_ratio
        )
        passed = absolute_ok | relative_ok
    elif policy.criterion == "both":
        ratio = jnp.maximum(absolute_ratio, jnp.where(active, relative_ratio, 0.0))
        passed = absolute_ok & (~active | relative_ok)
    elif policy.criterion == "relative":
        ratio = jnp.where(active, relative_ratio, absolute_ratio)
        passed = jnp.where(active, relative_ok, absolute_ok)
    else:
        ratio = absolute_ratio
        passed = absolute_ok
    finite = jnp.all(
        jnp.isfinite(current) & jnp.isfinite(initial) & (current >= 0) & (initial >= 0)
    )
    passed = passed & jnp.isfinite(ratio) & finite
    return dict(
        equation_relative=relative,
        relative_active=active,
        equation_passed=passed,
        equation_max_ratio=jnp.max(ratio),
        all_equations_passed=jnp.all(passed),
        finite=finite,
        absolute_satisfied=absolute_ok & finite,
        relative_satisfied=relative_ok & finite,
    )
