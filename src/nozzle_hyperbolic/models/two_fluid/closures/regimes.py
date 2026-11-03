"""Bubbly-to-droplet transition weights shared by drag and heat transfer.

Port reference: Alberto revision faafff704e54fd866abff95fcb88fc0a5217405c.
See docs/notes/two_fluid_migration.md for notation and retained policies.
"""

import jax.numpy as jnp


REGIME_BLENDS = ("linear", "smoothstep")


def regime_weight(alpha1, alpha_b, alpha_d, blend="linear"):
    """Return the droplet-regime weight between the two void thresholds.

    Parameters
    ----------
    alpha1 : array_like
        Gas volume fraction.
    alpha_b, alpha_d : float
        Upper bubbly and lower droplet thresholds, with ``0 < alpha_b < alpha_d < 1``.
    blend : {'linear', 'smoothstep'}, optional
        Linear interpolation or cubic interpolation with zero endpoint slopes.

    Returns
    -------
    array_like
        Weight in [0, 1], zero below ``alpha_b`` and one above ``alpha_d``.
    """
    w = jnp.clip((alpha1 - alpha_b) / (alpha_d - alpha_b), 0.0, 1.0)
    if blend == "smoothstep":
        return w * w * (3.0 - 2.0 * w)
    if blend == "linear":
        return w
    raise ValueError(f"unknown regime blend {blend!r}; choose from {REGIME_BLENDS}")
