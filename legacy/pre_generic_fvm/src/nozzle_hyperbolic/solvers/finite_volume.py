"""Integrated finite-volume assembly with explicit algebraic boundary rows."""

import operator

import jax.numpy as jnp


def flux_divergence(integrated_face_flux):
    """Return right-minus-left fluxes; geometry weighting is already applied."""
    return integrated_face_flux[1:] - integrated_face_flux[:-1]


def make_coupled_residual(
    storage, spatial_residual, boundary_residual, *, nphysical, nvar,
    content_scale=None, flux_scale=None,
):
    """Assemble one square system with two zero-storage boundary blocks.

    Callbacks receive cell-major arrays: storage(physical_W) returns integrated
    contents (N,nvar), spatial_residual(full_W) returns integrated spatial
    residuals (N,nvar), boundary_residual(full_W) returns two already-scaled
    boundary blocks (2,nvar). Full W has shape (N+2,nvar).

    Returns a dict: residual(new_flat,old_flat,dt) is the scaled backward-Euler
    system; steady(full_flat) is the original scaled full spatial/BC system;
    spatial(full_flat) returns raw integrated physical spatial rows (N,nvar).
    Boundary rows are never multiplied by dt and never contain storage. No
    boundary values are reconstructed, projected or overwritten here.
    """
    nphysical, nvar = map(operator.index, (nphysical, nvar))
    if nphysical < 1 or nvar < 1:
        raise ValueError("nphysical and nvar must be positive")
    shape = (nphysical + 2, nvar)
    content_scale = 1.0 if content_scale is None else jnp.asarray(content_scale)
    flux_scale = 1.0 if flux_scale is None else jnp.asarray(flux_scale)

    def pack(physical, boundaries):
        return jnp.concatenate([boundaries[:1], physical, boundaries[1:]], axis=0).ravel()

    def residual(new_flat, old_flat, dt):
        new = new_flat.reshape(shape)
        old = old_flat.reshape(shape)
        physical = (
            storage(new[1:-1]) - storage(old[1:-1]) + dt * spatial_residual(new)
        ) / content_scale
        return pack(physical, boundary_residual(new))

    def spatial(full_flat):
        return spatial_residual(full_flat.reshape(shape))

    def steady(full_flat):
        full = full_flat.reshape(shape)
        return pack(spatial_residual(full) / flux_scale, boundary_residual(full))

    return {"residual": residual, "steady": steady, "spatial": spatial}
