"""Model-owned Roe linearization of general-EOS homogeneous Euler fluxes.

The caller supplies density, sound speed and Gruneisen coefficient from its
pressure/enthalpy EOS calls. Geometry and property evaluation stay in the
physical model; reconstruction and numerical-flux contraction stay in numerics.
The annular form uses transverse momentum at one common face radius.
"""

import jax.numpy as jnp


def roe_metadata(reference):
    """Return reproducible, JSON-serializable averaging/entropy conventions."""
    rho, pressure, sound, enthalpy = map(float, reference)
    return dict(
        averaging="centered_secant_projection_v1",
        entropy_fix="acoustic_harten",
        invalid_face_policy="reject",
        small_jump_policy="c2_roundoff_blend_v1",
        small_jump_threshold="eps^(1/3), dimensionless thermodynamic jump",
        small_jump_transition="squared jump from tau^2 to 4*tau^2",
        secant_audit="regularization_remainder_v1",
        secant_audit_factor=4096,
        reference_convention="model_reservoir_centered_internal_energy",
        reference_density=rho,
        reference_pressure=pressure,
        reference_sound_speed=sound,
        reference_internal_energy=enthalpy - pressure / rho,
    )


def _centered_secant(left, right, thermo_left, thermo_right, reference):
    """Finite-jump thermodynamic slopes with a differentiable diagonal limit.

    Away from roundoff-size jumps the scaled orthogonal projection enforces
    dp=chi_c*drho+kappa*d(rho*(e-e_ref)). The symmetric provisional slopes
    differ from the exact secant by O(jump^3) for a smooth consistent EOS.
    Below eps^(1/3) they avoid amplifying EOS/subtraction noise by 1/jump.
    A C2 quintic transition restores the full projection above 2*eps^(1/3).
    The residual is audited against the remainder of this declared blend;
    the tiny-jump branch is not advertised as an exact finite secant. Independent
    tabulated acoustic properties need not have an O(jump^3) provisional defect.
    """
    rho_l, a_l, g_l = (thermo_left[key] for key in ("rho", "a", "G"))
    rho_r, a_r, g_r = (thermo_right[key] for key in ("rho", "a", "G"))
    rho_l, a_l, g_l, rho_r, a_r, g_r = jnp.broadcast_arrays(
        rho_l, a_l, g_l, rho_r, a_r, g_r
    )
    rho_ref, p_ref, a_ref, h_ref = reference
    energy_ref = h_ref - p_ref / rho_ref
    energy_scale = rho_ref * a_ref**2
    valid_sides = (
        (rho_l > 0)
        & (rho_r > 0)
        & (a_l > 0)
        & (a_r > 0)
        & jnp.isfinite(rho_l + rho_r + a_l + a_r + g_l + g_r)
    )
    # Safe inactive arithmetic does not make an invalid endpoint admissible.
    z_l = jnp.sqrt(jnp.where(rho_l > 0, rho_l, 1.0))
    z_r = jnp.sqrt(jnp.where(rho_r > 0, rho_r, 1.0))
    fraction = z_r / (z_l + z_r)

    def average(f_l, f_r):
        return f_l + fraction * (f_r - f_l)

    h_l, h_r = left[..., -1] - energy_ref, right[..., -1] - energy_ref
    chi_0 = average(a_l**2 - g_l * h_l, a_r**2 - g_r * h_r)
    kappa_0 = average(g_l, g_r)
    dp = right[..., 0] - left[..., 0]
    dh = right[..., -1] - left[..., -1]
    drho = rho_r - rho_l
    # Symmetric product identity avoids subtracting two large energy densities.
    di = drho * (0.5 * (h_l + h_r)) + 0.5 * (rho_l + rho_r) * dh - dp
    x, y, d = drho / rho_ref, di / energy_scale, dp / p_ref
    g_0, b_0 = rho_ref * chi_0 / p_ref, energy_scale * kappa_0 / p_ref
    defect = d - g_0 * x - b_0 * y
    norm2 = x * x + y * y
    eps = jnp.finfo(jnp.result_type(x, y, d)).eps
    threshold2 = eps ** (2.0 / 3.0)
    t = jnp.clip((norm2 / threshold2 - 1.0) / 3.0, 0.0, 1.0)
    blend = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
    denominator = jnp.where(norm2 > threshold2, norm2, 1.0)
    correction = blend * defect / denominator
    g, b = g_0 + correction * x, b_0 + correction * y
    chi_c, kappa = p_ref / rho_ref * g, p_ref / energy_scale * b
    pressure_defect = d - g * x - b * y
    audit_scale = 1.0 + jnp.abs(d) + jnp.abs(g * x) + jnp.abs(b * y)
    # Only the full projection promises a zero secant defect. In the diagonal
    # regularization, the uncorrected fraction is intentional, including for
    # independently interpolated EOS coefficients. Audit its algebraic remainder
    # without imposing an unrelated EOS-consistency tolerance on admissibility.
    expected_defect = (1.0 - blend) * defect
    audit_error = pressure_defect - expected_defect
    audit_tolerance = 4096 * eps * audit_scale
    secant_valid = jnp.abs(audit_error) <= audit_tolerance
    return (
        chi_c,
        kappa,
        average,
        energy_ref,
        pressure_defect,
        valid_sides & secant_valid,
        expected_defect,
        audit_error,
        blend,
        audit_tolerance,
    )


def euler_roe_data(
    left, right, thermo_left, thermo_right, reference, delta_factor, *, radius=None
):
    """Return three grouped Euler Roe eigenspaces and physical diagnostics.

    ``reference`` is (rho_ref,p_ref,a_ref,h_ref), fixed for a model instance.
    ``radius=None`` selects nozzle (p,u,h); otherwise the states are annular
    (p,u,ell,h) and ell/radius is the transverse velocity. Groups are ordered
    acoustic plus, acoustic minus, repeated convective/contact-and-shear.
    Invalid averages are flagged, never clipped to a different Roe method.
    """
    left, right = jnp.asarray(left), jnp.asarray(right)
    (
        chi_c,
        kappa,
        average,
        energy_ref,
        defect,
        valid,
        expected_defect,
        audit_error,
        blend,
        audit_tolerance,
    ) = _centered_secant(left, right, thermo_left, thermo_right, reference)
    u_l, u_r = left[..., 1], right[..., 1]
    u = average(u_l, u_r)
    zero, one = jnp.zeros_like(u), jnp.ones_like(u)
    if radius is None:
        v_l, v_r, v = zero, zero, zero
        nvar = 3
    else:
        v_l, v_r = left[..., 2] / radius, right[..., 2] / radius
        v = average(v_l, v_r)
        nvar = 4
    kinetic = 0.5 * (u**2 + v**2)
    h_l = left[..., -1] - energy_ref + 0.5 * (u_l**2 + v_l**2)
    h_r = right[..., -1] - energy_ref + 0.5 * (u_r**2 + v_r**2)
    total_h_c = average(h_l, h_r)
    sound2 = chi_c + kappa * (total_h_c - kinetic)
    valid = valid & (sound2 > 0) & jnp.isfinite(sound2 + u + v + chi_c + kappa)
    sound = jnp.sqrt(jnp.where(sound2 > 0, sound2, 1.0))
    # Construct in the centered-energy frame, then change energy reference by
    # similarity. This also avoids cancellation in sound2 after an h shift.
    if nvar == 3:
        matrix_c = jnp.stack(
            (
                jnp.stack((zero, one, zero), -1),
                jnp.stack((chi_c + kappa * kinetic - u**2, (2 - kappa) * u, kappa), -1),
                jnp.stack(
                    (
                        u * (chi_c + kappa * kinetic - total_h_c),
                        total_h_c - kappa * u**2,
                        (1 + kappa) * u,
                    ),
                    -1,
                ),
            ),
            -2,
        )
    else:
        matrix_c = jnp.stack(
            (
                jnp.stack((zero, one, zero, zero), -1),
                jnp.stack(
                    (
                        chi_c + kappa * kinetic - u**2,
                        (2 - kappa) * u,
                        -kappa * v,
                        kappa,
                    ),
                    -1,
                ),
                jnp.stack((-u * v, v, u, zero), -1),
                jnp.stack(
                    (
                        u * (chi_c + kappa * kinetic - total_h_c),
                        total_h_c - kappa * u**2,
                        -kappa * u * v,
                        (1 + kappa) * u,
                    ),
                    -1,
                ),
            ),
            -2,
        )
    identity = jnp.eye(nvar, dtype=matrix_c.dtype)
    centered = matrix_c - u[..., None, None] * identity
    square = centered @ centered
    sound2_safe = jnp.where(sound2 > 0, sound2, 1.0)[..., None, None]
    plus = (square + sound[..., None, None] * centered) / (2 * sound2_safe)
    minus = (square - sound[..., None, None] * centered) / (2 * sound2_safe)
    convective = identity - square / sound2_safe
    projectors_c = jnp.stack((plus, minus, convective), -3)
    transform = identity.at[-1, 0].set(energy_ref)
    inverse = identity.at[-1, 0].set(-energy_ref)
    projectors = transform @ projectors_c @ inverse
    matrix = transform @ matrix_c @ inverse
    eigenvalues = jnp.stack((u + sound, u - sound, u), -1)
    sound_l, sound_r = thermo_left["a"], thermo_right["a"]
    spread = jnp.stack(
        ((u_r + sound_r) - (u_l + sound_l), (u_r - sound_r) - (u_l - sound_l)), -1
    )
    base = delta_factor * sound[..., None]
    radicand = spread**2 + base**2
    root = jnp.where(
        radicand > 0, jnp.sqrt(jnp.where(radicand > 0, radicand, 1.0)), 0.0
    )
    acoustic_width = base + 0.25 * (spread + root)
    width = jnp.concatenate((acoustic_width, zero[..., None]), -1)
    valid = valid & jnp.all(jnp.isfinite(projectors), axis=(-3, -2, -1))
    return dict(
        projectors=projectors,
        eigenvalues=eigenvalues,
        width=width,
        valid=valid,
        matrix=matrix,
        sound_speed_squared=sound2,
        pressure_secant_defect=defect,
        pressure_secant_expected_defect=expected_defect,
        pressure_secant_audit_error=audit_error,
        secant_blend=blend,
        secant_audit_tolerance=audit_tolerance,
        chi=chi_c - kappa * energy_ref,
        kappa=kappa,
    )
