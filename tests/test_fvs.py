"""
Unit tests for the Städtke FVS flux, matrices, and EOS identities.

Spec reference: hem_nozzle_implementation_spec.md §12.

Tests use an ideal-gas mock fluid (γ=1.4, R=287 J/(kg·K)) so no LUT
is required and each test runs in < 1 s.
"""

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nozzle_hyperbolic.fvm_solver.fvm import _jax_F_fvs_stadtke


# ---------------------------------------------------------------------------
# Ideal-gas mock — implements the same get_state interface as jaxprop fluids
# ---------------------------------------------------------------------------

class _IdealGas:
    """Air-like ideal gas: γ=1.4, R=287 J/(kg·K). Accepts JAX arrays."""
    _gamma = 1.4
    _R     = 287.0
    _cp    = _gamma / (_gamma - 1.0) * _R   # 1004.5 J/(kg·K)

    def get_state(self, _input_type, h, p):
        """HmassP_INPUTS: (h [J/kg], p [Pa]) → state dict with JAX arrays."""
        T   = jnp.maximum(h / self._cp, 1.0)    # guard T > 0
        rho = p / (self._R * T)
        a   = jnp.sqrt(self._gamma * self._R * T)
        G   = jnp.full_like(rho, self._gamma - 1.0)
        return {"rho": rho, "a": a, "G": G}


_FLUID = _IdealGas()
_GAMMA = 1.4
_R     = 287.0
_CP    = _GAMMA / (_GAMMA - 1.0) * _R


# ---------------------------------------------------------------------------
# Helper: closed-form ideal-gas thermodynamics and matrices
# ---------------------------------------------------------------------------

def _pg_state(p, T, u):
    """Return (h, rho, a, G, rho_p, rho_h, H) for ideal gas at (p, T, u)."""
    h     = _CP * T
    rho   = p / (_R * T)
    a     = np.sqrt(_GAMMA * _R * T)
    G     = _GAMMA - 1.0
    rho_p = (1.0 + G) / a**2    # (∂ρ/∂p)_h
    rho_h = -(rho * G) / a**2   # (∂ρ/∂h)_p
    H     = h + 0.5 * u**2
    return h, rho, a, G, rho_p, rho_h, H


def _G_mat(u, rho, a):
    """Coefficient matrix G (theory §2.4)."""
    return np.array([
        [u,      rho * a**2,  0.0],
        [1/rho,  u,           0.0],
        [0.0,    a**2,        u  ],
    ])


def _J_mat(u, h, rho, rho_p, rho_h):
    """Conservative Jacobian J = dV/dU (theory §2.2)."""
    H = h + 0.5 * u**2
    return np.array([
        [rho_p,          0.0,    rho_h          ],
        [u * rho_p,      rho,    u * rho_h      ],
        [rho_p * H - 1,  rho*u,  rho_h * H + rho],
    ])


def _K_mat(u, h, rho, rho_p, rho_h):
    """Flux Jacobian K = dF/dU (theory §2.3)."""
    H = h + 0.5 * u**2
    return np.array([
        [u * rho_p,            rho,                u * rho_h              ],
        [u**2 * rho_p + 1.0,   2.0 * rho * u,      u**2 * rho_h           ],
        [u * rho_p * H,         rho * (H + u**2),   u * (rho_h * H + rho)  ],
    ])


def _G1_mat(u, rho, a):
    """Split matrix G₁ = (u+a) (...) (theory §4.2)."""
    return (u + a) * np.array([
        [0.5,           rho * a / 2,   0.0],
        [1/(2*rho*a),   0.5,           0.0],
        [1/(2*rho),     a / 2,         0.0],
    ])


def _G2_mat(u, rho, a):
    """Split matrix G₂ = (u-a) (...) (theory §4.2)."""
    return (u - a) * np.array([
        [0.5,             -rho * a / 2,  0.0],
        [-1/(2*rho*a),    0.5,           0.0],
        [1/(2*rho),       -a / 2,        0.0],
    ])


def _G3_mat(u, rho):
    """Split matrix G₃ = u (...) (theory §4.2)."""
    return u * np.array([
        [0.0,     0.0,  0.0],
        [0.0,     0.0,  0.0],
        [-1/rho,  0.0,  1.0],
    ])


# ---------------------------------------------------------------------------
# Test states
#   _STATES_ALL    — used for matrix/eigenvalue tests (any flow regime OK)
#   _STATES_NOSYNC — used for partition-unity: all |λ_k| >> δ (no entropy fix)
# ---------------------------------------------------------------------------

_STATES_ALL = [
    (2.0e5, 300.0,  50.0),   # clearly subsonic   (u/a ≈ 0.15)
    (1.0e5, 250.0, 280.0),   # near-sonic          (u/a ≈ 0.88)
    (5.0e4, 200.0, 600.0),   # supersonic          (u/a ≈ 2.10)
]

# For partition_unity the entropy fix must be inactive: |λ_k| > δ for all k.
# Condition on mode 2: |u-a| > 0.1*(|u|+a)  →  safe when u/a < 0.82 or u/a > 1.22
_STATES_NOSYNC = [
    (2.0e5, 300.0,  50.0),   # u/a ≈ 0.15  — all coeff = 1
    (5.0e4, 200.0, 600.0),   # u/a ≈ 2.10  — all coeff = 1
]


# ---------------------------------------------------------------------------
# §2.2  Sound-speed identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,T,u", _STATES_ALL)
def test_sound_speed_identity(p, T, u):
    """1/a² = ρ_p + ρ_h/ρ (spec §2.2, tol 1e-12)."""
    _, rho, a, _, rho_p, rho_h, _ = _pg_state(p, T, u)
    lhs = 1.0 / a**2
    rhs = rho_p + rho_h / rho
    assert abs(lhs - rhs) / lhs < 1e-12


# ---------------------------------------------------------------------------
# §4.5  Matrix partition and consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,T,u", _STATES_ALL)
def test_partition_G(p, T, u):
    """G₁ + G₂ + G₃ = G (spec §4.5, atol 1e-12)."""
    _, rho, a, _, _, _, _ = _pg_state(p, T, u)
    np.testing.assert_allclose(
        _G1_mat(u, rho, a) + _G2_mat(u, rho, a) + _G3_mat(u, rho),
        _G_mat(u, rho, a),
        atol=1e-12,
    )


@pytest.mark.parametrize("p,T,u", _STATES_ALL)
def test_consistency_JGK(p, T, u):
    """J @ G = K (spec §4.5, rtol 1e-10)."""
    h, rho, a, _, rho_p, rho_h, _ = _pg_state(p, T, u)
    J  = _J_mat(u, h, rho, rho_p, rho_h)
    Gm = _G_mat(u, rho, a)
    K  = _K_mat(u, h, rho, rho_p, rho_h)
    np.testing.assert_allclose(J @ Gm, K, rtol=1e-10)


@pytest.mark.parametrize("p,T,u", _STATES_ALL)
def test_eigenvalues_G(p, T, u):
    """Eigenvalues of G are {u+a, u-a, u} (spec §4.5, atol 1e-10)."""
    _, rho, a, _, _, _, _ = _pg_state(p, T, u)
    Gm       = _G_mat(u, rho, a)
    computed = sorted(np.real(np.linalg.eigvals(Gm)))
    expected = sorted([u + a, u - a, u])
    np.testing.assert_allclose(computed, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# §5.4  Flux properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,T,u", _STATES_NOSYNC)
def test_partition_unity(p, T, u):
    """R⁺ + R⁻ = I when entropy fix is inactive (spec §5.4, atol 1e-10).

    Only tested for states where all |λ_k| > δ so coeff_k = 1 exactly.
    """
    h, rho, a, _, rho_p, rho_h, _ = _pg_state(p, T, u)
    K  = _K_mat(u, h, rho, rho_p, rho_h)
    KT = K.T

    vs   = [np.array([ rho*a,  1.0,  a]), np.array([-rho*a, 1.0, -a]), np.array([0.0, 0.0, 1.0])]
    ls   = [np.array([0.5/(rho*a), 0.5, 0.0]), np.array([-0.5/(rho*a), 0.5, 0.0]), np.array([-1/rho, 0.0, 1.0])]
    lams = [u + a, u - a, u]
    delta = 0.1 * (abs(u) + a)

    # Verify all coefficients are 1 (entropy fix inactive)
    for lam in lams:
        coeff = abs(lam) / max(abs(lam), delta)
        assert abs(coeff - 1.0) < 1e-12, (
            f"Entropy fix active (coeff={coeff:.4f}) for λ={lam:.1f}, δ={delta:.1f}. "
            "Use a state further from sonic."
        )

    R_sum = np.zeros((3, 3))
    for vk, lk, lam in zip(vs, ls, lams):
        Kv   = K @ vk
        KTl  = np.linalg.solve(KT, lk)
        R_sum += np.outer(Kv, KTl)

    # atol=1e-9: at high u, K contains entries O(u²·H) which raise floating-point
    # errors in the solve; the identity still holds well within 1e-9.
    np.testing.assert_allclose(R_sum, np.eye(3), atol=1e-9)


def test_uniform_flow():
    """U_L = U_R => F̂ = F(U_L) (spec §5.4, rtol 1e-10)."""
    p0 = 2.0e5;  T0 = 300.0;  u0 = 150.0
    h0   = _CP * T0
    rho0 = p0 / (_R * T0)
    H0   = h0 + 0.5 * u0**2
    F_exact = np.array([rho0 * u0, rho0 * u0**2 + p0, u0 * rho0 * H0])

    U0    = jnp.array([[p0, u0, h0]], dtype=jnp.float64)
    F_hat = np.array(_jax_F_fvs_stadtke(U0, U0, _FLUID)[0])

    np.testing.assert_allclose(F_hat, F_exact, rtol=1e-10)


def test_supersonic():
    """u >> a => F̂ = F_L (all characteristics positive, spec §5.4, rtol 1e-6)."""
    T_L = 200.0;  p_L = 1.0e5
    h_L = _CP * T_L
    a_L = np.sqrt(_GAMMA * _R * T_L)
    u_L = 3.0 * a_L                            # M ≈ 3 — well supersonic
    rho_L = p_L / (_R * T_L)
    H_L   = h_L + 0.5 * u_L**2
    F_L_exact = np.array([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * rho_L * H_L])

    T_R = 195.0;  p_R = 0.9e5
    h_R = _CP * T_R
    u_R = 2.8 * a_L                            # also supersonic

    U_L = jnp.array([[p_L, u_L, h_L]], dtype=jnp.float64)
    U_R = jnp.array([[p_R, u_R, h_R]], dtype=jnp.float64)
    F_hat = np.array(_jax_F_fvs_stadtke(U_L, U_R, _FLUID)[0])

    # With M=3 and delta=0.1*(u+a), all |λ_k| >> δ → coeff = 1, R⁻ = 0 → F̂ = F_L
    np.testing.assert_allclose(F_hat, F_L_exact, rtol=1e-6)
