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

from nozzle_hyperbolic.fvm_solver.fvm import _jax_F_fvs_stadtke, muscl_van_leer_ph
from nozzle_hyperbolic.fvm_solver.two_fluid.flux import fvs_interface_flux
from nozzle_hyperbolic.fvm_solver.two_fluid.solver import _apply_ramped_inlet


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
        # The phase-property interface also consumes temperature. Return the
        # temperature already used to evaluate density and sound speed.
        return {"rho": rho, "a": a, "G": G, "T": T}


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





def test_lowmach_flux_dissipation():
    """A small low-Mach dissipation blend should change the interface flux."""
    U_h_L = jnp.array([1.0e5, 10.0, 10.0, 0.3, 3.0e5, 3.0e5], dtype=jnp.float64)
    U_h_R = U_h_L + jnp.array([0.0, 0.1, -0.1, 0.0, 200.0, -200.0], dtype=jnp.float64)

    F_base = fvs_interface_flux(U_h_L, U_h_R, _FLUID, _FLUID, 0.5, 1e-2, lowmach_diss=0.0)
    F_stab = fvs_interface_flux(U_h_L, U_h_R, _FLUID, _FLUID, 0.5, 1e-2, lowmach_diss=0.2)

    assert jnp.isfinite(F_stab).all()
    assert not jnp.allclose(F_stab, F_base)


def test_apply_ramped_inlet():
    """The ramped inlet helper should blend the imposed ghost state into the first cells."""
    U_h = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
    ], dtype=jnp.float64)
    U_h_bc = jnp.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=jnp.float64)

    U_h_new = _apply_ramped_inlet(U_h, U_h_bc, n_cells=3)

    np.testing.assert_allclose(np.array(U_h_new[0]), np.array(U_h_bc))
    np.testing.assert_allclose(np.array(U_h_new[1]), 0.5 * np.array(U_h[1]) + 0.5 * np.array(U_h_bc))
    np.testing.assert_allclose(np.array(U_h_new[2]), np.array(U_h[2]))


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


# ---------------------------------------------------------------------------
# MUSCL Van Leer reconstruction tests
# ---------------------------------------------------------------------------

def test_muscl_constant_field():
    """Constant field → sigma = 0, U_L = U[:-1], U_R = U[1:] (atol 1e-12)."""
    N = 10
    U = jnp.tile(jnp.array([2.0e5, 100.0, 3.0e5]), (N, 1))
    U_L, U_R = muscl_van_leer_ph(U)
    np.testing.assert_allclose(np.array(U_L), np.array(U[:-1]), atol=1e-12,
                               err_msg="U_L should equal cell-centred value for constant field")
    np.testing.assert_allclose(np.array(U_R), np.array(U[1:]),  atol=1e-12,
                               err_msg="U_R should equal cell-centred value for constant field")


def test_muscl_linear_field():
    """Linear profile → interior face values are exact midpoints (rtol 1e-10).

    For a linear field, van Leer = full slope (no limiting), so the
    reconstructed face value from both sides is exactly the cell-edge midpoint.
    Boundary faces (0 and N-2) are 1st-order on one side — excluded.
    """
    N  = 12
    dp = 1.0e3          # Pa per cell
    p  = 2.0e5 + np.arange(N) * dp
    u  = np.full(N, 50.0)
    h  = 3.0e5 + np.arange(N) * 100.0   # also linear
    U  = jnp.array(np.column_stack([p, u, h]))
    U_L, U_R = muscl_van_leer_ph(U)

    # At interior faces i=1..N-3: van Leer is unclipped, so U_L[i] = U_R[i] = face midpoint
    for i in range(1, N - 2):
        p_face = 0.5 * (p[i] + p[i + 1])   # exact face midpoint value
        h_face = 0.5 * (h[i] + h[i + 1])
        np.testing.assert_allclose(float(U_L[i, 0]), p_face, rtol=1e-10,
                                   err_msg=f"p: U_L wrong at face {i}")
        np.testing.assert_allclose(float(U_R[i, 0]), p_face, rtol=1e-10,
                                   err_msg=f"p: U_R wrong at face {i}")
        np.testing.assert_allclose(float(U_L[i, 2]), h_face, rtol=1e-10,
                                   err_msg=f"h: U_L wrong at face {i}")
        np.testing.assert_allclose(float(U_R[i, 2]), h_face, rtol=1e-10,
                                   err_msg=f"h: U_R wrong at face {i}")


def test_muscl_extremum_limited():
    """At a non-monotone extremum the limiter clamps sigma to 0 (atol 1.0).

    Cell 3 is a peak: neighbours on both sides are lower, so dU[2] and dU[3]
    have opposite signs → van Leer returns 0 → U_L[3] = U[3] exactly.
    """
    N = 8
    p = np.array([1.0, 2.0, 3.0, 6.0, 3.0, 2.0, 1.0, 0.5]) * 1e5   # peak at i=3
    U = jnp.array(np.column_stack([p, np.full(N, 50.0), np.full(N, 3.0e5)]))
    U_L, U_R = muscl_van_leer_ph(U)

    # sigma[3] uses dU[2] = p[3]-p[2] > 0 and dU[3] = p[4]-p[3] < 0 → product < 0 → 0
    # U_L[3] = U[3] + 0.5 * 0 = p[3]  (left state at face 3)
    np.testing.assert_allclose(float(U_L[3, 0]), p[3], atol=1.0,
                               err_msg="Peak cell: U_L should equal cell-centred p (no slope)")
    # U_R[2] = U[3] - 0.5 * sigma[3] = U[3] (sigma[3]=0 because face 2 right = cell 3 = peak)
    np.testing.assert_allclose(float(U_R[2, 0]), p[3], atol=1.0,
                               err_msg="Peak cell right-side: U_R should equal cell-centred p")
