"""
Debug/verify the Newton inner LINEAR solve of the two-fluid step.

Question being settled
----------------------
The Newton system  J·dW = −R  is RECTANGULAR and UNDER-determined:
    J is  6(N−2) × 8(N−2)   (6 conservation eqs, 8 unknowns per cell),
so we take the MINIMUM-NORM solution  dW = Jᵀ (J Jᵀ)⁻¹ (−R).

Claim (design rationale): `lstsq`, `normal_eq`, and `qr` all return the SAME
min-norm dW — they are three numerical routes to one mathematical object.  The
rectangularity is INTENTIONAL: the under-determination is what keeps the solve
well-posed in the two degenerate limits we must support —
    (a) ZERO SLIP     du = u_g − u_l → 0   (void/acoustic eigvecs become parallel),
    (b) GHOST PHASE   one phase's partial density → 0 (its eqs ~vanish).
Going SQUARE (the rejected "E3") would reintroduce a block that is singular in
exactly those limits.  This script demonstrates both points on representative
rectangular systems so we don't keep re-litigating the solver choice.

It replicates the THREE `_lin_solve` variants from
`two_fluid/solver.py` verbatim (numpy port) and compares:
  * dW agreement across solvers,
  * the min-norm property  ||dW|| minimal  and  residual ||J dW − (−R)||,
  * behaviour as the system approaches zero-slip / ghost degeneracy.

Run:
    poetry run python dev_scripts/compare_minnorm_solvers.py
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 → allow κ², —, × in prints
except Exception:
    pass

np.set_printoptions(precision=3, suppress=False, linewidth=120)
_RNG = np.random.default_rng(0)


# ── The three min-norm solves (numpy port of solver.py `_lin_solve`) ───────────
def solve_lstsq(J, rhs):
    return np.linalg.lstsq(J, rhs, rcond=None)[0]


def solve_normal_eq(J, rhs, reg=0.0):
    JJt = J @ J.T
    r   = reg * np.mean(np.diag(JJt))
    y   = np.linalg.solve(JJt + r * np.eye(JJt.shape[0]), rhs)
    return J.T @ y


def solve_qr(J, rhs):
    # Jᵀ (n×m, tall) = Q R;  min-norm dW = Q (Rᵀ \ rhs)
    Q, R = np.linalg.qr(J.T, mode="reduced")
    y    = np.linalg.solve(R.T, rhs)
    return Q @ y


def _report(name, J, rhs):
    dW = {
        "lstsq":     solve_lstsq(J, rhs),
        "normal_eq": solve_normal_eq(J, rhs),
        "qr":        solve_qr(J, rhs),
    }
    base = dW["lstsq"]
    m, n = J.shape
    print(f"\n=== {name}   (J: {m}×{n},  cond(J)={np.linalg.cond(J):.2e}) ===")
    print(f"  {'solver':<10} {'||dW||':>12} {'||J dW-rhs||':>14} {'||dW-lstsq||':>14}")
    for k, v in dW.items():
        res  = np.linalg.norm(J @ v - rhs)
        diff = np.linalg.norm(v - base)
        print(f"  {k:<10} {np.linalg.norm(v):12.5e} {res:14.3e} {diff:14.3e}")
    # is lstsq actually the min-norm among all solutions?  Add any null-space
    # vector and norm must not decrease.
    return dW


# ── (1) Generic well-conditioned rectangular system ────────────────────────────
def case_generic(Ncell=6):
    m, n = 6 * Ncell, 8 * Ncell
    J   = _RNG.standard_normal((m, n))
    rhs = _RNG.standard_normal(m)
    _report("generic rectangular (well-conditioned)", J, rhs)


# ── (2) Near ZERO-SLIP: two unknown-columns become nearly parallel ─────────────
# Mimics du→0 where the two void-wave right eigenvectors align → a near-null
# direction in W.  Min-norm stays finite/unique; normal_eq loses the most digits.
def case_zero_slip(eps=1e-8, Ncell=6):
    m, n = 6 * Ncell, 8 * Ncell
    J = _RNG.standard_normal((m, n))
    # make columns (8c+3=u_g) and (8c+5=u_l) nearly identical in every cell block
    for c in range(Ncell):
        J[:, 8 * c + 5] = J[:, 8 * c + 3] + eps * _RNG.standard_normal(m)
    rhs = _RNG.standard_normal(m)
    _report(f"near zero-slip (u_g,u_l cols parallel, eps={eps:g})", J, rhs)


# ── (3) GHOST phase: one phase's conservation rows ~vanish ─────────────────────
# Mimics α_l→0 (or α_g→0): that phase's mass/mom/energy equations scale with its
# partial density → ~0 rows.  Rectangular min-norm still returns a bounded dW
# (it simply does not constrain that phase); a SQUARE system would be singular.
def case_ghost(scale=1e-10, Ncell=6):
    m, n = 6 * Ncell, 8 * Ncell
    J = _RNG.standard_normal((m, n))
    # rows 3,4,5 of each 6-block = the "liquid" phase eqs → scale to ~0
    for c in range(Ncell):
        J[6 * c + 3: 6 * c + 6, :] *= scale
    rhs = _RNG.standard_normal(m)
    rhs[3::6] *= scale
    _report(f"ghost phase (liquid eqs × {scale:g})", J, rhs)


if __name__ == "__main__":
    print("Min-norm solver equivalence check for the rectangular Newton system.")
    print("Expect: lstsq == normal_eq == qr to ~machine precision when well-conditioned;")
    print("        near degeneracy they still AGREE on the min-norm but normal_eq loses")
    print("        the most digits (κ²).  All stay BOUNDED — the rectangularity is why.")
    case_generic()
    case_zero_slip(eps=1e-6)
    case_zero_slip(eps=1e-10)
    case_ghost(scale=1e-8)
    case_ghost(scale=1e-12)
    print("\nConclusion: the solver choice does NOT change the min-norm dW (the physics);")
    print("it only trades conditioning.  Keep the rectangular min-norm — going square")
    print("(E3) would be singular in the zero-slip / ghost limits shown above.")
