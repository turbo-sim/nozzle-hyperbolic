# Port reference: origin/alberto-branch faafff704e54fd866abff95fcb88fc0a5217405c.
# Intentional safety changes are documented in docs/two_fluid_migration.md.
"""
Closed-form matrices for Städtke's two-fluid FVS scheme in enthalpy primitives.

All algebra is transcribed from stadtke_enthalpy_frame_derivation_latest.md.
State ordering throughout this module:

    U_h = (p, u_g, u_l, alpha_g, h_g, h_l)     primitive (6,)
    V   = (ag*rho_g, al*rho_l,
           ag*rho_g*u_g, al*rho_l*u_l,
           ag*rho_g*E_g, al*rho_l*E_l)          conservative (6,)
    F   = (ag*rho_g*u_g, al*rho_l*u_l,
           ag*(rho_g*u_g^2+p), al*(rho_l*u_l^2+p),
           ag*rho_g*u_g*H_g, al*rho_l*u_l*H_l)  flux (6,)

where  E_i = h_i - p/rho_i + u_i^2/2,  H_i = h_i + u_i^2/2.

Virtual-mass coefficient k controls the magnitude of the non-conservative
coupling.  Typical value k = 0.5 (added-mass coefficient for spheres).
"""


# TODO: check matrixes with the derivation

from typing import NamedTuple
import jax.numpy as jnp
import jaxprop as jxp


# ---------------------------------------------------------------------------
# Lightweight container for pre-computed per-phase EOS quantities
# ---------------------------------------------------------------------------

class PhaseProps(NamedTuple):
    """
    Pre-computed EOS quantities for one phase at a single cell state.

    Populated by phase_props() from a FluidBicubic.get_state() call.
    rho_p/rho_h retain upstream sound-speed/Gruneisen derivative policy.
    Bicubic properties need not have these exact AD derivatives.
    """
    rho:   jnp.ndarray  # density           [kg/m^3]
    a:     jnp.ndarray  # speed of sound    [m/s]
    Gamma: jnp.ndarray  # Grüneisen coeff   [-]
    T:     jnp.ndarray  # temperature       [K]
    rho_p: jnp.ndarray  # (drho/dp)_h       [kg/(m^3 Pa)]
    rho_h: jnp.ndarray  # (drho/dh)_p       [kg^2/(m^5 J)]


def phase_props(fluid, p, h) -> PhaseProps:
    """
    Call jaxprop and assemble PhaseProps for one phase.

    Parameters
    ----------
    fluid : FluidBicubic  (metastable vapor or liquid table)
    p     : scalar JAX float — pressure [Pa]
    h     : scalar JAX float — specific enthalpy [J/kg]
    """
    p_arr = jnp.expand_dims(p, 0)
    h_arr = jnp.expand_dims(h, 0)

    # No property clipping: the model checks the declared PH domain.
    st    = fluid.get_state(jxp.HmassP_INPUTS, h_arr, p_arr)
    rho   = st["rho"][0]
    a     = st["a"][0]
    Gamma = st["G"][0]
    T     = st["T"][0]
    rho_p = (1.0 + Gamma) / a**2
    rho_h = -(rho * Gamma) / a**2
    return PhaseProps(rho=rho, a=a, Gamma=Gamma, T=T, rho_p=rho_p, rho_h=rho_h)


# ---------------------------------------------------------------------------
# Shared abbreviations (derivation §2.1, used by every matrix)
# ---------------------------------------------------------------------------

class CommonQty(NamedTuple):
    """Abbreviations that appear in G_h, J, K, X."""
    rho_mix: jnp.ndarray  # alpha_g*rho_g + alpha_l*rho_l  (mixture density)
    rho_s:   jnp.ndarray  # alpha_g*rho_l + alpha_l*rho_g  (cross-weighted)
    rh2:     jnp.ndarray  # rho_g*rho_l + k*rho_mix^2      (virtual-mass modified)
    rg_hat:  jnp.ndarray  # rho_g + k*rho_mix
    rl_hat:  jnp.ndarray  # rho_l + k*rho_mix
    a0sq:    jnp.ndarray  # frozen sound-speed combination
    du:      jnp.ndarray  # u_g - u_l  (slip)
    u1:      jnp.ndarray  # alpha_g*u_g + alpha_l*u_l  (volume-weighted velocity)
    u2:      jnp.ndarray  # mass-weighted velocity
    G_pp:    jnp.ndarray  # pressure-row (1,1) entry of G_4
    G_uu:    jnp.ndarray  # shared diagonal entry of velocity rows
    G_aa:    jnp.ndarray  # void-fraction diagonal entry


def common_quantities(U_h, eos_g: PhaseProps, eos_l: PhaseProps, k) -> CommonQty:
    """
    Pre-compute all shared abbreviations needed by the matrices.

    Parameters
    ----------
    U_h   : (6,) primitive state  [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g : PhaseProps for gas
    eos_l : PhaseProps for liquid
    k     : virtual-mass coefficient (scalar)
    """
    _, u_g, u_l, ag, _, _ = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  ag_ = eos_g.a;  rl = eos_l.rho;  al_ = eos_l.a

    rho_mix = ag * rg + al * rl
    rho_s   = ag * rl + al * rg
    rg_hat  = rg + k * rho_mix
    rl_hat  = rl + k * rho_mix
    rh2     = rg * rl + k * rho_mix**2

    # Frozen (zero-slip) sound speed combination — derivation §2.1
    a0sq = rho_s / (ag * rl / ag_**2 + al * rg / al_**2)

    du = u_g - u_l
    u1 = ag * u_g + al * u_l
    u2 = (ag * rg * u_g + al * rl * u_l) / rho_mix

    # Diagonal/coupling scalars in G_4 — derivation §2.2
    G_pp = (al * u_l * rg / rho_s * a0sq / al_**2
          + ag * u_g * rl / rho_s * a0sq / ag_**2)
    G_uu = u1 * rl * rg / rh2 + u2 * k * rho_mix**2 / rh2
    G_aa = (al * u_g * rg / rho_s * a0sq / al_**2
          + ag * u_l * rl / rho_s * a0sq / ag_**2)

    return CommonQty(rho_mix=rho_mix, rho_s=rho_s, rh2=rh2,
                     rg_hat=rg_hat, rl_hat=rl_hat, a0sq=a0sq,
                     du=du, u1=u1, u2=u2, G_pp=G_pp, G_uu=G_uu, G_aa=G_aa)


# ---------------------------------------------------------------------------
# G_h  — quasi-linear coefficient matrix in enthalpy primitives
# Derivation §4.4.  Eigenvalues: {u_g (×2), u_l (×2), u_mix±a}
# ---------------------------------------------------------------------------

def G_h_matrix(U_h, eos_g: PhaseProps, eos_l: PhaseProps, cq: CommonQty) -> jnp.ndarray:
    """
    6×6 quasi-linear coefficient matrix G_h = Phi^{-1} G_s Phi.

    Block structure (derivation §4.4):
        G_h = [[G_4,    0_{4×2}  ],
               [G_h21,  diag(u_g, u_l)]]

    The upper-left G_4 is identical to the verified G_s upper block.
    The lower-left G_h21 is the Gibbs-relation transform of G_s row 1.
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  ag_ = eos_g.a
    rl = eos_l.rho;  al_ = eos_l.a

    rho_s = cq.rho_s;  rh2 = cq.rh2;  rg_hat = cq.rg_hat;  rl_hat = cq.rl_hat
    a0sq  = cq.a0sq;   du  = cq.du;   G_pp   = cq.G_pp;    G_uu   = cq.G_uu;  G_aa = cq.G_aa

    # ── Upper-left G_4 (4×4) — rows/cols: p, u_g, u_l, alpha_g ─────────────
    # Row 1 — pressure equation
    r1 = jnp.array([
        G_pp,
        ag  * a0sq * rl * rg / rho_s,
        al  * a0sq * rl * rg / rho_s,
        a0sq * du  * rl * rg / rho_s,
    ])
    # Row 2 — gas velocity equation
    r2 = jnp.array([
        rl_hat / rh2,
        G_uu,
        -al * du * rl * rl_hat / rh2,
        0.0,
    ])
    # Row 3 — liquid velocity equation
    r3 = jnp.array([
        rg_hat / rh2,
        ag * du * rg * rg_hat / rh2,
        G_uu,
        0.0,
    ])
    # Row 4 — void fraction equation
    r4 = jnp.array([
        al * ag * du / rho_s * a0sq / (ag_**2 * al_**2),
        al * ag * rg / rho_s * a0sq / al_**2,
        -al * ag * rl / rho_s * a0sq / ag_**2,
        G_aa,
    ])

    # ── Lower-left G_h21 (2×4) — enthalpy rows ───────────────────────────────
    # Row 5 — gas enthalpy: (G_pp-u_g)/rho_g for col p; ratio*rho_l for cols u,al,a
    rh21_g = jnp.array([
        (G_pp - u_g) / rg,
        ag * a0sq * rl / rho_s,
        al * a0sq * rl / rho_s,
        a0sq * du  * rl / rho_s,
    ])
    # Row 6 — liquid enthalpy: (G_pp-u_l)/rho_l; ratio*rho_g for cols u,al,a
    rh21_l = jnp.array([
        (G_pp - u_l) / rl,
        ag * a0sq * rg / rho_s,
        al * a0sq * rg / rho_s,
        a0sq * du  * rg / rho_s,
    ])

    z4 = jnp.zeros(4)
    z2 = jnp.zeros(2)
    o  = jnp.ones(1)

    # Assemble full 6×6 row by row
    row1 = jnp.concatenate([r1,     z2])
    row2 = jnp.concatenate([r2,     z2])
    row3 = jnp.concatenate([r3,     z2])
    row4 = jnp.concatenate([r4,     z2])
    row5 = jnp.concatenate([rh21_g, jnp.array([u_g, 0.0])])
    row6 = jnp.concatenate([rh21_l, jnp.array([0.0, u_l])])

    return jnp.stack([row1, row2, row3, row4, row5, row6])


# ---------------------------------------------------------------------------
# Phi_inv  — inverse thermodynamic transform (entropy → enthalpy primitives)
# Derivation §3.2.  Only rows 5, 6 differ from identity.
# ---------------------------------------------------------------------------

def Phi_inv_matrix(eos_g: PhaseProps, eos_l: PhaseProps) -> jnp.ndarray:
    """
    6×6 inverse thermodynamic transform Phi^{-1} = d(U_h)/d(U_s).

    Phi^{-1} = [[I_4,     0_{4×2}  ],
                [[1/rho_g, 0, 0, 0], [T_g, 0 ]],
                [[1/rho_l, 0, 0, 0], [0,   T_l]]]

    Derivation §3.2 — from the inverse Gibbs relation dh_i = T_i ds_i + dp/rho_i.
    """
    rg = eos_g.rho;  T_g = eos_g.T
    rl = eos_l.rho;  T_l = eos_l.T

    I4 = jnp.eye(4)
    z4 = jnp.zeros((4, 2))

    row5 = jnp.array([1.0/rg, 0.0, 0.0, 0.0, T_g, 0.0])
    row6 = jnp.array([1.0/rl, 0.0, 0.0, 0.0, 0.0, T_l])

    top = jnp.concatenate([I4, z4], axis=1)             # (4,6)
    bot = jnp.stack([row5, row6])                         # (2,6)
    return jnp.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
# J  — conservative bridge  dV/dU_h
# Derivation §8.  Phase-decoupled in (u_g, h_g) / (u_l, h_l) columns.
# ---------------------------------------------------------------------------

def J_matrix(U_h, eos_g: PhaseProps, eos_l: PhaseProps) -> jnp.ndarray:
    """
    6×6 conservative bridge J = dV/dU_h.

    Column ordering: (p, u_g, u_l, alpha_g, h_g, h_l)
    Row ordering:    (ag*rho_g, al*rho_l, ag*rho_g*u_g, al*rho_l*u_l,
                      ag*rho_g*E_g, al*rho_l*E_l)
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  rg_p = eos_g.rho_p;  rg_h = eos_g.rho_h
    rl = eos_l.rho;  rl_p = eos_l.rho_p;  rl_h = eos_l.rho_h

    E_g = h_g - p / rg + 0.5 * u_g**2
    E_l = h_l - p / rl + 0.5 * u_l**2
    H_g = h_g + 0.5 * u_g**2
    H_l = h_l + 0.5 * u_l**2

    # Rows 1-2: mass partial densities
    row1 = jnp.array([ag*rg_p,              0.0,        0.0,         rg,      ag*rg_h,              0.0          ])
    row2 = jnp.array([al*rl_p,              0.0,        0.0,        -rl,      0.0,                   al*rl_h     ])
    # Rows 3-4: phase momenta
    row3 = jnp.array([ag*u_g*rg_p,          ag*rg,      0.0,         rg*u_g,  ag*u_g*rg_h,          0.0          ])
    row4 = jnp.array([al*u_l*rl_p,          0.0,        al*rl,      -rl*u_l,  0.0,                   al*u_l*rl_h  ])
    # Rows 5-6: phase total energies
    row5 = jnp.array([ag*H_g*rg_p - ag,     ag*rg*u_g,  0.0,         rg*E_g,  ag*H_g*rg_h + ag*rg,  0.0          ])
    row6 = jnp.array([al*H_l*rl_p - al,     0.0,        al*rl*u_l,  -rl*E_l,  0.0,                   al*H_l*rl_h + al*rl])

    return jnp.stack([row1, row2, row3, row4, row5, row6])


# ---------------------------------------------------------------------------
# K  — flux Jacobian  dF/dU_h
# Derivation §9.  Same column ordering as J.
# ---------------------------------------------------------------------------

def K_matrix(U_h, eos_g: PhaseProps, eos_l: PhaseProps) -> jnp.ndarray:
    """
    6×6 flux Jacobian K = dF/dU_h.

    The "2" in the velocity-diagonal entries (2*ag*rho_g*u_g) and the
    (H_g + u_g^2) in the energy rows are the signature of total-energy flux.
    K is singular when u_g=0 or u_l=0 (h_g/h_l columns vanish).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  rg_p = eos_g.rho_p;  rg_h = eos_g.rho_h
    rl = eos_l.rho;  rl_p = eos_l.rho_p;  rl_h = eos_l.rho_h

    H_g = h_g + 0.5 * u_g**2
    H_l = h_l + 0.5 * u_l**2

    # Rows 1-2: mass flux
    row1 = jnp.array([ag*u_g*rg_p,              ag*rg,           0.0,          rg*u_g,      ag*u_g*rg_h,              0.0              ])
    row2 = jnp.array([al*u_l*rl_p,              0.0,             al*rl,        -rl*u_l,     0.0,                       al*u_l*rl_h     ])
    # Rows 3-4: momentum flux  (extra +alpha_i from d(alpha_i*p)/dp = alpha_i)
    row3 = jnp.array([ag*u_g**2*rg_p + ag,      2*ag*rg*u_g,    0.0,          rg*u_g**2+p, ag*u_g**2*rg_h,           0.0              ])
    row4 = jnp.array([al*u_l**2*rl_p + al,      0.0,             2*al*rl*u_l, -rl*u_l**2-p, 0.0,                     al*u_l**2*rl_h  ])
    # Rows 5-6: energy flux
    row5 = jnp.array([ag*u_g*H_g*rg_p,          ag*rg*(H_g+u_g**2), 0.0,     rg*u_g*H_g,  ag*u_g*(H_g*rg_h + rg),  0.0              ])
    row6 = jnp.array([al*u_l*H_l*rl_p,          0.0,             al*rl*(H_l+u_l**2), -rl*u_l*H_l, 0.0,              al*u_l*(H_l*rl_h + rl)])

    return jnp.stack([row1, row2, row3, row4, row5, row6])


# ---------------------------------------------------------------------------
# X  — non-conservative defect  X = J*G_h - K
# Derivation §11.2.  Sparse: zero mass rows (1-2), zero void row (4),
# zero h_g/h_l columns (5-6).  Only 16 nonzero entries.
# ---------------------------------------------------------------------------

def X_matrix(U_h, eos_g: PhaseProps, eos_l: PhaseProps, cq: CommonQty) -> jnp.ndarray:
    """
    6×6 non-conservative defect X = J*G_h - K.

    Applied as X*K^{-1}*dF (never formed dense via K^{-1}):
        y = solve(K, dF)
        nc_contribution = X @ y

    The momentum-conservation identity X[3,:] = -X[2,:] is exact.
    Energy rows are exact linear combinations of momentum rows:
        X[4,:] = u_g*X[2,:] - p*G_4_row
        X[5,:] = u_l*X[3,:] + p*G_4_row
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  rl = eos_l.rho
    rg_hat = cq.rg_hat;  rl_hat = cq.rl_hat;  rh2 = cq.rh2
    rho_mix = cq.rho_mix;  rho_s = cq.rho_s;  a0sq = cq.a0sq
    du = cq.du

    ag_ = eos_g.a;  al_ = eos_l.a

    # Momentum-row nonzero entries (derivation §11.2)
    # X_{3p} = alpha_g*alpha_l*k*rho_mix*(rho_g-rho_l)/rh2
    # k*rho_mix is not stored in CommonQty; derive it from rg_hat - rg.
    k_rho_mix = rg_hat - rg
    X3p = ag * al * (rg - rl) * k_rho_mix / rh2

    # Velocity-column entries: X[gas_mom, u_i] = ag*rg*(G_uu - u_g),
    # where G_uu - u_g = -al*du*rl*rg_hat/rh2  (rl comes from both terms in G_uu - u_g).
    # Mixture momentum conservation forces X[liq_mom,:] = -X[gas_mom,:].
    X3g = -ag * al * du * rg * rl * rg_hat / rh2
    X3l = -ag * al * du * rg * rl * rl_hat / rh2
    X4g = X3g
    X4l = X3l

    # alpha_g column: ±p in momentum rows (derivation §11.2)
    # Void-row entries of G_h (needed for energy rows)
    G4p  =  al * ag * du / rho_s * a0sq / (ag_**2 * al_**2)
    G4g  =  al * ag * rg / rho_s * a0sq / al_**2
    G4l  = -al * ag * rl / rho_s * a0sq / ag_**2
    G4aa = (al * u_g * rg / rho_s * a0sq / al_**2
           + ag * u_l * rl / rho_s * a0sq / ag_**2)

    z = 0.0

    # Rows: (zero, zero, gas_mom, liq_mom, gas_E, liq_E)
    # Cols: (p,    u_g,  u_l,    alpha_g, h_g, h_l)
    row1 = jnp.array([z,     z,     z,     z,      z, z])
    row2 = jnp.array([z,     z,     z,     z,      z, z])
    row3 = jnp.array([X3p,   X3g,   X3l,  -p,      z, z])
    row4 = jnp.array([-X3p, -X4g,  -X4l,   p,      z, z])
    # Energy rows are u_i * momentum_row ± p * void_row
    row5 = jnp.array([
        u_g*X3p  - p*G4p,
        u_g*X3g  - p*G4g,
        u_g*X3l  - p*G4l,
        -p*G4aa,
        z, z,
    ])
    row6 = jnp.array([
        -u_l*X3p + p*G4p,
        -u_l*X4g + p*G4g,
        -u_l*X4l + p*G4l,
         p*G4aa,
        z, z,
    ])

    return jnp.stack([row1, row2, row3, row4, row5, row6])


# ---------------------------------------------------------------------------
# flux_vector  — physical flux F(U_h)
# Derivation §7.  Used to supply F_L, F_R to the FVS kernel.
# ---------------------------------------------------------------------------

def flux_vector(U_h, eos_g: PhaseProps, eos_l: PhaseProps) -> jnp.ndarray:
    """
    Physical flux vector F (6,) from the enthalpy-primitive state U_h.

    F = (ag*rho_g*u_g,  al*rho_l*u_l,
         ag*(rho_g*u_g^2 + p),  al*(rho_l*u_l^2 + p),
         ag*rho_g*u_g*H_g,      al*rho_l*u_l*H_l)

    with H_i = h_i + u_i^2/2 (total specific enthalpy).
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  rl = eos_l.rho

    H_g = h_g + 0.5 * u_g**2
    H_l = h_l + 0.5 * u_l**2

    return jnp.array([
        ag * rg * u_g,
        al * rl * u_l,
        ag * (rg * u_g**2 + p),
        al * (rl * u_l**2 + p),
        ag * rg * u_g * H_g,
        al * rl * u_l * H_l,
    ])


# ---------------------------------------------------------------------------
# eigenstructure  — closed-form eigenvalues + right/left eigenvectors of G_h
# Derivation §13 of stadtke_enthalpy_frame_derivation_complete.md.
# Verified in exact rational arithmetic at the nominal water/steam state.
# ---------------------------------------------------------------------------

def eigenstructure(U_h, eos_g: PhaseProps, eos_l: PhaseProps,
                   cq: CommonQty) -> tuple:
    """
    Closed-form eigenstructure of G_h in enthalpy primitives.

    Returns (eigvals, R_h, L_h, D) where:
      eigvals : (6,) real — [u_g, u_l, u_mix+a, u_mix-a, u_g, u_l]
      R_h     : (6,6) right eigenvectors as columns
      L_h     : (6,6) left eigenvectors as rows
      D       : (6,)  biorthogonality normalizers, L_h @ R_h = diag(D)

    Column/row ordering (derivation §13.1):
      1 — void gas wave     (λ = u_g)
      2 — void liquid wave  (λ = u_l)
      3 — acoustic+         (λ = u_mix + a)
      4 — acoustic−         (λ = u_mix − a)
      5 — entropy gas wave  (λ = u_g)
      6 — entropy liquid wave (λ = u_l)

    Typo note: a_1^2 carries hat_rho_l **squared** (not first power).
    See derivation §13.1 and §13.5 verification ledger.
    """
    _, u_g, u_l, ag, _, _ = U_h
    al = 1.0 - ag

    rg = eos_g.rho;  ag_sq = eos_g.a**2;  T_g = eos_g.T
    rl = eos_l.rho;  al_sq = eos_l.a**2;  T_l = eos_l.T

    rg_hat  = cq.rg_hat;  rl_hat  = cq.rl_hat;  rh2    = cq.rh2
    rho_mix = cq.rho_mix; rho_s   = cq.rho_s;   a0sq   = cq.a0sq
    du      = cq.du

    # ── Extra abbreviations for eigenstructure (§13.1) ──────────────────────
    # k*rho_mix is not stored in cq; derive from rg_hat = rg + k*rho_mix.
    k_rho_mix  = rg_hat - rg
    rho_hat_s  = rho_s + k_rho_mix          # hat_rho_s = rho_s + k*rho_mix
    rh         = jnp.sqrt(rh2)              # hat_rho = sqrt(hat_rho^2)

    a_tilde_sq = (rho_hat_s / rho_s) * (rg * rl / rh2) * a0sq
    delta_a_sq = ag * al * rg * rl * rg_hat * rl_hat / rh2**2 * du**2
    a_sq       = a_tilde_sq - delta_a_sq
    a          = jnp.sqrt(a_sq)  # real mixture sound speed

    hat_ag = ag * rg * rl_hat / rh2         # volume-fraction weight for u_g
    hat_al = al * rl * rg_hat / rh2         # volume-fraction weight for u_l
    u_mix  = hat_ag * u_g + hat_al * u_l    # mixture velocity

    # For left eigenvectors (§13.1)
    a1_sq   = a_sq - ag**2 * rl * rg * rl_hat**2 / rh2**2 * du**2
    a2_sq   = a_sq - al**2 * rl * rg * rg_hat**2 / rh2**2 * du**2
    delta_a = (ag**2 * rl_hat * a_tilde_sq / (rh * ag_sq)
             - al**2 * rg_hat * a_tilde_sq / (rh * al_sq))

    # ── Eigenvalues ─────────────────────────────────────────────────────────
    eigvals = jnp.array([u_g, u_l, u_mix + a, u_mix - a, u_g, u_l])

    # ── Right eigenvectors R_h (§13.2) ──────────────────────────────────────
    # Acoustic-block entries
    R_23 = rl_hat * (a  - al * rl * rg_hat / rh2 * du)
    R_24 = rl_hat * (-a - al * rl * rg_hat / rh2 * du)
    R_33 = rg_hat * (a  + ag * rg * rl_hat / rh2 * du)
    R_34 = rg_hat * (-a + ag * rg * rl_hat / rh2 * du)

    chi = (-ag * al * rh2 / rho_hat_s * a_tilde_sq
           * (rg_hat / (rg * ag_sq) - rl_hat / (rl * al_sq)))

    # col 1: void gas wave (λ = u_g)
    c1 = jnp.array([
        al * rl * du**2,
        0.0,
        du,
        -al * (1.0 - al * du**2 / al_sq),
        al * rl * du**2 / rg,
        al * du**2,
    ])
    # col 2: void liquid wave (λ = u_l)
    c2 = jnp.array([
        ag * rg * du**2,
        -du,
        0.0,
        ag * (1.0 - ag * du**2 / ag_sq),
        ag * du**2,
        ag * rg * du**2 / rl,
    ])
    # col 3: acoustic+ (λ = u_mix + a)
    c3 = jnp.array([
        rh2 * a_tilde_sq,
        R_23,
        R_33,
        chi,
        rh2 * a_tilde_sq / rg,
        rh2 * a_tilde_sq / rl,
    ])
    # col 4: acoustic− (λ = u_mix − a)
    c4 = jnp.array([
        rh2 * a_tilde_sq,
        R_24,
        R_34,
        chi,
        rh2 * a_tilde_sq / rg,
        rh2 * a_tilde_sq / rl,
    ])
    # col 5: entropy gas wave (λ = u_g) — trivial
    c5 = jnp.array([0.0, 0.0, 0.0, 0.0, T_g, 0.0])
    # col 6: entropy liquid wave (λ = u_l) — trivial
    c6 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, T_l])

    R_h = jnp.stack([c1, c2, c3, c4, c5, c6], axis=1)  # (6,6)

    # ── Left eigenvectors L_h (§13.3) ───────────────────────────────────────
    # Acoustic-block entries
    L_32 = ag * rh2 * (a2_sq + a * du * rg * rho_hat_s / rh2)
    L_33 = al * rh2 * (a1_sq - a * du * rl * rho_hat_s / rh2)
    L_42 = ag * rh2 * (a2_sq - a * du * rg * rho_hat_s / rh2)
    L_43 = al * rh2 * (a1_sq + a * du * rl * rho_hat_s / rh2)

    # row 1: dual to void gas (λ = u_g)
    l1 = jnp.array([
        ag * du / ag_sq,
        ag * rg,
        -ag * rg * rl_hat / rg_hat,
        rg * du,
        0.0,
        0.0,
    ])
    # row 2: dual to void liquid (λ = u_l)
    l2 = jnp.array([
        al * du / al_sq,
        al * rl * rg_hat / rl_hat,
        -al * rl,
        -rl * du,
        0.0,
        0.0,
    ])
    # row 3: dual to acoustic+ (λ = u_mix + a)
    l3 = jnp.array([
        a * rho_hat_s + rh * du * delta_a,
        L_32,
        L_33,
        rh2 * du * a_tilde_sq,
        0.0,
        0.0,
    ])
    # row 4: dual to acoustic− (λ = u_mix − a)
    l4 = jnp.array([
        -a * rho_hat_s + rh * du * delta_a,
        L_42,
        L_43,
        rh2 * du * a_tilde_sq,
        0.0,
        0.0,
    ])
    # row 5: dual to entropy gas — from Phi rows (derivation §3)
    l5 = jnp.array([-1.0 / (rg * T_g), 0.0, 0.0, 0.0, 1.0 / T_g, 0.0])
    # row 6: dual to entropy liquid
    l6 = jnp.array([-1.0 / (rl * T_l), 0.0, 0.0, 0.0, 0.0, 1.0 / T_l])

    L_h = jnp.stack([l1, l2, l3, l4, l5, l6])  # (6,6)

    # Biorthogonality normalizers D = diag(L_h @ R_h) (§13.4)
    # D_5 = D_6 = 1 exactly (T_i * 1/T_i cancellation);
    # D_1, D_2 ~ du^2;  D_3, D_4 ~ rh2 * a^2
    D = jnp.diag(L_h @ R_h)
    # TODO: do we need this matrix D?

    return eigvals, R_h, L_h, D
