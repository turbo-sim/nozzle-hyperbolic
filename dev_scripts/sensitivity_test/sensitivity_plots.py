"""
sensitivity_plots.py

Figures for the AD closure-sensitivity study.  Pure plotting: every function
takes plain numpy arrays and returns a Figure, so nothing here needs JAX or the
solver and the module can be exercised on canned data.

Why these figures exist
-----------------------
`sensitivity_closure.py` computes dW/dtheta — the sensitivity of the FULL state
field, 8 components x Nc cells — and then immediately contracts it against
dJ/dW down to a 2x2 table of scalars.  That contraction throws away the entire
spatial story: WHERE in the nozzle the closure parameter acts, and through which
variable.  These plots put it back.

The scalar table says "10x more bubbles gives ~13% less mass flow".  The field
plot says whether that came from the subsonic run-up, the throat, or the
flashing front — which is the part that tells you what to model better.

Conventions
-----------
* Plotted quantity is d(phi)/d(ln theta) = theta * d(phi)/d(theta): the change in
  a field per e-fold (x2.718) of the parameter, in the field's OWN physical
  units.  No normalization by phi, so nothing blows up where a field crosses
  zero (u_g and du both do).
* Parameter identity is carried by colour AND dash pattern, never colour alone —
  the palette is only certified colour-safe for its first three slots when every
  pair can appear together, which is exactly the case in a small-multiples grid.
* State and sensitivity are drawn as STACKED panels sharing x, never as two
  y-scales on one axes.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical slots in fixed order (never cycled, never reassigned when the
# parameter set changes — colour follows the parameter, not its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Secondary encoding.  Required past slot 3: with all pairs on screen at once the
# 4th slot puts yellow beside orange, which is not separable under CVD.  The dash
# pattern makes identity readable regardless.
DASHES = [(None, None), (5, 2), (1.4, 1.4), (7, 2, 1.4, 2),
          (4, 1.4, 1.4, 1.4), (2.4, 1.6), (6, 1.6, 1.6, 1.6), (1.0, 1.6)]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ZERO_LINE = "#c3c2b7"
# Regime wash — neutral greys, ordinal in alpha_g, deliberately recessive.
REGIME_WASH = {0: "#f4f3f0", 1: "#e8e7e1", 2: "#dbdad2"}
REGIME_NAME = {0: "bubbly", 1: "transition", 2: "droplet"}


def _style(name, j):
    """Colour + dash for parameter slot j."""
    return dict(color=PALETTE[j % len(PALETTE)],
                dashes=DASHES[j % len(DASHES)],
                lw=1.6, label=name)


# ── Regime shading ────────────────────────────────────────────────────────────

def regime_labels(alpha_g, alpha_b=0.3, alpha_d=0.7):
    """0 = bubbly, 1 = transition, 2 = droplet, per cell."""
    return np.where(alpha_g <= alpha_b, 0,
                    np.where(alpha_g >= alpha_d, 2, 1)).astype(int)


def _runs(lab):
    """Contiguous [start, end) index runs of a constant label."""
    if lab.size == 0:
        return []
    cut = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [lab.size]])
    return list(zip(starts, ends))


def shade_regimes(ax, x, lab, annotate=False):
    """Wash the axes background by flow regime.  Chrome, not data."""
    for s, e in _runs(lab):
        x0 = x[s] if s == 0 else 0.5 * (x[s - 1] + x[s])
        x1 = x[-1] if e == lab.size else 0.5 * (x[e - 1] + x[e])
        ax.axvspan(x0, x1, color=REGIME_WASH[lab[s]], zorder=0, lw=0)
        if annotate and (x1 - x0) > 0.12 * (x[-1] - x[0]):
            ax.text(0.5 * (x0 + x1), 0.96, REGIME_NAME[lab[s]],
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color=MUTED)


def _decorate(ax, x, lab, x_throat=None):
    shade_regimes(ax, x, lab)
    if x_throat is not None:
        ax.axvline(x_throat, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(x[0], x[-1])


# ── Figure 1 — where the parameter acts ───────────────────────────────────────

def plot_sensitivity_fields(x, state, sens, param_names, inert=(),
                            alpha_g=None, x_throat=None, title=""):
    """State profiles over their own sensitivity profiles, one column per field.

    Parameters
    ----------
    x : (Nc,)            interior-cell centres [m]
    state : dict name -> (Nc,)            baseline field
    sens  : dict name -> (n_param, Nc)    d(field)/d(ln theta_j), field units
    param_names : list[str]
    inert : names of parameters that are structurally inert (flat zero, labelled)
    alpha_g : (Nc,) for the regime wash; None disables it
    """
    fields = list(state.keys())
    ncol = 3
    nsup = int(np.ceil(len(fields) / ncol))
    lab = (regime_labels(alpha_g) if alpha_g is not None
           else np.zeros(x.size, int))

    # A spacer row between super-rows: the state/sensitivity pair inside one
    # super-row shares an x-axis and must sit tight, while consecutive pairs are
    # unrelated and need air.  A uniform hspace cannot express both.
    fig = plt.figure(figsize=(4.6 * ncol, 3.7 * nsup))
    gs = GridSpec(3 * nsup - 1, ncol, figure=fig,
                  height_ratios=([1.0, 2.0, 0.55] * nsup)[:3 * nsup - 1],
                  hspace=0.16, wspace=0.32)

    for k, f in enumerate(fields):
        r, c = divmod(k, ncol)
        ax_s = fig.add_subplot(gs[3 * r, c])                   # state
        ax_d = fig.add_subplot(gs[3 * r + 1, c], sharex=ax_s)  # sensitivity

        unit, scale = FIELD_UNITS.get(f, ("", 1.0))
        ax_s.plot(x, np.asarray(state[f]) / scale, color=INK, lw=1.4, zorder=3)
        ax_s.set_ylabel(f"{FIELD_LABEL.get(f, f)}{unit}", fontsize=10)
        ax_s.tick_params(labelbottom=False)
        _decorate(ax_s, x, lab, x_throat)
        if k == 0:
            shade_regimes(ax_s, x, lab, annotate=True)

        ax_d.axhline(0.0, color=ZERO_LINE, lw=1.0, zorder=1)
        for j, nm in enumerate(param_names):
            lbl = nm + ("  (inert)" if nm in inert else "")
            st = _style(lbl, j)
            ax_d.plot(x, np.asarray(sens[f][j]) / scale, zorder=3, **st)
        ax_d.set_ylabel(rf"$\partial${FIELD_LABEL.get(f, f)}$/\partial\ln\theta$"
                        + unit, fontsize=10)
        if r == nsup - 1:
            ax_d.set_xlabel("$x$ [m]", fontsize=10)
        _decorate(ax_d, x, lab, x_throat)

    handles = [Line2D([], [], **_style(nm + ("  (inert)" if nm in inert else ""), j))
               for j, nm in enumerate(param_names)]
    fig.legend(handles=handles, loc="upper right", ncol=len(param_names),
               bbox_to_anchor=(0.995, 0.995), frameon=True)
    if title:
        fig.suptitle(title, y=1.005, fontsize=12, color=INK2)
    fig.tight_layout()
    return fig


FIELD_LABEL = {"p": "$p$", "alpha_g": r"$\alpha_g$", "u_g": "$u_g$",
               "u_l": "$u_l$", "du": r"$u_g-u_l$", "mdot": r"$\dot m$",
               "m_g": r"$\alpha_g\rho_g$", "m_l": r"$\alpha_l\rho_l$"}
FIELD_UNITS = {"p": (" [bar]", 1e5), "alpha_g": (" [-]", 1.0),
               "u_g": (" [m/s]", 1.0), "u_l": (" [m/s]", 1.0),
               "du": (" [m/s]", 1.0), "mdot": (" [kg/s]", 1.0),
               "m_g": (" [kg/m$^3$]", 1.0), "m_l": (" [kg/m$^3$]", 1.0)}


# ── Figure 2 — the closure mechanism and its switches ─────────────────────────

def plot_regime_map(x, alpha_g, ai, d, dai_dlnN, param_names,
                    alpha_b=0.3, alpha_d=0.7, alpha_b_star=None,
                    alpha_d_star=None, x_throat=None, title=""):
    """Void fraction against the regime switches, and the interfacial-area
    response that is the ONLY channel through which N_b / N_d reach the physics.

    A parameter whose branch the flow never enters has an exactly zero
    sensitivity.  That is a structural fact about the regime, not a numerical
    artefact, and this figure is where it is read off.
    """
    lab = regime_labels(alpha_g, alpha_b, alpha_d)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4), sharex=True)

    ax = axes[0]
    ax.plot(x, alpha_g, color=INK, lw=1.5, zorder=3)
    for v, nm in ((alpha_b, r"$\alpha_b$"), (alpha_d, r"$\alpha_d$")):
        ax.axhline(v, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(x[0], v, f" {nm}={v:g}", fontsize=8, color=MUTED, va="bottom")
    # The nucleation / breakup floors are drawn only when they are actually in
    # play.  For the usual N and d_min they sit within ~1e-7 of 0 and 1, where a
    # line would be indistinguishable from the axis and would collide with the
    # regime labels while telling the reader nothing; the caption carries the
    # value instead.
    off = []
    for v, nm in ((alpha_b_star, r"$\alpha_b^*$ nucleation floor"),
                  (alpha_d_star, r"$\alpha_d^*$ breakup floor")):
        if v is None:
            continue
        if 0.02 < v < 0.98:
            ax.axhline(v, color=INK2, lw=0.9, ls=(0, (1.2, 1.6)), zorder=2)
            ax.text(x[-1], v, f"{nm} ", fontsize=8, color=INK2,
                    va="bottom", ha="right")
        else:
            off.append(f"{nm} at {v:.2e} — never reached")
    ax.set_ylabel(r"$\alpha_g$ [-]")
    ax.set_ylim(-0.03, 1.03)
    _decorate(ax, x, lab, x_throat)
    shade_regimes(ax, x, lab, annotate=True)
    if off:
        ax.text(0.985, 0.06, "\n".join(off), transform=ax.transAxes,
                fontsize=8, color=MUTED, ha="right", va="bottom")

    ax = axes[1]
    ax.plot(x, ai, color=INK, lw=1.5, zorder=3, label=r"$a_i$")
    ax.set_ylabel(r"$a_i$ [1/m]")
    ax.set_yscale("log")
    _decorate(ax, x, lab, x_throat)
    ax.set_title(rf"dispersed diameter $d$ = {d.min():.2e} .. {d.max():.2e} m",
                 fontsize=9, color=INK2)

    ax = axes[2]
    ax.axhline(0.0, color=ZERO_LINE, lw=1.0, zorder=1)
    for j, nm in enumerate(param_names):
        ax.plot(x, dai_dlnN[j], zorder=3, **_style(nm, j))
    ax.set_ylabel(r"$\partial a_i/\partial\ln\theta$ [1/m]")
    ax.set_xlabel("$x$ [m]")
    ax.legend(loc="best")
    _decorate(ax, x, lab, x_throat)

    if title:
        fig.suptitle(title, y=0.995, fontsize=12, color=INK2)
    fig.tight_layout()
    return fig


# ── Figure 3 — AD vs FD, field by field (the strong form of the §4 gate) ──────

def plot_ad_vs_fd(x, sens_ad, sens_fd, fields, param_name,
                  alpha_g=None, x_throat=None, title=""):
    """Overlay the AD field sensitivity on a central finite difference of two
    re-converged runs.

    Comparing the two SCALARS only tells you they disagree.  Comparing the two
    FIELDS tells you where — a disagreement confined to a few cells at an
    outflow boundary or across a regime switch is a different finding from one
    spread over the whole domain.
    """
    lab = (regime_labels(alpha_g) if alpha_g is not None
           else np.zeros(x.size, int))
    n = len(fields)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, f in enumerate(fields):
        ax = axes[k // ncol][k % ncol]
        unit, scale = FIELD_UNITS.get(f, ("", 1.0))
        ax.axhline(0.0, color=ZERO_LINE, lw=1.0, zorder=1)
        ax.plot(x, np.asarray(sens_ad[f]) / scale, color=PALETTE[0],
                lw=1.8, zorder=3, label="AD (implicit)")
        ax.plot(x, np.asarray(sens_fd[f]) / scale, color=PALETTE[1],
                lw=1.6, dashes=(4, 2), zorder=4, label="FD (central)")
        ax.set_ylabel(rf"$\partial${FIELD_LABEL.get(f, f)}"
                      rf"$/\partial\ln${param_name}" + unit, fontsize=10)
        ax.set_xlabel("$x$ [m]", fontsize=10)
        _decorate(ax, x, lab, x_throat)
        if k == 0:
            ax.legend(loc="best")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if title:
        fig.suptitle(title, y=1.005, fontsize=12, color=INK2)
    fig.tight_layout()
    return fig


# ── Figure 4 — is the linear algebra trustworthy ──────────────────────────────

def plot_diagnostics(S, rcond_rows, dt_rows, qoi_names, param_names, title=""):
    """Singular spectrum of the scaled Jacobian, plus the two invariance sweeps.

    The sweeps are the actual verification: a sensitivity that MOVES when the
    pseudo-inverse cut or the steady-Jacobian dt is changed is not determined by
    the steady state and must not be reported.  Both are plotted normalized to
    their own largest magnitude, so "flat" is the pass in every panel regardless
    of the elasticities' absolute size.

    The sweeps are FACETED by quantity of interest rather than drawn as one
    n_qoi x n_param bundle: past three series on a single axes the categorical
    hues stop being separable, and faceting is the fix that keeps every curve
    readable instead of leaning on the dash pattern alone.

    rcond_rows : list of (rcond, rank, flat elasticities ordered [qoi][param])
    dt_rows    : list of (dt/dt_cfl, cond, same ordering)
    """
    nq = len(qoi_names)
    nrow = 2 if dt_rows else 1
    fig, axes = plt.subplots(nrow, 1 + nq, figsize=(4.7 * (1 + nq), 3.9 * nrow),
                             squeeze=False)

    ax = axes[0][0]
    ax.semilogy(np.arange(1, S.size + 1), S / S[0], color=INK, lw=1.4)
    ax.set_xlabel("index"); ax.set_ylabel(r"$\sigma_i/\sigma_1$ [-]")
    ax.set_title("scaled-Jacobian spectrum", fontsize=10, color=INK2)
    for rc in (1e-8, 1e-12):
        ax.axhline(rc, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        ax.text(1, rc, f" rcond={rc:.0e}", fontsize=8, color=MUTED, va="bottom")
    if nrow == 2:
        axes[1][0].axis("off")

    def _sweep(ax, rows, i_qoi, xlab, ref_x):
        """Plot the RELATIVE departure from the production setting.

        Normalizing each curve by its own maximum would put every passing sweep
        at a flat +-1 and let the autoscale magnify float noise into what looks
        like structure.  Against a fixed reference, zero is the pass and the
        y-axis reads directly as "how wrong would this setting have made me".
        """
        xs = np.array([r[0] for r in rows], float)
        vals = np.array([r[2] for r in rows], float)   # (n_row, n_qoi*n_param)
        i_ref = int(np.argmin(np.abs(np.log10(xs) - np.log10(ref_x))))
        ax.axhspan(-0.01, 0.01, color="#f0efec", zorder=0, lw=0)
        ax.axhline(0.0, color=ZERO_LINE, lw=1.0, zorder=1)
        worst = 0.02
        for j, nm in enumerate(param_names):
            col = vals[:, i_qoi * len(param_names) + j]
            ref = col[i_ref]
            dev = (col - ref) / abs(ref) if ref != 0 else np.zeros_like(col)
            worst = max(worst, float(np.max(np.abs(dev))) * 1.25)
            ax.plot(xs, dev, marker="o", ms=4, zorder=3, **_style(nm, j))
        ax.set_xscale("log")
        ax.set_ylim(-worst, worst)
        ax.set_xlabel(xlab)
        ax.set_ylabel("relative change in elasticity [-]", fontsize=9)
        ax.legend(loc="best", fontsize=8)

    for i, q in enumerate(qoi_names):
        _sweep(axes[0][1 + i], rcond_rows, i, "pseudo-inverse cut  rcond", 1e-12)
        axes[0][1 + i].set_title(f"rcond invariance — {q}  (band = $\\pm$1%)",
                                 fontsize=10, color=INK2)
        if dt_rows:
            _sweep(axes[1][1 + i], dt_rows, i, r"$dt/dt_{\mathrm{CFL}}$",
                   max(r[0] for r in dt_rows))
            axes[1][1 + i].set_title(f"$dt$ invariance — {q}", fontsize=10,
                                     color=INK2)

    if title:
        fig.suptitle(title, y=1.005, fontsize=12, color=INK2)
    fig.tight_layout()
    return fig
