"""
plot_gasdom_journal.py

Journal-quality figures for the GAS-DOMINANT (perfect-gas) nozzle validation.
Same visual family as plot_phase_change_journal.py / plot_two_component_journal.py
(one colour per back-pressure regime, nozzle silhouette, serif style), but this
one is a VALIDATION plot: the analytic ideal-gas solution is drawn on top of the
solver result for every case that has one.

  1. gasdom_p_mach.png   — pressure (solid, left axis) and gas Mach number
                           (dashed, right axis) vs x, one colour per case, with
                           the analytic reference in black and the sonic line.
  2. gasdom_error.png    — validation panels: (p - p_ana)/p0 [%] (top) and
                           M - M_ana (bottom).  Boundary cells and the smeared
                           shock zone are masked, as in check_shock.py.
  3. gasdom_velocity.png — gas velocity u_g vs x with the analytic reference
                           (MAKE_VELOCITY_FIG = False to skip).

Analytic references (all from isentropic_ic.py, i.e. the SAME construction the
ICs and check_*.py use, with gamma and R derived from the run's own gas EOS at
the reservoir state):
  * 'isentropic' — choked perfect (adapted) expansion, area-Mach relation.
  * 'shock'      — choked flow with a standing normal shock: isentropic to the
                   shock, Rankine-Hugoniot jump, post-shock subsonic branch on
                   the p02 isentrope.  The shock is placed at the cell DETECTED
                   in the solver field (SHOCK_REF='detected', the a-posteriori
                   comparison used by check_shock.py) or at the designed
                   ic.x_shock from the settings (SHOCK_REF='design').
  * 'subsonic'   — fully subsonic, UNCHOKED isentropic flow: M_exit from
                   p_out/p0, virtual sonic area A* = A_exit/(A/A*)(M_exit), then
                   the subsonic area-Mach branch everywhere.  Built here (it is
                   not in isentropic_ic.py) and skipped automatically if the
                   back-pressure would in fact choke the nozzle.

Every case in RUNS whose folder has no solution.csv is skipped with a note, so a
placeholder path for a run that does not exist yet is harmless.

Run:  poetry run python plot_gasdom_journal.py
"""
import pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
import jaxprop as jxp

from config_loader import load_config
from nozzle_silhouette import add_nozzle
from isentropic_ic import isentropic_ic, isentropic_shock_ic, _area_mach

# ── KNOBS: the cases to overlay (path relative to dev_scripts, label, ref) ─────
# ref: 'auto' reads ic.type from the run's own settings yaml
#      ('isentropic' → adapted, 'isentropic_shock' → shock, anything else →
#      unchoked subsonic reference);  or force 'isentropic' | 'shock' |
#      'subsonic' | None (no analytic line for that case).
# NOTE: all three entries currently point at the SHOCK run — swap in the real
# subsonic / adapted folders when those simulations are done; nothing else needs
# to change ('auto' picks the right reference from each run's own settings).
_SUBSONIC_RUN = "results_single_phase_validation/perfect_gas_N500_muscl_van_albada_20260728_180246"

_SHOCK_RUN = "results/nozzle_valid_gasdom_N600_muscl_van_albada_20260718_180641"
_ADAPTED_RUN = "results_single_phase_validation/perfect_gas_N401_muscl_van_albada_20260727_185925"
RUNS = [
    (_SUBSONIC_RUN, "subsonic",             "auto"),
    (_SHOCK_RUN, "supersonic-shock",     "auto"),
    (_ADAPTED_RUN, "supersonic-adapted", "auto"),
]
INCLUDE_SUBSONIC = True   # False = drop the first entry without editing RUNS
OUT_DIR = "results_gasdom_journal"   # where the overlay figures are written
PREFIX  = "gasdom"                   # output file name prefix
# Shared folder (relative to dev_scripts) collecting every paper figure, next to
# the ones from plot_bernoulli_journal.py.  None = write to OUT_DIR only.
EXTRA_OUT_DIR = "nicfd_figures"
# Nozzle silhouette proportion.  False = schematic 16% band (radius NOT to scale).
# True = proportion-true (1 m in x == 1 m of radius) — sensible for the Elliot
# geometry (r_max/L ~ 0.08); nonsensical for the symmetric placeholder.
NOZZLE_TRUE_SCALE = False

SHOCK_REF = "detected"    # 'detected' = shock cell found in the solver field
                          # 'design'   = ic.x_shock from the settings yaml
ANA_STYLE = "line"        # 'line' = analytic as a thin dark line (behind the
                          # solver curve) | 'markers' = open circles (the
                          # convention of plot_bernoulli_journal.py)
NPOINTS   = 30            # number of analytic markers when ANA_STYLE='markers'
MASK_HALF = 4             # cells masked either side of the shock in the errors
MAKE_VELOCITY_FIG = True

# case colours are sampled from a colormap: pick the map and how many colours
CMAP_NAME = "plasma"
N_COLORS  = 4
CMAP_LO, CMAP_HI = 0.0, 0.85
# ──────────────────────────────────────────────────────────────────────────────

# ── Journal style (matches the other plot_*_journal.py scripts) ───────────────
plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   10,
    "legend.fontsize":  9.5,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.8,
    "legend.frameon":   False,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})
_cmap = plt.get_cmap(CMAP_NAME)
CASE_COLORS = [_cmap(t) for t in np.linspace(CMAP_LO, CMAP_HI, N_COLORS)]
C_ANA = "0.15"            # analytic reference colour (near-black)
_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

_here = pathlib.Path(__file__).parent


def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"])
    return jxp.FluidBicubic(**lut)


def _area_ratio(M, g):
    """A/A* from the area-Mach relation (inverse of _area_mach)."""
    return (1.0 / M) * ((2.0 / (g + 1.0))
                        * (1.0 + 0.5 * (g - 1.0) * M * M)
                        ) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def subsonic_profile(x, A, *, p0, T0, p_out, gamma, R):
    """Analytic UNCHOKED subsonic isentropic profile set by the back-pressure.

    M_exit from p0/p_out, virtual sonic area A* = A_exit / (A/A*)(M_exit), then
    the subsonic area-Mach branch at every station.  Returns None (with a note)
    if the pressure ratio would choke the nozzle, i.e. if the required A* is
    larger than the actual throat area.
    """
    g = gamma
    if not (p_out < p0):
        print("    subsonic reference: p_out >= p0 — skipped")
        return None
    M_e = np.sqrt(2.0 / (g - 1.0) * ((p0 / p_out) ** ((g - 1.0) / g) - 1.0))
    if M_e >= 1.0:
        print(f"    subsonic reference: exit would be supersonic "
              f"(M_exit={M_e:.3f}) — skipped")
        return None
    A_star = float(A[-1]) / _area_ratio(M_e, g)
    if A_star > float(A.min()) * (1.0 + 1e-12):
        print(f"    subsonic reference: back-pressure chokes the nozzle "
              f"(A*={A_star:.4e} > A_throat={A.min():.4e}) — skipped")
        return None
    M = np.array([_area_mach(a / A_star, g, "sub") for a in A])
    T = T0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    p = p0 * (T / T0) ** (g / (g - 1.0))
    u = M * np.sqrt(g * R * T)
    print(f"    subsonic reference: M_exit={M_e:.4f}  M_throat={M.min():.4f}"
          f"..{M.max():.4f}  A*/A_throat={A_star/float(A.min()):.4f}")
    return dict(x=x, A=A, M=M, p=p, T=T, u=u, gamma=g, R=R)


def load_case(rel_path, ref_mode):
    """Return dict with x, A, r, p, u_g, M (solver) and the analytic profile."""
    run_dir = (_here / rel_path) if not pathlib.Path(rel_path).is_absolute() \
        else pathlib.Path(rel_path)
    if not (run_dir / "solution.csv").exists():
        print(f"  SKIP (no solution.csv): {run_dir}")
        return None
    yml = sorted(run_dir.glob("settings*.yaml"))[0]
    cfg = load_config(yml)
    raw = yaml.safe_load(open(yml, encoding="utf-8"))

    D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    x   = np.asarray(D["x"]);   N = x.size
    A   = np.asarray(D["A"], float)      # solver's own area column
    p   = np.asarray(D["p"])
    u_g = np.asarray(D["u_g"]); h_g = np.asarray(D["h_g"])
    r   = np.sqrt(A / np.pi)             # radius for the nozzle silhouette

    fluid_g = _build_fluid(cfg.LUT_G)
    fluid_l = _build_fluid(cfg.LUT_L)
    p0 = float(cfg.BC_P0); T0_g = float(cfg.BC_T_G0)
    # gamma, R from the run's own gas EOS at the reservoir state (as in
    # isentropic_ic) — the reference is then self-consistent with the solver.
    st0   = fluid_g.get_state(jxp.PT_INPUTS, jnp.array([p0]), jnp.array([T0_g]))
    R     = p0 / (float(st0["density"][0]) * T0_g)
    gamma = float(st0["speed_of_sound"][0]) ** 2 / (R * T0_g)

    # solver Mach from (h_g, p) through the gas EOS — same as plot_gasdom_compare
    T_g = np.asarray(fluid_g.get_state(jxp.HmassP_INPUTS,
                                       jnp.array(h_g), jnp.array(p))["temperature"])
    M = np.abs(u_g) / np.sqrt(gamma * R * T_g)

    # ── analytic reference ────────────────────────────────────────────────────
    ic_type = str(raw.get("ic", {}).get("type", "")).strip().lower()
    mode = ref_mode
    if mode == "auto":
        mode = {"isentropic": "isentropic",
                "isentropic_shock": "shock"}.get(ic_type, "subsonic")

    void = cfg.BC_VOID if cfg.BC_VOID is not None else float(D["alpha_g"][0])
    ic_kw = dict(p0=p0, T0_g=T0_g, T0_l=float(cfg.BC_T_L0),
                 alpha_g=float(void), slip=float(cfg.BC_SLIP),
                 fluid_g=fluid_g, fluid_l=fluid_l)

    i_th = int(np.argmin(A))
    prof, i_sh, prof_isen = None, None, None
    if mode == "isentropic":
        _, prof = isentropic_ic(x, A, **ic_kw)
    elif mode == "shock":
        _, prof_isen = isentropic_ic(x, A, **ic_kw)   # unshocked isentrope too
        if SHOCK_REF == "design" and raw.get("ic", {}).get("x_shock") is not None:
            x_sh = float(raw["ic"]["x_shock"])
        else:
            # detect the discrete shock in the solver field (check_shock.py)
            dpx = np.abs(np.diff(p)); dpx[:i_th + 2] = 0.0
            x_sh = float(x[int(np.argmax(dpx))])
        _, prof = isentropic_shock_ic(x, A, **ic_kw, x_shock=x_sh)
        i_sh = int(prof["i_shock"])
    elif mode == "subsonic":
        prof = subsonic_profile(x, A, p0=p0, T0=T0_g, p_out=float(cfg.P_OUT),
                                gamma=gamma, R=R)
    elif mode is not None:
        raise ValueError(f"unknown reference mode {ref_mode!r}")

    # error mask: boundary cells + the smeared shock zone
    ok = np.ones(N, dtype=bool)
    ok[:2] = ok[-2:] = False
    if i_sh is not None:
        ok[max(i_sh - MASK_HALF, 0): i_sh + MASK_HALF + 1] = False

    print(f"  {run_dir.name}:  N={N}  p0={p0/1e5:.3f} bar  p_out={cfg.P_OUT/1e5:.3f} bar"
          f"  gamma={gamma:.4f}  max M={M[1:-1].max():.3f}  ref={mode}")
    if i_sh is not None:
        print(f"    shock at cell {i_sh} (x={x[i_sh]:.4f}, {SHOCK_REF}):"
              f"  M1={prof['M1']:.4f} -> M2={prof['M2']:.4f}"
              f"   p2/p1={prof['p_ratio']:.4f}")
    if prof is not None:
        dp = np.max(np.abs(p[ok] - prof["p"][ok])) / p0
        dM = np.max(np.abs(M[ok] - prof["M"][ok]))
        print(f"    masked error:  max|dp|/p0 = {dp:.3e}   max|dM| = {dM:.3e}")

    return dict(name=run_dir.name, x=x, A=A, r=r, p=p, u_g=u_g, M=M,
                prof=prof, prof_isen=prof_isen, i_shock=i_sh, ok=ok,
                p0=p0, mode=mode)


# add_nozzle now lives in nozzle_silhouette.py (shared by all journal scripts);
# NOZZLE_TRUE_SCALE (top of file) toggles schematic vs proportion-true.


def save_fig(fig, name):
    """Write the figure into OUT_DIR and, if EXTRA_OUT_DIR is set, into the
    shared paper-figure folder as well."""
    out = out_dir / name
    fig.savefig(out)
    print(f"Plot saved: {out}")
    if EXTRA_OUT_DIR:
        shared = _here / EXTRA_OUT_DIR
        shared.mkdir(parents=True, exist_ok=True)
        fig.savefig(shared / name)
        print(f"Plot saved: {shared / name}")


def plot_ana(ax, x, y, dashed=False):
    """Draw an analytic reference in the chosen style."""
    if ANA_STYLE == "markers":
        idx = np.linspace(1, x.size - 2, NPOINTS).round().astype(int)
        ax.plot(x[idx], y[idx], color=C_ANA, ls="none", marker="o",
                mfc="white", ms=5.0, mew=1.0, zorder=3)
    else:
        ax.plot(x, y, color=C_ANA, ls="--" if dashed else "-", lw=1.0,
                zorder=3, alpha=0.9)


# ── Load every case ───────────────────────────────────────────────────────────
print("Loading cases:")
cases = []
for slot, (rel, label, ref) in enumerate(RUNS):
    if label == "subsonic" and not INCLUDE_SUBSONIC:
        print("  SKIP (INCLUDE_SUBSONIC=False): subsonic")
        continue
    c = load_case(rel, ref)
    if c is not None:
        c["label"] = label
        # colour follows the RUNS SLOT, not the load order, so the shock and
        # adapted curves keep their colour whether or not the subsonic run exists
        c["color"] = CASE_COLORS[slot % len(CASE_COLORS)]
        cases.append(c)
if not cases:
    sys.exit("No cases loaded — check the RUNS paths.")
out_dir = (_here / OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

_nz = max(cases, key=lambda c: float(c["x"][-1]))     # widest domain → silhouette
case_handles = [Line2D([], [], color=c["color"], ls="-", lw=1.9, label=c["label"])
                for c in cases]
_ana_handle = (Line2D([], [], color=C_ANA, ls="none", marker="o", mfc="white",
                      ms=5.0, mew=1.0, label="analytic")
               if ANA_STYLE == "markers" else
               Line2D([], [], color=C_ANA, ls="-", lw=1.0, label="analytic"))

# ── FIGURE 1: pressure (solid, left) + Mach (dashed, right) + analytic ────────
fig1, axp = plt.subplots(figsize=(7.0, 4.3))
axm = axp.twinx()
for c in cases:
    col = c["color"]
    axp.plot(c["x"], c["p"] / 1e5, color=col, ls="-",  lw=1.9, zorder=2)
    axm.plot(c["x"], c["M"],       color=col, ls="--", lw=1.6, zorder=2)
    if c["prof"] is not None:
        plot_ana(axp, c["x"], c["prof"]["p"] / 1e5)
        plot_ana(axm, c["x"], c["prof"]["M"], dashed=True)
axm.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)          # sonic line
add_nozzle(axp, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
axm.annotate(r"$M=1$", xy=(axp.get_xlim()[0], 1.0), xytext=(3, 3),
             textcoords="offset points", color="0.4", fontsize=9,
             ha="left", va="bottom")
axp.set_xlabel(r"$x$ [m]")
axp.set_ylabel(r"pressure $p$ [bar]")
axm.set_ylabel(r"Mach number $M = u_g / \sqrt{\gamma R T_g}$")

style_handles = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label="pressure"),
                 Line2D([], [], color="0.3", ls="--", lw=1.6, label="Mach"),
                 _ana_handle]
# _leg1 = axp.legend(handles=case_handles, loc="upper center")
# axp.add_artist(_leg1)
axp.legend(handles=style_handles, loc="center left")
fig1.tight_layout()
save_fig(fig1, f"{PREFIX}_p_mach.png")

# ── FIGURE 2: validation errors vs the analytic reference ────────────────────
_val = [c for c in cases if c["prof"] is not None]
if _val:
    fig2, (axdp, axdM) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    for c in _val:
        col = c["color"]
        m = c["ok"]                       # masked: boundaries + shock zone
        e_p = np.where(m, (c["p"] - c["prof"]["p"]) / c["p0"] * 100.0, np.nan)
        e_M = np.where(m, c["M"] - c["prof"]["M"], np.nan)
        axdp.plot(c["x"], e_p, color=col, ls="-", lw=1.4, zorder=2)
        axdM.plot(c["x"], e_M, color=col, ls="-", lw=1.4, zorder=2)
        if c["i_shock"] is not None:      # shade the masked shock zone
            i = c["i_shock"]
            lo = c["x"][max(i - MASK_HALF, 0)]
            hi = c["x"][min(i + MASK_HALF, c["x"].size - 1)]
            for ax in (axdp, axdM):
                ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0, zorder=0)
    for ax in (axdp, axdM):
        ax.axhline(0.0, color="0.55", ls=":", lw=1.0, zorder=1)
    axdp.set_ylabel(r"$(p - p_\mathrm{ana})/p_0$  [%]")
    axdM.set_ylabel(r"$M - M_\mathrm{ana}$")
    axdM.set_xlabel(r"$x$ [m]")
    axdp.legend(handles=[Line2D([], [], color=c["color"], ls="-", lw=1.4,
                                label=c["label"]) for c in _val],
                loc="best")
    fig2.tight_layout()
    save_fig(fig2, f"{PREFIX}_error.png")
else:
    print("No case has an analytic reference — error figure skipped.")

# ── FIGURE 3: gas velocity + analytic ────────────────────────────────────────
if MAKE_VELOCITY_FIG:
    fig3, axu = plt.subplots(figsize=(7.0, 4.3))
    if NOZZLE_TRUE_SCALE:
        add_nozzle(axu, _nz["x"], _nz["r"], true_scale=True)   # proportion-true
    else:
        axr = axu.twinx()                     # nozzle radius on a true mm axis
        axr.set_zorder(axu.get_zorder() - 1)
        axu.patch.set_visible(False)
        _rmm = _nz["r"] * 1e3
        axr.fill_between(_nz["x"], 0.0, _rmm, color="0.45", alpha=0.22, lw=0, zorder=0)
        axr.set_ylim(0.0, float(_rmm.max()) / 0.16)
        axr.set_ylabel("nozzle radius [mm]", color="0.35")
        axr.tick_params(axis="y", colors="0.35")
        axr.spines["right"].set_color("0.35")

    for c in cases:
        axu.plot(c["x"], c["u_g"], color=c["color"], ls="-", lw=1.9, zorder=2)
        if c["prof"] is not None:
            plot_ana(axu, c["x"], c["prof"]["u"])
    axu.set_xlabel(r"$x$ [m]")
    axu.set_ylabel(r"gas velocity $u_g$ [m/s]")
    _leg3 = axu.legend(handles=case_handles, loc="center left")
    axu.add_artist(_leg3)
    axu.legend(handles=[_ana_handle], loc="lower right")
    fig3.tight_layout()
    save_fig(fig3, f"{PREFIX}_velocity.png")

plt.show()
