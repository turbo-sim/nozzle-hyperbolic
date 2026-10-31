"""
Exact steady quasi-1D nozzle flow for a real-gas EOS, with an optional normal
shock in the divergent section.

Uses the same jaxprop fluid as the solver (e.g. the bicubic LUT), so a
comparison against the FVS solution measures the discretisation error only,
not EOS differences.  For a perfect gas it reduces to the classical
area–Mach and normal-shock relations (nozzle_exact2).

Relations (inviscid, adiabatic):
  - total enthalpy h0 is constant everywhere (also across the shock);
  - isentropic branch at entropy s:  p(h) from s(h, p) = s,
    u = sqrt(2 (h0 - h)),  G(h) = rho(h, p(h)) u  (mass flux per unit area);
  - choked throat: mdot = A_throat * max_h G(h) on the reservoir isentrope s0;
  - section x: G(h) = mdot / A(x), subsonic root (h > h*) or supersonic root
    (h < h*);
  - normal shock: Rankine–Hugoniot with the EOS,
        rho1 u1 = rho2 u2,  p1 + rho1 u1² = p2 + rho2 u2²,  h1 + u1²/2 = h2 + u2²/2;
  - downstream: subsonic branch of the post-shock isentrope s2 with the same mdot.

Useful identities along an isentrope: dp/dh|_s = rho, ds/dp|_h = -1/(rho T).
"""

import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq


def _props(fluid, h, p):
    st = fluid.get_state(jxp.HmassP_INPUTS, np.atleast_1d(h), np.atleast_1d(p))
    return {k: np.asarray(st[k], dtype=float) for k in ("rho", "a", "T", "s")}


class _Isentrope:
    """Real-gas isentrope s = const with total enthalpy h0."""

    def __init__(self, fluid, s, h0, p_bounds, h_lo):
        self.fluid, self.s, self.h0 = fluid, float(s), float(h0)
        self.p_lo, self.p_hi = p_bounds
        self.h_lo = float(h_lo)

    def p_of_h(self, h, p_guess=None, n_bisect=45, n_newton=5):
        """p such that s(h, p) = s.  Without a guess: bisection in log p (s
        decreases with p at fixed h), then Newton with ds/dp|_h = -1/(rho T).
        With a nearby guess (warm start): Newton only."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        if p_guess is None:
            lo = np.full_like(h, self.p_lo)
            hi = np.full_like(h, self.p_hi)
            for _ in range(n_bisect):
                mid = np.sqrt(lo * hi)                 # p spans decades
                s_m = _props(self.fluid, h, mid)["s"]
                lo = np.where(s_m > self.s, mid, lo)
                hi = np.where(s_m <= self.s, mid, hi)
            p = np.sqrt(lo * hi)
        else:
            p = np.asarray(p_guess, dtype=float).copy()
        for _ in range(n_newton):
            q = _props(self.fluid, h, p)
            p = np.clip(p + q["rho"] * q["T"] * (q["s"] - self.s), self.p_lo, self.p_hi)
        return p

    def state(self, h, p_guess=None):
        h = np.atleast_1d(np.asarray(h, dtype=float))
        p = self.p_of_h(h, p_guess)
        q = _props(self.fluid, h, p)
        u = np.sqrt(np.maximum(2.0 * (self.h0 - h), 0.0))
        return dict(h=h, p=p, rho=q["rho"], a=q["a"], T=q["T"], s=q["s"], u=u,
                    G=q["rho"] * u, M=u / q["a"])

    def sonic(self, n_grid=801, n_levels=4):
        """h* = argmax G(h) (the sonic state), by nested grid refinement."""
        a, b = self.h_lo, self.h0 - 1e-6 * abs(self.h0)
        for _ in range(n_levels):
            hg = np.linspace(a, b, n_grid)
            k = int(np.argmax(self.state(hg)["G"]))
            a, b = hg[max(k - 1, 0)], hg[min(k + 1, n_grid - 1)]
        return 0.5 * (a + b)

    def solve_G(self, G_target, branch, h_star, n_iter=60):
        """h on the given branch ('sub' | 'sup') with G(h) = G_target
        (vectorised bisection: G decreases with h on 'sub', increases on 'sup')."""
        G_target = np.atleast_1d(np.asarray(G_target, dtype=float))
        if branch == "sub":
            lo = np.full_like(G_target, h_star)
            hi = np.full_like(G_target, self.h0 - 1e-9 * abs(self.h0))
        else:
            lo = np.full_like(G_target, self.h_lo)
            hi = np.full_like(G_target, h_star)
        p_prev = None                # warm start: h moves little per bisection
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            st = self.state(mid, p_prev)
            p_prev = st["p"]
            above = st["G"] > G_target
            if branch == "sub":      # G decreasing in h: too much flux → raise h
                lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
            else:                    # G increasing in h: too much flux → lower h
                lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
        return self.state(0.5 * (lo + hi), p_prev)


def _rankine_hugoniot(fluid, st1, h0, p_bounds):
    """Post-shock state for the pre-shock state st1 (dict of 1-element arrays)."""
    rho1, u1, p1 = float(st1["rho"][0]), float(st1["u"][0]), float(st1["p"][0])
    G = rho1 * u1

    def f(r):                          # r = rho2 / rho1
        rho2 = r * rho1
        u2 = G / rho2
        p2 = p1 + G * (u1 - u2)
        h2 = h0 - 0.5 * u2**2
        return _props(fluid, h2, np.clip(p2, *p_bounds))["rho"][0] - rho2

    # trivial root at r = 1; the physical (compressive) one lies above it
    rs = np.linspace(1.0 + 1e-4, 8.0, 200)
    fs = np.array([f(r) for r in rs])
    k = np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:]))
    if k.size == 0:
        raise RuntimeError("Rankine-Hugoniot: no compressive root found")
    r = brentq(f, rs[k[0]], rs[k[0] + 1], xtol=1e-14, rtol=1e-14)
    rho2 = r * rho1
    u2 = G / rho2
    p2 = p1 + G * (u1 - u2)
    h2 = h0 - 0.5 * u2**2
    q = _props(fluid, h2, p2)
    return dict(h=h2, p=p2, rho=rho2, u=u2, a=float(q["a"][0]), T=float(q["T"][0]),
                s=float(q["s"][0]), M=u2 / float(q["a"][0]))


def nozzle_exact_real_gas(x, A, fluid, p0, T0, x_shock=None, A_throat=None):
    """
    Exact quasi-1D solution on the stations x with areas A.

    x, A      : (N,) stations and areas (any resolution; A should resolve the
                throat — pass A_throat for the true minimum if it does not).
    fluid     : jaxprop fluid (FluidBicubic LUT or Fluid).
    p0, T0    : reservoir state.
    x_shock   : normal-shock position in the divergent section (None = no shock).

    Returns a dict with 'sub', 'sup' (isentropic subsonic / supersonic branches
    downstream of the throat; upstream both are the subsonic solution) and, with
    a shock, 'shock' (the full shocked profile), 'p_out' (back pressure at x[-1]
    that holds the shock at x_shock), 'pre'/'post' shock states and 'mdot'.
    Each profile holds arrays h, p, rho, a, T, s, u, M.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    st0 = fluid.get_state(jxp.PT_INPUTS, float(p0), float(T0))
    h0, s0 = float(st0["h"]), float(st0["s"])
    p_bounds = (float(getattr(fluid, "p_min", 1e3)) * 1.0001,
                float(getattr(fluid, "p_max", 1e8)) * 0.9999)
    h_lo = float(getattr(fluid, "h_min", h0 - 0.9 * abs(h0))) + 1.0

    i_thr = int(np.argmin(A))
    A_thr = float(A[i_thr] if A_throat is None else A_throat)
    x_thr = x[i_thr]

    iso0 = _Isentrope(fluid, s0, h0, p_bounds, h_lo)
    h_star = iso0.sonic()
    G_star = float(iso0.state(h_star)["G"][0])
    mdot = G_star * A_thr
    G_x = np.minimum(mdot / A, G_star)            # stations at the throat: G*

    sub = iso0.solve_G(G_x, "sub", h_star)
    sup_div = iso0.solve_G(G_x, "sup", h_star)
    div = x >= x_thr
    sup = {k: np.where(div, sup_div[k], sub[k]) for k in sub}

    out = dict(x=x, A=A, h0=h0, s0=s0, p0=float(p0), T0=float(T0),
               rho0=float(st0["rho"]), mdot=mdot, x_throat=x_thr, A_throat=A_thr,
               sub=sub, sup=sup)
    if x_shock is None:
        return out

    if x_shock <= x_thr or x_shock >= x[-1]:
        raise ValueError("x_shock must lie inside the divergent section")
    A_s = float(np.interp(x_shock, x, A))
    pre = iso0.solve_G(np.array([mdot / A_s]), "sup", h_star)
    post = _rankine_hugoniot(fluid, pre, h0, p_bounds)

    iso2 = _Isentrope(fluid, post["s"], h0, p_bounds, h_lo)
    h_star2 = iso2.sonic()
    down = x > x_shock
    post_prof = iso2.solve_G(mdot / A[down], "sub", h_star2)
    shock = {k: sup[k].copy() for k in sup}
    for k in shock:
        shock[k][down] = post_prof[k]
    out.update(shock=shock, x_shock=float(x_shock), A_shock=A_s,
               pre={k: float(v[0]) for k, v in pre.items()}, post=post,
               p_out=float(shock["p"][-1]))
    return out
