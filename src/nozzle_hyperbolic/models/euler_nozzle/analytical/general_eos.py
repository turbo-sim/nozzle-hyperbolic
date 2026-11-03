"""Semi-analytical steady quasi-1D nozzle reference using a supplied EOS.

Uses JAXProp ``fluid.get_state`` for all properties; no constant-gamma or
perfect-gas relation appears in this module. The one-dimensional conservation
laws are standard (e.g. E. F. Toro, *Riemann Solvers and Numerical Methods for
Fluid Dynamics*, 3rd ed., 2009, chapter 1, https://doi.org/10.1007/b79761).
The enthalpy-parametrized reduction and bracketing algorithms below were
derived/implemented in this project, rather than copied from a published
general-EOS nozzle formula. They are numerical root evaluations of continuous
invariants, not a closed-form solution and not a second FVM discretization.

Relations (inviscid, adiabatic, no shaft work):
  - total enthalpy h0 is constant everywhere (also across the shock);
  - isentropic branch at entropy s:  p(h) from s(h, p) = s,
    u = sqrt(2 (h0 - h)),  G(h) = rho(h, p(h)) u  (mass flux per unit area);
  - choked throat: mdot = A_throat * max_h G(h) on the reservoir isentrope s0;
  - section x: G(h) = mdot / A(x), subsonic root (h > h*) or supersonic root
    (h < h*);
  - normal shock: Rankine–Hugoniot with the EOS,
        rho1 u1 = rho2 u2,  p1 + rho1 u1² = p2 + rho2 u2²,  h1 + u1²/2 = h2 + u2²/2;
  - downstream: subsonic branch of the post-shock isentrope s2 with the same mdot.

The Gibbs identity ``dh=T ds+dp/rho`` gives ``dp/dh|s=rho`` and
``ds/dp|h=-1/(rho*T)``. Along an isentrope the mass-flux extremum satisfies
``u=a`` because ``d(rho*u)/dh=rho*(u/a**2-1/u)``. These reductions are stated
here as project derivations from thermodynamic identities and conservation.

Limits: one stable single-phase, classical/convex branch, positive forward
flow, one mass-flux maximum, and a compressive normal shock when requested.
Multi-phase/nonclassical EOS and multiple choking states are not supported.
EOS pressure/enthalpy bounds must include every state. See the public function
for explicit bounds and the inherited fallback/search limits. No plot or FVM
operator is imported. Agreement still includes root/geometry-interpolation
errors; using the same EOS does not isolate discretization error perfectly.
"""

import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq


def _props(fluid, h, p):
    """Evaluate dimensional EOS leaves at enthalpy [J/kg] and pressure [Pa]."""
    st = fluid.get_state(jxp.HmassP_INPUTS, np.atleast_1d(h), np.atleast_1d(p))
    props = {k: np.asarray(st[k], dtype=float) for k in ("rho", "a", "T", "s")}
    if any(np.any(~np.isfinite(value)) for value in props.values()):
        raise ValueError("Non-finite EOS state while evaluating nozzle reference")
    if any(np.any(props[key] <= 0) for key in ("rho", "a", "T")):
        raise ValueError(
            "Nozzle reference requires positive EOS density, sound speed and temperature"
        )
    return props


class _Isentrope:
    """Real-gas isentrope s = const with total enthalpy h0."""

    def __init__(self, fluid, s, h0, p_bounds, h_lo):
        self.fluid, self.s, self.h0 = fluid, float(s), float(h0)
        self.p_lo, self.p_hi = p_bounds
        self.h_lo = float(h_lo)

        # Exclude enthalpies whose isentropic pressure lies below the EOS
        # pressure domain. Derived from ds/dh|p=1/T>0; no EOS substitution.
        def entropy_at_lower_pressure(h):
            return float(_props(fluid, h, self.p_lo)["s"][0]) - self.s

        if entropy_at_lower_pressure(self.h_lo) < 0:
            boundary = brentq(entropy_at_lower_pressure, self.h_lo, self.h0)
            self.h_lo = boundary + max(1e-8, 1e-10 * (self.h0 - boundary))

    def p_of_h(self, h, p_guess=None, n_bisect=45, n_newton=5):
        """p such that s(h, p) = s.  Without a guess: bisection in log p (s
        decreases with p at fixed h), then Newton with ds/dp|_h = -1/(rho T).
        With a nearby guess: try Newton first, then fall back to bracketed
        bisection for any entries that have not met the entropy tolerance."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        if p_guess is None:
            if np.any(_props(self.fluid, h, self.p_lo)["s"] < self.s) or np.any(
                _props(self.fluid, h, self.p_hi)["s"] > self.s
            ):
                raise ValueError(
                    "EOS pressure bounds do not bracket the requested isentrope"
                )
            lo = np.full_like(h, self.p_lo)
            hi = np.full_like(h, self.p_hi)
            for _ in range(n_bisect):
                mid = np.sqrt(lo * hi)  # p spans decades
                s_m = _props(self.fluid, h, mid)["s"]
                lo = np.where(s_m > self.s, mid, lo)
                hi = np.where(s_m <= self.s, mid, hi)
            p = np.sqrt(lo * hi)
        else:
            p = np.asarray(p_guess, dtype=float).copy()
        for _ in range(n_newton):
            q = _props(self.fluid, h, p)
            p = np.clip(p + q["rho"] * q["T"] * (q["s"] - self.s), self.p_lo, self.p_hi)
        failed = np.abs(_props(self.fluid, h, p)["s"] - self.s) > 1e-6
        if np.any(failed):
            if p_guess is None:
                raise RuntimeError(
                    "Isentrope pressure solve did not converge within the EOS bounds"
                )
            # A warm start may have moved too far for five Newton corrections.
            # Recover a bracketed root instead of accepting the clipped iterate.
            p[failed] = self.p_of_h(h[failed], p_guess=None)
        return p

    def state(self, h, p_guess=None):
        """Recover p(h,s), EOS properties and u=sqrt(2*(h0-h)) from invariants."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        p = self.p_of_h(h, p_guess)
        q = _props(self.fluid, h, p)
        u = np.sqrt(np.maximum(2.0 * (self.h0 - h), 0.0))
        return dict(
            h=h,
            p=p,
            rho=q["rho"],
            a=q["a"],
            T=q["T"],
            s=q["s"],
            u=u,
            G=q["rho"] * u,
            M=u / q["a"],
        )

    def sonic(self, n_grid=801, n_levels=4):
        """h* = argmax G(h) (the sonic state), by nested grid refinement."""
        a, b = self.h_lo, self.h0 - 1e-6 * (self.h0 - self.h_lo)
        for level in range(n_levels):
            hg = np.linspace(a, b, n_grid)
            k = int(np.argmax(self.state(hg)["G"]))
            if level == 0 and k in (0, n_grid - 1):
                raise ValueError(
                    "EOS pressure/enthalpy search range does not contain an interior choking state"
                )
            a, b = hg[max(k - 1, 0)], hg[min(k + 1, n_grid - 1)]
        return 0.5 * (a + b)

    def solve_G(self, G_target, branch, h_star, n_iter=60):
        """h on the given branch ('sub' | 'sup') with G(h) = G_target
        (vectorised bisection: G decreases with h on 'sub', increases on 'sup')."""
        G_target = np.atleast_1d(np.asarray(G_target, dtype=float))
        if branch not in ("sub", "sup"):
            raise ValueError("branch must be 'sub' or 'sup'")
        maximum = float(self.state(h_star)["G"][0])
        minimum = (
            float(self.state(self.h_lo)["G"][0])
            if branch == "sup"
            else float(self.state(self.h0 - 1e-9 * (self.h0 - self.h_lo))["G"][0])
        )
        if np.any(G_target < minimum) or np.any(G_target > maximum * (1 + 1e-12)):
            raise ValueError(
                "Mass-flux root is outside the requested branch's EOS search range"
            )
        if branch == "sub":
            lo = np.full_like(G_target, h_star)
            hi = np.full_like(G_target, self.h0 - 1e-9 * (self.h0 - self.h_lo))
        else:
            lo = np.full_like(G_target, self.h_lo)
            hi = np.full_like(G_target, h_star)
        p_prev = None  # warm start: h moves little per bisection
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            st = self.state(mid, p_prev)
            p_prev = st["p"]
            above = st["G"] > G_target
            if branch == "sub":  # G decreasing in h: too much flux → raise h
                lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
            else:  # G increasing in h: too much flux → lower h
                lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
        return self.state(0.5 * (lo + hi), p_prev)


def _rankine_hugoniot(fluid, st1, h0, p_bounds, compression_max=8.0):
    """Find a compressive shock by its density ratio, excluding the trivial root.

    Derived here: for r=rho2/rho1, mass and momentum give u2=u1/r and
    p2=p1+rho1*u1**2*(1-1/r), while energy gives h2=h0-u2**2/2.
    The remaining scalar EOS equation is rho(h2,p2)-r*rho1=0.
    A finite scan over 1.0001<=r<=compression_max brackets the first root;
    it cannot establish uniqueness or cover arbitrary EOS shocks.
    """
    rho1, u1, p1 = float(st1["rho"][0]), float(st1["u"][0]), float(st1["p"][0])
    G = rho1 * u1

    def f(r):  # r = rho2 / rho1
        rho2 = r * rho1
        u2 = G / rho2
        p2 = p1 + G * (u1 - u2)
        h2 = h0 - 0.5 * u2**2
        if not p_bounds[0] <= p2 <= p_bounds[1]:
            return np.nan
        return _props(fluid, h2, p2)["rho"][0] - rho2

    # trivial root at r = 1; the physical (compressive) one lies above it
    rs = np.linspace(1.0 + 1e-4, compression_max, 200)
    fs = np.array([f(r) for r in rs])
    k = np.flatnonzero(
        np.isfinite(fs[:-1])
        & np.isfinite(fs[1:])
        & (np.sign(fs[:-1]) != np.sign(fs[1:]))
    )
    if k.size == 0:
        raise RuntimeError("Rankine-Hugoniot: no compressive root found")
    r = brentq(f, rs[k[0]], rs[k[0] + 1], xtol=1e-14, rtol=1e-14)
    rho2 = r * rho1
    u2 = G / rho2
    p2 = p1 + G * (u1 - u2)
    h2 = h0 - 0.5 * u2**2
    q = _props(fluid, h2, p2)
    if float(q["s"][0]) <= float(st1["s"][0]):
        raise RuntimeError("Selected Rankine-Hugoniot root is not entropy increasing")
    return dict(
        h=h2,
        p=p2,
        rho=rho2,
        u=u2,
        a=float(q["a"][0]),
        T=float(q["T"][0]),
        s=float(q["s"][0]),
        M=u2 / float(q["a"][0]),
    )


def nozzle_exact_real_gas(
    x,
    A,
    fluid,
    p0,
    T0,
    x_shock=None,
    A_throat=None,
    *,
    p_bounds=None,
    h_min=None,
    compression_max=8.0
):
    """
    Evaluate a choked quasi-1D continuous reference on sampled geometry.

    Parameters
    ----------
    x, A : array_like, shape (N,)
        Stations and areas. Resolve the throat or supply ``A_throat`` for the
        true minimum area when the sampled geometry misses it.
    fluid : object
        JAXProp fluid, including FluidBicubic tables or host fluids.
    p0, T0 : float
        Reservoir pressure in Pa and temperature in K.
    x_shock : float, optional
        Normal-shock position in the divergent section; None selects no shock.
    A_throat : float, optional
        Physical minimum area, overriding the sampled minimum.
    p_bounds : tuple of float, optional
        Minimum/maximum search pressure in Pa. Defaults to fluid bounds inset
        by 0.01%, or (1000.1, 99990000) Pa without metadata. These are search
        limits, not EOS physics.
    h_min : float, optional
        Lower search enthalpy in J/kg. Defaults to ``fluid.h_min + 1`` or,
        without metadata, the historical ``h0 - 0.9*abs(h0) + 1``. That fallback
        depends on enthalpy datum; specify a bound for shifted/negative datums
        or a different EOS validity range.
    compression_max : float, optional
        Upper density ratio searched for a shock; default 8.

    Returns a dict with 'sub', 'sup' (isentropic subsonic / supersonic branches
    downstream of the throat; upstream both are the subsonic solution) and, with
    a shock, 'shock' (the full shocked profile), 'p_out' (back pressure at x[-1]
    that holds the shock at x_shock), 'pre'/'post' shock states and 'mdot'.
    Each profile holds arrays h, p, rho, a, T, s, u, M.

    ``sub`` is the sonic-throat/subsonic-exit limiting branch, not an arbitrary
    unchoked operating point. ``sup`` changes to the supersonic branch after
    the sampled minimum area. Pass A_throat when the grid misses the true area
    minimum; the throat coordinate still comes from argmin(A). The shock area
    is linearly interpolated from (x,A), so resolve the geometry adequately.
    The sonic state uses four levels of 801-point mass-flux maximization;
    profile roots use 60 bisections with entropy completion. Pressure roots or
    shocks outside the search bounds raise instead of silently clipping states.
    This host-side reference is not differentiable/JIT-compatible and is not a
    guarantee that an arbitrary supplied EOS has the assumed branch structure.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 3
        or A.shape != x.shape
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(A))
        or np.any(np.diff(x) <= 0)
        or np.any(A <= 0)
    ):
        raise ValueError(
            "Need matching finite station/area arrays, increasing x and positive area"
        )
    st0 = fluid.get_state(jxp.PT_INPUTS, float(p0), float(T0))
    h0, s0 = float(st0["h"]), float(st0["s"])
    if p_bounds is None:
        p_bounds = (
            float(getattr(fluid, "p_min", 1e3)) * 1.0001,
            float(getattr(fluid, "p_max", 1e8)) * 0.9999,
        )
    h_lo = (
        float(getattr(fluid, "h_min", h0 - 0.9 * abs(h0))) + 1.0
        if h_min is None
        else float(h_min)
    )
    if (
        not 0 < p_bounds[0] < p_bounds[1]
        or h_lo >= h0
        or not np.all(np.isfinite([*p_bounds, h_lo]))
        or compression_max <= 1.0001
    ):
        raise ValueError(
            "Invalid pressure, enthalpy or shock-compression search bounds"
        )

    i_thr = int(np.argmin(A))
    A_thr = float(A[i_thr] if A_throat is None else A_throat)
    if not 0 < A_thr <= np.min(A) * (1 + 1e-12):
        raise ValueError(
            "A_throat must be positive and no larger than the sampled minimum area"
        )
    x_thr = x[i_thr]

    iso0 = _Isentrope(fluid, s0, h0, p_bounds, h_lo)
    h_star = iso0.sonic()
    G_star = float(iso0.state(h_star)["G"][0])
    mdot = G_star * A_thr
    G_x = np.minimum(mdot / A, G_star)  # stations at the throat: G*

    sub = iso0.solve_G(G_x, "sub", h_star)
    sup_div = iso0.solve_G(G_x, "sup", h_star)
    div = x >= x_thr
    sup = {k: np.where(div, sup_div[k], sub[k]) for k in sub}

    out = dict(
        x=x,
        A=A,
        h0=h0,
        s0=s0,
        p0=float(p0),
        T0=float(T0),
        rho0=float(st0["rho"]),
        mdot=mdot,
        x_throat=x_thr,
        A_throat=A_thr,
        sub=sub,
        sup=sup,
    )
    if x_shock is None:
        return out

    if x_shock <= x_thr or x_shock >= x[-1]:
        raise ValueError("x_shock must lie inside the divergent section")
    A_s = float(np.interp(x_shock, x, A))
    pre = iso0.solve_G(np.array([mdot / A_s]), "sup", h_star)
    post = _rankine_hugoniot(fluid, pre, h0, p_bounds, compression_max)

    iso2 = _Isentrope(fluid, post["s"], h0, p_bounds, h_lo)
    h_star2 = iso2.sonic()
    down = x > x_shock
    post_prof = iso2.solve_G(mdot / A[down], "sub", h_star2)
    shock = {k: sup[k].copy() for k in sup}
    for k in shock:
        shock[k][down] = post_prof[k]
    out.update(
        shock=shock,
        x_shock=float(x_shock),
        A_shock=A_s,
        pre={k: float(v[0]) for k, v in pre.items()},
        post=post,
        p_out=float(shock["p"][-1]),
    )
    return out
