"""Continuous quasi-1D Euler reference: pressure-coordinate ODE and scalar roots.

No finite-volume grid, flux, time integrator or production solver is used.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class Nozzle:
    x_in: float = -0.03201
    x_out: float = 0.06636
    throat_radius: float = 0.0015
    inlet_ratio: float = 17.1
    exit_ratio: float = 1.69

    @property
    def At(self):
        return np.pi * self.throat_radius**2

    def area(self, x):
        x = np.asarray(x)
        return self.At * np.where(x < 0, 1 + (self.inlet_ratio - 1) * (x / self.x_in)**2,
                                 1 + (self.exit_ratio - 1) * (x / self.x_out)**2)

    def derivative(self, x):
        x = np.asarray(x)
        return self.At * np.where(x < 0, 2 * (self.inlet_ratio - 1) * x / self.x_in**2,
                                 2 * (self.exit_ratio - 1) * x / self.x_out**2)


class SmoothPath:
    """dh/d(log p)=p/rho, so dp+rho*u*du=0 and h+u²/2=h0.

    A numerical primitive of the steady equations; dense output allows root
    evaluation without interpolating a sampled nozzle field.
    """

    def __init__(self, eos, h0, anchor_p, anchor_h, end_p, *, rtol=2e-11, max_step=0.008):
        self.eos, self.h0 = eos, float(h0)
        self.anchor_p, self.anchor_h = float(anchor_p), float(anchor_h)
        self.p_low, self.p_high = sorted((float(anchor_p), float(end_p)))
        self.rtol, self.max_step = rtol, max_step
        self.solution = solve_ivp(
            lambda logp, h: [np.exp(logp) / eos.rho(h[0], np.exp(logp))],
            (np.log(anchor_p), np.log(end_p)), [anchor_h], method="DOP853",
            dense_output=True, rtol=rtol, atol=2e-6, max_step=max_step,
        )
        if not self.solution.success:
            raise RuntimeError(self.solution.message)

    def enthalpy(self, p):
        p = np.asarray(p)
        # Small relative allowance only for floating point exponentiation.
        if np.any(p < self.p_low * (1 - 1e-13)) or np.any(p > self.p_high * (1 + 1e-13)):
            raise ValueError("Pressure outside integrated smooth path")
        return self.solution.sol(np.log(p))[0]

    def flux(self, p):
        h = float(self.enthalpy(p))
        energy = 2 * (self.h0 - h)
        if energy < -1e-6:
            raise ValueError("Negative kinetic energy")
        return self.eos.rho(h, p) * np.sqrt(max(energy, 0.0))

    def state(self, p):
        q = self.eos.state(self.enthalpy(p), p)
        energy = 2 * (self.h0 - q["h"])
        if np.any(energy < -1e-6):
            raise ValueError("Negative kinetic energy")
        q["u"] = np.sqrt(np.maximum(energy, 0.0))
        q["G"] = q["rho"] * q["u"]
        q["M_euler"] = q["u"] / q["a_euler"]
        q["M_table"] = q["u"] / q["a_table"]
        return q

    def choke(self):
        """Locate all acoustic extrema, require one maximum and monotone limbs."""
        ps = np.geomspace(self.p_low, self.p_high * (1 - 1e-10), 1201)
        q = self.state(ps)
        f = q["u"]**2 * q["inverse_a2"] - 1
        brackets = [(a, b) for a, b, fa, fb in zip(ps[:-1], ps[1:], f[:-1], f[1:])
                    if fa * fb < 0]
        if len(brackets) != 1:
            raise RuntimeError(f"Expected one choking maximum; found {len(brackets)} extrema")
        self.p_star = brentq(lambda p: self.state(p)["M_euler"] - 1,
                             *brackets[0], xtol=1e-5, rtol=2e-13)
        self.G_star = self.flux(self.p_star)
        if not self.G_star > max(self.flux(self.p_low), self.flux(self.p_high)):
            raise RuntimeError("Choking extremum is not an interior global maximum")
        # Sample monotonicity independently of the acoustic root sign check.
        for a, b, sign in ((self.p_low, self.p_star, 1), (self.p_star, self.p_high, -1)):
            gs = np.array([self.flux(p) for p in np.geomspace(a, b, 301)])
            if np.any(sign * np.diff(gs) < -1e-8 * self.G_star):
                raise RuntimeError("Nonmonotone mass-flux limb; simple nozzle branch unsupported")
        return self.state(self.p_star)

    def pressure_for_flux(self, target, branch):
        if branch == "sub":
            limits = (self.p_star, self.p_high)
        elif branch == "sup":
            limits = (self.p_low, self.p_star)
        else:
            raise ValueError("branch must be sub or sup")
        if abs(target / self.G_star - 1) < 2e-13:
            return self.p_star
        return brentq(lambda p: self.flux(p) - target, *limits, xtol=1e-5, rtol=2e-13)


def normal_shock(eos, pre, h0, compression_max=30.0):
    """Scalar RH solve in compression r=rho2/rho1; excludes r=1 explicitly."""
    rho1, u1, p1 = pre["rho"], pre["u"], pre["p"]
    if pre["M_euler"] <= 1:
        raise ValueError("Shock upstream state is not supersonic")

    def candidate(r):
        u2 = u1 / r
        p2 = p1 + rho1 * u1**2 * (1 - 1 / r)
        h2 = h0 - 0.5 * u2**2
        return h2, p2, u2

    def residual(r):
        h2, p2, _ = candidate(r)
        eos.check(h2, p2)
        return eos.rho(h2, p2) / rho1 - r

    rs = 1 + np.geomspace(1e-5, compression_max - 1, 401)
    brackets = []
    previous = None
    for r in rs:
        h2, p2, _ = candidate(r)
        if not (eos.settings["h_min"] <= h2 <= eos.settings["h_max"] and
                eos.settings["p_min"] <= p2 <= eos.settings["p_max"]):
            previous = None
            continue
        f = residual(r)
        if previous is not None and f * previous[1] < 0:
            brackets.append((previous[0], r))
        previous = (r, f)
    admissible = []
    for bracket in brackets:
        r = brentq(residual, *bracket, xtol=1e-13, rtol=2e-13)
        h2, p2, u2 = candidate(r)
        q = eos.state(h2, p2)
        q.update(u=u2, G=q["rho"] * u2, M_euler=u2 / q["a_euler"],
                 M_table=u2 / q["a_table"], compression=r)
        if q["p"] > p1 and q["rho"] > rho1 and q["s"] > pre["s"] and q["M_euler"] < 1:
            admissible.append(q)
    if len(admissible) != 1:
        raise RuntimeError(f"Expected one admissible compressive shock, found {len(admissible)}")
    return admissible[0]


def downstream_exit(eos, post, h0, mdot, exit_area, **ode_options):
    """Integrate the decelerating downstream limb until its exit-flux root."""
    def zero_energy(logp, h):
        return h[0] - h0
    zero_energy.terminal = True
    zero_energy.direction = 1
    # A physical stagnation point bounds the subsonic pressure root above p2.
    end_p = eos.settings["p_max"] * (1 - 1e-12)
    solution = solve_ivp(
        lambda logp, h: [np.exp(logp) / eos.rho(h[0], np.exp(logp))],
        (np.log(post["p"]), np.log(end_p)), [post["h"]], events=zero_energy,
        method="DOP853", dense_output=True,
        rtol=ode_options.get("rtol", 2e-11), atol=2e-6,
        max_step=ode_options.get("max_step", 0.008),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    path = SmoothPath.__new__(SmoothPath)
    path.eos, path.h0 = eos, h0
    path.anchor_p, path.anchor_h = post["p"], post["h"]
    path.p_low, path.p_high = post["p"], float(np.exp(solution.t[-1]))
    path.solution = solution
    target = mdot / exit_area
    p_exit = brentq(lambda p: path.flux(p) - target, path.p_low, path.p_high,
                    xtol=1e-5, rtol=2e-13)
    return path, path.state(p_exit)


def solve_shock_location(eos, upstream, nozzle, mdot, backpressure,
                         bracket_fractions=(0.15, 0.95), **ode_options):
    """Outer backpressure iteration, each evaluation containing a fresh RH root."""
    history = []
    def evaluate(x):
        pre = upstream.state(upstream.pressure_for_flux(mdot / nozzle.area(x), "sup"))
        post = normal_shock(eos, pre, upstream.h0)
        downstream, exit_state = downstream_exit(eos, post, upstream.h0, mdot,
                                                 nozzle.area(nozzle.x_out), **ode_options)
        residual = exit_state["p"] - backpressure
        history.append(dict(x_shock=float(x), p_exit=exit_state["p"],
                            backpressure_residual=residual, compression=post["compression"]))
        return residual, pre, post, downstream, exit_state
    limits = tuple(f * nozzle.x_out for f in bracket_fractions)
    x = brentq(lambda x: evaluate(x)[0], *limits, xtol=2e-12, rtol=2e-12)
    _, pre, post, downstream, exit_state = evaluate(x)
    return dict(x=x, pre=pre, post=post, downstream=downstream,
                exit=exit_state, history=history, bracket=limits)


def sample_profile(upstream, nozzle, mdot, *, shock=None, stations=501):
    """Include the throat and both coincident shock traces explicitly in CSV."""
    x = np.unique(np.r_[np.linspace(nozzle.x_in, nozzle.x_out, stations), 0.0])
    if shock is not None:
        x = np.sort(np.r_[x[x != shock["x"]], shock["x"], shock["x"]])
    ps, sides = [], []
    shock_seen = False
    for xi in x:
        target = mdot / nozzle.area(xi)
        if shock is not None and xi == shock["x"]:
            p = shock["post" if shock_seen else "pre"]["p"]
            side = "shock_post" if shock_seen else "shock_pre"
            shock_seen = True
        elif shock is not None and xi > shock["x"]:
            p = brentq(lambda p: shock["downstream"].flux(p) - target,
                        shock["post"]["p"], shock["downstream"].p_high,
                        xtol=1e-5, rtol=2e-13)
            side = "downstream"
        else:
            p = upstream.pressure_for_flux(target, "sub" if xi < 0 else "sup")
            side = "upstream"
        ps.append(p)
        sides.append(side)
    # Evaluate paths by side to preserve the distinct postshock h(p) primitive.
    fields = {}
    for downstream_side in (False, True):
        mask = np.array([s in ("shock_post", "downstream") for s in sides]) == downstream_side
        if not mask.any():
            continue
        path = shock["downstream"] if downstream_side else upstream
        q = path.state(np.asarray(ps)[mask])
        for key, values in q.items():
            fields.setdefault(key, np.empty(len(x)))[mask] = values
    fields.update(x=x, A=nozzle.area(x), side=np.asarray(sides))
    return fields
