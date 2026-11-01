"""Bounded, double-precision JAXProp bicubic evaluations only."""

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import hashlib
import io
import re

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from scipy.optimize import brentq


class BicubicCO2:
    """The table density defines the mechanical EOS; other fields are diagnostics.

    Derivatives below are AD derivatives of the density interpolant, rather
    than independently interpolated derivative properties.
    """

    def __init__(self, settings, output):
        self.settings = settings
        cache = Path(output) / "fluid_tables"
        self.log = io.StringIO()
        # tqdm writes on stderr. Keep progress out of the numerical run log.
        with redirect_stdout(self.log), redirect_stderr(io.StringIO()):
            self.fluid = jxp.FluidBicubic(
                "CO2", backend="HEOS", N_h=100, N_p=100,
                table_dir=str(cache), table_name=settings["name"],
                **{k: settings[k] for k in ("h_min", "h_max", "p_min", "p_max")},
            )
        log = self.log.getvalue()
        print(log, flush=True)
        generated = re.search(r"Successfully evaluated (\d+)/(\d+) points", log)
        log_file = Path(output) / f"{settings['name']}_table.log"
        if generated or not log_file.exists():
            log_file.write_text(log, encoding="utf-8")
        if generated and generated.group(1) != generated.group(2):
            raise RuntimeError("Incomplete JAXProp table: see table generation log")
        self.table_file = cache / f"{settings['name']}.pkl"
        self.table_sha256 = hashlib.sha256(self.table_file.read_bytes()).hexdigest()
        for name in ("density", "temperature", "entropy", "speed_of_sound", "quality_mass"):
            for part in ("value", "coeffs"):
                if not np.isfinite(np.asarray(self.fluid.table[name][part])).all():
                    raise ValueError(f"Nonfinite table {name}/{part}")
        if np.any(np.asarray(self.fluid.table["density"]["value"]) <= 0):
            raise ValueError("Nonpositive table density")

        def raw(h, p):
            return self.fluid.get_state(jxp.HmassP_INPUTS, h, p)

        self._rho = jax.jit(lambda h, p: raw(h, p)["rho"])

        def diagnostic(h, p):
            q = raw(h, p)
            rho_h, rho_p = jax.grad(lambda hh, pp: raw(hh, pp)["rho"], (0, 1))(h, p)
            s_h, s_p = jax.grad(lambda hh, pp: raw(hh, pp)["s"], (0, 1))(h, p)
            # Gibbs-consistent Euler acoustic derivative of this density EOS.
            inverse_a2 = rho_p + rho_h / q["rho"]
            return jnp.array([
                q["rho"], q["T"], q["s"], q["a"], q["Q"], q["is_two_phase"],
                q["compressibility_factor"], inverse_a2, rho_h, rho_p, s_h, s_p,
            ])

        self._diagnostic = jax.jit(diagnostic)
        self._batch = jax.jit(jax.vmap(diagnostic))
        self.keys = ("rho", "T", "s", "a_table", "Q", "phase_flag", "Z_table",
                     "inverse_a2", "rho_h", "rho_p", "s_h", "s_p")

    def check(self, h, p):
        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(p))):
            raise ValueError("Nonfinite property inputs")
        if not (np.all(np.asarray(h) >= self.settings["h_min"]) and
                np.all(np.asarray(h) <= self.settings["h_max"]) and
                np.all(np.asarray(p) >= self.settings["p_min"]) and
                np.all(np.asarray(p) <= self.settings["p_max"])):
            raise ValueError(f"Property evaluation outside bicubic table: h={h}, p={p}")

    def rho(self, h, p):
        self.check(h, p)
        result = float(self._rho(float(h), float(p)))
        if not np.isfinite(result) or result <= 0:
            raise ValueError("Invalid bicubic density")
        return result

    def state(self, h, p):
        h, p = np.broadcast_arrays(np.asarray(h, float), np.asarray(p, float))
        self.check(h, p)
        if h.ndim == 0:
            values = np.asarray(self._diagnostic(float(h), float(p)))
        else:
            values = np.asarray(self._batch(h.ravel(), p.ravel())).reshape(*h.shape, -1)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite bicubic properties")
        q = {k: values[..., i] for i, k in enumerate(self.keys)}
        if np.any(q["inverse_a2"] <= 0) or np.any(q["rho"] <= 0) or np.any(q["T"] <= 0):
            raise ValueError("Loss of hyperbolicity or invalid state")
        q.update(h=h, p=p, a_euler=1 / np.sqrt(q["inverse_a2"]))
        return {k: float(v) if np.ndim(v) == 0 else v for k, v in q.items()}

    def reservoir(self, p, T):
        """Invert table T(h,p) with a scalar bracket, never a direct HEOS call."""
        def residual(h):
            # Avoid derivative/hyperbolicity constraints at unrelated bracket ends.
            return float(self.fluid.get_state(jxp.HmassP_INPUTS, h, p)["T"]) - T
        hs = np.linspace(self.settings["h_min"], self.settings["h_max"], 201)
        fs = [residual(h) for h in hs]
        intervals = [(a, b) for a, b, fa, fb in zip(hs[:-1], hs[1:], fs[:-1], fs[1:])
                     if fa * fb < 0]
        if len(intervals) != 1:
            raise ValueError(f"Reservoir T inversion requires one root, found {len(intervals)}")
        h = brentq(residual, *intervals[0], xtol=1e-7, rtol=1e-13)
        return self.state(h, p)
