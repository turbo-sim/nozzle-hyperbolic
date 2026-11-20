"""Check the K-free kernels in src against the two_fluid_test_v2 prototype.

Run from the repository root: PYTHONPATH=. poetry run python -B development/two-fluid/check_k_free_kernels.py
Face states come from the saved K-inverse solutions in outputs/k_inverse_N100.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import CompatibilityConfig, primitives
from nozzle_hyperbolic.models.two_fluid.flux import fvs_interface_flux, k_free_interface_flux, k_free_path_nonconservative
from development.two_fluid_test_v2.scripts.numerics import face, path_nc

HERE = Path(__file__).resolve().parent
K = 0.5

# 1. Fluids (same tables as run.py)
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward",
           table_dir=str(HERE / "outputs" / "fluid_tables"))
gas = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
liquid = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)

# 2. Face pairs: adjacent cells of the saved solutions, plus slip-amplified copies for the blend/full branches
pairs = []
for name in ("subsonic", "shocked", "adapted"):
    U = np.asarray(primitives(jnp.asarray(np.load(HERE / "outputs" / "k_inverse_N100" / f"{name}.npz")["full_state"])))
    pairs.append(np.stack((U[:-1], U[1:]), axis=1))
    for factor in (1.5, 3.0):
        V = U.copy()
        V[:, 1] = V[:, 2] + factor * (V[:, 1] - V[:, 2]) + 10.0 * (factor - 1.0)
        pairs.append(np.stack((V[:-1], V[1:]), axis=1))
pairs = jnp.asarray(np.concatenate(pairs))
du = np.abs(np.asarray(pairs[:, :, 1] - pairs[:, :, 2]).mean(axis=1))
print(f"{len(pairs)} faces: |du|<5: {np.sum(du < 5)}, 5-20: {np.sum((du >= 5) & (du < 20))}, >=20: {np.sum(du >= 20)}")

# 3. Face flux and path integral: src vs prototype
for smooth in (False, True):
    for floor in (1.0, 0.0):
        cfg = CompatibilityConfig(lowmach_diss=floor)
        new = jax.lax.map(lambda q: k_free_interface_flux(q[0], q[1], gas, liquid, K, cfg.delta_factor, floor,
                                                          cfg.du_blend_lo, cfg.du_blend_hi, smooth=smooth), pairs)
        old = jax.lax.map(lambda q: face(q[0], q[1], gas, liquid, K, cfg, floor, True, smooth), pairs)
        scale = jnp.max(jnp.abs(old), axis=0)
        print(f"flux smooth={smooth}, lowmach floor {floor}: max |src - prototype| / max|F| per row = "
              f"{np.asarray(jnp.max(jnp.abs(new - old), axis=0) / scale)}")
for smooth in (False, True):
    cfg = CompatibilityConfig()
    new = jax.lax.map(lambda q: fvs_interface_flux(q[0], q[1], gas, liquid, K, cfg.delta_factor, 1.0,
                                                   cfg.du_blend_lo, cfg.du_blend_hi, smooth=smooth), pairs)
    old = jax.lax.map(lambda q: face(q[0], q[1], gas, liquid, K, cfg, 1.0, False, smooth), pairs)
    ok = jnp.all(jnp.isfinite(old), axis=1)
    scale = jnp.max(jnp.abs(old[ok]), axis=0)
    print(f"K-inverse flux smooth={smooth} ({int(ok.sum())} finite faces): max |src - prototype| / max|F| per row = "
          f"{np.asarray(jnp.max(jnp.abs(new[ok] - old[ok]), axis=0) / scale)}")
new = jax.lax.map(lambda q: k_free_path_nonconservative(q[0], q[1], gas, liquid, K), pairs)
old = jax.lax.map(lambda q: path_nc(q[0], q[1], gas, liquid, K), pairs)
print(f"path NC: max |src - prototype| = {float(jnp.max(jnp.abs(new - old))):.3e} (max |NC| = {float(jnp.max(jnp.abs(old))):.3e})")

# 4. Path NC structure: mass rows zero, momentum rows cancel
print(f"path NC mass rows max = {float(jnp.max(jnp.abs(new[:, :2]))):.3e}, "
      f"momentum sum max = {float(jnp.max(jnp.abs(new[:, 2] + new[:, 3]))):.3e} vs row max {float(jnp.max(jnp.abs(new[:, 2]))):.3e}")

# 5. Uniform state: flux equals the native flux (= F(U)), path NC zero
cfg = CompatibilityConfig()
q = pairs[len(pairs) // 3]
uniform_new = k_free_interface_flux(q[0], q[0], gas, liquid, K, cfg.delta_factor, 1.0, cfg.du_blend_lo, cfg.du_blend_hi)
uniform_old = fvs_interface_flux(q[0], q[0], gas, liquid, K, cfg.delta_factor, 1.0, cfg.du_blend_lo, cfg.du_blend_hi)
print(f"uniform: max |k_free - native| / |F| = {float(jnp.max(jnp.abs(uniform_new - uniform_old) / jnp.abs(uniform_old))):.3e}, "
      f"path NC = {float(jnp.max(jnp.abs(k_free_path_nonconservative(q[0], q[0], gas, liquid, K)))):.3e}")

# 6. Finite AD through both kernels on every branch
for i in (0, int(np.argmax((du >= 5) & (du < 20))), int(np.argmax(du >= 20))):
    q = pairs[i]
    jf = jax.jacfwd(lambda a, b: k_free_interface_flux(a, b, gas, liquid, K, 1e-6, 1.0, 5.0, 20.0), argnums=(0, 1))(q[0], q[1])
    js = jax.jacfwd(lambda a, b: k_free_interface_flux(a, b, gas, liquid, K, 1e-6, 1.0, 5.0, 20.0, smooth=True), argnums=(0, 1))(q[0], q[1])
    jn = jax.jacfwd(lambda a, b: k_free_path_nonconservative(a, b, gas, liquid, K), argnums=(0, 1))(q[0], q[1])
    finite = all(bool(jnp.all(jnp.isfinite(m))) for m in (*jf, *js, *jn))
    print(f"AD at |du| = {du[i]:.2f} m/s: finite = {finite}")
