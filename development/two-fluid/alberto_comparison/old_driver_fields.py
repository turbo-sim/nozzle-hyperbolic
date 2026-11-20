"""Run alberto_branch's own two-fluid driver with run-time back pressure and correction fields.

alberto_branch is not modified. Its dev_scripts/main_two_fluid.py is copied to outputs/_old_driver/ with
three textual patches, then run unchanged otherwise (its own time control, boundaries and stopping):
  1. _sources_fn takes an optional per-cell row (beta_ai, beta_cd, beta_htc_g, beta_htc_l) and passes it to
     two_fluid_closure as ai_scale, beta_cd, beta_htc_g, beta_htc_l;
  2. _step passes p_out and the fields to step(W, dt, p_out, ai_scale) at run time, as its API allows;
  3. after the time loop the final state is saved to OLD_OUT_NPZ and the run stops (no post-processing).
Run (alberto_branch first on the path):
    PYTHONPATH="<alberto_branch>;<alberto_branch>/nozzle_hyperbolic/fvm_solver;<alberto_branch>/dev_scripts" \
    SETTINGS_YAML=<settings.yaml> OLD_OUT_NPZ=<out.npz> [P_OUT_RUNTIME=<Pa>] [FIELDS_NPY=<fields.npy>] \
    python -B old_driver_fields.py
FIELDS_NPY holds an (Nx, 4) array including the two ghost rows.
"""

import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path("C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/main_two_fluid.py")
TARGET = HERE / "outputs" / "_old_driver" / "main_two_fluid_patched.py"

# 1. Patch a copy of the driver
text = SOURCE.read_text(encoding="utf-8")
patches = [
    ("def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):",
     "def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l, _f=None):"),
    ("            smooth_slip_u0 = cfg.SMOOTH_SLIP_U0,\n        )\n        if cfg.MASS_TRANSFER:\n"
     "            D = D + SRC_SCALE",
     "            smooth_slip_u0 = cfg.SMOOTH_SLIP_U0,\n"
     "            **({} if _f is None else dict(ai_scale=_f[0], beta_cd=_f[1], beta_htc_g=_f[2], beta_htc_l=_f[3])),\n"
     "        )\n        if cfg.MASS_TRANSFER:\n            D = D + SRC_SCALE"),
    ("def _step(W, dt_):\n    if _ramp_n > 0:\n        return step(W, dt_, jnp.float64(_p_out_now))\n    return step(W, dt_)\n",
     "_P_RUN = float(os.environ.get('P_OUT_RUNTIME', cfg.P_OUT))\n"
     "_FIELDS = (jnp.asarray(np.load(os.environ['FIELDS_NPY'])) if 'FIELDS_NPY' in os.environ else None)\n"
     "def _step(W, dt_):\n    return step(W, dt_, jnp.float64(_P_RUN), _FIELDS)\n"),
    ('print(f"\\nStopped: {_stop_reason}")',
     'print(f"\\nStopped: {_stop_reason}")\n'
     "np.savez(os.environ['OLD_OUT_NPZ'], x=x, W=np.asarray(W_n), U=np.asarray(U_h_from_W_batch(W_n)),\n"
     "         ssr=ssr, steps=n, reason=_stop_reason, p_out=_P_RUN)\n"
     "print(f'  steps={n}  SSR={ssr:.3e}  saved {os.environ[\"OLD_OUT_NPZ\"]}')\n"
     "raise SystemExit(0)"),
]
for old, new in patches:
    if text.count(old) != 1:
        raise RuntimeError(f"Patch anchor not found exactly once: {old[:60]!r}")
    text = text.replace(old, new)
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(text, encoding="utf-8")

# 2. Run the patched driver as a script
runpy.run_path(str(TARGET), run_name="__main__")
