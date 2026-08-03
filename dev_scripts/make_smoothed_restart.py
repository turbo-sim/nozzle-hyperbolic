"""
BASIN-ESCAPE experiment: write a SMOOTHED restart IC from a checkerboarded run.

Background: the throat checkerboard is a spurious FINITE-AMPLITUDE discrete
steady state (zero standing residual there), while the linearized operator
around the SMOOTH state is stable (max Re(eig) ~ -1300/s even with eps_void=0).
=> the solver fell into the wrong attractor during the startup transient.

This script takes a run's final solution.csv, applies a few passes of a 1-2-1
filter (kills 2Dx content, barely touches the smooth profile), and writes it in
the nozzlex jump-pickle format that `ic.type: nozzlex` already consumes:

    {'pass1': {x, p, v1, v2, alpha1, alpha2, h1, h2, rho1, rho2}, 'pass2': None, 'pass3': None}

(phase1 = LIQUID, matching the Elliot two-component pickles; ic_from_nozzlex_pickle
auto-detects the ordering from the densities.)

Usage:
    poetry run python make_smoothed_restart.py [run_dir] [n_smooth_passes]
Then point the YAML at it and rerun:
    ic:
      type: nozzlex
      pickle: results/smoothed_restart.pkl

PREDICTION: the march relaxes to the CLEAN smooth steady state and the
checkerboard does NOT return.  If instead it grows back, the smooth state is
NOT in the basin -> the spurious branch is reachable from arbitrarily close
-> back to scheme surgery.
"""
import os, sys, glob, pathlib, pickle
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

_here = pathlib.Path(__file__).parent
arg = sys.argv[1] if len(sys.argv) > 1 else None
n_pass = int(sys.argv[2]) if len(sys.argv) > 2 else 6

if arg:
    run_dir = pathlib.Path(arg)
else:
    run_dir = pathlib.Path(sorted(glob.glob(
        str(_here / "results" / "nozzle_valid_*" / "solution.csv")),
        key=os.path.getmtime)[-1]).parent
D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
print(f"Source: {run_dir.name}   smoothing passes: {n_pass}")

def smooth(f, k=n_pass):
    g = f.astype(float).copy()
    for _ in range(k):
        g[1:-1] = 0.25*g[:-2] + 0.5*g[1:-1] + 0.25*g[2:]
    return g

fields = {k: smooth(D[k]) for k in
          ("p", "u_g", "u_l", "alpha_g", "T_g", "T_l", "h_g", "h_l", "rho_g", "rho_l")}
x = D["x"].astype(float)

# argv[4] == 'slaveghost' (ghost-limit cases): REPLACE the ghost-gas fields with
# their drag/heat-slaved values instead of restarting the transient's garbage
# (observed: smoothed h_g carried a 60 K hump → stiff heat source → NaN at t=0).
#   u_g := u_l,  T_g := T_l  (h = cp·T for the perfect gas — cp, R fitted from
#   the run's own data), alpha := its median (≈ the BC seed).
if len(sys.argv) > 4 and sys.argv[4] == "slaveghost":
    cp_est = float(np.median(D["h_g"] / D["T_g"]))
    R_est  = float(np.median(D["p"] / (D["rho_g"] * D["T_g"])))
    fields["u_g"]     = fields["u_l"].copy()
    fields["h_g"]     = cp_est * fields["T_l"]
    fields["rho_g"]   = fields["p"] / (R_est * fields["T_l"])
    fields["alpha_g"] = np.full_like(x, float(np.median(D["alpha_g"])))
    print(f"SLAVED ghost: u_g=u_l, T_g=T_l (cp={cp_est:.1f}, R={R_est:.1f}), "
          f"alpha={fields['alpha_g'][0]:.2e} const")

# report the 2Dx suppression on the diagnostic fields
def osc_pct(f):
    o = np.abs(f[1:-1] - 0.5*(f[:-2] + f[2:]))
    return o.max() / (f.max() - f.min() + 1e-300) * 100
du_b = D["u_g"] - D["u_l"];  du_a = fields["u_g"] - fields["u_l"]
print(f"2Dx content  du: {osc_pct(du_b):.1f}% -> {osc_pct(du_a):.1f}%   "
      f"alpha: {osc_pct(D['alpha_g']):.1f}% -> {osc_pct(fields['alpha_g']):.1f}%")

# nozzlex jump-pickle layout: phase1 = LIQUID (denser), phase2 = GAS
pass1 = dict(
    x=x,
    p=fields["p"],
    v1=fields["u_l"],     v2=fields["u_g"],
    alpha1=1.0 - fields["alpha_g"], alpha2=fields["alpha_g"],
    h1=fields["h_l"],     h2=fields["h_g"],
    rho1=fields["rho_l"], rho2=fields["rho_g"],
)
# argv[3]: output filename (default smoothed_restart.pkl — careful, the
# two-component yaml points at that name; give case-specific names).
out = _here / "results" / (sys.argv[3] if len(sys.argv) > 3 else "smoothed_restart.pkl")
with open(out, "wb") as f:
    pickle.dump({"pass1": pass1, "pass2": None, "pass3": None}, f)
print(f"Wrote: {out}")
print("Set in the YAML:   ic: {type: nozzlex, pickle: results/smoothed_restart.pkl}")
