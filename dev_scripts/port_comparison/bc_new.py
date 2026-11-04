import json, sys, yaml, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import BoundaryConfig
from nozzle_hyperbolic.models.two_fluid.constrained_state import from_primitives
from cases.NICFD2026.scripts.case import reservoir_states

TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
ROOT = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/"
names = ["entropy/s_scale", "u_g - S u_l", "u_mix - u_mix(cell1)", "composition", "H_g - H_g0", "H_l - H_l0"]

for case in sys.argv[1:]:
    cfg = yaml.safe_load(open(ROOT + case + "/settings.yaml"))
    eos = {ph: {k: (float(v) if k in ("h_min", "h_max", "p_min", "p_max") else v)
                for k, v in cfg["eos"][ph].items() if k != "table_dir"} for ph in ("gas", "liquid")}
    eos["gas"].setdefault("N_h", eos["liquid"]["N_h"])
    fg = jxp.FluidBicubic(**eos["gas"], table_dir=TAB)
    fl = jxp.FluidBicubic(**eos["liquid"], table_dir=TAB)
    bc = cfg["bc"]
    family = case.split("/")[0]
    (sg, sl) = reservoir_states(family, float(bc["p0"]), float(bc["T_g0"]), float(bc["T_l0"]), fg, fl)
    slip = float(bc["slip"])
    if "quality" in bc:
        quality = float(bc["quality"])
    else:
        a0 = float(bc["void_fraction"])
        quality = a0 * sg["rho"] * slip / (a0 * sg["rho"] * slip + (1 - a0) * sl["rho"])
    dom = "gas" if quality >= 0.5 else "liquid"
    d = np.genfromtxt(ROOT + case + "/reference.csv", delimiter=",", names=True)
    x = d["x"]; U = np.column_stack([d[k] for k in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])
    faces = 0.5 * (x[:-1] + x[1:]); A = d["A"]; Af = 0.5 * (A[:-1] + A[1:]); dx = x[1] - x[0]
    mesh = Mesh(faces=faces, centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=Af, area_centers=A[1:-1],
                darea_centers=np.gradient(A, x)[1:-1], area_derivative_integrals=np.diff(Af))
    W = np.asarray(from_primitives(jnp.asarray(U), fg, fl))
    bcfg = BoundaryConfig(inlet=bc["inlet"], outlet=bc["outlet"], quality=quality, slip=slip,
                          H_g0=sg["h"], H_l0=sl["h"], s0=(sg if dom == "gas" else sl)["s"],
                          dominant_phase=dom, back_pressure=float(bc["p_out"]),
                          pressure_relaxation=float(bc["p_out_relax"]))
    model = hyp.TwoFluidModel(mesh, fg, fl, initial_state=W, boundaries=bcfg, virtual_mass=0.5)
    R = np.asarray(model.boundary_residual(jnp.asarray(W), 0.0))
    print(f"\n=== {case}  dominant={dom}  quality={quality:.4g}  Nx={len(x)}")
    print("  inlet  rows (dimensionless):", "  ".join(f"{n}={r:+.2e}" for n, r in zip(names, R[0, :6])))
    print("  outlet rows (U_ghost - target)/scale:", np.array2string(R[1, :6], precision=2))
    # physical-unit view of the inlet
    gas = fg.get_state(jxp.HmassP_INPUTS, U[0, 4], U[0, 0]); liq = fl.get_state(jxp.HmassP_INPUTS, U[0, 5], U[0, 0])
    s_dom = float((gas if dom == "gas" else liq)["s"]); s0 = (sg if dom == "gas" else sl)["s"]
    print(f"  inlet ghost: p={U[0,0]:.6e}  s_dom-s0={s_dom - s0:+.3e} J/kgK  "
          f"H_g-H_g0={U[0,4] + 0.5*U[0,1]**2 - sg['h']:+.3e}  H_l-H_l0={U[0,5] + 0.5*U[0,2]**2 - sl['h']:+.3e} J/kg")
    ev = model._wave_data(jnp.asarray(U[-2]))[0]
    print(f"  outlet: p_ghost={U[-1,0]:.6e} p_out={float(bc['p_out']):.6e}  Mach(cell N-2)={abs(float(ev[2]+ev[3]))/float(ev[2]-ev[3]):.3f}")
    from dataclasses import replace
    m2 = hyp.TwoFluidModel(mesh, fg, fl, initial_state=W, boundaries=replace(bcfg, outlet="characteristic_old"), virtual_mass=0.5)
    print("  outlet rows with characteristic_old:", np.array2string(np.asarray(m2.boundary_residual(jnp.asarray(W), 0.0))[1, :6], precision=2))
