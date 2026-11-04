import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jaxprop as jxp
from nozzle_hyperbolic.models.two_fluid.matrices import phase_props, common_quantities
from nozzle_hyperbolic.models.two_fluid.sources import (two_fluid_closure, D_VE_algebric,
    D_VE_algebric_two_component, D_VE_conservative)
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
for d in sys.argv[1:]:
    spec = json.load(open(d + "/spec.json")); s = np.load(d + "/state.npz")
    fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB); fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)
    cl = [t for t in spec["tests"] if t["closure"]][0]["closure"]; mt = cl["mass_transfer"]
    def one(u):
        eg, el = phase_props(fg, u[0], u[4]), phase_props(fl, u[0], u[5]); cq = common_quantities(u, eg, el, 0.5)
        if mt:
            sg, sl = fg.get_state_saturation(u[0]), fl.get_state_saturation(u[0]); Ts, hgs, hls = sg["temperature"], sg["enthalpy"], sl["enthalpy"]
        else:
            Ts = hgs = hls = 0.0
        c = two_fluid_closure(u, eg, el, fg, fl, Ts, hgs, hls, **cl)
        asm = D_VE_algebric if mt else D_VE_algebric_two_component
        D = asm(u, cq, c, eg, el, 0.5)
        Dz = asm(u, cq, c._replace(dlnA_dx=0.0), eg, el, 0.5)
        Dc = D_VE_conservative(u, eg, el, cq, c._replace(dlnA_dx=0.0, F_g=0.0, F_l=0.0, Q_g=0.0, Q_l=0.0), 0.5)
        # p*D_alpha = half the antisymmetric part of the energy rows vs the pre-faafff7 assembly is not
        # directly available: recompute it as D_l-energy row minus the same row with p -> 0 in the work term
        return D, Dc, c.sigma_Qg, c.sigma_Ql, c.sigma_Mg
    U = jnp.asarray(s["U_h"][1:-1])
    D, Dc, Qg, Ql, Mg = jax.lax.map(one, U)
    D, Dc = np.asarray(D), np.asarray(Dc)
    np.savez(d + "/pdalpha.npz", D=D, Dc=Dc, Qg=np.asarray(Qg), Ql=np.asarray(Ql), Mg=np.asarray(Mg), U=np.asarray(U))
    err = np.max(np.abs(D - Dc), axis=0) / np.max(np.abs(Dc), axis=0)
    print(d.split("/")[-1], "mass_transfer", mt, "| algebraic vs conservative route, max rel diff per row:", np.array2string(err, precision=1))
