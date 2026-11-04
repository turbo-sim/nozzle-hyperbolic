import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jaxprop as jxp
from nozzle_hyperbolic.models.two_fluid.matrices import phase_props, common_quantities
from nozzle_hyperbolic.models.two_fluid.sources import two_fluid_closure
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
for d in sys.argv[1:]:
    spec = json.load(open(d + "/spec.json")); s = np.load(d + "/state.npz"); r = np.load(d + "/pdalpha.npz")
    fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB); fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)
    cl = [t for t in spec["tests"] if t["closure"]][0]["closure"]; mt = cl["mass_transfer"]
    def one(u):
        p, ug, ul, ag, hg, hl = u; al = 1 - ag
        eg, el = phase_props(fg, p, hg), phase_props(fl, p, hl); cq = common_quantities(u, eg, el, 0.5)
        if mt:
            sg, sl = fg.get_state_saturation(p), fl.get_state_saturation(p); Ts, hgs, hls = sg["temperature"], sg["enthalpy"], sl["enthalpy"]
        else:
            Ts = hgs = hls = 0.0
        c = two_fluid_closure(u, eg, el, fg, fl, Ts, hgs, hls, **cl)
        sMg, sMl, Fg = c.sigma_Mg, -c.sigma_Mg, c.Fv_g
        Tg = sMg * ((c.h_ex_g - hg) + 0.5 * (c.u_ex - ug)**2) + Fg * (c.u_int - ug) + c.sigma_Qg
        Tl = sMl * ((c.h_ex_l - hl) + 0.5 * (c.u_ex - ul)**2) - Fg * (c.u_int - ul) + c.sigma_Ql
        Mg = sMg - eg.rho_h / eg.rho * Tg; Ml = sMl - el.rho_h / el.rho * Tl
        k1 = al * cq.a0sq / (cq.rho_s * el.a**2); k2 = ag * cq.a0sq / (cq.rho_s * eg.a**2)
        Da = k1 * Mg - k2 * Ml
        return p * Da, p * k1 * Mg, -p * k2 * Ml, p * k1 * sMg, -p * k2 * sMl
    pDa, pg, pl, pgm, plm = map(np.asarray, jax.lax.map(one, jnp.asarray(s["U_h"][1:-1])))
    D = r["D"]; E = np.maximum(np.abs(D[:, 4]), np.abs(D[:, 5]))
    x = s["x"][1:-1]
    rest = np.maximum(np.abs(D[:, 4] + pDa), np.abs(D[:, 5] - pDa))   # energy rows WITHOUT the p*Dalpha term (pre-faafff7)
    i = np.argmax(np.abs(pDa) / np.maximum(rest, 1e-300 + 1e-12 * rest.max()))
    print(f"\n== {d.split('/')[-1]}  (mass transfer {mt})")
    print(f"  max |p*Dalpha| = {np.abs(pDa).max():.3e} W/m3   max |other energy source| = {rest.max():.3e} W/m3   ratio of maxima = {np.abs(pDa).max()/rest.max():.3f}")
    j = np.argmax(np.abs(pDa))
    print(f"  at x={x[j]:.4f} m (alpha={s['U_h'][1+j,3]:.3e}): p*Dalpha={pDa[j]:+.3e}, from gas-side M_g {pg[j]:+.3e} [mass part {pgm[j]:+.3e}], from liquid-side M_l {pl[j]:+.3e} [mass part {plm[j]:+.3e}]; other E_g src {D[j,4]+pDa[j]:+.3e}, E_l src {D[j,5]-pDa[j]:+.3e}")
    w = rest > 1e-3 * rest.max()
    rel = np.abs(pDa[w]) / rest[w]
    print(f"  where sources are active ({w.sum()} cells): |p*Dalpha|/|other energy source| median {np.median(rel):.3f}, max {rel.max():.3f}")
