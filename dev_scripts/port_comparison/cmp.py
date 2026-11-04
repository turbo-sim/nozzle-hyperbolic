import numpy as np, sys
d = sys.argv[1]
o, n = np.load(d + "/R_old.npz"), np.load(d + "/R_new.npz")
rows = ["mass_g", "mass_l", "mom_g", "mom_l", "E_g", "E_l"]
for k in o.files:
    Ro, Rn = o[k], n[k]
    # scale: per-equation magnitude of the individual terms (conservative flux difference)
    sc = np.max(np.abs(n[k + "__conservative"]), axis=0)
    e = np.abs(Ro - Rn) / sc
    i = np.argmax(e, axis=0)
    print(f"{k:24s} max rel err per eq: " + "  ".join(f"{r}={e[i[j], j]:.1e}@{i[j]}" for j, r in enumerate(rows)))
