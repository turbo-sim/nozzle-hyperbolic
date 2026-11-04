import numpy as np, sys
d = sys.argv[1]
o, n = np.load(d + "/R_old.npz"), np.load(d + "/R_new.npz")
for k in ("full_vanalbada_wb", "full_const_wb_filter"):
    sc = np.max(np.abs(n[k + "__conservative"]), axis=0)
    e = np.max(np.abs(o[k] - n[k]) / sc, axis=1)
    N = len(e)
    print(k, "first 4 cells:", np.array2string(e[:4], precision=1), " last 4:", np.array2string(e[-4:], precision=1))
    print("   interior (4..N-5): max", f"{e[4:-4].max():.2e}", " median", f"{np.median(e[4:-4]):.2e}", " #cells>1e-6:", int(np.sum(e[4:-4] > 1e-6)), "/", N - 8)
    if "filter" in k:
        st = n[k + "__stabilization"]
        print("   new filter contribution / conservative scale, max:", np.array2string(np.max(np.abs(st), axis=0) / sc, precision=2))
