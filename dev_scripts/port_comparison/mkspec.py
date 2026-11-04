import json, sys, yaml, os
case, out = sys.argv[1], sys.argv[2]
root = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/" + case
cfg = yaml.safe_load(open(root + "/settings.yaml"))
eos = {ph: {k: v for k, v in cfg["eos"][ph].items() if k != "table_dir"} for ph in ("gas", "liquid")}
for ph in eos:
    for k in ("h_min", "h_max", "p_min", "p_max"): eos[ph][k] = float(eos[ph][k])
mt = bool(cfg["sources"]["mass_transfer"])
cl = dict(Nb=float(cfg["sources"]["bubble_number"]), Nd=float(cfg["sources"]["droplet_number"]),
          mass_transfer=mt, area_law="original")
tests = []
for name, recon, nc, ev, closure, area in [
    ("area_only_const_cell", "constant", "cell", 0.0, None, True),
    ("full_const_cell",      "constant", "cell", 0.0, cl, True),
    ("full_const_wb",        "constant", "wb",   0.0, cl, True),
    ("full_const_staedtke",  "constant", "staedtke", 0.0, cl, True),
    ("full_vanleer_wb",      "muscl_van_leer", "wb", 0.0, cl, True),
    ("full_vanalbada_wb",    "muscl_van_albada", "wb", 0.0, cl, True),
    ("full_const_wb_filter", "constant", "wb", 0.05, cl, True),
]:
    tests.append(dict(name=name, recon=recon, nc_form=nc, eps_void=ev, closure=closure, area_source=area))
os.makedirs(out, exist_ok=True)
json.dump(dict(reference=root + "/reference.csv", geometry=cfg["geometry"]["function"], eos=eos, tests=tests),
          open(out + "/spec.json", "w"), indent=1)
