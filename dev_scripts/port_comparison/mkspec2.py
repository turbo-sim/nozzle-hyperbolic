import json, sys, os
src, out, names, extra = sys.argv[1], sys.argv[2], sys.argv[3].split(","), json.loads(sys.argv[4])
spec = json.load(open(src + "/spec.json"))
spec["tests"] = [t for t in spec["tests"] if t["name"] in names]
spec.update(extra)
os.makedirs(out, exist_ok=True); json.dump(spec, open(out + "/spec.json", "w"), indent=1)
