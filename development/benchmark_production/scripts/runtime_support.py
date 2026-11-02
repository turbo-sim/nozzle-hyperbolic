"""Archived assessment and IPC helper bodies, relocated without numerical changes.

These functions were previously shared with the retired benchmarking client.
The original measurement-time files remain in source_and_cases.zip.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def save(path, value):
    from time import monotonic, sleep

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    deadline = monotonic() + 5.0
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers can temporarily deny DELETE sharing. Retain the
            # complete temporary JSON and retry the atomic replacement.
            if os.name != "nt" or monotonic() >= deadline:
                raise
            sleep(0.05)

def identity(value):
    return sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def source_snapshot():
    directory = ROOT / "src/nozzle_hyperbolic"
    files = {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
             for path in sorted(directory.rglob("*"))
             if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts}
    return dict(sha256=identity(files), files=files)

def error_norms(numeric, exact, weights, scale):
    numeric,exact=np.asarray(numeric,float),np.asarray(exact,float)
    difference=numeric-exact;weights=np.asarray(weights,float);weights=weights/weights.sum()
    l1=float(np.sum(weights*np.abs(difference)));l2=float(np.sqrt(np.sum(weights*difference*difference)))
    linf=float(np.max(np.abs(difference)));denominator=float(np.sum(weights*np.abs(exact)))
    active=denominator>max(float(scale),1e-300)*1e-14
    return dict(l1=l1,l2=l2,linf=linf,normalized_l1=l1/scale,normalized_l2=l2/scale,normalized_linf=linf/scale,
                relative_l1=l1/denominator if active else None,relative_active=active,scale=scale)

def assess(problem, result, backend, cells):
    x=np.asarray(result["x"]);volumes=np.asarray(result["volumes"]);fields={k:np.asarray(v) for k,v in result["fields"].items()}
    fields["velocity"]=fields["v_m"] if problem.definition.model_name=="euler_annular" else fields["u"]
    if problem.definition.model_name=="euler_annular":
        inlet_pressure=float(problem.reference_fields([float(problem.mesh.faces[0])])["p"][0])
        reservoir_pressure=float(problem.definition.data["inlet"]["p0"])
        fields["Cp"]=(fields["p"]-inlet_pressure)/(reservoir_pressure-inlet_pressure)
    # Match representation: current FV cells use exact conserved-volume
    # projection; the native nodal nozzle uses independent point values.
    projected=backend=="current" or problem.definition.model_name=="euler_annular"
    exact=problem.reference_cell_fields() if projected else problem.reference_fields(x)
    exact_q=problem.cell_reference()["q"] if projected else problem.reference_fields(x)["q"]
    norms={}
    for name,scale in problem.field_scales.items():
        if name in fields and name in exact and np.shape(fields[name])==np.shape(exact[name]):
            weights=np.asarray(problem.mesh.widths) if name=="Cp" else volumes
            norms[name]=error_norms(fields[name],exact[name],weights,float(scale))
    q=np.asarray(result["q"]);conservative={}
    for index,label in enumerate(problem.model.equation_names):
        conservative[label]=error_norms(q[:,index],exact_q[:,index],volumes,float(problem.model.conservative_scales[index]))
    max_conservative=max((item["relative_l1"] if item["relative_active"] else item["normalized_l1"] for item in conservative.values()),default=float("inf"))
    max_conservative_normalized=max((item["normalized_l1"] for item in conservative.values()),default=float("inf"))
    targets=problem.definition.data["accuracy"];checks={}
    if "normalized_field_l1" in targets:
        selected=[norms[name]["normalized_l1"] for name in ("p","rho","velocity","M","T") if name in norms]
        checks["normalized_field_l1"]=max(selected,default=float("inf"))<=targets["normalized_field_l1"]
    if "conservative_normalized_l1" in targets:
        checks["conservative_normalized_l1"]=max_conservative_normalized<targets["conservative_normalized_l1"]
    if "Cp_length_mean_absolute" in targets:
        checks["Cp_length_mean_absolute"]=norms.get("Cp",{}).get("l1",float("inf"))<=targets["Cp_length_mean_absolute"]
    shock={"expected":problem.shock_position,"computed":None,"error_m":None,"error_cells":None,"method":None}
    if problem.shock_position is not None:
        mach=np.asarray(fields.get("M_m",fields.get("M")))
        crossings=np.where((mach[:-1]>1)&(mach[1:]<=1))[0]
        if len(crossings):
            index=int(crossings[0]);position=float(.5*(x[index]+x[index+1]))
            shock["method"]="measured interior Mach crossing"
        elif abs(problem.shock_position-float(problem.mesh.faces[-1]))<=2*np.max(np.diff(problem.mesh.faces)) and mach[-1]>1:
            position=float(problem.mesh.faces[-1])
            shock["method"]="inferred outgoing boundary limit; no interior crossing"
        else:position=None
        if position is not None:
            error=abs(position-problem.shock_position);spacing=float(np.max(np.diff(problem.mesh.faces)))
            shock.update(computed=position,error_m=error,error_cells=error/spacing)
        checks["shock_position"]=shock["error_cells"] is not None and shock["error_cells"]<=targets.get("shock_position_cells",2.)
    finest=cells>=targets["recommended_finest_cells"]
    analytical=bool(all(checks.values()))
    mach=np.asarray(fields.get("M_m",fields.get("M")))
    boundaries=problem.definition.data["boundaries"]
    # Endpoint regimes belong to the physical traces (an exit shock can leave
    # the last physical cell supersonic). Use the model's trace/EOS audit.
    branch_checks=dict(forward_meridional=bool(np.all(fields["velocity"]>0)))
    if problem.shock_position is not None:
        branch_checks["shock_present"]=shock["computed"] is not None
    elif (problem.definition.model_name=="euler_annular" and boundaries["inlet"]=="subsonic") or problem.definition.data["reference"].get("condition") in ("throat_mach","critical"):
        branch_checks["subsonic_branch"]=bool(np.max(mach)<=1+1e-8)
    branch_checks["model_solution_audit"]=bool(problem.model.solution_audit(np.asarray(result["state"])))
    dense_x=np.linspace(float(problem.mesh.faces[0]),float(problem.mesh.faces[-1]),1001)
    if problem.shock_position is not None:
        epsilon=float(problem.mesh.faces[-1]-problem.mesh.faces[0])*1e-10
        dense_x=np.unique(np.r_[dense_x,max(dense_x[0],problem.shock_position-epsilon),min(dense_x[-1],problem.shock_position+epsilon)])
    dense=problem.reference_fields(dense_x)
    reference_fields={k:v for k,v in dense.items() if np.shape(v)==dense_x.shape and np.issubdtype(np.asarray(v).dtype,np.number)}
    return dict(errors=norms,conservative_errors=conservative,metrics=dict(result["metrics"],max_conservative_relative_l1=max_conservative,max_conservative_normalized_l1=max_conservative_normalized),
                analytical_passed=analytical,accuracy_assessed=finest,accuracy_checks=checks,accuracy_targets=targets,
                branch_passed=bool(all(branch_checks.values())),branch_checks=branch_checks,
                shock=shock,reference_representation="conservative finite-volume projection" if projected else "native coordinate point sampling",
                reference=dict(x=dense_x,fields=reference_fields),
                state=dict(x=x,fields=fields,full=result["state"],conserved_density=q),
                known_limit=problem.definition.data.get("known_limit"))

def stats(samples):
    values=np.asarray(samples,float)
    if not len(values):return dict(samples_s=[],mean_s=None,min_s=None,max_s=None,std_s=None)
    return dict(samples_s=values.tolist(),mean_s=float(values.mean()),min_s=float(values.min()),max_s=float(values.max()),std_s=float(values.std(ddof=1)) if len(values)>1 else 0.)

def rotating_order(methods,index):
    if not methods:return []
    offset=index%len(methods)
    return methods[offset:]+methods[:offset]


class WorkerProtocol:
    def _read(self):
            queue=Queue()
            Thread(target=lambda:queue.put(self.process.stdout.readline()),daemon=True).start()
            try:line=queue.get(timeout=self.timeout)
            except Empty:
                self.process.kill();self.process.wait(timeout=15)
                raise TimeoutError(f"{self.backend} complete solve exceeded {self.timeout}s; see worker log")
            if not line:raise RuntimeError(f"{self.backend} worker exited ({self.process.poll()}); see worker log")
            return json.loads(line)

    def ask(self, command):
            self.process.stdin.write(json.dumps(clean(command),allow_nan=False)+"\n");self.process.stdin.flush()
            result=self._read()
            if not result.get("ok"):
                error=result.get("traceback",result)
                if "UnsupportedLegacyCase" in str(error):raise NotImplementedError(error)
                raise RuntimeError(error)
            return result.get("result")

    def close(self):
            if getattr(self,"process",None) is not None and self.process.poll() is None:
                try:
                    self.process.stdin.write('{"command":"shutdown"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
                except Exception:
                    self.process.kill();self.process.wait(timeout=10)
            if getattr(self,"log",None):self.log.close()
