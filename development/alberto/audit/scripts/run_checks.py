"""Execute the legacy development entry points with bounded, headless subprocesses.

Run from the repository root: python -m development.alberto.audit.scripts.run_checks
No numerical settings/functions are patched. Status is execution coverage,
not a mathematical accuracy certificate. Detailed logs retain each failure.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "development/alberto/outputs/audit"


def task(name, module=None, code=None, args=(), purpose="", coverage="script"):
    command = [sys.executable, "-u"] + (
        ["-m", module, *args] if module else ["-c", code]
    )
    return dict(file=name, command=command, purpose=purpose, coverage=coverage)


def tasks():
    config_code = """from pathlib import Path
import json
from development.alberto.two_fluid.config_loader import load_config,resolved_dict
results=[]
for path in sorted(Path("development/alberto/two_fluid/settings").glob("*.yaml")):
 if path.name.startswith("defaults"): continue
 try:
  cfg=load_config(path)
  assert cfg.Nx>0
  results.append(dict(file=str(path),works=True,Nx=cfg.Nx))
 except Exception as exc:
  results.append(dict(file=str(path),works=False,error=repr(exc)))
Path("development/alberto/outputs/audit/configurations.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
assert any(x["works"] for x in results)
"""
    initial_code = """import numpy as np
from development.alberto.two_fluid.initial_conditions import uniform_ic,stagnation_ic,simple_ic
arrays=[uniform_ic(7,(2e5,5.,5.,3e5,1e5),.1),stagnation_ic(7,2e5,.1,3e5,1e5),simple_ic(7,2e5,1e5,.1,1.,3e5,1e5)]
assert all(a.shape==(7,6) and np.isfinite(a).all() for a in arrays)
print("Uniform, stagnation and simple initial conditions pass; external NozzleX/restart paths not certified.")
"""
    info_code = """from pathlib import Path
import yaml
from development.alberto.support.run_info import write_run_info
from development.alberto.two_fluid.config_loader import load_config
cfg=load_config("development/alberto/two_fluid/settings/settings_two_fluid.yaml")
out=Path("development/alberto/outputs/audit/provenance");out.mkdir(parents=True,exist_ok=True)
p=write_run_info(out,cfg,input_files={"config":cfg._settings.path})
d=yaml.safe_load(p.read_text(encoding="utf-8"))
assert d["versions"]["python"] and d["inputs"]["config"]["sha256"]
print("Run provenance written and read back:",p)
"""
    affinity_code = """import os,sys
from development.alberto.support.cpu_affinity import pin_cpus
if sys.platform=="win32":
 import ctypes
 from ctypes import wintypes
 dll=ctypes.WinDLL("kernel32",use_last_error=True)
 dll.GetCurrentProcess.restype=wintypes.HANDLE
 dll.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
 current=ctypes.c_size_t();system=ctypes.c_size_t()
 assert dll.GetProcessAffinityMask(dll.GetCurrentProcess(),ctypes.byref(current),ctypes.byref(system))
 cpu=(current.value & -current.value).bit_length()-1
else:
 cpu=min(os.sched_getaffinity(0))
pin_cpus(str(cpu))
print("Pinned this disposable child process only.")
"""
    plot_code = """import numpy as np
import matplotlib.pyplot as plt
from development.alberto.sensitivity.sensitivity_plots import plot_sensitivity_fields
x=np.linspace(0,.2,15)
state={"p":2e5-1e5*x/.2,"u_g":5+10*x/.2,"u_l":5+2*x/.2}
sens={k:np.vstack([np.zeros_like(x),np.ones_like(x)]) for k in state}
fig=plot_sensitivity_fields(x,state,sens,["N_b","N_d"],alpha_g=np.linspace(.1,.9,len(x)),title="Legacy plotting audit")
fig.savefig("development/alberto/outputs/audit/sensitivity_plot.png")
plt.close(fig)
print("Rendered a canned-data sensitivity figure.")
"""
    return [
        task(
            "euler_nozzle/main.py",
            "development.alberto.euler_nozzle.main",
            purpose="Legacy single-fluid Euler/HEM nozzle driver",
        ),
        task(
            "euler_nozzle/shock_tube_fvs.py",
            "development.alberto.euler_nozzle.shock_tube_fvs",
            purpose="Explicit FVS Sod shock tube with tabulated air",
        ),
        task(
            "two_fluid/main_two_fluid.py",
            "development.alberto.two_fluid.main_two_fluid",
            purpose="Two-fluid flashing nitrogen default campaign",
        ),
        task(
            "two_fluid/main_two_fluid_grid_sequence.py",
            "development.alberto.two_fluid.main_two_fluid_grid_sequence",
            args=("settings/settings_flashing_nitrogen_simoneau.yaml",),
            purpose="Coarse-to-fine two-fluid restart campaign",
        ),
        task(
            "sensitivity/sensitivity_study.py",
            "development.alberto.sensitivity.sensitivity_study",
            purpose="Solve and differentiate interphase-closure parameters",
        ),
        task(
            "sensitivity/sensitivity_closure.py",
            "development.alberto.sensitivity.sensitivity_closure",
            purpose="Differentiate an existing saved converged two-fluid field",
        ),
        task(
            "two_fluid/config_loader.py",
            code=config_code,
            purpose="Resolve settings inheritance and typed configuration",
            coverage="all case YAML files; defaults treated as fragments",
        ),
        task(
            "two_fluid/initial_conditions.py",
            code=initial_code,
            purpose="Build prescribed, cold, restart or external-solver initial fields",
            coverage="three self-contained initializers",
        ),
        task(
            "support/run_info.py",
            code=info_code,
            purpose="Save code, environment and input provenance",
            coverage="write/read a real provenance file",
        ),
        task(
            "support/cpu_affinity.py",
            code=affinity_code,
            purpose="Pin a solver process to selected logical CPUs",
            coverage="one allowed CPU in a disposable process",
        ),
        task(
            "sensitivity/sensitivity_plots.py",
            code=plot_code,
            purpose="Render sensitivity profiles and diagnostics",
            coverage="render canned-data field figure",
        ),
        task(
            "two_fluid/settings/settings_two_fluid.py",
            "development.alberto.two_fluid.settings.settings_two_fluid",
            purpose="Legacy Python configuration wrapper",
        ),
        task(
            "two_fluid/settings/settings_phase_change.py",
            "development.alberto.two_fluid.settings.settings_phase_change",
            purpose="Legacy phase-change configuration wrapper",
        ),
    ]


def run(item, timeout):
    name = item["file"].replace("/", "__") + ".log"
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("SETTINGS_YAML", None)
    env.update(item.get("environment", {}))
    began = perf_counter()
    with (OUTPUT / name).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            item["command"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = perf_counter() + timeout
        while process.poll() is None and perf_counter() < deadline:
            sleep(0.1)
        if process.poll() is None:
            state = "timeout"
            code = None
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        else:
            code = process.returncode
            state = "completed" if code == 0 else "error"
    return dict(
        **item,
        works=state == "completed",
        state=state,
        returncode=code,
        wall_s=perf_counter() - began,
        log=name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "results.json"
    previous = json.loads(destination.read_text()) if destination.exists() else []
    selected = [x for x in tasks() if not args.only or x["file"] in args.only]
    result = [x for x in previous if x["file"] not in [i["file"] for i in selected]]
    for item in selected:
        print("Checking", item["file"], flush=True)
        entry = run(item, args.timeout)
        result.append(entry)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            " ", entry["works"], entry["state"], f"{entry['wall_s']:.1f}s", flush=True
        )


if __name__ == "__main__":
    main()
