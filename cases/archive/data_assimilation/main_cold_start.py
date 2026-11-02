"""
Cold start of the design case 4 nozzle from the HEM design, refined in stages.
================================================================================

Stage 1 starts from the nozzlex HEM inverse design (design_case4/hem_seed_*.csv)
on a coarse grid; every later stage restarts from the previous one (interpolated
onto its grid) with the numerics of the STAGES table below:

    stage  grid  recon             alpha_eps  inlet void   notes
    1      100   muscl_van_albada  1e-3       3e-3         HEM seed; stops at the SSR plateau
    2      200   muscl_van_albada  1e-6       1e-4         thinner ghost
    3      300   muscl_van_albada  1e-6       1e-4         va_rel 0.01 + SER CFL ramp to 10
                                                           (not run by default, see below)

Tested 2026-10-14 with the polynomial area law, d_min 65 um: stage 1 plateaus at SSR 1.4e-5
(6449 steps, 6 min), stage 2 at SSR 3.85e-8 (11547 steps, 20 min; it is there
after ~7000, the rest was the plateau window, since cut from 4000 to 2000 steps
in settings_cold_start.yaml).  Stage 3
(N = 300) stalls at SSR ~1e-4 with this law, so the default is --stages 1-2.
Not used any more: the hand-made chain for the original law (c4_try03 -> 04b ->
08 -> 12) had an N = 200 *constant*-reconstruction stage that thinned the ghost
to 1e-5; with the new law that stage blows up (gas enthalpy at step ~2800).

The interfacial area law comes from the case file (sources.interfacial_area,
sources.interfacial_area_params -- settings_cold_start.yaml: polynomial, d_min
65 um); --area / --d-min / --p0 override it for every stage:

    --area polynomial  Bernstein-polynomial diameter (source_two_fluid.interfacial_area_poly);
                       --d-min sets its nucleation floor, --p0 its first control
                       diameter d_ctrl[0].  With d_min 10 um (p0 160 um) the onset
                       has ~10x less interface than the original law with
                       Nb = 1e13 and case 4 blows up at the flashing front.
                       d_min 65 um works; 33 um (= original law's area at the ghost
                       level) still blows up in stage 1, because the cold start
                       sits at alpha ~ 3e-3 where it gives ~3x less area.
    --area original    number-density law (sources.bubble_number / droplet_number)
    --area parabola    inverted parabola in alpha (interfacial_area_para)

    python main_cold_start.py                          # case-file law, stages 1-2
    python main_cold_start.py --d-min 45e-6            # other floor
    python main_cold_start.py --area original
    python main_cold_start.py --stages 3 --restart results/<stage-2 run folder>
    python main_cold_start.py other_case.yaml --tag try2

Each stage is an ordinary main_two_fluid.py run (own results folder).  The
generated stage settings and a summary.txt go into
results/<repo_run>_cold_<area>_<timestamp>/.  The sequence stops at the first
stage that ends in NaN/Inf, crashes or diverges (SSR blown up while the state
stays finite -- the driver only stops on NaN, so the guard here kills it).

Environment: the project's poetry venv (jax, jaxprop, CoolProp).
"""

import argparse
import copy
import datetime as dt
import inspect
import os
import pathlib
import re
import runpy
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
DEV_SCRIPTS = HERE.parents[1] / "dev_scripts"
DRIVER = DEV_SCRIPTS / "main_two_fluid.py"
sys.path.insert(0, str(DEV_SCRIPTS))

MUSCL = "muscl_van_albada"

# Divergence guard (see launch_stage): a stage is killed when its SSR exceeds
# both DIVERGE_SSR and DIVERGE_FACTOR x the best SSR it has reached.
DIVERGE_SSR = 1.0e-2
DIVERGE_FACTOR = 1.0e3
STEP_LINE = re.compile(r"^\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+\(")   # step  t  dt  SSR  (wall)

# Overrides applied on top of the case file, one dict per stage (nested like the
# settings file).  Stage 1 keeps the case file's ic (the HEM seed); later stages
# get ic: restart from the previous stage.
STAGES = [
    {"grid": {"Nx": 100},
     "numerics": {"recon": MUSCL, "alpha_eps": 1.0e-3},
     "bc": {"void_fraction": 3.0e-3},
     "run": {"n_steps": 40000}},
    {"grid": {"Nx": 200},
     "numerics": {"recon": MUSCL, "alpha_eps": 1.0e-6},
     "bc": {"void_fraction": 1.0e-4},
     "run": {"n_steps": 15000}},
    {"grid": {"Nx": 300},
     "numerics": {"recon": MUSCL, "alpha_eps": 1.0e-6, "va_rel": 0.01},
     "bc": {"void_fraction": 1.0e-4},
     "time_control": {"ser_beta": 0.7, "ser_cfl_max": 10},
     "run": {"n_steps": 15000}},
]
DEFAULT_STAGES = "1-2"


# ─────────────────────────────────────────────────────────────────────────────
# Interfacial area (sources.interfacial_area / interfacial_area_params)
# ─────────────────────────────────────────────────────────────────────────────
def apply_area_overrides(sources, cli):
    """Write --area / --d-min / --p0 into the sources block. Returns a short tag."""
    if cli.area:
        sources["interfacial_area"] = cli.area
        if cli.area == "original":
            sources["interfacial_area_params"] = None   # Nb / Nd come from the sources block
    law = sources.get("interfacial_area", "polynomial")
    params = dict(sources.get("interfacial_area_params") or {})
    if cli.d_min is not None and law != "original":
        params["d_min"] = cli.d_min
    if cli.p0 is not None and law == "polynomial":
        from nozzle_hyperbolic.fvm_solver.source_two_fluid import interfacial_area_poly
        d_ctrl = list(params.get("d_ctrl") or
                      inspect.signature(interfacial_area_poly).parameters["d_ctrl"].default)
        d_ctrl[0] = cli.p0
        params["d_ctrl"] = d_ctrl
    sources["interfacial_area_params"] = params or None
    tag = law
    if "d_min" in params:
        tag += f"_dmin{params['d_min'] * 1e6:g}um"
    if cli.p0 is not None and "d_ctrl" in params:
        tag += f"_p0{params['d_ctrl'][0] * 1e6:g}um"
    return tag


def run_stage(settings_yaml):
    """Stage process: run main_two_fluid.py on settings_yaml from this folder."""
    import matplotlib
    matplotlib.use("Agg")
    os.environ["SETTINGS_YAML"] = str(settings_yaml)
    os.chdir(HERE)                     # table_dir / geometry / seed paths resolve here
    sys.argv = [str(DRIVER)]
    runpy.run_path(str(DRIVER), run_name="__main__")


# ─────────────────────────────────────────────────────────────────────────────
# Stage sequence (parent process)
# ─────────────────────────────────────────────────────────────────────────────
def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def parse_stages(spec):
    a, _, b = spec.partition("-")
    first, last = int(a), int(b or a)
    if not 1 <= first <= last <= len(STAGES):
        raise SystemExit(f"--stages {spec}: stages are 1..{len(STAGES)}")
    return list(range(first, last + 1))


def launch_stage(settings_yaml):
    """Run one stage in a fresh process, streaming its output. Returns (out_dir, stop, summary)."""
    cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()),
           "--run-stage", str(settings_yaml)]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    out_dir = stop = summary = None
    ssr_best, diverged = float("inf"), None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Output directory:"):
            out_dir = pathlib.Path(line.split(":", 1)[1].strip())
        elif line.startswith("Stopped:"):
            stop = line.strip()
        elif line.strip().startswith("steps="):
            summary = line.strip()
        elif (m := STEP_LINE.match(line)):
            # Divergence guard: a blown-up state can stay finite (SSR ~1e16) and the
            # driver only stops on NaN, so it would march on and hand garbage to the
            # next stage.  Kill the stage once the SSR has jumped far above its best.
            ssr = float(m.group(2))
            ssr_best = min(ssr_best, ssr)
            if not ssr < max(DIVERGE_SSR, DIVERGE_FACTOR * ssr_best):
                diverged = f"diverged at step {m.group(1)} (SSR {ssr:.3e}, best {ssr_best:.3e})"
                print(f"\n[main_cold_start] {diverged} -- killing the stage", flush=True)
                proc.kill()
                break
    proc.wait()
    if diverged:
        stop = diverged
    elif proc.returncode != 0:
        stop = f"crashed (exit code {proc.returncode})"
    return out_dir, stop, summary


def main(cli):
    from config_loader import resolved_dict

    case_path = (HERE / cli.settings).resolve()
    base = resolved_dict(case_path)
    base.pop("grid_sequence", None)
    area_tag = apply_area_overrides(base["sources"], cli)
    stages = parse_stages(cli.stages)

    prev_dir = None
    if stages[0] > 1:
        if not cli.restart:
            raise SystemExit(f"--stages {cli.stages} starts after stage 1: give --restart <run folder>")
        prev_dir = (HERE / cli.restart).resolve()

    results = HERE / "results"
    prefix = cli.tag or str((base.get("save_options") or {}).get("repo_run") or "c4_cold")
    seq_dir = results / f"{prefix}_cold_{area_tag}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cold start {case_path.name}: stages {stages}, area {area_tag}  ->  {seq_dir}")

    rows = []
    for i in stages:
        cfg = deep_merge(copy.deepcopy(base), STAGES[i - 1])
        if cli.cpus:
            cfg["run"]["cpu_affinity"] = cli.cpus
        if prev_dir is not None:
            cfg["ic"] = {"type": "restart", "run": str(prev_dir)}
            cfg["bc"]["p_out_ramp_steps"] = 0
        cfg["save_options"] = {"repo_main": str(results), "repo_run": f"{prefix}_S{i}"}
        stage_yaml = seq_dir / f"stage{i}_N{cfg['grid']['Nx']}.yaml"
        with open(stage_yaml, "w", encoding="utf-8") as f:
            f.write(f"# Stage {i} of main_cold_start.py on {case_path.name} "
                    f"(defaults chain resolved), interfacial area: {area_tag}\n")
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(f"\n{'=' * 78}\nStage {i}/{len(STAGES)}: Nx={cfg['grid']['Nx']}  "
              f"recon={cfg['numerics']['recon']}  alpha_eps={cfg['numerics']['alpha_eps']:g}  "
              f"void={cfg['bc']['void_fraction']:g}  IC={cfg['ic']['type']}\n{'=' * 78}", flush=True)
        out_dir, stop, summary = launch_stage(stage_yaml)
        rows.append((i, cfg["grid"]["Nx"], stop or "?", summary or "?", out_dir))
        if out_dir is None or stop is None or re.search(r"NaN|crashed|interrupted|diverged", stop):
            print(f"\nStage {i} failed ({stop}) -- stopping the sequence.")
            break
        prev_dir = out_dir

    last = rows[-1][2] if rows else ""
    status = ("CONVERGED (SSR < tol)" if re.search(r"SSR=.*< tol", last)
              else "on an SSR plateau" if "plateau" in last
              else "max steps reached" if "max steps" in last else "FAILED")
    lines = [f"Cold start of {case_path.name}, interfacial area {area_tag}", ""]
    lines += [f"S{i}  N={nx:<4d} {summary:<55s} {stop}\n     {out}" for i, nx, stop, summary, out in rows]
    lines += ["", f"last stage: {status}"]
    (seq_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", nargs="?", default="settings_cold_start.yaml",
                    help="case file, relative to this folder")
    ap.add_argument("--area", choices=["original", "polynomial", "parabola"],
                    help="interfacial area law (default: sources.interfacial_area of the case file)")
    ap.add_argument("--d-min", type=float,
                    help="nucleation floor d_min of the polynomial / parabola law [m]")
    ap.add_argument("--p0", type=float,
                    help="polynomial law: first control diameter d_ctrl[0] [m] (code default 160e-6)")
    ap.add_argument("--stages", default=DEFAULT_STAGES,
                    help=f"e.g. 1-2, 2-3, 3 (stages 1..{len(STAGES)}, default {DEFAULT_STAGES})")
    ap.add_argument("--restart", help="run folder to start from when --stages does not begin at 1")
    ap.add_argument("--tag", help="results name prefix (default: save_options.repo_run)")
    ap.add_argument("--cpus", help='cpu_affinity for every stage, e.g. "0-13"')
    ap.add_argument("--run-stage", help=argparse.SUPPRESS)   # internal: one stage process
    cli = ap.parse_args()

    if cli.run_stage:
        run_stage(cli.run_stage)
    else:
        main(cli)
