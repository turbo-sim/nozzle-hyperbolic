"""
Cold start of the design case 4 nozzle from the HEM design: refine, then thin the ghost.
=========================================================================================

From no knowledge to the production state (N = 400, MUSCL van Albada, inlet ghost
void 1e-5) in one call.  Stage 1 starts from the nozzlex HEM inverse design
(design_case4/hem_seed_*.csv) on a coarse grid; every later stage restarts from the
previous one (interpolated onto its grid).  The case file holds the physics (fluid,
tables, geometry, reservoir, interphase closure, area law); THIS script holds the
numerics of every stage, so a new area law runs through the same ladder:

    stage  grid  inlet void  ghost numerics  CFL  SER   stop
    1      80    1e-3        thick           0.2  1     SSR < 1e-8   (HEM seed)
    2      200   1e-3        thick           0.5  1     SSR < 1e-8
    3      400   1e-3        thick           2    1     SSR < 1e-8
    4..    400   --voids     thin            4    1     t = 0.05 s, SSR on its floor
                 (default 1e-5: one rung)

    thick = alpha_eps 1e-6, eps_ghost_alo 1e-5, pin_mom_alpha null   (code defaults)
    thin  = alpha_eps 1e-8, eps_ghost_alo 1e-8, pin_mom_alpha 1e-4

Measured 2026-10-15 (case 4, parabola area law, Nb 1e12, smoothstep blend; steps of
one run alone on the machine take ~0.03 / 0.06 / 0.12 s at N = 80 / 200 / 400):

    stage                    this table          hand-made chain (CFL 0.2/0.5, no SER)
    1  HEM -> N80               647 steps           8650 steps
    2  N80 -> N200              876                 6750
    3  N200 -> N400             387                10676
    4  1e-3 -> 1e-5 (N400)     1020                 4537 for 1e-3 -> 7e-4 only
    (5  1e-5 -> 3e-6            315,  --voids 1e-5,3e-6)
    run c4_cold_S1..S5 (2026-10-15 19:02): 9.5 min wall on 10 shared logical CPUs.
    final states identical to the hand-made ones to ~1e-6 (stages 1-3); the 1e-5 state
    is the same to 1e-14 whichever thinning path reaches it.

Why the thinning rungs look like this (full record: LADDER_HISTORY.md):
  * A rung has a fixed PHYSICAL cost: the inlet void front must convect through the
    converging section; the SSR peaks at t ~ 0.015 s (front at the throat) and is on
    its floor by t ~ 0.035-0.045 s, for any jump size and CFL.  So the lever is dt:
    CFL 0.5 needs ~4500-5800 steps, CFL 2 ~1600, CFL 4 ~1000, CFL 8 ~650; and the
    lever is the number of rungs: 1e-3 -> 1e-5 works in ONE rung at CFL 4.
  * CFL 0.5 is not safer: the 3e-4 rung peaked at SSR 1.5e-4 at CFL 0.5 vs 1.3e-5
    at CFL 2 (the implicit step damps the front passage).
  * Thin-ghost runs never reach SSR 1e-8: the algebraic ghost momentum pin leaves a
    fixed residual floor (1.1e-8 at 3e-4, 7.7e-8 at 1e-4, 1.5e-8 at 1e-5), so a rung
    stops at run.t_end and is accepted when its SSR < ACCEPT_SSR.
  * Below 1e-5 at CFL 4: 1e-5 -> 3e-6 (~300 steps) and 3.2e-6 -> 1.8e-6 (274) work;
    1e-6 dies at t ~ 0.015 s (ghost h_g blow-up as the front reaches the flashing
    onset) from 1e-5, 3e-6 and 1.8e-6 alike, with or without SER -- a limit of the
    level, not of the jump.  It survives at CFL <= 2 without SER (LADDER_HISTORY.md).
  * Stage 2 at CFL 2 dies (the N80 state carries a 28 % du checkerboard): keep 0.5.

Robustness: a stage that ends in NaN / crash / divergence / SSR above ACCEPT_SSR is
retried automatically -- a thinning rung is split at the geometric mean of its
start and target void (up to MAX_SPLITS times), a stage 1-3 is rerun once without
SER (the hand-made recipe).  The interfacial area law comes from the case file
(sources.interfacial_area, sources.interfacial_area_params); --area / --d-min / --p0 /
--blend override it for every stage:

    --area polynomial  Bernstein-polynomial diameter (source_two_fluid.interfacial_area_poly);
                       --d-min sets its nucleation floor, --p0 its first control
                       diameter d_ctrl[0].
    --area original    number-density law (sources.bubble_number / droplet_number)
    --area parabola    inverted parabola in alpha (interfacial_area_para)

    python main_cold_start.py                              # stages 1-4, void 1e-5
    python main_cold_start.py --voids 1e-4,1e-5            # two thinning rungs
    python main_cold_start.py --voids 1e-5,3e-6            # one rung further (tested)
    python main_cold_start.py --area polynomial --d-min 45e-6
    python main_cold_start.py --stages 4 --restart results/<N400 run folder> --voids 1e-5
    python main_cold_start.py other_case.yaml --tag try2 --cpus 0-15

Each stage is an ordinary main_two_fluid.py run (own results folder).  The generated
stage settings and a summary.txt go into results/<repo_run>_cold_<area>_<timestamp>/.
The previous recipe of this script (polynomial law, N100 -> N200 with a 3e-3 ghost)
is in git history (commit 6fe085c).

Environment: the project's poetry venv (jax, jaxprop, CoolProp).
"""

import argparse
import copy
import datetime as dt
import inspect
import math
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
SUMMARY_SSR = re.compile(r"SSR=(\S+)")

# A stage counts as converged when its final SSR is below this (the thin-ghost
# floors are 1e-8..1e-7; the rung's front passage peaks at 1e-5..3e-5).
ACCEPT_SSR = 1.0e-6
MAX_SPLITS = 2                      # times a failed thinning rung may be split

# Ghost numerics: thick for the refinement stages (a 1e-3 ghost), thin for the rungs.
THICK = {"alpha_eps": 1.0e-6, "eps_ghost_alo": 1.0e-5, "pin_mom_alpha": None}
THIN = {"alpha_eps": 1.0e-8, "eps_ghost_alo": 1.0e-8, "pin_mom_alpha": 1.0e-4}

# Overrides applied on top of the case file, one dict per stage (nested like the
# settings file).  Stage 1 keeps the case file's ic (the HEM seed); later stages
# get ic: restart from the previous stage.
STAGES = [
    {"grid": {"Nx": 80},
     "numerics": {"recon": MUSCL, "CFL": 0.2, **THICK},
     "bc": {"void_fraction": 1.0e-3},
     "time_control": {"ser_beta": 1.0},
     "run": {"n_steps": 20000, "ssr_tol": 1.0e-8, "t_end": None}},
    {"grid": {"Nx": 200},
     "numerics": {"recon": MUSCL, "CFL": 0.5, **THICK},
     "bc": {"void_fraction": 1.0e-3},
     "time_control": {"ser_beta": 1.0},
     "run": {"n_steps": 15000, "ssr_tol": 1.0e-8, "t_end": None}},
    {"grid": {"Nx": 400},
     "numerics": {"recon": MUSCL, "CFL": 2.0, **THICK},
     "bc": {"void_fraction": 1.0e-3},
     "time_control": {"ser_beta": 1.0},
     "run": {"n_steps": 15000, "ssr_tol": 1.0e-8, "t_end": None}},
]
# Retry of a failed stage 1-3: the hand-made recipe (constant CFL, no SER).
STAGE_RETRY = {"time_control": {"ser_beta": 0.0}}

# One thinning rung (bc.void_fraction is set per rung from --voids).
THIN_RUNG = {"grid": {"Nx": 400},
             "numerics": {"recon": MUSCL, "CFL": 4.0, **THIN},
             "time_control": {"ser_beta": 1.0},
             "run": {"n_steps": 20000, "ssr_tol": 1.0e-8, "t_end": 0.05}}
DEFAULT_VOIDS = "1e-5"


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
    if cli.blend:
        sources["regime_blend"] = cli.blend
    if sources.get("regime_blend", "linear") != "linear":
        tag += f"_{sources['regime_blend']}"
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


def build_plan(voids):
    """[(label, overrides)] for stages 1-3 and one thinning rung per void."""
    plan = [(f"N{s['grid']['Nx']}", s) for s in STAGES]
    for v in voids:
        plan.append((f"v{v:g}", deep_merge(copy.deepcopy(THIN_RUNG),
                                           {"bc": {"void_fraction": v}})))
    return plan


def parse_stages(spec, n):
    a, _, b = spec.partition("-")
    first, last = int(a), int(b or a)
    if not 1 <= first <= last <= n:
        raise SystemExit(f"--stages {spec}: stages are 1..{n}")
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


def run_void(run_dir):
    """bc.void_fraction of a finished run (its resolved settings file is named
    after the case file it ran on; run_info.yaml records which)."""
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "run_info.yaml", encoding="utf-8") as f:
        name = (yaml.safe_load(f).get("settings") or {}).get("resolved_file")
    files = [run_dir / name] if name else [p for p in run_dir.glob("*.yaml")
                                           if p.name != "run_info.yaml"]
    with open(files[0], encoding="utf-8") as f:
        return float(yaml.safe_load(f)["bc"]["void_fraction"])


def stage_ok(out_dir, stop, summary):
    """Converged: SSR < tol, or (t_end / plateau / max steps) with SSR < ACCEPT_SSR."""
    if out_dir is None or stop is None or summary is None:
        return False
    if re.search(r"NaN|crashed|interrupted|diverged", stop):
        return False
    m = SUMMARY_SSR.search(summary)
    return bool(m) and float(m.group(1)) < ACCEPT_SSR


def main(cli):
    from config_loader import resolved_dict

    case_path = (HERE / cli.settings).resolve()
    base = resolved_dict(case_path)
    base.pop("grid_sequence", None)
    area_tag = apply_area_overrides(base["sources"], cli)
    voids = [float(v) for v in cli.voids.split(",") if v.strip()]
    plan = build_plan(voids)
    stages = parse_stages(cli.stages or f"1-{len(plan)}", len(plan))

    prev_dir, prev_void = None, None
    if stages[0] > 1:
        if not cli.restart:
            raise SystemExit(f"--stages {cli.stages} starts after stage 1: give --restart <run folder>")
        prev_dir = (HERE / cli.restart).resolve()
        prev_void = run_void(prev_dir)

    results = HERE / "results"
    prefix = cli.tag or str((base.get("save_options") or {}).get("repo_run") or "c4_cold")
    seq_dir = results / f"{prefix}_cold_{area_tag}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cold start {case_path.name}: stages {stages} of {[p[0] for p in plan]}, "
          f"area {area_tag}  ->  {seq_dir}")

    rows, n_run = [], 0

    def attempt(i, label, overrides, retry_note=""):
        """Run one stage on top of prev_dir. Returns (ok, out_dir)."""
        nonlocal n_run
        n_run += 1
        cfg = deep_merge(copy.deepcopy(base), overrides)
        if cli.cpus:
            cfg["run"]["cpu_affinity"] = cli.cpus
        if prev_dir is not None:
            cfg["ic"] = {"type": "restart", "run": str(prev_dir)}
            cfg["bc"]["p_out_ramp_steps"] = 0
        # the driver appends _N<Nx>_<recon>_<timestamp>; rungs also carry their void
        run_name = f"{prefix}_S{i}" if i <= len(STAGES) else f"{prefix}_S{i}_{label}"
        cfg["save_options"] = {"repo_main": str(results), "repo_run": run_name}
        stage_yaml = seq_dir / f"run{n_run:02d}_stage{i}_{label}.yaml"
        with open(stage_yaml, "w", encoding="utf-8") as f:
            f.write(f"# Stage {i} ({label}{retry_note}) of main_cold_start.py on {case_path.name} "
                    f"(defaults chain resolved), interfacial area: {area_tag}\n")
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        nm = cfg["numerics"]
        print(f"\n{'=' * 78}\nStage {i} {label}{retry_note}: Nx={cfg['grid']['Nx']}  "
              f"void={cfg['bc']['void_fraction']:g}  CFL={nm['CFL']:g}  "
              f"ser_beta={cfg['time_control']['ser_beta']:g}  alpha_eps={nm['alpha_eps']:g}  "
              f"IC={cfg['ic']['type']}\n{'=' * 78}", flush=True)
        out_dir, stop, summary = launch_stage(stage_yaml)
        ok = stage_ok(out_dir, stop, summary)
        rows.append((i, f"{label}{retry_note}", cfg["grid"]["Nx"], cfg["bc"]["void_fraction"],
                     nm["CFL"], "ok" if ok else "FAILED", summary or "?", stop or "?", out_dir))
        return ok, out_dir

    failed = False
    for i in stages:
        label, overrides = plan[i - 1]
        target = overrides["bc"]["void_fraction"]
        if i <= len(STAGES):
            ok, out = attempt(i, label, overrides)
            if not ok:
                ok, out = attempt(i, label, deep_merge(copy.deepcopy(overrides), STAGE_RETRY),
                                  " retry: no SER")
        else:
            # Thinning rung: on failure, insert the geometric mean of the start and
            # target voids (from the last converged state) and climb down again.
            todo, splits, ok = [target], 0, True
            while todo:
                v = todo[0]
                ok, out = attempt(i, f"v{v:g}", deep_merge(copy.deepcopy(overrides),
                                                            {"bc": {"void_fraction": v}}))
                if ok:
                    todo.pop(0)
                    prev_dir, prev_void = out, v
                    continue
                if prev_void is None or splits >= MAX_SPLITS or not prev_void > v:
                    break
                splits += 1
                todo.insert(0, float(f"{math.sqrt(prev_void * v):.2g}"))
                print(f"\n[main_cold_start] rung to {v:g} failed -- splitting at "
                      f"{todo[0]:.3g}", flush=True)
            if ok:
                continue
        if not ok:
            print(f"\nStage {i} ({label}) failed -- stopping the sequence.")
            failed = True
            break
        prev_dir, prev_void = out, target

    status = "FAILED" if failed or not rows else "CONVERGED (every stage accepted)"
    if failed and prev_dir is not None:
        status += f" -- last converged state (void {prev_void:g}): {prev_dir}"
    lines = [f"Cold start of {case_path.name}, interfacial area {area_tag}",
             f"accept: SSR < tol, or t_end/plateau with SSR < {ACCEPT_SSR:g}", ""]
    lines += [f"S{i:<2d} {lab:<22s} N={nx:<4d} void={v:<8g} CFL={c:<4g} {res:<6s} {summ}\n"
              f"     {stop}\n     {out}"
              for i, lab, nx, v, c, res, summ, stop, out in rows]
    lines += ["", f"sequence: {status}"]
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
    ap.add_argument("--blend", choices=["linear", "smoothstep"],
                    help="drag / regime-HTC transition weight (default: sources.regime_blend)")
    ap.add_argument("--voids", default=DEFAULT_VOIDS,
                    help=f"inlet ghost voids of the thinning rungs after stage 3, comma-separated "
                         f"(default {DEFAULT_VOIDS})")
    ap.add_argument("--stages",
                    help="e.g. 1-4, 2-3, 4 (default: all = 3 + one per --voids entry)")
    ap.add_argument("--restart", help="run folder to start from when --stages does not begin at 1")
    ap.add_argument("--tag", help="results name prefix (default: save_options.repo_run)")
    ap.add_argument("--cpus", help='cpu_affinity for every stage, e.g. "0-13"')
    ap.add_argument("--run-stage", help=argparse.SUPPRESS)   # internal: one stage process
    cli = ap.parse_args()

    if cli.run_stage:
        run_stage(cli.run_stage)
    else:
        # the stages' output (box-drawing characters) is re-printed here: a
        # redirected Windows stdout would be cp1252 and raise on it
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        main(cli)
