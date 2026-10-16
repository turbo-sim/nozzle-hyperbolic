"""
Ladder of stages from the HEM design to the N = 400 thin-ghost solution.
=========================================================================

The stages are defined in the settings file (`ladder:` block of settings_ladder.yaml),
each one as ordinary settings overrides on top of the case.  Stage 1 starts from the
case's ic (the nozzlex HEM seed); every later stage restarts from the previous one,
interpolated onto its grid.  Each stage is an ordinary main_two_fluid.py run with its
own results folder; the generated stage settings and summary.txt go into
results/<repo_run>_ladder_<time>/.

    python main_ladder.py                                  # settings_ladder.yaml, all stages
    python main_ladder.py other.yaml                       # another ladder file
    python main_ladder.py --stages 3-4 --restart results/<N200 run folder>
    python main_ladder.py --stages 4 --restart results/<N400 run folder> \
                          --set numerics.CFL=8 --set run.t_end=0.08 --tag try_cfl8
    python main_ladder.py --dry-run                        # write the stage files only
    python main_ladder.py --cpus 0-15                      # pin every stage

A stage is accepted when it stops with SSR < run.ssr_tol, or at t_end / plateau /
n_steps with SSR < ladder.accept_ssr.  It fails on NaN, a crash, an interrupt, or
divergence (SSR > DIVERGE_SSR and > DIVERGE_FACTOR x its best: killed here, since the
driver only stops on NaN).  A failed stage is retried with its `retry:` overrides
(once), or, with `split: true`, split at the geometric mean of the previous and the
target inlet void (up to ladder.max_splits times).  The sequence stops at the first
stage that cannot be recovered.

Environment: the project's poetry venv (jax, jaxprop, CoolProp).
"""

import argparse
import copy
import datetime as dt
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

DIVERGE_SSR = 1.0e-2
DIVERGE_FACTOR = 1.0e3
STEP_LINE = re.compile(r"^\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+\(")   # step  t  dt  SSR  (wall)
SUMMARY_SSR = re.compile(r"SSR=(\S+)")
STAGE_KEYS = {"name", "retry", "split"}                          # besides settings sections


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def set_dotted(d, dotted, value):
    *path, last = dotted.split(".")
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def load_ladder(settings_path, seq_dir):
    """(resolved case dict, ladder dict).  The `ladder:` block is not a settings
    section, so the case is written without it to seq_dir/case.yaml (base made
    absolute) and resolved through config_loader like any settings file."""
    from config_loader import resolved_dict, _find_base

    with open(settings_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ladder = raw.pop("ladder", None)
    if not ladder or not ladder.get("stages"):
        raise SystemExit(f"{settings_path.name}: no `ladder: stages:` block")
    if str(raw.get("base", "none")).strip().lower() not in ("none", "null", ""):
        raw["base"] = str(_find_base(raw["base"], settings_path))
    case_file = seq_dir / "case.yaml"
    with open(case_file, "w", encoding="utf-8") as f:
        f.write(f"# case part of {settings_path} (the ladder: block removed)\n")
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    base = resolved_dict(case_file)
    base.pop("grid_sequence", None)
    for i, st in enumerate(ladder["stages"], 1):
        if "name" not in st:
            raise SystemExit(f"ladder stage {i}: give it a name")
        bad = [k for k in st if k not in STAGE_KEYS and k not in base]
        if bad:
            raise SystemExit(f"ladder stage {i} ({st['name']}): unknown keys {bad}")
    return base, ladder


def run_void(run_dir):
    """bc.void_fraction of a finished run (its resolved settings file is named
    after the settings file it ran on; run_info.yaml records which)."""
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "run_info.yaml", encoding="utf-8") as f:
        name = (yaml.safe_load(f).get("settings") or {}).get("resolved_file")
    files = [run_dir / name] if name else [p for p in run_dir.glob("*.yaml")
                                           if p.name != "run_info.yaml"]
    with open(files[0], encoding="utf-8") as f:
        return float(yaml.safe_load(f)["bc"]["void_fraction"])


# ─────────────────────────────────────────────────────────────────────────────
# One stage
# ─────────────────────────────────────────────────────────────────────────────
def run_stage(settings_yaml):
    """Stage process: run main_two_fluid.py on settings_yaml from this folder."""
    import matplotlib
    matplotlib.use("Agg")
    os.environ["SETTINGS_YAML"] = str(settings_yaml)
    os.chdir(HERE)                     # table_dir / geometry / seed paths resolve here
    sys.argv = [str(DRIVER)]
    runpy.run_path(str(DRIVER), run_name="__main__")


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
            ssr = float(m.group(2))
            ssr_best = min(ssr_best, ssr)
            if not ssr < max(DIVERGE_SSR, DIVERGE_FACTOR * ssr_best):
                diverged = f"diverged at step {m.group(1)} (SSR {ssr:.3e}, best {ssr_best:.3e})"
                print(f"\n[main_ladder] {diverged} -- killing the stage", flush=True)
                proc.kill()
                break
    proc.wait()
    if diverged:
        stop = diverged
    elif proc.returncode != 0:
        stop = f"crashed (exit code {proc.returncode})"
    return out_dir, stop, summary


def stage_ok(out_dir, stop, summary, accept_ssr):
    if out_dir is None or stop is None or summary is None:
        return False
    if re.search(r"NaN|crashed|interrupted|diverged", stop):
        return False
    m = SUMMARY_SSR.search(summary)
    return bool(m) and float(m.group(1)) < accept_ssr


# ─────────────────────────────────────────────────────────────────────────────
# The sequence
# ─────────────────────────────────────────────────────────────────────────────
def parse_stages(spec, n):
    if not spec:
        return list(range(1, n + 1))
    a, _, b = spec.partition("-")
    first, last = int(a), int(b or a)
    if not 1 <= first <= last <= n:
        raise SystemExit(f"--stages {spec}: the ladder has stages 1..{n}")
    return list(range(first, last + 1))


def main(cli):
    bad_set = [s for s in cli.set if "=" not in s]
    if bad_set:
        raise SystemExit(f"--set needs SECTION.KEY=VALUE, got {bad_set}")
    cli_set = [(k, yaml.safe_load(v)) for k, _, v in (s.partition("=") for s in cli.set)]
    settings_path = (HERE / cli.settings).resolve()
    results = HERE / "results"
    with open(settings_path, encoding="utf-8") as f:
        _rr = (yaml.safe_load(f).get("save_options") or {}).get("repo_run")
    prefix = cli.tag or str(_rr or "ladder")
    seq_dir = results / f"{prefix}_ladder_{dt.datetime.now():%Y%m%d_%H%M%S}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    base, ladder = load_ladder(settings_path, seq_dir)
    stages_def = ladder["stages"]
    accept_ssr = float(ladder.get("accept_ssr", 1.0e-6))
    max_splits = int(ladder.get("max_splits", 2))
    stages = parse_stages(cli.stages, len(stages_def))

    prev_dir = prev_void = None
    if stages[0] > 1:
        if not cli.restart:
            raise SystemExit(f"--stages {cli.stages} starts after stage 1: give --restart <run folder>")
        prev_dir = (HERE / cli.restart).resolve()
        prev_void = run_void(prev_dir)

    names = [s["name"] for s in stages_def]
    print(f"Ladder {settings_path.name}: stages {stages} of {names}  ->  {seq_dir}")
    if cli_set:
        print(f"  --set on every stage: {dict(cli_set)}")

    rows, n_run = [], 0

    def attempt(i, stage, extra=None, note=""):
        """One run of stage i on top of prev_dir. Returns (ok, out_dir)."""
        nonlocal n_run
        n_run += 1
        ov = {k: v for k, v in stage.items() if k not in STAGE_KEYS}
        cfg = deep_merge(copy.deepcopy(base), ov)
        if extra:
            deep_merge(cfg, extra)
        for k, v in cli_set:
            set_dotted(cfg, k, v)
        if cli.cpus:
            cfg["run"]["cpu_affinity"] = cli.cpus
        if prev_dir is not None:
            cfg["ic"] = {"type": "restart", "run": str(prev_dir)}
            cfg["bc"]["p_out_ramp_steps"] = 0
        name = stage["name"] if not extra or "bc" not in extra else \
            f"v{cfg['bc']['void_fraction']:g}"
        # the driver appends _N<Nx>_<recon>_<time>
        run_name = f"{prefix}_S{i}" + ("" if name == f"N{cfg['grid']['Nx']}" else f"_{name}")
        cfg["save_options"] = {"repo_main": str(results), "repo_run": run_name}
        stage_yaml = seq_dir / f"run{n_run:02d}_stage{i}_{name}.yaml"
        with open(stage_yaml, "w", encoding="utf-8") as f:
            f.write(f"# Stage {i} ({name}{note}) of main_ladder.py on {settings_path.name}\n")
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        nm = cfg["numerics"]
        print(f"\n{'=' * 78}\nStage {i} {name}{note}: Nx={cfg['grid']['Nx']}  "
              f"void={float(cfg['bc']['void_fraction']):g}  CFL={nm['CFL']}  "
              f"ser_beta={cfg['time_control']['ser_beta']}  alpha_eps={nm['alpha_eps']}  "
              f"t_end={cfg['run']['t_end']}  IC={cfg['ic']['type']}\n{'=' * 78}", flush=True)
        if cli.dry_run:
            rows.append((i, name + note, cfg["grid"]["Nx"], cfg["bc"]["void_fraction"],
                         nm["CFL"], "dry", "", "", stage_yaml))
            return True, f"<run folder of stage {i}>"      # what the next stage restarts from
        out_dir, stop, summary = launch_stage(stage_yaml)
        ok = stage_ok(out_dir, stop, summary, accept_ssr)
        rows.append((i, name + note, cfg["grid"]["Nx"], cfg["bc"]["void_fraction"],
                     nm["CFL"], "ok" if ok else "FAILED", summary or "?", stop or "?", out_dir))
        return ok, out_dir

    failed = False
    for i in stages:
        stage = stages_def[i - 1]
        target = float(stage.get("bc", {}).get("void_fraction", base["bc"]["void_fraction"]))
        if stage.get("split"):
            todo, splits, ok = [target], 0, True
            while todo:
                v = todo[0]
                extra = None if v == target else {"bc": {"void_fraction": v}}
                ok, out = attempt(i, stage, extra, "" if extra is None else " (split)")
                if ok:
                    todo.pop(0)
                    prev_dir, prev_void = out, v
                    continue
                if prev_void is None or splits >= max_splits or not prev_void > v:
                    break
                splits += 1
                todo.insert(0, float(f"{math.sqrt(prev_void * v):.2g}"))
                print(f"\n[main_ladder] stage {i} to void {v:g} failed -- "
                      f"splitting at {todo[0]:g}", flush=True)
        else:
            ok, out = attempt(i, stage)
            if not ok and stage.get("retry"):
                ok, out = attempt(i, stage, stage["retry"], " (retry)")
            if ok:
                prev_dir, prev_void = out, target
        if not ok:
            print(f"\nStage {i} ({stage['name']}) failed -- stopping the sequence.")
            failed = True
            break

    if cli.dry_run:
        status = "DRY RUN (stage files written, nothing run)"
    else:
        status = "FAILED" if failed or not rows else "CONVERGED (every stage accepted)"
        if failed and prev_dir is not None:
            status += f" -- last converged state (void {prev_void:g}): {prev_dir}"
    lines = [f"Ladder {settings_path.name}",
             f"accept: SSR < run.ssr_tol, or t_end/plateau/n_steps with SSR < {accept_ssr:g}"]
    if cli_set:
        lines.append(f"--set on every stage: {dict(cli_set)}")
    lines.append("")
    lines += [f"S{i:<2d} {lab:<16s} N={nx:<4d} void={float(v):<8g} CFL={c:<4g} {res:<6s} {summ}\n"
              f"     {stop}\n     {out}"
              for i, lab, nx, v, c, res, summ, stop, out in rows]
    lines += ["", f"sequence: {status}"]
    (seq_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", nargs="?", default="settings_ladder.yaml",
                    help="ladder settings file, relative to this folder")
    ap.add_argument("--stages", help="e.g. 1-4, 3-4, 4 (default: all)")
    ap.add_argument("--restart", help="run folder to start from when --stages does not begin at 1")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override on every selected stage, e.g. numerics.CFL=8 (repeatable)")
    ap.add_argument("--tag", help="results name prefix (default: save_options.repo_run)")
    ap.add_argument("--cpus", help='cpu_affinity for every stage, e.g. "0-15"')
    ap.add_argument("--dry-run", action="store_true",
                    help="write the stage settings and the summary, run nothing")
    ap.add_argument("--run-stage", help=argparse.SUPPRESS)   # internal: one stage process
    cli = ap.parse_args()

    if cli.run_stage:
        run_stage(cli.run_stage)
    else:
        # the stages' output (box-drawing characters) is re-printed here: a
        # redirected Windows stdout would be cp1252 and raise on it
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        main(cli)
