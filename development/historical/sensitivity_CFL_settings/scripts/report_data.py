"""Read and audit retained sensitivity records without importing a solver."""

import hashlib
import json
import math
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
CASE_ORDER = ("subsonic", "sonic", "shock35", "shock70", "supersonic")
TITLES = {"subsonic": "Fully subsonic", "sonic": "Sonic throat / subsonic exit",
          "shock35": "Shock at 35%", "shock70": "Shock at 70%",
          "supersonic": "Adapted supersonic exit"}
PAGES = (("newton", "Newton caps"), ("cfl100", "CFL / cap 100"),
         ("cfl1", "CFL / cap 1"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(output):
    paths = [output/"summary.json", output/"results.csv"]
    paths += sorted(p for p in (output/"runs").rglob("*") if p.suffix in (".json", ".csv"))
    return {p.relative_to(output).as_posix(): digest(p) for p in paths if p.is_file()}


def load_saved(output):
    """Normalize historical Windows paths only for locating retained files."""
    output = output.resolve()
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    loaded = {}
    for row in summary["runs"]:
        path = (output/row["file"].replace("\\", "/")).resolve()
        if not path.is_relative_to(output):
            raise ValueError(f"Run file escapes the selected output directory: {path}")
        loaded[row["run_id"]] = json.loads(path.read_text(encoding="utf-8"))
    return summary, loaded


def study_runs(rows, page, case=None):
    rows = [r for r in rows if case is None or r["case"] == case]
    if page == "newton":
        return sorted((r for r in rows if r["cfl_variant"] == "baseline"), key=lambda r:r["cap"])
    cap = 100 if page == "cfl100" else 1
    return [r for r in rows if r["cap"] == cap]


def audit_saved(summary, loaded):
    """Verify saved arithmetic/provenance, not current-source equation physics."""
    rows = summary["runs"]
    results = []
    for row in rows:
        run = loaded[row["run_id"]]
        hist, trials = run["history"], run["trial_history"]
        accepted = [t for t in trials if t["accepted"]]
        rejects = len(trials)-len(accepted)
        work = sum(t["inner_solves"] for t in trials)
        anchors = [r for r in rows if r["case"]==row["case"] and r["cap"]==100 and r["cfl_variant"]=="baseline"]
        anchor = anchors[0]
        baseline = loaded[anchor["run_id"]]["x"]
        if len(run["x"]) != len(baseline):
            raise ValueError(f"Inconsistent state dimension: {row['run_id']}")
        distance = math.sqrt(sum((a-b)**2 for a,b in zip(run["x"],baseline))/len(baseline))
        residual = run["diagnostics"]["residual_rms"]
        root = residual <= 1e-8
        accuracy = max(run["diagnostics"]["normalized_L1"].values()) < .03
        agreement = bool(root and anchor["root_success"] and distance <= 1e-4)
        prefix = [0]
        for t in trials: prefix.append(prefix[-1]+t["inner_solves"])
        flags = dict(
            accepted_count=len(accepted)==row["updates"]==len(hist)-1==run["iterations"],
            trial_count=len(trials)==row["trial_steps"], rejected_count=rejects==row["rejected_steps"],
            linear_work=work==row["linear_solves"]==hist[-1]["cumulative_linear_solves"],
            cap_respected=all(0<=t["inner_solves"]<=row["cap"] for t in trials),
            inner_flags=all(t["inner_converged"]==(t["inner_relative_update"] is not None
                            and math.isfinite(t["inner_relative_update"])
                            and t["inner_relative_update"]<=1e-8) for t in trials),
            accepted_inner_count=sum(t["inner_converged"] for t in accepted)==row["inner_converged_accepted"],
            history_work=all(h["cumulative_linear_solves"]==prefix[h["implicit_step_trials"]] for h in hist),
            recorded_root=root==row["root_success"]==run["native_success"],
            recorded_accuracy=accuracy==row["analytic_pass"]==run["diagnostics"]["analytical_pass"],
            final_history_residual=abs(residual-hist[-1]["residual_rms"])<=1e-14,
            timing=abs(row["solve_s"]-run["wall_s"])<=1e-12,
            recorded_distance=abs(distance-row["scaled_distance_from_baseline"])<=1e-12,
            recorded_agreement=agreement==row["same_baseline_root"],
            finite_fields=all(math.isfinite(v) and v>0 for f in run["diagnostics"]["fields"].values() for v in f),
            mesh_shape=all(len(f)==len(summary["cases"][row["case"]]["x_m"])
                           for f in run["diagnostics"]["fields"].values()),
        )
        results.append(dict(run_id=row["run_id"], pass_=all(flags.values()), checks=flags,
                            original_spatial_root=root, analytical_accuracy=accuracy,
                            baseline_agreement=agreement, scaled_distance=distance))
    seed_consistent = all(len({r["initial_sha256"] for r in rows if r["case"]==case})==1
                          for case in summary["cases"])
    unique_ids=len({r["run_id"] for r in rows})==len(rows)
    return dict(pass_=all(r["pass_"] for r in results) and seed_consistent and unique_ids,
                kind="Saved-record consistency audit; no current-source solver evaluations",
                unique_runs=len(rows), unique_ids=unique_ids,
                original_spatial_roots=sum(r["original_spatial_root"] for r in results),
                baseline_agreements=sum(r["baseline_agreement"] for r in results),
                analytical_passes=sum(r["analytical_accuracy"] for r in results),
                analytical_failures=[r["run_id"] for r in results if not r["analytical_accuracy"]],
                consistent_cold_seeds=seed_consistent, timing_samples_per_configuration=1,
                results=results)


def payload(summary, loaded):
    """Keep exact plotted arrays and compact records; no envelope/downsampling."""
    runs = {}
    for row in summary["runs"]:
        run = loaded[row["run_id"]]
        hist = run["history"]
        runs[row["run_id"]] = dict(
            **row, diagnostics={k:v for k,v in run["diagnostics"].items() if k!="fields"},
            pressure_Pa=run["diagnostics"]["fields"]["p"],
            history={k:[h.get(k) for h in hist] for k in
                ("accepted_index", "elapsed_s", "residual_rms", "effective_cfl",
                 "cumulative_linear_solves", "inner_solves", "inner_relative_update")},
            trials={k:[t[k] for t in run["trial_history"]] for k in
                ("trial_index", "anchor_index", "accepted", "effective_cfl", "inner_solves", "inner_relative_update")},
        )
    return dict(cases=summary["cases"], order=list(summary["cases"]), titles=TITLES, runs=runs,
                settings=summary["settings"], versions=summary["versions"],
                studies={key:[r["run_id"] for r in study_runs(summary["runs"],key)] for key,_ in PAGES})
