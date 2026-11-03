"""Portable evidence and independent original-equation reductions."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
GRIDS = (50,100,200,400)
METHODS = ("global", "local")
MODES = ("sequence", "single_400")


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if hasattr(value, "tolist"): return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def source_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/"src/nozzle_hyperbolic").rglob("*")) if p.suffix in (".py",".yaml")}


def work(history):
    return dict(accepted=sum(r.get("phase")=="accepted" for r in history),
        rejected=sum(r.get("phase")=="rejected" for r in history),
        corrections=sum(r.get("inner_iterations",0) for r in history),
        backtracks=sum(r.get("rejected_newton_trials",0) for r in history))


def audit(solution, problem, policy):
    op = solution.operator
    raw = np.asarray(op.spatial_residual(jnp.asarray(solution.state)))
    volumes = np.asarray(problem.mesh.volumes)
    scaled = raw*problem.model.reference_time/(volumes[:,None]*problem.model.conservative_scales)
    norms = np.sqrt(np.sum(volumes[:,None]/volumes.sum()*scaled**2,axis=0))
    initial = np.asarray(solution.history[0]["equation_norm"])
    relative = norms/np.where(initial>0,initial,1.)
    rows_passed = (norms<=policy.absolute_tolerance) | ((initial>policy.initial_floor)&(relative<=policy.relative_tolerance))
    bc = np.asarray(op.boundary_residual(jnp.asarray(solution.state),0.))
    budget = np.abs(raw.sum(axis=0)/problem.model.global_flux_scales)
    branch = bool(problem.model.solution_audit(solution.state))
    agrees = bool(np.allclose(norms,solution.metrics["equation_norm"],rtol=1e-9,atol=1e-15))
    return dict(passed=bool(solution.success and np.all(rows_passed) and np.isfinite(scaled).all()
        and max(abs(bc).ravel())<=policy.boundary_tolerance and max(budget)<=policy.global_tolerance
        and branch and agrees),equation_l2=norms,equation_relative=relative,
        boundary_max=float(np.max(abs(bc))),global_defect=budget,branch_passed=branch,reduction_agrees=agrees)
