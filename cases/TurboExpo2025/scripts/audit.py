"""Audit saved campaign coverage and evidence without advancing a simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-converged", action="store_true")
    args = parser.parse_args()
    definitions = read(ROOT / "case_definitions.json")
    records = [(p, read(p)) for p in sorted((ROOT / "outputs").glob("**/result.json"))]
    checks, coverage = [], []
    for case in definitions["cases"]:
        for flux in case["fluxes"]:
            found = [
                (p, r)
                for p, r in records
                if r.get("case_id") == case["id"]
                and r.get("flux") == flux
                and r.get("cells") == case["cells"]
            ]
            current = (
                max(found, key=lambda pair: pair[0].stat().st_mtime) if found else None
            )
            verified = bool(
                current and current[1].get("verification", {}).get("verified")
            )
            coverage.append(
                dict(
                    case_id=case["id"],
                    flux=flux,
                    cells=case["cells"],
                    attempted=bool(found),
                    verified=verified,
                    outcomes=[
                        dict(
                            file=str(p.relative_to(ROOT)),
                            success=r.get("success"),
                            reason=r.get("reason"),
                            error=r.get("error"),
                        )
                        for p, r in found
                    ],
                )
            )
    mesh_coverage = []
    for study in definitions["mesh_studies"]:
        for cells in study["cells"]:
            found = [
                (p, r)
                for p, r in records
                if r.get("case_id") == study["case_id"]
                and r.get("flux") == study["flux"]
                and r.get("cells") == cells
            ]
            current = (
                max(found, key=lambda pair: pair[0].stat().st_mtime) if found else None
            )
            mesh_coverage.append(
                dict(
                    case_id=study["case_id"],
                    flux=study["flux"],
                    cells=cells,
                    attempted=bool(found),
                    verified=bool(
                        current and current[1].get("verification", {}).get("verified")
                    ),
                )
            )
    for path, row in records:
        tag = str(path.parent.relative_to(ROOT))
        if "scheme" in row:
            checks.append(
                dict(
                    name=tag + ":paper_spatial_implicit_time",
                    passed=row["scheme"]["time_integrator"] == "euler-implicit"
                    and row["scheme"]["reconstruction"] == "constant"
                    and row["scheme"]["flux"] == row["flux"],
                )
            )
        solution = path.with_name("solution.npz")
        if solution.exists():
            with np.load(solution) as saved:
                shapes = saved["state"].shape == (row["cells"] + 2, 3)
                finite = np.isfinite(saved["state"]).all()
                checks.append(
                    dict(
                        name=tag + ":actual_final_state", passed=bool(shapes and finite)
                    )
                )
            audit = row.get("verification", {})
            if audit.get("verified"):
                thresholds = row["steady"]["residual"]
                checks.append(
                    dict(
                        name=tag + ":original_equation_targets",
                        passed=bool(
                            row["success"]
                            and audit["admissible"]
                            and audit["physical_branch_pass"]
                            and max(audit["equation_l2"])
                            <= thresholds["absolute_tolerance"]
                            and audit["boundary_max"]
                            <= thresholds["boundary_tolerance"]
                            and max(audit["global_defect"])
                            <= thresholds["global_tolerance"]
                            and max(audit["telescoping_identity_defect"]) < 1e-12
                        ),
                    )
                )
        elif row.get("success"):
            checks.append(dict(name=tag + ":missing_success_state", passed=False))
    sources = {
        r["source"]["identity"]: r["source"]
        for _, r in records
        if r.get("source", {}).get("identity")
    }
    production_hashes = {}
    for identity, source in sources.items():
        archive_path = ROOT / "outputs" / f"source_{identity}.zip"
        bad = []
        if archive_path.exists():
            with zipfile.ZipFile(archive_path) as archive:
                members = set(archive.namelist())
                for name, digest in source["files"].items():
                    normalized = name.replace("\\", "/")
                    member = normalized if normalized in members else name
                    if (
                        member not in members
                        or hashlib.sha256(archive.read(member)).hexdigest() != digest
                    ):
                        bad.append(name)
                    if normalized.startswith("src/"):
                        production_hashes.setdefault(normalized, set()).add(digest)
        checks.append(
            dict(
                name=f"source_archive:{identity}",
                passed=archive_path.exists() and not bad,
                detail=bad,
            )
        )
    changed_production = [
        name
        for name, hashes in production_hashes.items()
        if len(hashes) != 1
        or hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest()
        not in hashes
    ]
    checks.append(
        dict(
            name="same_current_production_source_across_all_attempts",
            passed=not changed_production,
            detail=changed_production,
        )
    )
    package_epochs = {
        json.dumps(s.get("packages"), sort_keys=True) for s in sources.values()
    }
    checks.append(
        dict(
            name="same_recorded_package_versions_across_all_attempts",
            passed=len(package_epochs) == 1,
        )
    )
    manifest = read(ROOT / "data/source_manifest.json")
    altered_inputs = [
        name
        for name, entry in manifest["files"].items()
        if hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
        != entry["sha256"]
    ]
    checks.append(
        dict(
            name="immutable_archived_input_hashes",
            passed=not altered_inputs,
            detail=altered_inputs,
        )
    )
    report = dict(
        case_method_count=len(coverage),
        attempted=sum(x["attempted"] for x in coverage),
        verified=sum(x["verified"] for x in coverage),
        coverage=coverage,
        checks=checks,
        source_epochs=len(sources),
        production_source_unchanged=not changed_production,
        package_epochs=len(package_epochs),
        mesh_coverage=mesh_coverage,
        mesh_attempted=sum(x["attempted"] for x in mesh_coverage),
        mesh_verified=sum(x["verified"] for x in mesh_coverage),
        evidence_checks_pass=all(c["passed"] for c in checks),
        full_paper_mesh_coverage=all(c["attempted"] for c in coverage + mesh_coverage),
        all_requested_roots_verified=all(
            c["verified"] for c in coverage + mesh_coverage
        ),
    )
    target = ROOT / "outputs/audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"Paper meshes: {report['attempted']}/{len(coverage)} attempted; {report['verified']}/{len(coverage)} independently verified"
    )
    print(f"Evidence checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
    print(
        f"Appendix mesh points: {report['mesh_attempted']}/{len(mesh_coverage)} attempted; {report['mesh_verified']}/{len(mesh_coverage)} verified"
    )
    if not report["evidence_checks_pass"] or not report["full_paper_mesh_coverage"]:
        raise SystemExit(1)
    if args.require_converged and not report["all_requested_roots_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
