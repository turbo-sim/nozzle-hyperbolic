"""Summarize saved steady refinement, branch and acceptance evidence; no solves."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .summarize import independently_converged


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _orders(points,field):
    result=[]
    for coarse,fine in zip(points,points[1:]):
        left,right=coarse.get(field),fine.get(field)
        if _finite(left) and _finite(right) and left>0 and right>0 and fine["cells"]>coarse["cells"]:
            result.append(dict(coarse=coarse["cells"],fine=fine["cells"],coarse_error=left,fine_error=right,
                               observed_order=math.log(left/right)/math.log(fine["cells"]/coarse["cells"]),
                               reduced=right<=left*(1+1e-10)))
    return result


def analyze(path):
    path=Path(path).resolve();campaign=json.loads(path.read_text(encoding="utf-8"));protocol=campaign["protocol"]
    rows=[]
    for index in campaign.get("results",[]):
        row=dict(index)
        if index.get("file"):
            filename=(path.parent/index["file"]).resolve()
            if not filename.is_relative_to(path.parent):raise ValueError("Unsafe scenario path")
            raw=filename.read_bytes()
            if index.get("scenario_sha256") and sha256(raw).hexdigest()!=index["scenario_sha256"]:
                raise ValueError("Scenario evidence changed: "+index["file"])
            row={**json.loads(raw),**index}
        rows.append(row)
    definitions={case["id"]:case for case in campaign.get("cases",[])}
    series=[]
    for case in protocol["cases"]:
      for reconstruction in protocol["reconstructions"]:
       for flux in protocol["fluxes"]:
        selected=sorted((row for row in rows if row.get("case_id")==case and row.get("reconstruction")==reconstruction and row.get("flux")==flux),key=lambda row:row["cells"])
        points=[]
        for row in selected:
            points.append(dict(cells=row["cells"],status=row.get("status"),success=row.get("success"),
                               original_residual_passed=independently_converged(row,protocol),branch_passed=row.get("branch_passed"),
                               accuracy_assessed=row.get("accuracy_assessed"),analytical_passed=row.get("analytical_passed"),
                               failed_accuracy_checks=[name for name,value in row.get("accuracy_checks",{}).items() if value is False],
                               conservative_max_normalized_l1=row.get("metrics",{}).get("max_conservative_normalized_l1"),
                               cp_l1=row.get("errors",{}).get("Cp",{}).get("l1"),
                               shock=row.get("shock"),work=row.get("work"),invalid_roe_faces=(row.get("roe_diagnostics") or {}).get("invalid_faces")))
        q_orders=_orders(points,"conservative_max_normalized_l1");cp_orders=_orders(points,"cp_l1")
        flags=[]
        if any(point["success"] is not True or point["original_residual_passed"] is not True for point in points):flags.append("nonlinear_or_original_residual_failure")
        if any(point["branch_passed"] is not True for point in points):flags.append("physical_branch_failure")
        if any(point["accuracy_assessed"] and point["analytical_passed"] is not True for point in points):flags.append("assessed_accuracy_failure")
        if any(not row["reduced"] and max(row["coarse_error"],row["fine_error"])>1e-12 for row in q_orders):flags.append("nonmonotone_conservative_error")
        if any(not row["reduced"] and max(row["coarse_error"],row["fine_error"])>1e-12 for row in cp_orders):flags.append("nonmonotone_cp_error")
        series.append(dict(case_id=case,reconstruction=reconstruction,flux=flux,
                           known_limit=definitions.get(case,{}).get("known_limit"),
                           missing_grids=sorted(set(protocol["grids"])-{point["cells"] for point in points}),
                           points=points,conservative_error_orders=q_orders,cp_error_orders=cp_orders,review_flags=flags))
    return dict(schema_version=1,scope="Saved-data numerical refinement and physical-branch context; no fresh residual/flow solve and no timing claim",
                input=str(path),completed=campaign.get("completed"),verification_only=campaign.get("verification_only"),
                source_sha256=campaign.get("provenance",{}).get("source",{}).get("sha256"),
                harness_sha256=campaign.get("provenance",{}).get("harness",{}).get("sha256"),
                records_retained=len(rows),series=series,
                interpretation="Orders use scalar maximum component normalized conservative L1 and Cp L1 where available. Dominant components and discontinuities can change observed order; flags invite inspection rather than proving a solver defect.")


def markdown(data):
    lines=["# Saved steady refinement and branch assessment","",data["interpretation"],"",
           f"Completed campaign: **{data['completed']}**; retained records: **{data['records_retained']}**. No performance timings are assessed.","",
           "| Case | Reconstruction | Grids | Conservative L1 orders | Cp L1 orders | Review flags |",
           "|---|---|---|---|---|---|"]
    for row in data["series"]:
        orders=lambda key:", ".join(f"{item['coarse']}->{item['fine']}: {item['observed_order']:.2f}" for item in row[key]) or "unavailable"
        flags=", ".join(row["review_flags"]) or ("incomplete" if row["missing_grids"] else "none")
        lines.append(f"| `{row['case_id']}` | `{row['reconstruction']}` | {', '.join(str(point['cells']) for point in row['points']) or 'not run'} | {orders('conservative_error_orders')} | {orders('cp_error_orders')} | {flags} |")
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path);args=parser.parse_args();data=analyze(args.input)
    output=args.output or args.input.with_name("refinement_audit.json");output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(data),encoding="utf-8")
    print("Saved refinement evidence:",len(data["series"]),"series;",data["records_retained"],"records;",sum(bool(row["review_flags"]) for row in data["series"]),"flagged series")


if __name__=="__main__":main()
