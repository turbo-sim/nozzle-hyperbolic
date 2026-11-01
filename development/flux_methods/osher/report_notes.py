"""Data-only findings for one configurable Osher flux; no numerical imports."""
CONFIGURATIONS=("gauss_3","gauss_5","gauss_9","sonic_partitioned_5")
CASES=("euler_annular/shock35","euler_annular/shock70","euler_nozzle/shock_35","euler_nozzle/shock_70")


def make_notes(records,audits,originals,profile,helpers):
    derived=bool(records) and all(r["record_type"]=="api_migration_view" for r in records)
    scope=("Derived configuration view of preserved numeric evidence; not new cold solves" if derived
           else "Native cold-solve records with separate saved-state Operator evaluations")
    findings={"verified":False,"statements":[],"scope":scope}
    expected={(case,n,order,config) for case in CASES for n in (50,100,200,400)
              for order in ("constant","muscl_van_leer") for config in CONFIGURATIONS}
    actual={(r["case"],r["cells"],r["reconstruction"],r["variant"]) for r in records}
    validated=helpers["audit_complete"] and len(records)==128 and actual==expected and all(r["success"] and r["alignment"]==0 and helpers["valid_audit"](r,audits) for r in records)
    if validated:
        peaks=sum(r["profile_metrics"]["pressure_local_peak_pct"]>0 for r in records)
        branch=max(r["profile_metrics"]["branch_excess_pct"] for r in records)
        enthalpy=100*max(r["profile_metrics"]["h0_relative_max"] for r in records)
        groups={}
        for r in records:groups.setdefault((r["case"],r["reconstruction"],r["variant"]),[]).append(r)
        end=adjacent=improves=0
        for group in groups.values():
            group.sort(key=lambda r:r["cells"])
            values=[r["errors"]["errors.cell.p.l1"] for r in group]
            end+=values[-1]<values[0]
            improves+=sum(b<a for a,b in zip(values,values[1:]))
            adjacent+=len(values)-1
        statements=[
            ("All 128 preserved solve records pass current unified-API Operator, branch, cold-state and numeric-origin checks. This is a configuration metadata view, not 128 new cold solves." if derived else
             "All 128 native cold solves and their saved-state Operator, branch, cold-state and execution-identity checks pass."),
            f"Immediate-neighbor pressure peaks occur in {peaks}/128 profiles. Branch excess still reaches {branch:.4g}% and maximum total-enthalpy defect {enthalpy:.4g}%; these are distinct diagnostics.",
            f"Pressure conservative-cell L1 improves N50 to N400 in {end}/{len(groups)} series, but only {improves}/{adjacent} adjacent refinements improve. No uniform monotone-refinement or global-order claim follows.",
            "More quadrature points do not uniformly improve solution accuracy. Points per path or per captured panel are not spatial order, and nonlinear work is not a timing ranking."]
        details=[]
        for order in ("constant","muscl_van_leer"):
            selected=[next(r for r in records if r["case"]=="euler_annular/shock70" and r["cells"]==400 and r["reconstruction"]==order and r["variant"]==v) for v in CONFIGURATIONS]
            details.append(f"N400 annular shock70 ({order}) maximum relative total-enthalpy defects for Gauss3/Gauss5/Gauss9/sonic-partitioned5: "+"/".join(f"{100*r['profile_metrics']['h0_relative_max']:.4g}%" for r in selected)+".")
        totals=[]
        for config in CONFIGURATIONS:
            rows=[r for r in records if r["variant"]==config]
            totals.append(profile["labels"][config]+" "+"/".join(str(sum(r["work"].get(key,0) for r in rows)) for key in ("accepted_steps","rejected_attempts","newton_corrections")))
        details.append("Accepted/rejected/Newton totals across 32 matched configurations: "+"; ".join(totals)+". These are nonlinear-work counts, not a timing ranking.")
        findings.update(verified=True,statements=statements,accuracy_details=details)
    scope=("Historical execution source hashes and origin hashes remain separate from current unified-API evaluation hashes. Numeric states, references, errors, histories and work are preserved; metadata migration and current Operator checks do not constitute new cold solves. Historical raw JSON is archive-only." if derived else
           "Native execution identities and saved-state evaluation identities are recorded separately. The renderer imports no solver and advances no flow; analytical accuracy remains distinct from original-equation convergence.")
    return {"findings":findings,"epoch_recovery":{"verified":False,"statement":scope}}
