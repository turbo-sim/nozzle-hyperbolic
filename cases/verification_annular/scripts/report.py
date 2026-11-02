"""Build offline HTML and exportable scientific figures from saved results only."""

import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

STUDY=Path(__file__).resolve().parents[1]
OUT=STUDY/"outputs"
COLORS=plt.get_cmap("plasma")(np.linspace(.2,.8,5))
LABELS=dict(inviscid="Inviscid",constant_wall="Constant wall friction",aungier="Aungier",
            aungier_cooled="Aungier, cooled wall",aungier_heated="Aungier, heated wall")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image(fig,name):
    folder=OUT/"figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/f"{name}.png",dpi=165,bbox_inches="tight")
    fig.savefig(folder/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)
    data=base64.b64encode((folder/f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{name.replace("_"," ")}"></figure>'


def number(v):
    return f"{v:.3g}"


def export_figures():
    result=read(OUT/"results.json")
    names=list(result["variants"])
    records={name:[read(OUT/name/f"hyperbolic_n{row['cells']}.json") for row in
                   result["variants"][name]["grids"]] for name in names}
    refs={name:read(OUT/name/f"nozzlex_{result['config']['numerics']['reference_tolerances'][-1]:.0e}.json")
          ["fields"] for name in names}
    finest={name:rows[-1] for name,rows in records.items()}
    first=refs[names[0]]
    plt.rcdefaults()
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans",
                         "axes.spines.top":False,"axes.spines.right":False})

    fig,(ax,area)=plt.subplots(1,2,figsize=(11,3.6),layout="constrained")
    for wall in ("hub","shroud"):
        ax.plot(first[f"z_{wall}"],first[f"r_{wall}"],color="#192636",lw=2)
    ax.plot(first["z"],first["r"],"--",color=COLORS[2],label="Imported cubic midline")
    ax.set(xlabel="Axial coordinate z [m]",ylabel="Radius r [m]",title="Radial–axial bend",aspect="equal")
    ax.legend(fontsize=8,loc="center left",bbox_to_anchor=(1.07,.5),frameon=False)
    area.plot(np.asarray(first["m"])*1e3,np.asarray(first["A"])/first["A"][0],color=COLORS[2])
    area.set(xlabel="Meridional arclength m [mm]",ylabel="A / A inlet",title="Normal flow area; b = 15 mm")
    geometry_image=image(fig,"geometry")

    fig,axes=plt.subplots(2,3,figsize=(13,7.2),layout="constrained")
    fields=[("p","Static pressure [kPa]",.001),("v_m","Meridional velocity [m/s]",1),
            ("v_theta","Tangential velocity [m/s]",1),("M_m","Meridional Mach number",1),
            ("T","Static temperature [K]",1),("h0","Stagnation enthalpy [kJ/kg]",.001)]
    refnames=dict(v_theta="v_t",M_m="Ma_m")
    for ax,(key,label,factor) in zip(axes.flat,fields):
        for color,name in zip(COLORS,names):
            ref=refs[name]
            row=finest[name]
            ax.plot(np.asarray(ref["m"])*1e3,np.asarray(ref[refnames.get(key,key)])*factor,
                    color=color,lw=1.7,label=LABELS[name])
            ax.plot(np.asarray(row["m"])*1e3,np.asarray(row["fields"][key])*factor,"--",
                    color=color,lw=1,marker="o",markevery=max(1,row["cells"]//14),ms=2)
        ax.set(xlabel="Meridional arclength [mm]",ylabel=label)
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    axes[0,1].set_title("Solid: NozzleX ODE; dashed/markers: finest FVM",fontsize=10)
    profiles=image(fig,"solution_comparison")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for color,name in zip(COLORS,names):
        rows=records[name]
        n=[r["cells"] for r in rows]
        error=[max(r["errors"]["conservative_normalized_l1"]) for r in rows]
        axes[0].loglog(n,error,"o-",color=color,label=LABELS[name])
        history=[h for h in rows[-1]["history"] if "equation_norm" in h]
        axes[1].semilogy([h["step"] for h in history],
                          [max(h["equation_norm"]) for h in history],color=color)
    axes[0].axhline(1e-3,color="gray",ls=":",label="Conservative error target")
    axes[1].axhline(1e-9,color="gray",ls=":",label="Absolute residual target")
    axes[0].set(xlabel="Number of cells",ylabel="Max component of volume-weighted normalized L1 error",
                title="Refinement against point-sampled ODE states")
    axes[1].set(xlabel="Accepted pseudo-time update",ylabel="Max fixed-scale equation RMS",
                title="Finest-grid original spatial residual")
    axes[0].legend(fontsize=8)
    convergence=image(fig,"refinement_and_residuals")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for color,name in zip(COLORS,names):
        row=finest[name]
        x=np.asarray(row["m"])*1e3
        axes[1].plot(x,np.broadcast_to(row["wall_loads"]["q_w"],x.shape)*.001,color=color,label=LABELS[name])
    aungier="aungier" if "aungier" in names else names[0]
    row=finest[aungier]
    for color,key,label in zip(COLORS,("cf_wall","cf_diffusion","cf_curvature"),
                              ("Wall friction","Diffusion loss","Curvature loss")):
        values=np.broadcast_to(row["wall_loads"][key],(row["cells"],))
        axes[0].plot(np.asarray(row["m"])*1e3,values,color=color,label=label)
    axes[0].set(xlabel="Meridional arclength [mm]",ylabel="Fanning / empirical coefficient",
                title=f"Imported closure components: {LABELS[aungier]}")
    axes[1].set(xlabel="Meridional arclength [mm]",ylabel="Heat flux into fluid [kW/m²]")
    for ax in axes: ax.legend(fontsize=8)
    closures=image(fig,"closure_loads")

    # Project the 1D field uniformly across each section; no transverse resolution is implied.
    fig,axes=plt.subplots(1,len(names),figsize=(3*len(names),4.2),layout="constrained",squeeze=False)
    vmin=min(min(refs[n]["Ma_m"]) for n in names)
    vmax=max(max(refs[n]["Ma_m"]) for n in names)
    for ax,name in zip(axes.flat,names):
        ref=refs[name]
        z0,r0,z1,r1=[np.asarray(ref[k]) for k in ("z_hub","r_hub","z_shroud","r_shroud")]
        polygons=[[(z0[i],r0[i]),(z0[i+1],r0[i+1]),(z1[i+1],r1[i+1]),(z1[i],r1[i])]
                  for i in range(len(z0)-1)]
        collection=PolyCollection(polygons,cmap="plasma",edgecolors="face",rasterized=True)
        collection.set_array((np.asarray(ref["Ma_m"])[1:]+np.asarray(ref["Ma_m"])[:-1])/2)
        collection.set_clim(vmin,vmax)
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set(aspect="equal",xlabel="Axial coordinate [m]",ylabel="Radius [m]",title=LABELS[name])
    fig.colorbar(collection,ax=axes.ravel().tolist(),label="Meridional Mach number",shrink=.85)
    contour=image(fig,"duct_rendering")

    summary=dict(all_solved=result["all_solved"],
        all_finest_accuracy_passed=all(max(row["errors"]["conservative_normalized_l1"])<1e-3 for row in finest.values()),
        finest_errors={name:max(row["errors"]["conservative_normalized_l1"]) for name,row in finest.items()},
        max_discrete_budget_defect=max(max(abs(np.asarray(row["budget_normalized"]))) for row in finest.values()),
        max_geometry_radius_difference_m=result["geometry_audit"]["r"],
        sources_changed_during_run=result.get("sources_changed_during_run",[]))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Exported PNG/PDF figures from saved data",flush=True)


def report_payload():
    """Lossless presentation projection of existing runs; never import a solver."""
    results=read(OUT/"results.json")
    variants=[]
    input_hashes={"outputs/results.json":sha256((OUT/"results.json").read_bytes()).hexdigest()}
    fields=dict(p="p",rho="d",v_m="v_m",v_theta="v_t",T="T",M_m="Ma_m",
                h="h",h0="h0",s="s",alpha_deg="alpha",M="Ma")
    geometry_keys=("z","r","z_hub","r_hub","z_shroud","r_shroud","A","b")
    palette=["#6a00a8","#9b179e","#cc4778","#ed7953","#fca636"]
    for color,(name,variant) in zip(palette,results["variants"].items()):
        reference_path=OUT/name/f"nozzlex_{results['config']['numerics']['reference_tolerances'][-1]:.0e}.json"
        reference=read(reference_path)["fields"]
        input_hashes[reference_path.relative_to(STUDY).as_posix()]=sha256(reference_path.read_bytes()).hexdigest()
        reference_fields={key:reference[source] for key,source in fields.items()}
        reference_fields["mdot"]=(np.asarray(reference["d"])*reference["v_m"]*reference["A"]).tolist()
        runs=[]
        for row in variant["grids"]:
            path=OUT/name/f"hyperbolic_n{row['cells']}.json"
            run=read(path)
            input_hashes[path.relative_to(STUDY).as_posix()]=sha256(path.read_bytes()).hexdigest()
            # Include the actual saved method; this does not invent other methods.
            method=sha256(json.dumps(run["scheme"],sort_keys=True).encode()).hexdigest()[:12]
            run["method_id"]=method
            scheme=run["scheme"]
            run["method_label"]=" · ".join((scheme["flux"].upper(),
                {"muscl_van_leer":"MUSCL van Leer","constant":"Constant"}.get(
                    scheme["reconstruction"],scheme["reconstruction"]),
                {"euler-implicit":"Implicit Euler","runge-kutta-3":"RK3"}.get(
                    scheme["time_integrator"],scheme["time_integrator"])))
            # Full values, matched references and histories remain available offline.
            runs.append({key:run[key] for key in ("cells","method_id","method_label","scheme",
                "fields","reference_fields","m","errors","metrics","history","initialization",
                "success","reason","wall_loads","elapsed_compile_inclusive_s","budget_normalized")})
        variants.append(dict(id=name,label=LABELS[name],color=color,runs=runs,
            reference=dict(m=reference["m"],fields=reference_fields,
                           geometry={k:reference[k] for k in geometry_keys}),
            reference_audits=variant["references"],
            reference_last_change=variant["reference_last_refinement_change"]))
    assets=Path(__file__).parent/"report_assets"
    vendor=STUDY.parents[1]/"development/flux_methods/shared/assets/vendor"
    data=dict(variants=variants,summary=read(OUT/"summary.json"),config=results["config"],
              geometry_audit=results["geometry_audit"],
              presentation=dict(numerical_cases_rerun=False,input_sha256=input_hashes,
                 plotly_version="2.35.2",external_assets=False,
                 plotly_sha256=sha256((vendor/"plotly-2.35.2.min.js.gz").read_bytes()).hexdigest(),
                 generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 asset_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}))
    return data


def build_report():
    """Build the interactive Plotly report without changing saved CFD evidence."""
    if not (OUT/"figures/duct_rendering.png").exists():
        export_figures()
    data=report_payload()
    assets=Path(__file__).parent/"report_assets"
    vendor=STUDY.parents[1]/"development/flux_methods/shared/assets/vendor"
    document=(assets/"template.html").read_text(encoding="utf-8")
    raw=json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    raw=raw.replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e")
    raw=raw.replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    plotly=gzip.decompress((vendor/"plotly-2.35.2.min.js.gz").read_bytes()).decode("utf-8")
    replacements={"STYLE":(assets/"style.css").read_text(encoding="utf-8"),
                  "PLOTLY":plotly.replace("</script","<\\/script"),"DATA":raw,
                  "APP":(assets/"app.js").read_text(encoding="utf-8"),
                  "DUCT_IMAGE":base64.b64encode((OUT/"figures/duct_rendering.png").read_bytes()).decode(),
                  "EVIDENCE":base64.b64encode((OUT/"results.json").read_bytes()).decode(),
                  "LICENSE":base64.b64encode((vendor/"PLOTLY_LICENSE.txt").read_bytes()).decode()}
    for key,value in replacements.items():
        document=document.replace(f"<!--{key}-->",value)
    (STUDY/"REPORT.html").write_text(document,encoding="utf-8",newline="\n")
    presentation=dict(data["presentation"],html_sha256=sha256(document.encode()).hexdigest(),
                      embedded_json_sha256=sha256(raw.encode()).hexdigest(),
                      bytes=len(document.encode()))
    (OUT/"presentation_manifest.json").write_text(json.dumps(presentation,indent=2),encoding="utf-8")
    print(f"Plotly report: {STUDY/'REPORT.html'} ({len(document.encode()):,} bytes)",flush=True)


if __name__=="__main__":build_report()
