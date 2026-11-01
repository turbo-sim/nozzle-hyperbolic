"""Plot retained Sod fields/reference cell averages without solver/EOS calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def plot(data, source, directory):
    """Return a PNG/PDF pair from the finest saved four-method Sod result.

    This plotting conversion is specific to the recorded perfect-gas reference.
    Its affine energy relation is recovered from the two unchanged exact end
    states, including any enthalpy reference offset, rather than assuming gamma
    or evaluating thermodynamics. The reference curve represents conservative
    cell averages converted to primitive form, matching the numerical storage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [row for row in data.get("records", []) if row.get("success")
            and "final_state" in row and "exact_final_conservative_cell_averages" in row]
    if not rows:
        return None
    cells = max(row["cells"] for row in rows)
    rows = [row for row in rows if row["cells"]==cells]
    order = {name:i for i,name in enumerate(("euler_explicit","rk3","implicit_coupled","implicit_segregated"))}
    rows.sort(key=lambda row:order.get(row["method"]["id"],99))
    first = rows[0]
    q = np.asarray(first["exact_final_conservative_cell_averages"])
    physical = data["settings"]["physical"]
    left,right = physical["left"],physical["right"]
    internal_l = q[0,-1]/left["rho"]-.5*left["u"]**2
    internal_r = q[-1,-1]/right["rho"]-.5*right["u"]**2
    coefficient = (internal_l-internal_r)/(left["p"]/left["rho"]-right["p"]/right["rho"])
    energy_offset = internal_l-coefficient*left["p"]/left["rho"]
    gamma = 1+1/coefficient
    sound = (gamma*left["p"]/left["rho"])**.5
    rho = q[:,0]
    velocity = q[:,1]/rho
    pressure = (q[:,-1]-.5*rho*velocity**2-energy_offset*rho)/coefficient
    x = np.asarray(first["physical_cell_centers"])/physical["length"]
    references = (pressure/left["p"],rho/left["rho"],velocity/sound)
    colors = plt.get_cmap("plasma")(np.linspace(.2,.8,len(rows)))
    styles = ("-","--","-.",":")
    markers = (None,None,"o","s")
    labels = dict(euler_explicit="Explicit Euler",rk3="SSPRK3",
                  implicit_coupled="Backward Euler · coupled BC",
                  implicit_segregated="Backward Euler · segregated BC")
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":"#253248","text.color":"#253248",
                         "axes.edgecolor":"#8793a4","xtick.color":"#526078",
                         "ytick.color":"#526078","savefig.facecolor":"white"}):
        fig,axes = plt.subplots(1,3,figsize=(12.4,4.1),sharex=True)
        for axis,exact in zip(axes,references):
            axis.plot(x,exact,color="#1b2637",lw=1.7,label="Exact conservative cell averages",zorder=5)
        for index,(row,color) in enumerate(zip(rows,colors)):
            state = np.asarray(row["final_state"])[1:-1]
            p,u,h = state.T
            density = (coefficient+1)*p/(h-energy_offset)
            for axis,value in zip(axes,(p/left["p"],density/left["rho"],u/sound)):
                axis.plot(x,value,color=color,lw=1.55,ls=styles[index%len(styles)],
                    marker=markers[index%len(markers)],markevery=(index,13),ms=3.1,
                    markerfacecolor="none",label=labels.get(row["method"]["id"],row["method"]["id"]))
        for axis,label,title in zip(axes,("$p/p_L$",r"$\rho/\rho_L$","$u/a_L$"),
                                    ("Pressure","Density","Velocity")):
            axis.set(xlabel="$x/L$",ylabel=label,title=title,xlim=(0,1))
            axis.grid(alpha=.17,lw=.6)
        fig.suptitle(f"Roe Sod verification · {cells} cells · CFL 0.5 · t = {first['time']*1e3:.4f} ms",
                     fontsize=13,fontweight="semibold",y=.99)
        handles,legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc="lower center",ncol=3,frameon=False,
                   bbox_to_anchor=(.5,-.005),fontsize=9)
        fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.79,wspace=.29)
        directory.mkdir(parents=True,exist_ok=True)
        png,pdf = directory/"sod_final_profiles.png",directory/"sod_final_profiles.pdf"
        fig.savefig(png,dpi=190,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        plt.close(fig)
    (directory/"sod_final_profiles.json").write_text(json.dumps(dict(
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=cells,methods=[row["method"]["id"] for row in rows],
        reference="Saved128-point Gauss conservative cell averages converted to primitive form",
        conversion="Affine perfect-gas energy coefficient/reference offset recovered from saved exact end states",
        recovered_gamma=float(gamma),recovered_energy_offset=float(energy_offset),
        numerical_or_EOS_evaluations=False),indent=2)+"\n",encoding="utf-8")
    return png,pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    study = Path(__file__).resolve().parents[1]
    parser.add_argument("--input",type=Path,default=study/"outputs/roe_transient.json")
    parser.add_argument("--output-dir",type=Path,default=study/"outputs/figures")
    args = parser.parse_args()
    paths = plot(json.loads(args.input.read_text(encoding="utf-8")),args.input,args.output_dir)
    for path in paths or ():
        print(path)


if __name__=="__main__":
    main()
