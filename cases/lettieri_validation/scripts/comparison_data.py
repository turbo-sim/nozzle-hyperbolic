"""Immutable original evidence and common geometry/properties for both solvers."""

from contextlib import redirect_stdout,redirect_stderr
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from .data import ROOT,settings
from .run import make_mesh,BoundedBicubic,audit_property_domain

OUT=ROOT/"outputs/solver_comparison"


def baseline_unchanged():
    path=OUT/"baseline_manifest.json"
    if not path.exists():
        raise RuntimeError("Freeze the original Lettieri evidence before starting this campaign")
    manifest=json.loads(path.read_text(encoding="utf-8"))
    changed=[name for name,digest in manifest.items()
             if sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:raise RuntimeError(f"Original Lettieri input files changed: {changed}")
    return dict(passed=True,files=len(manifest))


def cached_fluid():
    """Load the original table; never generate or rewrite property evidence."""
    cfg=settings()["properties"]
    table=ROOT/"outputs/fluid_tables"/(cfg["table_name"]+".pkl")
    if not table.is_file():raise RuntimeError("The original cached property table is required")
    digest=sha256(table.read_bytes()).hexdigest()
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        source=jxp.FluidBicubic(**cfg,table_dir=str(table.parent))
    audit=audit_property_domain(source,settings()["property_domain"])
    domain=settings()["property_domain"]
    h,p=np.asarray(source.h_vals),np.exp(np.asarray(source.logP_vals))
    i0,i1=np.searchsorted(h,[domain["h_min"],domain["h_max"]])+[-1,1]
    j0,j1=np.searchsorted(p,[domain["p_min"],domain["p_max"]])+[-1,1]
    values=np.asarray(source.table["viscosity"]["value"])[i0:i1+1,j0:j1+1]
    coeff=np.asarray(source.table["viscosity"]["coeffs"])[i0:i1,j0:j1]
    if not (np.isfinite(values).all() and (values>0).all() and np.isfinite(coeff).all()):
        raise RuntimeError("The active viscosity interpolation stencils are invalid")
    audit["viscosity"]={"nodes_positive":True,"coefficients_finite":True,
                        "node_min_Pa_s":float(values.min()),"node_max_Pa_s":float(values.max())}
    if digest!=sha256(table.read_bytes()).hexdigest():raise RuntimeError("Property cache was modified")
    return BoundedBicubic(source,domain),dict(table_sha256=digest,domain_audit=audit,
        loaded_existing_table=True,preparation_log=stream.getvalue(),configuration=cfg)


class NozzleGeometry:
    """JAX evaluation of the original measured cubic, shifted to x=0 at inlet."""

    def __init__(self):
        mesh,self.depth=make_mesh(2)
        spline=mesh.geometry
        self.origin=float(spline.x[0])
        self.length=float(spline.x[-1]-spline.x[0])
        self.breaks=jnp.asarray(spline.x-self.origin)
        self.coefficients=jnp.asarray(spline.c)
        roots=spline.derivative().roots(extrapolate=False)
        candidates=np.r_[spline.x[0],roots[(roots>=spline.x[0])&(roots<=spline.x[-1])],spline.x[-1]]
        self.throat_x=float(candidates[np.argmin(spline(candidates))]-self.origin)
        self.throat_half_height=float(spline(self.origin+self.throat_x))

    def __call__(self,x,length=None):
        x=jnp.asarray(x)
        index=jnp.clip(jnp.searchsorted(self.breaks,x,side="right")-1,0,len(self.breaks)-2)
        t=x-self.breaks[index]
        a,b,c,d=(self.coefficients[k,index] for k in range(4))
        height=((a*t+b)*t+c)*t+d
        slope=(3*a*t+2*b)*t+c
        area=2*self.depth*height
        perimeter=2*(self.depth+2*height)
        return area,2*self.depth*slope,perimeter,height,4*area/perimeter

    def audit(self):
        mesh,_=make_mesh(96)
        area,derivative,perimeter,height,diameter=self(jnp.asarray(mesh.centers-self.origin))
        np.testing.assert_allclose(area,mesh.area_centers,rtol=3e-14,atol=1e-18)
        np.testing.assert_allclose(derivative,mesh.darea_centers,rtol=3e-13,atol=1e-17)
        return dict(passed=True,origin_m=self.origin,length_m=self.length,
                    geometric_throat_x_m=self.throat_x+self.origin,
                    area_max_difference_m2=float(np.max(abs(np.asarray(area)-mesh.area_centers))),
                    normal_area="A=2*b*y",wetted_perimeter="P=2*(b+2*y)",hydraulic_diameter="4*A/P")


def original_fvm(case,cells):
    folder=ROOT/"outputs"/f"{case}_N{cells}_stretch1"
    saved=np.load(folder/"solution.npz")
    row=json.loads((folder/"result.json").read_text(encoding="utf-8"))
    verification=json.loads((folder/"verification.json").read_text(encoding="utf-8"))
    keys=("p","u","rho","T","h","h0","M","quality","mdot")
    return dict(id=folder.name,cells=cells,x_m=saved["x_m"].tolist(),
        fields={key:saved[key].tolist() for key in keys},success=row["success"],reason=row["reason"],
        metrics=row["metrics"],history=row["history"],scheme=row["scheme"],
        initialization=row["initialization"],source_epoch=row["source"],saved_verification=verification,
        mass_flow=row["mass_flow"],source_file=str(folder.relative_to(ROOT)),
        elapsed_s=row["elapsed_s"],physics="inviscid")
