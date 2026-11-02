"""Common physical geometry from NozzleX's canonical cubic midline.

The installed NURBSpy JAX arclength builder requires an unavailable Quadax
DirectAdjoint API. A fixed host arclength inverse supplies an audited Hermite
map to NozzleX's existing geometry evaluator. No dependency is patched.
"""

import jax
import jax.numpy as jnp
import numpy as np
import nurbspy.numpy as nrb
from dataclasses import replace
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from nozzlex.vaneless_channel.vaneless_channel_model import (
    make_vaneless_channel_midline, make_vaneless_channel_geometry,
)
from nozzle_hyperbolic.geometry.annular import make_annular_geometry, constant_height, AnnularMetrics


def shared_geometry(channel, map_points):
    source = make_vaneless_channel_midline(channel.geometry)
    host_curve = nrb.NurbsCurve(np.asarray(source.P), weights=np.asarray(source.W),
                                degree=source.p, knots=np.asarray(source.U))
    if float(channel.geometry.b_in) != float(channel.geometry.b_out):
        raise ValueError("This verification adapter currently requires constant height")
    physical = make_annular_geometry(host_curve, constant_height(float(channel.geometry.b_in)))
    distances = np.linspace(0,physical.length,map_points)
    # Integrate the inverse only once; thousands of repeated scalar quadratures
    # in mesh preprocessing otherwise dominate this small CFD study.
    def inverse_rate(m,u):
        # Endpoint rounding in this numerical coordinate solve is audited below.
        bounded=np.clip(u,0.0,1.0)
        return 1/np.linalg.norm(host_curve.get_derivative(bounded,order=1),axis=0)
    mapping=solve_ivp(inverse_rate,(0,physical.length),[0.0],method="DOP853",
                      rtol=2e-13,atol=2e-15,t_eval=distances)
    if not mapping.success or abs(mapping.y[0,-1]-1)>1e-10:
        raise RuntimeError("Fixed arclength inverse integration failed")
    parameters=mapping.y[0]
    parameters[0],parameters[-1]=0,1
    derivatives = 1/np.linalg.norm(host_curve.get_derivative(parameters,order=1),axis=0)
    table, values, slopes = map(jnp.asarray,(distances,parameters,derivatives))

    def inverse(m):
        index = jnp.clip(jnp.searchsorted(table,m,side="right")-1,0,len(table)-2)
        delta = table[index+1]-table[index]
        t = (m-table[index])/delta
        return ((2*t**3-3*t**2+1)*values[index]
                +(t**3-2*t**2+t)*delta*slopes[index]
                +(-2*t**3+3*t**2)*values[index+1]
                +(t**3-t**2)*delta*slopes[index+1])

    handle = make_vaneless_channel_geometry(channel.geometry,
        arclength_map=(inverse,jnp.asarray(physical.length)))
    probes = np.linspace(0,physical.length,103)
    host = physical.evaluate(probes)
    other = jax.device_get(handle(jnp.asarray(probes)))
    audit = {key:float(np.max(np.abs(np.asarray(other[key])-expected)))
             for key,expected in dict(r=host.r,z=host.x,b=host.b,A=host.A,
                                    dAds=host.dA_dm,curvature=abs(host.curvature)).items()}
    audit["map_points"] = map_points
    audit["length_m"] = physical.length
    audit["passed"] = (audit["r"] < 1e-9 and audit["z"] < 1e-9
                       and audit["dAds"] < 1e-9 and audit["curvature"] < 1e-6)
    if not audit["passed"]:
        raise RuntimeError(f"Geometry mapping discrepancy: {audit}")
    numpy_inverse=CubicHermiteSpline(distances,parameters,derivatives,extrapolate=False)
    height=float(channel.geometry.b_in)

    def evaluate_fast(m):
        shape=m.shape
        u=numpy_inverse(m.reshape(-1))
        position=host_curve.get_value(u)
        derivative=host_curve.get_derivative(u,order=1)
        second=host_curve.get_derivative(u,order=2)
        speed=np.linalg.norm(derivative,axis=0)
        xp,rp=derivative/speed
        x,r=position
        b=np.full_like(r,height)
        kappa=(derivative[0]*second[1]-derivative[1]*second[0])/speed**3
        values=(x,r,b,2*np.pi*r*b,xp,rp,np.zeros_like(r),2*np.pi*rp*b,
                np.arctan2(rp,xp),kappa)
        return AnnularMetrics(*(value.reshape(shape) for value in values))

    fast=replace(physical,xi_of_m=numpy_inverse,_evaluate=evaluate_fast)
    return fast,handle,audit
