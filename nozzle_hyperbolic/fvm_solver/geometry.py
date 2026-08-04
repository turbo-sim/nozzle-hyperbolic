"""
Geometry definitions for the 1D FVM solver.

Each function returns a Geometry object containing:
  A       — cell-centre cross-sectional areas  (N,)  [m²]
  dAdx    — analytical dA/dx                   (N,)  [m]
  D_h     — hydraulic diameter                 (N,)  [m]
  roughness — wall roughness                   scalar [m]

All functions accept (x, roughness=0.0, **kwargs) and return Geometry.
x must be a 1-D NumPy array of cell-centre positions.
"""

import jax
import nurbspy.jax as nrb
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple


class Geometry(NamedTuple):
    A:         np.ndarray  # cell-centre areas  (N,)  [m²]
    dAdx:      np.ndarray  # dA/dx              (N,)  [m]
    D_h:       np.ndarray  # hydraulic diameter (N,)  [m]
    roughness: float       # wall roughness           [m]


def get_pipe(x, roughness=0.0, radius=0.005):
    """Constant-area circular pipe."""
    x   = np.asarray(x, dtype=float)
    N   = len(x)
    A   = np.full(N, np.pi * radius**2)
    dAdx = np.zeros(N)
    D_h  = np.full(N, 2.0 * radius)
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_shock_tube(x, roughness=0.0, area=1.0):
    """Constant-area shock tube (circular equivalent diameter)."""
    x   = np.asarray(x, dtype=float)
    N   = len(x)
    A   = np.full(N, float(area))
    dAdx = np.zeros(N)
    D_h  = np.full(N, 2.0 * np.sqrt(float(area) / np.pi))
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_symmetric_nozzle(x, roughness=0.0, L=0.2, A_inlet=0.30, A_throat=0.15):
    """
    Symmetric parabolic converging-diverging nozzle.

    Area profile:  A(x) = A_inlet - 4*(A_inlet - A_throat) * (x/L) * (1 - x/L)
    Derivative:   dA/dx = (4*(A_inlet - A_throat) / L) * (2*x/L - 1)
    """
    x      = np.asarray(x, dtype=float)
    xi     = x / L
    A      = A_inlet - 4.0 * (A_inlet - A_throat) * xi * (1.0 - xi)
    dAdx   = (4.0 * (A_inlet - A_throat) / L) * (2.0 * xi - 1.0)
    D_h    = 2.0 * np.sqrt(A / np.pi)
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_lettieri_nozzle(x, roughness=0.0, file_path="Lettieri_nozzle.xlsx",
                         sheet_name="Sheet1", nozzle_type="axisymmetric",
                         width=6.48e-3):
    """
    Nozzle geometry from Excel spline fit (Lettieri et al.).

    nozzle_type : 'axisymmetric' or 'planar'
    width       : channel depth [m] (planar only)
    """
    import nozzle_hyperbolic as hyp
    x = np.asarray(x, dtype=float)

    _, _, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree=10)

    x_min = -0.03201
    r = np.array([spline(xi) if xi >= x_min else 0.006365 for xi in x])

    nozzle_type = str(nozzle_type).lower()
    if nozzle_type == "axisymmetric":
        A   = np.pi * r**2
        D_h = 2.0 * r
    elif nozzle_type == "planar":
        width = float(width)
        A   = 2.0 * r * width
        D_h = 4.0 * A / (2.0 * (width + 2.0 * r))
    else:
        raise ValueError(f"Unknown nozzle_type '{nozzle_type}'. "
                         f"Choose 'axisymmetric' or 'planar'.")

    # dAdx from central differences on the spline-derived A
    dx   = x[1] - x[0]
    dAdx = np.gradient(A, dx)

    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))

# Nakagawa geometry
def get_nozzle_nakagawa(
    x,
    roughness=0.0,
    L=83.50e-3,
    L_convergent=15.0e-3,
    height_in=5e-3,
    height_throat=0.2e-3,
    height_out=0.72e-3,
    width=1e-3,
    theta=0.48,
):
    """
    JAX-safe linear convergent–divergent nozzle (planar geometry).
    Uses jax.lax.cond (JAX's version of 'if') so only the active branch runs.
    Returns (A, dAdx, perimeter, height).
    """
    L_divergent = L - L_convergent
    theta_rad = jnp.deg2rad(theta)

    def convergent(x_):
        h = height_in + (height_throat - height_in) * x_ / L_convergent
        A = 2.0 * h * width
        dAdx = 2.0 * width * (height_throat - height_in) / L_convergent
        return A, dAdx, h

    def divergent(x_):
        l = x_ - L_convergent
        h = height_throat + l * jnp.tan(theta_rad)
        A = 2.0 * h * width
        dAdx = 2.0 * width * jnp.tan(theta_rad)
        return A, dAdx, h

    # The branch logic (lax.cond) is per-scalar; vmap it over the position array x.
    def _scalar(x_):
        A, dAdx, h = jax.lax.cond(x_ <= L_convergent, convergent, divergent, operand=x_)
        perimeter = 2.0 * (width + 2.0 * h)
        D_h = 4.0 * A / perimeter
        return A, dAdx, D_h

    A, dAdx, D_h = jax.vmap(_scalar)(jnp.asarray(x, dtype=jnp.float64))
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))

def get_nozzle_elliot(
    length,
    roughness=0.0,
    convergent_length=0.09940,
    divergent_length=0.16652,
    radius_in=0.0254778,
    radius_throat=0.00655,
    radius_out=0.01413,
    throat_first_length = 0.04223,
    radius_throat_in = 0.007952,
    throat_second_length = 0.011512,
    radius_throat_out = 0.0068617,
):
    """
    JAX-compatible version of get_nozzle_elliot.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants


    def region_convergent(length):
        # area = jnp.pi * (radius_in + ((radius_throat_in - radius_in) / convergent_length) * length) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        # jax.debug.print("{r}",r = length)
        radius = radius_in + (((radius_throat_in - radius_in) / convergent_length) * length)
        area = jnp.pi * radius**2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter *( (radius_throat_in - radius_in) / convergent_length)
        return area, area_slope, perimeter, radius

    def region_throat_1(length):
        l = length - convergent_length
        # print((radius_throat - radius_throat_in) )
        # area = jnp.pi * (radius_throat_in + ((radius_throat - radius_throat_in) / throat_first_length) * l) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        points2 = jnp.array([
            [0.0, 0.005, 0.04, 0.04223],
            [radius_throat_in, 0.006995, radius_throat, radius_throat]
        ])

        l = jnp.atleast_1d(l)
        nurbs2 = nrb.NurbsCurve(control_points=points2)
        u = nurbs2.reparametrize_by_coordinate()

        _, radius  = nurbs2.get_value(u(l))
        dxdu, drdu  = nurbs2.get_derivative(u(l), 1)

        radius = radius[0]

        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius

        drdx = drdu[0] / dxdu[0]
        area_slope = perimeter * drdx
        
        return area, area_slope, perimeter, radius

    def region_throat_2(length):
        l = length - convergent_length - throat_first_length
        # print((radius_throat_out - radius_throat))
        # area = jnp.pi * (radius_throat + ((radius_throat_out - radius_throat) / throat_second_length) * l) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat + (((radius_throat_out - radius_throat) / throat_second_length) * l)
        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * ((radius_throat_out - radius_throat) / throat_second_length)
        return area, area_slope, perimeter, radius

    def region_divergent(length):
        # print((radius_out - radius_throat_out))
        l = length - convergent_length - throat_first_length - throat_second_length
        area = jnp.pi * (radius_throat_out + ((radius_out - radius_throat_out) / divergent_length) * l) ** 2
        radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat_out + (((radius_out - radius_throat_out) / divergent_length) * l)
        perimeter = 2 * jnp.pi * radius
        area = jnp.pi * radius ** 2
        area_slope = perimeter * ((radius_out - radius_throat_out) / divergent_length)
        return area, area_slope, perimeter, radius

    funcs = [region_convergent, region_throat_1, region_throat_2, region_divergent]

    # Select the correct region via lax.switch, which needs a scalar index →
    # build the scalar core and vmap it over the position array.
    def _scalar(length):
        conds = [
            length >= convergent_length,
            length >= convergent_length + throat_first_length,
            length >= convergent_length + throat_first_length + throat_second_length,
        ]
        idx = jnp.sum(jnp.array(conds))  # 0→convergent, 1→throat_1, 2→throat_2, 3→divergent
        area, area_slope, perimeter, radius = jax.lax.switch(idx, funcs, length)
        return area, area_slope, radius

    area, area_slope, radius = jax.vmap(_scalar)(jnp.asarray(length, dtype=jnp.float64))
    return Geometry(A=area, dAdx=area_slope, D_h=radius*2, roughness=float(roughness))


def get_nozzle_smd_divergent_smooth(
    length,
    roughness=0.0,
    z_transition=0.089805825,
    r_const=0.010065,
    z_const_end=0.463106796,
    z_div_end=0.900106796,
    r_outlet=0.03685,
    blend_width=0.02,
):
    """
    Same geometry as get_nozzle_smd_divergent but with a C1-smooth cubic
    Hermite blend over [z_const_end, z_const_end + blend_width] so that
    dA/dx ramps smoothly from 0 to the divergent slope after the throat.
    The constant-radius section is preserved up to exactly z_const_end.
    """
    _breaks = jnp.array([
        0.0, 0.00970874, 0.01941748, 0.02961165, 0.04029126,
        0.04951456, 0.05970874, 0.07038835, 0.08058252, 0.08980583,
    ])
    _coeffs = jnp.array([
        [-5.15687535e+02,  2.36106419e+02,  2.13628410e+01,
          3.45126602e+02,  4.88402424e+02, -4.79989669e+02,
          3.25965960e+01, -2.52971271e+02,  7.58975837e+02],
        [-1.77635684e-15, -1.50200255e+01, -8.14313939e+00,
         -7.48980984e+00,  3.56764475e+00,  1.70816925e+01,
          2.40239639e+00,  3.44675339e+00, -4.28974607e+00],
        [ 1.44451489e-02, -1.31380343e-01, -3.56265442e-01,
         -5.15630447e-01, -5.57517648e-01, -3.67062596e-01,
         -1.68438384e-01, -1.05971734e-01, -1.14565348e-01],
        [ 3.35000000e-02,  3.31683170e-02,  3.06930690e-02,
          2.62376240e-02,  2.02970300e-02,  1.58415840e-02,
          1.33663370e-02,  1.18811880e-02,  1.08910890e-02],
    ])

    drdx_div = (r_outlet - r_const) / (z_div_end - z_const_end)
    x1_blend = z_const_end + blend_width

    def converging(length):
        n_seg = _coeffs.shape[1]
        idx = jnp.searchsorted(_breaks, length, side="right") - 1
        idx = jnp.clip(idx, 0, n_seg - 1)
        dx = length - _breaks[idx]
        c0, c1, c2, c3 = _coeffs[0, idx], _coeffs[1, idx], _coeffs[2, idx], _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def constant(length):
        radius = jnp.float64(r_const)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = jnp.float64(0.0)
        return area, area_slope, perimeter, radius

    def blend(length):
        # Cubic Hermite on [z_const_end, x1_blend]:
        #   at z_const_end: r = r_const, dr/dx = 0
        #   at x1_blend:    r = r_const + drdx_div * blend_width, dr/dx = drdx_div
        h = blend_width
        r0 = r_const
        r1 = r_const + drdx_div * blend_width
        t = (length - z_const_end) / h
        H00 =  2*t**3 - 3*t**2 + 1
        H01 = -2*t**3 + 3*t**2
        H11 =    t**3 -   t**2
        # dr0=0 so the H10 term vanishes
        radius = r0 * H00 + r1 * H01 + (drdx_div * h) * H11
        dH00 =  6*t**2 - 6*t
        dH01 = -6*t**2 + 6*t
        dH11 =  3*t**2 - 2*t
        drdx = (r0 * dH00 + r1 * dH01 + (drdx_div * h) * dH11) / h
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def diverging(length):
        radius = r_const + drdx_div * (length - z_const_end)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx_div
        return area, area_slope, perimeter, radius

    # lax.switch needs a scalar index → build the scalar core and vmap it over the array.
    def _scalar(length):
        conds = jnp.array([
            length >= z_transition,
            length >= z_const_end,
            length >= x1_blend,
        ], dtype=jnp.int32)
        idx = jnp.sum(conds)  # 0=converging, 1=constant, 2=blend, 3=diverging
        area, area_slope, perimeter, radius = jax.lax.switch(
            idx, [converging, constant, blend, diverging], length)
        return area, area_slope, radius

    area, area_slope, radius = jax.vmap(_scalar)(jnp.asarray(length, dtype=jnp.float64))
    return Geometry(A=area, dAdx=area_slope, D_h=radius*2, roughness=float(roughness))


def get_nozzle_simoneau_bspline(
    length,
    roughness=0.0,
    convergent_length=79.65e-3,
    divergent_length=220e-3,
    radius_in=11.26e-3,
    radius_throat=1.78e-3,
    radius_out=16.31e-3,
    throat_length=11.35e-3,
    bspline_in_convergent_length=30e-3,
):
    """
    Simoneau nozzle with a cubic B-spline raccordatura between the convergent
    section and the constant-area throat region.

    The linear convergent runs from x=0 to x = convergent_length - bspline_in_convergent_length.
    A cubic NURBS then transitions smoothly (C1) from that junction point to
    x = convergent_length + throat_length (throat exit), where radius = radius_throat
    and slope = 0.  The throat (minimum area, sonic point) is at the B-spline exit.

    Sections:
        0: Linear convergent,   x in [0, x_blend_start]
        1: Cubic B-spline,      x in [x_blend_start, x_throat_exit]
        2: Linear divergent,    x in [x_throat_exit, total_length]

    Note: bspline_in_convergent_length must be large enough so that the first
    B-spline control point (P1) stays above radius_throat.  With the default
    geometry, values >= ~10 mm are safe; 30 mm gives a comfortable margin.
    """

    x_blend_start = convergent_length - bspline_in_convergent_length
    x_throat_exit = convergent_length + throat_length

    # Convergent slope and radius at the blend start
    m0 = (radius_throat - radius_in) / convergent_length          # < 0  (converging)
    r_blend = radius_in + m0 * x_blend_start                      # r at x_blend_start

    # B-spline span in local coordinates (0 → delta_x)
    delta_x = x_throat_exit - x_blend_start

    # Cubic Bézier control-point radii for C1 continuity at both ends:
    #   At P0: tangent matches convergent slope  → P1 = P0 + (delta_x/3) * (1, m0)
    #   At P3: tangent is horizontal (slope = 0) → P2 has same r as P3
    r_P1 = r_blend + m0 * (delta_x / 3.0)   # always > radius_throat when bspline_in_convergent_length is large enough
    r_P2 = radius_throat                     # horizontal tangent into P3

    def region_convergent(length):
        radius = radius_in + m0 * length
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * m0
        return area, area_slope, perimeter, radius

    def region_bspline(length):
        l = jnp.atleast_1d(length - x_blend_start)
        points = jnp.array([
            [0.0,     delta_x / 3.0,  2.0 * delta_x / 3.0,  delta_x],
            [r_blend, r_P1,           r_P2,                  radius_throat],
        ])
        nurbs = nrb.NurbsCurve(control_points=points)
        u = nurbs.reparametrize_by_coordinate()
        _, radius_arr = nurbs.get_value(u(l))
        dxdu, drdu = nurbs.get_derivative(u(l), 1)
        radius = radius_arr[0]
        drdx = drdu[0] / dxdu[0]
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * drdx
        return area, area_slope, perimeter, radius

    def region_divergent(length):
        l = length - x_throat_exit
        dr_dx = (radius_out - radius_throat) / divergent_length
        radius = radius_throat + dr_dx * l
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * dr_dx
        return area, area_slope, perimeter, radius

    funcs = [region_convergent, region_bspline, region_divergent]

    # lax.switch needs a scalar index → build the scalar core and vmap it over the array.
    def _scalar(length):
        conds = [
            length >= x_blend_start,
            length >= x_throat_exit,
        ]
        idx = jnp.sum(jnp.array(conds))   # 0 → convergent, 1 → bspline, 2 → divergent
        area, area_slope, perimeter, radius = jax.lax.switch(idx, funcs, length)
        return area, area_slope, radius

    area, area_slope, radius = jax.vmap(_scalar)(jnp.asarray(length, dtype=jnp.float64))
    return Geometry(A=area, dAdx=area_slope, D_h=radius*2, roughness=float(roughness))


def get_southwest_nozzle(
    length,
    roughness=0.0,
    convergent_length=0.250,
    throat_length=0.044,
    divergent_length=0.300,
    radius_in=0.0333250,
    radius_throat=0.0101600,
    radius_out=0.0333250,
    cubic_on_area=False,
):
    """
    Converging-diverging nozzle of Table 1.
 
        convergent   L = 250 mm   D: 66.65 -> 20.32 mm   cubic
        throat       L =  44 mm   D  = 20.32 mm          constant
        divergent    L = 300 mm   D: 20.32 -> 66.65 mm   linear
 
    The cubic is the smoothstep S(s) = 3s^2 - 2s^3, the unique cubic matching
    the two end radii with zero slope at both ends.  Zero slope at the
    downstream end is what lets the contraction meet the constant-area throat
    without a kink.  The table does not say whether the cubic is laid on the
    diameter or on the area, so `cubic_on_area` switches between the two
    (static Python flag, resolved at trace time).
 
    Sections:
        0: cubic convergent,  x in [0, convergent_length]
        1: constant throat,   x in [convergent_length, +throat_length]
        2: linear divergent,  x in [throat exit, total_length]
 
    The throat -> divergent junction is C0 only: the straight wall starts at a
    finite slope, so dA/dx jumps from 0 to about 15 A there.  That is what the
    table prescribes.  If the reconstruction dislikes it, blend `drdx` over a
    few mm the way get_nozzle_smd_divergent_smooth does, rather than changing
    the profile itself.
    """
 
    x_throat_in = convergent_length
    x_throat_out = convergent_length + throat_length
    drdx_div = (radius_out - radius_throat) / divergent_length
 
    def region_convergent(length):
        s = jnp.clip(length / convergent_length, 0.0, 1.0)
        smoothstep = s * s * (3.0 - 2.0 * s)
        dsmoothstep = 6.0 * s * (1.0 - s) / convergent_length
        if cubic_on_area:
            radius = jnp.sqrt(
                radius_in**2 + (radius_throat**2 - radius_in**2) * smoothstep
            )
            drdx = 0.5 * (radius_throat**2 - radius_in**2) * dsmoothstep / radius
        else:
            radius = radius_in + (radius_throat - radius_in) * smoothstep
            drdx = (radius_throat - radius_in) * dsmoothstep
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * drdx
        return area, area_slope, perimeter, radius
 
    def region_throat(length):
        radius = jnp.float64(radius_throat)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = jnp.float64(0.0)
        return area, area_slope, perimeter, radius
 
    def region_divergent(length):
        l = length - x_throat_out
        radius = radius_throat + drdx_div * l
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * drdx_div
        return area, area_slope, perimeter, radius
 
    funcs = [region_convergent, region_throat, region_divergent]
 
    # lax.switch needs a scalar index → build the scalar core and vmap it over
    # the position array.
    def _scalar(length):
        conds = [
            length >= x_throat_in,
            length >= x_throat_out,
        ]
        idx = jnp.sum(jnp.array(conds))  # 0 → convergent, 1 → throat, 2 → divergent
        area, area_slope, perimeter, radius = jax.lax.switch(idx, funcs, length)
        return area, area_slope, radius
 
    area, area_slope, radius = jax.vmap(_scalar)(jnp.asarray(length, dtype=jnp.float64))
    return Geometry(A=area, dAdx=area_slope, D_h=radius * 2, roughness=float(roughness))
