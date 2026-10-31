"""Exact special-case transient annular references and their cell projections.

The compact translating contact and the constant wall-source trajectory below
were derived for this package by substitution in the annular Euler balances.
No external publication is claimed as the source of either constructed test
problem. The derivations and restrictions are stated beside each function.
These host-side verification routines do not call production state maps,
numerical fluxes, residuals, or time integrators.
"""

import jaxprop as jxp
import numpy as np


def exact_homogeneous(
    time,
    fluid,
    *,
    density,
    temperature,
    v_m,
    v_theta,
    height,
    radius,
    tau_w,
    q_w,
):
    """Return a homogeneous wall-source trajectory before the flow stops.

    This package-derived solution applies in a uniform, constant-radius
    annulus, with spatial gradients suppressed, fixed density ``rho``,
    prescribed wall-shear magnitude ``tau_w >= 0`` opposing velocity, and
    prescribed heat flux ``q_w`` positive *into* the fluid. Both walls contribute,
    so the wall-area/volume ratio is ``2/height``. The velocity direction is
    fixed. Integrating the momentum and total-energy ODEs gives::

        V(t) = V(0) - 2*tau_w*t/(rho*height)
        E(t) = E(0) + 2*q_w*t/(rho*height)

    Wall shear here converts resolved kinetic energy into internal energy;
    stationary walls do not add a total-energy source. This is a reference
    for the package's prescribed-source equations, not a friction-factor law.
    The solution ends at ``V=0`` and does not describe reversal after stopping.

    ``fluid`` must be calorically perfect with constant ``gamma`` and ``R``.
    Pressure and thermal energy use that explicit analytic EOS. Its enthalpy
    datum is obtained from ``get_state(PT_INPUTS, p, T)`` and retained in both
    ``h`` and ``E``. This fixes the old verification helper's implicit zero
    enthalpy-datum assumption. Backend agreement is checked at the initial and
    final states, rather than silently applying these formulas to a real gas.

    All inputs are SI scalars; ``time`` is seconds. Returns
    ``(q, W, T)`` with conserved density
    ``q=(rho,rho*v_m,rho*ell,rho*E)``, primitive
    ``W=(p,v_m,ell,h)``, ``ell=radius*v_theta``, and temperature in kelvin.
    """
    scalars = (time, density, temperature, v_m, v_theta, height, radius, tau_w, q_w)
    if not np.isfinite(scalars).all():
        raise ValueError("Homogeneous-reference inputs must be finite")
    if min(density, temperature, height, radius) <= 0 or min(time, tau_w) < 0:
        raise ValueError(
            "Positive density/temperature/geometry and nonnegative time/shear are required"
        )
    try:
        gas_constant = float(fluid.constants.R)
        gamma = float(fluid.constants.gamma)
    except AttributeError as error:
        raise ValueError(
            "The homogeneous reference requires perfect-gas constants"
        ) from error
    if not np.isfinite((gamma, gas_constant)).all() or gamma <= 1 or gas_constant <= 0:
        raise ValueError("Perfect-gas constants require finite gamma > 1 and R > 0")
    cv = gas_constant / (gamma - 1)
    cp = cv + gas_constant
    pressure_initial = density * gas_constant * temperature
    initial = fluid.get_state(jxp.PT_INPUTS, pressure_initial, temperature)
    if not np.isclose(float(initial["rho"]), density, rtol=2e-9, atol=1e-12):
        raise ValueError("The fluid backend is not the reference's perfect gas")
    h_shift = float(initial["h"]) - cp * temperature

    initial_speed = np.hypot(v_m, v_theta)
    if initial_speed == 0 and tau_w != 0:
        raise ValueError("A nonzero shear direction requires a nonzero initial speed")
    speed = initial_speed - 2 * tau_w * time / (density * height)
    if speed < 0:
        raise ValueError("The analytic shear solution is used only before stopping")
    direction = (
        np.array([v_m, v_theta]) / initial_speed if initial_speed else np.zeros(2)
    )
    meridional, tangential = speed * direction
    total_energy = (
        cv * temperature
        + h_shift
        + 0.5 * initial_speed**2
        + 2 * q_w * time / (density * height)
    )
    internal_energy = total_energy - 0.5 * speed**2
    temp = (internal_energy - h_shift) / cv
    if temp <= 0:
        raise ValueError("The homogeneous reference reached nonpositive temperature")
    pressure = density * gas_constant * temp
    backend = fluid.get_state(jxp.PT_INPUTS, pressure, temp)
    if not np.isclose(
        float(backend["rho"]), density, rtol=2e-9, atol=1e-12
    ) or not np.isclose(float(backend["h"]), cp * temp + h_shift, rtol=2e-9, atol=1e-7):
        raise ValueError(
            "The fluid backend is not the reference's calorically perfect gas"
        )
    conservative = density * np.array(
        [1.0, meridional, radius * tangential, total_energy]
    )
    primitive = np.array(
        [pressure, meridional, radius * tangential, cp * temp + h_shift]
    )
    return conservative, primitive, temp


def density_pulse(m, time, density_base, settings):
    """Sample a package-constructed smooth contact advected at fixed velocity.

    Define ``xi=(m-center-v_m*time)/half_width`` and the compact smooth bump
    ``b=exp(1-1/(1-xi**2))`` for ``abs(xi)<1``, zero outside. Then
    ``rho=density_base*(1+density_amplitude*b)``. With uniform pressure,
    ``v_m`` and tangential velocity in a constant-radius/area annulus, this
    translated density profile is an exact contact solution of the inviscid
    Euler equations. Its shape is chosen here, not taken from a publication.
    ``settings`` stores the four named parameters and velocity in SI units.
    """
    coordinate = (
        np.asarray(m) - settings["center"] - settings["v_m"] * time
    ) / settings["half_width"]
    bump = np.zeros_like(coordinate)
    active = np.abs(coordinate) < 1.0
    bump[active] = np.exp(1.0 - 1.0 / (1.0 - coordinate[active] ** 2))
    return density_base * (1.0 + settings["density_amplitude"] * bump)


def cell_reference(fluid, mesh, time, density_base, settings, order=32):
    """Conservatively project the translating contact into uniform-annulus cells.

    Integrates ``q*A dm`` using ``order``-point Gauss-Legendre quadrature on
    each panel, split at both pulse edges. This numerical quadrature evaluates
    the analytic profile; it is independent of the FVM discretization. The
    required settings additionally contain ``radius``, ``width``, ``pressure``
    and ``v_theta``. The physical mesh must have the matching fixed radius and
    constant area ``A=2*pi*radius*width``. SI units are used throughout.

    ``get_state(DmassP_INPUTS, rho, pressure)`` supplies enthalpy. The exact
    pointwise contact exists for a general EOS, but the returned equivalent
    cell state retains constant pressure only if ``rho*h(rho,p)-p`` is affine
    in ``rho`` at fixed pressure. This holds for a calorically perfect gas,
    including an enthalpy offset; it need not hold for a general EOS. The
    energy reconstruction is checked and an incompatible EOS is rejected.

    Returns primitive ``W``, conserved density ``q``, integrated conserved
    amounts, projected ``fields``, and the quadrature metadata. ``W`` is
    equivalent to the conservative average, not a primitive volume average.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    center = settings["center"] + settings["v_m"] * time
    edges = (center - settings["half_width"], center + settings["half_width"])
    coordinates, quadrature_weights, owners = [], [], []
    area = 2.0 * np.pi * settings["radius"] * settings["width"]
    if not np.allclose(
        mesh.volumes, area * np.diff(mesh.m_faces), rtol=2e-10, atol=1e-15
    ):
        raise ValueError(
            "The translating contact requires a matching constant-area mesh"
        )
    if not np.allclose(mesh.r_center, settings["radius"], rtol=2e-10, atol=1e-14):
        raise ValueError(
            "The translating contact requires a matching constant-radius mesh"
        )
    for index, (lower, upper) in enumerate(zip(mesh.m_faces[:-1], mesh.m_faces[1:])):
        splits = [lower, *(edge for edge in edges if lower < edge < upper), upper]
        for a, b in zip(splits[:-1], splits[1:]):
            coordinates.extend(0.5 * (a + b) + 0.5 * (b - a) * nodes)
            quadrature_weights.extend(area * 0.5 * (b - a) * weights)
            owners.extend([index] * order)
    coordinates, quadrature_weights = map(np.asarray, (coordinates, quadrature_weights))
    owners = np.asarray(owners, dtype=int)
    density = density_pulse(coordinates, time, density_base, settings)
    pressure = np.full_like(density, settings["pressure"])
    state = fluid.get_state(jxp.DmassP_INPUTS, density, pressure)
    enthalpy = np.asarray(state["h"])
    v_m, v_theta = settings["v_m"], settings["v_theta"]
    ell = settings["radius"] * v_theta
    kinetic = 0.5 * (v_m**2 + v_theta**2)
    values = np.stack(
        (
            density,
            density * v_m,
            density * ell,
            density * (enthalpy + kinetic) - pressure,
        ),
        axis=-1,
    )
    integrated = np.zeros((mesh.n_cells, 4))
    np.add.at(integrated, owners, quadrature_weights[:, None] * values)
    conservative = integrated / mesh.volumes[:, None]
    density_average = conservative[:, 0]
    projected = fluid.get_state(
        jxp.DmassP_INPUTS, density_average, np.full(mesh.n_cells, settings["pressure"])
    )
    enthalpy_average = np.asarray(projected["h"])
    energy_check = density_average * (enthalpy_average + kinetic) - settings["pressure"]
    if not np.allclose(energy_check, conservative[:, 3], rtol=2e-12, atol=2e-9):
        raise ValueError(
            "The contact's constant-pressure cell recovery is incompatible with this EOS"
        )
    W = np.column_stack(
        (
            np.full(mesh.n_cells, settings["pressure"]),
            np.full(mesh.n_cells, v_m),
            np.full(mesh.n_cells, ell),
            enthalpy_average,
        )
    )
    fields = dict(
        p=W[:, 0],
        v_m=W[:, 1],
        ell=W[:, 2],
        h=W[:, 3],
        v_theta=np.full(mesh.n_cells, v_theta),
        rho=density_average,
        T=np.asarray(projected["T"]),
        s=np.asarray(projected["s"]),
        h0=enthalpy_average + kinetic,
    )
    return dict(
        W=W,
        q=conservative,
        integrated=integrated,
        fields=fields,
        quadrature_order=order,
        quadrature_split_at_pulse_edges=True,
    )
