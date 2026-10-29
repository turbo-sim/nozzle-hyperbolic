"""Independent geometry oracles for the canonical annular NURBS pipeline."""

import numpy as np
from numpy.testing import assert_allclose
import nurbspy.numpy as nrb
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

from nozzle_hyperbolic.geometry.annular import (
    circular_arc_channel, conical_channel, constant_height,
    make_annular_geometry, radial_channel,
)


@pytest.mark.parametrize("r_in,r_out,sign", [(0.1, 0.2, 1), (0.2, 0.1, -1)])
def test_radial_orientations_and_nonuniform_exact_volumes(r_in, r_out, sign):
    geometry = radial_channel(r_in, r_out, 0.005, x=0.3)
    assert_allclose(geometry.length, 0.1, atol=2e-12)
    faces = np.array([0., .01, .037, .08, geometry.length])
    mesh = geometry.mesh(m_faces=faces)
    expected_r = r_in + sign * np.asarray(mesh.m_centers)
    assert mesh.n_cells == 4
    assert_allclose(mesh.r_center, expected_r, atol=2e-11)
    assert_allclose(mesh.r_prime, sign, atol=2e-12)
    assert_allclose(mesh.x_prime, 0., atol=2e-12)
    assert_allclose(mesh.dA_dm, 2 * np.pi * .005 * sign, atol=2e-12)
    primitive = lambda m: 2 * np.pi * .005 * (r_in * m + sign * m * m / 2)
    assert_allclose(mesh.volumes, np.diff(primitive(faces)), rtol=2e-10, atol=2e-14)
    assert_allclose(mesh.A_face, 2 * np.pi * .005 * (r_in + sign * faces), atol=2e-12)
    assert mesh.m_faces.shape == (5,)  # Exactly the physical faces, without ghosts.


def test_conical_angles_chain_rule_and_volume():
    length, radius, width, phi, delta = .12, .15, .006, .4, .01
    geometry = conical_channel(length, radius, width, phi, delta, x_in=.02)
    mesh = geometry.mesh(17)
    m = np.asarray(mesh.m_centers)
    r1, b1 = np.sin(phi), 2 * np.tan(delta)
    assert_allclose(mesh.x_center, .02 + m * np.cos(phi), atol=2e-11)
    assert_allclose(mesh.r_center, radius + m * r1, atol=2e-11)
    assert_allclose(mesh.b, width + m * b1, atol=2e-11)
    assert_allclose(mesh.dA_dm, 2 * np.pi * (r1 * (width + b1*m) + b1*(radius+r1*m)), atol=2e-11)
    primitive = lambda z: 2*np.pi*(radius*width*z + (radius*b1+r1*width)*z*z/2 + r1*b1*z**3/3)
    assert_allclose(mesh.volumes, np.diff(primitive(np.asarray(mesh.m_faces))), rtol=2e-10, atol=2e-14)
    metrics = geometry.evaluate(np.array([0., geometry.length]))
    assert_allclose(metrics.phi, phi, atol=2e-12)
    assert_allclose(metrics.b_prime, b1, atol=2e-12)


def test_circular_arc_length_curvature_and_independent_volume():
    radius, center_r, width = .05, .16, .004
    start, end = -np.pi / 2, 0.
    geometry = circular_arc_channel((.02, center_r), radius, start, end, width)
    expected_length = radius * (end - start)
    assert_allclose(geometry.length, expected_length, atol=2e-11)
    mesh = geometry.mesh(12)
    angle = start + np.asarray(mesh.m_centers) / radius
    metrics = geometry.evaluate(mesh.m_centers)
    assert_allclose(metrics.x, .02 + radius*np.cos(angle), atol=2e-10)
    assert_allclose(metrics.r, center_r + radius*np.sin(angle), atol=2e-10)
    assert_allclose(metrics.curvature, 1/radius, atol=2e-9)
    assert_allclose(metrics.r_prime, np.cos(angle), atol=2e-9)
    expected_volume = 2*np.pi*width*(center_r*expected_length + radius**2*(np.cos(start)-np.cos(end)))
    assert_allclose(np.sum(mesh.volumes), expected_volume, rtol=2e-9, atol=2e-13)
    step = 1e-6 * geometry.length
    slope = (geometry.xi_of_m(mesh.m_centers + step)
             - geometry.xi_of_m(mesh.m_centers - step)) / (2 * step)
    derivative = geometry.midline.get_derivative(geometry.xi_of_m(mesh.m_centers), order=1)
    assert_allclose(slope * np.linalg.norm(derivative, axis=0), 1., atol=3e-7)


def test_nonlinear_parameter_arclength_and_height_coordinate():
    # x=xi+xi^2 gives nonconstant spline speed. The independent height uses
    # sigma=m/L, so its physical derivative is constant despite that speed.
    midline = nrb.NurbsCurve(np.array([[0., .5, 2.], [.2, .2, .2]]))
    height = nrb.NurbsCurve(np.array([[.004, .008]]), degree=1,
                            knots=np.array([2., 2., 5., 5.]))
    geometry = make_annular_geometry(midline, height)
    m = np.linspace(0., geometry.length, 11)
    metrics = geometry.evaluate(m)
    assert_allclose(metrics.x, m, atol=3e-9)
    assert_allclose(metrics.b, .004 + .004*np.asarray(m)/2, atol=2e-12)
    assert_allclose(metrics.b_prime, .002, atol=2e-12)
    assert_allclose(geometry.xi_of_m(m) + geometry.xi_of_m(m)**2, m, atol=3e-9)
    assert geometry.evaluate(np.array(.7)).r.shape == ()


def test_affine_inverse_is_exact_and_rational_line_keeps_nonlinear_map():
    controls = np.array([[.02, .08], [.12, .20]])
    geometry = make_annular_geometry(nrb.NurbsCurve(controls, np.array([3., 3.]), degree=1),
                                     constant_height(.004))
    distances = np.linspace(0., geometry.length, 8193)
    assert_allclose(geometry.length, np.linalg.norm(controls[:,1]-controls[:,0]), rtol=0, atol=0)
    assert_allclose(geometry.xi_of_m(distances), distances/geometry.length, rtol=0, atol=0)
    assert_allclose(geometry.xi_of_m.arclength_at(.37), .37*geometry.length, rtol=0, atol=0)
    rational = make_annular_geometry(nrb.NurbsCurve(controls, np.array([1., 2.]), degree=1),
                                     constant_height(.004))
    fractions = np.linspace(0., 1., 17)
    assert_allclose(rational.xi_of_m(fractions*rational.length), fractions/(2-fractions), atol=2e-12)


def test_general_rational_geometry_against_independent_scipy_quadrature():
    controls = np.array([[0., .03, .10, .15], [.13, .12, .16, .18]])
    weights = np.array([1., .8, 1.2, 1.])
    curve = nrb.NurbsCurve(controls, weights)
    geometry = make_annular_geometry(curve, constant_height(.003))
    # SciPy evaluates homogeneous polynomials independently of NURBSpy/JAX.
    spline = BSpline(np.asarray(curve.U), np.vstack((controls*weights, weights)).T, curve.p)
    def position_speed(xi):
        value, derivative = spline(xi), spline(xi, 1)
        position = value[:-1]/value[-1]
        tangent = (derivative[:-1]*value[-1]-value[:-1]*derivative[-1])/value[-1]**2
        return position, np.linalg.norm(tangent)
    length = quad(lambda xi: position_speed(xi)[1], 0., 1., epsabs=1e-13, epsrel=1e-12)[0]
    volume = quad(lambda xi: 2*np.pi*.003*position_speed(xi)[0][1]*position_speed(xi)[1],
                  0., 1., epsabs=1e-14, epsrel=1e-12)[0]
    assert_allclose(geometry.length, length, rtol=2e-10, atol=2e-12)
    assert_allclose(np.sum(geometry.mesh(15).volumes), volume, rtol=2e-9, atol=2e-13)


@pytest.mark.parametrize("count", [0, -1, 2.5, True, None])
def test_invalid_physical_cell_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        radial_channel(.1, .2, .005).mesh(count)


def test_invalid_channels_and_faces_are_rejected():
    with pytest.raises(Exception):
        radial_channel(.1, .1, .005)
    with pytest.raises(ValueError, match="positive"):
        radial_channel(-.2, -.1, .005)
    with pytest.raises(ValueError, match="positive"):
        radial_channel(.1, .2, -.005)
    geometry = radial_channel(.1, .2, .005)
    for faces in ([0., .05, .04, geometry.length], [-.01, geometry.length], [0., .09]):
        with pytest.raises(ValueError, match="m_faces"):
            geometry.mesh(m_faces=faces)
