"""NURBSpy geometry and physical finite-volume meshes for annular ducts.

The meridional curve stores ``(x, r)``. The independent scalar height curve
uses normalized arclength, not the midline's spline parameter. All particular
channel constructors use this same evaluator and volume-integration path.
Geometry preprocessing is performed once on the host; sampled fields are NumPy
arrays, converted once by the FVM to JAX arrays.
The continuous geometry evaluator is host-only; geometry AD is not provided.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import nurbspy
import nurbspy.numpy as nrb
from scipy.integrate import fixed_quad, quad
from scipy.optimize import brentq


class AnnularMetrics(NamedTuple):
    x: np.ndarray
    r: np.ndarray
    b: np.ndarray
    A: np.ndarray
    x_prime: np.ndarray
    r_prime: np.ndarray
    b_prime: np.ndarray
    dA_dm: np.ndarray
    phi: np.ndarray
    curvature: np.ndarray


@dataclass(frozen=True)
class AnnularGeometry:
    """Fixed, normalized NURBS definitions and their arclength-map snapshot.

    Rebuild this object after changing either input curve. The two curves are
    copied into normalized knot domains on construction; input arrays are not
    modified. Geometry-optimization sensitivities are outside this host builder.
    """

    midline: nrb.NurbsCurve
    height: nrb.NurbsCurve
    length: float
    xi_of_m: Callable = field(repr=False, compare=False)
    _evaluate: Callable = field(repr=False, compare=False)
    knot_distances: tuple[float, ...]
    inverse_tol: float
    quadrature_tol: float

    def evaluate(self, m):
        """Return metrics at dimensional arclengths, preserving input shape."""
        return self._evaluate(np.asarray(m, dtype=np.float64))

    __call__ = evaluate

    def mesh(self, n_cells=None, *, m_faces=None):
        return make_annular_mesh(self, n_cells, m_faces=m_faces)


@dataclass(frozen=True)
class AnnularMesh:
    """Physical cells only; endpoints are physical boundary faces, not ghosts."""

    geometry: AnnularGeometry
    m_faces: np.ndarray
    m_centers: np.ndarray
    widths: np.ndarray
    volumes: np.ndarray
    x_center: np.ndarray
    x_face: np.ndarray
    r_center: np.ndarray
    r_face: np.ndarray
    A_center: np.ndarray
    A_face: np.ndarray
    b: np.ndarray
    b_face: np.ndarray
    dA_dm: np.ndarray
    r_prime: np.ndarray
    x_prime: np.ndarray
    phi: np.ndarray

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != "geometry":
                value.setflags(write=False)

    @property
    def n_cells(self):
        return self.m_centers.size


class _ArclengthInverse:
    """Adapter for the NumPy backend's absent controlled arclength inverse.

    NURBSpy NumPy's public ``get_arclength`` uses a fixed ten-point rule and
    offers neither tolerance control nor an inverse. A vectorized higher-order
    check, adaptive fallback over its public analytical derivative, and bracketed
    roots add those capabilities without another NURBS/basis implementation.
    """

    def __init__(self, curve, tolerance, samples_per_span):
        if (isinstance(samples_per_span, (bool, np.bool_))
                or not isinstance(samples_per_span, (int, np.integer))
                or samples_per_span < 2):
            raise ValueError("steps_per_span must be an integer >= 2")
        self.curve = curve
        self.tolerance = tolerance
        self.knots = np.unique(curve.U)
        samples = np.unique(np.concatenate([
            np.linspace(a, b, max(65, samples_per_span + 1))
            for a, b in zip(self.knots[:-1], self.knots[1:])]))
        speeds = np.linalg.norm(curve.get_derivative(samples, order=1), axis=0)
        if np.any(~np.isfinite(speeds)) or np.any(speeds <= 0):
            raise ValueError("The midline must have positive finite speed on every span")
        # Exact affine parameterizations need no numerical inversion. This is a
        # property of the canonical spline, independent of radial/axial/conical
        # channel labels. Unequal rational weights and higher-degree straight
        # curves deliberately retain the general arclength/root path.
        self._affine = (curve.p == 1 and curve.P.shape[1] == 2
                        and np.array_equal(curve.W, np.full(2, curve.W[0]))
                        and np.array_equal(curve.U, np.array([0., 0., 1., 1.])))
        self._affine_speed = None
        if self._affine:
            delta = np.asarray(curve.P[:, 1] - curve.P[:, 0])
            derivatives = curve.get_derivative(np.array([0., .5, 1.]), order=1)
            if not np.allclose(derivatives, delta[:, None], rtol=2e-14, atol=0):
                raise ValueError("The affine NURBS derivative verification failed")
            self._affine_speed = float(np.linalg.norm(delta))
        lengths = [self._length(a, b) for a, b in zip(self.knots[:-1], self.knots[1:])]
        if np.any(np.asarray(lengths) <= 0):
            raise ValueError("The midline must have positive-length knot spans")
        self.cumulative = np.r_[0., np.cumsum(lengths)]
        self.length = float(self.cumulative[-1])

    def _length(self, lower, upper):
        if lower == upper:
            return 0.
        if self._affine:
            return self._affine_speed * (upper - lower)
        absolute_tolerance = max(self.tolerance * 1e-3, 5e-15)
        relative_tolerance = max(self.tolerance * 1e-2, 5e-14)

        def vector_speed(xi):
            values = np.linalg.norm(self.curve.get_derivative(xi, order=1), axis=0)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("The midline must have positive finite speed")
            return values

        # Reuse the public ten-point method, checking it against a vectorized
        # higher-order rule before accepting it. A failed comparison invokes
        # adaptive QUADPACK below. This avoids hundreds of scalar NURBS calls
        # for every inverse query on ordinary smooth channel definitions.
        coarse = float(self.curve.get_arclength(lower, upper))
        fine = float(fixed_quad(vector_speed, lower, upper, n=32)[0])
        if abs(fine - coarse) <= absolute_tolerance + relative_tolerance * abs(fine):
            return fine

        def speed(xi):
            value = float(np.linalg.norm(self.curve.get_derivative(xi, order=1)))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("The midline must have positive finite speed")
            return value
        result = quad(speed, lower, upper, epsabs=absolute_tolerance,
                      epsrel=relative_tolerance, limit=200, full_output=1)
        if len(result) != 3:
            raise ValueError(f"Midline arclength quadrature failed: {result[3]}")
        return float(result[0])

    def arclength_at(self, xi):
        if not np.isfinite(xi) or not 0 <= xi <= 1:
            raise ValueError("Midline parameter must lie in [0,1]")
        if self._affine:
            return self.length * xi
        panel = min(np.searchsorted(self.knots, xi, side="right") - 1, self.knots.size - 2)
        return float(self.cumulative[panel] + self._length(self.knots[panel], xi))

    def __call__(self, m):
        distances = np.asarray(m, dtype=float)
        if np.any(~np.isfinite(distances)) or np.any((distances < 0) | (distances > self.length)):
            raise ValueError("Arclength queries must lie between zero and geometry.length")
        if self._affine:
            return distances / self.length
        output = np.empty(distances.size)
        for index, distance in enumerate(distances.reshape(-1)):
            if distance == 0:
                output[index] = 0.
            elif distance == self.length:
                output[index] = 1.
            else:
                panel = min(np.searchsorted(self.cumulative, distance, side="right") - 1,
                            self.knots.size - 2)
                lower, upper = self.knots[panel:panel + 2]
                target = distance - self.cumulative[panel]
                output[index] = brentq(lambda xi: self._length(lower, xi) - target,
                                       lower, upper, xtol=5e-15, rtol=4*np.finfo(float).eps)
        return output.reshape(distances.shape)


def _normalized_curve(curve, dimensions, name):
    if not isinstance(curve, nrb.NurbsCurve):
        raise TypeError(f"{name} must be a nurbspy.numpy.NurbsCurve")
    controls, weights, knots = map(np.asarray, (curve.P, curve.W, curve.U))
    if (controls.ndim != 2 or controls.shape[0] != dimensions
            or controls.shape[1] < 2):
        raise ValueError(f"{name} controls must have shape ({dimensions}, n), n >= 2")
    if any(np.iscomplexobj(a) or not np.isfinite(a).all()
           for a in (controls, weights, knots)):
        raise ValueError(f"{name} requires finite real controls, weights and knots")
    if isinstance(curve.p, (bool, np.bool_)) or int(curve.p) != curve.p:
        raise ValueError(f"{name} degree must be an integer")
    degree = int(curve.p)
    if not 1 <= degree < controls.shape[1]:
        raise ValueError(f"{name} degree must satisfy 1 <= degree < control count")
    if (weights.shape != (controls.shape[1],) or np.any(weights <= 0)
            or knots.shape != (controls.shape[1] + degree + 1,)
            or np.any(np.diff(knots) < 0)):
        raise ValueError(f"{name} requires positive weights and a valid knot vector")
    start, end = knots[degree], knots[-degree - 1]
    if end <= start or np.any(knots[:degree + 1] != start) or np.any(knots[-degree - 1:] != end):
        raise ValueError(f"{name} requires a nonempty clamped active knot domain")
    _, counts = np.unique(knots[degree + 1:-degree - 1], return_counts=True)
    if np.any(counts > degree):
        raise ValueError(f"{name} must be continuous across internal knots")
    return nrb.NurbsCurve(np.array(controls, dtype=np.float64, copy=True),
                         np.array(weights, dtype=np.float64, copy=True), degree,
                         np.array((knots - start) / (end - start), dtype=np.float64, copy=True))


def make_annular_geometry(midline, height, *, inverse_tol=1e-10,
                          quadrature_tol=1e-11, steps_per_span=64):
    """Build a fixed geometry from a 2D midline and scalar normal-height spline.

    Arbitrary clamped input knot domains are normalized once. The height's
    normalized coordinate is always ``sigma=m/length``. NURBSpy >= 1.6 supplies
    curve values and analytical derivatives. A thin adaptive-quadrature/root
    adapter supplies the absent NumPy arclength inverse, without radial or
    axial inversion bypasses. Geometry AD is not supported by this host builder.
    Invalid curves, offset walls and failed inversions raise explicitly.
    """
    version = tuple(int(part) for part in nurbspy.__version__.split(".")[:2])
    if version < (1, 6):
        raise ImportError("Annular geometry requires NURBSpy >= 1.6 for fixed-geometry preprocessing")
    if (not np.isfinite(inverse_tol) or inverse_tol <= 0
            or not np.isfinite(quadrature_tol) or quadrature_tol <= 0):
        raise ValueError("Geometry tolerances must be finite and positive")
    midline = _normalized_curve(midline, 2, "midline")
    height = _normalized_curve(height, 1, "height")
    xi_of_m = _ArclengthInverse(midline, inverse_tol, steps_per_span)
    length = xi_of_m.length
    if not np.isfinite(length) or length <= 0:
        raise ValueError("The meridional curve must have positive finite length")

    def evaluate(m):
        shape = m.shape
        flat = m.reshape(-1)
        xi = xi_of_m(flat)
        position = midline.get_value(xi)
        derivative = midline.get_derivative(xi, order=1)
        second = midline.get_derivative(xi, order=2)
        speed = np.linalg.norm(derivative, axis=0)
        tangent = derivative / speed
        b = height.get_value(flat / length)[0]
        b_prime = height.get_derivative(flat / length, order=1)[0] / length
        x, r = position
        x_prime, r_prime = tangent
        area = 2 * np.pi * r * b
        dA_dm = 2 * np.pi * (r_prime * b + r * b_prime)
        curvature = (derivative[0] * second[1] - derivative[1] * second[0]) / speed**3
        values = (x, r, b, area, x_prime, r_prime, b_prime, dA_dm,
                  np.arctan2(r_prime, x_prime), curvature)
        return AnnularMetrics(*(value.reshape(shape) for value in values))

    # The inverse adapter validates positive speed per knot span. Check channel
    # admissibility on a dense, span-aware grid as well. Sampling is explicitly
    # a geometry admissibility check, not a proof for every possible spline.
    mid_knots = np.unique(np.asarray(midline.U))
    height_knots = np.unique(np.asarray(height.U))
    knot_m = [0.0, length]
    for knot in mid_knots[1:-1]:
        knot_m.append(xi_of_m.arclength_at(float(knot)))
    knot_m.extend((length * height_knots[1:-1]).tolist())
    knot_m = np.unique(knot_m)
    samples = np.unique(np.concatenate([
        np.linspace(lo, hi, 129) for lo, hi in zip(knot_m[:-1], knot_m[1:])]))
    metrics = evaluate(np.asarray(samples))
    if any(not np.isfinite(np.asarray(value)).all() for value in metrics):
        raise ValueError("Annular geometry metrics must be finite")
    if np.any(np.asarray(metrics.r) <= 0) or np.any(np.asarray(metrics.b) <= 0):
        raise ValueError("Annular radius and normal height must remain positive")
    if np.any(np.asarray(metrics.r - np.abs(metrics.x_prime) * metrics.b / 2) <= 0):
        raise ValueError("The normal-offset walls must remain at positive radius")
    if np.any(np.asarray(np.abs(metrics.curvature) * metrics.b / 2) >= 1):
        raise ValueError("Height is too large for regular normal-offset walls")
    for knot in mid_knots[1:-1]:
        gap = min(knot - mid_knots[mid_knots < knot][-1],
                  mid_knots[mid_knots > knot][0] - knot) * 1e-8
        d = np.asarray(midline.get_derivative(np.array([knot - gap, knot + gap]), order=1))
        directions = d / np.linalg.norm(d, axis=0)
        if not np.allclose(directions[:, 0], directions[:, 1], atol=2e-6, rtol=0):
            raise ValueError("The midline must have a continuous tangent across knots")
    return AnnularGeometry(midline, height, length, xi_of_m, evaluate,
                           tuple(float(x) for x in knot_m), inverse_tol, quadrature_tol)


_GL16 = np.polynomial.legendre.leggauss(16)
_GL32 = np.polynomial.legendre.leggauss(32)


def _integrate_area(geometry, lower, upper, scale, depth=0):
    def rule(nodes_weights):
        nodes, weights = nodes_weights
        points = lower + (upper - lower) * (nodes + 1) / 2
        areas = np.asarray(geometry.evaluate(np.asarray(points)).A)
        return (upper - lower) / 2 * float(weights @ areas)
    low, high = rule(_GL16), rule(_GL32)
    tolerance = geometry.quadrature_tol * (abs(high) + scale * (upper - lower) / geometry.length)
    if abs(high - low) <= tolerance:
        return high
    if depth >= 12:
        raise ValueError("Annular cell-volume quadrature did not meet its tolerance")
    middle = (lower + upper) / 2
    return (_integrate_area(geometry, lower, middle, scale, depth + 1)
            + _integrate_area(geometry, middle, upper, scale, depth + 1))


def make_annular_mesh(geometry, n_cells=None, *, m_faces=None):
    """Sample uniform or supplied nonuniform physical arclength faces.

    Volumes integrate ``2*pi*r*b dm`` with controlled Gauss quadrature split
    at all height/midline knot distances. They are not midpoint-area products.
    The discrete pressure source should use ``diff(A_face)``, while ``dA_dm``
    is retained for continuous diagnostics and manufactured-solution checks.
    """
    if not isinstance(geometry, AnnularGeometry):
        raise TypeError("geometry must be an AnnularGeometry")
    if m_faces is None:
        if (isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer))
                or n_cells < 1):
            raise ValueError("n_cells must be a positive integer")
        faces = np.linspace(0., geometry.length, n_cells + 1)
    else:
        if n_cells is not None:
            raise ValueError("Supply n_cells or m_faces, not both")
        if np.iscomplexobj(m_faces):
            raise ValueError("m_faces must be real")
        faces = np.array(m_faces, dtype=float, copy=True)
        endpoint_tolerance = 16 * np.finfo(float).eps * geometry.length
        if (faces.ndim != 1 or faces.size < 2 or not np.isfinite(faces).all()
                or np.any(np.diff(faces) <= 0) or abs(faces[0]) > endpoint_tolerance
                or abs(faces[-1] - geometry.length) > endpoint_tolerance):
            raise ValueError("m_faces must increase strictly from 0 to geometry.length")
        # Canonical physical endpoints; permit only floating-point roundoff.
        faces[0], faces[-1] = 0., geometry.length
    centers = (faces[:-1] + faces[1:]) / 2
    face_metrics = geometry.evaluate(np.asarray(faces))
    center_metrics = geometry.evaluate(np.asarray(centers))
    scale = float(np.max(face_metrics.A)) * geometry.length
    volumes = []
    for lower, upper in zip(faces[:-1], faces[1:]):
        splits = [lower, *(k for k in geometry.knot_distances if lower < k < upper), upper]
        volumes.append(sum(_integrate_area(geometry, a, b, scale)
                           for a, b in zip(splits[:-1], splits[1:])))
    if not np.isfinite(volumes).all() or np.any(np.asarray(volumes) <= 0):
        raise ValueError("Annular cell volumes must be finite and positive")
    return AnnularMesh(geometry, np.asarray(faces), np.asarray(centers),
                       np.asarray(np.diff(faces)), np.asarray(volumes),
                       center_metrics.x, face_metrics.x, center_metrics.r, face_metrics.r,
                       center_metrics.A, face_metrics.A, center_metrics.b, face_metrics.b,
                       center_metrics.dA_dm, center_metrics.r_prime,
                       center_metrics.x_prime, center_metrics.phi)


def constant_height(b):
    """A scalar degree-one NURBS height definition."""
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Height must be finite and positive")
    return nrb.NurbsCurve(np.array([[float(b), float(b)]]), degree=1)


def radial_channel(r_in, r_out, b, *, x=0., **geometry_options):
    """Outward or inward radial channel with constant normal height."""
    midline = nrb.NurbsCurve(np.array([[x, x], [r_in, r_out]], dtype=np.float64), degree=1)
    return make_annular_geometry(midline, constant_height(b), **geometry_options)


def conical_channel(length, r_in, b_in, phi, semi_divergence=0., *, x_in=0., **geometry_options):
    """Straight meanline; wall half-angle sets ``b=b_in+2*m*tan(delta)``.

    ``phi`` is the meanline cant angle from the positive axial direction and
    ``semi_divergence`` is the wall half-angle; both are in radians. ``length``
    is meridional arclength, not axial length. Zero cant gives an axial annulus.
    """
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Meridional length must be finite and positive")
    if not np.isfinite(phi) or not np.isfinite(semi_divergence) or abs(semi_divergence) >= np.pi / 2:
        raise ValueError("Angles must be finite with abs(semi_divergence) < pi/2")
    midline = nrb.NurbsCurve(np.array([
        [x_in, x_in + length * np.cos(phi)],
        [r_in, r_in + length * np.sin(phi)]], dtype=np.float64), degree=1)
    height = nrb.NurbsCurve(np.array([[b_in, b_in + 2 * length * np.tan(semi_divergence)]],
                                     dtype=np.float64), degree=1)
    return make_annular_geometry(midline, height, **geometry_options)


def circular_arc_channel(center, radius, theta_start, theta_end, b, **geometry_options):
    """Counterclockwise circular midline in the (x,r) plane; angles in radians.

    For reverse orientation, supply a reversed NURBS midline to the general
    builder. This constructor uses NURBSpy's exact rational circular-arc data.
    """
    if (not np.isfinite(radius) or radius <= 0 or not np.isfinite(theta_start)
            or not np.isfinite(theta_end) or not 0 < theta_end - theta_start <= 2 * np.pi):
        raise ValueError("Circular arcs require positive radius and angle span in (0, 2*pi]")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Arc center must contain finite (x,r) coordinates")
    arc = nrb.CircularArc(center, np.array([1., 0.]), np.array([0., 1.]),
                          radius, theta_start, theta_end)
    return make_annular_geometry(arc.NurbsCurve, constant_height(b), **geometry_options)
