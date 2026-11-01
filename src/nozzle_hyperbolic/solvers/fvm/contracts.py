"""Structural interface for user-programmed conservative physical models.

No base-class inheritance, fixed equation identities, thermodynamic library or
fixed variable count is required by numerical assembly. Models declare their
own equation names and variable count. The supported system is square,
cell-local storage on a fixed one-dimensional mesh with two endpoint traces.
Nonconservative/rectangular systems require a later, explicit contract extension.
"""

from typing import Protocol, Sequence, Any
from ...numerics.mesh import Mesh


class BalanceModel(Protocol):
    """Required model data and callbacks; shapes use N cells and V variables.

    W is both the iteration and reconstruction representation. Conserved storage
    may be nonlinear in W. Geometry is already included in integrated_source;
    face_conserved/face_physical_flux are unweighted and use one declared face
    frame. Boundary rows are dimensionless and have no storage.

    Instead of the independent face callbacks, face_data(W) may return
    conserved, physical_flux, speed_bounds and opaque auxiliary data. Optional
    characteristic information, endpoint fluxes, conservative filters,
    cell/boundary admissibility, explicit source timestep bounds, monitoring
    quantities and final solution checks are documented in MODEL_INTERFACE.md.
    Operator validates the callbacks selected by Scheme before compiling.
    """

    mesh: Mesh
    nvar: int
    equation_names: Sequence[str]
    variable_scales: Any
    conservative_scales: Any
    global_flux_scales: Any
    reference_time: float
    source_dependency_radius: int
    boundary_dependency_radius: int
    boundary_coupling: str

    def cell_conserved(self, physical):
        """(N,V) -> (N,V), cell-local densities in the transported frame."""
        ...

    def flux_to_conserved_frame(self, face_flux): ...
    def integrated_source(self, full): ...
    def boundary_residual(self, full, time=0.0): ...
    def admissible(self, full, left_faces, right_faces): ...
    def wave_speed(self, physical): ...
    def initial_state(self): ...


class ConservativeModel(BalanceModel, Protocol):
    """Simple face callbacks, with no mandatory eigensystem or EOS."""

    def face_conserved(self, face_states):
        """(N+1,V) -> (N+1,V), densities in the model's face frame."""
        ...

    def face_physical_flux(self, face_states): ...
    def face_speed_bounds(self, face_states): ...


class FusedConservativeModel(BalanceModel, Protocol):
    """Alternative bundled interface for reusing expensive physical data."""

    def face_data(self, face_states):
        """Return conserved,physical_flux,speed_bounds,auxiliary in one dict."""
        ...


class CharacteristicModel(Protocol):
    """Optional characteristic/Osher eigendata in the conservative face frame.

    Return projectors (faces,waves,V,V), eigenvalues and sonic widths
    (faces,waves). The original characteristic flux requires positive widths; Osher also
    permits zero and evaluates this callback at coincident path states.
    """

    def face_characteristics(self, left, right, delta_factor): ...


class FusedCharacteristicModel(Protocol):
    """Alternative eigendata callback reusing opaque face_data payloads."""

    def face_characteristics_from_data(
        self, left, right, delta_factor, aux_left, aux_right
    ): ...


class FaceStatePathModel(Protocol):
    """Optional path for Osher; otherwise reconstruction states interpolate.

    parameter has shape (faces,1). Return state and dstate/dparameter with
    shape (faces,V), maintaining an admissible hyperbolic path in the model's
    reconstruction variables. Sampled-node checks do not prove whole-path
    admissibility or capture every possible sonic crossing. Scheme.integration
    selects an unsplit Gauss rule or sonic-partitioned panels; both use the
    declared Scheme.quadrature_points without changing the physical path.
    """

    def face_state_path(self, left, right, parameter): ...


class RoeModel(Protocol):
    """Optional Roe capability; no physical averaging is inferred by the solver.

    The dict contains projectors (faces,waves,V,V), eigenvalues/width
    (faces,waves) and boolean valid (faces,). Width is nonnegative; zero
    selects the exact absolute value. Projectors act in the same conservative
    face frame as face_conserved/face_physical_flux. Models own a consistent
    finite-jump Roe linearization, real spectrum and any diagnostic extras.
    """

    def face_roe_data(self, left, right, delta_factor): ...


class FusedRoeModel(Protocol):
    """Alternative Roe capability reusing opaque side ``face_data`` payloads."""

    def face_roe_data_from_data(
        self, left, right, delta_factor, aux_left, aux_right
    ): ...
