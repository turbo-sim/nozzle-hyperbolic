"""Structural interface for user-programmed constrained balance models.

No base-class inheritance, fixed equation identities, thermodynamic library or
fixed variable count is required by numerical assembly. Models declare their
own equation names and variable count. Ntransport balances plus Nconstraints
cell-local algebraic rows form a square state system. Storage itself can be
rectangular. Physical traces each have Nvar algebraic boundary equations.
"""

from typing import Protocol, Sequence, Any
from ...numerics.mesh import Mesh


class BalanceModel(Protocol):
    """Required model data and callbacks; shapes use N cells and V variables.

    W is the iteration representation; optional local forward/inverse callbacks
    can select different reconstruction variables. Conserved storage may be
    nonlinear in W. Geometry is already included in integrated_source;
    face_conserved/face_physical_flux are unweighted and use one declared face
    frame. Boundary rows are dimensionless and have no storage.

    Instead of the independent face callbacks, face_data(W) may return
    conserved, physical_flux, speed_bounds and opaque auxiliary data. Optional
    characteristic information, endpoint fluxes, conservative filters,
    cell/boundary admissibility, explicit source timestep bounds, monitoring
    quantities and final solution checks are documented in docs/model_interface.md.
    Operator validates the callbacks selected by Scheme before compiling.
    """

    mesh: Mesh
    nvar: int
    ntransport: int
    nconstraints: int
    nboundary: int
    nonconservative_mode: str
    equation_names: Sequence[str]
    variable_scales: Any
    conservative_scales: Any
    global_flux_scales: Any
    reference_time: float
    source_dependency_radius: int
    boundary_dependency_radius: int
    boundary_coupling: str

    def cell_conserved(self, physical):
        """Evaluate cell-local conserved densities in the transported frame.

        Parameters
        ----------
        physical : jax.Array, shape (N, V)
            Physical cell iteration states, excluding endpoint traces.

        Returns
        -------
        jax.Array, shape (N, T)
            Conserved densities per physical volume. The operator applies
            cell volumes once; this callback must not couple distinct cells.
        """
        ...

    def flux_to_conserved_frame(self, face_flux):
        """Convert face-frame fluxes to the cell storage frame.

        Parameters
        ----------
        face_flux : jax.Array, shape (N + 1, T)
            Unweighted numerical fluxes in the model's physical face frame.

        Returns
        -------
        jax.Array, shape (N + 1, T)
            Unweighted transported fluxes; the operator applies face area.
            The transformation must act independently on each face.
        """
        ...

    def integrated_source(self, full):
        """Evaluate physical sources integrated over each control volume.

        Parameters
        ----------
        full : jax.Array, shape (N + 2, V)
            Full iteration state, including independent endpoint traces.

        Returns
        -------
        jax.Array, shape (N, T)
            Integrated sources subtracted from flux divergence in the spatial
            residual. All required physical geometry measures are included.
        """
        ...

    def boundary_residual(self, full, time=0.0):
        """Evaluate the canonical zero-storage physical endpoint equations.

        Parameters
        ----------
        full : jax.Array, shape (N + 2, V)
            Full iteration state with traces in the first and last rows.
        time : float or jax.Array, optional
            Physical time for prescribed boundary data, default zero.

        Returns
        -------
        jax.Array, shape (2, V)
            Dimensionless algebraic defects ordered inlet then outlet.
            Neither timestep nor control-volume storage multiplies these rows.
        """
        ...

    def admissible(self, full, left_faces, right_faces):
        """Check cells, endpoint traces and reconstructed face states safely.

        Parameters
        ----------
        full : jax.Array, shape (N + 2, V)
            Full iteration state.
        left_faces, right_faces : jax.Array, shape (N + 1, V)
            Reconstructed states on both sides of each face.

        Returns
        -------
        jax.Array
            Scalar boolean. Property-domain checks must safely reject invalid
            inputs before calling an EOS that cannot evaluate those inputs.
        """
        ...

    def wave_speed(self, physical):
        """Return nonnegative cell signal speeds for timestep selection.

        Parameters
        ----------
        physical : jax.Array, shape (N, V)
            Physical cell iteration states.

        Returns
        -------
        jax.Array, shape (N,)
            Finite upper bounds on absolute physical signal speeds.
        """
        ...

    def initial_state(self):
        """Construct the model's physical initial guess.

        Returns
        -------
        jax.Array, shape (N + 2, V)
            Initial iteration variables, including guesses for both algebraic
            endpoint traces. No analytical verification profile is inferred.
        """
        ...


class ConservativeModel(BalanceModel, Protocol):
    """Simple face callbacks, with no mandatory eigensystem or EOS."""

    def face_conserved(self, face_states):
        """Evaluate face-local conserved densities without area weighting.

        Parameters
        ----------
        face_states : jax.Array, shape (N + 1, V)
            Reconstructed iteration states for one side of every face.

        Returns
        -------
        jax.Array, shape (N + 1, T)
            Conserved densities in the same frame as the physical face flux.
        """
        ...

    def face_physical_flux(self, face_states):
        """Evaluate physical face fluxes without area weighting.

        Parameters
        ----------
        face_states : jax.Array, shape (N + 1, V)
            Reconstructed iteration states for one side of every face.

        Returns
        -------
        jax.Array, shape (N + 1, T)
            Face-local physical fluxes in the declared conservative face frame.
        """
        ...

    def face_speed_bounds(self, face_states):
        """Bound the signed physical characteristic speeds at each face.

        Parameters
        ----------
        face_states : jax.Array, shape (N + 1, V)
            Reconstructed iteration states for one side of every face.

        Returns
        -------
        lower, upper : jax.Array, shape (N + 1,)
            Finite lower and upper signal-speed bounds for HLL/Rusanov.
        """
        ...


class FusedConservativeModel(BalanceModel, Protocol):
    """Alternative bundled interface for reusing expensive physical data."""

    def face_data(self, face_states):
        """Reuse one physical-property evaluation for a face-local data bundle.

        Parameters
        ----------
        face_states : jax.Array, shape (N + 1, V)
            Reconstructed iteration states for one side of every face.

        Returns
        -------
        dict
            Keys ``conserved`` and ``physical_flux`` hold ``(N + 1, T)`` arrays;
            ``speed_bounds`` holds two ``(N + 1,)`` arrays. ``auxiliary`` is an
            opaque JAX pytree reusable by fused characteristic/Roe callbacks.
        """
        ...


class CharacteristicModel(Protocol):
    """Optional Stadtke/Osher eigendata in the conservative face frame.

    Return projectors (faces,waves,T,T), eigenvalues and sonic widths
    (faces,waves). Stadtke requires positive widths; Osher also permits zero.
    Endpoint-envelope speeds must retain the mean's family indexing. Osher
    also evaluates this callback at coincident path states.
    """

    def face_characteristics(self, left, right, delta_factor):
        """Build model-owned characteristic data for a pair of face states.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed iteration states on both sides of every face.
        delta_factor : float
            Positive characteristic regularization coefficient.

        Returns
        -------
        projectors : jax.Array, shape (N + 1, K, T, T)
            Spectral projectors acting on transported conserved jumps.
        eigenvalues, width : jax.Array, shape (N + 1, K)
            Ordered family speeds and sonic regularization widths.
        """
        ...


class FusedCharacteristicModel(Protocol):
    """Alternative eigendata callback reusing opaque face_data payloads."""

    def face_characteristics_from_data(
        self, left, right, delta_factor, aux_left, aux_right
    ):
        """Build characteristic data while reusing both endpoint property bundles.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed iteration states.
        delta_factor : float
            Positive characteristic regularization coefficient.
        aux_left, aux_right : object
            Opaque JAX pytrees from the corresponding ``face_data`` calls.

        Returns
        -------
        projectors : jax.Array, shape (N + 1, K, T, T)
            Conservative spectral projectors.
        eigenvalues, width : jax.Array, shape (N + 1, K)
            Family speeds and sonic widths, as in ``face_characteristics``.
        """
        ...


class FaceStatePathModel(Protocol):
    """Optional path for Osher; otherwise reconstruction states interpolate.

    parameter has shape (faces,1). Return state and dstate/dparameter with
    shape (faces,V), maintaining an admissible hyperbolic path in the model's
    iteration variables after face reconstruction. Sampled-node checks do not prove whole-path
    admissibility or capture every possible sonic crossing. Scheme.integration
    selects an unsplit Gauss rule or sonic-partitioned panels; both use the
    declared Scheme.quadrature_points without changing the physical path.
    """

    def face_state_path(self, left, right, parameter):
        """Evaluate the physical model's admissible Osher path and tangent.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed face states in the model's iteration variables.
        parameter : jax.Array, shape (N + 1, 1)
            Path coordinates between zero at left and one at right.

        Returns
        -------
        state, tangent : jax.Array, shape (N + 1, V)
            Iteration state along the path and its derivative with respect to
            the path coordinate. Storage differentiation uses these variables.
        """
        ...


class RoeModel(Protocol):
    """Optional Roe capability; no physical averaging is inferred by the solver.

    The dict contains projectors (faces,waves,T,T), eigenvalues/width
    (faces,waves) and boolean valid (faces,). Width is nonnegative; zero
    selects the exact absolute value. Projectors act in the same conservative
    face frame as face_conserved/face_physical_flux. Models own a consistent
    finite-jump Roe linearization, real spectrum and any diagnostic extras.
    """

    def face_roe_data(self, left, right, delta_factor):
        """Evaluate a model-owned finite-jump Roe linearization.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed iteration states on both sides of every face.
        delta_factor : float
            Positive entropy regularization coefficient.

        Returns
        -------
        dict
            ``projectors`` has shape ``(N + 1, K, T, T)``;
            ``eigenvalues`` and nonnegative ``width`` have shape ``(N + 1, K)``;
            ``valid`` is a boolean vector of shape ``(N + 1,)``.
        """
        ...


class FusedRoeModel(Protocol):
    """Alternative Roe capability reusing opaque side ``face_data`` payloads."""

    def face_roe_data_from_data(
        self, left, right, delta_factor, aux_left, aux_right
    ):
        """Evaluate Roe data using existing endpoint property bundles.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed iteration states.
        delta_factor : float
            Positive entropy regularization coefficient.
        aux_left, aux_right : object
            Opaque JAX pytrees from the corresponding ``face_data`` calls.

        Returns
        -------
        dict
            The projector, speed, width and validity bundle defined by
            ``face_roe_data``, using the same conservative face frame.
        """
        ...


class ConstrainedCellModel(Protocol):
    """Zero-storage algebraic cells, scaled independently of dt and volume.

    The Jacobian of (cell_conserved, cell_constraints) must be locally
    invertible. Constraint scales are fixed physical scales, never state/phase
    weighted. Newton packs transported rows first, then these constraint rows.
    """

    nconstraints: int
    constraint_names: Sequence[str]
    constraint_scales: Any

    def cell_constraints(self, physical):
        """Evaluate dimensional cell-local algebraic defects.

        Parameters
        ----------
        physical : jax.Array, shape (N, V)
            Physical cell iteration states.

        Returns
        -------
        jax.Array, shape (N, C)
            Algebraic defects before division by fixed ``constraint_scales``.
            These equations have zero storage and no timestep multiplier.
        """
        ...


class FluxDifferenceNonconservativeModel(Protocol):
    """Explicit compatibility discretization, not an inferred path scheme.

    integrated_nonconservative receives the unweighted transported numerical
    face flux BEFORE additional stabilization and endpoint flux overrides.
    Return already integrated (N,T) contributions ADDED to the spatial residual.
    The model owns geometry pairing and the validity/singular limits of its
    correction. Supported fluxes must be explicitly justified and declared.
    """

    nonconservative_mode: str  # 'flux_difference'; conservative models: 'none'
    supported_fluxes: Sequence[str]
    nonconservative_dependency_radius: int

    def integrated_nonconservative(self, full, unfiltered_flux):
        """Evaluate the declared flux-difference nonconservative contribution.

        Parameters
        ----------
        full : jax.Array, shape (N + 2, V)
            Full iteration state including endpoint traces.
        unfiltered_flux : jax.Array, shape (N + 1, T)
            Unweighted numerical flux in the transported storage frame, before
            extra stabilization and endpoint physical-flux overrides.

        Returns
        -------
        jax.Array, shape (N, T)
            Already integrated nonconservative rows added to the spatial
            residual. The model owns the entire geometry pairing.
        """
        ...


class CompatibilityFluxModel(Protocol):
    """Optional explicitly selected model-owned historical flux formulation.

    Used only for Scheme(flux='model_compatibility'). Keep physical averaging,
    eigenstructure and regularization in the model; never transplant a solver.
    This callback is face-local and returns (N+1,T) unweighted face-frame flux.
    """

    def numerical_face_flux(self, left, right, scheme):
        """Evaluate the explicitly selected physical compatibility flux.

        Parameters
        ----------
        left, right : jax.Array, shape (N + 1, V)
            Reconstructed iteration states.
        scheme : Scheme
            Fixed numerical settings with ``flux="model_compatibility"``.

        Returns
        -------
        jax.Array, shape (N + 1, T)
            Unweighted face-local flux in the model's physical face frame,
            before conversion to the transported storage frame.
        """
        ...
