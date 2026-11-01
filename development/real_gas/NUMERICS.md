# Numerical formulation and verification

The unknowns are $(p,u,h)$. With bicubic density $\rho(p,h)$ and
$E=h+u^2/2-p/\rho$, the homogeneous conservative state and physical flux are

$$
U=\begin{bmatrix}\rho\\\rho u\\\rho E\end{bmatrix},\qquad
F=\begin{bmatrix}\rho u\\p+\rho u^2\\\rho u(h+u^2/2)\end{bmatrix}.
$$

Uniform axial cells use exact integrated area/volume measures. The steady
integrated cell residual is

$$
R_i=A_{i+1/2}\widehat F_{i+1/2}-A_{i-1/2}\widehat F_{i-1/2}
-\begin{bmatrix}0\\p_i(A_{i+1/2}-A_{i-1/2})\\0\end{bmatrix}.
$$

Face states use constant reconstruction or linear MUSCL with the Van Leer
limiter. Rusanov, HLL, Roe, corrected Stadtke and Osher are the production
numerical flux families. The geometric
source uses midpoint cell pressure. Reservoir and outlet equations are solved
together with the cell equations by implicit Euler/Newton iteration.

## Corrected Stadtke

Both custom corrections are enabled: the conservative-jump correction and the
endpoint characteristic-speed envelope. For model-provided mean projectors
$P_k$, characteristic speeds $\lambda_k$, Harten absolute speeds $\phi_k$ and
endpoint speeds $\lambda_{k,L},\lambda_{k,R}$, the resulting contraction is

$$
\widehat F=\frac{F_L+F_R}{2}
-\frac12\sum_k\nu_kP_k(U_R-U_L),\qquad
\nu_k=\max\left(\phi_k,|\lambda_{k,L}|,|\lambda_{k,R}|\right).
$$

These corrections are project additions to the original Stadtke FVS formula.
The mean characteristic construction does not impose Roe's finite-jump secant
condition. The exact options are saved in each stage's `scheme` object.

## Osher path viscosity

The model's primitive path is $W(s)=W_L+s(W_R-W_L)$, with $W=(p,u,h)$ and
$0\le s\le1$. Storage differentiation supplies $dU(W(s))/ds$. The flux is

$$
\widehat F=\frac{F_L+F_R}{2}
-\frac12\int_0^1\sum_k\phi_k(W(s))P_k(W(s))
\frac{dU(W(s))}{ds}\,ds.
$$

The Gauss configuration applies a five-point rule to the whole path. The
sonic-partitioned configuration applies a five-point rule on each panel bounded
by captured characteristic zeroes and Harten-transition crossings. Quadrature
and integration policy are independent of constant/MUSCL reconstruction.
The production validity check rejects inadmissible sampled path states;
finite sampling does not certify every point of an arbitrary EOS path.

## General-EOS Roe data

Density, sound speed and Grüneisen coefficient are evaluated from the bicubic
tables. For fixed reference values $(\rho_*,p_*,a_*,h_*)$, define
$e_*=h_*-p_*/\rho_*$, centered energy density $I=\rho(h-e_*)-p$, and jumps

$$
x=\frac{\Delta\rho}{\rho_*},\qquad
y=\frac{\Delta I}{\rho_*a_*^2},\qquad d=\frac{\Delta p}{p_*}.
$$

Square-root-density averages supply provisional centered pressure slopes
$g_0,b_0$. With $r_0=d-g_0x-b_0y$, their scaled secant projection is

$$
g=g_0+w\frac{r_0x}{x^2+y^2},\qquad
b=b_0+w\frac{r_0y}{x^2+y^2}.
$$

The C2 weight is zero for $x^2+y^2\le\epsilon^{2/3}$ and one for
$x^2+y^2\ge4\epsilon^{2/3}$. Between these limits,
$w=6t^5-15t^4+10t^3$ with $t=((x^2+y^2)/\epsilon^{2/3}-1)/3$.
When $w=0$, safe inactive arithmetic makes the correction zero.

The pressure-secant remainder satisfies $r=d-gx-by=(1-w)r_0$, and the
admissibility audit requires

$$
\left|r-(1-w)r_0\right|
\le4096\epsilon\left(1+|d|+|gx|+|by|\right).
$$

Full projection enforces the pressure secant to roundoff. Tiny jumps use the
smooth provisional limit; independently interpolated acoustic properties need
not be exact derivatives of the density interpolant. Density, endpoint sound
speed and Roe sound speed squared must be positive and finite, and projectors
must be finite. Acoustic waves use Harten entropy regularization; contact waves
retain their convective eigenvalue. Roe metadata and secant diagnostics are
saved with each applicable stage.

## Convergence and accuracy

The independent audit recomputes the original signed integrated spatial residual
and applies NumPy reductions using fixed conservation scales. Each equation
uses a volume-weighted RMS and satisfies relative $10^{-6}$ or absolute
$10^{-9}$. Boundary and global-conservation defects must separately satisfy
$10^{-9}$. All accepted states and reconstructed faces must satisfy the
[property-domain policy](TABLE_BOUNDS.md).

For a field $q$ with reference $q^*$, scale $s_q$ and cell volume $V_i$,

$$
L_1=\frac{\sum_iV_i|q_i-q_i^*|}{s_q\sum_iV_i},\qquad
L_2=\left(\frac{\sum_iV_i(q_i-q_i^*)^2}{s_q^2\sum_iV_i}\right)^{1/2},\qquad
L_\infty=\max_i\frac{|q_i-q_i^*|}{s_q}.
$$

Reference values are sampled at cell centers. The report's mass-flow error uses
the mean numerical face mass flux relative to the reference choked mass flow.
Conservation spread and physical cell-state mass-flow variation are reported
separately. Each grid has its own initial residual normalization and history.
