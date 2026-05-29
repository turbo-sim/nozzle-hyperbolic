## 4. Fully implicit time integration

### 4.1 Backward Euler (implicit Euler)

The fully implicit backward Euler update requires finding $\mathbf{Q}^{n+1}$ satisfying:

$$\mathbf{Q}_i^{n+1} = \mathbf{Q}_i^n + \Delta t\,\mathcal{L}_i(\mathbf{Q}^{n+1})$$

where $\mathcal{L} = \mathcal{R} + \mathbf{S}_A/A$ is the full spatial operator evaluated at the **unknown** state. This is a nonlinear system; linearising around $\mathbf{Q}^n$ gives:

$$\mathcal{L}(\mathbf{Q}^{n+1}) \approx \mathcal{L}(\mathbf{Q}^n) + \frac{\partial \mathcal{L}}{\partial \mathbf{Q}}\bigg|^n (\mathbf{Q}^{n+1} - \mathbf{Q}^n)$$

Substituting and rearranging:

$$\left(\mathbf{I} - \Delta t\,\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}\bigg|^n\right)(\mathbf{Q}^{n+1} - \mathbf{Q}^n) = \Delta t\,\mathcal{L}(\mathbf{Q}^n)$$

This is a linear system solved at each time step. The diagonal implicit scheme of §3.5 is a further approximation that retains only the diagonal of the Jacobian $\partial\mathcal{L}/\partial\mathbf{Q}$, avoiding the full matrix solve. The Jacobian $\partial\mathcal{L}/\partial\mathbf{Q}$ can be calculated using JAX.

### 4.2 Full nonlinear solve via root finding

Without linearisation, define the residual over all $N_x$ cells:

$$\mathbf{G}(\mathbf{Q}^{n+1}) = \mathbf{Q}^{n+1} - \mathbf{Q}^n - \Delta t\,\mathcal{L}(\mathbf{Q}^{n+1}) = \mathbf{0}$$

This nonlinear system can be solved with Newton or Gauss-Newton methods (e.g. via the `optimistix` library), with the exact Jacobian $\partial\mathbf{G}/\partial\mathbf{Q}$ computed automatically by JAX (no need to implement the Jacobian calculation, it is handled by optimistix)

### 4.3 Direct steady-state solve

Since the steady-state solution satisfies $\mathcal{L}(\mathbf{Q}^{\infty}) = \mathbf{0}$ regardless of $\Delta t$, two equivalent formulations are available:

**Large $\Delta t$ limit.** Dividing the residual $\mathbf{G}$ by $\Delta t$ and taking $\Delta t \to \infty$:

$$\frac{\mathbf{G}}{\Delta t} = \frac{\mathbf{Q}^{n+1} - \mathbf{Q}^n}{\Delta t} - \mathcal{L}(\mathbf{Q}^{n+1}) \xrightarrow{\Delta t\to\infty} -\mathcal{L}(\mathbf{Q}^{n+1}) = \mathbf{0}$$
Equivalently, one can pose the steady-state problem directly as:

$$ \mathcal{L}(\mathbf{Q}) = \mathbf{0}$$
and apply the root finder without any reference to a time step. Many people prefer the pseudo-transient continuation approach (large but finite $\Delta t$, see 4.1 and 4.2) because it provides better conditioned Jacobians and a natural continuation path from the initial condition to the solution.