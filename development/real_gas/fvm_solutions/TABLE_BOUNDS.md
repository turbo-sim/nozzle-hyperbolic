# Behavior at property-table bounds

For these verification calculations, every accepted cell, boundary trace and
reconstructed face state must remain inside the bicubic $(p,h)$ domain. The
production model rejects invalid trial states. Its Newton backtracking and
timestep retry logic attempt a smaller update; the study does not clip
thermodynamic properties or substitute another backend. Persistent failure is
reported with the original spatial residual and distance to each table bound.

A future bound-aware update limiter could choose a step fraction $\alpha$ so
that

$$
p_{\min}<p+\alpha\Delta p<p_{\max},\qquad
h_{\min}<h+\alpha\Delta h<h_{\max},\qquad 0<\alpha\le1.
$$

Reconstructed states and thermodynamic admissibility would still need checking;
componentwise cell bounds alone are insufficient for MUSCL. After accepting an
update, convergence must be tested using the original, unlimited conservation
residual. This is a possible solver improvement, not a change implemented by
this study.

Clamping only properties at an out-of-table state creates an extension of the
EOS that is different from the reference. It can also make density, enthalpy,
sound speed and their derivatives inconsistent. Such an explicit robustness
mode would need separate tests, a defined thermodynamic extension, counters
and warnings, and proof that limiting is inactive in the converged verification
solution. It is not enabled here.

Fluent documents two distinct mechanisms. Its general solution limits can
limit pressure and temperature, and its density-based positivity control can
reduce timesteps when temperature/pressure decreases too quickly; see
[Fluent solution limits](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_ug/flu_ug_sec_solve_limit.html).
For NIST real-gas lookup tables, Fluent documents direct NIST evaluations when
states fall outside the lookup range; this is a property-backend fallback,
rather than holding properties at their nearest table value. See
[Fluent NIST real-gas tables](https://ansyshelp.ansys.com/public/views/secured/corp/v252/en/flu_ug/flu_ug_sec_real_gas.html).
That fallback is outside this study's bicubic-only constraint.

The original tables ended at 32 MPa for the 30 MPa supercritical reservoir and
9 MPa for the 8.474 MPa condensation reservoir. Adapted cold-start attempts
reached those ceilings before satisfying steady conservation. Their complete
states and histories are retained. The final study extends the respective
ceilings to 60 and 18 MPa, retains 100 by 100 nodes, and recomputes each
continuum reference with its identical table. These are documented table
setup changes; they do not enlarge accepted domains through extrapolation or
change the underlying physical reservoir/nozzle problems.
