# Property-table domain

Every accepted physical cell, boundary trace and reconstructed face must remain
within the JAXProp bicubic $(p,h)$ domain:

$$
p_{\min}\le p\le p_{\max},\qquad h_{\min}\le h\le h_{\max}.
$$

| Case | Pressure | Enthalpy | Nodes |
|---|---:|---:|---:|
| Supercritical CO2 | 4–60 MPa | 260–600 kJ/kg | 100 × 100 |
| Equilibrium condensation | 1–18 MPa | 250–520 kJ/kg | 100 × 100 |

The production model rejects inadmissible trial states. Newton backtracking and
timestep retries reduce the attempted update. Properties are evaluated directly
from the bicubic tables throughout the accepted domain, without clipping,
extrapolation or substitution of another property backend.

The continuous reference and FVM solution share the same table bytes for each
case. Final audits check table bounds for the full numerical state and
reconstructed faces, positive finite thermodynamic properties and the original
conservation residual. Table settings and SHA-256 fingerprints are saved with the
reference and campaign data.
