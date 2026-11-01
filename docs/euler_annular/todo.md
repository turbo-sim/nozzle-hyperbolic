# Annular development backlog draft

These planning notes have not been reconciled with the current implementation.
Wall shear, prescribed wall heat flux and automatic outlet selection are already
implemented; see the [physical closures](../physical_closures.md) and
[model interface](../model_interface.md) guides. The campaign behind the counts
below is not identified, so those counts are not current validation claims.
Consult the owning study reports before treating an item as missing capability.

Original notes, retained for review:
Priority	Missing or incomplete
1. Wall friction	Compute wall shear and apply meridional drag and angular-momentum torque, with viscosity, Reynolds number and roughness inputs.
2. Heat transfer	Add prescribed wall heat flux and/or a wall-temperature/heat-transfer-coefficient model to the energy balance.
3. Additional diffuser losses	Implement the documented diffusion and curvature loss correlations. Boundary-layer blockage and separation/stall prediction are also absent.
4. Accuracy improvements	Resolve the sonic-exit discrepancy and shock-profile overshoot. The saved campaign converged 72/72 discrete solutions, but passed only 11/18 finest-grid accuracy targets.
5. Broader validation	Validate real-fluid diffuser applications. Reverse flow and automatic boundary-regime switching remain unsupported.
