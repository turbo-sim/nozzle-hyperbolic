

# Validation cases

## Perfect gas
- Spinelli nozzle with air at different exit pressures
- Adequate for modeling of perfect gas conditions. Simple case but verifies the model implementation is correct
- It is verification, not validation

## Real gas
- Experimental data from Spinelli et al.
- Relevant for modeling of single-phase ORC systems
- This is validation without shock waves

## Two-phase flows
- R134a data from Zhu and Elbel
  - We can only use cases with no swirl
  - Cases 1, 2, 7, 11 from Table 3 in J. Zhu and S. Elbel (2020)
  - Cases 1 and 2 have the same inlet conditions, but different exit pressures. Both cannot be adapted expansions, so we can use one to illustrate shock waves with the HEM. Also, maybe both are non-adapted, we can compare against the adaptes solution by setting a very low exit pressures
  - The findings would illustrate shock waves with HEM. Would shock waves structure not be affected by DEM since it is already in equilibrium when shock wave starts?
  - We can also do simulations for cases with positive inlet quality: Figure 8 in J. Zhu and S. Elbel (2019)
  - Nozzle might be too small? high friction losses?
  - We should include a friction factor with the Lockhard Martinelli parameter as Carlotta did.
  - Carlotta included several correlations in her study. Identify 1 or 2 that work well?
- CO2 from Berana
  - Experimental data with shock waves at the exit of the nozzle
  - It is condensation flows, so we cannot claim that it is similar to the conditions in PEORC? (some are condensation, some are flashing)
  - We could include it to claim that it is validation for the both cases of flashing and condensation?