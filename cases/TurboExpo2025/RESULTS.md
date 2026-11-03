# Turbo Expo 2025 reproduction results

Saved results from the current `roberto-branch` HEM and implicit backward Euler. See [REPORT.html](REPORT.html) for fields, original-equation checks, every retained attempt, reference comparisons and the embedded evidence archive.

Primary configurations: **52/52 verified**, 52 attempted. Appendix mesh points: **23/25 verified**, 25 attempted. The campaign retains 83 attempts, including pilots and superseded settings.

Verification requires the original spatial equations, boundary equations, global flux balances, EOS admissibility and physical branch. It is separate from published-data agreement.

| Case | Paper cells | Method | Latest numerical outcome |
|---|---:|---|---|
| G_subsonic | 500 | rusanov | Verified steady root |
| G_subsonic | 500 | roe | Verified steady root |
| G_subsonic | 500 | hll | Verified steady root |
| G_adapted | 500 | rusanov | Verified steady root |
| G_adapted | 500 | roe | Verified steady root |
| G_adapted | 500 | hll | Verified steady root |
| G_shock_pr156 | 500 | rusanov | Verified steady root |
| G_shock_pr156 | 500 | roe | Verified steady root |
| G_shock_pr156 | 500 | hll | Verified steady root |
| G_shock_25 | 500 | rusanov | Verified steady root |
| G_shock_25 | 500 | roe | Verified steady root |
| G_shock_25 | 500 | hll | Verified steady root |
| G_shock_50 | 500 | rusanov | Verified steady root |
| G_shock_50 | 500 | roe | Verified steady root |
| G_shock_50 | 500 | hll | Verified steady root |
| G_shock_75 | 500 | rusanov | Verified steady root |
| G_shock_75 | 500 | roe | Verified steady root |
| G_shock_75 | 500 | hll | Verified steady root |
| S1 | 250 | rusanov | Verified steady root |
| S1 | 250 | hll | Verified steady root |
| S2 | 250 | rusanov | Verified steady root |
| S2 | 250 | hll | Verified steady root |
| S3 | 250 | rusanov | Verified steady root |
| S3 | 250 | hll | Verified steady root |
| L1 | 200 | rusanov | Verified steady root |
| L1 | 200 | hll | Verified steady root |
| L1_inviscid | 200 | rusanov | Verified steady root |
| L1_inviscid | 200 | hll | Verified steady root |
| L5 | 200 | rusanov | Verified steady root |
| L5 | 200 | hll | Verified steady root |
| L5_inviscid | 200 | rusanov | Verified steady root |
| L5_inviscid | 200 | hll | Verified steady root |
| P1 | 600 | rusanov | Verified steady root |
| P1 | 600 | hll | Verified steady root |
| P2 | 600 | rusanov | Verified steady root |
| P2 | 600 | hll | Verified steady root |
| P3 | 600 | rusanov | Verified steady root |
| P3 | 600 | hll | Verified steady root |
| P4 | 600 | rusanov | Verified steady root |
| P4 | 600 | hll | Verified steady root |
| B1 | 1600 | rusanov | Verified steady root |
| B1 | 1600 | hll | Verified steady root |
| B2 | 1600 | rusanov | Verified steady root |
| B2 | 1600 | hll | Verified steady root |
| B3 | 1600 | rusanov | Verified steady root |
| B3 | 1600 | hll | Verified steady root |
| B3_k0 | 1600 | rusanov | Verified steady root |
| B3_k0 | 1600 | hll | Verified steady root |
| B3_k1 | 1600 | rusanov | Verified steady root |
| B3_k1 | 1600 | hll | Verified steady root |
| B3_k3 | 1600 | rusanov | Verified steady root |
| B3_k3 | 1600 | hll | Verified steady root |

## Coarse-mesh flow regimes

These meshes satisfy the original steady equations but do not recover the requested shock regime. They remain recorded as converged discrete roots with failed physical-branch verification. They are excluded from a claim that every mesh reproduces the paper's regime.

| Case | Cells | Maximum interior Mach | Original equation targets | Intended shock regime |
|---|---:|---:|---|---|
| B3 | 100 | 0.860435 | Pass | Absent |
| B3 | 200 | 0.964743 | Pass | Absent |

## Published scalars

The selected ideal contour matches the published critical mass flow and outlet/throat area ratio. Its exact shock-entropy values differ from Table 1; numerical error against the selected contour and disagreement with the historical paper scalar are therefore reported separately.

| Case | Method | Current entropy rise [J/(kg K)] | Selected-contour exact | Paper | Difference from contour [%] |
|---|---|---:|---:|---:|---:|
| G_shock_25 | rusanov | 15.5264 | 15.7182 | 18.66 | -1.22019 |
| G_shock_25 | roe | 15.9068 | 15.7182 | 18.66 | 1.20021 |
| G_shock_25 | hll | 15.9264 | 15.7182 | 18.66 | 1.3247 |
| G_shock_50 | rusanov | 63.9681 | 64.2577 | 60.66 | -0.450764 |
| G_shock_50 | roe | 64.5437 | 64.2577 | 60.66 | 0.445005 |
| G_shock_50 | hll | 64.573 | 64.2577 | 60.66 | 0.490677 |
| G_shock_75 | rusanov | 100.229 | 100.639 | 105.79 | -0.406947 |
| G_shock_75 | roe | 101.037 | 100.639 | 105.79 | 0.395449 |
| G_shock_75 | hll | 101.077 | 100.639 | 105.79 | 0.435736 |

Lettieri's historical Table 3 values use half-channel flow per unit depth, despite the paper's kg/s label. Current full-channel flow and the comparable per-depth quantity are both shown.

| Case | Method | Full channel [kg/s] | Half channel per depth [kg/(s m)] | Paper HEM per depth | Difference [%] |
|---|---|---:|---:|---:|---:|
| L1 | rusanov | 0.379803 | 29.3058 | 29.37 | -0.218538 |
| L1 | hll | 0.376925 | 29.0838 | 29.41 | -1.10929 |
| L5 | rusanov | 0.638554 | 49.2711 | 49.06 | 0.430351 |
| L5 | hll | 0.626655 | 48.353 | 48.5 | -0.303054 |

## Visible Petruccelli paper curves

Figure 9's embedded raster plots supply independently reviewed, partial pressure-curve samples. They are approximate graph digitizations, with stroke/calibration uncertainty and visual overlays in the HTML evidence. Upstream overlaps, sharp jump interiors and ambiguous/merged tails are omitted. Only observed points are scored; these metrics do not describe complete historical profiles.

| Case | Current method | Visible reference | Samples | Pressure RMSE [bar] | RMSE / reservoir pressure [%] |
|---|---|---|---:|---:|---:|
| P1 | rusanov | Paper P1 Rusanov HEM (digitized) | 120 | 0.419627 | 0.32279 |
| P1 | rusanov | Paper P1 CFD (digitized) | 8 | 5.54212 | 4.26317 |
| P1 | hll | Paper P1 Rusanov HEM (digitized) | 120 | 1.35864 | 1.04511 |
| P1 | hll | Paper P1 CFD (digitized) | 8 | 5.44621 | 4.18939 |
| P3 | rusanov | Paper P3 Rusanov HEM (digitized) | 144 | 0.276899 | 0.276899 |
| P3 | rusanov | Paper P3 CFD (digitized) | 16 | 3.02104 | 3.02104 |
| P3 | hll | Paper P3 Rusanov HEM (digitized) | 144 | 1.09541 | 1.09541 |
| P3 | hll | Paper P3 CFD (digitized) | 16 | 2.89798 | 2.89798 |
| P2 | rusanov | Paper P2 Rusanov HEM (digitized) | 64 | 1.0033 | 0.771765 |
| P2 | rusanov | Paper P2 CFD (digitized) | 13 | 3.78457 | 2.91121 |
| P2 | hll | Paper P2 Rusanov HEM (digitized) | 64 | 3.50278 | 2.69445 |
| P2 | hll | Paper P2 CFD (digitized) | 13 | 9.19813 | 7.07549 |
| P4 | rusanov | Paper P4 Rusanov HEM (digitized) | 113 | 0.432821 | 0.432821 |
| P4 | rusanov | Paper P4 CFD (digitized) | 10 | 1.91657 | 1.91657 |
| P4 | hll | Paper P4 Rusanov HEM (digitized) | 113 | 0.686844 | 0.686844 |
| P4 | hll | Paper P4 CFD (digitized) | 10 | 1.87084 | 1.87084 |

Historical solver fields were not committed, and P1–P4 reference CFD arrays were not recovered. Spinelli's coordinate mapping remains uncertain. These gaps preclude a complete historical field-regression claim. All recoverable measured/CFD reference series and current discrepancies are shown in the HTML; no missing curve or uncertainty is fabricated.

Current source epochs and dirty implementation snapshots are retained with each attempt. The stable polynomial-volume integration correction was checked independently: the original spatial residual was bitwise unchanged and the corrected-volume residual norms still passed. Reported solve times are single compilation-inclusive observations; they establish no speedup.
