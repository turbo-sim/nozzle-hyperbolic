# NICFD nitrogen–water: 50–400-cell comparison

Open [REPORT.html](REPORT.html) for interactive Plotly profiles, published/new/both
visibility controls, mesh/regime selectors, difference plots, histories and downloads.

Every new mesh starts independently from the published profile. The physical model,
MUSCL Van Albada reconstruction, Städtke compatibility flux, wb NC pairing, void filter,
coupled backward-Euler solve, 20-correction Newton cap and stopping criteria match
the previous fine-grid comparison. Only the mesh count changes.

| Regime | Physical cells | Result | Pressure RMSE [bar] | Gas / liquid velocity RMSE [m/s] | Accepted / rejected |
|---|---:|---|---:|---:|---:|
| subsonic | 50 | verified | 0.179619 | 4.209 / 1.619 | 33 / 0 |
| subsonic | 100 | verified | 0.0790947 | 1.87 / 0.9006 | 36 / 0 |
| subsonic | 200 | verified | 0.0416195 | 1.06 / 0.6988 | 40 / 0 |
| subsonic | 400 | verified | 0.0267963 | 0.7716 / 0.512 | 43 / 0 |
| supersonic_shock | 50 | stagnation | 0.314934 | 13.58 / 1.08 | 77 / 61 |
| supersonic_shock | 100 | verified | 0.137682 | 5.196 / 1.016 | 35 / 1 |
| supersonic_shock | 200 | stagnation | 0.0908363 | 3.861 / 0.2019 | 64 / 57 |
| supersonic_shock | 400 | verified | 0.0446022 | 1.967 / 0.7997 | 47 / 2 |
| supersonic_adapted | 50 | max_steps | 0.125191 | 2.849 / 0.5425 | 500 / 137 |
| supersonic_adapted | 100 | verified | 0.0774139 | 2.005 / 0.2758 | 37 / 0 |
| supersonic_adapted | 200 | stagnation | 0.0385979 | 0.9954 / 0.4271 | 85 / 62 |
| supersonic_adapted | 400 | verified | 0.0125119 | 0.3589 / 0.2067 | 44 / 0 |

The published reference has 500 total rows, including two endpoint ghosts. Its
498 interior samples are comparison data; new meshes contain the stated number of
physical cells plus two endpoint traces. Pressure errors compare point profiles
using bounded linear interpolation, not exact cell averages. Errors of unfinished
states are retained as diagnostics. No grid-independence claim is made.

Previous investigation records and the former 498-cell report remain in `outputs/`.
The original cold-to-fine subsonic test found a numerically converged root with a
supersonic pocket. Independent cold-start recovery of the published branches remains
unverified; this mesh comparison uses explicit published-profile seeding.
