# Coverage relative to main's validation collection

`TurboExpo2025` reproduces the operating points and comparisons selected in
GT2025-153250. It is not a replacement for all of `main:validation_cases/`.

This comparison was checked against local `main` at
`9f9f07fe796034ac6c13284162f83d684fc3e5b0` and GitHub `main` at
`878420dc50680f8492eaac4fda047a8ac12e382b`. The latter revision was confirmed
against the remote on 3 November 2026. The validation trees differ only by four
generated Spinelli plot images.

| Main collection | Files on GitHub main | TurboExpo2025 coverage | Broader material |
|---|---:|---|---|
| `Barana2009` (Berana) | 18 | Paper's 90 bar, 318 K B1–B3 pressure cases, roughness variants and selected experimental/CFD workbooks | Additional original summaries, pressure workbooks, source article, figures and conversion/plotting scripts |
| `Simoneau1979` | 115 | Outside this paper's reproduction matrix | Nitrogen validation collection: 113 pressure-profile CSVs, summary workbook and source article |
| `Spinelli2019` | 13 | Three paper MDM cases S1–S3 and their pressure/compressibility references | Additional Mach-2 reference workbooks, full case summary, source article, plotting script and generated plots |
| `Zhu2019` | 3 | Outside this paper's reproduction matrix | Vortex-nozzle article and image, plus an OS metadata file |

The broader datasets are available in the current working tree at
[`../archive/validation/`](../archive/validation/). Every file in the four
dataset directories above was compared with GitHub main: binary assets match
exactly, and text files match after CRLF/LF normalization. This is a data
preservation check, not evidence that the archived drivers run on the current
solver. Four generated Spinelli plot images and the `Thumbs.db` OS file exist
locally but are ignored in the relocated archive; they remain recoverable from
the pinned main revision. This TurboExpo2025 commit does not change that archive.

Main also contains `validation_cases/notes.md` and the exploratory
`validation_cases/test_friction_factors.py`. The notes are preserved in the
archive, apart from line endings. The exploratory friction plotting script is
not part of the new case or the relocated archive. Current wall-law unit tests
live under `tests/models/euler_shared/`; that is not a claim that the historical
plotting script itself was ported.

For the Turbo Expo paper specifically, the case contains the full requested
operating-point/method matrix, settings, geometries, immutable selected source
workbooks, analytical comparisons, scalar tables, mesh studies, measured
results and standalone HTML evidence. The report's embedded ZIP retains all
83 attempts, states, histories and source snapshots. Generated outputs and
large property caches remain ignored as separate files.

The original historical solver arrays and original P1–P4 CFD arrays were not
available. Reviewed Figure 9 digitizations provide explicitly approximate,
partial pressure comparisons. Spinelli coordinates and some historical
geometry/scalar conventions remain uncertain. These limits, and the two
coarse Berana meshes that miss the shock regime, are documented in
[README.md](README.md), [RESULTS.md](RESULTS.md) and [REPORT.html](REPORT.html).
