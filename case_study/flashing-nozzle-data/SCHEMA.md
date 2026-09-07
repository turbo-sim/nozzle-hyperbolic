# SCHEMA

The contract every file in `sources/` must satisfy. `scripts/build_index.py --strict` enforces
the parts of it that a machine can check.

## Conventions that apply to every file

**Units are SI, always, and the unit is in the column name.** `p0_Pa`, not `p0` with a note
somewhere. No bar, no psi, no mm, no degC. Convert at digitisation time and record the original
unit in a `#` comment.

**Encoding** is UTF-8, **newline** is `\n`, **separator** is `,`, **decimal mark** is `.`.

**Missing values are empty**, not `NaN`, `-`, `999` or `N/A`.

**Comment lines.** Any number of lines starting with `#` may precede the header row. They are
provenance notes and are skipped by every reader in `scripts/`. A `#` line *after* the header is
not allowed — it would silently drop a data row in a naive reader.

```csv
# digitised from Fig. 7, WebPlotDigitizer 4.7, 2026-08-28
# original units: x [mm], p [bar]
x_m,p_Pa,p_Pa_unc
0.0000,5.01e5,
```

**Uncertainty columns** are the value column plus `_unc`: `p0_Pa_unc` is the absolute (not
relative) 1-sigma uncertainty on `p0_Pa`, in the same unit. Always optional; leave empty when
the source does not report one. Never guess one.

**Identifiers** (`case_id`, `nozzle_id`, `bibkey`, file stems) are lowercase
`[a-z0-9_]`, no spaces, no hyphens inside an identifier. Hyphens are reserved for the
repository folder name itself.

## `sources/<bibkey>/meta.yaml`

Top-level scalar keys only — `build_index.py` ships a deliberately dumb parser so that PyYAML is
not a dependency. Nesting is allowed for `notes` (block scalar) but nothing else is read.

| key | required | meaning |
| --- | --- | --- |
| `bibkey` | yes | citation key, must equal the folder name |
| `fluid` | yes | CoolProp fluid name, e.g. `Water`, `CO2`, `R134a` |
| `doi` | yes | DOI, or empty if the source has none (reports often do not) |
| `report_id` | no | report number when there is no DOI, e.g. `NUREG/CR-1864` |
| `reference` | yes | full human-readable citation, one line |
| `x_origin` | yes | where `x = 0` sits and which way it grows, e.g. `throat, positive downstream` |
| `x_units_original` | no | the unit in the paper before conversion |
| `digitizer` | yes | who digitised it, with what tool, when |
| `verified_by` | no | second person who checked the numbers; empty means nothing here is `verified` |
| `notes` | yes | anything a future reader needs: figure quality, known errata, sign conventions |

`x_origin` is required because a digitised axial coordinate with an unstated origin cannot be
compared against a simulation. Say it explicitly even when it seems obvious.

## `sources/<bibkey>/cases.csv`

One row per experimental run.

| column | required | unit | meaning |
| --- | --- | --- | --- |
| `case_id` | yes | — | unique **within the source**; the index prefixes it with the bibkey |
| `nozzle_id` | yes | — | must match a file `geometry/<nozzle_id>.csv` |
| `fluid` | yes | — | CoolProp name; normally equal to `meta.fluid` |
| `p0_Pa` | yes | Pa | inlet **stagnation** pressure |
| `T0_K` | yes | K | inlet **stagnation** temperature |
| `subcooling_K` | no | K | `T_sat(p0) - T0`; positive means subcooled liquid at inlet |
| `p_back_Pa` | no | Pa | imposed downstream / back pressure |
| `p_exit_Pa` | no | Pa | measured static pressure at the nozzle exit plane |
| `mdot_kg_s` | no | kg/s | measured mass flow rate |
| `G_throat_kg_m2_s` | no | kg/(m^2 s) | mass flux at the throat, as reported |
| `choked` | no | — | `true` / `false` / empty when the source does not say |
| `profile_p` | no | — | file stem in `profiles/`, without `.csv` |
| `profile_T` | no | — | file stem in `profiles/`, without `.csv` |
| `figure_ref` | no | — | where in the paper this row came from, e.g. `Fig. 12b` |
| `status` | yes | — | `verified`, `provisional`, or `placeholder` |
| `notes` | no | — | free text |

Any `<col>_unc` column is accepted alongside its base column.

`status` semantics:

- `placeholder` — structure only, the numbers are not real. Never use for validation.
- `provisional` — digitised once, not checked. Usable, but say so in the paper.
- `verified` — a second person re-read it against the source and `meta.verified_by` is filled.

## `sources/<bibkey>/geometry/<nozzle_id>.csv`

The nozzle wall, as a table. Axisymmetric nozzles give `r_m`; rectangular ones give `h_m` and
`w_m`. Give `A_m2` when the source states an area directly; otherwise leave it empty and let the
consumer compute it.

| column | required | unit | meaning |
| --- | --- | --- | --- |
| `x_m` | yes | m | axial coordinate, origin as declared in `meta.x_origin`, monotonically increasing |
| `r_m` | one of | m | wall radius (axisymmetric) |
| `h_m`, `w_m` | one of | m | channel height and width (rectangular) |
| `A_m2` | no | m^2 | flow area when reported directly |

The throat is *not* flagged by a column — it is wherever the area is minimum. If the source
reports a throat diameter that disagrees with the digitised curve, say so in `meta.notes` and
trust the reported diameter.

## `sources/<bibkey>/profiles/<file_id>.csv`

One measured axial profile of one quantity.

File name, by convention: `<src><yy>_<nozzle tag>_<run>_<quantity>.csv`, e.g.
`ab81_a_003_p.csv` = Abuaf 1981, nozzle `a`, run `003`, pressure. The nozzle tag is a short form
of `nozzle_id` (`nozzle_a` -> `a`); `<quantity>` is one of `p` (static pressure), `t`
(temperature), `a` (void fraction), `v` (velocity). The name is a convention for humans — what
`build_index.py` actually enforces is that the file referenced by `profile_p` / `profile_T`
exists.

| column | required | unit | meaning |
| --- | --- | --- | --- |
| `x_m` | yes | m | axial coordinate, same origin as the geometry file |
| `p_Pa` | quantity `p` | Pa | static pressure |
| `T_K` | quantity `t` | K | temperature |
| `void_fraction` | quantity `a` | — | 0 to 1 |
| `u_m_s` | quantity `v` | m/s | velocity |

Rows are sorted by `x_m`. A profile file holds exactly one run and one quantity; do not put two
runs side by side in wide format, since it makes the per-point uncertainty columns ambiguous.

## Generated files

`index/cases_index.csv` is `build_index.py` output: every `cases.csv` concatenated, with
`bibkey`, `doi` and a globally unique `uid = <bibkey>__<case_id>` prepended. It is derived —
never edit it, never commit it.
