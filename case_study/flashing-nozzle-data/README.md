# flashing-nozzle-data

A small, plain-text archive of **experimental flashing / cavitating nozzle data** digitised from
the literature, in a shape that model-validation scripts can consume without hand editing.

Everything is CSV + YAML. No binary formats, no spreadsheets, no notebooks. The point is that a
diff shows what changed and a `grep` finds what you need ten years from now.

## Layout

```
flashing-nozzle-data/
├── README.md              this file
├── SCHEMA.md              the contract: column names, units, file naming, required fields
├── sources/               one folder per publication (the only place data lives)
│   └── <bibkey>/
│       ├── meta.yaml      provenance: doi, fluid, bibkey, x-origin, digitizer, notes
│       ├── cases.csv      one row per experimental run
│       ├── geometry/      one CSV per nozzle used in that publication
│       └── profiles/      one CSV per measured axial profile
├── scripts/
│   ├── build_index.py     concatenates every sources/*/cases.csv into one table
│   └── plot_ts.py         regenerates every T-s figure
├── papers/                local PDFs (git-ignored, see below)
└── .gitignore             excludes papers/, index/ and figures/
```

Two directories are **generated** and are not committed:

- `index/cases_index.csv` — written by `build_index.py`
- `figures/` — written by `plot_ts.py`

## Quickstart

```bash
cd case_study/flashing-nozzle-data

# validate every source folder and build the master case table
python scripts/build_index.py

# stricter: turn warnings into a non-zero exit code (use this in CI)
python scripts/build_index.py --strict

# regenerate the T-s diagrams into figures/
python scripts/plot_ts.py
python scripts/plot_ts.py --skip-placeholder      # only real data
python scripts/plot_ts.py --case abuaf1981__run_003
python scripts/plot_ts.py --full-dome             # show the vapour branch as well
```

`build_index.py` is standard library only. `plot_ts.py` needs `CoolProp` and `matplotlib`,
both of which are already dependencies of `nozzle_hyperbolic`. It builds the index itself if
one is missing.

### What the T-s figure shows

The saturation dome, the inlet stagnation state, the equilibrium isentrope down to the lowest
pressure the case reports, the point where that isentrope crosses saturation, and the measured
pressure taps mapped onto the isentrope. The dashed red branch is the *metastable* liquid: the
same isentrope continued below the saturation pressure on the liquid root of the equation of
state, which is the path a subcooled flashing flow actually follows. It looks degenerate — an
isentropic metastable liquid barely changes temperature — so the endpoint marker is what
carries the number: the liquid superheat at the exit pressure.

Only the pressure on those figures is measured. Every temperature is a property evaluation.

## Adding a new source

1. `mkdir -p sources/<bibkey>/{geometry,profiles}` where `<bibkey>` is the citation key you use
   in the paper, lowercase, e.g. `abuaf1981`.
2. Write `meta.yaml`. Copy `sources/abuaf1981/meta.yaml` and replace every field. The
   `x_origin` field is not optional — a digitised axial coordinate is meaningless without it.
3. Write `cases.csv` with the columns in [SCHEMA.md](SCHEMA.md). One row per run.
4. Drop the nozzle geometry in `geometry/<nozzle_id>.csv` and the measured profiles in
   `profiles/<file_id>.csv`.
5. Run `python scripts/build_index.py --strict` and fix whatever it complains about.

## Data status and honesty rules

Digitised data is a *derived* product and it is only as good as the digitiser's hand. Two rules
keep that honest:

- Every `cases.csv` row carries a `status` column: `verified`, `provisional` or `placeholder`.
  Nothing is `verified` until a second person has re-read the numbers against the source.
- Every CSV may carry `#` comment lines **before** the header row. Use them to record where the
  numbers came from (figure number, table number, digitiser tool). All readers in `scripts/`
  skip them. See [SCHEMA.md](SCHEMA.md).

> **Current state:** `sources/abuaf1981` is a *scaffold*. Its structure, units and provenance
> fields are real; its numeric rows are marked `status = placeholder` and must be replaced with
> values read off the actual report before any of it is used for validation.

## papers/

Put the source PDFs here so the digitiser can check a number without hunting. They are **not**
committed — most are copyrighted — and the local `.gitignore` already excludes them, along with
`index/` and `figures/`. The `doi` / `report_id` fields in each `meta.yaml` are what make a
source findable; the PDF is just a local convenience.
