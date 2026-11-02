"""
Two-fluid (six-equation) run of the inverse-designed CO2 nozzle.
=================================================================

Thin wrapper around dev_scripts/main_two_fluid.py, so the case always runs on
the maintained driver (same settings format, `base:` defaults chain, run
record, plots).  This folder only holds the case:

    settings.yaml   the case definition (builds on dev_scripts/settings/
                    defaults_flashing_cold_start.yaml)
    design_hem/     the nozzlex inverse-design output, incl. the geometry
                    nozzle_geometry_case_3.csv read by get_nozzle_from_csv
    fluid_tables/   EOS tables, built on the first run (tens of minutes)
    results/        one timestamped folder per run

    python main.py                 # settings.yaml
    python main.py other.yaml      # another settings file in this folder

Relative paths in the settings file (eos table_dir, geometry file_path) are
resolved from this folder.  The results folder is set by save_options.repo_main,
which main_two_fluid.py resolves from dev_scripts/ - hence the '../cases/...'
there.

Environment: the project's poetry venv (jax, jaxprop, nurbspy, CoolProp 7.2).
"""

import os
import sys
import pathlib
import runpy

HERE = pathlib.Path(__file__).resolve().parent
DEV_SCRIPTS = HERE.parents[1] / "dev_scripts"
DRIVER = DEV_SCRIPTS / "main_two_fluid.py"

settings = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "settings.yaml")
if not settings.is_absolute():
    settings = HERE / settings
if not settings.exists():
    raise SystemExit(f"settings file not found: {settings}")

os.environ["SETTINGS_YAML"] = str(settings)   # absolute -> overrides the driver's default
os.chdir(HERE)                                 # table_dir / geometry file_path resolve here
sys.path.insert(0, str(DEV_SCRIPTS))           # config_loader, initial_conditions, run_info
sys.argv = [str(DRIVER)]

runpy.run_path(str(DRIVER), run_name="__main__")
