"""
Thin wrapper: loads settings_phase_change.yaml and re-exports every attribute
at module level so the driver script can use

    import settings_phase_change as cfg
    cfg.P_INIT, cfg.H_L_OFFSET, ...
"""
import sys
import pathlib
from config_loader import load_config as _load_config

_here = pathlib.Path(__file__).parent
_cfg  = _load_config(_here / "settings_phase_change.yaml")

_this = sys.modules[__name__]
for _k, _v in vars(_cfg).items():
    setattr(_this, _k, _v)
