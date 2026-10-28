"""
Thin wrapper: loads settings_two_fluid.yaml and re-exports every attribute
at module level so existing code can continue to use

    import settings_two_fluid as cfg
    cfg.K_VM, cfg.LUT_G, cfg.USE_RIEMANN_IC, ...

Edit settings_two_fluid.yaml to change any parameter.
"""
import sys
import pathlib
from development.alberto.two_fluid.config_loader import load_config as _load_config

_here = pathlib.Path(__file__).parent
_cfg  = _load_config(_here / "settings_two_fluid.yaml")

_this = sys.modules[__name__]
for _k, _v in vars(_cfg).items():
    setattr(_this, _k, _v)
