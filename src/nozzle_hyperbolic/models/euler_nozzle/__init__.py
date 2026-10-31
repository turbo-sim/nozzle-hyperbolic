"""General-EOS nozzle model for the shared conservative FVM machinery."""

__all__ = ["NozzleModel"]


def __getattr__(name):
    if name == "NozzleModel":
        from .model import NozzleModel

        return NozzleModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
