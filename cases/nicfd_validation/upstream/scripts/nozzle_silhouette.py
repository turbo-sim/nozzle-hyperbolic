"""
nozzle_silhouette.py — shared nozzle-radius silhouette.

`add_nozzle(ax, x, r, true_scale=...)` draws the grey nozzle-radius band behind
the data.
"""
import warnings
import numpy as np
from matplotlib.transforms import blended_transform_factory


def add_nozzle(ax, x, r, *, true_scale=False, height=0.16,
               color="0.45", alpha=0.22):
    """Draw the nozzle-radius silhouette behind the data on `ax`.

    Parameters
    ----------
    ax          : the host Axes.
    x, r        : position [m] and radius [m] arrays (r = sqrt(A/pi)).
    true_scale  : False → schematic 16% band (default); True → equal-aspect,
                  proportion-true overlay in metres.
    height      : schematic-mode band height as a fraction of the panel (0.16).
    color, alpha: silhouette fill style.

    Returns
    -------
    The overlay Axes when true_scale=True (so the caller can tweak it), else None.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)

    if not true_scale:
        # ── SCHEMATIC: radius → bottom `height` band, proportion NOT preserved ─
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        yr = (r / float(np.max(r))) * height
        ax.fill_between(x, 0.0, yr, transform=tr, color=color, alpha=alpha,
                        lw=0, zorder=0)
        return None

    # ── TRUE PROPORTION: equal-aspect overlay glued to the host rectangle ──────
    warnings.filterwarnings(
        "ignore",
        message="This figure includes Axes that are not compatible with tight_layout",
        category=UserWarning)
    fig = ax.figure
    band = fig.add_axes(ax.get_position().bounds, sharex=ax,
                        frameon=False, zorder=-5)
    band.set_aspect("equal", adjustable="box", anchor="S")
    band.fill_between(x, 0.0, r, color=color, alpha=alpha, lw=0)
    band.set_ylim(0.0, float(np.max(r)))
    band.axis("off")
    band.set_in_layout(False)   # aspect-locked overlay: keep it out of tight_layout
    # let the below-host overlay show through the host axes
    ax.patch.set_visible(False)

    def _sync(_evt=None, host=ax, follower=band):
        b = tuple(host.get_position().bounds)
        if getattr(follower, "_nz_last_host_box", None) != b:
            follower.set_position(b)
            follower._nz_last_host_box = b

    _sync()
    fig.canvas.mpl_connect("draw_event", _sync)
    return band
