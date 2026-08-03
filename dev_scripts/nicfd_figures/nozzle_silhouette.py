"""
nozzle_silhouette.py — shared nozzle-radius silhouette for the journal figures.

`add_nozzle(ax, x, r, true_scale=...)` draws the grey nozzle-radius band behind
the data in one of two modes:

  true_scale=False  (DEFAULT — the historical behaviour)  ── SCHEMATIC.
      The radius is normalized to its own maximum and drawn to fill the bottom
      `height` fraction (16%) of the panel, via a blended (dataX, axesY)
      transform.  x is in true metres but the LENGTH:RADIUS proportion is NOT
      preserved.  This is intentional: a full-scale silhouette would either be
      unreadable or (for the non-dimensional symmetric validation nozzle, whose
      area numbers give r_max > L) taller than the panel is wide.

  true_scale=True  ── TRUE PROPORTION.
      The radius is drawn in metres on a frameless, equal-aspect overlay axis
      glued to `ax` (1 m along x  ==  1 m of radius), anchored to the bottom.
      Use this only for a physically long-and-thin nozzle — e.g. the Elliot
      geometry (r_max/L ~ 0.08 → a thin, honest strip).  For the symmetric
      placeholder (r_max > L) it would dominate the panel.

Both modes are drawn *behind* the data (zorder 0 / a below-host overlay), so the
host axes' patch is made transparent in the true-scale branch to let it show.
"""
import warnings
import numpy as np
from matplotlib.transforms import blended_transform_factory


def add_nozzle(ax, x, r, *, true_scale=False, height=0.16,
               color="0.45", alpha=0.22):
    """Draw the nozzle-radius silhouette behind the data on `ax`.

    Parameters
    ----------
    ax          : the host Axes (may be a twinx primary — the band is drawn
                  behind, so put it on whichever axis owns the bottom).
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
    # The overlay is a free (non-subplot) axes, so a later fig.tight_layout() in
    # the caller flags it as "not compatible with tight_layout".  Harmless here:
    # the draw-sync below re-glues the band onto the host rectangle AFTER layout
    # runs (verified: band box == host box).  Silence just that one message.
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

    # Keep the overlay locked onto the host rectangle across tight_layout /
    # resize / savefig(bbox="tight").  Guard on the LAST-APPLIED host box (not
    # the follower's own, which shrinks under equal aspect) → no redraw loop.
    def _sync(_evt=None, host=ax, follower=band):
        b = tuple(host.get_position().bounds)
        if getattr(follower, "_nz_last_host_box", None) != b:
            follower.set_position(b)
            follower._nz_last_host_box = b

    _sync()
    fig.canvas.mpl_connect("draw_event", _sync)
    return band