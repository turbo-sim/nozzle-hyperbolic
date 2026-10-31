"""
Metastability restriction on the interfacial mass-transfer closure.

Stadtke (Gasdynamic Aspects of Two-Phase Flow, eq. 3.34-3.36, repeated for the
ATFM code as eq. 8.18-8.20) writes the evaporation rate as

    sigma_Mg = -sigma_Ml = a_int / dh_s * ( H_g^q * dT_g + H_l^q * dT_l )

but takes ONLY THE METASTABLE CONTRIBUTIONS in the driving potentials:

    dT_g = min(T_g - T_sat, 0)      nonzero only for SUBCOOLED   vapour
    dT_l = max(T_l - T_sat, 0)      nonzero only for SUPERHEATED liquid

i.e. a phase drives phase change only when it sits on its METASTABLE side of the
saturation line.  A superheated vapour and a subcooled liquid are both
thermodynamically stable, so in that quadrant the closure gives sigma_Mg = 0 by
construction.

`calculate_mass_transfer_lv` implements the same formula with the RAW
temperature differences and no clamps, so it deviates from Stadtke in three of
the four quadrants.  The tests below pin that: the agreement in the
doubly-metastable quadrant is asserted normally, the three deviations are
xfail(strict=True) so the suite stays green today and flags loudly the moment a
clamp is introduced.

Sign convention throughout: sigma_Mg > 0 is evaporation (liquid -> gas).
"""
import numpy as np
import pytest

from nozzle_hyperbolic.fvm_solver.source_two_fluid import calculate_mass_transfer_lv


# --------------------------------------------------------------------------
# Reference: Stadtke's clamped closure
# --------------------------------------------------------------------------

def stadtke_sigma_Mg(T_g, T_l, T_sat, htc_g, htc_l, a_int, dh_s):
    """Stadtke eq. 3.34-3.36 with the metastability clamps applied."""
    dT_g = min(T_g - T_sat, 0.0)          # subcooled vapour only
    dT_l = max(T_l - T_sat, 0.0)          # superheated liquid only
    return a_int * (htc_g * dT_g + htc_l * dT_l) / dh_s


def code_sigma_Mg(T_g, T_l, T_sat, htc_g, htc_l, a_int, dh_s):
    """The closure as the solver evaluates it (sources.py:307-311)."""
    Qvi = htc_g * a_int * (T_sat - T_g)
    Qli = htc_l * a_int * (T_sat - T_l)
    return float(calculate_mass_transfer_lv(
        T_l, T_g, T_sat, htc_l, htc_g, a_int,
        H_SAT_L, H_SAT_G, 0.0, 0.0, Qvi, Qli,
    ))


# Representative water/steam numbers at ~10 bar; only the signs and the relative
# magnitudes matter for these tests.
T_SAT   = 453.0          # K
H_SAT_G = 2.778e6        # J/kg
H_SAT_L = 7.626e5        # J/kg
DH_S    = H_SAT_G - H_SAT_L
HTC_G   = 1.2e4          # W/(m^2 K)
HTC_L   = 8.0e3          # W/(m^2 K)
A_INT   = 5.0e3          # 1/m

_kw = dict(T_sat=T_SAT, htc_g=HTC_G, htc_l=HTC_L, a_int=A_INT, dh_s=DH_S)

# The four quadrants of (vapour, liquid) relative to saturation.
#   name                       T_g            T_l
QUADRANTS = {
    "both_metastable":        (T_SAT - 30.0, T_SAT + 5.0),   # subcooled g, superheated l
    "vapour_stable":          (T_SAT + 30.0, T_SAT + 5.0),   # superheated g, superheated l
    "liquid_stable":          (T_SAT - 30.0, T_SAT - 5.0),   # subcooled g, subcooled l
    "both_stable":            (T_SAT + 30.0, T_SAT - 5.0),   # superheated g, subcooled l
}


def test_reference_and_code_share_the_same_algebra():
    """Both forms must be the SAME expression away from the clamps."""
    T_g, T_l = QUADRANTS["both_metastable"]
    assert code_sigma_Mg(T_g, T_l, **_kw) == pytest.approx(
        stadtke_sigma_Mg(T_g, T_l, **_kw), rel=1e-12)


def test_doubly_metastable_quadrant_agrees():
    """Subcooled vapour + superheated liquid: neither clamp fires."""
    T_g, T_l = QUADRANTS["both_metastable"]
    sig = code_sigma_Mg(T_g, T_l, **_kw)
    assert sig == pytest.approx(stadtke_sigma_Mg(T_g, T_l, **_kw), rel=1e-12)
    # both drivers pull the same way here: net condensation, gas term dominates
    assert sig < 0.0


def test_superheated_liquid_alone_drives_evaporation():
    """Sanity on the sign convention: superheated liquid at T_g = T_sat."""
    assert code_sigma_Mg(T_SAT, T_SAT + 5.0, **_kw) > 0.0
    assert stadtke_sigma_Mg(T_SAT, T_SAT + 5.0, **_kw) > 0.0


def test_subcooled_vapour_alone_drives_condensation():
    """Sanity on the sign convention: subcooled vapour at T_l = T_sat."""
    assert code_sigma_Mg(T_SAT - 30.0, T_SAT, **_kw) < 0.0
    assert stadtke_sigma_Mg(T_SAT - 30.0, T_SAT, **_kw) < 0.0


@pytest.mark.xfail(strict=True, reason=(
    "calculate_mass_transfer_lv uses the RAW (T_g - T_sat); Stadtke clamps it "
    "with min(.,0) so a SUPERHEATED (stable) vapour contributes nothing."))
def test_superheated_vapour_must_not_drive_evaporation():
    T_g, T_l = QUADRANTS["vapour_stable"]
    assert code_sigma_Mg(T_g, T_l, **_kw) == pytest.approx(
        stadtke_sigma_Mg(T_g, T_l, **_kw), rel=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "calculate_mass_transfer_lv uses the RAW (T_l - T_sat); Stadtke clamps it "
    "with max(.,0) so a SUBCOOLED (stable) liquid contributes nothing."))
def test_subcooled_liquid_must_not_drive_condensation():
    T_g, T_l = QUADRANTS["liquid_stable"]
    assert code_sigma_Mg(T_g, T_l, **_kw) == pytest.approx(
        stadtke_sigma_Mg(T_g, T_l, **_kw), rel=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "Superheated vapour + subcooled liquid: BOTH phases stable, so Stadtke's "
    "closure gives sigma_Mg = 0 identically.  The unclamped code returns a "
    "finite rate whose sign is set by whichever raw dT term is larger."))
def test_both_phases_stable_gives_zero_mass_transfer():
    T_g, T_l = QUADRANTS["both_stable"]
    assert stadtke_sigma_Mg(T_g, T_l, **_kw) == 0.0        # reference is exact
    assert code_sigma_Mg(T_g, T_l, **_kw) == pytest.approx(0.0, abs=1e-12)


def test_deviation_is_confined_to_the_clamped_quadrants():
    """Map the whole quadrant table: exactly one quadrant may agree."""
    agree = {
        name: np.isclose(code_sigma_Mg(T_g, T_l, **_kw),
                         stadtke_sigma_Mg(T_g, T_l, **_kw), rtol=1e-9)
        for name, (T_g, T_l) in QUADRANTS.items()
    }
    assert agree["both_metastable"] is np.True_ or agree["both_metastable"]
    assert not any(v for k, v in agree.items() if k != "both_metastable"), (
        f"expected deviation in the three clamped quadrants, got {agree}")
