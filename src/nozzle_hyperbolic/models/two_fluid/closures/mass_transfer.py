"""Liquid-vapor mass exchange from the interfacial heat balance.

Port reference: Alberto revision faafff704e54fd866abff95fcb88fc0a5217405c.
See docs/notes/two_fluid_migration.md for notation and retained policies.
"""


def calculate_mass_transfer_lv(
    T_l, T_v, T_sat, htc_l, htc_v, A_i, h_sat_l, h_sat_v, h_l, h_v, Qvi, Qli
):
    """Convert interfacial heat exchange to a liquid-vapor mass source.

    Parameters
    ----------
    T_l, T_v, T_sat : scalar
        Phase and saturation temperatures in K; retained in Alberto's signature.
    htc_l, htc_v : scalar
        Phase-side HTCs in W/(m**2 K); retained for call compatibility.
    A_i : scalar
        Interfacial area density in 1/m; retained for call compatibility.
    h_sat_l, h_sat_v : scalar
        Saturated liquid and vapor enthalpies in J/kg.
    h_l, h_v : scalar
        Phase enthalpies in J/kg; retained for call compatibility.
    Qvi, Qli : scalar
        Interfacial heat sources into vapor and liquid in W/m**3.

    Returns
    -------
    scalar
        Gas mass source in kg/(m**3 s); positive for evaporation.

    Notes
    -----
    The heat sources are already assembled by the caller. Only ``Qvi``,
    ``Qli`` and the saturation enthalpies enter this expression. Both transfer
    directions use the same positive saturation latent heat; no clipping occurs.
    """
    # Eq. (4): m_dot = -[htc_l * ai * (Tsat - Tl) + htc_v * ai * (Tsat - Tv)] / (h_qs - h_ps)
    # Distribute the minus sign:
    # numerator = -htc_l * A_i * (T_sat - T_l) - htc_v * A_i * (T_sat - T_v)

    numerator = -Qvi - Qli

    # The retained law uses saturation latent heat in both transfer directions.
    denominator = h_sat_v - h_sat_l

    m_dot = numerator / denominator

    return m_dot
