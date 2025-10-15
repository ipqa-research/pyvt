"""
Simulation of a Constant Composition Expansion Experiment (CCE)
"""

import numpy as np

from yaeos.core import ArModel


def cce(
    model: ArModel, z: np.ndarray, sat: dict, P0: float, Pf: float
) -> dict:
    """Constant Composition Expansion (CCE) simulation.

    Starts from a starting pressure `P0` and goes up to a higher final pressure
    `Pf`.  For a conventional oil, it will start in the two-phase region and
    end in the single-phase region. Flashes will be performed in the two-phase
    region and volumes will be calculated in the single-phase region.

    Parameters
    ----------
    model : ArModel
        An `ArModel` object with the equation of state and interaction
        parameters already set.
    z : np.ndarray
        Overall composition of the fluid.
    sat : dict
        Saturation point dictionary as returned by `model.saturation_point`.
    P0 : float
        Initial pressure of the CCE simulation. Must be lower than the
        saturation pressure.
    Pf : float
        Final pressure of the CCE simulation. Must be higher than the
        saturation pressure.

    Returns
    -------
    dict
        A dictionary with the following
        keys and values:
        - `P2`: np.ndarray
            Pressures in the two-phase region.
        - `Vx2`: np.ndarray
            Molar volumes of the liquid phase in the two-phase region.
        - `Vy2`: np.ndarray
            Molar volumes of the vapor phase in the two-phase region.
        - `Vrel`: np.ndarray
            Molar volumes of the overall fluid in the two-phase region.
        - `P1`: np.ndarray
            Pressures in the single-phase region.
        - `V1`: np.ndarray
            Molar volumes of the fluid in the single-phase region.
    """

    T = sat["T"]
    Psat = sat["P"]
    P = P0

    two_phase_steps = []
    Voils = []
    mono_ps = []

    # Iniciamos calculando flashes y aumentando presión
    # hasta alcanzar/pasar la presión de saturación.
    while P < Psat:
        step = model.flash_pt(z, pressure=P, temperature=T)
        if (Psat - P) < 7:
            P += 1
        else:
            P += 5

        two_phase_steps.append(step)

    # We go a bit above the saturation pressure to ensure correct calculations
    P += 2

    # Volume calculations until reaching the final pressure
    while P < Pf:
        Voil = model.volume(z, pressure=P, temperature=T, root="liquid")
        Voils.append(Voil)
        mono_ps.append(P)
        P += 5

    # Reorder the data to return relevant information
    p2 = np.array([step["P"] for step in two_phase_steps])
    vxs = np.array([step["Vx"] for step in two_phase_steps])
    vys = np.array([step["Vy"] for step in two_phase_steps])
    vrel = np.array(
        [
            step["beta"] * step["Vy"] + (1 - step["beta"]) * step["Vx"]
            for step in two_phase_steps
        ]
    )

    V1 = np.array(Voils)
    p1 = np.array(mono_ps)

    return {"P2": p2, "Vx2": vxs, "Vy2": vys, "Vrel": vrel, "P1": p1, "V1": V1}
