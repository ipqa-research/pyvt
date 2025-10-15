"""
Differential Liberation experiment (DL) module.
"""

import numpy as np

from pyvt.constants import P_STD, T_STD

from yaeos.core import ArModel


def dl(
    model: ArModel, z: np.ndarray, T: float, P0: float, bubble: dict
) -> dict:
    """Differential Liberation (DL) simulation.

    Starts from a starting pressure `P0` and goes down to a very low pressure
    (1 bar).  For a conventional oil, it will start in the single-phase region
    and end in the two-phase region. Volumes will be calculated in the
    single-phase region and flashes will be performed in the two-phase region.

    Parameters
    ----------
    model : ArModel
        An `ArModel` object with the equation of state and interaction
        parameters already set.
    z : np.ndarray
        Overall composition of the fluid.
    T : float
        Temperature of the DL simulation.
    P0 : float
        Initial pressure of the DL simulation. Must be higher than the
        bubble point pressure.
    bubble : dict
        Bubble point dictionary as returned by `model.bubble_point`.
    
    Returns
    -------
    dict
        A dictionary with the results of the DL simulation.
        The dictionary has the following keys and values:
        - `Voil`: np.ndarray
            Molar volumes of the oil phase in the DL simulation.
        - `Vgas`: np.ndarray
            Molar volumes of the gas phase in the DL simulation.
        - `Vgas_std`: np.ndarray
            Molar volumes of the gas phase at standard conditions in the DL
            simulation.
        - `Vres`: float
            Molar volume of the residual oil at standard conditions.
        - `moles_oil`: float
            Moles of oil remaining at the end of the DL simulation.
        - `P`: np.ndarray
            Pressures in the DL simulation.
        - `Rs`: list
    """

    P = P0

    n_oil = 1
    n_gas = 0

    bubble_P = bubble["P"]

    gas_volume = []
    gas_volume_std = []
    oil_volume = []
    pressure = []

    # ========================================================================
    # Monophasic region
    # ------------------------------------------------------------------------
    monopoints = 0
    while P > bubble_P:
        monopoints = monopoints + 1
        V = model.volume(moles=z, pressure=P, temperature=T, root="liquid")

        gas_volume.append(0)
        gas_volume_std.append(0)
        oil_volume.append(V)
        pressure.append(P)

        P = P - 5

    # ========================================================================
    # Phase-equilibria region
    # ------------------------------------------------------------------------
    P = bubble_P - 1
    x = z
    k0 = bubble["y"] / bubble["x"]

    moles_oil = []
    moles_gas = []

    pepoints = 0
    cumgas = 0
    while P > P_STD:
        pepoints = pepoints + 1

        # Solve the flash
        step = model.flash_pt(x, temperature=T, pressure=P, k0=k0)

        y = step["y"]
        x = step["x"]
        k0 = y / x

        # Mass balances of moles numbers
        n_gas = step["beta"] * n_oil
        n_oil -= step["beta"] * n_oil
        cumgas += n_gas

        # Save points
        Vgas_std = model.volume(
            y, pressure=P_STD, temperature=T_STD, root="vapor"
        )
        pressure.append(P)
        oil_volume.append(step["Vx"] * n_oil)
        gas_volume.append(step["Vy"] * n_gas)
        gas_volume_std.append(Vgas_std * n_gas)
        moles_oil.append(n_oil)
        moles_gas.append(n_gas)

        P -= 5

    # Calculate the residual oil properties at standard conditions
    k0 = y / x
    step = model.flash_pt(x, temperature=T_STD, pressure=P_STD, k0=k0)
    Vres = model.volume(x, pressure=P_STD, temperature=T_STD, root="liquid")
    Vres = Vres * n_oil

    # Calculation of Rs
    Rs = []
    for i in range(len(gas_volume_std)):
        up = sum(gas_volume_std[i:])
    Rs.append(up / Vres)

    return {
        "Voil": np.array(oil_volume),
        "Vgas": np.array(gas_volume),
        "Vgas_std": np.array(gas_volume_std),
        "Vres": Vres,
        "moles_oil": n_oil,
        "P": np.array(pressure),
        "Rs": Rs,
    }
