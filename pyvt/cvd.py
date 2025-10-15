"""
Consttant Volume Depletion (CVD) simulation.
"""

import numpy as np

from yaeos.core import ArModel


def cvd(model: ArModel, z: np.ndarray, sat: dict) -> dict:
    """Simulation of a Constant Volume Depletion (CVD) experiment.

    Starts from a saturation point and goes down in pressure removing gas
    until a very low pressure is reached. The total volume is kept constant
    between steps.

    Parameters
    ----------
    model : ArModel
        An `ArModel` object with the equation of state and interaction
        parameters already set.
    z : np.ndarray
        Overall composition of the fluid.
    sat : dict
        Saturation point dictionary as returned by `model.saturation_point`.

    Returns
    -------
    dict
        A dictionary with the following
        keys and values:
        - `P`: np.ndarray
            Pressures in the CVD simulation.
        - `Voil`: np.ndarray
            Molar volumes of the oil phase in the CVD simulation.
        - `Vrel`: np.ndarray
            Molar oil volume relative to the molar volume at saturation.
    """
    Psat = sat["P"]
    T = sat["T"]

    P = Psat - 5

    cum = 0

    # Asume that we start with 1 mol of fluid
    n = 1
    z_i = z

    ps = []
    Voil = []

    while P > 1:
        step = model.flash_pt(z_i, P, T)

        n_oil = n * (1 - step["beta"])
        n_gas = n * step["beta"]

        # Calculation of the moles of gas that leave the system to keep
        # the total volume constant and equal to the volume of the oil at
        # saturation conditions.
        # V = n_oil * step["Vx"] + n_gas * step["Vy"]
        nout = (n_oil * step["Vx"] + n_gas * step["Vy"] - sat["Vx"]) / step[
            "Vy"
        ]

        # Remaining gas in the cell.
        n_gas = n_gas - nout

        n = n_oil + n_gas

        z_i = (n_oil * step["x"] + n_gas * step["y"]) / (n)
        cum += nout

        ps.append(P)
        Voil.append(n_oil * step["Vx"])
        P -= 1

    ps = np.array(ps)
    Voil = np.array(Voil)
    return {"P": ps, "Voil": Voil, "Vrel": Voil / sat["Vx"]}
