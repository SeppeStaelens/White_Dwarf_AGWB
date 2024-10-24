"""!
@file SeBa_pre_process.py
@date 2024-07-29
@author Seppe Staelens
@brief This program takes the output of the SeBa population synthesis code and calculates other values from it. The results are saved in a dataframe that can be used in the main code.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from modules.physics import *
from modules.auxiliary import tau_syst


def main() -> None:
    """!
    @brief Function to calculate values from the SeBa output and save them in a dataframe.
    """

    # --- Settings --- #

    # save the resulting file or not
    SAVE_FILE = False

    # metallicity, can be 'z02', 'z01', 'z005', 'z001', 'z0001' or 'z00001'
    metallicity = "z02"

    # population model, can be 'AlphaAlpha' or 'GammaAlpha'
    pop_synth = "AlphaAlpha"

    # alpha, can be 'Alpha1' or 'Alpha4'
    alpha = "Alpha1"

    # which population file to use
    data_file = Path(
        f"../data/{pop_synth}/{alpha}/{metallicity}/{metallicity}_t0aim1m1.dat.gz"
    )  # add _Seppe before .dat.gz to run his data
    # where and how to save the data
    save_filename = Path(
        f"../data/{pop_synth}/{alpha}/{metallicity}/Initials_{metallicity}.txt"
    )  # add _Seppe before .dat.gz when running his data

    # --- Main code --- #

    # Load data
    population = pd.read_csv(data_file, names=["t0", "a", "m1", "m2"], sep="\s+")

    # Calculate the initial orbital frequency from a_i, based on Kepler's law.
    # The GW frequency is twice the orbital frequency.
    orbital_frequencies = 1 / Period(population.a, population.m1, population.m2)
    population["nu0"] = orbital_frequencies

    # Calculate the chirp masses
    population["M_ch"] = chirp(population.m1, population.m2)

    # Calculate the factor K
    population["K"] = K(population.M_ch)

    # Calculate the maximal frequencies. The loop is because of a boolean operation in Kepler
    nu_list = []
    for ma, mb in zip(population.m1, population.m2):
        nu_list.append(Kepler(ma, mb))
    population["nu_max"] = np.array(nu_list)

    # Calculate the maximal time to coalescence
    population["Dt_max"] = tau_syst(
        2 * population.nu0, 2 * population.nu_max, population.K
    )

    if SAVE_FILE:
        population.to_csv(save_filename, index=False)


main()
