"""!
@file SFH.py
@date 2024-07-24
@brief This file contains analytic functions to determine the star formation rate.
@details The file contains analytic functions to determine the star formation rate. 
The functions SFH_MD, SFH2, SFH3, and SFH4 are star formation histories that can be selected in the SFRInterpolator class.
The other SFHs are obtained from a data file.
@author Seppe Staelens
"""


def SFH_MD(z: float) -> float:
    """!
    @brief Star formation history from [Madau, Dickinson 2014].
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.015 * (1 + z) ** (2.7) / (1 + ((1 + z) / 2.9) ** (5.6))


def SFH2(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.143 * (1 + z) ** (0.3) / (1 + ((1 + z) / 2.9) ** (3.2))


def SFH3(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.00533 * (1 + z) ** (2.7) / (1 + ((1 + z) / 2.9) ** (3.0))


def SFH4(z: float) -> float:
    """!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    """
    return 0.00245 * (1 + z) ** (2.7) / (1 + ((1 + z) / 5.0) ** (5.6))
