"""!
@file SFH.py
@date 2024-07-24
@brief This file contains the functions to determine the star formation rate.
@details The file contains the functions to determine the star formation rate. The function representative_SFH determines an appropriate value for the star formation rate at a given age. 
It allows for an optional additional time delay, due to a delay in formation of the binary, or if time is required to move to the correct frequency bin.
The functions SFH_MD, SFH2, SFH3, and SFH4 are star formation histories that can be selected.
@author Seppe Staelens
"""

import modules.RedshiftInterpolator as ri

def representative_SFH(age: float, redshift_interpolator: ri.RedshiftInterpolator, Delta_t: float = 0., SFH_num: int = 1, max_z: float = 8.):
    '''!
    @brief Determines an appropriate value for the star formation rate at a given age.
    @details The function looks for a representative value of the star formation rate given the age of the system, and takes into account an optional additional time delay.
    @param age: age of the system in Myr.
    @param redshift_interpolator: RedshiftInterpolator object that interpolates the redshift at a given age.
    @param Delta_t: time delay due to formation of binary or time required to reach the correct frequency bin, in Myr.
    @param SFH_num: which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01.
    @param max_z: maximum redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    new_age = age - Delta_t
    z_new = redshift_interpolator.get_z_fast(new_age)
    if z_new > max_z:
        print(f"z larger than {max_z}")

    if SFH_num == 1:
        return SFH_MD(z_new)
    if SFH_num == 2:
        return SFH2(z_new)
    if SFH_num == 3:
        return SFH3(z_new)
    if SFH_num == 4:
        return SFH4(z_new)
    if SFH_num == 5:
        return 0.01

def SFH_MD(z: float) -> float:
    '''!
    @brief Star formation history from [Madau, Dickinson 2014].
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.015*(1+z)**(2.7)/(1+((1+z)/2.9)**(5.6)) 

def SFH2(z: float) -> float:
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.143*(1+z)**(0.3)/(1+((1+z)/2.9)**(3.2)) 

def SFH3(z: float) -> float:
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.00533*(1+z)**(2.7)/(1+((1+z)/2.9)**(3.))

def SFH4(z: float) -> float:
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.00245*(1+z)**(2.7)/(1+((1+z)/5.)**(5.6))
