"""!
@file auxiliary.py
@author Seppe Staelens
@date 2024-07-29
@brief This module contains auxiliary physics functions that are used to pre-process the population synthesis data.
"""

import numpy as np
import pandas as pd
from astropy import constants as cst

def chirp(m1: float, m2: float) -> float:
    '''!
    @brief Calculate the chirp mass in solar masses.
    @param m1: mass of the first object in solar masses.
    @param m2: mass of the second object in solar masses.
    @return The chirp mass in solar masses.
    '''
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def WD_radius(m: float) -> float:
    '''!
    @brief Calculate the radius of a white dwarf of mass m.
    @param m: mass of the white dwarf in solar masses.
    @details Eggleton 1986 fit to Nauenberg for high m and ZS for low m.
    @return the radius in solar radii.
    '''
    r = 0.0114*np.sqrt( (m/1.44)**(-2/3) - (m/1.44)**(2/3) ) * (1 + 3.5*(m/0.00057)**(-2/3) + 0.00057/m)**(-2/3)
    return r

def a_min(m1: float, m2: float) -> float:
    '''!
    @brief Calculate minimum separation between two WDs of masses m1 and m2 (solar units).
    @param m1: mass of the first WD in solar masses.
    @param m2: mass of the second WD in solar masses.
    @return The minimal separation in solar radii.
    '''
    r1 = WD_radius(m1)
    r2 = WD_radius(m2)

    q = m2/m1

    # Calculate the minimal separation for q and q^{-1}
    ap_min = r1*(0.6 + q**(2/3) * np.log(1 + q**(-1/3)))/0.49
    as_min = r2*(0.6 + q**(-2/3) * np.log(1 + q**(1/3)))/0.49

    return max(ap_min, as_min)

def Kepler(m1: float, m2: float) -> float:
    '''!
    @brief Calculate the orbital frequency of a binary with separation a_min and masses m1, m2.
    @param m1: mass of the first WD in solar masses.
    @param m2: mass of the second WD in solar masses.
    @return the orbital frequency in Hz.
    '''
    nu = np.sqrt( cst.G * cst.M_sun*(m1+m2) / (4*np.pi**2 * (a_min(m1, m2)* cst.R_sun)**3) )
    return nu.value 

def K(M: float) -> float:
    '''!
    @brief Calculate the factor K.
    @param M: chirp mass in solar masses.
    @return The factor K.
    '''
    return (96/5) * (2*np.pi)**(8/3) * (cst.G * M*cst.M_sun)**(5/3) / cst.c**5

def Period(a: float, m1: float, m2: float) -> np.array:
    '''!
    @brief Calculate the orbital period of a binary system from Kepler's law.
    @param a: separation in solar radii.
    @param m1: mass of the first WD in solar masses.
    @param m2: mass of the second WD in solar masses.
    @return The orbital periods in years.
    '''
    return ( (a * cst.R_sun)**3 * 4 * np.pi**2 / (cst.G * cst.M_sun * (m1 + m2) ) )**(1/2)
