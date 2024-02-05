import numpy as np
import pandas as pd
from astropy import constants as cst

def chirp(m1, m2):
    '''
    Calculate the chirp mass in solar masses.
    '''
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def WD_radius(m):
    '''
    Eggleton 1986 fit to Nauenberg for high m and ZS for low m.
    Returns the radius in solar radii.
    '''
    r = 0.0114*np.sqrt( (m/1.44)**(-2/3) - (m/1.44)**(2/3) ) * (1 + 3.5*(m/0.00057)**(-2/3) + 0.00057/m)**(-2/3)
    return r

def a_min(m1, m2):
    '''
    Calculate minimum separation between two WDs of masses m1 and m2 (solar units).
    Assumes m2<m1.
    Return the minimal separation in solar radii.
    '''
    r1 = WD_radius(m1)
    r2 = WD_radius(m2)
    q = m2/m1
    ap_min = r1*(0.6 + q**(2/3) * np.log(1 + q**(-1/3)))/0.49
    as_min = r2*(0.6 + q**(-2/3) * np.log(1 + q**(1/3)))/0.49
    return min(ap_min, as_min)      ### !!! Indeed, I did pick the min :-(, should be max

def Kepler(m1, m2):
    '''
    Calculate orbital frequency of a binary with separation a_min and masses m1, m2.
    Returns the frequency in Hz.
    '''
    nu = np.sqrt( cst.G * cst.M_sun*(m1+m2) / (4*np.pi**2 * (a_min(m1, m2)* cst.R_sun)**3) )
    return nu.value 

def K(M):
    '''
    Calculate the factor K.
    '''
    return (96/5) * (2*np.pi)**(8/3) * (cst.G * M*cst.M_sun)**(5/3) / cst.c**5

# Load data Gijs
data = pd.read_csv("Pop_Synth/t0aim1m2.dat", names = ["t0", "a", "m1", "m2"], sep = "\s+")

# Calculate the initial period from a_i, based on Kepler's law. The GW frequency is twice the orbital frequency.
Periods = ( (data["a"]*cst.R_sun)**3 * 4*np.pi**2 / (cst.G * cst.M_sun * (data["m1"]+data["m2"]) ) )**(1/2)
GW_frequencies = 2 / Periods        ### !!! Indeed, seems like I saved the f_0 instead of nu_0. This 2 should be 1
data["nu0"] = GW_frequencies

# Calculate the chirp masses
data["M_ch"] = chirp(data.m1, data.m2)

# Calculate the factor K
data["K"] = K(data.M_ch)

# Calculate the maximal frequencies.
nus = []
for ma, mb in zip(data.m1, data.m2):
    nus.append(Kepler(ma, mb))      ### !!! Here I did calculate the orbital frequency, i.e. nu_0
data["nu_max"] = np.array(nus)

# Save data
data.to_csv("Pop_Synth/initials_final_2.txt", index = False)
