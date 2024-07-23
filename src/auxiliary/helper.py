import numpy as np
from numpy import interp
import pandas as pd
from astropy.cosmology import Planck18 as cosmo

########### LOAD Z_AT_VALUE FILE #############
z_at_val_data = pd.read_csv("../Data/z_at_age.txt", names=["age", "z"], header=1)
interp_age, interp_z = z_at_val_data.age.values, z_at_val_data.z.values


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''!
    @brief Calculate the coefficients of a parabola given three points.
    @param x1, y1: x and y coordinates of the first point.
    @param x2, y2: x and y coordinates of the second point.
    @param x3, y3: x and y coordinates of the third point.
    @return A, B, C: coefficients of the parabola.
    '''
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A, B, C

def parabola(x, a, b, c):
    """!
    @brief Calculate the value of a parabola given the coefficients.
    @param x: x value.
    @param a, b, c: coefficients of the parabola.
    @return y: y value.
    """
    return a*x**2 + b*x+c

def get_z_fast(age):
    return interp(age, interp_age, interp_z)

def get_bin_factors(freqs, bins):
    '''
    Determine bin factors that often recur in the calculation to store them.
    '''
    factors = []
    for i, f in enumerate(freqs):
        fac = f * (bins[i+1]**(2/3) - bins[i]**(2/3))/(bins[i+1]-bins[i])
        factors.append(fac)
    return np.array(factors)

def get_width_z_shell_from_z(z_vals):
    '''
    Returns the widths of the z_shells in Mpc.
    '''
    widths = cosmo.comoving_distance(z_vals).value 
    shells = [widths[i+1] - widths[i] for i in range(len(widths)-1)]
    return np.array(shells)
