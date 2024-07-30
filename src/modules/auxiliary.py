"""!
@file auxiliary.py
@author Seppe Staelens
@date 2024-07-24
@brief This module contains auxiliary functions that are used in the main code.
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt
from astropy import units as u

global s_in_Myr 
s_in_Myr = (u.Myr).to(u.s)

def calc_parabola_vertex(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> tuple:
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

def parabola(x: float, a: float, b: float, c: float) -> float:
    """!
    @brief Calculate the value of a parabola given the coefficients.
    @param x: x value.
    @param a, b, c: coefficients of the parabola.
    @return y: y value.
    """
    return a*x**2 + b*x+c

def get_bin_factors(freqs: np.array, bins: np.array) -> np.array:
    '''!
    @brief Determine bin factors that often recur in the calculation to store them.
    @param freqs: central frequencies.
    @param bins: frequency bin edges.
    @return factors: factors to multiply the contributions with.
    '''
    factors = []
    for i, f in enumerate(freqs):
        fac = f * (bins[i+1]**(2/3) - bins[i]**(2/3))/(bins[i+1]-bins[i])
        factors.append(fac)
    return np.array(factors)

def get_width_z_shell_from_z(z_vals: np.array) -> np.array:
    '''!
    @brief Returns the widths of the redshift shells in Mpc.
    @param z_vals: redshift values.
    @return shells: shell widths in Mpc.
    '''
    widths = cosmo.comoving_distance(z_vals).value 
    shells = [widths[i+1] - widths[i] for i in range(len(widths)-1)]
    return np.array(shells)

def Omega(Omega_ref: float, f_ref: float, freq: np.array) -> np.array:
    '''!
    @brief Create a f^{2/3} spectrum line.
    @param Omega_ref: reference Omega value.
    @param f_ref: reference frequency.
    @param freq: frequency array.
    @return Omega: Omega array.
    '''
    return Omega_ref*10**((2/3) * (np.log10(freq) - np.log10(f_ref)))

def make_Omega_plot_unnorm(f: np.array, Omega_sim: np.array, save: bool = False, save_name: str = "void", show: bool = False) -> None:
    '''!
    @brief Make a plot showing Omega for BWD.
    @param f: frequency array.
    @param Omega_sim: Omega array.
    @param save: save the figure.
    @param save_name: name of the saved figure.
    @param show: show the figure.
    '''
    fig, ax = plt.subplots(1, 1, figsize = (10,8))

    ax.plot(np.log10(f), Omega_sim, color = "green", linewidth = 3, label = "Sim BWD")
    ax.grid(color = "gainsboro", alpha = 0.7)
    ax.set_xlabel(r"$\log_{10}(f$ / Hz$)$")
    ax.set_ylabel(r"$\Omega_{GW}$")

    ax.legend()
    ax.set_yscale("log")
    # ax.set_ylim(10**(-16), 10**(-9))
    ax.set_xlim(-6, 0)
    if save:
        plt.tight_layout()
        fig.savefig("../output/Figures/" + save_name + ".png")
    if show:
        plt.show()

def tau_syst(f_0: float, f_1: float, K: float) -> float:
    '''!
    @brief Calculates tau, the time it takes a binary with K to evolve from f_0 to f_1 (GW frequencies).
    @param f_0: initial frequency.
    @param f_1: final frequency.
    @param K: constant depending on the binary.
    @return tau: time in Myr.
    '''
    tau = 2.381*(f_0**(-8/3) - f_1**(-8/3)) / K
    return tau / s_in_Myr

def determine_upper_freq(nu_low: float, evolve_time: float, K: float, DEBUG: bool = False) -> float:
    '''!
    @brief Determines upper ORBITAL frequency for a binary with K, starting from nu_0, evolving over evolve_time.
    @param nu_low: initial orbital frequency.
    @param evolve_time: time it takes to evolve in Myr.
    @param K: constant depending on the binary.
    @return nu_upp: upper orbital frequency.
    '''
    if DEBUG:
        assert (nu_low**(-8/3)) > (8 * K * evolve_time * s_in_Myr / 3)
    nu_upp = (nu_low**(-8/3) - 8 * K * evolve_time * s_in_Myr / 3)**(-3/8)
    if DEBUG:
        assert nu_upp > nu_low
    return nu_upp
