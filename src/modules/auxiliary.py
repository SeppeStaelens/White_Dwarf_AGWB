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

def Omega(Omega_ref, f_ref, freq):
    '''
    Create a f^2/3 spectrum line
    '''
    return Omega_ref*10**((2/3) * (np.log10(freq) - np.log10(f_ref)))

def make_Omega_plot_unnorm(f, Omega_sim, save = False, save_name = "void"):
    '''!
    @brief Make a plot showing Omega for BWD.
    @param f: frequency array.
    @param Omega_sim: Omega array.
    @param save: save the figure.
    @param save_name: name of the saved figure.
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
        fig.savefig("../Output/Figures/" + save_name + ".png")

    #plt.show()

def tau_syst(f_0, f_1, K):
    '''
    Calculates tau, the time it takes a binary with K to evolve from f_0 to f_1 (GW frequencies).
    Returns tau in Myr.
    '''
    tau = 2.381*(f_0**(-8/3) - f_1**(-8/3)) / K
    return tau/s_in_Myr



def determine_upper_freq(nu_low, evolve_time, K):
    '''
    Determines upper ORBITAL frequency for a binary with K, starting from nu_0, evolving over evolve_time.
    Takes evolve_time in Myr, so needs to be converted.
    '''
    if DEBUG:
        assert (nu_low**(-8/3)) > (8 * K * evolve_time * s_in_Myr / 3)
    nu_upp = (nu_low**(-8/3) - 8 * K * evolve_time * s_in_Myr / 3)**(-3/8)
    if DEBUG:
        assert nu_upp > nu_low
    return nu_upp

# DEPRECATED FUNCTION
# 
# def safe_determine_upper_freq(nu_low, evolve_time, K):
#     '''
#     Determines upper ORBITAL frequency for a binary with K, starting from nu_0, evolving over evolve_time.
#     However, the binary can have merged within less than evolve time, in which case the code returns -1.
#     Takes evolve_time in Myr, so needs to be converted.
#     '''
#     if ((nu_low**(-8/3)) > (8 * K * evolve_time * s_in_Myr / 3)):
#         return determine_upper_freq(nu_low, evolve_time, K)
#     else:
#         return -1
