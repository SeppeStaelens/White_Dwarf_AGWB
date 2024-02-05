#########################################
#       SIMULATE GWB FROM BWDS          #
#########################################

# This program calculates the GWB based on the method described in my thesis.

############### INITIALS ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy import constants as cst
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from numpy import interp
from scipy.integrate import trapezoid as trap

# matplotlib globals
plt.rc('font',   size=16)          # controls default text sizes
plt.rc('axes',   titlesize=18)     # fontsize of the axes title
plt.rc('axes',   labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=14)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=14)     # fontsize of the tick labels
plt.rc('legend', fontsize=18)      # legend fontsize
plt.rc('figure', titlesize=18)     # fontsize of the figure title

########## PARAMETERS AND DATA ##########

# LVK O3, alpha = 2/3 at 25 Hz
Omega_upp = 3.4*10**(-9)
Omega_BBH = 4.7*10**(-10)
Omega_BNS = 2.*10**(-10)

Omega_BBH_up = 6.3*10**(-10)
Omega_BNS_up = 5.2*10**(-10)
Omega_BBH_low = 3.3*10**(-10)
Omega_BNS_low = 0.6*10**(-10)

# Farmer and Phinney, at 1 mHz
Omega_BWD = 3.57*10**(-12)
Omega_BWD_up = 6.*10**(-12)
Omega_BWD_low = 1.*10**(-12)

# other quantities
s_in_Myr = (u.Myr).to(u.s)

# LISA sensitivity curve: parabola approximation
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A,B,C

a, b, c = calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)

########### AUXILIARY FUNCTIONS ##############

def get_width_z_shell_from_z(z_vals):
    '''
    Returns the widths of the z_shells in Mpc.
    '''
    widths = cosmo.comoving_distance(z_vals).value 
    shells = [widths[i+1] - widths[i] for i in range(len(widths)-1)]
    return np.array(shells)

def SFH(z):
    return 0.015*(1+z)**(2.7)/(1+((1+z)/2.9)**(5.6))  # solar mass / yr / Mpc^3 [Madau, Dickinson 2014]

def SFH2(z):
    return 0.143*(1+z)**(0.3)/(1+((1+z)/2.9)**(3.2))  # solar mass / yr / Mpc^3 

def SFH3(z):
    return 0.00533*(1+z)**(2.7)/(1+((1+z)/2.9)**(3.))  # solar mass / yr / Mpc^3 

def SFH4(z):
    return 0.00245*(1+z)**(2.7)/(1+((1+z)/5.)**(5.6))  # solar mass / yr / Mpc^3

def chirp(m1, m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def make_Omega_plot_unnorm(f, Omega_sim, save = False, save_name = "void"):

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
        fig.savefig("Thesis_Gijs/Figures_final/" + save_name + ".png")

    #plt.show()

def get_bin_factors(freqs, bins):
    factors = []
    for i, f in enumerate(freqs):
        fac = f * (bins[i+1]**(2/3) - bins[i]**(2/3))/(bins[i+1]-bins[i])
        factors.append(fac)
    return np.array(factors)

def tau_syst(f_0, f_1, K):
    '''
    Calculates tau, the time it takes a binary with K to evolve from f_0 to f_1 (GW frequencies).
    Returns tau in Myr.
    '''
    tau = 2.4*(f_0**(-8/3) - f_1**(-8/3)) / K
    return tau/s_in_Myr

def representative_SFH(age, Delta_t, SFH_num, max_z):
    new_age = age - Delta_t
    z_new = z_at_value(cosmo.age, new_age*u.Myr).value
    if z_new > max_z:
        print(f"z larger than {max_z}")

    if SFH_num == 1:
        return SFH(z_new)
    if SFH_num == 2:
        return SFH2(z_new)
    if SFH_num == 3:
        return SFH3(z_new)
    if SFH_num == 4:
        return SFH4(z_new)

def Omega(Omega_ref, f_ref, freq):
    '''
    Create a f^2/3 spectrum line
    '''
    return Omega_ref*10**((2/3) * (np.log10(freq) - np.log10(f_ref)))

def parabola(freq, a, b, c):
    return a*freq**2 + b*freq+c

def get_SFH(SFH_num, z, age, t0, max_z):
    new_age = age - t0
    z_new = z_at_value(cosmo.age, new_age*u.Myr).value
    if z_new > max_z:
        print(f"z larger than {max_z}")
    if SFH_num == 1:
        return SFH(z_new)
    if SFH_num == 2:
        return SFH2(z_new)
    if SFH_num == 3:
        return SFH3(z_new)
    if SFH_num == 4:
        return SFH4(z_new)
    
######## MODEL CLASS ############
    
class sim_model:
    '''
    This class contains information about the run that needs to be shared over the different subroutines.
    '''
    def __init__(self, N = 50, N_z = 20, max_z = 8, SFH_num = 1, log_f_low = -5, log_f_high = 0) -> None:
        self.N = N
        self.N_z = N_z
        self.max_z = max_z
        self.SFH_num = SFH_num
        assert log_f_low < log_f_high, "log_f_low should be smaller than log_f_high"
        self.log_f_low = log_f_low
        self.log_f_high = log_f_high
        self.calculate_f_bins()
        self.calculate_z_bins()
        self.calculate_cosmology()

    def calculate_f_bins(self):
        '''
        Calculates the f bins and the bin factors.
        '''    
        f_range = np.logspace(self.log_f_low, self.log_f_high, 2*self.N + 1, base = 10)
        self.f_plot = np.array([f_range[2*i+1] for i in range(self.N)])
        self.f_bins = np.array([f_range[2*i] for i in range(self.N+1)])
        self.f_bin_factors = get_bin_factors(self.f_plot, self.f_bins)

        print(f"The frequencies are {self.f_plot}\n")

    def calculate_z_bins(self):
        '''
        Calculates the z bins.
        '''
        z_range = np.linspace(0, self.max_z, 2*self.N_z+1)  
        self.z_list = np.array([z_range[2*i+1] for i in range(self.N_z)])
        self.z_bins = np.array([z_range[2*i] for i in range(self.N_z+1)])

        print(f"The redshifts are {self.z_list}\n")

    def calculate_cosmology(self):
        self.z_widths = get_width_z_shell_from_z(self.z_bins)    
        self.z_time_since_max_z = (cosmo.lookback_time(self.max_z) - cosmo.lookback_time(self.z_list)).to(u.Myr)
        self.ages = (cosmo.age(0) - cosmo.lookback_time(self.z_list)).to(u.Myr)



############ MAIN SUB FUNCTIONS ############

def main_add_bulk(model, data, SAVE_FIG, tag):
    '''
    This routine calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating bulk part of the code.\n")

    # array that will store the values for Omega
    Omega_plot = np.zeros_like(model.f_plot)        

    # array that will store the contributions of each shell
    z_contr = pd.DataFrame({"z" : model.z_list})

    for i in range(model.N):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We now loop over the received frequency values f_r.
    for j, f_r in enumerate(model.f_plot):

        low_f_r, upp_f_r = model.f_bins[j], model.f_bins[j+1]      # Bin edges
        Omega = 0                                                  

        # We calculate the contribution to the frequency bin for every redshift bin
        for i, z in enumerate(model.z_list):
            time_since_max_z = model.z_time_since_max_z[i].value
            age = model.ages[i].value
            bin_low_f_e = low_f_r * (1+z)                          # Emission frequency bin edges
            bin_upp_f_e = upp_f_r * (1+z)                          # 
            z_fac = 0
            num_syst = 0

            # We calculate the contribution for every type of binary in the Population Synthesis
            for index, row in data.iterrows():

                # Working on generic case, so strictly f_0 <  low_f_e < high_f_e < f_max
                if 2*row.nu0> bin_low_f_e or 2*row.nu_max < bin_upp_f_e:
                    continue

                tau = tau_syst(2*row.nu0, bin_low_f_e, row.K)      # Time to evolve from WD binary formation to lower edge of bin
                time_since_ZAMS = tau + row.t0                     # Both quantities are in Myr

                # Binary can't be older then the beginning of the Universe (with max_z ~ the beginning) 
                if time_since_ZAMS >= time_since_max_z:
                    continue

                psi = representative_SFH(age, time_since_ZAMS, model.SFH_num, model.max_z)
    
                z_fac += psi*row.M_ch**(5/3)
                num_syst += psi * tau_syst(bin_low_f_e, bin_upp_f_e, row.K) * 10**6 # tau is given in Myr, psi in ... /yr

            Omega_cont = z_fac * (1+z)**(-4/3) * model.z_widths[i]
            Omega += Omega_cont
            z_contr[f"freq_{j}"][i] = Omega_cont
            z_contr[f"freq_{j}_num"][i] = (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
        
        Omega_plot[j] = 2e-15 * Omega * model.f_bin_factors[j]
        print(f"At frequency {f_r:.5f}: {Omega_plot[j]}.")

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}")

    # Save GWB
    GWB = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWB.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_{model.N}_{model.N_z}_z_contr_{tag}.txt", index = False)

def main_add_birth(model, data, SAVE_FIG, tag):
    '''
    This routine add the contribution of the 'birth bins' to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating birth bin part of the code.\n")
    
    previous_Omega = pd.read_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list})

    for i in range(model.N):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no birth bin for binaries that have f_0 below our region of interest
    lowest_bin = model.f_bins[0]

    # We go over the rows in the data and determine the birth bins, and their contribution
    for index, row in data.iterrows():
        if index % 500 == 0:                   # there are ~ 14k rows
            print(f"At row {index}.")

        # Determine birth bins for every z bin
        for i, z in enumerate(model.z_list):
            time_since_max_z = model.z_time_since_max_z[i].value

            # Binaries can't be older than the Universe
            if row.t0 >= time_since_max_z:
                continue

            # Birth frequency out of our region of interest
            if 2*row.nu0/(1+z) < lowest_bin:
                continue
            
            age = model.ages[i].value

            # determine the birth bin
            bin_index = np.digitize(2*row.nu0/(1+z), model.f_bins)-1
            low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 
            freq_fac = ((upp_f_r*(1+z)/2)**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)

            psi = get_SFH(model.SFH_num, z, age, row.t0, model.max_z)

            # contributions
            Omega_cont = 3.2e-15 * model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-2) * psi * model.z_widths[i]

            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (2e-15 * model.f_bin_factors[bin_index]) # The denominator is to keep the relative size wrt the bulk
            
            num_syst = psi * tau_syst(2*row.nu0, upp_f_r*(1+z), row.K) * 10**6 # tau is given in Myr, psi in ... /yr
            z_contr[f"freq_{bin_index}_num"][i] += (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
            
            Omega_plot[bin_index] += Omega_cont

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_wbirth_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_wbirth_{tag}.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_z_contr_birth_{tag}.txt", index = False)


def main_add_merge(model, data, SAVE_FIG, tag):
    '''
    This routine add the contribution of the 'merger bins' to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating merger bin part of the code.\n")

    previous_Omega = pd.read_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_wbirth_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list})

    for i in range(model.N):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no merger bin for binaries that have f_max above our region of interest
    highest_bin = model.f_bins[-1]

    # We go over the rows in the data and determine the merger bins, and their contribution
    for index, row in data.iterrows():
        if index % 500 == 0:                   # there is ~ 14k rows
            print(f"At row {index}.")
        
        # Determine merger bins for every z bin
        for i, z in enumerate(model.z_list):

            # Don't consider mergers that happen at a frequency beyond our region of interest
            if 2*row.nu_max/(1+z) > highest_bin:
                continue

            # find merger bin
            bin_index = np.digitize(2*row.nu_max/(1+z), model.f_bins)-1
            low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 
            freq_fac = (row.nu_max**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
            tau = tau_syst(2*row.nu0, low_f_r*(1+z), row.K)

            if tau >= model.z_time_since_max_z[i].value:
                continue

            psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

            # contributions
            Omega_cont = 3.2e-15* model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-2) * psi * model.z_widths[i]

            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (2e-15 * model.f_bin_factors[bin_index])

            num_syst = psi * tau_syst(low_f_r*(1+z), 2*row.nu_max, row.K) * 10**6 # tau is given in Myr, psi in ... /yr
            z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / 4e6)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]

            Omega_plot[bin_index] += Omega_cont

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_wmerge_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_wmerge_{tag}.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{model.SFH_num}_corr_{model.N}_{model.N_z}_z_contr_merge_{tag}.txt", index = False)



############ ACTUAL MAIN FUNCTION #############

def main():
    '''
    The actual main function. Combines the three different components
    '''

    ### initiate ###

    N = 25              # number of bins
    N_z = 20            # number of z bins
    max_z = 8           # max_redshift
    SFH_num = 1         # which SFH
    tag = "net"         # identifier for filenames

    SAVE_FIG = True

    # create the simulation model
    model = sim_model(N, N_z, max_z, SFH_num)

    # data. initial file with some added calculations
    data = pd.read_csv("Thesis_Gijs/Pop_Synth/initials_final_2.txt", sep = ",")

    main_add_bulk(model, data, SAVE_FIG, tag)
    main_add_birth(model, data, SAVE_FIG, tag)
    main_add_merge(model, data, SAVE_FIG, tag)

main()

