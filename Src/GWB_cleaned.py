#########################################
#       SIMULATE GWB FROM BWDS          #
#########################################

# This program calculates the GWB based on the method described in my thesis.

############### INITIALS ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from numpy import interp
from scipy.integrate import trapezoid as trap
from warnings import simplefilter 
import os

os.chdir("/home/seppe/data/Papers/2310.19448/White_Dwarf_AGWB/Src/")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

########### LOAD Z_AT_VALUE FILE #############
z_at_val_data = pd.read_csv("../Data/z_at_age.txt", names=["age", "z"], header=1)
interp_age, interp_z = z_at_val_data.age.values, z_at_val_data.z.values

########### AUXILIARY FUNCTIONS ##############

def get_width_z_shell_from_z(z_vals):
    '''
    Returns the widths of the z_shells in Mpc.
    '''
    widths = cosmo.comoving_distance(z_vals).value 
    shells = [widths[i+1] - widths[i] for i in range(len(widths)-1)]
    return np.array(shells)

def SFH(z):
    '''
    Star formation history from [Madau, Dickinson 2014].
    Units: solar mass / yr / Mpc^3
    '''
    return 0.015*(1+z)**(2.7)/(1+((1+z)/2.9)**(5.6)) 

def SFH2(z):
    '''
    Made-up star formation history.
    Units: solar mass / yr / Mpc^3
    '''
    return 0.143*(1+z)**(0.3)/(1+((1+z)/2.9)**(3.2)) 

def SFH3(z):
    '''
    Made-up star formation history.
    Units: solar mass / yr / Mpc^3
    '''
    return 0.00533*(1+z)**(2.7)/(1+((1+z)/2.9)**(3.))

def SFH4(z):
    '''
    Made-up star formation history.
    Units: solar mass / yr / Mpc^3
    '''
    return 0.00245*(1+z)**(2.7)/(1+((1+z)/5.)**(5.6))

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
        fig.savefig("../Output/Figures/" + save_name + ".png")

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
    '''
    Looks for a representative value of the SFH given the age of the system, and an additional time delay in reaching the bin.
    age and Delta_t should be given in Myr.
    '''
    new_age = age - Delta_t
    z_new = get_z_fast(new_age)
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
    if SFH_num == 5:
        return 0.01

def Omega(Omega_ref, f_ref, freq):
    '''
    Create a f^2/3 spectrum line
    '''
    return Omega_ref*10**((2/3) * (np.log10(freq) - np.log10(f_ref)))

def parabola(freq, a, b, c):
    return a*freq**2 + b*freq+c

def get_SFH(SFH_num, z, age, t0, max_z):
    new_age = age - t0
    z_new = get_z_fast(new_age)
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
    if SFH_num == 5:
        return 0.01
    
def get_z_fast(age):
    return interp(age, interp_age, interp_z)

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

def safe_determine_upper_freq(nu_low, evolve_time, K):
    '''
    Determines upper ORBITAL frequency for a binary with K, starting from nu_0, evolving over evolve_time.
    However, the binary can have merged within less than evolve time, in which case the code returns -1.
    Takes evolve_time in Myr, so needs to be converted.
    '''
    if ((nu_low**(-8/3)) > (8 * K * evolve_time * s_in_Myr / 3)):
        return determine_upper_freq(nu_low, evolve_time, K)
    else:
        return -1
    
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

        print(f"\nThe frequencies are {self.f_plot}\n")

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

def main_add_bulk(model, data, tag):
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

                if TEST_FOR_ONE and (index>0):
                    break

                # Working on generic case, so strictly f_0 <  low_f_e < high_f_e < f_max
                if 2*row.nu0> bin_low_f_e or 2*row.nu_max < bin_upp_f_e:
                    continue

                tau = tau_syst(2*row.nu0, bin_upp_f_e, row.K)      # Time to evolve from WD binary formation to upper edge of bin
                time_since_ZAMS = tau + row.t0                     # Both quantities are in Myr

                # Binary can't be older then the beginning of the Universe (with max_z ~ the beginning) 
                if time_since_ZAMS >= time_since_max_z:
                    continue

                psi = representative_SFH(age, time_since_ZAMS, model.SFH_num, model.max_z)
    
                z_fac += psi*row.M_ch**(5/3)
                num_syst += psi * tau_syst(bin_low_f_e, bin_upp_f_e, row.K) * 10**6 # tau is given in Myr, psi in ... /yr

            # print(f"{err_count} errors for z {z}")
            Omega_cont = z_fac * (1+z)**(-4/3) * model.z_widths[i]
            Omega += Omega_cont
            z_contr[f"freq_{j}"][i] = Omega_cont
            z_contr[f"freq_{j}_num"][i] = (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
        
        Omega_plot[j] = 2e-15 * Omega * model.f_bin_factors[j]
        print(f"At frequency {f_r:.5f}: {Omega_plot[j]:.3E}.")

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}")

    # Save GWB
    GWB = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWB.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs//SFH{model.SFH_num}_{model.N}_{model.N_z}_z_contr_{tag}.txt", index = False)

def main_add_birth(model, data, tag):
    '''
    This routine add the contribution of the 'birth bins' to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating birth bin part of the code.\n")
    
    previous_Omega = pd.read_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_{tag}.txt", sep = ",")
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
        if TEST_FOR_ONE and (index>0):
            break

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
            
            # determine the birth bin
            bin_index = np.digitize(2*row.nu0/(1+z), model.f_bins)-1
            low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 
            if TEST_FOR_ONE:
                print(f"Bin frequencies for z {z:.2f}: [{low_f_r:.2E}, {upp_f_r:.2E}]")

            age = model.ages[i].value
            psi = get_SFH(model.SFH_num, z, age, row.t0, model.max_z)

            # The time it would take the binary to evolve from nu_0 to the upper bin edge
            tau_to_bin_edge = tau_syst(2*row.nu0, upp_f_r*(1+z), row.K)

            # If this time is larger than the time the binary has had to evolve since max_z,
            # the latter duration is used.
            max_evolve_time = time_since_max_z - row.t0
            if tau_to_bin_edge >= max_evolve_time:
                tau_in_bin = max_evolve_time
                upp_freq = determine_upper_freq(row.nu0, max_evolve_time, row.K)
                freq_fac = ((upp_freq*(1+z)/2)**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)
            else:
                tau_in_bin = tau_to_bin_edge
                freq_fac = ((upp_f_r*(1+z)/2)**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)

            # contributions
            Omega_cont = 3.2e-15 * model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-2) * psi * model.z_widths[i]

            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (2e-15 * model.f_bin_factors[bin_index]) # The denominator is to keep the relative size wrt the bulk
            
            num_syst = psi * tau_in_bin * 10**6 # tau is given in Myr, psi in ... /yr
            z_contr[f"freq_{bin_index}_num"][i] += (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
            
            Omega_plot[bin_index] += Omega_cont

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_z}_wbirth_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_wbirth_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_z_contr_birth_{tag}.txt", index = False)

def main_add_merge_at_max(model, data, tag):
    '''
    This routine add the contribution of the 'merger bins' due to Kepler max to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitiating merger bin part of the code.\n")

    previous_Omega = pd.read_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_wbirth_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list})

    for i in range(model.N):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no merger bin for binaries that have f_max above our region of interest
    highest_bin = model.f_bins[-1]
    lowest_bin = model.f_bins[0]

    # We go over the rows in the data and determine the merger bins, and their contribution
    for index, row in data.iterrows():

        if TEST_FOR_ONE and (index>0):
            break

        if index % 500 == 0:                   # there is ~ 14k rows
            print(f"At row {index}.")
        
        MERGER_CAN_BE_REACHED = True

        # Determine merger bins for every z bin
        for i, z in enumerate(model.z_list):

            # Don't consider mergers that happen at a frequency beyond our region of interest
            if 2*row.nu_max/(1+z) > highest_bin:
                continue
            
            evolve_time = model.z_time_since_max_z[i].value - row.t0

            if evolve_time <=0:
                    continue

            if MERGER_CAN_BE_REACHED:

                # find merger bin
                bin_index = np.digitize(2*row.nu_max/(1+z), model.f_bins)-1
                low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 

                # The time it takes to evolve from birth to merger
                tau = tau_syst(2*row.nu0, 2*row.nu_max, row.K)

                if tau >= evolve_time:
                    MERGER_CAN_BE_REACHED = False
                    if TEST_FOR_ONE:
                        print("Did not reach merger.")

                else:
                    if TEST_FOR_ONE:
                        print("Reached merger.")

                    psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                    # contributions
                    freq_fac = (row.nu_max**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
                    num_syst = psi * tau_syst(low_f_r*(1+z), 2*row.nu_max, row.K) * 10**6 # tau is given in Myr, psi in ... /yr


            if not MERGER_CAN_BE_REACHED:

                nu_max_b_ini = safe_determine_upper_freq(row.nu0, evolve_time, row.K)

                if nu_max_b_ini == -1:
                    continue

                if DEBUG:
                    tolerance = 0.01
                    if (nu_max_b_ini > (1+tolerance) * row.nu_max):
                        # The first means that evolve_time too large, second should be caught in previous part
                        raise "Error"

                # for safety
                nu_max_b = min(nu_max_b_ini, row.nu_max)

                # Don't consider mergers that happen at a frequency beyond our region of interest
                if (2*nu_max_b/(1+z) > highest_bin) or (2*nu_max_b / (1+z) < lowest_bin):
                    continue

                # find merger bin
                bin_index = np.digitize(2*nu_max_b/(1+z), model.f_bins)-1
                if bin_index == -1:
                    print("oops")

                low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 

                if 2*row.nu0/(1+z) > low_f_r:
                    # in this case the merger is already included as a special case in add birth contribution
                    continue

                freq_fac = (nu_max_b**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
                tau = tau_syst(2*row.nu0, low_f_r*(1+z), row.K)
                psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                num_syst = psi * (evolve_time - tau) * 10**6 # tau is given in Myr, psi in ... /yr

                if DEBUG:
                    np.testing.assert_allclose(evolve_time, tau + tau_syst(low_f_r*(1+z), 2*nu_max_b, row.K), rtol=1e-2)

            # contributions

            Omega_cont = 3.2e-15* model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-2) * psi * model.z_widths[i]
            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (2e-15 * model.f_bin_factors[bin_index])
            Omega_plot[bin_index] += Omega_cont

            z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / 4e6)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]


    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_z}_wmerge_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_wmerge_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_z}_z_contr_merge_{tag}.txt", index = False)

############ ACTUAL MAIN FUNCTION #############

def main():
    '''
    The actual main function. Combines the three different components
    '''

    ### initiate ###

    N = 50             # number of bins
    N_z = 20           # number of z bins
    max_z = 8           # max_redshift
    SFH_num = 1         # which SFH
    tag = "check"       # identifier for filenames

    global SAVE_FIG
    SAVE_FIG = False

    global DEBUG
    DEBUG = False

    # Run script for only one system if True
    global TEST_FOR_ONE
    TEST_FOR_ONE = False

    # create the simulation model
    model = sim_model(N, N_z, max_z, SFH_num)

    # data. initial file with some added calculations
    data = pd.read_csv("../Data/initials_final_2.txt", sep = ",")

    # Some binaries will never make it to our frequency window
    initial_check = data[data["nu0"] < 5e-6]
    can_not_be_seen = (tau_syst(2*initial_check["nu0"], 1e-5, initial_check["K"]) > 13000)
    actual = data.drop(initial_check[can_not_be_seen].index)
    print(f"Out of {len(initial_check)} binaries below 1e-5 Hz, only {len(initial_check) - np.sum(can_not_be_seen)} enters our window.")
    print(f"Dataset reduced from {len(data)} rows to {len(actual)} rows.")


    if TEST_FOR_ONE:
        # info on the first row of data
        print(data.iloc[0])

    main_add_bulk(model, actual, tag)
    main_add_birth(model, actual, tag)
    main_add_merge_at_max(model, actual, tag)

main()

