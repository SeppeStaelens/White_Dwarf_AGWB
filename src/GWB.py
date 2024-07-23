"""!@file GWB.py
@brief This program calculates the GWB based on the method described in my thesis.
@details The program calculates the GWB based on the method described in my thesis. It is divided into three main parts: the bulk part, the birth part, and the merger part. The bulk part calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'. The birth part adds the contribution of the 'birth bins' to the bulk GWB. The merger part adds the contribution of the 'merger bins' due to Kepler max to the bulk GWB. The program saves a dataframe with all the essential information.
@author Seppe Staelens
"""

############### INITIALS ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from numpy import interp
from warnings import simplefilter 
import auxiliary.helper as hp
import auxiliary.SFH as sfh
import auxiliary.SimModel as sm

# ignore pandas warning
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
a, b, c = hp.calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)

########### AUXILIARY FUNCTIONS ##############

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

def Omega(Omega_ref, f_ref, freq):
    '''
    Create a f^2/3 spectrum line
    '''
    return Omega_ref*10**((2/3) * (np.log10(freq) - np.log10(f_ref)))

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

############ MAIN SUB FUNCTIONS ############

def main_add_bulk(model, data, tag):
    '''
    This routine calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitiating bulk part of the code.\n")

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

                psi = sfh.representative_SFH(age, time_since_ZAMS, model.SFH_num, model.max_z)
    
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
            psi = sfh.get_SFH(model.SFH_num, z, age, row.t0, model.max_z)

            # The time it would take the binary to evolve from nu_0 to the upper bin edge
            tau_to_bin_edge = tau_syst(2*row.nu0, upp_f_r*(1+z), row.K)

            # If this time is larger than the time the binary has had to evolve since max_z,
            # the latter duration is used.
            max_evolve_time = time_since_max_z - row.t0
            if tau_to_bin_edge >= max_evolve_time:
                tau_in_bin = max_evolve_time
                upp_freq = determine_upper_freq(row.nu0, max_evolve_time, row.K)
                freq_fac = (upp_freq**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)
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

    # To check numerics
    NUM_ERRORS = 0

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
                tau = row.Dt_max

                if tau >= evolve_time:
                    MERGER_CAN_BE_REACHED = False
                    if TEST_FOR_ONE:
                        print("Did not reach merger.")

                else:
                    if TEST_FOR_ONE:
                        print("Reached merger.")

                    psi = sfh.representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                    # contributions
                    freq_fac = (row.nu_max**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
                    num_syst = psi * tau_syst(low_f_r*(1+z), 2*row.nu_max, row.K) * 10**6 # tau is given in Myr, psi in ... /yr


            if not MERGER_CAN_BE_REACHED:

                nu_max_b_ini = determine_upper_freq(row.nu0, evolve_time, row.K)

                if nu_max_b_ini == -1:
                    # Should not be possible
                    NUM_ERRORS += 1
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
                psi = sfh.representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                num_syst = psi * (evolve_time - tau) * 10**6 # tau is given in Myr, psi in ... /yr

                if DEBUG:
                    np.testing.assert_allclose(evolve_time, tau + tau_syst(low_f_r*(1+z), 2*nu_max_b, row.K), rtol=1e-2)

            # contributions

            Omega_cont = 3.2e-15* model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-2) * psi * model.z_widths[i]
            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (2e-15 * model.f_bin_factors[bin_index])
            Omega_plot[bin_index] += Omega_cont

            z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / 4e6)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]


    if DEBUG:
        print(f"Number of numerical errors: {NUM_ERRORS}\n")

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
    tag = "final"       # identifier for filenames

    global SAVE_FIG
    SAVE_FIG = False

    global DEBUG
    DEBUG = False

    # Run script for only one system if True
    global TEST_FOR_ONE
    TEST_FOR_ONE = False

    # create the simulation model
    model = sm.SimModel(N, N_z, max_z, SFH_num)

    # data. initial file with some added calculations
    data = pd.read_csv("../Data/initials_final_3.txt", sep = ",")

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

