'''!
@file add_birth.py
@brief This file contains a routine that adds the contribution of the 'birth bins' to the bulk GWB.
@author Seppe Staelens
@date 2024-07-24
'''

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from modules.auxiliary import make_Omega_plot_unnorm, tau_syst, determine_upper_freq
import modules.SFH as sfh
import modules.SimModel as sm
import modules.RedshiftInterpolator as ri

# omega prefactors
normalisation = 3.4e6 # in solar masses, change if necessary, 4e6 for Seppe
omega_prefactor_bulk = 8.10e-9 / normalisation # value = 2.4e-15, value = 2e-15 for Seppe
omega_prefactor_birth_merger = 1.28e-8 / normalisation # value = 3.75e-15 # value = 3.2e-15 for Seppe

def add_birth(model: sm.SimModel, data: pd.DataFrame, z_interp: ri.RedshiftInterpolator, tag: str) -> None:
    '''!
    @brief This routine adds the contribution of the 'birth bins' to the bulk GWB.
    @param model: instance of SimModel, containing the necessary information for the run.
    @param z_interp: instance of RedshiftInterpolator, used in the SFH calculations.
    @param data: dataframe containing the binary population data.
    @param tag: tag to add to the output files.
    @return Saves a dataframe that contains the GWB at all freqyencies, and a dataframe that has the breakdown for the different redshift bins.
    '''
   
    print("\nInitating birth bin part of the code.\n")
    
    # read the bulk part of the GWB
    previous_Omega = pd.read_csv(f"../output/{model.pop_synth}/{model.alpha}/{model.metallicity}/{model.SFH_type}/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list})
    if model.INTEG_MODE == "time":
        z_contr["T"] = model.T_list

    for i in range(model.N_freq):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no birth bin for binaries that have f_0 below our region of interest
    lowest_bin = model.f_bins[0]

    # We go over the rows in the data and determine the birth bins, and their contribution
    for index, row in data.iterrows():

        if model.TEST_FOR_ONE and (index>0):
            break

        if index % 1000 == 0:                   
            print(f"At row {index} out of {len(data)}.")

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
            if model.TEST_FOR_ONE:
                print(f"Bin frequencies for z {z:.2f}: [{low_f_r:.2E}, {upp_f_r:.2E}]")

            age = model.ages[i].value
            # calculate representative SFH at the time of formation
            psi = sfh.representative_SFH(age, z_interp, Delta_t=row.t0, SFH_num=model.SFH_num, max_z=model.max_z, SFH_type = model.SFH_type, metallicity = model.metallicity)

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
            Omega_cont = model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * psi
            if model.INTEG_MODE == "redshift":
                Omega_cont *=  omega_prefactor_birth_merger * (1+z)**(-1) * model.z_widths[i]
            
            num_syst = psi * tau_in_bin * 10**6 # tau is given in Myr, psi in ... /yr

            if model.INTEG_MODE == "redshift":
                z_contr[f"freq_{bin_index}"][i] += Omega_cont / (omega_prefactor_bulk * model.f_bin_factors[bin_index]) # The denominator is to keep the relative size wrt the bulk
                z_contr[f"freq_{bin_index}_num"][i] += (4*np.pi / normalisation) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
            elif model.INTEG_MODE == "time":
                z_contr[f"freq_{bin_index}"][i] += Omega_cont / model.f_bin_factors[bin_index] # The denominator is to keep the relative size wrt the bulk
                z_contr[f"freq_{bin_index}_num"][i] += (4*np.pi / normalisation) * num_syst * (cosmo.comoving_distance(z).value ** 2) * model.light_speed * (1+z) * model.dT
            
            if model.INTEG_MODE == "time":
                Omega_cont *= model.light_speed * omega_prefactor_birth_merger * model.dT

            Omega_plot[bin_index] += Omega_cont

    # Plots
    if model.SAVE_FIG:
        make_Omega_plot_unnorm(model.f_plot, Omega_plot, model.SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_wbirth_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"../output/{model.pop_synth}/{model.alpha}/{model.metallicity}/{model.SFH_type}/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_wbirth_{tag}.txt", index = False)
    z_contr.to_csv(f"../output/{model.pop_synth}/{model.alpha}/{model.metallicity}/{model.SFH_type}/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_z_contr_birth_{tag}.txt", index = False)

