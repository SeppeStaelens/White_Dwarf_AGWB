'''!
@file add_merge.py
@brief This file contains a routine that adds the contribution of the 'merger bins' due to Kepler max to the bulk+birth GWB.
@author Seppe Staelens
@date 2024-07-24
'''

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from modules.auxiliary import make_Omega_plot_unnorm, tau_syst, determine_upper_freq
import modules.SimModel as sm
from pathlib import Path

# omega prefactors
normalisation = 3.4e6 # in solar masses, change if necessary, 4e6 for Seppe
omega_prefactor_bulk = 8.10e-9 / normalisation # value = 2.4e-15, value = 2e-15 for Seppe
omega_prefactor_birth_merger = 1.28e-8 / normalisation # value = 3.75e-15 # value = 3.2e-15 for Seppe

def add_merge(model: sm.SimModel, data: pd.DataFrame) -> None:
    '''!
    @brief This routine adds the contribution of the 'merger bins' due to Kepler max to the bulk+birth GWB.
    @param model: instance of SimModel, containing the necessary information for the run.
    @param data: dataframe containing the binary population data.
    @return Saves a dataframe that contains the GWB at all freqyencies, and a dataframe that has the breakdown for the different redshift bins.
    '''
   
    print("\nInitiating merger bin part of the code.\n")

    previous_Omega = pd.read_csv(Path(f"../output/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_wbirth_{model.tag}.txt"), sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list})
    if model.INTEG_MODE == "time":
        z_contr["T"] = model.T_list

    for i in range(model.N_freq):
        z_contr[f"freq_{i}"] = np.zeros_like(model.z_list)
        z_contr[f"freq_{i}_num"] = np.zeros_like(model.z_list)

    # We will have no merger bin for binaries that have f_max above our region of interest
    highest_bin = model.f_bins[-1]
    lowest_bin = model.f_bins[0]

    # To check numerics
    NUM_ERRORS = 0

    # We go over the rows in the data and determine the merger bins, and their contribution
    for index, row in data.iterrows():

        if model.TEST_FOR_ONE and (index>0):
            break

        if index % 1000 == 0:                  
            print(f"At row {index} out of {len(data)}.")
        
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
                    if model.TEST_FOR_ONE:
                        print("Did not reach merger.")

                else:
                    if model.TEST_FOR_ONE:
                        print("Reached merger.")

                    # calculate representative SFH at the time of formation
                    psi = model.sfr_interp.representative_SFH(model.ages[i].value, Delta_t=tau)

                    # contributions
                    freq_fac = (row.nu_max**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
                    num_syst = psi * tau_syst(low_f_r*(1+z), 2*row.nu_max, row.K) * 10**6 # tau is given in Myr, psi in ... /yr


            if not MERGER_CAN_BE_REACHED:

                nu_max_b_ini = determine_upper_freq(row.nu0, evolve_time, row.K)

                if nu_max_b_ini == -1:
                    # Should not be possible
                    NUM_ERRORS += 1
                    continue

                if model.DEBUG:
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
                psi = model.sfr_interp.representative_SFH(model.ages[i].value, Delta_t=tau)

                num_syst = psi * (evolve_time - tau) * 10**6 # tau is given in Myr, psi in ... /yr

                if model.DEBUG:
                    np.testing.assert_allclose(evolve_time, tau + tau_syst(low_f_r*(1+z), 2*nu_max_b, row.K), rtol=1e-2)

            # contributions
            Omega_cont = model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * psi
            if model.INTEG_MODE == "redshift":
                Omega_cont *= omega_prefactor_birth_merger * (1+z)**(-1) * model.z_widths[i]
            
            if model.INTEG_MODE == "redshift":
                z_contr[f"freq_{bin_index}"][i] += Omega_cont / (omega_prefactor_bulk * model.f_bin_factors[bin_index])
                z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / normalisation)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
            elif model.INTEG_MODE == "time":
                z_contr[f"freq_{bin_index}"][i] += Omega_cont / model.f_bin_factors[bin_index]
                z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / normalisation)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.light_speed * (1+z) * model.dT

            if model.INTEG_MODE == "time":
                Omega_cont *= model.light_speed * omega_prefactor_birth_merger * model.dT

            Omega_plot[bin_index] += Omega_cont

    if model.DEBUG:
        print(f"Number of numerical errors: {NUM_ERRORS}\n")

    # Plots
    if model.SAVE_FIG:
        make_Omega_plot_unnorm(model.f_plot, Omega_plot, model.SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_wmerge_{model.tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(Path(f"../output/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_wmerge_{model.tag}.txt"), index = False)
    z_contr.to_csv(Path(f"../output/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_z_contr_merge_{model.tag}.txt"), index = False)
