'''!
@file add_bulk.py
@brief This file contains a routine that calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
@author Seppe Staelens
@date 2024-07-24
'''

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from auxiliary import make_Omega_plot_unnorm, tau_syst
import SFH as sfh

def add_bulk(model, z_interp, data, tag):
    '''!
    @brief This routine calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
    @param model: instance of SimModel, containing the necessary information for the run.
    @param z_interp: instance of RedshiftInterpolator, used in the SFH calculations.
    @param data: dataframe containing the binary population data.
    @param tag: tag to add to the output files.
    @return Saves a dataframe with all the essential information.
    '''
   
    print("\nInitiating bulk part of the code.\n")

    # array that will store the values for Omega
    Omega_plot = np.zeros_like(model.f_plot)        

    # array that will store the contributions of each shell
    z_contr = pd.DataFrame({"z" : model.z_list})
    if model.INTEG_MODE == "time":
        z_contr["T"] = model.T_list

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

                psi = sfh.representative_SFH(age, z_interp, Delta_t=time_since_ZAMS, SFH_num=model.SFH_num, max_z=model.max_z)
    
                z_fac += psi*row.M_ch**(5/3)
                num_syst += psi * tau_syst(bin_low_f_e, bin_upp_f_e, row.K) * 10**6 # tau is given in Myr, psi in ... /yr

            # the contribution if we integrate over T
            Omega_cont = z_fac * (1+z)**(-1/3)
            # if we integrate over z, we need to add another factor (1+z)^(-1) Delta z
            if model.INTEG_MODE == "redshift":
                Omega_cont *= model.z_widths[i]*(1+z)**(-1)
            Omega += Omega_cont
            z_contr[f"freq_{j}"][i] = Omega_cont

            # the contribution to the number of systems
            pre_num = (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2)
            if model.INTEG_MODE == "redshift":
                z_contr[f"freq_{j}_num"][i] = pre_num * model.z_widths[i]
            elif model.INTEG_MODE == "time":
                z_contr[f"freq_{j}_num"][i] = pre_num * light_speed * (1+z) * model.dT
        
        Omega_plot[j] = 2e-15 * Omega * model.f_bin_factors[j]
        if model.INTEG_MODE == "time":
            Omega_plot[j] *= light_speed * model.dT

        print(f"At frequency {f_r:.5f}: {Omega_plot[j]:.3E}.")

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, model.SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_{tag}")

    # Save GWB
    GWB = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWB.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N_freq}_{model.N_int}_z_contr_{tag}.txt", index = False)
