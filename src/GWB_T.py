

def main_add_bulk(model, data, tag):
    '''
    This routine calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating bulk part of the code.\n")

    # array that will store the values for Omega
    Omega_plot = np.zeros_like(model.f_plot)        

    # array that will store the contributions of each shell
    z_contr = pd.DataFrame({"z":model.z_list, "T":model.T_list})

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
            Omega_cont = z_fac * (1+z)**(-1/3) 
            Omega += Omega_cont
            z_contr[f"freq_{j}"][i] = Omega_cont
            z_contr[f"freq_{j}_num"][i] = (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * light_speed * (1+z) * model.dT
        
        Omega_plot[j] = light_speed * 2e-15 * Omega * model.f_bin_factors[j] * model.dT

        print(f"At frequency {f_r:.5f}: {Omega_plot[j]:.3E}.")

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_t}_{tag}")

    # Save GWB
    GWB = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWB.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs//SFH{model.SFH_num}_{model.N}_{model.N_t}_T_contr_{tag}.txt", index = False)

def main_add_birth(model, data, tag):
    '''
    This routine add the contribution of the 'birth bins' to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitating birth bin part of the code.\n")
    
    previous_Omega = pd.read_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list, "T":model.T_list})

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
                freq_fac = (upp_freq**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)
            else:
                tau_in_bin = tau_to_bin_edge
                freq_fac = ((upp_f_r*(1+z)/2)**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)

            # contributions
            if DEBUG:
                if freq_fac < 0:
                    print("freq fac")
                if psi < 0:
                    print("psi")
            Omega_cont =  model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * psi

            z_contr[f"freq_{bin_index}"][i] += Omega_cont / model.f_bin_factors[bin_index] # The denominator is to keep the relative size wrt the bulk
            
            num_syst = psi * tau_in_bin * 10**6 # tau is given in Myr, psi in ... /yr
            z_contr[f"freq_{bin_index}_num"][i] += (4*np.pi / 4e6) * num_syst * (cosmo.comoving_distance(z).value ** 2) * light_speed * (1+z) * model.dT
            
            Omega_plot[bin_index] += light_speed * 3.2e-15 * Omega_cont * model.dT

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_t}_T_wbirth_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_wbirth_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_contr_birth_{tag}.txt", index = False)

def main_add_merge_at_max(model, data, tag):
    '''
    This routine add the contribution of the 'merger bins' due to Kepler max to the bulk GWB.
    Saves a dataframe with all the essential information.
    '''
   
    print("\nInitiating merger bin part of the code.\n")

    previous_Omega = pd.read_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_wbirth_{tag}.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # Create dataframe to store results
    z_contr = pd.DataFrame({"z":model.z_list, "T":model.T_list})

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

                    psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

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
                    if DEBUG:
                        print("Error here")
                    continue

                # find merger bin
                bin_index = np.digitize(2*nu_max_b/(1+z), model.f_bins)-1
                if bin_index == -1:
                    print("oops")

                low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 

                if 2*row.nu0/(1+z) > low_f_r:
                    # in this case the merger is already included as a special case in add birth contribution
                    if DEBUG:
                        print("Already included")
                    continue

                freq_fac = (nu_max_b**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
                tau = tau_syst(2*row.nu0, low_f_r*(1+z), row.K)
                psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                num_syst = psi * (evolve_time - tau) * 10**6 # tau is given in Myr, psi in ... /yr

                if DEBUG:
                    np.testing.assert_allclose(evolve_time, tau + tau_syst(low_f_r*(1+z), 2*nu_max_b, row.K), rtol=1e-2)

            # contributions

            Omega_cont = model.f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * psi
            z_contr[f"freq_{bin_index}"][i] += Omega_cont / model.f_bin_factors[bin_index]
            Omega_plot[bin_index] += light_speed * 3.2e-15 * Omega_cont * model.dT

            z_contr[f"freq_{bin_index}_num"][i] += (4 * np.pi / 4e6)* num_syst * (cosmo.comoving_distance(z).value ** 2) * light_speed * (1+z) * model.dT


    if DEBUG:
        print(f"Number of numerical errors: {NUM_ERRORS}\n")

    # Plots
    make_Omega_plot_unnorm(model.f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{model.SFH_num}_{model.N}_{model.N_t}_T_wmerge_{tag}")

    # Save GWB
    GWBnew = pd.DataFrame({"f":model.f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_wmerge_{tag}.txt", index = False)

    z_contr.to_csv(f"../Output/GWBs/SFH{model.SFH_num}_{model.N}_{model.N_t}_T_contr_merge_{tag}.txt", index = False)
