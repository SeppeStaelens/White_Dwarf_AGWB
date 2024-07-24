import SFH as sfh

def add_birth(model, data, tag):
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
            psi = sfh.representative_SFH(age, row.t0, model.SFH_num, model.max_z)

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
