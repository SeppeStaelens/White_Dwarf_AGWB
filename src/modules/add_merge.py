def add_merge(model, data, tag):
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
