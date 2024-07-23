def add_bulk(model, data, tag):
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
