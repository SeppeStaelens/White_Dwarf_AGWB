#########################################
#       SIMULATE GWB FROM BWDS          #
#########################################

# this program calculates the GWB based on the method described in Farmer and Phinney

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
Omega_prefactor = 7.7e-9
s_in_Myr = (u.Myr).to(u.s)

# LISA parabola approximation
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A,B,C

a, b, c = calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)

########### AUXILIARY FUNCTIONS ##############

def get_volume_z_shell_from_z(z_vals):
    volumes = cosmo.comoving_volume(z_vals).value # in Mpc3
    shells = [volumes[i+1] - volumes[i] for i in range(len(volumes)-1)]
    return np.array(shells)

def get_width_z_shell_from_z(z_vals):
    widths = cosmo.comoving_distance(z_vals).value # in Mpc
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

def K(M):
    return 3.7e-6 * M**(5/3)

def timespan(nu_1, nu_2, M):
    return 3*(nu_1**(-8/3) - nu_2**(-8/3))/(8*K(M))


def representative_SFH(z, nu_0, nu_1, nu_2, M):
    age_at_z = cosmo.age(0) - cosmo.lookback_time(z)
    t_until_nu_1 = (timespan(nu_0, nu_1, M)*u.s).to(u.Myr)
    t_until_nu_2 = (timespan(nu_0, nu_2, M)*u.s).to(u.Myr)

    av_t = age_at_z - (t_until_nu_1+ t_until_nu_2) / 2

    if av_t < 0.001*u.Gyr:
        av_t = 0.001*u.Gyr

    # print(age_low)

    z_1 = z_at_value(cosmo.age, av_t).value

    return SFH2(z_1)


def F(z, M, f_e):
    return 1.8e-9 * (M*f_e)**(10/3) / ((1+z) * cosmo.comoving_transverse_distance(z).value**2)

def Omega_at_z_for_syst(z, M, f_e):
    return 7.7e-9 * M**(10/3) * f_e**(13/3) / ((1+z)**2 * cosmo.comoving_transverse_distance(z).value**2)

def make_Omega_plot_unnorm(f, Omega_sim, save = False, save_name = "void"):

    ### PLOTS ###

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

    plt.show()

def f_bounds(bin_low, bin_upp, row, z):
    f_0 = 2*row.nu0
    max_f = determine_max_f(row, z)
    if max_f <= bin_low or f_0 >= bin_upp:
        return -1, -1
    if bin_low <= f_0:
        low_f_e = f_0
    else:
        low_f_e = bin_low
    if bin_upp >= max_f:
        upp_f_e = max_f
    else:
        upp_f_e = bin_upp
    return low_f_e, upp_f_e

def determine_max_f(row, z):
    max_duration  = (cosmo.age(0).value - cosmo.lookback_time(z).value) * 10**9 * 365.25 * 24 * 3600 # in seconds
    nu_max1 = (row.nu0**(-8/3) - 8*row.K*max_duration/3 )**(-3/8) # if it were able to coalesce forever
    if np.isnan(nu_max1):
        nu_max = row.nu_max
    else:
        nu_max = min(nu_max1, row.nu_max)
    
    return 2*nu_max

def get_bin_factors(freqs, bins):
    factors = []
    for i, f in enumerate(freqs):
        fac = f * (bins[i+1]**(2/3) - bins[i]**(2/3))/(bins[i+1]-bins[i])
        factors.append(fac)
    return np.array(factors)

def tau_syst(f_0, f_1, K): # returns tau in Myr 
    tau = 2.4*(f_0**(-8/3) - f_1**(-8/3)) / K
    return tau/s_in_Myr

def representative_SFH_v2(age, Delta_t, SFH_num, max_z):
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
    
def make_Omega_plot(f, Omega_sim, save = False, save_name = "void"):

    ### PLOTS ###
    freq = np.logspace(-5, 0, 1000, base = 10)

    fig, ax = plt.subplots(1, 1, figsize = (10,8))

    ax.plot(np.log10(freq), Omega(Omega_upp, 25, freq), linestyle = "dashed", label = "LVK Upp", color = "black", linewidth = 3)
    ax.plot(np.log10(freq), Omega(Omega_BBH, 25, freq), label = "LVK BBH", color = "red", linewidth = 3)
    ax.plot(np.log10(freq), Omega(Omega_BNS, 25, freq), label = "LVK BNS", color = "blue", linewidth = 3)
    ax.plot(np.log10(f), Omega_sim, color = "green", linewidth = 3, label = "Sim BWD")
    ax.plot(np.log10(freq), 10**parabola(np.log10(freq), a, b, c), linestyle = "dashdot", color = "purple", label = "LISA", linewidth = 3)
    ax.fill_between(np.log10(freq), Omega(Omega_BBH_low, 25, freq), Omega(Omega_BBH_up, 25, freq), color = "red", alpha = 0.2)
    ax.fill_between(np.log10(freq), Omega(Omega_BNS_low, 25, freq), Omega(Omega_BNS_up, 25, freq), color = "blue", alpha = 0.2)
    ax.fill_between(np.log10(f), Omega_sim*10**(-0.2), Omega_sim*10**(0.2), color = "green", alpha = 0.2)
    ax.grid(color = "gainsboro", alpha = 0.7)
    ax.set_xlabel(r"$\log_{10}(f$ / Hz$)$")
    ax.set_ylabel(r"$\Omega_{GW}$")

    # plot comparative f**(10/3) line
    # ! separation moet eigenlijk rond 10^-4
    # test = np.logspace(-5,-4,100, base = 10)
    # ax.plot(np.log10(test), Omega_BWD*(0.1)**(2/3) *(test / 0.0001)**(10/3), color = 'orange', linewidth = 3)

    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(10**(-16), 10**(-9))
    ax.set_xlim(-5, 0)
    if save:
        plt.tight_layout()
        fig.savefig("Thesis_Gijs/Figures/" + save_name + ".png")

    plt.show()

def Omega(Omega_ref, f_ref, freq):
    # create quick f^2/3 spectrum
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


############ MAIN FUNCTION ############

def main():
    ### initiate ###

    N = 25      # number of f bins
    N_t = 20    # number of T bins
    max_z = 8   # max_redshift
    SFH_num = 1 # which SFH

    ### automate ###

    print("\nInitating code.\n")

    # freq and z bins
    f_range = np.logspace(-5, 0, 2*N+1, base = 10)

    f_plot = np.array([f_range[2*i+1] for i in range(N)])
    f_bins = np.array([f_range[2*i] for i in range(N+1)])

    print(f"The frequencies are {f_plot}\n")

    T0 = cosmo.lookback_time(max_z)

    T_range = np.linspace(0, T0.value, 2*N_t+1)
  
    T_list = np.array([T_range[2*i+1] for i in range(N_t)])
    T_bins = np.array([T_range[2*i] for i in range(N_t+1)])

    dT = (T_list[1] - T_list[0])

    print(f"The cosmic timestep is {dT} Gyr\n")

    print(f"The times are {T_list}\n")

    _z_list = []
    # _z_bins = []
    for time in T_list:
        _z_list.append(z_at_value(cosmo.lookback_time, time * u.Gyr).value)
    # for time in T_bins:
    #     _z_bins.append(z_at_value(cosmo.lookback_time, time).value)
    
    z_list = np.array(_z_list)
    # z_bins = np.array(_z_bins)

    # z_widths = get_width_z_shell_from_z(z_bins)  

    print(f"The redshifts are {z_list}\n")

    # data. initial file with some added calculations
    data = pd.read_csv("Thesis_Gijs/Pop_Synth/initials_final_2.txt", sep = ",")

    ### body ###

    Omega_plot = np.zeros_like(f_plot)    

    z_time_since_max_z = T0 - T_list * u.Gyr
    ages = cosmo.age(0) - T_list * u.Gyr

    z_contr = pd.DataFrame({"z":z_list, "T":T_list})

    f_bin_factors = get_bin_factors(f_plot, f_bins)

    for j, f_r in enumerate(f_plot):
        low_f_r, upp_f_r = f_bins[j], f_bins[j+1]
        Omega = 0
        z_contr_for_f = np.zeros_like(z_list)
        num_syst_for_f = np.zeros_like(z_list)
        for i, z in enumerate(z_list):
            time_since_max_z = z_time_since_max_z[i].value*1e3
            age = ages[i].value*1e3
            bin_low_f_e = low_f_r * (1+z)
            bin_upp_f_e = upp_f_r * (1+z)
            z_fac = 0
            num_syst = 0
            for index, row in data.iterrows():
                if 2*row.nu0 > bin_low_f_e or 2*row.nu_max < bin_upp_f_e:
                    continue
                tau = tau_syst(2*row.nu0, bin_low_f_e, row.K)
                time_since_ZAMS = tau + row.t0 # both tau and t0 are in Myr
                if time_since_ZAMS >= time_since_max_z:
                    continue
                psi = representative_SFH_v2(age, time_since_ZAMS, SFH_num, max_z)
                z_fac += psi*row.M_ch**(5/3)
                num_syst += psi
            Omega_cont = z_fac*(1+z)**(-1/3)
            Omega += Omega_cont
            z_contr_for_f[i] = Omega_cont
            num_syst_for_f[i] = num_syst
            print("Redshift_check")            
        print(f"At frequency {f_r:.5f}.")
        z_contr[f"freq_{j}"] = z_contr_for_f
        z_contr[f"freq_{j}_nums"] = num_syst_for_f
        Omega_plot[j] = 306.6 * 2e-15 * Omega * f_bin_factors[j] * dT
        print(Omega_plot[j])

    # Plots
    SAVE_FIG = False

    #make_Omega_plot_unnorm(f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{SFH_num}_{N}_{N_t}_T")

    # Save GWB
    GWB = pd.DataFrame({"f":f_plot, "Om":Omega_plot})
    GWB.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_contr.txt", index = False)

def main_add_birth():
    # THIS ADDS THE CONTRIBUTION OF THE BIRTH FREQUENCY BINS TO THE RESULT OF MAIN2
    ### initiate ###

    N = 25       # number of bins
    N_t = 20    # number of z bins
    max_z = 8   # max_redshift
    SFH_num = 1 # which SFH
    SAVE_FIG = False

    ### automate ###

    print("\nInitating code.\n")

    # freq and z bins
    f_range = np.logspace(-5, 0, 2*N+1, base = 10)

    f_plot = np.array([f_range[2*i+1] for i in range(N)])
    f_bins = np.array([f_range[2*i] for i in range(N+1)])

    print(f"The frequencies are {f_plot}\n")

    T0 = cosmo.lookback_time(max_z)

    T_range = np.linspace(0, T0.value, 2*N_t+1)
  
    T_list = np.array([T_range[2*i+1] for i in range(N_t)])
    T_bins = np.array([T_range[2*i] for i in range(N_t+1)])

    dT = (T_list[1] - T_list[0])

    print(f"The cosmic timestep is {dT} Gyr\n")

    print(f"The times are {T_list}\n")

    _z_list = []
    # _z_bins = []
    for time in T_list:
        _z_list.append(z_at_value(cosmo.lookback_time, time * u.Gyr).value)
    # for time in T_bins:
    #     _z_bins.append(z_at_value(cosmo.lookback_time, time).value)
    
    z_list = np.array(_z_list)

    # data
    data = pd.read_csv("Thesis_Gijs/Pop_Synth/initials_final_2.txt", sep = ",")
    previous_Omega = pd.read_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # check frequencies
    sum_check = np.sum(np.abs(f_plot - previous_Omega.f.values))
    if sum_check >= 1e-7:
        print("Frequencies do not match")
        exit()

    ### body ###

    z_time_since_max_z = T0 - T_list * u.Gyr
    ages = cosmo.age(0) - T_list * u.Gyr

    z_contr = pd.DataFrame({"z":z_list, "T":T_list})

    for i in range(N):
        z_contr[f"freq_{i}"] = np.zeros_like(z_list)
    
    f_bin_factors = get_bin_factors(f_plot, f_bins)

    for index, row in data.iterrows():
        if index % 500 == 0:                   # there is ~ 14k rows
            print(f"At row {index}.")
        for i, z in enumerate(z_list):
            time_since_max_z = z_time_since_max_z[i].value*1e3
            if row.t0 >= time_since_max_z:
                continue
            if 2*row.nu0/(1+z) < f_bins[0]:
                continue
            age = ages[i].value*1e3
            bin_index = np.digitize(2*row.nu0/(1+z), f_bins)-1
            low_f_r, upp_f_r = f_bins[bin_index], f_bins[bin_index + 1] 
            freq_fac = ((upp_f_r*(1+z)/2)**(2/3) - row.nu0**(2/3))/(upp_f_r - low_f_r)
            Omega_cont = 306.6 * 3.2e-15* f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * get_SFH(SFH_num, z, age, row.t0, max_z) * dT
            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (306.6 * 2e-15 * f_bin_factors[bin_index])
            Omega_plot[bin_index] += Omega_cont

    # Plots
    #make_Omega_plot_unnorm(f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{SFH_num}_{N}_{N_t}_T_wbirth")

    # Save GWB
    GWBnew = pd.DataFrame({"f":f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_wbirth.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_contr_birth.txt", index = False)


def main_add_merge():

    # THIS ADDS THE CONTRIBUTION OF THE MERGER FREQUENCY BINS TO THE RESULT OF MAIN_ADD_BIRTH

    ### initiate ###

    N = 25       # number of bins
    N_t = 20    # number of z bins
    max_z = 8   # max_redshift
    SFH_num = 1 # which SFH
    SAVE_FIG = False

    ### automate ###

    print("\nInitating code.\n")

    # freq and z bins
    f_range = np.logspace(-5, 0, 2*N+1, base = 10)

    f_plot = np.array([f_range[2*i+1] for i in range(N)])
    f_bins = np.array([f_range[2*i] for i in range(N+1)])

    print(f"The frequencies are {f_plot}\n")

    T0 = cosmo.lookback_time(max_z)

    T_range = np.linspace(0, T0.value, 2*N_t+1)
  
    T_list = np.array([T_range[2*i+1] for i in range(N_t)])
    T_bins = np.array([T_range[2*i] for i in range(N_t+1)])

    dT = (T_list[1] - T_list[0])

    print(f"The cosmic timestep is {dT} Gyr\n")

    print(f"The times are {T_list}\n")

    _z_list = []
    # _z_bins = []
    for time in T_list:
        _z_list.append(z_at_value(cosmo.lookback_time, time * u.Gyr).value)
    # for time in T_bins:
    #     _z_bins.append(z_at_value(cosmo.lookback_time, time).value)
    
    z_list = np.array(_z_list)

    # data
    data = pd.read_csv("Thesis_Gijs/Pop_Synth/initials_final_2.txt", sep = ",")
    previous_Omega = pd.read_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_wbirth.txt", sep = ",")
    Omega_plot = previous_Omega.Om.values

    # check frequencies
    sum_check = np.sum(np.abs(f_plot - previous_Omega.f.values))
    if sum_check >= 1e-7:
        print("Frequencies do not match")
        exit()

    ### body ###

    z_time_since_max_z = T0 - T_list * u.Gyr
    ages = cosmo.age(0) - T_list * u.Gyr

    z_contr = pd.DataFrame({"z":z_list, "T":T_list})

    for i in range(N):
        z_contr[f"freq_{i}"] = np.zeros_like(z_list)
    
    f_bin_factors = get_bin_factors(f_plot, f_bins)
   
    for index, row in data.iterrows():
        if index % 500 == 0:                   # there is ~ 14k rows
            print(f"At row {index}.")
        for i, z in enumerate(z_list):
            if 2*row.nu_max/(1+z) > f_bins[-1]:
                continue
            bin_index = np.digitize(2*row.nu_max/(1+z), f_bins)-1
            low_f_r, upp_f_r = f_bins[bin_index], f_bins[bin_index + 1] 
            freq_fac = (row.nu_max**(2/3) - (low_f_r*(1+z)/2)**(2/3))/(upp_f_r - low_f_r)
            tau = tau_syst(2*row.nu0, low_f_r*(1+z), row.K)
            if tau >= z_time_since_max_z[i].value*1e3:
                continue
            psi = representative_SFH_v2(ages[i].value*1e3, tau, SFH_num, max_z)
            Omega_cont = 306.6 * 3.2e-15* f_plot[bin_index] * row.M_ch**(5/3) * freq_fac * (1+z)**(-1) * psi * dT
            z_contr[f"freq_{bin_index}"][i] += Omega_cont / (306.6 * 2e-15 * f_bin_factors[bin_index])
            Omega_plot[bin_index] += Omega_cont

    # Plots
    # make_Omega_plot_unnorm(f_plot, Omega_plot, SAVE_FIG, f"GWB_SFH{SFH_num}_{N}_{N_t}_wmerge")

    # Save GWB
    GWBnew = pd.DataFrame({"f":f_plot, "Om":Omega_plot})
    GWBnew.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_wmerge.txt", index = False)

    z_contr.to_csv(f"Thesis_Gijs/GWBs_final2/SFH{SFH_num}_corr_{N}_{N_t}_T_contr_merge.txt", index = False)

main()
main_add_birth()
main_add_merge()
