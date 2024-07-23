############### INITIALS ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from numpy import interp
from warnings import simplefilter 
import os

os.chdir("/home/seppe/data/Papers/2310.19448/White_Dwarf_AGWB/Src/")

# matplotlib globals
plt.rc('font',   size=16)          # controls default text sizes
plt.rc('axes',   titlesize=18)     # fontsize of the axes title
plt.rc('axes',   labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=14)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=14)     # fontsize of the tick labels
plt.rc('legend', fontsize=18)      # legend fontsize
plt.rc('figure', titlesize=18)     # fontsize of the figure title

# other quantities
s_in_Myr = (u.Myr).to(u.s)

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

def tau_syst(f_0, f_1, K):
    '''
    Calculates tau, the time it takes a binary with K to evolve from f_0 to f_1 (GW frequencies).
    Returns tau in Myr.
    '''
    tau = 2.381*(f_0**(-8/3) - f_1**(-8/3)) / K
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

################ SUB FUNCTIONS ################

def num_merge_bins(model1, model2, data, tag):

    # We will have no merger bin for binaries that have f_max above our region of interest
    highest_bin = model1.f_bins[-1]
    lowest_bin = model2.f_bins[0]
    assert model2.f_bins[-1] == highest_bin
    assert model2.f_bins[0] == lowest_bin

    COUNT1 = 0
    COUNT2 = 0

    for index, row in data.iterrows():

        MERGER_CAN_BE_REACHED = True

        for i, z in enumerate(model1.z_list):

            # Don't consider mergers that happen at a frequency beyond our region of interest
            if 2*row.nu_max/(1+z) > highest_bin:
                continue
            
            evolve_time = model1.z_time_since_max_z[i].value - row.t0
            assert model2.z_time_since_max_z[i].value - row.t0 == evolve_time

            if evolve_time <=0:
                    continue
            
            if MERGER_CAN_BE_REACHED:

                print("-----------------------------")

                for model in [model1, model2]:

                    # find merger bin
                    bin_index = np.digitize(2*row.nu_max/(1+z), model.f_bins)-1
                    low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 

                    print(f"For {model.N}")
                    print(low_f_r, 2*row.nu_max/(1+z),upp_f_r)

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
                        num_syst = psi * tau_syst(low_f_r*(1+z), 2*row.nu_max, row.K) * 10**6 # tau is given in Myr, psi in ... /yr
                        print(num_syst)


            if not MERGER_CAN_BE_REACHED:

                nu_max_b_ini = determine_upper_freq(row.nu0, evolve_time, row.K)

                if nu_max_b_ini == -1:
                    # Should not be possible
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
                prev_num = 0
                for model in [model1, model2]:
                    bin_index = np.digitize(2*nu_max_b/(1+z), model.f_bins)-1
                    if bin_index == -1:
                        print("oops")

                    low_f_r, upp_f_r = model.f_bins[bin_index], model.f_bins[bin_index + 1] 

                    if 2*row.nu0/(1+z) > low_f_r:
                        # in this case the merger is already included as a special case in add birth contribution
                        if model.N == 25:
                            COUNT1 += 1
                        elif model.N == 50:
                            COUNT2 += 1
                        continue

                    tau = tau_syst(2*row.nu0, low_f_r*(1+z), row.K)
                    psi = representative_SFH(model.ages[i].value, tau, model.SFH_num, model.max_z)

                    print(f"For {model.N}")
                    print(tau, psi)

                    num_syst = psi * (evolve_time - tau) * 10**6 # tau is given in Myr, psi in ... /yr
                    print(num_syst)

                    if DEBUG:
                        np.testing.assert_allclose(evolve_time, tau + tau_syst(low_f_r*(1+z), 2*nu_max_b, row.K), rtol=1e-2)

                    num = (4 * np.pi / 4e6)* num_syst * (cosmo.comoving_distance(z).value ** 2) * model.z_widths[i]
                    if num > prev_num and prev_num !=0:
                        assert 0 == 1
                    prev_num = num
                    print(num)        
    
    print(COUNT1)
    print(COUNT2)


############ ACTUAL MAIN FUNCTION #############

def main():
    '''
    The actual main function. Combines the three different components
    '''

    ### initiate ###

    N1 = 25             # number of bins
    N2 = 50
    N_z = 20           # number of z bins
    max_z = 8           # max_redshift
    SFH_num = 1         # which SFH
    tag = "final"       # identifier for filenames
    SMALL_DATA = True

    global SAVE_FIG
    SAVE_FIG = False

    global DEBUG
    DEBUG = False

    # Run script for only one system if True
    global TEST_FOR_ONE
    TEST_FOR_ONE = False

    # create the simulation model
    model1 = sim_model(N1, N_z, max_z, SFH_num)
    model2 = sim_model(N2, N_z, max_z, SFH_num)


    # data. initial file with some added calculations
    data = pd.read_csv("../Data/initials_final_3.txt", sep = ",")

    # Some binaries will never make it to our frequency window
    initial_check = data[data["nu0"] < 5e-6]
    can_not_be_seen = (tau_syst(2*initial_check["nu0"], 1e-5, initial_check["K"]) > 13000)
    actual = data.drop(initial_check[can_not_be_seen].index)
    if SMALL_DATA:
        actual = actual[:10]
    print(f"Out of {len(initial_check)} binaries below 1e-5 Hz, only {len(initial_check) - np.sum(can_not_be_seen)} enters our window.")
    print(f"Dataset reduced from {len(data)} rows to {len(actual)} rows.")


    if TEST_FOR_ONE:
        # info on the first row of data
        print(data.iloc[0])

    num_merge_bins(model1, model2, actual, tag)

main()
