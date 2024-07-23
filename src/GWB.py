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
from warnings import simplefilter 
import time

import modules.auxiliary as aux
import modules.SimModel as sm
from modules.add_bulk import add_bulk
from modules.add_birth import add_birth
from modules.add_merge import add_merge

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
a, b, c = aux.calc_parabola_vertex(-3, -12, -2.5, -12.5, -2, -12)


def main():
    '''!
    @brief Main function.
    @details The main functions sets the details of the simulation and runs the three main parts of the program.
    '''

    ## Start time of the program
    start_time = time.time()

    ### initiate ###

    ## number of bins
    N = 50              
    N_z = 20            ## number of z bins
    max_z = 8           ## max_redshift
    SFH_num = 1         ## which SFH
    tag = "final"       ## identifier for filenames

    # normalisation = 3.4e6 # in solar masses, change if necessary, 4e6 voor Seppe
    # omega_prefactor_bulk = 8.10e-9 / normalisation # waarde = 2.4e-15,  2e-15 voor Seppe
    # omega_prefactor_birth_merger = 1.28e-8 / normalisation # waarde = 3.75e-15 # 3.2e-15 voor Seppe

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
    can_not_be_seen = (aux.tau_syst(2*initial_check["nu0"], 1e-5, initial_check["K"]) > 13000)
    actual = data.drop(initial_check[can_not_be_seen].index)
    print(f"Out of {len(initial_check)} binaries below 1e-5 Hz, only {len(initial_check) - np.sum(can_not_be_seen)} enters our window.")
    print(f"Dataset reduced from {len(data)} rows to {len(actual)} rows.")


    if TEST_FOR_ONE:
        # info on the first row of data
        print(data.iloc[0])

    add_bulk(model, actual, tag)
    add_birth(model, actual, tag)
    add_merge(model, actual, tag)

    #@var Total run time
    duration = time.time() - start_time
    print("--- duration: %s minutes %s seconds ---" % (duration//60, duration%60))

main()
