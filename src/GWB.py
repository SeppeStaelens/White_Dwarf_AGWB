"""!@file GWB.py
@date 2024-07-26
@brief This program calculates the GWB based on the method described in my thesis, using uniform redshift bins.
@details The program calculates the GWB based on the method described in my thesis, using uniform redshift bins. It is divided into three main parts: the bulk part, the birth part, and the merger part. The bulk part calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'. The birth part adds the contribution of the 'birth bins' to the bulk GWB. The merger part adds the contribution of the 'merger bins' due to Kepler max to the bulk GWB. The program saves a dataframe with all the essential information.
@author Seppe Staelens
"""

# -------- INITIALS -------- #
import numpy as np
import pandas as pd
from astropy import units as u
from warnings import simplefilter 
import time
import matplotlib.pyplot as plt

import modules.auxiliary as aux
import modules.SimModel as sm
import modules.RedshiftInterpolator as ri
from modules.add_bulk import add_bulk
from modules.add_birth import add_birth
from modules.add_merge import add_merge

# matplotlib globals
plt.rc('font',   size=16)          # controls default text sizes
plt.rc('axes',   titlesize=18)     # fontsize of the axes title
plt.rc('axes',   labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick',  labelsize=14)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=14)     # fontsize of the tick labels
plt.rc('legend', fontsize=18)      # legend fontsize
plt.rc('figure', titlesize=18)     # fontsize of the figure title

# ignore pandas warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

global s_in_Myr 
s_in_Myr = (u.Myr).to(u.s)

def main():
    '''!
    @brief Main function.
    @details The main functions sets the details of the simulation and runs the three main parts of the program.
    '''

    ## Start time of the program
    start_time = time.time()

    # ----- initiate ----- #

    ## number of frequency bins
    N_freq = 50
    ## number of integration bins (z or T)              
    N_int = 20
    ## max_redshift            
    max_z = 8
    ## which SFH
    SFH_num = 1
    ## population file
    population_file_name = "../Data/aa_4_0p02_MD/initials_final.txt"      
    ## tag for filenames
    tag = ""       

    # normalisation = 3.4e6 # in solar masses, change if necessary, 4e6 voor Seppe
    # omega_prefactor_bulk = 8.10e-9 / normalisation # waarde = 2.4e-15,  2e-15 voor Seppe
    # omega_prefactor_birth_merger = 1.28e-8 / normalisation # waarde = 3.75e-15 # 3.2e-15 voor Seppe

    # Integrate over "redshift" or (cosmic) "time"
    INTEGRATION_MODE = "redshift"
    # Run script with(out) saving figures
    SAVE_FIG = False
    # Run script with(out) more output
    DEBUG = False
    # Run script for only one system if True
    TEST_FOR_ONE = False

    # create the simulation 
    z_interp = ri.RedshiftInterpolator("../data/z_at_age.txt")
    model = sm.SimModel(INTEGRATION_MODE, z_interp, N_freq, N_int, max_z, SFH_num)
    model.set_mode(SAVE_FIG, DEBUG, TEST_FOR_ONE)

    # data. initial file with some added calculations
    population = pd.read_csv(population_file_name, sep = ",")

    # Some binaries will never make it to our frequency window within a Hubble time
    initial_check = population[population["nu0"] < 5e-6]
    can_not_be_seen = (aux.tau_syst(2*initial_check["nu0"], 1e-5, initial_check["K"]) > 13000)
    relevant_population = population.drop(initial_check[can_not_be_seen].index)
    print(f"Out of {len(initial_check)} binaries below 1e-5 Hz, only {len(initial_check) - np.sum(can_not_be_seen)} enters our window.")
    print(f"Dataset reduced from {len(population)} rows to {len(relevant_population)} rows.")

    if TEST_FOR_ONE:
        # info on the first row of data
        print(population.iloc[0])

    # ----- main part of the program ----- #
    add_bulk(model, relevant_population, z_interp, tag)
    add_birth(model, relevant_population, z_interp, tag)
    add_merge(model, relevant_population, z_interp, tag)

    # total run time
    duration = time.time() - start_time
    print("--- duration: %s minutes %s seconds ---" % (duration//60, duration%60))

main()
