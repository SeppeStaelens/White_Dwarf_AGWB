"""!@file GWB.py
@date 2024-07-26
@brief This program calculates the GWB based on the method described in my thesis, using uniform redshift bins.
@details The program calculates the GWB based on the method described in my thesis, using uniform redshift bins. It is divided into three main parts: the bulk part, the birth part, and the merger part. The bulk part calculates the majority of the GWB, what is referred to in my thesis as the 'generic case'. The birth part adds the contribution of the 'birth bins' to the bulk GWB. The merger part adds the contribution of the 'merger bins' due to Kepler max to the bulk GWB. The program saves a dataframe with all the essential information.
@author Seppe Staelens
"""

# -------- INITIALS -------- #
import sys
# sys.path.append("src/modules")
import os

# change directory to src
if os.getcwd().split("/")[-1] != "src":
    print("Changing directory to src")
    os.chdir("src")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import simplefilter

# ignore pandas warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)

import modules.auxiliary as aux
import modules.SimModel as sm
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

def main():
    '''!
    @brief Main function.
    @details The main functions sets the details of the simulation and runs the three main parts of the program.
    '''

    ## Start time of the program
    start_time = time.time()

    # create the simulation from the parameter file
    model = sm.SimModel(input_file = sys.argv[1])

    # population data
    population = pd.read_csv(model.population_file_name, sep = ",")

    # Some binaries will never make it to our frequency window within a Hubble time
    relevant_population = aux.drop_redundant_binaries(population, model.log_f_low, model.T0)

    if model.TEST_FOR_ONE:
        # info on the first row of data
        print(relevant_population.iloc[0])

    # ----- main part of the program ----- #
    add_bulk(model, relevant_population)
    add_birth(model, relevant_population)
    add_merge(model, relevant_population)

    # total run time
    duration = time.time() - start_time
    print(f"--- duration: {duration//60:.0f} minutes {duration%60:.0f} seconds ---")

metallicities = ['z0001', 'z001', 'z005', 'z01', 'z02', 'z03']
for m in range(len(metallicities)):
    metallicity = metallicities [m]
    main()
