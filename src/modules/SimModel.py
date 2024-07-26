"""!
@package SimModel
@brief This module contains the class SimModel.
@details The class SimModel contains information about the run that needs to be shared over the different subroutines.
@author Seppe Staelens
@date 2024-07-24
"""

from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u
from auxiliary import get_bin_factors, get_width_z_shell_from_z

class SimModel:
    """
    ! This class contains information about the run that needs to be shared over the different subroutines.
    """
    def __init__(self, INTEG_MODE, z_interp, N_freq = 50, N_int = 20, max_z = 8, SFH_num = 1, log_f_low = -5, log_f_high = 0) -> None:
        '''!
        Initializes the SimModel object.
        @param N_freq: number of frequency bins.
        @param N_int: number of integration bins (z or T).
        @param max_z: maximum redshift.
        @param SFH_num: which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01.
        @param log_f_low: lower bound of the frequency bins in log10 space.
        @param log_f_high: upper bound of the frequency bins in log10 space.
        @return instance of SimModel, with frequency and redshift bins calculated, and cosmology set.
        '''
        self.N_freq = N_freq
        self.N_int = N_int
        self.max_z = max_z
        self.SFH_num = SFH_num
        assert log_f_low < log_f_high, "log_f_low should be smaller than log_f_high"
        self.log_f_low = log_f_low
        self.log_f_high = log_f_high

        self.calculate_f_bins()
        self.INTEG_MODE = INTEG_MODE

        if INTEG_MODE == "redshift":
            self.calculate_z_bins()
            self.calculate_cosmology_from_z()
        elif INTEG_MODE == "time":
            self.calculate_T_bins()
            self.calculate_cosmology_from_T(z_interpolator=z_interp)

    def calculate_f_bins(self):
        '''!
        Calculates the f bins and the bin factors.
        '''    
        f_range = np.logspace(self.log_f_low, self.log_f_high, 2*self.N_freq + 1, base = 10)
        ## The frequencies at which we will plot
        self.f_plot = np.array([f_range[2*i+1] for i in range(self.N_freq)])
        ## The frequency bins
        self.f_bins = np.array([f_range[2*i] for i in range(self.N_freq+1)])
        ## The frequency bin factors that appear in the calculation
        self.f_bin_factors = get_bin_factors(self.f_plot, self.f_bins)

        print(f"\nThe frequencies are {self.f_plot}\n")

    def calculate_z_bins(self):
        '''!
        Calculates the z bins.
        '''
        z_range = np.linspace(0, self.max_z, 2*self.N_int+1)  
        ## The central values of the redshift bins
        self.z_list = np.array([z_range[2*i+1] for i in range(self.N_int)])
        ## The redshift bins
        self.z_bins = np.array([z_range[2*i] for i in range(self.N_int+1)])

        print(f"The redshifts are {self.z_list}\n")

    def calculate_T_bins(self):
        self.T0 = cosmo.lookback_time(self.max_z).to(u.Myr)

        self.T_range = np.linspace(0, self.T0.value, 2*self.N_t+1)
    
        self.T_list = np.array([self.T_range[2*i+1] for i in range(self.N_t)])
        self.T_bins = np.array([self.T_range[2*i] for i in range(self.N_t+1)])

        self.dT = (self.T_list[1] - self.T_list[0])

        print(f"The cosmic timestep is {self.dT} Myr\n")
        print(f"The times are {self.T_list}\n")

    def calculate_cosmology_from_z(self):
        '''!
        Calculations depending on the cosmology. Sets the widths of the z bins and the time since max z, as well
        as the age of the universe at each redshift.
        '''
        ## The width of the redshift bins in Mpc
        self.z_widths = get_width_z_shell_from_z(self.z_bins)  
        ## The time since the maximum redshift  
        self.z_time_since_max_z = (cosmo.lookback_time(self.max_z) - cosmo.lookback_time(self.z_list)).to(u.Myr)
        ## The age of the universe at each redshift
        self.ages = (cosmo.age(0) - cosmo.lookback_time(self.z_list)).to(u.Myr)
    
    def calculate_cosmology_from_T(self, z_interpolator):

        self.ages = cosmo.age(0).to(u.Myr) - self.T_list * u.Myr
        self.z_list = z_interpolator.get_z_fast(self.ages.value)
        self.z_time_since_max_z = (self.T0.value - self.T_list) * u.Myr

        print(f"The redshifts are {self.z_list}\n")

    def set_mode(self, SAVE_FIG, DEBUG, TEST_FOR_ONE):
        '''!
        Sets the mode of the simulation.
        @param SAVE_FIG: whether to save the figures.
        @param DEBUG: whether to print more output.
        @param TEST_FOR_ONE: whether to test for only one system.
        @param INT_MODE: whether to integrate over redshift or time.
        '''
        self.SAVE_FIG = SAVE_FIG
        self.DEBUG = DEBUG
        self.TEST_FOR_ONE = TEST_FOR_ONE
