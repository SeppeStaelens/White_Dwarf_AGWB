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
    def __init__(self, N = 50, N_z = 20, max_z = 8, SFH_num = 1, log_f_low = -5, log_f_high = 0) -> None:
        '''!
        Initializes the SimModel object.
        @param N: number of frequency bins.
        @param N_z: number of redshift bins.
        @param max_z: maximum redshift.
        @param SFH_num: which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01.
        @param log_f_low: lower bound of the frequency bins in log10 space.
        @param log_f_high: upper bound of the frequency bins in log10 space.
        @return instance of SimModel, with frequency and redshift bins calculated, and cosmology set.
        '''
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
        '''!
        Calculates the f bins and the bin factors.
        '''    
        f_range = np.logspace(self.log_f_low, self.log_f_high, 2*self.N + 1, base = 10)
        ## The frequencies at which we will plot
        self.f_plot = np.array([f_range[2*i+1] for i in range(self.N)])
        ## The frequency bins
        self.f_bins = np.array([f_range[2*i] for i in range(self.N+1)])
        ## The frequency bin factors that appear in the calculation
        self.f_bin_factors = get_bin_factors(self.f_plot, self.f_bins)

        print(f"\nThe frequencies are {self.f_plot}\n")

    def calculate_z_bins(self):
        '''!
        Calculates the z bins.
        '''
        z_range = np.linspace(0, self.max_z, 2*self.N_z+1)  
        ## The central values of the redshift bins
        self.z_list = np.array([z_range[2*i+1] for i in range(self.N_z)])
        ## The redshift bins
        self.z_bins = np.array([z_range[2*i] for i in range(self.N_z+1)])

        print(f"The redshifts are {self.z_list}\n")

    def calculate_cosmology(self):
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
