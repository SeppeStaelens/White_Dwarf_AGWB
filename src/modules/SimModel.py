from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u
from auxiliary import get_bin_factors, get_width_z_shell_from_z

class SimModel:
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
        self.f_bin_factors = get_bin_factors(self.f_plot, self.f_bins)

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
