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
from  modules.auxiliary import get_bin_factors, get_width_z_shell_from_z
import modules.RedshiftInterpolator as ri
import modules.SFRInterpolator as sfri
import configparser as cfg

class SimModel:
    """
    ! This class contains information about the run that needs to be shared over the different subroutines.
    """

    ## lower bound of the frequency bins in log10 space.
    log_f_low: float
    ## upper bound of the frequency bins in log10 space.
    log_f_high: float
    ## number of frequency bins
    N_freq: int
    ## number of integration bins (z or T)
    N_int: int
    ## max_redshift
    max_z: float
    ## which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01.
    SFH_num: int
    ## type of star formation history. Can be 'MZ19', 'LZ19', 'HZ19', 'LZ21', 'HZ21' or 'MD' (Madau & Dickinson). Only used in case SFH_num = 6
    SFH_type: str
    ## metallicity, can be 'z0001', 'z001', 'z005', 'z01', 'z02' or 'z03'
    metallicity: str
    ## population synthesis model, can be 'AlphaAlpha' or 'GammaAlpha'
    pop_synth: str
    ## numerical value for alpha, can be 'Alpha1' or 'Alpha4'
    alpha: str
    ## normalisation for the population synthesis files in solar masses. 3.4e6 for Sophie, 4e6 for Seppe.
    normalisation: float
    ## population file
    population_file_name: str
    ## redshift interpolator file
    ri_file: str
    ## tag for filenames
    tag: str
    ## Integrate over "redshift" or (cosmic) "time"
    INTEG_MODE: str
    ## Run script with(out) saving figures
    SAVE_FIG: bool
    ## Run script with(out) more output
    DEBUG: bool
    ## Run script for only one system if True
    TEST_FOR_ONE: bool       

    ## Redshift interpolator used for quick conversions in the cosmology
    z_interp: ri.RedshiftInterpolator
    ## Star Formation Rate interpolator to determine the SFR at a given redshift
    sfr_interp: sfri.SFRInterpolator

    ## The speed of light in units of Mpc/Myr
    light_speed = 0.30660139 
    ## lookback time to maximal redshift in Myr
    T0: float
    ## prefactor for the bulk calculations. value = 2.4e-15, value = 2e-15 for Seppe
    omega_prefactor_bulk: float
    ## prefactor for the birth and merger calculations. value = 3.76e-15, value = 3.2e-15 for Seppe
    omega_prefactor_birth_merger: float

    def __init__(self, input_file: str, metallicity: str) -> None:
        '''!
        Initializes the SimModel object: reads parameters, sets interpolators and calculates the bins and cosmology.
        @param input_file: parameter file.
        @param metallicity: metallicity of the simulation.
        '''
        self.metallicity = metallicity
        self.read_params(input_file)

        self.z_interp = ri.RedshiftInterpolator(self.ri_file)
        self.sfr_interp = sfri.SFRInterpolator(self.z_interp, self.SFH_num, self.SFH_type, self.metallicity, self.max_z)
        
        self.calculate_f_bins()

        if self.INTEG_MODE == "redshift":
            self.calculate_z_bins()
            self.calculate_cosmology_from_z()
        elif self.INTEG_MODE == "time":
            self.calculate_T_bins()
            self.calculate_cosmology_from_T(self.z_interp)

    def read_params(self, input_file: str) -> None:
        '''!
        Reads the parameters from the config file.
        @param input_file: parameter file.
        '''
        config = cfg.ConfigParser()
        config.read(input_file)

        self.log_f_low = config.getfloat('integration', 'log_f_low', fallback=-5)
        self.log_f_high = config.getfloat('integration', 'log_f_high', fallback=0)
        assert self.log_f_low < self.log_f_high, "log_f_low should be smaller than log_f_high"
        self.N_freq = config.getint('integration', 'N_freq', fallback=50)
        self.N_int = config.getint('integration', 'N_int', fallback=20)
        self.max_z = config.getfloat('integration', 'max_z', fallback=8)

        self.SFH_num = config.getint('physics', 'SFH_num', fallback=1)
        if self.SFH_num != 6:
            print("SFH_num is not 6, so SFH_type and metallicity are ignored.")
        self.SFH_type = config.get('physics', 'SFH_type', fallback='MZ19')
        self.pop_synth = config.get('physics', 'pop_synth', fallback='GammaAlpha')
        self.alpha = config.get('physics', 'alpha', fallback='Alpha4')
        self.normalisation = config.getfloat('physics', 'normalisation', fallback=4e6)  
        self.omega_prefactor_bulk = 8.10e-9 / self.normalisation                        
        self.omega_prefactor_birth_merger = 1.28e-8 / self.normalisation                

        self.population_file_name = "../data/" + self.pop_synth + "/" + self.alpha + "/" + self.metallicity + "/" + "Initials_" + self.metallicity
        if config.getboolean('files', 'use_data_Seppe', fallback=False):
            self.population_file_name += "_Seppe"
        self.population_file_name += ".txt.gz"
        self.ri_file = "../data/" + config.get('files', 'ri_file', fallback="z_at_age.txt")

        self.tag = config.get('settings', 'tag', fallback="")
        self.INTEG_MODE = config.get('settings', 'integration_mode', fallback="redshift")
        self.SAVE_FIG = config.getboolean('settings', 'save_fig', fallback=False)
        self.DEBUG = config.getboolean('settings', 'debug', fallback=False)
        self.TEST_FOR_ONE = config.getboolean('settings', 'test_for_one', fallback=False)

    def calculate_f_bins(self) -> None:
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

    def calculate_z_bins(self) -> None:
        '''!
        Calculates the z bins.
        '''
        z_range = np.linspace(0, self.max_z, 2*self.N_int+1)  
        ## The central values of the redshift bins
        self.z_list = np.array([z_range[2*i+1] for i in range(self.N_int)])
        ## The redshift bins
        self.z_bins = np.array([z_range[2*i] for i in range(self.N_int+1)])

        print(f"The redshifts are {self.z_list}\n")

    def calculate_T_bins(self) -> None:
        '''!
        Calculates the T bins.
        '''
        self.T0 = cosmo.lookback_time(self.max_z).to(u.Myr)

        self.T_range = np.linspace(0, self.T0.value, 2*self.N_int+1)
    
        self.T_list = np.array([self.T_range[2*i+1] for i in range(self.N_int)])
        self.T_bins = np.array([self.T_range[2*i] for i in range(self.N_int+1)])

        self.dT = (self.T_list[1] - self.T_list[0])

        print(f"The cosmic timestep is {self.dT} Myr\n")
        print(f"The times are {self.T_list}\n")

    def calculate_cosmology_from_z(self) -> None:
        '''!
        Calculations depending on the cosmology, starting from redshift bins. 
        Sets the widths of the z bins and the time since max z, as well
        as the age of the universe at each redshift.
        '''
        ## The width of the redshift bins in Mpc
        self.z_widths = get_width_z_shell_from_z(self.z_bins)  
        ## The time since the maximum redshift  
        self.z_time_since_max_z = (cosmo.lookback_time(self.max_z) - cosmo.lookback_time(self.z_list)).to(u.Myr)
        ## The age of the universe at each redshift
        self.ages = (cosmo.age(0) - cosmo.lookback_time(self.z_list)).to(u.Myr)
    
        self.T0 = cosmo.lookback_time(self.max_z).to(u.Myr)

    def calculate_cosmology_from_T(self, z_interpolator: ri.RedshiftInterpolator) -> None:
        '''!
        Calculations depending on the cosmology, starting from cosmic time bins. 
        Calculates the redshifts, the time since the maximum redshift, and the ages of the universe at each time.
        '''
        self.ages = cosmo.age(0).to(u.Myr) - self.T_list * u.Myr
        self.z_list = z_interpolator.get_z_fast(self.ages.value)
        self.z_time_since_max_z = (self.T0.value - self.T_list) * u.Myr

        print(f"The redshifts are {self.z_list}\n")
