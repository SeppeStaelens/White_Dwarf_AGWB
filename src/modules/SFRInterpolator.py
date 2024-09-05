"""!
@package SFRInterpolator
@brief This module contains the class RedshiftInterpolator.
@details The class SFRIinterpolator is used to quickly determine the SFR at a given age of the Universe.
@author Seppe Staelens
@date 2024-09-01
"""

from numpy import interp
import pandas as pd
from pathlib import Path
import modules.SFH as sfh
import modules.RedshiftInterpolator as ri

class SFRInterpolator:
    """!
    This class is used to quickly determine the SFR at a given age of the Universe.
    """

    def __init__(self, redshift_interpolator: ri.RedshiftInterpolator, SFH_num: int = 1, SFH_type: str = 'MZ19', metallicity: str = 'z02', max_z: float = 8.,) -> None:
        """!
        Initializes the SFRInterpolator object.
        @param redshift_interpolator: RedshiftInterpolator object that interpolates the redshift at a given age.
        @param SFH_num: which star formation history to select. 1: Madau & Dickinson 2014, 2-4: made up, 5: constant 0.01, 6: alternatives Sophie.
        @param SFH_type: type of SFH/SFRD (LZ19, MZ19, HZ19, LZ21, HZ21. Only used in case SFH_num = 6.
        @param metallicity: metallicity range around z0001 (z = 0.0001), z001 (z = 0.001), z005 (z = 0.005), z01 (z = 0.01), z02 (z = 0.02) or z03 (z = 0.03). Only used in case SFH_num = 6.
        @param max_z: maximum redshift.
        """

        self.redshift_interpolator = redshift_interpolator
        self.max_z = max_z

        if SFH_num == 1:
            def SFRimpl(z: float) -> float:
                return sfh.SFH_MD(z)
        elif SFH_num == 2:
            def SFRimpl(z: float) -> float:
                return sfh.SFH2(z)
        elif SFH_num == 3:
            def SFRimpl(z: float) -> float:
                return sfh.SFH3(z)
        elif SFH_num == 4:
            def SFRimpl(z: float) -> float:
                return sfh.SFH4(z)
        elif SFH_num == 5:
            def SFRimpl(z: float) -> float:
                return 0.01
            
        elif SFH_num == 6:
            SFR_at_val_data = pd.read_csv(Path(f"../data/SFRD/{SFH_type}_SFRD_allbins.txt"))
            self.interp_z = SFR_at_val_data.redshift.values

            # Number in square brackets corresponds to different metallicities
            if metallicity == 'z03':
                self.interp_SFR = SFR_at_val_data['0'].values 
            elif metallicity == 'z02':
                self.interp_SFR = SFR_at_val_data['1'].values
            elif metallicity == 'z01':
                self.interp_SFR = SFR_at_val_data['2'].values
            elif metallicity == 'z005':
                self.interp_SFR = SFR_at_val_data['3'].values
            elif metallicity == 'z001':
                self.interp_SFR = SFR_at_val_data['4'].values
            elif metallicity == 'z0001':
                self.interp_SFR = SFR_at_val_data['5'].values
            else:
                raise ValueError("Invalid metallicity value. Choose from 'z03', 'z02', 'z01', 'z005', 'z001' or 'z0001'.")
            
            def SFRimpl(z: float) -> float:
                return interp(z, self.interp_z, self.interp_SFR)
        
        else:
            raise ValueError("Invalid SFH_num value. Choose from 1, 2, 3, 4, 5 or 6.")
    
        self.SFR = SFRimpl

    def SFR(self, z: float) -> float:
        '''!
        @brief Determines the star formation rate at a given redshift.
        @param z: redshift.
        @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
        '''
        print("This is a placeholder function.")

    def representative_SFH(self, age: float, Delta_t: float = 0.) -> float:
        '''!
        @brief Determines an appropriate value for the star formation rate at a given age.
        @details The function looks for a representative value of the star formation rate given the age of the system, and takes into account an optional additional time delay.
        @param age: age of the system in Myr.
        @param Delta_t: time delay due to formation of binary or time required to reach the correct frequency bin, in Myr.
        @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
        '''
        new_age = age - Delta_t
        z_new = self.redshift_interpolator.get_z_fast(new_age)
        if z_new > self.max_z:
            print(f"z larger than {self.max_z}")
        
        return self.SFR(z_new)
