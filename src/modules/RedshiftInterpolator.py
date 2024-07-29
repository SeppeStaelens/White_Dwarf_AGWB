"""!
@package RedshiftInterpolator
@brief This module contains the class RedshiftInterpolator.
@details The class RedshiftInterpolator is used to quickly determine the redshift at a given age of the Universe.
@author Seppe Staelens
@date 2024-07-24
"""

from numpy import interp
import pandas as pd

class RedshiftInterpolator:
    """!
    This class is used to quickly determine the redshift at a given age of the Universe.
    """
    def __init__(self, z_at_age_file: str) -> None:
        """!
        Initializes the RedshiftInterpolator object.
        @param z_at_age_file: file containing the redshift at a given age of the Universe.
        """
        z_at_val_data = pd.read_csv(z_at_age_file, names=["age", "z"], header=1)
        ## The age of the Universe at which the redshift is determined
        self.interp_age = z_at_val_data.age.values
        ## The redshift at the given age of the Universe
        self.interp_z = z_at_val_data.z.values
    
    def get_z_fast(self, age: float) -> float:
        """!
        Quickly determine the redshift at a given age of the Universe.
        @param age: age of the Universe in Myr.
        @return redshift at the given age of the Universe.
        """
        return interp(age, self.interp_age, self.interp_z)
