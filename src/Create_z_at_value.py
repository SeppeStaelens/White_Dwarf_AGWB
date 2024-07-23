import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import os

print("Current dir: " + os.getcwd())

### SETTINGS ###
max_z = 8
nr_interp = 10000

def main():

    initial_age = cosmo.age(max_z).to(u.Myr)
    current_age = cosmo.age(1e-5).to(u.Myr)

    ages = np.linspace(initial_age.value, current_age.value, nr_interp)
    z_vals = np.array(z_at_value(cosmo.age, ages * u.Myr).value)

    data = pd.DataFrame({"Age (Myr)" : ages, "Redshift" : z_vals})
    data.to_csv("Data/z_at_age.txt", index=False)

def test_fast():

    z_at_val_data = pd.read_csv("../Data/z_at_age.txt", names=["age", "z"], header=1)
    interp_age, interp_z = z_at_val_data.age.values, z_at_val_data.z.values

    print(interp_age)

    initial_age = cosmo.age(max_z).to(u.Myr)
    current_age = cosmo.age(1e-5).to(u.Myr)

    ages = np.linspace(initial_age.value, current_age.value, nr_interp)
    sum = 0
    for age in ages:
        z = np.interp(age, interp_age, interp_z)
        sum += z
    
    print(sum)
    print("done")


test_fast()


