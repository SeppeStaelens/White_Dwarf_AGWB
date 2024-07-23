from auxiliary import get_z_fast

def representative_SFH(age, Delta_t, SFH_num, max_z):
    '''
    Looks for a representative value of the SFH given the age of the system, and an additional time delay in reaching the bin.
    age and Delta_t should be given in Myr.
    '''
    new_age = age - Delta_t
    z_new = get_z_fast(new_age)
    if z_new > max_z:
        print(f"z larger than {max_z}")

    if SFH_num == 1:
        return SFH(z_new)
    if SFH_num == 2:
        return SFH2(z_new)
    if SFH_num == 3:
        return SFH3(z_new)
    if SFH_num == 4:
        return SFH4(z_new)
    if SFH_num == 5:
        return 0.01
    
def get_SFH(SFH_num, z, age, t0, max_z):
    new_age = age - t0
    z_new = get_z_fast(new_age)
    if z_new > max_z:
        print(f"z larger than {max_z}")
    if SFH_num == 1:
        return SFH(z_new)
    if SFH_num == 2:
        return SFH2(z_new)
    if SFH_num == 3:
        return SFH3(z_new)
    if SFH_num == 4:
        return SFH4(z_new)
    if SFH_num == 5:
        return 0.01

def SFH(z):
    '''!
    @brief Star formation history from [Madau, Dickinson 2014].
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.015*(1+z)**(2.7)/(1+((1+z)/2.9)**(5.6)) 

def SFH2(z):
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.143*(1+z)**(0.3)/(1+((1+z)/2.9)**(3.2)) 

def SFH3(z):
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.00533*(1+z)**(2.7)/(1+((1+z)/2.9)**(3.))

def SFH4(z):
    '''!
    @brief Made up star formation history.
    @param z: redshift.
    @return SFR: star formation rate. Units: solar mass / yr / Mpc^3.
    '''
    return 0.00245*(1+z)**(2.7)/(1+((1+z)/5.)**(5.6))
