import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy import optimize

# Global constants
c = 3.e10              # cm/s
Ro = 6.957e10          # cm
Mo = 1.989e30          # kg
secperyr = 3.1556926
m_p = 1.67e-27         # kg
m_e = 9.11e-31         # kg
e = 1.602e-19          # coulombs
epsilon = 8.854e-12    # F/m

def density_profile():
    """
    PRELIMINARY!
    Fits function to dispersion measure value to estimate density profile
    """
    def dist(x,y,z):
        return (x**2+y**2+z**2)**.5

    def fitfunc(p,x):
        y_vals = np.linspace(-1.e11, 1.e11, 2000)
        p_arr = np.empty((y_vals.size, x.size))
        for y in np.arange(y_vals.size):
            yy = y_vals[y]
            r = dist(x,yy,0)          # Should be able to change z to our LoS ??
            density = p[0] * np.exp(-r/p[1])
            p_arr[y,:] = density * 100100100.1
        return np.sum(p_arr,axis=0)

    errfunc = lambda p,x,y: fitfunc(p,x)-y
    p1,success = optimize.leastsq(errfunc,p0[:],args=(R_E,N_e))

def cyclotron(a,E_dot,inclination=90.):
    """
    a in units of solar radius
    """
    d = a*Ro*np.cos(inclination*np.pi/180.) # distance to eclipse region in cm
    U_E = E_dot/(4*np.pi*c*d**2)
    B_E = (U_E*8*np.pi)**0.5 # magnetic field in eclipse region
    v_B = e*B_E / (2*np.pi*m_e*c) # fundamental cyclotron freq
    return 'U_E =',U_E,'ergs/cm^3. B_E =',B_E,'G. v_B =',v_B,'MHz.'

def free_free(freq,N_e,L):

def plasma_freq(n_e):
    """
    n_e = electron density (cm^-3)
    """
    n_e = n_e * 100**3    # convert to m^-3
    f_p = (1 / (2*np.pi)) * ((n_e * e**2) / (m_e * epsilon))**0.5
    return 'Plasma frequency, f_p =', f_p

def mass_loss(R_E,n_e,U_E):
    """
    Mass loss assuming ablated material entrained in pulsar wind
    (Thompson et al., 1994).
    n_e = Density in cm^-3
    R_E = Radius of mass loss projected circle in solar radii units
    U_E = Energy density of pulsar wind at companion in ergs/cm^-3
    """
    U_E = U_E * 1e-7     # convert to Joules
    R_E = R_E * Ro       # convert to cm
    V_w = (U_E / (n_e * m_p))**0.5  # Wind velocity for balanced momentum fluxes
    M_dot = np.pi * (R_E**2) * m_p * n_e * V_w  # Mass loss through circle of radiu R_E
    M_dot = M_dot * secperyr / Mo    # convert to solar mass per year
    return 'Mass loss =', M_dot, 'Mo/yr.'
