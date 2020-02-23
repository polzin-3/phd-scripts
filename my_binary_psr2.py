#!/usr/bin/env python
import numpy as np
import scipy, scipy.optimize
import argparse
import parfile


ARCSECTORAD = float('4.8481368110953599358991410235794797595635330237270e-6')
RADTOARCSEC = float('206264.80624709635515647335733077861319665970087963')
SECTORAD    = float('7.2722052166430399038487115353692196393452995355905e-5')
RADTOSEC    = float('13750.987083139757010431557155385240879777313391975')
RADTODEG    = float('57.295779513082320876798154814105170332405472466564')
DEGTORAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RADTOHRS    = float('3.8197186342054880584532103209403446888270314977710')
HRSTORAD    = float('2.6179938779914943653855361527329190701643078328126e-1')
PI          = float('3.1415926535897932384626433832795028841971693993751')
TWOPI       = float('6.2831853071795864769252867665590057683943387987502')
PIBYTWO     = float('1.5707963267948966192313216916397514420985846996876')
SECPERDAY   = float('86400.0')
SECPERJULYR = float('31557600.0')
KMPERPC     = float('3.0856776e13')
KMPERKPC    = float('3.0856776e16')
Tsun        = float('4.925490947e-6') # sec
Msun        = float('1.9891e30')      # kg
Mjup        = float('1.8987e27')      # kg
Rsun        = float('6.9551e8')       # m
Rearth      = float('6.378e6')        # m
SOL         = float('299792458.0')    # m/s
MSUN        = float('1.989e+30')      # kg
G           = float('6.673e-11')      # m^3/s^2/kg 
C           = SOL


#------------------------------------------------------------------------------
##### My own version of binary_psr, adapted from PRESTO

def myasarray(a):
    #if type(a) in [type(1.0),type(1L),type(1),type(1j)]:
    #    a = np.asarray([a])
    #if len(a) == 0:
    #    a = np.asarray([a])
    a = np.atleast_1d(a)
    return a

def shapR(m2):
    """
    shapR(m2):
        Return the Shapiro 'R' parameter (in sec) with m2 in
            solar units.
    """
    return Tsun * m2

def shapS(m1, m2, x, pb):
    """
    shapS(m1, m2, x, pb):
        Return the Shapiro 'S' parameter with m1 and m2 in
            solar units, x (asini/c) in sec, and pb in days.
            The Shapiro S param is also equal to sin(i).
    """
    return x * (pb*SECPERDAY/TWOPI)**(-2.0/3.0) * \
           Tsun**(-1.0/3.0) * (m1 + m2)**(2.0/3.0) * 1.0/m2

class binary_psr:
    """
    This class reads in a parfile (the only option for instantiation) of
    a binary pulsar.  It allows access the calculation of the mean,
    eccentric, and true anomalies, orbital position, radial velocity,
    and predicted spin period as a function of time.
    If 'leap' is set to true, the calculations will take into account
    the occasional leap seconds. Those are listed in the leap_MJDs
    variable.
            
    Example:
        bpsr = binary_psr('parfile.par', leap=True)
        next_eclipse = bpsr.next_eclipse(56550.)
    """
    def __init__(self, parfilenm, leap=True):
        self.par = parfile.psr_par(parfilenm)
        if not hasattr(self.par, 'BINARY'):
            print "'%s' doesn't contain parameters for a binary pulsar!"
            return None
        self.PBsec = self.par.PB*SECPERDAY
        self.T0 = self.par.T0
        self.leap = leap
        self.leap_MJDs = np.r_[41499., 41683., 42048., 42413., 42778., 43144., 43509., 43874., 44239., 44786., 45151., 45516., 46247., 47161., 47892., 48257., 48804., 49169., 49534., 50083., 50630., 51179., 53736., 54832., 56109., 57204.]

    def approx_anoms(self, MJD, radians=False):
        """
        Return the true anomaly using a third order approximation to the 
        eccentric anomaly given the barycentric epoch in MJD(s).

        Accuracy is within +/-10% at eccentricity = 0.7.
        Accuracy is within +/-5% at eccentricity = 0.6.
        Accuracy is within +/-2% at eccentricity = 0.5.
        Accuracy is within +/-0.3% at eccentricity = 0.3.

        MJD : float, ndarray, list
            Barycentric MJD(s).
        radians : bool
            True will return radians, False will return orbital cycle.
        """
        difft = self.diff_time(MJD)
        mean_anom = np.mod(difft/self.PBsec, 1)*TWOPI
        e = self.par.E
        sinM = np.sin(mean_anom)
        cosM = np.cos(mean_anom)
        ecc_anom = mean_anom + e*sinM + e**2*sinM*cosM + 0.5*e**3*sinM*(3*cosM**2-1)
        ta = 2*np.arctan2(np.sqrt(1+e)*np.sin(ecc_anom*0.5), np.sqrt(1-e)*np.cos(ecc_anom*0.5))
        ta = np.mod(ta, TWOPI)
        if not radians:
            mean_anom /= TWOPI
            ecc_anom /= TWOPI
            true_anom /= TWOPI
        return np.array([mean_anom, ecc_anom, true_anom])

    def calc_anoms(self, MJD, radians=False):
        """
        Return the mean, eccentric, and true anomalies at the 
        barycentric epoch MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        radians : bool
            True will return radians, False will return orbital cycle.
        """
        mean_anom = self.mean_anomaly(MJD, radians=True)
        ecc_anom = self.eccentric_anomaly(mean_anom, radians=True)
        true_anom = self.true_anomaly(ecc_anom, radians=True)
        if not radians:
            mean_anom /= TWOPI
            ecc_anom /= TWOPI
            true_anom /= TWOPI
        return np.array([mean_anom, ecc_anom, true_anom])

    def calc_omega(self, MJD):
        """
        Return the argument of periastron (in radians) at the barycentric
        MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        """
        difft = self.diff_time(MJD)
        if hasattr(self.par, 'OMDOT'):
            # Note:  This is an array
            return (self.par.OM + difft/SECPERJULYR*self.par.OMDOT)*DEGTORAD
        elif hasattr(self.par, 'OM'):
            return self.par.OM*DEGTORAD
        else:
            return 0.

    def calc_orbph(self, MJD, radians=False):
        """
        Return the orbital phase as seen from an observer on Earth. The phase
        corresponds to the true anomaly to which the argument of the ascending
        node is added.

        By defaults, phases are in units of orbital cycle (0-1). Phases are
        defined from the ascending node of the pulsar (0.25 being the superior
        conjunction).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        radians : bool
            True will return radians, False will return orbital cycle.
        """
        mean_anom, ecc_anom, true_anom = self.calc_anoms(MJD, radians=True)
        ws = self.calc_omega(MJD)
        orb_phs = np.mod(true_anom + ws, TWOPI)
        if not radians:
            orb_phs /= TWOPI
        return orb_phs

    def demodulate_TOAs(self, MJD):
        """
        Return arrival times correctly orbitally de-modulated using
        the iterative procedure described in Deeter, Boynton, and Pravdo
        (1981ApJ...247.1003D, thanks, Deepto!).  This corrects for the
        fact that the emitted times are what you want when you only
        have the arrival times.

        MJD : float, ndarray, list
            Barycentric MJD(s).        
        """
        ts = MJD[:]  # start of iteration
        dts = np.ones_like(MJD)
        # This is a simple Newton's Method iteration based on
        # the code orbdelay.c written by Deepto Chakrabarty
        while (np.maximum.reduce(np.fabs(dts)) > 1e-10):
            # radial position in lt-days
            xs = -self.position(ts, inc=90.0)[0]/86400.0
            # radial velocity in units of C
            dxs = self.radial_velocity(ts)*1000.0/SOL
            dts = (ts + xs - MJD) / (1.0 + dxs)
            ts = ts - dts
        return ts

    def diff_time(self, MJD):
        """
        Returns the time difference, in seconds, between an MJD(s) and the
        reference epoch of the binary, T0, from the parfile.

        MJD : float, ndarray, list
            Barycentric MJD(s).
        MJDref : float
            Barycentric reference epoch.
        """
        MJD = myasarray(MJD)
        MJDref = self.T0
        difft = (MJD - MJDref)*SECPERDAY
        if self.leap:
            nleap = (MJD[...,np.newaxis] > self.leap_MJDs[np.newaxis,...]).sum(-1) - (MJDref > self.leap_MJDs).sum(-1)
            difft += nleap
        return difft

    def doppler_period(self, MJD):
        """
        Return the observed pulse spin period in sec at the given MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        """
        vs = self.radial_velocity(MJD)*1000.0 # m/s
        return self.par.P0*(1.0+vs/SOL)

    def eccentric_anomaly(self, ma, radians=False):
        """
        Return the eccentric anomaly in radians, given the mean anomaly.

        ma : float, ndarray, list
            Mean anomaly(ies) in radians.
        radians : bool
            True will return radians, False will return orbital cycle.
        """
        ma = myasarray(ma)
        ma = np.mod(ma, TWOPI)
        ## Method 1
        if 1:
            ecc = self.par.ECC
            ea_old = ma
            ea = ma + ecc*np.sin(ea_old)
            ## This is a simple iteration to solve Kepler's Equation
            while (np.maximum.reduce(np.fabs(ea-ea_old)) > 5e-15):
                ea_old = ea[:]
                ea = ma + ecc*np.sin(ea_old)
            ea = np.mod(ea, TWOPI)
        ## Method 2
        else:
            e = self.par.ECC
            ea = np.empty_like(ma)
            for i in np.arange(ma.size):
                ea[i] = scipy.optimize.bisect(lambda E: E-e*np.sin(E)-ma[i], 0., TWOPI)
        if not radians:
            ea /= TWOPI
        return ea

    def mean_anomaly(self, MJD, radians=False):
        """
        Return the mean anomaly, in radians at the barycentric epoch MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        radians : bool
            True will return radians, False will return orbital cycle.
        """
        difft = self.diff_time(MJD)
        mean_anom = np.mod(difft/self.PBsec, 1)
        if radians:
            mean_anom *= TWOPI
        return mean_anom

    def most_recent_peri(self, MJD):
        """
        Return the MJD(s) of the most recent periastrons that occurred
        before the input MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        """
        mean_anom = self.mean_anomaly(MJD, radians=False)
        days_since_peri = mean_anom * self.par.PB
        return MJD - days_since_peri

    def next_eclipse(self, MJD):
        """
        Return the MJD(s) of the next eclipse following the input MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        """
        ## For the bisection to work, we need to find the right interval
        ## which will run from the current point down to one orbit before
        MJD = myasarray(MJD)
        next_eclipse = np.empty_like(MJD)
        for i in np.arange(MJD.size):
            current_phs = self.calc_orbph(MJD[i], radians=False)
            eclipse_phs = np.mod(0.25-current_phs, 1)
            func = lambda x: np.mod(self.calc_orbph(x)-current_phs, 1) - eclipse_phs
            next_eclipse[i] = scipy.optimize.bisect(func, MJD[i], MJD[i]+self.par.PB*0.9999)
        return next_eclipse

    def position(self, MJD, inc=60.0):
        """
        Return the 'x' (along the LOS with + being towards us) and 'y' (in the
        plane of the sky with + being away from the line of nodes and -
        being in the direction of the line of nodes) positions of the
        pulsar with respect to the center of mass in units of lt-sec.
        (Note:  This places the observer at (+inf,0.0) and the line of nodes
        extending towards (0.0,-inf) with the pulsar orbiting (0.0,0.0)
        clockwise).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        inc : float
            Orbital inclination
        """
        ma, ea, ta = self.calc_anoms(MJD, radians=True)
        orb_phs = self.calc_orbph(MJD, radians=True)
        sini = np.sin(inc*DEGTORAD)
        x = self.par.A1/sini
        r = x*(1.0-self.par.E**2)/(1.0+self.par.E*np.cos(ta))
        return -r*np.sin(orb_phs)*sini, -r*np.cos(orb_phs)

    def radial_velocity(self, MJD):
        """
        Return the radial velocity of the pulsar (km/s) at the given MJD(s).

        MJD : float, ndarray, list
            Barycentric MJD(s).
        """
        ma, ea, ta = self.calc_anoms(MJD, radians=True)
        ws = self.calc_omega(MJD)
        c1 = TWOPI*self.par.A1/self.PBsec
        c2 = np.cos(ws)*np.sqrt(1-self.par.E**2)
        sws = np.sin(ws)
        cea = np.cos(ea)
        return SOL/1000.0*c1*(c2*cea - sws*np.sin(ea)) / (1.0 - self.par.E*cea)

    def shapiro_delays(self, R, S, ecc_anoms):
        """
        Return the predicted Shapiro delay (in us) for a variety of eccentric
        anomalies (in radians) given the R and S parameters.
        """
        canoms = np.cos(ecc_anoms)
        sanoms = np.sin(ecc_anoms)
        ecc = self.par.E
        omega = self.par.OM * DEGTORAD
        cw = np.cos(omega)
        sw = np.sin(omega)
        delay = -2.0e6*R*np.log(1.0 - ecc*canoms -
                                 S*(sw*(canoms-ecc) +
                                    np.sqrt((1.0 - ecc*ecc)) * cw * sanoms))
        return delay

    def shapiro_measurable(self, R, S, mean_anoms):
        """
        Return the predicted _measurable_ Shapiro delay (in us) for a variety
        of mean anomalies (in radians) given the R and S parameters. This is
        eqn 28 in Freire & Wex 2010 and is only valid in the low eccentricity
        limit.
        """
        Phi = mean_anoms + self.par.OM * DEGTORAD
        cbar = np.sqrt(1.0 - S**2.0)
        zeta = S / (1.0 + cbar)
        h3 = R * zeta**3.0
        sPhi = np.sin(Phi)
        delay = -2.0e6 * h3 * (
            np.log(1.0 + zeta*zeta - 2.0 * zeta * sPhi) / zeta**3.0 +
            2.0 * sPhi / zeta**2.0 -
            np.cos(2.0 * Phi) / zeta)
        return delay

    def true_anomaly(self, ea, radians=False):
        """
        Return the true anomaly given the eccentric anomaly.

        radians : bool
            True will return radians, False will return orbital cycle.
        """
        ta = 2*np.arctan2(np.sqrt(1+self.par.ECC)*np.sin(ea*0.5), np.sqrt(1-self.par.ECC)*np.cos(ea*0.5))
        ta = np.mod(ta, TWOPI)
        if not radians:
            ta /= TWOPI
        return ta


##### -----------------------------------------------------------------
##### The command line processing
##### -----------------------------------------------------------------

if __name__=='__main__':
    
    ## The command-line argument parser
    parser = argparse.ArgumentParser(description='Calculate the next eclipse time (barycentric MJD) from a certain epoch (barycentric MJD).', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parfile', action='store', help='Input parfile to calculate the ephemerides.')
    parser.add_argument('epoch', action='store', type=float, help='Input epoch (barycentric MJD) from which to calculate the next eclipse time.')
    args = parser.parse_args()

    ## Parsing the inputs
    parfilenm = args.parfile
    epoch = args.epoch

    ## Creating the binary_psr object and evaluating the next eclipse time
    bpsr = binary_psr(parfilenm, leap=True)
    next_eclipse = bpsr.next_eclipse(epoch)
    
    ## Output the results
    print('The next eclipse time is: {0}'.format(next_eclipse[0]))



