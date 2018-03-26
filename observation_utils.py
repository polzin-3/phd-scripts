import numpy as np
import matplotlib.pyplot as plt
import pycurl
import base64
import re
import os
from StringIO import StringIO
from bs4 import BeautifulSoup
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.io import fits
import my_binary_psr2

###########################################
# Contents:
#	altaz_calc
#	mjd_to_orbph
#       utc_to_bary
#       eclipse_visibility_plotter
#       TASC_search
#	LOFAR_obs_parser
###########################################

def altaz_calc(RADEC, TIME, latitude=52.91, longitude=6.87, save=False):
    """
    Calculate the altitude and azimuth of beam pointing for a given target
    pulsar at a given date and time.
    RADEC : string
           "00h00m00s +00d00m00s"
    TIME : array
           Array of MJD values to calculate alt, az.
    latitude : float, optional
           Telescope latitude in degrees. Default LOFAR.
    longitude : float, optional
           Telescope longitude in degrees. Default LOFAR.
    save : bool, optional
           For True, alt and az arrays will be saved as .txt files.
    """
    PSR = SkyCoord(RADEC)
    telescope = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)
    TIME = np.asarray((TIME))
    altitude = np.empty(TIME.size)
    azimuth = np.empty(TIME.size)
    for i in range(TIME.size):
        time = Time(TIME[i],format='mjd')
        PSRaltaz = PSR.transform_to(AltAz(obstime=time, location=telescope))
        altitude[i] = PSRaltaz.alt.value
        azimuth[i] = PSRaltaz.az.value
    if save==True:
        np.savetxt('altitude.txt', altitude)
        np.savetxt('azimuth.txt', azimuth)
    else:
        return altitude, azimuth

###########################################################################

def mjd_to_orbph(par, mjd, savefile=False, outfile='orbph.txt'):
    """
    Returns array of orbital phases at given MJDs using Rene Breton's 
    my_binary_psr script.
    par : string
         Path to ephemeris (.par) file for given pulsar.
    mjd : array
         Array of MJD values to calculate orbital phase.
    savefile : bool, optional
         If True saves orbph array to .txt file
    outfile : string, optional
         Filename of output .txt file containing orbital phase array.
    """
    mjd = np.asarray((mjd))
    bpsr = my_binary_psr2.binary_psr(par, leap=True)
    orbph = np.empty(mjd.size)
    for i, mjd_i in enumerate(mjd):
        orbph[i] = bpsr.calc_orbph(mjd_i)
    if savefile==True:
        np.savetxt(outfile, orbph)
    else:
        return orbph

###########################################################################

def utc_to_bary(t_utc, psr_coords, dish_lat=52.91, dish_long=6.87):
    """
    Adapted from function by Mark Kennedy.
    Convert UTC MJD pulse arrival times to barycentric.
    t_utc : float or astropy Time object
           UTC MJD(s). If floats are given, then these are converted to astropy.
    psr_coords : string
           "00h00m00s +00d00m00s"
    dish_lat : float, optional
          Latitude (deg) of telescope. Default = LOFAR Core
    dish_long : float, optional
          Longitude (deg) of telescope. Default = LOFAR Core
    """
    psr_coords = SkyCoord(psr_coords)
    telescope = EarthLocation(lat=dish_lat*u.deg, lon=dish_long*u.deg)
    if type(t_utc) == float or type(t_utc) == np.ndarray:
        t_utc = Time(t_utc, format='mjd')
    ltt_bary = t_utc.light_travel_time(psr_coords, location=telescope)
    time_bary = t_utc.tdb + ltt_bary
    return time_bary.mjd

###########################################################################

def eclipse_visibility_plotter(radec, mjd_start, mjd_end, time_step=2,
                               obs_length=None, parfile=None,
                               eclipse_phase_start=None,
                               eclipse_phase_end=None, dish_lat=52.91,
                               dish_long=6.87, line_col='b'):
    """
    Generate plot of source elevation vs time, with eclipsed orbital phases
    highlighted.
    radec : str
          '00h00m00s +/-00d00m00s' RA and DEC of target source.
    mjd_start : float
          UTC MJD for beginning of plot
    mjd_end : float
          UTC MJD for end of plot
    time_step : float, optional
          Temporal resolution of plot in minutes. Default = 2.0 minutes.
    obs_length : float, optional
          Duration of observation in hours, assuming centred around meridian.
    parfile : str, optional
          Path to ephemeris file e.g. 'file.par'. Required for marking eclipse.
    eclipse_phase_start : float, optional
          Orbital phase corresponding to beginning of eclipse
    eclipse_phase_end : float, optional
          Orbital phase corresponding to end of eclipse
    dish_lat : float, optional
          Latitude (deg) of telescope. Default = LOFAR Core
    dish_long : float, optional
          Longitude (deg) of telescope. Default = LOFAR Core
    line_col : string
          Colour to plot line. Default = 'b' (blue).
    """
    # Define MJD array with resolution of 2 minute
    steps = time_step / 1440.
    mjd_arr = np.arange(mjd_start, mjd_end, steps)
    alt, az = altaz_calc(radec, mjd_arr, latitude=dish_lat, longitude=dish_long)
    try:
        # Find minimum observing altitude assuming observation of given length
        min_alt = observing_threshold(alt, obs_length, steps)
    except:
        print 'No observation length specified'
        min_alt = 91.
    #fig, ax = plt.subplots()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Elevation (degrees)')
    ax.set_xlabel('Hours from start')
    ax.hlines(0, 0, (mjd_end-mjd_start) * 24., colors='.5', linestyles='dashed')
    hours_plot = (mjd_arr - mjd_arr.min()) * 24.
    if eclipse_phase_start is None or eclipse_phase_end is None:
        print 'No eclipse range specified by user'
        for i in range(hours_plot.size - 1):
            if alt[i] >= min_alt:
                lw=3
            else:
                lw=1
            ax.plot([hours_plot[i], hours_plot[i+1]], [alt[i], alt[i+1]],
                    c=line_col, lw=lw)
    else:
        mjd_arr_barycent = utc_to_bary(mjd_arr, radec, dish_lat, dish_long)
        orbph_arr = mjd_to_orbph(parfile, mjd_arr_barycent, savefile=False)
        for i in range(hours_plot.size - 1):
            if np.logical_and(orbph_arr[i] > eclipse_phase_start,
                              orbph_arr[i] < eclipse_phase_end):
                col='r'
            else:
                col=line_col
            if alt[i] >= min_alt:
                lw=3
            else:
                lw=1
            ax.plot([hours_plot[i], hours_plot[i+1]], [alt[i], alt[i+1]], c=col,
                    lw=lw)
    plt.show()
    return

def observing_threshold(alt, obs_length, steps):
    """ Calculates minimum altitude of source for observing, assuming that you
    wish to observe centred around the meridian """
    if alt.size*steps < 1:
        print "Warning: Plotting less than 1 day so may not see max altitude \
                  of source. Not showing planned observing time on plot."
        return 91.
    obs_length_days = obs_length / 24.
    obs_length_steps = np.int(np.round(obs_length_days / steps))
    alt_argmax = np.argmax(alt)
    if alt_argmax >= obs_length_steps/2:
        alt_thresh = alt[alt_argmax - obs_length_steps/2]
    else:
        alt_thresh = alt[alt_argmax + obs_length_steps/2]
    return alt_thresh

###########################################################################

def TASC_search(datafile, parfile, range=0.1, steps=5e-4):
    """
    Step through intervals of the Time of ascending node (TASC) parameter to
    refold an observation that has lost orbital timing coherence. Searches for
    the highest S/N profile. NOTE: may need PB=0 in original parfile.
    datafile : str
             Name of file containing data to fold.
    parfile : str
             Name of original ephemeris file used to fold data.
    range : float, optional
             Half fraction of orbital period to search
             (TASC - PB*range : TASC + PB*range). Default = 10%
    steps : float, optional
             Search intervals in units of fractions of orbit. Default = 0.05%
    """
    # Read in original .par file
    with open(parfile,"r+") as f:
        original = f.read()

    # Extract values
    try:
        PB = re.findall('PB\s+0.\d+', original)[0]
        PB = float(re.findall('0.\d+', PB)[0])
    except:
        raise NameError('Could not find	PB in parfile')
    if 'TASC' in original:
        TASC = re.findall('TASC\s+\d+.\d+', original)[0]
        TASC_str = re.findall('\d+.\d+', TASC)[0]
        TASC = float(TASC_str)
    elif 'T0' in original:
        TASC = re.findall('T0\s+\d+.\d+', original)[0]
        TASC_str = re.findall('\d+.\d+', TASC)[0]
       	TASC = float(TASC_str)
    else:
        raise NameError('Could not find TASC or T0 in parfile')

    steps *= PB  # Convert steps to fraction of PB

    TASC_start = TASC - PB*range
    TASC_end = TASC + PB*range
    TASC_arr = np.arange(TASC_start, TASC_end, steps)
    signal_arr = []

    # Make copy of data to temporary file to work with
    os.system('cp ' + datafile + ' data_TASC_search.ar')

    for i, tasc in enumerate(TASC_arr):
        # Modify TASC parameter in .par file
        new = original.replace(TASC_str, '%.9f' % tasc)
        temporary_file = parfile + '.tmp'
        with open(temporary_file, "w+") as f:
            f.write(new)

        # Fold with new ephemeris
        os.system('pam -m -E ' + temporary_file + ' data_TASC_search.ar')
        # Convert to PSRFITS for python
        os.system('psrconv -o PSRFITS data_TASC_search.ar')
        # Find max signal
        hdu = fits.open('data_TASC_search.rf')
        data = hdu[4].data['DATA']
        signal = np.nanmean(np.nanmean(data[:, 0], axis=0), axis=0).max()
        signal_arr.append(signal)
        # Make counter to output completion every 10%
        pc_complete = 100. * i / TASC_arr.size
        if pc_complete % 10 == 0.0:
            print str(int(pc_complete))+'%'

    np.savetxt('signal_arr.txt', signal_arr)
    np.savetxt('TASC_arr.txt', TASC_arr)
    print TASC_arr[np.array(signal_arr).argmax()]

###########################################################################

def LOFAR_obs_parser(psr, parfile=None, eclipse_only=False):
    """
    Find LOFAR LTA observations of given pulsar from lofarpwg website.
    Returns obsID, MJD, duration and orbital phases (if parfile input).
    psr : string
         Pulsar name to search observations e.g. 'J1810+1744'.
    parfile : string, optional
         Path to ephemeris (.par) file to calculate orbital phases.
    eclipse_only : bool, optional
         True to return observations in phase range 0.2 - 0.35 only
    """
    # Set orbital phase range
    ph_thresh_1 = 0.2
    ph_thresh_2 = 0.35
    if eclipse_only is True and parfile is None:
        print 'Provide parfile to restrict orbital phase range.'

    # Retrieve data from LTA. "username:pwd"
    headers = { 'Authorization' : 'Basic %s' % base64.b64encode("pwg:pulpme") }
    buffer = StringIO()
    c = pycurl.Curl()
    c.setopt(c.HTTPHEADER, ["%s: %s" % t for t in headers.items()]) # For username/pwd
    c.setopt(c.URL, 'www.astron.nl/lofarpwg/megapulsars-hba-time.html')
    c.setopt(c.POST, 1)
    c.setopt(c.WRITEDATA, buffer)
    #c.setopt(c.VERBOSE,True)
    c.perform()
    c.close()
    
    # Extract data from buffer - should be a string containing the url metadata
    body = buffer.getvalue()
    buffer.close()

#### Try re.findall('\d{2}-\d{2}', 'string'), or use re to get e.g. obsID

    soup = BeautifulSoup(body)
    body_text = soup.text
    body_text_splits = body_text.split('\n\n\n')
    # Info for each observation split in half so need to recombine
    body_text_splits_obs = []
    for i in np.arange(5, len(body_text_splits)-1, 2):
        combine = body_text_splits[i] + body_text_splits[i+1]
        body_text_splits_obs.append(combine)
    psr_obs = []
    # Extract only observations of input pulsar
    for entry in body_text_splits_obs:
        if psr in entry:
            psr_obs.append(entry)
    print 'Total observations of '+psr+' =',len(psr_obs)
    # Extract useful info from each observation
    for obs in psr_obs:
        obs_parts = obs.split('\n')
        obsID = obs_parts[1]
        start_date = obs_parts[4]
        #start_time = obs_parts[16][obs_parts[16].index('Time: ')+6 : \
        #                 obs_parts[16].index('Clock:')]
        start_time = re.findall('\d{2}:\d{2}:\d{2}', obs)[0]
        duration = obs_parts[6]
        if duration[-1] == 'm':   # if units == minutes
            duration_days = float(duration[:-1]) / (60*24)
        else:    # assume units == hours
            duration_days = float(duration[:-1]) / 24
        t_start = Time(start_date+'T'+start_time, format='isot', scale='utc')
        start_mjd = t_start.mjd
        end_mjd = start_mjd + duration_days

        if parfile is not None:
            orbphases = observation_orbph.mjd_to_orbph(
                            parfile, np.asarray((start_mjd, end_mjd)),
                            savefile=False)
            orbph_start = orbphases[0].copy()
            orbph_end = orbphases[1].copy()
            if eclipse_only is True:
                overlap = test_overlap(ph_thresh_1, ph_thresh_2, orbph_start,
                                       orbph_end)
                if not overlap:
                    continue
            if orbph_end < orbph_start:
                orbph_end += 1
            print obsID, start_mjd, duration
            print '    Orb. phase range =', "{0:.2f}".format(orbph_start), '-', "{0:.2f}".format(orbph_end)
        else:
       	    print obsID, start_mjd, duration
            print '    Orbital phase output requires input parfile.'

def test_overlap(fixed_ph1,fixed_ph2,test_ph1,test_ph2):
    """
    For use with LOFAR_obs_parser()
    Tests for overlap of orbital phase ranges
    NOT ROBUST TO OBSERVATIONS SPANNING > ONE ORBIT
    """
    if test_ph1<test_ph2:
        if fixed_ph1<test_ph2 and test_ph1<fixed_ph2:
            return True
        else:
            return False
    else:
        if fixed_ph1<1. and test_ph1<fixed_ph2:
       	    return True
        elif fixed_ph1<test_ph2 and 0.<fixed_ph2:
            return True
       	else:
       	    return False
