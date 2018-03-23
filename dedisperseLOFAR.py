import numpy as np
from scipy import optimize
import time
import psr_utils
import scatter_fft
import dispersion_measure

def dm_2d(
        data, template, currdm=0, dmmin=0, dmmax=0, dmint=0, P=0.0016627,
        subfreqs=None, EMsearch=False, free_base=True, free_subbands=False,
        fscr=False, correct_data=False):
    """
    Find best DM and scattering timescale, tau, for each sub-interval in data.
    Takes input template and disperses + scatters it over the specified range of
    values, and performs least-squares fit of each template to the sub-intervals
    of the data. Input template must be phase-aligned and dedispered to same as
    the out-of-eclipse input data sub-intervals.
    Function generates, and saves to files, arrays of best-fit chi-squares,
    template fit amplitudes, and uncertainties on amplitudes for each fit.
    data : numpy.array
          Data from which DM and tau are measured
          data.shape = (intervals, channels, bins)
    template : numpy.array
          Template profile to fit to the data. template.shape = (channels, bins)
    currdm : float, optional
          DM that data is currently dedispersed at in pc cm^-3.
    dmmin : float, optional
          Minimum DM to search in pc cm^-3.
    dmmax : float, optional
          Maximum DM to search in pc cm^-3.
    dmint : float, optional
          Size of DM in pc cm^-3.
    P : float, optional
          Spin period of pulsar in seconds.
    subfreqs : numpy.array, optional
          Array of channel centre frequencies (MHz) if different from LOFAR
          HBA (400 channels).
    EMsearch : bool, optional
          If True then includes additional EM term in dispersion, and searches
          over range 0 - 1 x 10^6 pc cm^-6.
    free_base : bool, optional
          If True then baseline of template is free parameter in the fitting.
    free_subbands : bool, optional
          If True then channels grouped into 4 'sub-bands', with a separate
          amplitude free parameter for each group of channels. This goes some
          way towards accounting for a variable frequency spectrum.
    fscr : bool, optional
          If True then the channels in data and template are scrunched to 4
          sub-bands (scrunch applied to template array after dispersion and
          scattering to avoid smearing).
    correct_data : bool, optional
          If True then the input data array will be returned with the best
          DM's applied; note - this does not correct for scattering!
    """
    start_time = time.time()
    intervals, channels, bins = data.shape[0], data.shape[1], data.shape[2]
    binspersec = bins / P
    steps = int(round((dmmax-dmmin) / dmint))
    tscat = np.append(np.arange(1., 50., 2.), np.arange(50., 200., 50.)) / bins
    #tscat=np.append(np.arange(0.1,1.,0.2),np.arange(1.,70.,5.))  # Karastergiou scattering
    EMsteps = np.append(np.arange(0., 1.e5, 2.e4), np.arange(1.e5, 1.e6, 5.e4))  # Need this even if not EM searching, originally max = 2.e6
    tmp_fp = np.empty((channels, bins))
    scat_template = np.empty(np.shape(template))
    rfimask = ~np.isnan(np.nanmean(template, axis=1))
    if subfreqs is None:
        subfreqs = np.arange(109.9609376, 187.9882813, 0.1953125)
        #subfreqs = subfreqs[280:]  # Sub-band filter
        #rfimask[66:75] = 0  # Needed for sub-band filter [280:]
        rfimask[:16] = rfimask[138] = rfimask[306] = 0
        rfimask[346:355] = 0
    else:
        ######## German station ########
        ##rfimask[95:98] = rfimask[219:227] = rfimask[304:314] = 0
        ##rfimask[331:341] = 0
        ############# WSRT #############
        ##rfimask[52/2 : 74/2] = rfimask[(125 - 1)/2 : 136/2] = 0
        ##rfimask[(178)/2 : (201 - 1)/2] = rfimask[(247 - 1)/2 : (264)/2] = 0
        ##rfimask[(310)/2 : (328)/2] = 0
        ##rfimask[(376)/2 : (394)/2] = rfimask[(441 - 1)/2 : (463 - 1)/2] = 0
        ##subfreqs = subfreqs[354/2 :]
        ############ Lovell ############
        rfimask[0] = 0
    if subfreqs[1] - subfreqs[0] < 0.:
        subfreqs = subfreqs[::-1].copy()
        rfimask = rfimask[::-1].copy()
        data = data[:, ::-1, :].copy()
        template = template[::-1, :].copy()
    max_dm_chan = dm_smear(subfreqs, P, dmmax-currdm)
    if correct_data:
        data_new = np.empty(np.shape(data))

    if not EMsearch:
        snr_arr = np.zeros((intervals, steps, tscat.size+1))
        chi_arr = np.zeros((intervals, steps, tscat.size+1))
        sigma_snr_arr = np.zeros((intervals, steps, tscat.size+1))
        fit_points_arr = np.zeros((intervals, steps, tscat.size+1))
        start_template_time = time.time()
        print('Pre-template generation time =',start_template_time-start_time)
        template_arr = create_templates(
            template, steps, EMsteps, tscat,channels, bins, currdm, dmint,
            dmmin, subfreqs, binspersec, tmp_fp, scat_template)
        end_template_time = time.time()
        print('Time taken to generate templates =',end_template_time-start_template_time)
        for ii in range(intervals):
            start_int_time = time.time()
            if ii%10 == 0:
                print ii
            flux_err = sigma_flux(data[ii], channels, bins)
            rfiblock = mask_rfi(data[ii], rfimask)
            for i in range(int(steps)):
                dm = dmint*i + dmmin
                rfi_dm_mask = mask_dms(dm-currdm, rfiblock, max_dm_chan)
                if free_subbands is False:
                    snr_arr[ii,i,0], chi_arr[ii,i,0], sigma_snr_arr[ii,i,0], \
                        fit_points_arr[ii,i,0] = template_fit1d(
                            template_arr[i,0][rfi_dm_mask],
                            data[ii][rfi_dm_mask], flux_err[rfi_dm_mask],
                            free_base=free_base)
                else:
                    if fscr:
                        data_input, sigma_input = scrunch_freq(
                                                    data[ii][rfi_dm_mask],
                                                    flux_err[rfi_dm_mask])
                    else:
                        data_input = data[ii][rfi_dm_mask]
                        sigma_input = flux_err[rfi_dm_mask]
                    snr_arr[ii,i,0], chi_arr[ii,i,0], sigma_snr_arr[ii,i,0], \
                        fit_points_arr[ii,i,0] = template_fit_subbands(
                            rfi_dm_mask, template_arr[i,0][rfi_dm_mask],
                            data_input, sigma_input, nchan=channels, fscr=fscr)
                for yy in np.arange(tscat.size):
                    if free_subbands is False:
                        snr_arr[ii,i,yy+1], chi_arr[ii,i,yy+1], \
                            sigma_snr_arr[ii,i,yy+1], \
                            fit_points_arr[ii,i,yy+1] = template_fit1d(
                                template_arr[i,yy+1][rfi_dm_mask],
                                data[ii][rfi_dm_mask], flux_err[rfi_dm_mask],
                                free_base=free_base)
                    else:
                        snr_arr[ii,i,yy+1], chi_arr[ii,i,yy+1], \
                            sigma_snr_arr[ii,i,yy+1], \
                            fit_points_arr[ii,i,yy+1] = template_fit_subbands(
                                rfi_dm_mask, template_arr[i,yy+1][rfi_dm_mask],
                                data_input, sigma_input, nchan=channels,
                                fscr=fscr)
            if correct_data:
                min_chi_dm = np.unravel_index(chi_arr[ii].argmin(),
                                              chi_arr[ii].shape)[0]
                dm_estimate = dmmin + min_chi_dm*dmint
                data_new[ii] = dm_rotate(data[ii], -dm_estimate, -currdm, 0.,
                                         subfreqs, binspersec, channels, tmp_fp)
            end_int_time = time.time()
            print('Interval duration =', end_int_time-start_int_time)
        save_arrays(snr_arr, chi_arr, sigma_snr_arr, fit_points_arr)
        end_save_time = time.time()
        print('Total time =',end_save_time-start_time)
        if correct_data:
            return data_new
    else:
        snr_arr = np.zeros((intervals, steps, EMsteps.size, tscat.size+1))
        chi_arr = np.zeros((intervals, steps, EMsteps.size, tscat.size+1))
        sigma_snr_arr = np.zeros((intervals, steps, EMsteps.size,
                                  tscat.size+1))
        fit_points_arr = np.zeros((intervals, steps, EMsteps.size,
                                   tscat.size+1))
        template_arr = create_templates(
            template, steps, EMsteps, tscat, channels, bins, currdm, dmint,
            dmmin, subfreqs, binspersec, tmp_fp, scat_template, EMsearch=True)
        for ii in range(intervals):
            if ii%10 == 0:
                print ii
            flux_err = sigma_flux(data[ii], channels, bins)
            rfiblock = mask_rfi(data[ii], rfimask)
            for i in range(int(steps)):
                dm = dmint*i + dmmin
                rfi_dm_mask = mask_dms(dm-currdm, rfiblock, max_dm_chan)
                for EM in np.arange(EMsteps.size):
                    if not free_subbands:
                        snr_arr[ii,i,EM,0], chi_arr[ii,i,EM,0], \
                            sigma_snr_arr[ii,i,EM,0], \
                            fit_points_arr[ii,i,EM,0] = template_fit1d(
                                template_arr[i,EM,0][rfi_dm_mask],
                                data[ii][rfi_dm_mask], flux_err[rfi_dm_mask],
                                free_base=free_base)
                    else:
                        if fscr:
                            data_input, sigma_input = scrunch_freq(
                                                        data[ii][rfi_dm_mask],
                                                        flux_err[rfi_dm_mask])
                        else:
                            data_input = data[ii][rfi_dm_mask]
                            sigma_input = flux_err[rfi_dm_mask]
                        snr_arr[ii,i,EM,0], chi_arr[ii,i,EM,0], \
                            sigma_snr_arr[ii,i,EM,0], \
                            fit_points_arr[ii,i,EM,0] = template_fit_subbands(
                                rfi_dm_mask, template_arr[i,EM,0][rfi_dm_mask],
                                data_input, sigma_input, nchan=channels,
                                fscr=fscr)
                    for yy in np.arange(tscat.size):
                        if not free_subbands:
                            snr_arr[ii,i,EM,yy+1], chi_arr[ii,i,EM,yy+1], \
                                sigma_snr_arr[ii,i,EM,yy+1], \
                                fit_points_arr[ii,i,EM,yy+1] = template_fit1d(
                                    template_arr[i,EM,yy+1][rfi_dm_mask],
                                    data[ii][rfi_dm_mask],
                                    flux_err[rfi_dm_mask],
                                    free_base=free_base)
                        else:
                            snr_arr[ii,i,EM,yy+1], chi_arr[ii,i,EM,yy+1], \
                                sigma_snr_arr[ii,i,EM,yy+1], \
                                fit_points_arr[ii,i,EM,yy+1] = \
                                    template_fit_subbands(
                                        rfi_dm_mask,
                                        template_arr[i,EM,yy+1][rfi_dm_mask],
                                        data_input, sigma_input, nchan=channels,
                                        fscr=fscr)
        return snr_arr, chi_arr, sigma_snr_arr

def create_templates(
        template, steps, EMsteps, tscat, channels, bins, currdm, dmint, dmmin,
        subfreqs, binspersec, tmp_fp, scat_template, EMsearch=False):
    """
    Using the original dedispersed template, generate a series of new templates
        each dispersed and scattered in steps defined by dmint and tscat.
    Returns array of templates, including the original.
    """
    if not EMsearch:
        template_arr = np.empty((steps, tscat.size+1, channels, bins))
        for i in range(int(steps)):
            dm = dmint*i + dmmin
            template_arr[i,0] = dm_rotate(template, dm, currdm, 0., subfreqs,
                                          binspersec, channels, tmp_fp)
            for yy in np.arange(tscat.size):
                for jj in range(channels):
                    tau = tscat[yy] / (subfreqs[jj]/subfreqs[-1])**4.
                    scat_template[jj] = scatter_fft.scatter_fft(
                        template_arr[i,0,jj], tau=tau)
                    #tau=tscat[yy]/(subfreqs[jj]/subfreqs[-1])**4.  # Karastergiou scat
                    #scat_template[jj]=scatter_fft.scatter_fft(template_arr[i,0,jj],tau=tau,kscat=True)
                template_arr[i,yy+1] = scat_template
    else:
        template_arr = np.empty((steps, EMsteps.size, tscat.size+1, channels,
                                 bins))
        for i in range(int(steps)):
            dm = dmint*i + dmmin
            for EM in np.arange(EMsteps.size):
                template_arr[i,EM,0] = dm_rotate(template, dm, currdm,
                                                 EMsteps[EM], subfreqs,
                                                 binspersec, channels, tmp_fp)
                for yy in np.arange(tscat.size):
                    for jj in range(channels):
                        tau = tscat[yy] / (subfreqs[jj]/subfreqs[-1])**4.
                        scat_template[jj] = scatter_fft.scatter_fft(
                            template_arr[i,EM,0,jj], tau=tau)
                        #tau=tscat[yy]/(subfreqs[jj]/subfreqs[-1])**4.  # Karastergiou scat
                        #scat_template[jj]=scatter_fft.scatter_fft(template_arr[i,EM,0,jj],tau=tau,kscat=True)
                    template_arr[i,EM,yy+1] = scat_template
    return template_arr

def dm_rotate(template, dm, currdm, EM, subfreqs, binspersec, channels, tmp_fp):
    """
    Rotate channels of input template in pulse phase. Delay in each channel
        assumes cold plasma dispersion law (i.e. freq^{-2}) - see psr_utils.py
    """
    subdelays = (psr_utils.delay_from_DM(dm-currdm, subfreqs) 
                 + EM/(4*subfreqs**4))
    #hifreqdelay = subdelays[-1]
    #subdelays = subdelays-hifreqdelay
    delaybins = subdelays * binspersec
    tmp_fp = np.array([psr_utils.fft_rotate(row, -delaybins[num])
                       for num,row in enumerate(template)])
    return tmp_fp

def dm_smear(subfreqs, P, max_dm_search):
    """
    Returns minimum delta DM for which the pulse would smear over full pulse
        period in each individual channel.
    """
    freqBW = subfreqs[1] - subfreqs[0]
    subfreqs_low = subfreqs - 0.5*freqBW
    subfreqs_high = subfreqs + 0.5*freqBW
    max_dm_chan = (2.41e-4)*P / (np.reciprocal(subfreqs_low**2)
                                 -np.reciprocal(subfreqs_high**2))
    return max_dm_chan

def mask_rfi(freqphase, mask_arr):
    """
    Updates input mask to block channels in which |flux| > |mean + 3 sigma|,
        which we asume to be RFI dominated.
    Also blocks channels that are NaN in this subint of data.
    """
    spectrum = np.nanmean(freqphase, axis=1)
    mu = np.nanmean(spectrum)
    norm_spec = spectrum - mu
    diff = np.abs(spectrum - mu)
    threesigma = 3 * np.nanstd(norm_spec)
    tmp_mask = np.isnan(diff)
    diff[tmp_mask] = 0
    mask_three_sig = diff < threesigma
    masknan = ~np.isnan(np.nanmean(freqphase, axis=1))
    new_mask = mask_three_sig * masknan * mask_arr
    return new_mask

def mask_dms(deltaDM, mask, max_dm_chan):
    """
    Updates input mask to block channels in which pulse becomes fully smeared
        at given DM
    """
    mask_dm = max_dm_chan > deltaDM
    new_mask = mask_dm * mask
    return new_mask

def save_arrays(snr_arr, chi_arr, sigma_snr_arr, fit_points_arr):
    """
    Save 3D arrays of best-fit amplitudes and chi-sq's
    This flattens as ((subints, dm_steps, tau_steps)) --> 
        ((subints * dm_steps, tau_steps)).
    """
    with file('snr_arr.txt','w') as outfile:
        for data_slice in snr_arr:
            np.savetxt(outfile,data_slice)
            outfile.write('# New slice\n')
    with file('chi_arr.txt','w') as outfile:
        for data_slice in chi_arr:
            np.savetxt(outfile,data_slice)
            outfile.write('# New slice\n')
    with file('sigma_snr_arr.txt','w') as outfile:
        for data_slice in sigma_snr_arr:
            np.savetxt(outfile,data_slice)
            outfile.write('# New slice\n')
    with file('fit_points_arr.txt','w') as outfile:
        for data_slice in fit_points_arr:
            np.savetxt(outfile,data_slice)
            outfile.write('# New slice\n')

def scrunch_freq(data_in, sigma_in):
    """
    Scrunches the input 2D array to 4 frequency channels
    """
    if data_in.shape[0] == 0:
        return data_in
    nbin = data_in.shape[1]
    data_scr = np.empty((4, nbin))
    sigma_scr = np.empty((4, nbin))
    chanset_size = data_in.shape[0] / 4
    if chanset_size > 4:
        for subset in range(4):
            data_scr[subset] = np.nanmean(
                data_in[subset*chanset_size : (subset+1)*chanset_size],
                axis=0)
            sigma_scr[subset] = np.nanmean(
                sigma_in[subset*chanset_size : (subset+1)*chanset_size],
                axis=0) / np.sqrt(chanset_size)
    else:
        return data_in
    return data_scr, sigma_scr

def sigma_flux(fphase, channels, bins):
    """
    Estimates 1 sigma uncertainty on flux values in each phase bin by
        calculating noise level in each channel using high-pass filter (this
        assumes that pulse width is significantly wider than the noise)
    """
    sigma_arr = np.empty((channels, bins))
    fft = np.fft.rfft(fphase, axis=1)
    fft_mag = abs(fft[:, 1:])**2
    for chan_i in np.arange(channels):
        sigma_arr[chan_i] = (np.nanmean(fft_mag[chan_i, 10:]) / bins)**0.5
    #sigma_arr = (np.nanmean(fft_mag[:, 10:], axis = 1) / bins)**0.5
    #sigma_arr.shape = channels, 1
    return sigma_arr

def template_fit1d(x, y, sigma=1., free_base=False):
    """
    Analytical least-squares template fit for 1 or 2-dimensional input arrays.
    If template and data are 2D, the template is treated as if the baseline
        and amplitude relationships between the channels are locked i.e. finds
        a SINGLE best fit amplitude and baseline for whole template.
    See Numerical Recipe for more information.
    """
    sx = (x * np.reciprocal(sigma**2)).sum()
    ss = np.reciprocal(sigma**2).sum()
    if free_base:
        ######### For fit of amplitude & baseline ###########
        sxoss = sx / ss
        t = (x-sxoss) * np.reciprocal(sigma)
        st2 = (t**2).sum()
        sy = (y * np.reciprocal(sigma**2)).sum()
        amplitude = (t * y * np.reciprocal(sigma)).sum() / st2
        baseline = (sy - sx*amplitude) / ss
        sigma_amp = np.sqrt(1 / st2)
        #sigma_base = np.sqrt((1 + sx*sx/(ss*st2)) / ss)
    else:
        ############ For fit of amplitude only ##############
        sxy = (x * y * np.reciprocal(sigma**2)).sum()
        sxx = (x * x * np.reciprocal(sigma**2)).sum()
        amplitude = sxy / sxx
        baseline = 0
        sigma_amp = np.sqrt(ss) * (sx/sxx)
    chi2 = (((y - baseline - amplitude*x) * np.reciprocal(sigma))**2).sum()
    fit_points = y.size  # Number of data points used in fit
    return amplitude, chi2, sigma_amp, fit_points

def template_fit_subbands(
        rfi_dm_mask, template_i, data_subint, sigma=1., nchan=400, fscr=False):
    """
    Least squares template fit for 2-dimensional input arrays.
    Template is split into 4 frequency sub-bands, each with its own
        free-parameter for amplitude.
    Only a single free parameter is used for the baseline (applied to full
        template).
    If fscr is True then the channels are scrunched into a single profile for
        each sub-band before performing fit -- increases signal-to-noise.
    """
    if data_subint.shape[0] == 0:
        return np.nan, 0., np.nan, 0.
    nchan_band = nchan / 4
    nchan_rem = nchan % 4
    if fscr:
        nbin = template_i.shape[1]
        template_tmp = np.empty((4, nbin))
        chanset_size = template_i.shape[0] / 4
        if chanset_size > 4:
            for subset in range(4):
                template_tmp[subset] = np.nanmean(
                    template_i[subset*chanset_size : (subset+1)*chanset_size],
                    axis=0)
            template_i = template_tmp.copy()
            def fitfunc(x,a,b,c,d,e): \
                return a + (x.T * np.array([b,c,d,e])).T.flatten()
        else:
            fscr = False
    data_subint_flat = data_subint.flatten()
    sigma_flat = sigma.flatten()
    if not fscr:
        def fitfunc(x,a,b,c,d,e): \
            return a + (x.T * np.array([b]*nchan_band + [c]*nchan_band
                   + [d]*nchan_band
                   + [e]*(nchan_band + nchan_rem))[rfi_dm_mask]).T.flatten()
    p0 = [0., 1., 1., 1., 1.]
    p1,p1_cov = optimize.curve_fit(
        fitfunc, template_i, data_subint_flat, p0[:], sigma_flat)
    uncertainties = np.sqrt(np.diag(p1_cov))
    sigma_amp = np.sqrt(np.sum(uncertainties[1:]**2))
    best_temp = fitfunc(template_i, p1[0], p1[1], p1[2], p1[3], p1[4])
    chi2 = (((data_subint_flat - best_temp)*np.reciprocal(sigma_flat))**2).sum()
    #amplitude = np.nanmean(best_temp)
    amplitude = np.sum(p1[1:])
    fit_points = data_subint.size
    return amplitude, chi2, sigma_amp, fit_points

