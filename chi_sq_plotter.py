import numpy as np
import matplotlib.pyplot as plt

def bestparams(chi_arr, dmmin=0., dmint=1.):
    Nsubint, Ndm, Ntscat = chi_arr.shape[0], chi_arr.shape[1], chi_arr.shape[2]
    best_dm = np.empty(Nsubint)
    best_scat = np.empty(Nsubint)
    dm_1sig_low = np.empty(Nsubint)
    dm_1sig_high = np.empty(Nsubint)
    tscat_1sig_low = np.empty(Nsubint)
    tscat_1sig_high = np.empty(Nsubint)
    tmp_arr = chi_arr.copy()
    tscats = np.append(np.arange(1.,50.,2.),np.arange(50.,200.,50.)) / 256.
    tscats = np.append(0,tscats)
    onesig = 2.3

    for i in np.arange(Nsubint):
        tmp_arr[i] -= np.min(tmp_arr[i])
        dm_time = tmp_arr[i].min(axis=1)
        tscat_time = tmp_arr[i].min(axis=0)
        best_dm[i] = dm_time.argmin()
        best_scat[i] = tscat_time.argmin()
        for jj in range(np.int(best_dm[i]) + 1):
            if dm_time[jj-1] > onesig and dm_time[jj] < onesig:
                diff = (onesig - dm_time[jj]) / (dm_time[jj-1] - dm_time[jj])
                dm_1sig_low[i] = jj - diff
        for jj in range(np.int(best_dm[i]) + 1, Ndm):
            if dm_time[jj-1] < onesig and dm_time[jj] > onesig:
                diff = (dm_time[jj] - onesig) / (dm_time[jj] - dm_time[jj-1])
                dm_1sig_high[i] = jj - diff
        for jj in range(np.int(best_scat[i])):
            if tscat_time[jj] > onesig and tscat_time[jj+1] < onesig:
                diff = ((onesig - tscat_time[jj+1]) /
                        (tscat_time[jj] - tscat_time[jj+1]))
                tscat_1sig_low[i] = jj + 1 - diff
        for jj in range(np.int(best_scat[i]) + 1, Ntscat):
            if tscat_time[jj-1] < onesig and tscat_time[jj] > onesig:
                diff = ((tscat_time[jj] - onesig) /
                        (tscat_time[jj] - tscat_time[jj-1]))
                tscat_1sig_high[i] = jj - diff
    best_dm = dmmin + best_dm * dmint
    dm_1sig_low = dmmin + dm_1sig_low * dmint
    dm_1sig_high = dmmin + dm_1sig_high * dmint

    # To be conservative, assign to error the largest of 1sig_high and 1sig_low
    
    dm_err = _uncertainty_assign(best_dm, dm_1sig_high, dm_1sig_low)
    tscat_err = _uncertainty_assign(best_scat, tscat_1sig_high, tscat_1sig_low)
    return best_dm, best_scat, dm_err, tscat_err, dm_1sig_low, dm_1sig_high, \
           tscat_1sig_low, tscat_1sig_high

def params_vs_time(
        chi_arr, dmmin=0., dmint=1., currdm=0., snr_arr=None, sigma_arr=None,
        subint_start=0, orbph=None, fignum=1, bins=256., telescope=None):
    """
    Outputs contour plots of DM and tau vs subint with 1, 2 and 3 sigma
    contours at 68.3%, 95.4% and 99.73% uncertainty levels, based on chi-sq
    values of fits to data using different combinations of DM and tau.
    
    Parameters
    ----------
    chi_arr : numpy.ndarray
        Array of chi-square values (see dedisperse_eclipse.py) with shape
        ((N_subints, N_dm, N_tau))
    dmmin : float, optional
        Minimum DM used in search in pc cm^-3
    dmint : float, optional
        Size of DM steps in pc cm^-3
    currdm : float, optional
        DM that data is currently dedispersed at in pc cm^-3
    snr_arr : numpy.ndarray, optional
        Array of template scale-factors (see dedisperse_eclipse.py) with
        shape ((N_subints, N_dm, N_tau)). If provided then the values
        corresponding to the lowest chi-sq fit per sub-integration will be
        plotted along with the DM and tau contour plots
    sigma_arr : numpy.ndarray, optional
        Array of template scale-factors uncertainties
        (see dedisperse_eclipse.py) with shape ((N_subints, N_dm, N_tau)).
        If provided then the values corresponding to the lowest chi-sq fit
        per sub-integration will be plotted as errorbars for the
        scale-factor plot
    subint_start : int, optional
        First x-axis tick label, if the fits were made to only a subset of
        the data
    orbph : numpy.array or list, optional
        Orbital phases for each sub-integration of data to use as x-axis. If
        not provided then sub-integration number is used. If subint-start and
        orbph are provided then orbph will be subset as
        orbph[subint_start:subint_start+Nsubint]
    fignum : int, optional
        Matplotlib figure number, if different from 1
    bins : int, optional
        Number of pulse phase bins in data that was used in fits, if different
        from 256
    telescope : string
        Used to scale tau to centre frequency of observation.
        options : 'LOFAR' (148.9 MHz), 'WSRT' (345.6 MHz), 'GMRT' (650 MHz),
                  'Parkes' (1369 MHz), 'Lovell' (1532 MHz)
    """
    Nsubint, Ndm, Ntscat = chi_arr.shape
    tmp_arr = chi_arr.copy()
    dm_time = np.empty((Ndm, Nsubint))
    tscat_time = np.empty((Ntscat, Nsubint))
    dms = np.arange(dmmin, dmmin + dmint * (Ndm - .5), dmint)
    ##tscats=np.append(np.arange(0.1,1.,0.2),np.arange(1.,70.,5.)) ## Karastergiou scattering
    # Old tscat range
    ##tscats=np.append(np.arange(1.,50.,2.),np.arange(50.,200.,50.)) / np.float(bins)
    # New tscat range
    tscats = np.append(np.arange(1., 50., 2.),
                       np.arange(50., np.float(bins), 50.)) / np.float(bins)
    tscats = np.append(0, tscats)
    # ratio of band centre to band top
    tel_dict = {'LOFAR':(148.9 / 187.9), 'WSRT':(345.6 / 381.2),
                'GMRT':(650. / 750.), 'Parkes':(1369. / 1496.75),
                'Lovell':(1532.0 / 1731.9)}
    try:
        ratio = tel_dict[telescope]
    except:
        raise ValueError("Invalid telescope name provided")
    # tau at centre of telescope band
    tau_cf = tscats / ratio**4.
    # sub-integration indices
    subints = np.arange(subint_start, subint_start + Nsubint)
    # Delta chi-square values for 2 variables:
    onesig = 2.3     # was 1., 4., 9. originally?
    twosig = 6.17
    threesig = 11.8
    for i in range(Nsubint):
        tmp_arr[i] -= np.min(tmp_arr[i])
        dm_time[:, i] = tmp_arr[i].min(axis=1)
        tscat_time[:, i] = tmp_arr[i].min(axis=0)
    if orbph is not None:
        x_ax = orbph[subint_start : subint_start+Nsubint]
        x_lab = 'Orbital Phase'
    else:
        x_ax = subints
        x_lab = 'Sub-integration'
    f = plt.figure(fignum)
    f.subplots_adjust(hspace=0)
    if snr_arr is None:
        ax = f.add_subplot(211, ylabel=r'$\Delta$DM (pc cm$^{-3}$)',
                           xticklabels=[])
        ax.contourf(x_ax, dms-currdm, dm_time,
                    levels=[0, onesig, twosig, threesig], cmap='copper')
        ax.contour(x_ax, dms-currdm, dm_time,
                    levels=[onesig, twosig, threesig], colors='black')
        ax.tick_params(direction='out')
        ax2 = f.add_subplot(212, xlabel=x_lab, ylabel=r'$\Delta\tau$ (P)')
        ax2.contourf(x_ax, tau_cf, tscat_time,
                     levels=[0, onesig, twosig, threesig], cmap='copper')
        ax2.contour(x_ax, tau_cf, tscat_time,
                    levels=[onesig, twosig, threesig], colors='black')
        ax2.tick_params(direction='out')
        plt.show()
    else:
        # Find template scale-factor corresponding to lowest chi-sq fits
        snr_fit = np.empty(Nsubint)
        sigma = np.empty(Nsubint)
        for i in range(Nsubint):
            coord = np.unravel_index(chi_arr[i].argmin(), chi_arr[i].shape)
            snr_fit[i] = snr_arr[i][coord]
            if sigma_arr is not None:
                sigma[i] = sigma_arr[i][coord]
        ax = f.add_subplot(311, ylabel='Best fit amplitude', xticklabels=[])
        if sigma_arr is not None:
            ax.errorbar(x_ax, snr_fit, sigma)
        else:
            ax.plot(x_ax, snr_fit)
        ax = f.add_subplot(312, ylabel=r'$\Delta$DM (pc cm$^{-3}$)',
                            xticklabels=[])
        ax.contourf(x_ax, dms-currdm, dm_time,
                    levels=[0, onesig, twosig, threesig], cmap='copper')
        ax.contour(x_ax, dms-currdm, dm_time,
                    levels=[onesig, twosig, threesig], colors='black')
        ax.tick_params(direction='out')
        ax2 = f.add_subplot(313, xlabel=x_lab, ylabel=r'$\Delta\tau$ (P)')
        ax2.contourf(x_ax, tau_cf, tscat_time,
                     levels=[0, onesig, twosig, threesig], cmap='copper')
        ax2.contour(x_ax, tau_cf, tscat_time,
                    levels=[onesig, twosig, threesig], colors='black')
        ax2.tick_params(direction='out')
        plt.show()

def best_fit_amplitude(chi_arr, snr_arr, subint_start=0, fignum=1):
    """
    Plots the amplitude of the template that produces the best fit to the data
    i.e. that which results in lowest chi-square.
    
    Parameters
    ----------
    chi_arr : numpy.ndarray
        Array of chi-square values (see dedisperse_eclipse.py) with shape
        ((N_subints, N_dm, N_tau))
    snr_arr : numpy.ndarray, optional
        Array of template scale-factors (see dedisperse_eclipse.py) with
        shape ((N_subints, N_dm, N_tau)). If provided then the values
        corresponding to the lowest chi-sq fit per sub-integration will be
        plotted along with the DM and tau contour plots
    subint_start : int, optional
        First x-axis tick label, if the fits were made to only a subset of
        the data
    fignum : int, optional
        Matplotlib figure number, if different from 1
        
    """
    Nsubint = chi_arr.shape[0]
    subints = np.arange(subint_start, subint_start+Nsubint)
    snr_fit = np.empty(Nsubint)
    for i in range(Nsubint):
        coord = np.unravel_index(chi_arr[i].argmin(), chi_arr[i].shape)
        snr_fit[i] = snr_arr[i][coord]
    f = plt.figure(fignum)
    ax = f.add_subplot(111, title = 'Amplitude of minimum chi-square fit',
                       xlabel='Sub-integration', ylabel='Best fit amplitude')
    ax.plot(subints, snr_fit)
    plt.show()

def convert_tscat_steps(tscats,arr):
    int_arr=arr.astype(int)
    int_arr_next=int_arr+1
    int_arr_next[int_arr_next>38.]=38
    new=tscats[int_arr]+(tscats[int_arr_next]-tscats[int_arr])*(arr-int_arr)
    return new

def _uncertainty_assign(vals, err_high, err_low):
    diff_low = np.abs(vals - err_low)
    diff_high = np.abs(vals - err_high)
    new_err = np.empty_like(vals)
    for i in range(len(vals)):
        if diff_low[i] > diff_high[i]:
            new_err[i] = diff_low[i]
        else:
            new_err[i] = diff_high[i]
    return new_err