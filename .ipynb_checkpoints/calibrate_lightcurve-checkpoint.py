import numpy as np

def calibrate_lightcurve(src_flux, src_flux_err, w_mean=False):
    """
    Takes array of flux of multiple sources vs time and normalises primary
    source (src_flux[0,:]) using the other sources in the array.
    src_flux : array
          Array of source fluxes with shape [#src, time]
    src_flux_err : array
          Corresponding array of flux uncertainties
    w_mean : bool
          If True normalises by weighted mean. Default = False
    """
    T = src_flux.shape[1]
    Snum = src_flux.shape[0]
    norm = np.empty(T)
    ext_src = src_flux[1:].copy()
    ext_src_err = src_flux_err[1:].copy()
    for i in np.arange(T):
        mask = ext_src[:,i]<.01 # Mask values of flux where sources not detected
        if w_mean is False:
            norm[i] = np.mean(ext_src[:,i][~mask]) / np.mean(ext_src[~mask])
        else:
            wsum_i = np.reciprocal(ext_src_err[:,i][~mask]**2).sum()
            wflux_i = (np.reciprocal(ext_src_err[:,i][~mask]**2)
                       * ext_src[:,i][~mask]).sum()
            wsum_tot = np.reciprocal(ext_src_err[:,:226][~mask]**2).sum() 
            wflux_tot = (np.reciprocal(ext_src_err[:,:226][~mask]**2)
                          * ext_src[:,:226][~mask]).sum()
            norm[i] = (wflux_i / wsum_i) / (wflux_tot / wsum_tot)
    flux_corr = src_flux[0] / norm
    flux_err_corr = src_flux_err[0] / norm
    return flux_corr, flux_err_corr
