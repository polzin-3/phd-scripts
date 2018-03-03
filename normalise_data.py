import numpy as np
import psr_utils
from astropy.io import fits

#-----------------------------------------------------------
#   Functions to extract data from fits files and normalise
#   by various methods.
#   Contents:
#	Extract_data
#	Combine_data
#	norm_multibeam
#	norm_dm_baseline (inc. baseline_search)
#	norm_fft_baseline
#	fluxcal_fit
#	template_prof
#	dedisperse_subint
#-----------------------------------------------------------

def Extract_data(hdu,combine=False,field=5):
    """
    If combine=False, takes hdu(=fits.open('<fitsfile>')) and extracts the
        raw data, scale factors, offsets and weights and returns these.
    If combine=True, takes hdu and extracts as above, then calls Combine_data
        function to return the observed data with scales, offsets and weights
        applied.
    """
    dataval=hdu[field].data['DATA']
    scl=hdu[field].data['DAT_SCL']
    offs=hdu[field].data['DAT_OFFS']
    wts=hdu[field].data['DAT_WTS']
    if combine is False:
        return dataval,scl,offs,wts
    else:
        fulldata=Combine_data(dataval,scl,offs,wts)
        return fulldata

#-----------------------------------------------------------------------------

def Combine_data(dataval,scl,offs,wts):
    """
    Scales subint data from psrfits file read into python
    Scales as: outval = dataval*scale + offset
    Applies weights from file, replacing bad data with np.nan
    """
    nsub=dataval.shape[0]
    npol=dataval.shape[1]
    nchan=dataval.shape[2]
    nbin=dataval.shape[3]
    # Re-shape arrays to (nsub,npol,nchan,nbin)
    scl.shape=nsub,npol,nchan,1
    offs.shape=nsub,npol,nchan,1
    outval=dataval*scl + offs
    mask=wts<.1 # Apply weights to block RFI etc.
    wts[mask]=np.nan
    wts[~mask]=1
    wts.shape=nsub,1,nchan,1
    data=outval*wts
    #with file('data_calib.txt','w') as outfile:
    #    for data_slice in outval[:,0]:
    #        np.savetxt(outfile,data_slice)
    #        outfile.write('# New slice\n')
    return data

#-----------------------------------------------------------------------------

def norm_multibeam(on_src_data,off_src_data,b_line=False):
    """
    Takes two data sets:
        on_src_data - ((nsubint,npol,nchan,nbin)) centred on pulsar
        off_src_data - ((nsubint,npol,nchan,nbin)) offset from pulsar
    Uses mean flux per subint,chan in off-source beam to normalise that in the
    on-source beam (c.f. optical flat-fielding) in order to remove influence
    from telescope, RFI, sky etc.
    If b_line is True then data will be baselined by subracting unity.
    Returns Stokes I of the corrected on-source data.
    """
    nsub=on_src_data.shape[0]
    nchan=on_src_data.shape[2]
    off_avflux=np.nanmean(off_src_data[:,0],axis=2)
    off_avflux.shape=nsub,nchan,1
    on_corr = on_src_data[:,0] / off_avflux
    if b_line is True:
        on_corr -= 1.
    return on_corr

#-----------------------------------------------------------------------------

def norm_dm_baseline(in_data,off_pulse_size=30,b_line=False,subfreqs=None,GLOW=False):
    """
    Takes in_data ((nsubint,npol,nchan,nbin)) and normalises by off-pulse baseline of
    phase bin width = off_pulse_size.
    Iterates over a range of DMs and pulse phases and fits a mask matching the size
    of the expected off-pulse region in order to find the phases in a given channel
    that provide the lowest mean, which is assumed to be the off-pulse baseline.
    Normalises each subint and channel by the corresponding baseline level.
    If b_line is true then unity is subtracted in order to baseline the array.
    Returns Stokes I of the corrected data.
    NOTE: This is susceptible to scattering over a full pulse phase (would remove
    pulsar flux scattered into baseline)
    """
    nsub=in_data.shape[0]
    nchan=in_data.shape[2]
    baseline_arr=np.empty((nsub,nchan))
    dm_arr=np.empty(nsub)
    LOFAR=False
    if subfreqs is None:
        if GLOW is True:
            subfreqs=np.linspace(118.06640625,189.55078125,366) # German station
            print 'Using GLOW sub-frequencies'
        else:
            subfreqs = np.arange(109.9609376,187.9882813,0.1953125) # For LOFAR
            print 'Using LOFAR sub-frequencies'
        LOFAR=True
    for ii in np.arange(nsub):
        if ii%10==0:
            print ii
        baseline_arr[ii],dm_arr[ii]=baseline_search(in_data[ii,0],nchan,off_pulse_size,subfreqs,LOFAR=LOFAR)
    baseline_arr.shape=nsub,nchan,1
    out_data = in_data[:,0] / baseline_arr
    if b_line is True:
        out_data -= 1.
    return out_data,dm_arr

def baseline_search(prof_2d,nchan,off_pulse_size,subfreqs,LOFAR=True):
    nbin=prof_2d.shape[1]
    binspersec = nbin/0.0016627
    tmp_fp=np.empty(np.shape(prof_2d))
    mask=np.zeros(np.shape(prof_2d),dtype=bool)
    mask[:,:off_pulse_size]=1
    binstep=3
    binshift_range=np.arange(0,nbin,binstep)
    if LOFAR is False:
        dmstep=0.002 ## WSRT
        dm_range=np.arange(0,0.05,dmstep) ## WSRT
    else:
	dmstep=0.0005 ## LOFAR
        dm_range=np.arange(0,0.01,dmstep) ## LOFAR
    fit=np.empty((binshift_range.size,dm_range.size))
    for binshift in binshift_range:
        for dm in dm_range:
            subdelays = psr_utils.delay_from_DM(dm, subfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays-hifreqdelay
            delaybins = subdelays*binspersec + binshift
            for jj in range(nchan):
                tmp_prof = prof_2d[jj,:].copy()
                tmp_fp[jj] = psr_utils.fft_rotate(tmp_prof, delaybins[jj])
            fit[binshift/binstep,int(dm/dmstep)]=np.nanmean(tmp_fp[mask])
    best=np.unravel_index(fit.argmin(),fit.shape)
    Bin=best[0]*binstep
    DM=best[1]*dmstep
    subdelays = psr_utils.delay_from_DM(DM, subfreqs)
    hifreqdelay = subdelays[-1]
    subdelays = subdelays-hifreqdelay
    delaybins = subdelays*binspersec + Bin
    for jj in range(nchan):
        tmp_prof = prof_2d[jj,:].copy()
        tmp_fp[jj] = psr_utils.fft_rotate(tmp_prof, delaybins[jj])
    base_perchan=np.nanmean(tmp_fp[:,:off_pulse_size],axis=1)
    return base_perchan,DM

#-----------------------------------------------------------------------------

def norm_fft_baseline(in_data,off_pulse_size=30,b_line=False):
    """
    Takes in_data ((nsubint,npol,nchan,nbin)) and normalises by off-pulse baseline of
    phase bin width = off_pulse_size.
    FFTs the profile in each subint and channel and low-pass filters to remove noise.
    Assumes baseline to be located around lowest point in smoothed profile, so finds
    minimum median value of the off-pulse range of bins around lowest point
    (+/- off-pulse width).
    Normalises each subint and channel by the corresponding baseline level.
    If b_line is true then unity is subtracted in order to baseline the array.
    Returns Stokes I of the corrected data.
    NOTE: This is susceptible to scattering over a full pulse phase (would remove
        pulsar flux scattered into baseline)
    NOTE 2: For low S/N this method seems to predict an off-pulse level too low,
        thus the normalised data is not fully baselined
    """
    nsub=in_data.shape[0]
    nbin=in_data.shape[1]
    nchan=in_data.shape[2]
    baseline_arr=np.empty((nsub,nchan))
    med_val=np.empty(off_pulse_size)
    for ii in np.arange(nsub):
        for jj in np.arange(nchan):
            prof=in_data[ii,0,jj]
            fft=np.fft.rfft(prof)
            fft[3:]=0 ## Low pass filter to remove noise
            prof_recover=np.fft.irfft(fft)
            tmp=np.append(prof_recover,[prof_recover,prof_recover])
            end=np.argmin(prof_recover)+nbin
            for yy in np.arange(off_pulse_size):
                med_val[yy]=np.nanmedian(tmp[yy+end-off_pulse_size:yy+end])
            baseline_arr[ii,jj]=np.min(med_val)
    baseline_arr.shape=nsub,nchan,1
    out_data = in_data[:,0] / baseline_arr
    if b_line is True:
        out_data -= 1.
    return out_data

#-----------------------------------------------------------------------------

def fluxcal_fit(hdu,mjds):
    """
    Fits power law curves to the scale and offsets that are the products of flux
    calibration. These can be applied to the data using the extract function below.
    """
    dataval,scl,offs,wts=Extract_data(hdu)
    nsub=dataval.shape[0]
    npol=dataval.shape[1]
    nchan=dataval.shape[2]
    nbin=dataval.shape[3]
    x=mjds-mjds.astype(int)
    x2=x.copy(); x2.shape=nsub,1

    # Fit curves
    fit_scl=np.polyfit(x,scl,4)
    fit_offs=np.polyfit(x,offs,4)
    ##fit_offs[:,:400]=0 ######## Only do this if offset=0 for Stokes I (first 400 values)
    y_scl=fit_scl[0]*(x2**4)+fit_scl[1]*(x2**3)+fit_scl[2]*(x2**2)+fit_scl[3]*x2+fit_scl[4]
    y_offs=fit_offs[0]*(x2**4)+fit_offs[1]*(x2**3)+fit_offs[2]*(x2**2)+fit_offs[3]*x2+fit_offs[4]

    #fit_scl=np.polyfit(x,scl,2)
    #fit_offs=np.polyfit(x,offs,2)
    ###fit_offs[:,:400]=0 ######## Only do this if offset=0 for Stokes I (first 400 values)
    #y_scl=fit_scl[0]*(x2**2)+fit_scl[1]*x2+fit_scl[2]
    #y_offs=fit_offs[0]*(x2**2)+fit_offs[1]*x2+fit_offs[2]

    # Remove outlier points and re-fit
    diff_scl=abs(y_scl-scl)
    diff_offs=abs(y_offs-offs)
    for j in range(npol*nchan):
        mask_scl=np.ones(nsub,dtype=bool)
        mask_offs=np.ones(nsub,dtype=bool)
        sigma_scl=np.std(diff_scl[:,j])
        sigma_offs=np.std(diff_offs[:,j])
        for i in range(nsub):
            if diff_scl[i,j]>3.*sigma_scl:
                mask_scl[i]=0
            if diff_offs[i,j]>3.*sigma_offs:
                mask_offs[i]=0
        fit_scl[:,j]=np.polyfit(x[mask_scl],scl[:,j][mask_scl],4)
        #fit_scl[:,j]=np.polyfit(x[mask_scl],scl[:,j][mask_scl],2)
        ##if j>399: ###### Only do this if offset=0 for Stokes I (first 400 values)
        fit_offs[:,j]=np.polyfit(x[mask_offs],offs[:,j][mask_offs],4)
        #fit_offs[:,j]=np.polyfit(x[mask_offs],offs[:,j][mask_offs],2)
    y_scl=fit_scl[0]*(x2**4)+fit_scl[1]*(x2**3)+fit_scl[2]*(x2**2)+fit_scl[3]*x2+fit_scl[4]
    y_offs=fit_offs[0]*(x2**4)+fit_offs[1]*(x2**3)+fit_offs[2]*(x2**2)+fit_offs[3]*x2+fit_offs[4]
    #y_scl=fit_scl[0]*(x2**2)+fit_scl[1]*x2+fit_scl[2]
    #y_offs=fit_offs[0]*(x2**2)+fit_offs[1]*x2+fit_offs[2]

    np.savetxt('calibration_scale_factors.txt',y_scl)
    np.savetxt('calibration_offsets.txt',y_offs)
    #Combine_data(dataval,y_scl,y_offs,wts)

#-----------------------------------------------------------------------------

def baseline_search(prof_2d,nchan,off_pulse_size,subfreqs,LOFAR=True):
    nbin=prof_2d.shape[1]
    binspersec = nbin/0.0016627
    tmp_fp=np.empty(np.shape(prof_2d))
    mask=np.zeros(np.shape(prof_2d),dtype=bool)
    mask[:,:off_pulse_size]=1
    binstep=3
    binshift_range=np.arange(0,nbin,binstep)
    if LOFAR is False:
        dmstep=0.002 ## WSRT
        dm_range=np.arange(0,0.05,dmstep) ## WSRT
    else:
        dmstep=0.0005 ## LOFAR
        dm_range=np.arange(0,0.01,dmstep) ## LOFAR
    fit=np.empty((binshift_range.size,dm_range.size))
    for binshift in binshift_range:
        for dm in dm_range:
            subdelays = psr_utils.delay_from_DM(dm, subfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays-hifreqdelay
            delaybins = subdelays*binspersec + binshift
            for jj in range(nchan):
                tmp_prof = prof_2d[jj,:].copy()
                tmp_fp[jj] = psr_utils.fft_rotate(tmp_prof, delaybins[jj])
            fit[binshift/binstep,int(dm/dmstep)]=np.nanmean(tmp_fp[mask])
    best=np.unravel_index(fit.argmin(),fit.shape)
    Bin=best[0]*binstep
    DM=best[1]*dmstep
    subdelays = psr_utils.delay_from_DM(DM, subfreqs)
    hifreqdelay = subdelays[-1]
    subdelays = subdelays-hifreqdelay
    delaybins = subdelays*binspersec + Bin
    for jj in range(nchan):
        tmp_prof = prof_2d[jj,:].copy()
        tmp_fp[jj] = psr_utils.fft_rotate(tmp_prof, delaybins[jj])
    base_perchan=np.nanmean(tmp_fp[:,:off_pulse_size],axis=1)
    return base_perchan,DM

#-----------------------------------------------------------------------------

def template_prof(input_profs,sum=True):
    """
    Phase aligns and sums (if sum=True) multiple input profiles from different observations
    to create a single high S/N template profile.
    input_profs - np.asarray((prof1,prof2,...)) of input profiles to be combined.
        prof1 is the profile to which the others are aligned
    """
    Nprof=np.shape(input_profs)[0]
    bins=input_profs[0].size
    aligned_profs=np.empty(np.shape(input_profs))
    for i in range(Nprof):
        offset=psr_utils.measure_phase_corr(input_profs[i],input_profs[0])
        #print 'Phase offset '+str(i)+' = '+str(offset)
        bin_offset=-offset*bins
        aligned_profs[i]=psr_utils.fft_rotate(input_profs[i],bin_offset)
    if sum is False:
        return aligned_profs
    else:
	template_prof=aligned_profs.sum(0)
        for i in range(Nprof):
            offset=psr_utils.measure_phase_corr(input_profs[i],template_prof)
            bin_offset=-offset*bins
            aligned_profs[i]=psr_utils.fft_rotate(input_profs[i],bin_offset)
        template_prof=aligned_profs.sum(0)
        return template_prof

def template_2d(input_profs,sum=True):
    """
    Phase aligns and sums multiple input 2D profiles from different observations
    to create a single high S/N template profile, while keeping full frequency
    resolution.
    input_profs - np.asarray((prof1,prof2,...)) of input profiles to be combined.
        prof1 is the profile to which the others are aligned
    """
    Nprof=np.shape(input_profs)[0]
    bins=input_profs[0].shape[1]
    channels=input_profs[0].shape[0]
    aligned_profs=np.empty(np.shape(input_profs))
    for i in range(Nprof):
        #offset=psr_utils.measure_phase_corr(input_profs[i].sum(0),input_profs[0].sum(0))
        offset=psr_utils.measure_phase_corr(np.nanmean(input_profs[i],axis=0),np.nanmean(input_profs[0],axis=0))
        #print 'Phase offset '+str(i)+' = '+str(offset)
        bin_offset=-offset*bins
        print bin_offset ################
        for jj in np.arange(channels):
            aligned_profs[i][jj]=psr_utils.fft_rotate(input_profs[i][jj],bin_offset)
    if sum is False:
        return aligned_profs
    else:
	#template_prof=aligned_profs.sum(0)
        template_prof=np.nanmean(aligned_profs,axis=0)
        for i in range(Nprof):
            #offset=psr_utils.measure_phase_corr(input_profs[i].sum(0),template_prof.sum(0))
            offset=psr_utils.measure_phase_corr(np.nanmean(input_profs[i],axis=0),np.nanmean(template_prof,axis=0))
            bin_offset=-offset*bins
            for jj in np.arange(channels):
                aligned_profs[i][jj]=psr_utils.fft_rotate(input_profs[i][jj],bin_offset)
        #template_prof=aligned_profs.sum(0)
        template_prof=np.nanmean(aligned_profs,axis=0)
        return template_prof

#-----------------------------------------------------------------------------

def dedisperse_subint(chi_arr,data_in,DD=False,P=0.0016627,dmmin=39.6604,dmint=0.0002,currdm=39.6608,subfreqs=None):
    """
    Disperse or dedisperse individual subintegrations of an input data array
    If DD=True then data is dedispersed (DD=False disperses)
    """
    if np.ndim(data_in)==3:
        intervals=data_in.shape[0]
        channels=data_in.shape[1]
        bins=data_in.shape[2]
    if np.ndim(data_in)==4:
        intervals=data_in.shape[0]
        npol=data_in.shape[1]
        channels=data_in.shape[2]
        bins=data_in.shape[3]
    binspersec=bins/P
    if subfreqs is None:
        subfreqs=np.arange(109.9609376,187.9882813,0.1953125) ## LOFAR
    tmp_fp=np.empty((channels,bins))
    data_new=np.empty(np.shape(data_in))
    if DD is True:
        currdm*=-1
    if np.ndim(data_in)==3:
        for ii in np.arange(intervals):
            min_chi_dm=np.unravel_index(chi_arr[ii].argmin(),chi_arr[ii].shape)[0]
            ###min_chi_dm=chi_arr[ii]
            dm_estimate=dmmin + min_chi_dm*dmint
            if DD is True:
                dm_estimate*=-1
            data_new[ii]=dedisperseLOFAR.dm_rotate(data_in[ii],dm_estimate,currdm,0.,subfreqs,binspersec,channels,tmp_fp)
    if np.ndim(data_in)==4:
        for ii in np.arange(intervals):
            min_chi_dm=np.unravel_index(chi_arr[ii].argmin(),chi_arr[ii].shape)[0]
            dm_estimate=dmmin + min_chi_dm*dmint
            if DD is True:
                dm_estimate*=-1
            for jj in np.arange(npol):
                data_new[ii,jj]=dedisperseLOFAR.dm_rotate(data_in[ii,jj],dm_estimate,currdm,0.,subfreqs,binspersec,channels,tmp_fp)
    return data_new

