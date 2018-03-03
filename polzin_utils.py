import numpy as np
from scipy import *
from scipy import optimize

###########################################
# Contents:
#	Powerlaw_fit
#	Fermi_fit
#	fit_line_nr (Rene's function)
#	py_2_ascii (3 funcs)
###########################################

def Powerlaw_fit(x,y=1.,yerr=None):
    if yerr is None:
        yerr=np.ones(y.size)
    tmp_mask=y==0.
    y[tmp_mask]=np.nan
    mask=np.isnan(y)
    xdata=x[~mask]
    ydata=y[~mask]
    ydata_err=yerr[~mask]
    logx=log10(xdata)
    logy=log10(ydata)
    logyerr=ydata_err/ydata
    fit=fit_line_nr(logy,x=logx,err=logyerr)
    alpha=fit[0][1]; amp=10**fit[0][0]
    alpha_err=fit[1][1]; amp_err=fit[1][0]*amp
    print 'amplitude = '+str(amp)+' +/- '+str(amp_err)
    print 'alpha = '+str(alpha)+' +/- '+str(alpha_err)
    return alpha,alpha_err

#---------------------------------------------------------------------------------------

def Fermi_fit(x,y=1.,sigma=1.,a=1.,b=1.):
    """
    Fit Fermi-Dirac function, y = 1 / (exp((x + a)/b) + 1)
    x,y - input data
    a,b - initial parameter estimates (floats)
    """
    mask=~np.isnan(y)
    p0=[a,b]
    fitfunc=lambda x,a,b: 1./(np.exp((x+a)/b)+1)
    p1,p1_cov=optimize.curve_fit(fitfunc,x[mask],y[mask],p0[:],sigma[mask])
    uncertainty=np.sqrt(np.diag(p1_cov))
    print 'a =',p1[0],'+/-',uncertainty[0]
    print 'b =',p1[1],'+/-',uncertainty[1]
    return p1,p1_cov

#---------------------------------------------------------------------------------------

# From Rene Breton hulk:/home/bretonr/lib/python/bretonr_utils.py
def fit_line_nr(y, x=None, err=1.0, m=None, b=None, short=None, output=None):
    """
    fit_line_nr(y, x=None, err=1.0, m=None, b=None, short=None, output=None):
        Return ([b, m], [err_b, err_m], red_chi2)
    One or many linear equations can be fitted simultaneously.

    If one:
            y, x: vector (m,)
            err: scalar or vector (m,)
            m, b: scalar
    If many:
            y, x: array (m,n)
            err: scalar or array (m,n)
            m, b: vector (n,)
    If output: A plot will be displayed and information printed.
    If short: Only the best fit slope and intersect are returned (array (b,m)).

    N.B. Uses the numerical recipe algorithm.
    """
    if (type(err) == type(1.0)):
        err = (np.ones(y.T.shape)*err).T
    if x is None:
        x = np.arange(len(y),dtype='d')
    if x.shape != y.shape:
        x = (np.ones(y.T.shape)*x).T
    err2 = err**2
    if (m is None) and (b is None):
        S = np.sum(1/err2,0)
        Sx = np.sum(x/err2,0)
        Sy = np.sum(y/err2,0)
        #Sxx = np.sum(x**2/err2,0)
        #Sxy = np.sum(x*y/err2,0)
        t = (x-Sx/S)/err
        Stt = np.sum(t**2,0)
        #Delta = S*Sxx-Sx**2
        #m = (S*Sxy-Sx*Sy)/Delta
        #err_m = np.sqrt(S/Delta)
        m = np.sum(t*y/err,0)/Stt
        #b = (Sxx*Sy-Sx*Sxy)/Delta
        #err_b = np.sqrt(Sxx/Delta)
        b = (Sy-Sx*m)/S
        if short: return np.array([b,m])
        err_b = np.sqrt((1+Sx**2/(S*Stt))/S)
        err_m = np.sqrt(1/Stt)
        nfit = 2
    elif (m is not None) and (b is None):
        if len(y.shape) > 1:  m = (np.ones(y.shape[-1])*m).T
        S = np.sum(1/err2,0)
        Sx = np.sum(x/err2,0)
        Sy = np.sum(y/err2,0)
        #Sxx = np.sum(x**2/err2,0)
        #Sxy = np.sum(x*y/err2,0)
        #t = (x-Sx/S)/err
        #Stt = np.sum(t**2,0)
        #Delta = S*Sxx-Sx**2
        b = (Sy-m*Sx)/S
        if short: return np.array([b,m])
        err_b = np.sqrt(1/S)
        if len(y.shape) > 1: err_m = (np.zeros(y.shape[-1])).T
        else: err_m = 0.0
        nfit = 1
    elif (m is None) and (b is not None):
        if len(y.shape) > 1: b = (np.ones(y.T.shape[-1])*b).T
        #S = np.sum(1/err2,0)
        Sx = np.sum(x/err2,0)
        #Sy = np.sum(y/err2,0)
        Sxx = np.sum(x**2/err2,0)
        Sxy = np.sum(x*y/err2,0)
        #t = (x-Sx/S)/err
        #Stt = np.sum(t**2,0)
        #Delta = S*Sxx-Sx**2
        m = (Sxy-b*Sx)/Sxx
        if short: return np.array([b,m])
        err_m = np.sqrt(1/Sxx)
        if len(y.shape) > 1: err_b = (np.zeros(y.shape[-1])).T
        else: err_b = 0.0
        nfit = 1
    else:
	nfit = 0
        err_b = 0.
        err_m = 0.
    red_chi2 = np.sum(((m*x+b - y)/err)**2,0)/(len(y)-nfit)
    if output:
	#plotxy(y, x, line=None, symbol=2, color=2)
        #plotxy(m*x+b, x)
        print 'Reduced chi-square: ' + str(red_chi2)
        print 'b -> ' + str(b) + ' err_b -> ' +str(err_b)
        print 'm -> ' + str(m) + ' err_m -> ' +str(err_m)
    return np.array([b, m]), np.array([err_b, err_m]), red_chi2

#------------------------------------------------------------------------------

def py_2_ascii_intervals(timephase,int_duration=2):
    """
    Splits up python time vs. phase text file into ascii files each containing
    one profile summed over n consecutive intervals (e.g. intervals_0_1,
    intervals_2_3, ...)
    int_duration is the number of intervals to sum
    """
    intervals=timephase.shape[0]
    bins=timephase.shape[1]
    for j in np.arange(0,intervals,int_duration):
        f=open('intervals_'+str(j)+'_'+str(j+int_duration)+'.dat','w+')
        for i in range(bins):
            #f.write('0 0 '+str(i)+' '+str(timephase[j:j+int_duration].sum(0)[i])+'\n')
            f.write(str(i)+' '+str(timephase[j:j+int_duration].sum(0)[i])+'\n')
        f.close()

def py_2_ascii_chans(freqphase,sub_band_chans=50):
    """
    Splits up python time vs. phase text file into ascii files each containing
    one profile summed over n consecutive channels (e.g. channels_0_50,
    channels_50_100, ...)
    sub_band_chans is the number of channels to sum
    """
    channels=freqphase.shape[0]
    bins=freqphase.shape[1]
    for j in np.arange(0,channels,sub_band_chans):
        f=open('channels_'+str(j)+'_'+str(j+sub_band_chans)+'.dat','w+')
        for i in range(bins):
            #f.write('0 0 '+str(i)+' '+str(freqphase[j:j+sub_band_chans].sum(0)[i])+'\n')
            f.write(str(i)+' '+str(freqphase[j:j+sub_band_chans].sum(0)[i])+'\n')
        f.close()

def py_2_ascii_template(profile):
    """
    Takes python array of template profile and writes out as ascii file
    """
    bins=profile.size
    f=open('template_prof.dat','w+')
    for i in range(bins):
        f.write(str(i)+' '+str(profile[i])+'\n')
    f.close()

#---------------------------------------------------------------------------------------
