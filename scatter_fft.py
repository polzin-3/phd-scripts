import numpy as np

def scatter_fft(prof, tau=0, kscat=False):
    """
    Models the effect of scattering an input profile by convolving with an
    exponential.
    tau [float] = characteristic scattering time.
    n (optional) = number of sampling bins if desired to be larger than input
    profile.
    """
    if kscat is False:
        x = np.arange(0, 1, 1./np.size(prof))
        tail = (1/tau) * np.exp(-x/tau)
    else:
        x = np.arange(np.size(prof)) + 1
        tail = BF(tau, x)
    tail_norm = tail / np.sum(tail)
    prof_rfft = np.fft.rfft(prof)
    tail_rfft = np.fft.rfft(tail_norm)
    product = prof_rfft * tail_rfft
    scat_prof = np.fft.irfft(product)
    return scat_prof

def BF(tau, x): ##### Thick screen near source. See Karastergiou, 2009
    tail = (((np.pi*tau) / (4*x**3)) ** 0.5) * np.exp(-(tau*np.pi**2) / (16*x))
    return tail
