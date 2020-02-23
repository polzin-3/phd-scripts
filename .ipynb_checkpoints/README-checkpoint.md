# phd-scripts
Mix of python scripts developed throughout my PhD  for use with radio astronomy data. Note that these were made for Python2 and are likely not compatible with python3

The majority of these scripts are short, quick scripts to perform repeatable tasks on radio astronomy data. They were written specificly for my use cases, and definitely **not optimised**, so feel free to take them and improve.

Note that psr_utils.py and my_binary_psr.py (Rene Breton) are not my scripts, but copies are provided here as they are sourced by some functions and thus are required for some of my scripts to run.

dedidisperseLOFAR.py, observation_utils.py and normalise_data.py are the most in-depth, containing a number of functions that can be generally applied to data from different telescopes.

## dedisperseLOFAR.py
---
Most applicable to eclipsing pulsars (or others with small, short-timescale DM or pulse scattering variations). This script takes a 3-dimensional array ((time-intervals, freq-channels, pulse-phase-bins)) and searches over a specified range of DM and scattering timescales to find the best-fitting values in each time-interval by performing least-squares fits using an input pulse template. The chi-square array of all the fits is returned, giving a useful tool for identifying correlations and/or false results.

The performance of this script is highly dependent on the validity and quality of the input pulsar observation data and the input template. It is not particularly robust to low signal-to-noise data and can often produce spurious results in these cases. This is where the verbose output format of the results comes in handy, as the minimum chi-square and template amplitude for the fits using every combination of DM and tau are returned, making it easier to spot the problem areas. Due to this sensitivity, care should be taken when preparing your inputs, and an iterative process may be required.

The input data should have sub-integrations, frequency channelisation, and pulse-phase binning such that the signal-to-noise is fairly high, however you must balance this with respect to the biases that you may introduce by integrating data together. For example, longer sub-integrations give higher signal-to-noise, however if you are trying to measure a change in DM over time, then you will begin to integrate together data with differential DMs, causing the pulsations to be smeared beyond recovery, and the function will struggle to find any good fit. Equally, integrating along the pulse phase axis to give wider phase bins will reduce the sensitivity to small changes in DM or tau. Since the function uses incoherent (de-)dispersion methods, if you integrate together frequency channels then you risk smearing the profile beyond recovery in each channel. You should ensure that the data has been well cleaned of any RFI.

The input template is equally important in the success of the method. It should have as high signal-to-noise as possible, and be representative of the true pulse shape of the pulsar. A generally good method, if you have the available data, is to sum observed data over a long duration (out-of-eclipse and dedispersed) to give a relatively high S/N 2D template (freq chan, pulse phase) that can be further improved by applying a smoothing algorithm (e.g. Savitzky-Golay filter) to reduce noise. If you are using a full de-dispersed template, then it must be accurately phase-aligned with the data that you are trying to measure. For an eclipsing pulsar this can be achieved by cross-correlating the template with the de-dispersed, out-of-eclipse part of the observed data (see template_2d from normalise_data.py).

As the templates and data retain 2-dimensions (freq chan and pulse phase) in the fits, then they must both be correctly normalised prior to input. This means that the off-pulse baseline should be removed, and the pulse amplitudes scaled by a consistent method across all frequency channels. Some methods to do this are contained in the functions of normalise_data.py.

The output data: chi-square array (best-fit chi-square for each sub-integration, DM & tau), amplitude array (template scale factor for best-fit of each sub-int, DM & tau) and amplitude error array (corresponding scale factor fit errors, not accounting for uncertainty in DM & tau), are saved to disk (as .txt) upon completion. When reading these into python you will first need to reshape the arrays into ((# of sub-ints, # of DMs, # of tau)). Automatic contour plots of the DM and tau vs sub-integration (contours of 1,2 and 3 sigma) can be made with the params_vs_time() function from chi_sq_plotter.py.

## normalise_data.py
---
A selection of functions for loading FITS files of astronomy data into python and performing normalisation of the data (e.g. baseline removal) through a range of methods.


## observation_utils.py
---
A selection of functions that are useful for planning / investigating observations (inc. altitude / azimuth calculator, mjd -> orbital phases, LOFAR archival observation searcher, TASC searching, ...).
