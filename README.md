# phd-scripts
Mix of python scripts developed throughout my PhD  for use with radio astronomy data.

The majority of these scripts are short, quick scripts to perform repeatable tasks on radio astronomy data (often specific for my use).

dedidisperseLOFAR.py, observation_utils.py and normalise_data.py are the most in-depth, containing a number of functions that can be generally applied to data from different telescopes.

dedisperseLOFAR.py  --  Most applicable to eclipsing pulsars (or others with small, short-timescale DM or pulse scattering variations). This script takes a 3-dimensional array ((time-intervals, freq-channels, pulse-phase-bins)) and searches over a specified range of DM and scattering timescales to find the best-fitting values in each time-interval by performing least-squares fits using an input pulse template. The chi-square array of all the fits is returned, giving a useful tool for identifying correlations and/or false results.

observation_utils.py  --  A selection of functions that are useful for planning / investigating observations (inc. altitude / azimuth calculator, mjd > orbital phases, LOFAR archival observation searcher, ...).

normalise_data.py  --  A selection of functions for loading FITS files of astronomy data into python and performing normalisation of the data (e.g. baseline removal) through a range of methods.
