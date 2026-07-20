Make sure these scripts are run in a CMSSW environment after you call `cmsenv`.

Before this make sure to run:
  - `../plotting/makeSignalHistogram.py`
  - `../plotting/makeSignalDatacard.py`
  - `../plotting/getUncertainties.py`
  - `../plotting/makeBGDatacards.py`

The main scripts are:
  - `run_fits.sh` which combines datacards, does the CR fit to data, then does AsymptoticLimits
  - `run_limits.sh` which takes the output of AsymptoticLimits and makes a json file for each signal sample with the limits
  - `run_plot_limits.sh` which takes `plot_limits_new.py` and makes the limit plots

