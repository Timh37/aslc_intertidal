# aslc_intertidal
Repository for the conceptual analysis of the effect of (changes in) the range of the annual sea-level cycle on the frequency and duration of inundation and emergence in intertidal zones. This repository underlies the Brief Communication "Future Changes in Seasonal Sea-Level Variability Could Reshape Coastal Ecosystems" in revision for Nature Climate Change by Tim Hermans, Greg Fivash and Jim van Belzen.

## Repository structure / how-to
The repository is organized as follows:
- bin : contains the binary output of the conceptual model, used for plotting figures
- data : external data used for plotting figures
- figures : contains the output of the scripts for plotting figures
- src: contains the source-code, organised by figure. 

Each directory in aslc_intertidal/src contains Jupyter Notebooks that can be run by users to reproduce the figures in our manuscript. The code of the conceptual inundation and emergence model is retrieved from the script functions.py. If no output is already present in aslc_intertidal/bin, this output is produced first before performing the analysis.

## Dependencies
- xarray
- netcdf
- pandas
- numpy
- scipy
- matplotlib
- cmocean
- cartopy
