# UMDRDA_Gridded_Data
Provided here is code used to turn ICESat-2 UMD-RDA along-track data into monthly grids, calculate monthly climatologies, monthly anomalies, and the removal of outliers. Gridded data files are calculated at three spatial resolutions of 25, 10, and 3.125 km.

"Step1_Arctic_Grid_netCDF_Monthly_PyGMT_FINAL.py" contains code for gridding the ICESat-2 UMD-RDA along-track data using PyGMT

"Step2_UMDRDA_Grid_Monthly_Climatology.py" calculates monthly climatologies for January-December for the ICESat-2 period of October 2018 to September 2025

"Step3_UMDRDA_Grid_Monthly_Anomaly.py" calculates monthly anomalies based on the monthly climatologies from Step 2

"Step4_UMDRDA_Grid_Fix_Outliers.py" uses the monthly anomalies and sea ice concentration data to identify and remove outliers, producing a final monthly gridded netcdf file

The gridded deformation data product is available at PANGAEA (https://www.pangaea.de). The DOIs for the different resolution products are:

25 km: https://doi.org/10.1594/PANGAEA.990274 \
10 km: https://doi.org/10.1594/PANGAEA.990265 \
3.125 km: https://doi.org/10.1594/PANGAEA.990275

The product "User Guide" can be found at: https://download.pangaea.de/reference/137687/attachments/UMD_Arctic_SeaIceDeformation_UserGuide.pdf

Please email kduncan@umd.edu for any questions related to this repository

# References
Please reference the following publication, in addition to the dataset, when using this data:

Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice pressure ridge topography with ICESat-2.Geophysical Research Letters, 49, e2022GL100272. https://doi.org/10.1029/2022GL100272

# Acknowledgements & Funding:
This work was supported by NASA Cryosphere Program Grant 80NSSC24K1450 and by NOAA Grant NA24NESX432C0001 (Cooperative Institute for Satellite Earth System Studies – CISESS) at the University of Maryland/ESSIC.

