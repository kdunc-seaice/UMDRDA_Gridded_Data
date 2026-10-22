import numpy as np
import numpy.ma as ma
import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, warnings, time, csv, math
from datetime import datetime, date, timezone
from global_land_mask import globe
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.stats import binned_statistic_2d
from pyproj import Proj, transform
from netCDF4 import Dataset
from scipy.spatial import distance
import itertools

warnings.filterwarnings("ignore")
np.printoptions(suppress=True)

def reproject_to_PSG(x, y):
    inProj,outProj = Proj(init='epsg:4326'), Proj(init='epsg:6931')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2

def reproject_to_WGS84(x, y):
    inProj,outProj = Proj(init='epsg:6931'), Proj(init='epsg:4326')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2


np.set_printoptions(suppress=True)

start_time = time.time()

pathname = '//' ##Path to gridded netcdf files
pathname2 = '//' ##Path to climatology files from Step #2
pathname3 = '//' ##Path to save anomaly files to

nc_files = sorted(np.array(glob.glob(pathname + '*3125m*.nc')))
clim_files = sorted(np.array(glob.glob(pathname2 + '*SurfRough.txt')))

for c,i in enumerate(nc_files):
    print (i)

    UMD_data = Dataset(i, 'r')
    
    UMD_data.set_auto_mask(False)

    clim_data = pd.read_csv(clim_files[int(i[-5:-3])-1], sep=',', header=None)
    #clim_data = pd.read_csv(clim_files[c], sep=',', header=None)
    print (clim_files[int(i[-5:-3])-1])

    UMD_lon = UMD_data.variables['longitude'][:].ravel()
    UMD_lat = UMD_data.variables['latitude'][:].ravel()
    UMD_param = UMD_data.variables['surface_roughness_1km_ltT60_interp'][:].ravel()

    UMD_good = np.where(UMD_param > 0.0)[0]
    clim_good = np.where(clim_data[2].values > 0.0)[0]

    PSG_coords = reproject_to_PSG(UMD_lon[UMD_good],UMD_lat[UMD_good])

    B = np.vstack((np.round(PSG_coords[0],decimals=1),np.round(PSG_coords[1],decimals=1))).T
    A = np.vstack((clim_data[0].values[clim_good],clim_data[1].values[clim_good])).T

    arr1 = np.empty(len(B), dtype=[('x', float), ('y', float)])
    arr2 = np.empty(len(A), dtype=[('x', float), ('y', float)])

    # Add the data to the structured array
    arr1['x'] = B[:,0]
    arr1['y'] = B[:,1]

    arr2['x'] = A[:,0]
    arr2['y'] = A[:,1]

    same, arr1_ind, arr2_ind = np.intersect1d(arr1,arr2,return_indices=True)

    param_val_UMD = UMD_param[UMD_good][arr1_ind]
    param_val_clim = clim_data[2].values[clim_good][arr2_ind]

    same_lon = np.array([x[0] for x in same])
    same_lat = np.array([x[1] for x in same])

    monthly_anom = param_val_UMD - param_val_clim

    good_data = np.where(monthly_anom < 100000.0)[0]

    stacked = np.column_stack((same_lon[good_data],same_lat[good_data],monthly_anom[good_data]))
    np.savetxt('Path to save Anomaly file',stacked,delimiter=',',fmt=('%.1f','%.1f','%.8f'))

    


print("%s seconds" % (time.time() - start_time))
