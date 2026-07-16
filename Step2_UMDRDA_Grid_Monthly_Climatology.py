import numpy as np
import numpy.ma as ma
import pygmt
import pandas as pd
#import matplotlib.pyplot as plt
import glob, os, warnings, time, csv, math
from datetime import datetime, date, timezone
#from global_land_mask import globe
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.stats import binned_statistic_2d
from pyproj import Proj, transform
from netCDF4 import Dataset
from scipy.spatial import distance

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

pathname = '//' ##Path to gridded files
pathname2 = '//' ##Path to save files to

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
month_names = ['January','February','March','April','May','June','July','August','September',
              'October','November','December']


for c,month in enumerate(months):
    print (month)
    
    nc_files = sorted(np.array(glob.glob(pathname + '*' + month + '.nc')))
    #print (nc_files)

    param_list = []

    for i in nc_files:
        #print (i)

        data = Dataset(i, 'r')

        lon = data.variables['longitude'][:]
        lat = data.variables['latitude'][:]
        param = data.variables['ridge_L_T60_interp'][:].ravel()
        param[(param > 100000.0)] = 0.0

        param_list.append(param)

    param_mean = np.nanmean(param_list, axis=0).reshape(1080,1080)
    #param_mean = np.nansum(param_list, axis=0) / len(param_list)
    #param_mean = np.reshape(param_mean,shape=(432,432))
    param_mean[np.where(param_mean == 0.0)] = np.nan
    

    good_data = np.where(param_mean.ravel() > 0.0)[0]
    good_data2 = np.where(param_mean.ravel()[good_data] < 100000.0)[0]

    print (param_mean.ravel()[good_data][good_data2].size)

    PSG_coords = reproject_to_PSG(lon.ravel(),lat.ravel())

    stacked = np.column_stack((PSG_coords[0][good_data][good_data2],PSG_coords[1][good_data][good_data2],param_mean.ravel()[good_data][good_data2]))
    np.savetxt(pathname2 + 'Climatology/IS2-UMDRDA-NH-EASE2-10.0km-v01-2018_2025_' + month + '_Mean_RidgeLength_T60.txt',stacked,delimiter=',',fmt=('%.1f','%.1f','%.3f'))

    


print("%s seconds" % (time.time() - start_time))
