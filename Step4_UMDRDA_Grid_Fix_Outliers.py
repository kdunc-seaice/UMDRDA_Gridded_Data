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
from scipy.stats import binned_statistic_2d, median_abs_deviation
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
pathname2 = '//' ##Path to anomaly files

##clim_names = ['SailHeight_T20','SailHeight_T60','MaxSailHeight_T20','MaxSailHeight_T60',
##              'RidgingIntensity_T20','RidgingIntensity_T60','SailWidth_T20','SailWidth_T60',
##              'RidgeLength_T20','RidgeLength_T60','SurfRough','SurfRough_ltT20','SurfRough_ltT60']

clim_names = ['SailSpacing_T20', 'SailSpacing_T60']

##param_names = ['sail_H_T20','sail_H_T60','max_sail_H_1km_T20','max_sail_H_1km_T60',
##               'ridging_I_1km_T20','ridging_I_1km_T60','sail_W_T20','sail_W_T60',
##               'ridge_L_T20','ridge_L_T60','surface_roughness_1km','surface_roughness_1km_ltT20','surface_roughness_1km_ltT60']

param_names = ['sail_D_T20','sail_D_T60']

#clim_names = ['SurfRough']
#param_names = ['surface_roughness_1km']

for z,name in enumerate(clim_names):
    #print (name)
    
    nc_files = sorted(np.array(glob.glob(pathname + '*IS2-UMDRDA-NH-EASE2-25000m-v01-202510.nc')))
    clim_files = sorted(np.array(glob.glob(pathname2 + '*202510_Mean_' + name + '_Anomaly.txt')))

    grid_xy = 432 #3456#1080
    grid_res = 25000 #3125#10000

    for c,i in enumerate(nc_files):
        print (c,i)
        print (clim_files[c])

        UMD_data = Dataset(i, 'r+')
        
        UMD_data.set_auto_mask(False)

        clim_data = pd.read_csv(clim_files[c], sep=',', header=None)

        name_interp = param_names[z] + '_interp'
        name_orig = param_names[z]
        name_count = param_names[z] + '_n'

        UMD_lon = UMD_data.variables['longitude'][:].ravel()
        UMD_lat = UMD_data.variables['latitude'][:].ravel()
        UMD_param = UMD_data.variables[name_interp][:].ravel()

        UMD_param_orig = UMD_data.variables[name_orig][:].ravel()
        SIC = UMD_data.variables['sea_ice_conc'][:]

        UMD_good = np.where(UMD_param > 0.0)[0]
        UMD_good2 = np.where(UMD_param_orig > 0.0)[0]
        clim_good = np.where(clim_data[2].values > np.nanmin(clim_data[2].values))[0]
        UMD_param = UMD_param[UMD_good]

        UMD_param_orig = UMD_param_orig[UMD_good2]
        SIC_bad = np.where((SIC.ravel() >= 15.0) & (SIC.ravel() < 85.0))[0]

        #upper_thresh = np.nanmean(clim_data[2].values)+(np.nanstd(clim_data[2].values)*1.0)
        upper_thresh = np.nanmean(clim_data[2].values)-(np.nanstd(clim_data[2].values)*1.0) ## For Sail Spacing

        #thresh_idx = np.where(clim_data[2].values > upper_thresh)[0] ## For all params other than Sail Spacing ##
        thresh_idx = np.where(clim_data[2].values < upper_thresh)[0] ## For Sail Spacing ##

        if (thresh_idx.max() < UMD_param.size):
            UMD_param[thresh_idx] = np.nan

        elif (thresh_idx.max() >= UMD_param.size):
            thresh_idx_new = thresh_idx[(thresh_idx <= UMD_param.size-1)]
            UMD_param[thresh_idx_new] = np.nan

        PSG_coords_upthresh = reproject_to_PSG(UMD_lon[UMD_good][np.where(np.isnan(UMD_param) == True)[0]],UMD_lat[UMD_good][np.where(np.isnan(UMD_param) == True)[0]])
        PSG_coords_SIC_lt90 = reproject_to_PSG(UMD_lon[SIC_bad],UMD_lat[SIC_bad])
        #PSG_coords = reproject_to_PSG(UMD_lon[UMD_good],UMD_lat[UMD_good])

        B = np.vstack((np.round(PSG_coords_upthresh[0],decimals=1),np.round(PSG_coords_upthresh[1],decimals=1))).T
        A = np.vstack((np.round(PSG_coords_SIC_lt90[0],decimals=1),np.round(PSG_coords_SIC_lt90[1],decimals=1))).T

        #same = np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])


        arr1 = np.empty(len(B), dtype=[('x', float), ('y', float)])
        arr2 = np.empty(len(A), dtype=[('x', float), ('y', float)])

        # Add the data to the structured array
        arr1['x'] = B[:,0]
        arr1['y'] = B[:,1]

        arr2['x'] = A[:,0]
        arr2['y'] = A[:,1]

        same, arr1_ind, arr2_ind = np.intersect1d(arr1,arr2,return_indices=True)

        UMD_lon = UMD_data.variables['longitude'][:].ravel()
        UMD_lat = UMD_data.variables['latitude'][:].ravel()
        UMD_param = UMD_data.variables[name_interp][:].ravel()
        UMD_param_orig = UMD_data.variables[name_orig][:].ravel()
        UMD_param_count = UMD_data.variables[name_count][:].ravel()
        SIC = UMD_data.variables['sea_ice_conc'][:]

        UMD_good = np.where(UMD_param > 0.0)[0]
        UMD_good2 = np.where(UMD_param_orig > 0.0)[0]
        clim_good = np.where(clim_data[2].values > np.nanmin(clim_data[2].values))[0]
        #UMD_param = UMD_param[UMD_good]
        #UMD_param_orig = UMD_param_orig[UMD_good2]
        SIC_bad = np.where((SIC.ravel() >= 15.0) & (SIC.ravel() < 85.0))[0]
        SIC_good = np.where(SIC.ravel() >= 85.0)[0]

        UMD_SIC_bad = UMD_param[SIC_bad]
        UMD_SIC_bad_orig = UMD_param_orig[SIC_bad]
        UMD_SIC_bad_orig_count = UMD_param_count[SIC_bad]

        #stacked = np.column_stack((UMD_lon[SIC_bad][arr2_ind],UMD_lat[SIC_bad][arr2_ind],UMD_SIC_bad[arr2_ind]))
        #np.savetxt(pathname + '202504_SurfRough_Outliers.txt',stacked,delimiter=',',fmt='%.8f')

        UMD_SIC_bad[arr2_ind] = np.nan
        UMD_SIC_bad_orig[arr2_ind] = np.nan
        UMD_SIC_bad_orig_count[arr2_ind] = np.nan

        nan_val_idx = np.where(np.isnan(UMD_SIC_bad) == True)[0]
        
        print ('Interpolating...')

        data_interp_nn = pygmt.nearneighbor(x=PSG_coords_SIC_lt90[0], y=PSG_coords_SIC_lt90[1], z=UMD_SIC_bad,
                                                        spacing="25000+e",
                                                        region=[-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0),-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0)],
                                                        search_radius="100000",
                                                        sectors="36+m3")

        SIC = SIC.reshape(grid_xy,grid_xy)

        data_interp_nn.values.T[np.where(np.isnan(SIC) == True)] = np.nan

        UMD_lon_2d = UMD_lon.reshape(grid_xy,grid_xy)
        UMD_lat_2d = UMD_lat.reshape(grid_xy,grid_xy)
        
        data_interp_nn.values.T[np.where(UMD_lat_2d > 88.0)] = np.nan
        data_interp_nn.values.T[np.where(UMD_lat_2d < 60.0)] = np.nan

        UMD_SIC_bad[nan_val_idx] = data_interp_nn.values.T.ravel()[SIC_bad][nan_val_idx]

        UMD_SIC_param_final = np.hstack((UMD_SIC_bad,UMD_param[SIC_good]))
        UMD_lon_param_final = np.hstack((UMD_lon_2d.ravel()[SIC_bad],UMD_lon_2d.ravel()[SIC_good]))
        UMD_lat_param_final = np.hstack((UMD_lat_2d.ravel()[SIC_bad],UMD_lat_2d.ravel()[SIC_good]))

        data_interp_good = np.where(UMD_SIC_param_final > 0.0)[0]

        UMD_SIC_param_orig_final = np.hstack((UMD_SIC_bad_orig,UMD_param_orig[SIC_good]))
        UMD_lon_param_orig_final = np.hstack((UMD_lon_2d.ravel()[SIC_bad],UMD_lon_2d.ravel()[SIC_good]))
        UMD_lat_param_orig_final = np.hstack((UMD_lat_2d.ravel()[SIC_bad],UMD_lat_2d.ravel()[SIC_good]))

        data_orig_good = np.where(UMD_SIC_param_orig_final > 0.0)[0]

        UMD_param_interp = UMD_data.variables[name_interp][:].ravel()
        UMD_param_orig_2 = UMD_data.variables[name_orig][:].ravel()
        UMD_param_count_2 = UMD_data.variables[name_count][:].ravel()

        UMD_param_interp[SIC_bad] = UMD_SIC_bad
        UMD_param_interp[SIC_good] = UMD_param[SIC_good]

        UMD_param_orig_2[SIC_bad] = UMD_SIC_bad_orig
        UMD_param_orig_2[SIC_good] = UMD_param_orig[SIC_good]

        UMD_param_count_2[SIC_bad] = UMD_SIC_bad_orig_count
        UMD_param_count_2[SIC_good] = UMD_param_count[SIC_good]

        UMD_data.variables[name_interp][:,:] = UMD_param_interp.reshape(grid_xy,grid_xy)
        UMD_data.variables[name_orig][:,:] = UMD_param_orig_2.reshape(grid_xy,grid_xy)
        UMD_data.variables[name_count][:,:] = UMD_param_count_2.reshape(grid_xy,grid_xy)

        UMD_data.close()


print("%s seconds" % (time.time() - start_time))
